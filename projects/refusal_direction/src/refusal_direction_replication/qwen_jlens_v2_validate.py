"""Offline validator for the long-context (v2) Qwen Jacobian-lens run.

This module is deliberately independent from ``qwen_jlens_validate`` (the v1
project-root validator): it never calls it and it does not require the v1
pilot/dose/jlens result tree. It reads only the v2 profile, the v2 results and
the v2 artifacts directory, and raises a descriptive ``ValueError`` on the first
invalid condition. On success it returns a dictionary of named checks, all
``True``.

Two run shapes are accepted:

* ``pass``: fit on a selected candidate count, a 200-item holdout, a comparison
  of the v2 lens, the frozen v1 lens and the logit lens on that holdout.
* ``aborted`` before selection: the prepare artifact and corpus hash describe
  the full candidate pool, ``n_fit`` is null and no fit, holdout, comparison,
  checkpoint or final lens exists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .pipeline import load_config
from .qwen_dose import sha256_file
from .qwen_jlens import (
    DATASET_CONFIG,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DIM_BATCH_PROBE_ORDER,
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    EXPECTED_SOURCE_LAYERS,
    EXPECTED_TARGET_LAYER,
    FIT_COUNT,
    JLENS_LEAK_KEYS,
    JLENS_UPSTREAM_COMMIT,
    JLENS_V2_PROFILE,
    MAX_PROJECTED_SECONDS,
    QWEN_MODEL,
    QWEN_REVISION,
    SHA256_RE,
    SKIP_FIRST,
    SOURCE_DOSE_RUN_ID,
    V2_FINAL_LENS_FILENAME,
    V2_FIT_COUNT_CANDIDATES,
    V2_HOLDOUT_COUNT,
    V2_MAX_FIT_COUNT,
    V2_MAX_SEQ_LEN,
    config_hash,
    load_fit_checkpoint,
    require_dataset_identity,
    sha256_bytes,
    validate_checkpoint_identity,
    validate_checkpoint_payload,
    validate_fitted_lens,
)
from .qwen_lab_readout import FROZEN_LENS_SHA256
from .run_progress import EVENTS_FILENAME, STATUS_FILENAME

PASS_FILES = (
    "00_jlens_prepare.json",
    "01_jlens_probe.json",
    "02_jlens_fit.json",
    "03_jlens_holdout.json",
    "04_jlens_compare.json",
    "jlens_evaluation.json",
    "execution_manifest.json",
)
ABORTED_FILES = (
    "00_jlens_prepare.json",
    "01_jlens_probe.json",
    "jlens_evaluation.json",
    "execution_manifest.json",
)
COMPARISON_FILE = "04_jlens_compare.json"
ABORT_GATES = frozenset({"projected_runtime", "dim_batch_oom"})
COMPARISON_METHODS = frozenset({"jlens_v1", "jlens_v2", "logit_lens"})
CHECKPOINT_FILENAME = "jlens_fit_checkpoint.pt"
CORPUS_FILENAME = "corpus.jsonl"
HOLDOUT_PARTS = 2
FLOAT_TOLERANCE = 1e-9

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_STRING_PATTERNS = (
    SHA256_RE,
    COMMIT_RE,
    re.compile(r"^csn:\d+:[0-9a-f]{16}$"),
    re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.+Z-]+$"),
)
_CODE_MARKERS = re.compile(
    r"\bdef\s+\w+\s*\("
    r"|\bclass\s+\w+\s*[:(]"
    r"|^\s*(?:import\s+[\w.]+|from\s+[\w.]+\s+import\b.*)\s*$"
    r"|\"\"\"|'''"
    r"|\breturn\b"
    r"|\bprint\s*\("
    r"|\blambda\b"
    r"|\bself\.",
    re.MULTILINE,
)
_MIN_CORPUS_FRAGMENT_CHARS = 32
_MIN_CORPUS_FRAGMENT_TOKENS = 4
# Closed vocabularies written by the runtime; never raw text.
_ENUM_VALUES = frozenset(
    {
        "pass",
        "aborted",
        "completed",
        "running",
        "pending",
        "failed",
        "ok",
        "cuda_oom",
        "projected_runtime",
        "dim_batch_oom",
        "prepare",
        "probe",
        "fit",
        "holdout",
        "initialized",
        "state_changed",
        "progress",
        "probe_choice",
        "checkpoint",
        "stage_completed",
        "resume_recovered",
    }
)

# Stage initialization order. Resume re-enters ``prepare`` and then ``fit``;
# a restart before any selection may re-enter ``prepare`` from ``prepare`` or
# ``probe``. Nothing else is allowed.
ALLOWED_INIT_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"prepare"}),
    "prepare": frozenset({"prepare", "probe", "fit"}),
    "probe": frozenset({"prepare", "fit"}),
    "fit": frozenset({"holdout", "prepare", "fit"}),
    "holdout": frozenset({"prepare"}),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path.name} is missing") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} is unreadable: {type(exc).__name__}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _results_of(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{name} has no results object")
    return results


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    return float(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return str(value)


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= FLOAT_TOLERANCE


def _layers(config: Mapping[str, Any]) -> list[int]:
    return [int(layer) for layer in config["source_layers"]]


# ---------------------------------------------------------------------------
# Raw text leakage (keys and values)
# ---------------------------------------------------------------------------


def _flatten_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key)
            yield from _flatten_strings(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_strings(item)


def _load_corpus_texts(artifacts_dir: Path) -> list[str]:
    path = artifacts_dir / CORPUS_FILENAME
    if not path.is_file():
        return []
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        text = record.get("fit_text") if isinstance(record, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


class _LeakScanner:
    """Finds raw prompt/code text in keys or values without echoing it."""

    def __init__(self, *, allowed: set[str], corpus_texts: Sequence[str]) -> None:
        self._allowed = allowed
        self._texts = list(corpus_texts)
        self._blob = "\n".join(self._texts)
        self._shortest = min((len(text) for text in self._texts), default=0)
        self._memo: dict[tuple[str, bool, bool], str | None] = {}

    def _string_leaks(
        self, value: str, *, heuristics: bool, corpus: bool
    ) -> str | None:
        key = (value, heuristics, corpus)
        if key not in self._memo:
            self._memo[key] = self._classify(
                value, heuristics=heuristics, corpus=corpus
            )
        return self._memo[key]

    @staticmethod
    def _is_long_fragment(value: str) -> bool:
        """Prose-like and long enough that a corpus match cannot be a coincidence."""
        return (
            len(value) >= _MIN_CORPUS_FRAGMENT_CHARS
            and len(value.split()) >= _MIN_CORPUS_FRAGMENT_TOKENS
        )

    def _classify(
        self, value: str, *, heuristics: bool, corpus: bool
    ) -> str | None:
        if value in self._allowed or any(
            pattern.fullmatch(value) for pattern in _SAFE_STRING_PATTERNS
        ):
            return None
        if heuristics:
            if "\n" in value or "\r" in value:
                return "multi-line string"
            if _CODE_MARKERS.search(value):
                return "code-like string"
        # Schema keys and short identifiers (``error_message``, ``content_hash``)
        # occur in ordinary code, so only long multi-token VALUES are compared
        # with the raw corpus. Keys are never compared with it.
        if corpus and self._texts and self._is_long_fragment(value):
            if value in self._blob:
                return "string copied from the raw corpus"
            if len(value) >= self._shortest and any(
                len(text) >= _MIN_CORPUS_FRAGMENT_CHARS and text in value
                for text in self._texts
            ):
                return "string copied from the raw corpus"
        return None

    def find(self, obj: Any, where: str = "$") -> str | None:
        if isinstance(obj, Mapping):
            skip_message = obj.get("outcome") == "cuda_oom"
            for key, value in obj.items():
                key_text = str(key)
                location = f"{where}.{key_text[:40]}"
                if key_text.casefold() in JLENS_LEAK_KEYS:
                    return f"forbidden key at {where}: {key_text}"
                reason = self._string_leaks(key_text, heuristics=True, corpus=False)
                if reason:
                    return f"{reason} used as a key at {where}"
                if isinstance(value, str):
                    # CUDA OOM messages are free-form framework text. They keep
                    # the corpus-copy check but not the structural heuristics.
                    heuristics = not (skip_message and key_text == "error_message")
                    reason = self._string_leaks(
                        value, heuristics=heuristics, corpus=True
                    )
                    if reason:
                        return f"{reason} at {location}"
                else:
                    found = self.find(value, location)
                    if found:
                        return found
            return None
        if isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                if isinstance(item, str):
                    reason = self._string_leaks(item, heuristics=True, corpus=True)
                    if reason:
                        return f"{reason} at {where}[{index}]"
                else:
                    found = self.find(item, f"{where}[{index}]")
                    if found:
                        return found
            return None
        if isinstance(obj, str):
            reason = self._string_leaks(obj, heuristics=True, corpus=True)
            return f"{reason} at {where}" if reason else None
        return None


def _check_no_raw_text(
    named_payloads: Sequence[tuple[str, Any]],
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
) -> None:
    allowed = set(_flatten_strings(config)) | _ENUM_VALUES | {
        QWEN_MODEL,
        QWEN_REVISION,
        JLENS_UPSTREAM_COMMIT,
        DATASET_REPO_ID,
        DATASET_CONFIG,
        DATASET_SPLIT,
        DATASET_REVISION,
        SOURCE_DOSE_RUN_ID,
        FROZEN_LENS_SHA256,
        V2_FINAL_LENS_FILENAME,
        JLENS_V2_PROFILE,
    }
    scanner = _LeakScanner(
        allowed=allowed, corpus_texts=_load_corpus_texts(artifacts_dir)
    )
    for name, payload in named_payloads:
        found = scanner.find(payload)
        if found:
            raise ValueError(f"raw text found in {name}: {found}")


# ---------------------------------------------------------------------------
# Config, provenance and hashes
# ---------------------------------------------------------------------------


def _load_profile(root: Path) -> dict[str, Any]:
    try:
        return dict(load_config(root, profile=JLENS_V2_PROFILE))
    except Exception as exc:
        raise ValueError(
            f"v2 profile {JLENS_V2_PROFILE} is unreadable: {type(exc).__name__}"
        ) from None


def _check_profile(config: Mapping[str, Any]) -> None:
    _require(config.get("model_name") == QWEN_MODEL, "profile model_name is not pinned")
    _require(
        config.get("model_revision") == QWEN_REVISION,
        "profile model_revision is not pinned",
    )
    _require(
        config.get("jlens_upstream_commit") == JLENS_UPSTREAM_COMMIT,
        "profile jlens_upstream_commit is not pinned",
    )
    try:
        require_dataset_identity(config)
    except ValueError as exc:
        raise ValueError(f"profile dataset identity: {exc}") from None
    _require(
        [int(v) for v in config.get("fit_count_candidates") or []]
        == list(V2_FIT_COUNT_CANDIDATES),
        "profile fit count candidate list differs from the canonical list "
        f"{V2_FIT_COUNT_CANDIDATES}",
    )
    _require(
        int(config.get("max_fit_count", -1)) == V2_MAX_FIT_COUNT,
        f"profile max_fit_count must be {V2_MAX_FIT_COUNT}",
    )
    _require(
        int(config.get("holdout_count", -1)) == V2_HOLDOUT_COUNT,
        f"profile holdout count must be {V2_HOLDOUT_COUNT}",
    )
    _require(
        [int(v) for v in config.get("dim_batch_probe_order") or []]
        == list(DIM_BATCH_PROBE_ORDER),
        "profile dim_batch_probe_order differs from the canonical order",
    )
    _require(
        int(float(config.get("max_projected_hours", -1)) * 3600)
        == MAX_PROJECTED_SECONDS,
        "profile max_projected_hours differs from the 12 hour limit",
    )


def _packaged_profile() -> dict[str, Any]:
    package_path = (
        Path(__file__).resolve().parents[2] / "configs" / JLENS_V2_PROFILE
    )
    _require(package_path.is_file(), "packaged v2 profile is missing")
    return dict(json.loads(package_path.read_text(encoding="utf-8")))


def _check_root_profile_is_production(config: Mapping[str, Any]) -> None:
    """Strict mode: the run root profile must equal the packaged canonical profile.

    This covers the layer contract (n_layers, d_model, source_layers,
    target_layer, skip_first, max_seq_len, min tokens), every pin, the
    candidate list, and the profile-relative result, artifact and v1 lens paths.
    """
    package = _packaged_profile()
    for key in sorted(set(package) | set(config)):
        if key not in config:
            raise ValueError(f"production contract: root profile is missing {key}")
        if key not in package:
            raise ValueError(
                f"production contract: root profile has unexpected key {key}"
            )
        if config[key] != package[key]:
            raise ValueError(
                f"production contract: root profile {key} {config[key]!r} "
                f"differs from the packaged v2 profile {package[key]!r}"
            )


def _confined_path(root: Path, value: Any, label: str) -> Path:
    """Resolve a profile/artifact path under ``root``; reject escapes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path string")
    parts = [part for part in re.split(r"[\\/]+", value) if part not in ("", ".")]
    absolute = (
        value.startswith(("/", "\\"))
        or bool(re.match(r"^[A-Za-z]:", value))
        or Path(value).is_absolute()
    )
    if absolute:
        raise ValueError(
            f"{label} must be relative to the run root: absolute path rejected"
        )
    if ".." in parts:
        raise ValueError(
            f"{label} must stay inside the run root: parent traversal rejected"
        )
    if not parts:
        raise ValueError(f"{label} must name a path inside the run root")
    resolved_root = root.resolve()
    joined = root.joinpath(*parts)
    try:
        joined.resolve().relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            f"{label} must stay inside the run root: path resolves outside it"
        ) from None
    return joined


def _check_production_contract() -> None:
    package = _packaged_profile()
    _require(
        int(package.get("n_layers", -1)) == EXPECTED_N_LAYERS
        and int(package.get("d_model", -1)) == EXPECTED_D_MODEL
        and list(package.get("source_layers") or []) == EXPECTED_SOURCE_LAYERS
        and int(package.get("target_layer", -1)) == EXPECTED_TARGET_LAYER
        and int(package.get("skip_first", -1)) == SKIP_FIRST
        and int(package.get("max_seq_len", -1)) == V2_MAX_SEQ_LEN
        and int(package.get("min_tokens_before_truncation", -1)) == V2_MAX_SEQ_LEN,
        "production layer contract: packaged v2 profile differs from the "
        "28-layer, d_model 3584, source layers 0..26, target 27, 512 token contract",
    )
    _check_profile(package)


def _check_provenance(
    name: str,
    provenance: Any,
    *,
    run_id: str,
    commit: str,
) -> None:
    _require(isinstance(provenance, dict), f"provenance missing in {name}")
    expected = {
        "model_name": QWEN_MODEL,
        "requested_model_revision": QWEN_REVISION,
        "resolved_model_revision": QWEN_REVISION,
        "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "source_dose_run_id": SOURCE_DOSE_RUN_ID,
    }
    for field_name, wanted in expected.items():
        _require(
            provenance.get(field_name) == wanted,
            f"provenance {field_name} mismatch in {name}",
        )
    _require(
        provenance.get("run_id") == run_id,
        f"run_id mismatch in {name} provenance",
    )
    _require(
        provenance.get("implementation_commit") == commit
        and bool(COMMIT_RE.fullmatch(str(commit))),
        f"provenance implementation_commit mismatch in {name}",
    )


def _check_envelope(
    name: str,
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run_id: str,
    commit: str,
) -> None:
    _require(payload.get("schema_version") == 1, f"{name} schema_version must be 1")
    _check_provenance(name, payload.get("provenance"), run_id=run_id, commit=commit)
    _require(
        payload.get("config") == dict(config),
        f"config in {name} differs from the v2 profile",
    )


def _check_evaluation_identity(evaluation: Mapping[str, Any]) -> tuple[str, str]:
    run_id = evaluation.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id.strip()), "evaluation run_id is missing")
    commit = evaluation.get("implementation_commit")
    _require(
        isinstance(commit, str) and bool(COMMIT_RE.fullmatch(commit)),
        "evaluation implementation_commit is not a full commit hash",
    )
    _require(evaluation.get("schema_version") == 1, "evaluation schema_version must be 1")
    _require(
        evaluation.get("operational_success") is True,
        "evaluation operational_success must be true",
    )
    pins = {
        "model_revision": QWEN_REVISION,
        "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "source_dose_run_id": SOURCE_DOSE_RUN_ID,
        "source_dose_status": "pass",
    }
    for field_name, wanted in pins.items():
        _require(
            evaluation.get(field_name) == wanted,
            f"provenance {field_name} mismatch in jlens_evaluation.json",
        )
    return str(run_id), str(commit)


def _recompute_corpus_hash(prepare: Mapping[str, Any]) -> str:
    payload = [
        {
            "example_id": item["example_id"],
            "content_hash": item["content_hash"],
            "token_count": item["token_count"],
            "split": split,
        }
        for split in ("fit", "holdout")
        for item in prepare[split]
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _check_prepare_lists(prepare: Mapping[str, Any]) -> None:
    """Internal consistency of the compact corpus record; counts are checked elsewhere."""
    fit_ids = list(prepare.get("fit_ids") or [])
    holdout_ids = list(prepare.get("holdout_ids") or [])
    fit_hashes = list(prepare.get("fit_hashes") or [])
    holdout_hashes = list(prepare.get("holdout_hashes") or [])
    _require(
        not (set(fit_ids) & set(holdout_ids))
        and not (set(fit_hashes) & set(holdout_hashes)),
        "fit/holdout overlap in ids or content hashes",
    )
    _require(
        len(set(fit_ids)) == len(fit_ids)
        and len(set(holdout_ids)) == len(holdout_ids)
        and len(set(fit_hashes)) == len(fit_hashes)
        and len(set(holdout_hashes)) == len(holdout_hashes),
        "fit/holdout ids or content hashes are not unique",
    )
    for label, values in (("fit", fit_hashes), ("holdout", holdout_hashes)):
        _require(
            all(_is_sha256(item) for item in values),
            f"{label} content hashes are not SHA-256 digests",
        )
    for split, ids, hashes in (
        ("fit", fit_ids, fit_hashes),
        ("holdout", holdout_ids, holdout_hashes),
    ):
        items = prepare.get(split)
        _require(isinstance(items, list), f"prepare {split} item list is missing")
        _require(
            [item.get("example_id") for item in items] == ids
            and [item.get("content_hash") for item in items] == hashes,
            f"prepare {split} items disagree with {split}_ids/{split}_hashes",
        )


# ---------------------------------------------------------------------------
# Probe and selection
# ---------------------------------------------------------------------------


def _check_probe_common(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    _require(
        [int(v) for v in probe.get("probe_order") or []]
        == [int(v) for v in config["dim_batch_probe_order"]],
        "probe order differs from the profile",
    )
    _require(
        [int(v) for v in probe.get("fit_count_candidates") or []]
        == list(V2_FIT_COUNT_CANDIDATES),
        "probe fit count candidate list differs from the canonical list",
    )
    _require(
        int(probe.get("max_seq_len", -1)) == int(config["max_seq_len"]),
        "probe max_seq_len differs from the profile",
    )
    _require(
        _as_int(probe.get("projected_limit_seconds"), "probe projected_limit_seconds")
        == MAX_PROJECTED_SECONDS,
        "probe projected_limit_seconds differs from the 12 hour limit",
    )
    attempts = probe.get("attempts")
    order = list(probe["probe_order"])
    _require(
        isinstance(attempts, list) and 0 < len(attempts) <= len(order),
        "probe attempts are missing or exceed the probe order",
    )
    for index, attempt in enumerate(attempts):
        _require(
            int(attempt.get("dim_batch", -1)) == int(order[index]),
            "probe attempts do not follow the probe order",
        )
    wall = _as_float(probe.get("wall_seconds"), "probe wall_seconds")
    return list(attempts), wall


def _check_probe_pass(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    n_fit: int,
) -> int:
    attempts, wall = _check_probe_common(probe, config)
    _require(
        attempts[-1].get("outcome") == "ok"
        and all(item.get("outcome") == "cuda_oom" for item in attempts[:-1]),
        "probe attempts must be CUDA OOM retries followed by one ok attempt",
    )
    selected = _as_int(probe.get("selected_dim_batch"), "probe selected_dim_batch")
    _require(
        selected == int(attempts[-1]["dim_batch"]),
        "probe selected_dim_batch is not the successful attempt",
    )
    _require(
        _close(float(attempts[-1].get("wall_seconds", -1)), wall),
        "probe wall_seconds differs from the successful attempt",
    )
    limit = MAX_PROJECTED_SECONDS
    projected = _as_float(probe.get("projected_seconds"), "probe projected_seconds")
    _require(
        _close(projected, wall * n_fit),
        "probe projected_seconds is not wall_seconds times the selected fit count",
    )
    _require(
        probe.get("exceeds_limit") is False and projected <= limit,
        "a passed run cannot have a projection above the limit",
    )
    larger = [value for value in V2_FIT_COUNT_CANDIDATES if value > n_fit]
    _require(
        all(wall * value > limit for value in larger),
        f"selected fit count {n_fit} is not the largest safe candidate "
        f"for the probe wall_seconds",
    )
    return selected


def _check_probe_aborted(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    gate: str,
) -> int | None:
    attempts, wall = _check_probe_common(probe, config)
    _require(
        probe.get("selected_fit_count") is None,
        "aborted probe must have selected_fit_count null",
    )
    exceeds = probe.get("exceeds_limit")
    projected = _as_float(probe.get("projected_seconds"), "probe projected_seconds")
    selected = probe.get("selected_dim_batch")
    if gate == "projected_runtime":
        _require(
            exceeds is True,
            "abort gate projected_runtime requires exceeds_limit true",
        )
        _require(
            selected in config["dim_batch_probe_order"]
            and attempts[-1].get("outcome") == "ok"
            and int(attempts[-1]["dim_batch"]) == selected
            and all(item.get("outcome") == "cuda_oom" for item in attempts[:-1]),
            "abort gate projected_runtime requires a successful probe attempt",
        )
        _require(
            _close(float(attempts[-1].get("wall_seconds", -1)), wall),
            "probe wall_seconds differs from the successful attempt",
        )
    else:
        _require(
            selected is None
            and len(attempts) == len(probe["probe_order"])
            and all(item.get("outcome") == "cuda_oom" for item in attempts),
            "abort gate dim_batch_oom requires every probe attempt to be a CUDA OOM",
        )
        _require(
            _close(wall, sum(float(item.get("wall_seconds", 0)) for item in attempts)),
            "probe wall_seconds is not the sum of the OOM attempts",
        )
    minimum = min(V2_FIT_COUNT_CANDIDATES)
    _require(
        _close(projected, wall * minimum),
        "aborted probe projected_seconds is not wall_seconds times the minimum candidate",
    )
    _require(
        bool(exceeds) == (projected > MAX_PROJECTED_SECONDS),
        "probe exceeds_limit disagrees with projected_seconds",
    )
    return None if selected is None else int(selected)


# ---------------------------------------------------------------------------
# Lens and checkpoint
# ---------------------------------------------------------------------------


class _LoadedLens:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.jacobians = {int(layer): tensor for layer, tensor in payload["J"].items()}
        self.source_layers = [int(layer) for layer in payload["source_layers"]]
        self.n_prompts = int(payload["n_prompts"])
        self.d_model = int(payload["d_model"])


def _check_v2_lens(
    root: Path,
    artifacts_dir: Path,
    *,
    config: Mapping[str, Any],
    n_fit: int,
    evaluation: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> None:
    path = artifacts_dir / V2_FINAL_LENS_FILENAME
    _require(path.is_file(), f"v2 lens {V2_FINAL_LENS_FILENAME} is missing")
    try:
        recorded = _confined_path(root, fit.get("lens"), "v2 lens path in 02_jlens_fit.json")
    except ValueError as exc:
        raise ValueError(f"v2 lens: {exc}") from None
    _require(
        recorded == path,
        "v2 lens path in 02_jlens_fit.json is not exactly the artifacts_dir "
        f"path of {V2_FINAL_LENS_FILENAME}",
    )
    size = path.stat().st_size
    digest = sha256_file(path)
    for label, record in (("evaluation", evaluation), ("fit artifact", fit)):
        _require(
            _as_int(record.get("lens_bytes"), f"{label} lens_bytes") == size,
            f"v2 lens size differs from the {label}",
        )
        _require(
            _require_sha256(record.get("lens_sha256"), f"{label} lens_sha256") == digest,
            f"v2 lens SHA-256 differs from the {label}",
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raise ValueError("v2 lens is malformed or unreadable") from None
    _require(
        isinstance(payload, dict)
        and {"J", "source_layers", "n_prompts", "d_model"} <= set(payload)
        and isinstance(payload["J"], dict),
        "v2 lens is malformed: expected J, source_layers, n_prompts and d_model",
    )
    try:
        lens = _LoadedLens(payload)
        validate_fitted_lens(
            lens,
            source_layers=_layers(config),
            d_model=int(config["d_model"]),
            n_prompts=n_fit,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"v2 lens contract: {exc}") from None


def _check_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    n_fit: int,
    corpus_digest: str,
    config_digest: str,
    commit: str,
    run_id: str,
) -> None:
    try:
        validate_checkpoint_identity(
            checkpoint,
            source_layers=_layers(config),
            target_layer=int(config["target_layer"]),
            skip_first=int(config["skip_first"]),
            corpus_hash=corpus_digest,
            config_hash=config_digest,
            model_revision=QWEN_REVISION,
            implementation_commit=commit,
            run_id=run_id,
        )
        validate_checkpoint_payload(
            checkpoint,
            source_layers=_layers(config),
            d_model=int(config["d_model"]),
            dim_batch=int(checkpoint["dim_batch"]),
            fit_count=n_fit,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"checkpoint contract: {exc}") from None
    _require(
        int(checkpoint["n_done"]) == n_fit == int(checkpoint["next_idx"]),
        f"checkpoint n_done {checkpoint['n_done']} is not the completed fit count {n_fit}",
    )


# ---------------------------------------------------------------------------
# Holdout and comparison arithmetic
# ---------------------------------------------------------------------------

_STATS = ("rank_sum", "hits@1", "hits@5", "hits@10")


def _empty_counts() -> dict[str, int]:
    return {"n": 0, "rank_sum": 0, "hits@1": 0, "hits@5": 0, "hits@10": 0}


def _metric_from_counts(counts: Mapping[str, int]) -> dict[str, float | int]:
    n = counts["n"]
    return {
        "n_observations": n,
        "mean_rank": counts["rank_sum"] / n if n else 0.0,
        "pass@1": counts["hits@1"] / n if n else 0.0,
        "pass@5": counts["hits@5"] / n if n else 0.0,
        "pass@10": counts["hits@10"] / n if n else 0.0,
    }


def _item_counts(block: Mapping[str, Any], prefix: str, label: str) -> dict[str, int]:
    n = _as_int(block.get("n_positions"), f"{label} n_positions")
    counts = {"n": n}
    for stat in _STATS:
        counts[stat] = _as_int(block.get(f"{prefix}_{stat}"), f"{label} {prefix}_{stat}")
    _require(n > 0, f"{label} has no positions")
    _require(
        all(counts[stat] >= 0 for stat in _STATS)
        and counts["hits@1"] <= counts["hits@5"] <= counts["hits@10"] <= n
        and counts["rank_sum"] >= n,
        f"{label} {prefix} rank/hit counters are inconsistent",
    )
    return counts


def _add_counts(total: dict[str, int], counts: Mapping[str, int]) -> None:
    for key, value in counts.items():
        total[key] += value


def _recompose(
    by_item: Mapping[str, Any],
    ids: Sequence[str],
    layers: Sequence[int],
    prefix: str,
    label: str,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float | int]]:
    per_layer = {str(layer): _empty_counts() for layer in layers}
    overall = _empty_counts()
    for item_id in ids:
        item = by_item[item_id]
        for layer in layers:
            counts = _item_counts(
                item[str(layer)], prefix, f"{label} item {item_id} layer {layer}"
            )
            _add_counts(per_layer[str(layer)], counts)
            _add_counts(overall, counts)
    return (
        {layer: _metric_from_counts(counts) for layer, counts in per_layer.items()},
        _metric_from_counts(overall),
    )


def _metric_equal(stored: Any, expected: Mapping[str, float | int]) -> bool:
    if not isinstance(stored, dict) or set(stored) != set(expected):
        return False
    return all(
        isinstance(stored[key], (int, float))
        and not isinstance(stored[key], bool)
        and _close(stored[key], value)
        for key, value in expected.items()
    )


def _check_item_shape(
    by_item: Any,
    ids: Sequence[str],
    layers: Sequence[int],
    label: str,
) -> None:
    _require(isinstance(by_item, dict), f"{label} by_item is missing")
    _require(
        set(by_item) == set(ids) and len(by_item) == len(ids),
        f"{label} by_item ids differ from the holdout ids",
    )
    wanted = {str(layer) for layer in layers}
    for item_id in ids:
        item = by_item[item_id]
        _require(
            isinstance(item, dict) and set(item) == wanted,
            f"{label} item {item_id} does not cover every source layer",
        )


def _check_holdout(
    holdout: Mapping[str, Any],
    prepare: Mapping[str, Any],
    layers: Sequence[int],
) -> dict[str, Any]:
    ids = list(holdout.get("holdout_ids") or [])
    hashes = list(holdout.get("holdout_hashes") or [])
    _require(
        _as_int(holdout.get("n_items"), "holdout n_items") == V2_HOLDOUT_COUNT
        and len(ids) == V2_HOLDOUT_COUNT
        and len(set(ids)) == V2_HOLDOUT_COUNT
        and len(hashes) == V2_HOLDOUT_COUNT,
        f"holdout count in 03_jlens_holdout.json is not {V2_HOLDOUT_COUNT}",
    )
    _require(
        ids == list(prepare["holdout_ids"]) and hashes == list(prepare["holdout_hashes"]),
        "holdout ids/hashes in 03_jlens_holdout.json differ from the prepare artifact",
    )
    by_item = holdout.get("by_item")
    _check_item_shape(by_item, ids, layers, "holdout")
    by_layer = holdout.get("by_layer")
    overall = holdout.get("overall")
    _require(
        isinstance(by_layer, dict) and {int(k) for k in by_layer} == set(layers),
        "holdout by_layer does not cover every source layer",
    )
    total_positions = 0
    for item_id in ids:
        for layer in layers:
            total_positions += _as_int(
                by_item[item_id][str(layer)].get("n_positions"), "holdout n_positions"
            )
    _require(
        _as_int(holdout.get("n_records"), "holdout n_records") == total_positions,
        "holdout n_records differs from the per-item position counts",
    )
    for method in ("jlens", "logit_lens"):
        expected_layers, expected_overall = _recompose(
            by_item, ids, layers, method, "holdout"
        )
        for layer in layers:
            _require(
                _metric_equal((by_layer[str(layer)] or {}).get(method), expected_layers[str(layer)]),
                f"holdout metric {method} at layer {layer} does not recompose from by_item",
            )
        _require(
            isinstance(overall, dict)
            and _metric_equal(overall.get(method), expected_overall),
            f"holdout metric {method} overall does not recompose from by_item",
        )
    return {"ids": ids, "hashes": hashes, "by_item": by_item, "n_records": total_positions}


def _check_comparison(
    comparison: Mapping[str, Any],
    holdout_info: Mapping[str, Any],
    config: Mapping[str, Any],
    layers: Sequence[int],
) -> None:
    ids = holdout_info["ids"]
    methods = comparison.get("methods")
    _require(
        isinstance(methods, dict) and set(methods) == COMPARISON_METHODS,
        "comparison methods must be exactly jlens_v1, jlens_v2 and logit_lens",
    )
    _require(
        list(comparison.get("shared_holdout_ids") or []) == ids,
        "comparison shared_holdout_ids differ from 03_jlens_holdout.json",
    )
    _require(
        list(comparison.get("shared_holdout_hashes") or []) == holdout_info["hashes"],
        "comparison shared_holdout_hashes differ from 03_jlens_holdout.json",
    )
    _require(
        _as_int(comparison.get("n_items"), "comparison n_items") == len(ids)
        and _as_int(comparison.get("n_observations"), "comparison n_observations")
        == holdout_info["n_records"],
        "comparison n_items/n_observations differ from the holdout artifact",
    )
    by_item = comparison.get("by_item")
    _check_item_shape(by_item, ids, layers, "comparison")
    holdout_items = holdout_info["by_item"]
    for item_id in ids:
        for layer in layers:
            block = by_item[item_id][str(layer)]
            reference = holdout_items[item_id][str(layer)]
            _require(
                block.get("n_positions") == reference.get("n_positions"),
                f"comparison item {item_id} layer {layer} n_positions differ from the holdout",
            )
            for stat in _STATS:
                _require(
                    block.get(f"jlens_v2_{stat}") == reference.get(f"jlens_{stat}")
                    and block.get(f"logit_lens_{stat}") == reference.get(f"logit_lens_{stat}"),
                    f"comparison metric: jlens_v2/logit_lens {stat} of item {item_id} "
                    f"layer {layer} differs from the holdout",
                )
    for method in sorted(COMPARISON_METHODS):
        expected_layers, expected_overall = _recompose(
            by_item, ids, layers, method, "comparison"
        )
        stored = methods[method]
        _require(
            isinstance(stored, dict)
            and isinstance(stored.get("by_layer"), dict)
            and {int(k) for k in stored["by_layer"]} == set(layers),
            f"comparison methods {method} does not cover every source layer",
        )
        for layer in layers:
            _require(
                _metric_equal(stored["by_layer"][str(layer)], expected_layers[str(layer)]),
                f"comparison metric {method} at layer {layer} does not recompose from by_item",
            )
        _require(
            _metric_equal(stored.get("overall"), expected_overall),
            f"comparison metric {method} overall does not recompose from by_item",
        )
    _check_stability(comparison.get("stability"), by_item, ids, layers)
    _check_v1_lens(comparison.get("v1_lens"), config)


def _check_stability(
    stability: Any,
    by_item: Mapping[str, Any],
    ids: Sequence[str],
    layers: Sequence[int],
) -> None:
    _require(isinstance(stability, dict), "comparison stability is missing")
    parts = stability.get("parts")
    _require(
        stability.get("n_parts") == HOLDOUT_PARTS
        and isinstance(parts, list)
        and len(parts) == HOLDOUT_PARTS,
        f"comparison stability must have {HOLDOUT_PARTS} parts",
    )
    size = len(ids) // HOLDOUT_PARTS
    _require(
        stability.get("halves") == parts
        and [part.get("holdout_ids") for part in parts] == [ids[:size], ids[size:]],
        "comparison stability parts are not contiguous halves of the shared holdout",
    )
    for index, part in enumerate(parts):
        part_ids = list(part["holdout_ids"])
        overall = part.get("overall")
        _require(
            part.get("part") == index
            and part.get("n_items") == len(part_ids)
            and isinstance(overall, dict)
            and set(overall) == COMPARISON_METHODS,
            "comparison stability part is malformed",
        )
        n_records = 0
        for method in sorted(COMPARISON_METHODS):
            _, expected = _recompose(by_item, part_ids, layers, method, "comparison")
            n_records = expected["n_observations"]
            _require(
                _metric_equal(overall[method], expected),
                f"comparison metric stability part {index} {method} does not recompose",
            )
        _require(
            part.get("n_records") == n_records,
            f"comparison stability part {index} n_records is inconsistent",
        )


def _check_v1_lens(v1_lens: Any, config: Mapping[str, Any]) -> None:
    _require(isinstance(v1_lens, dict), "comparison v1_lens block is missing")
    _require(
        v1_lens.get("path") == config.get("source_v1_lens"),
        "comparison v1 lens path differs from the profile source_v1_lens",
    )
    _require(
        v1_lens.get("sha256") == FROZEN_LENS_SHA256,
        "comparison v1 lens sha256 differs from the canonical frozen v1 hash",
    )
    _require(
        isinstance(v1_lens.get("bytes"), int)
        and not isinstance(v1_lens.get("bytes"), bool)
        and v1_lens["bytes"] > 0,
        "comparison v1 lens bytes must be a positive integer",
    )
    _require(
        v1_lens.get("n_prompts") == FIT_COUNT,
        f"comparison v1 lens n_prompts must be {FIT_COUNT}",
    )


# ---------------------------------------------------------------------------
# Status, events and manifest
# ---------------------------------------------------------------------------


def _read_events(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{EVENTS_FILENAME} is missing")
    events: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            raise ValueError(f"{EVENTS_FILENAME} line {number} is not JSON") from None
        _require(isinstance(item, dict), f"{EVENTS_FILENAME} line {number} is not an object")
        events.append(item)
    return events


def _initialized_stages_resume_aware(stages: Sequence[Any]) -> bool:
    previous: str | None = None
    for stage in stages:
        if not isinstance(stage, str):
            return False
        allowed = ALLOWED_INIT_TRANSITIONS.get(previous)
        if allowed is None or stage not in allowed:
            return False
        previous = stage
    return True


def _check_events_common(events: Sequence[Mapping[str, Any]], run_id: str) -> None:
    _require(bool(events), f"{EVENTS_FILENAME} is empty")
    _require(
        all(item.get("run_id") == run_id for item in events),
        f"run_id mismatch in {EVENTS_FILENAME}",
    )
    stages = [item.get("stage") for item in events if item.get("event") == "initialized"]
    _require(
        _initialized_stages_resume_aware(stages),
        "events initialize stages in an invalid order",
    )


def _check_status_common(
    status: Mapping[str, Any], run_id: str
) -> None:
    _require(status.get("run_id") == run_id, "run_id mismatch in status.json")
    _require(
        status.get("profile") == JLENS_V2_PROFILE,
        "status.json profile is not the v2 profile",
    )


def _check_pass_progress(
    status: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    n_fit: int,
    n_holdout: int,
    selected_dim_batch: int,
) -> None:
    _check_status_common(status, run_id)
    _require(
        status.get("state") == "completed" and status.get("current_stage") == "holdout",
        "status must be completed at the holdout stage for a passed run",
    )
    _require(
        status.get("completed_units") == status.get("total_units") == n_holdout,
        "status completed_units/total_units must equal the holdout count",
    )
    _require(
        status.get("fit_completed_units") == status.get("fit_total_units") == n_fit,
        "status fit units must equal the selected fit count",
    )
    _check_events_common(events, run_id)
    stages = {item.get("stage") for item in events if item.get("event") == "initialized"}
    _require(
        {"prepare", "probe", "fit", "holdout"} <= stages,
        "events must initialize prepare, probe, fit and holdout",
    )
    checkpoints = [item for item in events if item.get("event") == "checkpoint"]
    done = [item.get("n_done") for item in checkpoints]
    _require(
        set(done) == set(range(1, n_fit + 1))
        and all(isinstance(value, int) and not isinstance(value, bool) for value in done),
        f"checkpoint events must cover n_done 1..{n_fit} exactly",
    )
    _require(
        all(item.get("total_units") == n_fit for item in checkpoints),
        "checkpoint events total_units differ from the selected fit count",
    )
    choices = [item for item in events if item.get("event") == "probe_choice"]
    _require(len(choices) == 1, "events must contain exactly one probe_choice")
    _require(
        choices[0].get("dim_batch") == selected_dim_batch,
        "selected dim_batch disagrees between probe_choice event and artifacts",
    )
    last = events[-1]
    _require(
        last.get("event") == "stage_completed"
        and last.get("stage") == "holdout"
        and last.get("state") == "completed",
        "events must end with the completed holdout stage",
    )


def _check_aborted_progress(
    status: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    gate: str,
    reason: Any,
) -> None:
    _check_status_common(status, run_id)
    _require(
        status.get("state") == "aborted"
        and status.get("abort_gate") == gate
        and status.get("abort_reason") == reason
        and status.get("current_stage") == "probe",
        "status must be aborted at the probe stage with the evaluation abort gate",
    )
    _check_events_common(events, run_id)
    stages = {item.get("stage") for item in events if item.get("event") == "initialized"}
    _require(
        stages <= {"prepare", "probe"}
        and not any(item.get("event") in {"checkpoint", "stage_completed"} for item in events),
        "events of an aborted pre-selection run must not contain fit work",
    )
    last = events[-1]
    _require(
        last.get("event") == "aborted" and last.get("abort_gate") == gate,
        "events must end with the aborted event of the evaluation gate",
    )


def _check_manifest(
    manifest: Mapping[str, Any],
    *,
    name: str,
    status: str,
    expected_files: Sequence[str],
) -> None:
    results = _results_of(manifest, name)
    _require(results.get("status") == status, "manifest status differs from the evaluation")
    _require(
        list(results.get("result_files") or []) == list(expected_files),
        f"manifest result_files differ from the documented set {list(expected_files)}",
    )


# ---------------------------------------------------------------------------
# Count agreement
# ---------------------------------------------------------------------------


def _require_agreement(label: str, values: Mapping[str, Any], expected: int) -> None:
    bad = {name: value for name, value in values.items() if value != expected}
    _require(
        not bad,
        f"{label} disagreement: expected {expected}, got {bad}",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_qwen_jlens_v2(
    root: Path, *, allow_fixture_contract: bool = False
) -> dict[str, bool]:
    """Validate the v2 results under ``root``; raise ``ValueError`` on the first problem.

    Strict production is the default: the run root profile must equal the
    packaged canonical v2 profile. ``allow_fixture_contract=True`` is only for
    CPU fixtures that intentionally use a tiny layer contract and a different
    directory layout; it relaxes that profile equality (the pins, candidate
    list, holdout count and path confinement are still enforced).
    """
    try:
        return _validate(root, allow_fixture_contract=allow_fixture_contract)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(
            f"malformed v2 profile or artifact: {type(exc).__name__}: {str(exc)[:120]}"
        ) from None


def _validate(root: Path, *, allow_fixture_contract: bool) -> dict[str, bool]:
    config = _load_profile(root)
    results_dir = _confined_path(root, config.get("results_dir"), "profile results_dir")
    artifacts_dir = _confined_path(
        root, config.get("artifacts_dir"), "profile artifacts_dir"
    )
    _confined_path(root, config.get("source_v1_lens"), "profile source_v1_lens")
    if not allow_fixture_contract:
        _check_root_profile_is_production(config)
    evaluation = _read_json(results_dir / "jlens_evaluation.json")
    status = evaluation.get("status")
    if status == "aborted":
        return _validate_aborted(
            config, results_dir, artifacts_dir, evaluation
        )
    _require(
        status == "pass",
        f"v2 evaluation status {status!r} is neither 'pass' nor 'aborted'",
    )
    return _validate_pass(root, config, results_dir, artifacts_dir, evaluation)


def _validate_pass(
    root: Path,
    config: dict[str, Any],
    results_dir: Path,
    artifacts_dir: Path,
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}

    n_fit = evaluation.get("n_fit")
    _require(
        isinstance(n_fit, int)
        and not isinstance(n_fit, bool)
        and n_fit in V2_FIT_COUNT_CANDIDATES,
        f"selected fit count candidate {n_fit!r} is not one of {V2_FIT_COUNT_CANDIDATES}",
    )
    checks["selected_fit_count_candidate"] = True
    _require(
        evaluation.get("n_holdout") == V2_HOLDOUT_COUNT,
        f"holdout count {evaluation.get('n_holdout')!r} is not {V2_HOLDOUT_COUNT}",
    )
    checks["holdout_count"] = True
    _check_profile(config)
    layers = _layers(config)
    checks["model_dataset_upstream_revisions"] = True

    missing = [name for name in PASS_FILES if not (results_dir / name).is_file()]
    if COMPARISON_FILE in missing:
        raise ValueError(
            f"a passed run requires the comparison artifact {COMPARISON_FILE}"
        )
    _require(not missing, f"required result artifacts are missing: {missing}")
    unexpected = sorted(
        path.name for path in results_dir.glob("*.json") if path.name not in PASS_FILES
    )
    _require(not unexpected, f"unexpected result artifacts: {unexpected}")
    checks["comparison_artifact"] = True

    run_id, commit = _check_evaluation_identity(evaluation)
    payloads = {name: _read_json(results_dir / name) for name in PASS_FILES}
    status_payload = _read_json(artifacts_dir / STATUS_FILENAME)
    events = _read_events(artifacts_dir / EVENTS_FILENAME)

    _check_no_raw_text(
        [
            *payloads.items(),
            (STATUS_FILENAME, status_payload),
            (EVENTS_FILENAME, events),
        ],
        config=config,
        artifacts_dir=artifacts_dir,
    )
    checks["no_raw_text_leakage"] = True

    for name, payload in payloads.items():
        if name != "jlens_evaluation.json":
            _check_envelope(name, payload, config=config, run_id=run_id, commit=commit)
    prepare = _results_of(payloads["00_jlens_prepare.json"], "00_jlens_prepare.json")
    probe = _results_of(payloads["01_jlens_probe.json"], "01_jlens_probe.json")
    fit = _results_of(payloads["02_jlens_fit.json"], "02_jlens_fit.json")
    holdout = _results_of(payloads["03_jlens_holdout.json"], "03_jlens_holdout.json")
    comparison = _results_of(payloads[COMPARISON_FILE], COMPARISON_FILE)

    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    _require(checkpoint_path.is_file(), f"checkpoint {CHECKPOINT_FILENAME} is missing")
    try:
        checkpoint = load_fit_checkpoint(checkpoint_path)
    except ValueError as exc:
        raise ValueError(f"checkpoint contract: {exc}") from None

    # Selected count and run id must agree everywhere before deeper checks.
    n_holdout = V2_HOLDOUT_COUNT
    checkpoint_events = [item for item in events if item.get("event") == "checkpoint"]
    probe_choices = [item for item in events if item.get("event") == "probe_choice"]
    _require_agreement(
        "selected fit count",
        {
            "evaluation.n_fit": evaluation.get("n_fit"),
            "probe.selected_fit_count": probe.get("selected_fit_count"),
            "fit.n_fit": fit.get("n_fit"),
            "fit.n_done": fit.get("n_done"),
            "prepare.fit_count": prepare.get("fit_count"),
            "prepare.len(fit_ids)": len(prepare.get("fit_ids") or []),
            "prepare.len(fit)": len(prepare.get("fit") or []),
            "prepare.len(fit_hashes)": len(prepare.get("fit_hashes") or []),
            "checkpoint.fit_count": checkpoint.get("fit_count"),
            "checkpoint.n_done": checkpoint.get("n_done"),
            "status.fit_total_units": status_payload.get("fit_total_units"),
            "status.fit_completed_units": status_payload.get("fit_completed_units"),
            **{
                f"probe_choice[{index}].fit_count": item.get("fit_count")
                for index, item in enumerate(probe_choices)
            },
            **{
                f"checkpoint_event[{index}].total_units": item.get("total_units")
                for index, item in enumerate(checkpoint_events[:1])
            },
        },
        n_fit,
    )
    _require_agreement(
        "holdout count",
        {
            "evaluation.n_holdout": evaluation.get("n_holdout"),
            "fit.n_holdout": fit.get("n_holdout"),
            "prepare.holdout_count": prepare.get("holdout_count"),
            "prepare.len(holdout_ids)": len(prepare.get("holdout_ids") or []),
            "prepare.len(holdout)": len(prepare.get("holdout") or []),
            "holdout.n_items": holdout.get("n_items"),
            "comparison.n_items": comparison.get("n_items"),
        },
        n_holdout,
    )
    checks["selected_count_agreement"] = True

    _check_prepare_lists(prepare)
    checks["fit_holdout_ids_hashes"] = True

    corpus_digest = _require_sha256(prepare.get("corpus_hash"), "prepare corpus_hash")
    config_digest = _require_sha256(prepare.get("config_hash"), "prepare config_hash")
    _require(
        _recompute_corpus_hash(prepare) == corpus_digest,
        "corpus hash in 00_jlens_prepare.json does not match its fit/holdout items",
    )
    _require(
        config_digest == config_hash(config),
        "config hash does not match the v2 profile",
    )
    for label, record in (
        ("evaluation", evaluation),
        ("fit artifact", fit),
        ("checkpoint", checkpoint),
    ):
        _require(
            record.get("corpus_hash") == corpus_digest,
            f"corpus hash differs between prepare and {label}",
        )
        _require(
            record.get("config_hash") == config_digest,
            f"config hash differs between prepare and {label}",
        )
    checks["corpus_and_config_hash"] = True

    selected_dim_batch = _check_probe_pass(probe, config, n_fit=n_fit)
    _require_agreement(
        "selected dim_batch",
        {
            "probe": probe.get("selected_dim_batch"),
            "fit.selected_dim_batch": fit.get("selected_dim_batch"),
            "evaluation.selected_dim_batch": evaluation.get("selected_dim_batch"),
            "checkpoint.dim_batch": checkpoint.get("dim_batch"),
        },
        selected_dim_batch,
    )
    checks["probe_selection"] = True

    _check_checkpoint(
        checkpoint,
        config=config,
        n_fit=n_fit,
        corpus_digest=corpus_digest,
        config_digest=config_digest,
        commit=commit,
        run_id=run_id,
    )
    checks["checkpoint_identity"] = True

    _check_v2_lens(
        root, artifacts_dir, config=config, n_fit=n_fit, evaluation=evaluation, fit=fit
    )
    checks["v2_lens_identity"] = True

    holdout_info = _check_holdout(holdout, prepare, layers)
    checks["holdout_complete_and_arithmetic"] = True

    _check_comparison(comparison, holdout_info, config, layers)
    checks["comparison_shared_holdout"] = True
    checks["comparison_methods"] = True
    checks["v1_lens_canonical_hash"] = True

    _check_pass_progress(
        status_payload,
        events,
        run_id=run_id,
        n_fit=n_fit,
        n_holdout=n_holdout,
        selected_dim_batch=selected_dim_batch,
    )
    _check_manifest(
        payloads["execution_manifest.json"],
        name="execution_manifest.json",
        status="pass",
        expected_files=PASS_FILES,
    )
    checks["run_status_events_manifest"] = True

    _check_production_contract()
    checks["production_layer_contract"] = True
    return checks


def _validate_aborted(
    config: dict[str, Any],
    results_dir: Path,
    artifacts_dir: Path,
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    gate = evaluation.get("abort_gate")
    _require(gate in ABORT_GATES, f"abort gate {gate!r} is not one of {sorted(ABORT_GATES)}")
    checks["abort_gate"] = True
    _require(
        evaluation.get("n_fit") is None,
        "aborted run must report n_fit null (no fit count was selected)",
    )
    _require(
        evaluation.get("n_holdout") == V2_HOLDOUT_COUNT,
        f"holdout count {evaluation.get('n_holdout')!r} is not {V2_HOLDOUT_COUNT}",
    )
    checks["holdout_count"] = True
    _check_profile(config)
    checks["model_dataset_upstream_revisions"] = True

    present = {path.name for path in results_dir.glob("*.json")}
    _require(
        present == set(ABORTED_FILES),
        "aborted run must contain only the documented pre-selection result set "
        f"{list(ABORTED_FILES)}, found {sorted(present)}",
    )
    stray = sorted(path.name for path in artifacts_dir.glob("*.pt"))
    _require(
        not stray,
        f"aborted run must have no pre-selection checkpoint or lens, found {stray}",
    )
    checks["aborted_pre_selection_artifacts"] = True

    run_id, commit = _check_evaluation_identity(evaluation)
    payloads = {name: _read_json(results_dir / name) for name in ABORTED_FILES}
    status_payload = _read_json(artifacts_dir / STATUS_FILENAME)
    events = _read_events(artifacts_dir / EVENTS_FILENAME)
    _check_no_raw_text(
        [
            *payloads.items(),
            (STATUS_FILENAME, status_payload),
            (EVENTS_FILENAME, events),
        ],
        config=config,
        artifacts_dir=artifacts_dir,
    )
    checks["no_raw_text_leakage"] = True

    for name, payload in payloads.items():
        if name != "jlens_evaluation.json":
            _check_envelope(name, payload, config=config, run_id=run_id, commit=commit)
    prepare = _results_of(payloads["00_jlens_prepare.json"], "00_jlens_prepare.json")
    probe = _results_of(payloads["01_jlens_probe.json"], "01_jlens_probe.json")

    pool = V2_MAX_FIT_COUNT + V2_HOLDOUT_COUNT
    _require(
        prepare.get("fit_count") == V2_MAX_FIT_COUNT
        and prepare.get("holdout_count") == V2_HOLDOUT_COUNT
        and prepare.get("n_selected") == pool
        and len(prepare.get("fit_ids") or []) == V2_MAX_FIT_COUNT
        and len(prepare.get("fit") or []) == V2_MAX_FIT_COUNT
        and len(prepare.get("holdout_ids") or []) == V2_HOLDOUT_COUNT
        and len(prepare.get("holdout") or []) == V2_HOLDOUT_COUNT,
        "aborted prepare artifact must describe the full candidate pool "
        f"({V2_MAX_FIT_COUNT} fit + {V2_HOLDOUT_COUNT} holdout)",
    )
    _check_prepare_lists(prepare)
    checks["fit_holdout_ids_hashes"] = True

    corpus_digest = _require_sha256(prepare.get("corpus_hash"), "prepare corpus_hash")
    config_digest = _require_sha256(prepare.get("config_hash"), "prepare config_hash")
    _require(
        _recompute_corpus_hash(prepare) == corpus_digest,
        "corpus hash in 00_jlens_prepare.json does not match the full candidate pool",
    )
    _require(
        config_digest == config_hash(config),
        "config hash does not match the v2 profile",
    )
    _require(
        evaluation.get("corpus_hash") == corpus_digest
        and evaluation.get("config_hash") == config_digest,
        "corpus/config hash differs between prepare and evaluation",
    )
    checks["corpus_and_config_hash"] = True

    probe_selected = _check_probe_aborted(probe, config, gate=str(gate))
    _require(
        evaluation.get("selected_dim_batch") == probe_selected
        or (probe_selected is None and "selected_dim_batch" not in evaluation),
        "selected dim_batch disagrees between probe and evaluation",
    )
    _require(
        evaluation.get("stage") == "probe",
        "aborted evaluation must be at the probe stage",
    )
    checks["probe_selection"] = True

    _check_aborted_progress(
        status_payload,
        events,
        run_id=run_id,
        gate=str(gate),
        reason=evaluation.get("abort_reason"),
    )
    _check_manifest(
        payloads["execution_manifest.json"],
        name="execution_manifest.json",
        status="aborted",
        expected_files=ABORTED_FILES,
    )
    checks["run_status_events_manifest"] = True

    _check_production_contract()
    checks["production_layer_contract"] = True
    return checks
