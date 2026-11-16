"""Model-independent helpers for the Qwen dose follow-up."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from .data import (
    PromptExample,
    content_hashes_for_examples,
    load_local_prompt_corpus,
    prompt_content_hash,
)
from .gates import (
    OPERATIONAL_ERROR_MESSAGES,
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    GateResult,
)


DOSE_SELECTION_SIZE = 32
DOSE_CONFIRMATION_SIZE = 32
STATUS_NO_DOSE_QUALIFIED = "no_dose_qualified"
EXPECTED_SOURCE_LAYER = 18
EXPECTED_SOURCE_OFFSET = -1
DEFAULT_MIN_HARMFUL_ABLATION_DROP = 0.20
DEFAULT_MAX_HARMLESS_ABLATION_KL = 0.10
DEFAULT_MIN_HARMLESS_ADDITION_INCREASE = 0.20
ORIGINAL_SPLIT_KEYS = (
    "harmful_train",
    "harmless_train",
    "harmful_val",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)
ORIGINAL_SOURCE_ENVELOPES = (
    "00_preflight.json",
    "01_baseline.json",
    "02_candidates.json",
    "03_interventions.json",
    "final_evaluation.json",
    "execution_manifest.json",
    "pilot_evaluation.json",
)
ABLATION_METRIC_FIELDS = (
    "coefficient",
    "harmful_refusal_drop",
    "harmless_kl",
)
ADDITION_METRIC_FIELDS = (
    "coefficient",
    "harmless_refusal_increase",
)
METADATA_LEAK_KEYS = frozenset(
    {
        "text",
        "instruction",
        "prompt",
        "prompts",
        "generation",
        "generations",
        "outputs",
        "responses",
        "raw",
        "generated",
        "completion",
        "completions",
    }
)
ALLOWED_OPERATIONAL_MESSAGES = frozenset(OPERATIONAL_ERROR_MESSAGES.values())
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")
_GATE_REASON_RE = re.compile(
    r"^[a-z0-9_]+: observed [-+0-9.eE]+(?: not)? (?:>=|<=) [-+0-9.eE]+$"
)
_GATE_PASSED_RE = re.compile(r"^[a-z0-9_]+ passed$")
_PREFIX_RE = re.compile(r"^[\w' ]{1,32}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.json$")
_NO_DOSE_REASON = "no qualifying dose pair"
_DOSE_STATUSES = frozenset(
    {
        STATUS_PASS,
        STATUS_CAUSAL_GATE_FAILED,
        STATUS_OPERATIONAL_ERROR,
        STATUS_NO_DOSE_QUALIFIED,
    }
)
_EXPERIMENTS = frozenset(
    {
        "dose_prepare",
        "dose_baseline",
        "dose_selection",
        "dose_checkpoint",
        "dose_confirmation",
        "execution_manifest",
    }
)
_ABORT_GATES = frozenset({"dose", "causal"})
_EVENT_NAMES = frozenset(
    {
        "initialized",
        "state_changed",
        "progress",
        "stage_completed",
        "aborted",
        "failed",
    }
)
_STATES = frozenset({"pending", "running", "completed", "aborted", "failed"})
_STAGES = frozenset({"prepare", "sweep", "confirm"})
_COMPACT_STRING_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("experiment",), "experiment"),
    (("config", "model_name"), "token"),
    (("config", "model_revision"), "hash"),
    (("config", "dtype"), "token"),
    (("config", "upstream_commit"), "hash"),
    (("config", "results_dir"), "path"),
    (("config", "artifacts_dir"), "path"),
    (("config", "format_style"), "token"),
    (("config", "selection_split"), "token"),
    (("config", "final_conditions_split"), "token"),
    (("config", "source_direction"), "path"),
    (("config", "source_evidence", "baseline"), "path"),
    (("config", "source_evidence", "interventions"), "path"),
    (("config", "refusal_prefixes", "*"), "prefix"),
    (("provenance", "run_id"), "token"),
    (("provenance", "implementation_commit"), "hash"),
    (("provenance", "model_name"), "token"),
    (("provenance", "requested_model_revision"), "hash"),
    (("provenance", "resolved_model_revision"), "hash"),
    (("provenance", "upstream_commit"), "hash"),
    (("provenance", "source_pilot_run_id"), "token"),
    (("provenance", "source_pilot_implementation_commit"), "hash"),
    (("provenance", "source_artifact_hashes", "*"), "hash"),
    (("results", "source_pilot_status"), "status"),
    (("results", "source_pilot_run_id"), "token"),
    (("results", "source_pilot_implementation_commit"), "hash"),
    (("results", "splits", "source_revision"), "token"),
    (("results", "completed_stages", "*"), "stage"),
    (("results", "result_files", "*"), "filename"),
    (("results", "status"), "status"),
)
_EVALUATION_STRING_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("status",), "status"),
    (("run_id",), "token"),
    (("implementation_commit",), "hash"),
    (("model_revision",), "hash"),
    (("source_pilot_run_id",), "token"),
    (("source_pilot_implementation_commit",), "hash"),
    (("source_artifact_hashes", "*"), "hash"),
    (("causal_gate", "name"), "token"),
    (("causal_gate", "reason"), "joined_gate_reason"),
    (("causal_gate", "checks", "*", "name"), "token"),
    (("causal_gate", "checks", "*", "reason"), "gate_reason"),
    (("causal_gate", "checks", "*", "comparator"), "token"),
    (("stage",), "stage"),
    (("exception_category",), "token"),
    (("error_code",), "token"),
    (("message",), "catalog_message"),
    (("abort_reason",), "abort_reason"),
)
_STATUS_STRING_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("profile",), "token"),
    (("run_id",), "token"),
    (("current_stage",), "stage"),
    (("state",), "state"),
    (("updated_at_utc",), "iso"),
    (("message",), "catalog_message"),
    (("abort_gate",), "abort_gate"),
    (("abort_reason",), "abort_reason"),
)
_EVENT_STRING_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("event",), "event_name"),
    (("state",), "state"),
    (("stage",), "stage"),
    (("run_id",), "token"),
    (("updated_at_utc",), "iso"),
    (("message",), "catalog_message"),
    (("abort_gate",), "abort_gate"),
    (("abort_reason",), "abort_reason"),
)
_SPLIT_STRING_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("example_ids", "*"), "id"),
    (("example_ids", "*", "*"), "id"),
    (("content_hashes", "*", "*"), "hash"),
    (("source_revision",), "token"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def coefficient_checkpoint_name(coefficient: float) -> str:
    return f"checkpoint_coeff_{format(float(coefficient), 'g')}.json"


def source_envelope_identity(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, str]:
    if name == "pilot_evaluation.json":
        run_id = payload.get("run_id")
        commit = payload.get("implementation_commit")
        missing = [
            field
            for field, value in (("run_id", run_id), ("implementation_commit", commit))
            if not value
        ]
        if missing:
            raise ValueError(
                "source envelope is missing identity fields: " + ", ".join(missing)
            )
        if payload.get("status") != STATUS_CAUSAL_GATE_FAILED:
            raise ValueError("source pilot must remain causal_gate_failed")
        return {
            "run_id": str(run_id),
            "implementation_commit": str(commit),
        }
    provenance = payload.get("provenance") or {}
    config = payload.get("config") or {}
    run_id = provenance.get("run_id") or payload.get("run_id")
    commit = provenance.get("implementation_commit") or payload.get(
        "implementation_commit"
    )
    model_name = (
        provenance.get("model_name")
        or config.get("model_name")
        or payload.get("model_name")
    )
    revision = (
        provenance.get("requested_model_revision")
        or config.get("model_revision")
        or payload.get("model_revision")
    )
    missing = [
        field
        for field, value in (
            ("run_id", run_id),
            ("implementation_commit", commit),
            ("model_name", model_name),
            ("model_revision", revision),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "source envelope is missing identity fields: " + ", ".join(missing)
        )
    return {
        "run_id": str(run_id),
        "implementation_commit": str(commit),
        "model_name": str(model_name),
        "model_revision": str(revision),
    }


def load_source_envelopes(root: Path) -> dict[str, dict[str, Any]]:
    results_dir = root / "results" / "qwen25_coder_7b"
    envelopes: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    for name in ORIGINAL_SOURCE_ENVELOPES:
        path = results_dir / name
        if not path.is_file():
            raise ValueError(f"source envelope missing: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"source envelope is not an object: {name}")
        envelopes[name] = payload
        identities[name] = source_envelope_identity(payload, name=name)
    compact_names = [
        name for name in ORIGINAL_SOURCE_ENVELOPES if name != "pilot_evaluation.json"
    ]
    first = identities[compact_names[0]]
    for name in compact_names:
        if identities[name] != first:
            raise ValueError(
                f"source envelope {name} does not share run_id/commit/model"
            )
    pilot = identities["pilot_evaluation.json"]
    if first["run_id"] != pilot["run_id"] or first["implementation_commit"] != pilot[
        "implementation_commit"
    ]:
        raise ValueError(
            "source envelope pilot_evaluation.json does not share run_id/commit"
        )
    return envelopes


def source_artifact_hashes(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    evidence = config["source_evidence"]
    return {
        "selected_direction": sha256_file(root / config["source_direction"]),
        "baseline": sha256_file(root / evidence["baseline"]),
        "interventions": sha256_file(root / evidence["interventions"]),
        "pilot_evaluation": sha256_file(
            root / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
        ),
    }


@dataclass(frozen=True)
class DoseSplits:
    harmful_dose_selection: tuple[PromptExample, ...]
    harmless_dose_selection: tuple[PromptExample, ...]
    harmful_dose_confirmation: tuple[PromptExample, ...]
    harmless_dose_confirmation: tuple[PromptExample, ...]


def _require_usage_list(value: object, *, field: str, key: str) -> list:
    if not isinstance(value, list):
        raise ValueError(
            f"{field}[{key!r}] must be a list, got {type(value).__name__}"
        )
    return value


def _extend_usage_groups(
    groups: dict[str, list[str]],
    mapping: object,
    *,
    field: str,
) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{field} must be an object mapping split names to lists")
    for key, value in mapping.items():
        items = _require_usage_list(value, field=field, key=str(key))
        groups.setdefault(str(key), []).extend(str(item) for item in items)


def collect_original_usage(
    results_dir: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    id_groups: dict[str, list[str]] = {}
    hash_groups: dict[str, list[str]] = {}
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        results = payload.get("results") or {}
        if not isinstance(results, dict):
            continue
        example_ids = results.get("example_ids")
        content_hashes = results.get("content_hashes")
        if example_ids is None and content_hashes is None:
            continue
        if example_ids is None or content_hashes is None:
            raise ValueError(
                f"{path.name} must record both example_ids and content_hashes"
            )
        _extend_usage_groups(id_groups, example_ids, field="example_ids")
        _extend_usage_groups(hash_groups, content_hashes, field="content_hashes")
    missing = [
        key
        for key in ORIGINAL_SPLIT_KEYS
        if not id_groups.get(key) or not hash_groups.get(key)
    ]
    if missing:
        raise ValueError(
            "original usage missing required split keys: " + ", ".join(missing)
        )
    ids = {item for group in id_groups.values() for item in group}
    hashes = {item for group in hash_groups.values() for item in group}
    return frozenset(ids), frozenset(hashes)


def original_source_revision(results_dir: Path) -> str:
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        provenance = payload.get("provenance") or {}
        revision = provenance.get("upstream_commit")
        if revision:
            return str(revision)
    raise ValueError("original results do not record a source revision")


def eligible_examples(
    examples: Sequence[PromptExample],
    *,
    used_ids: Set[str],
    used_hashes: Set[str],
) -> list[PromptExample]:
    leftover: list[PromptExample] = []
    for example in examples:
        if example.id in used_ids:
            continue
        if prompt_content_hash(example.text) in used_hashes:
            continue
        leftover.append(example)
    return leftover


def sample_dose_halves(
    examples: Sequence[PromptExample],
    *,
    seed: int,
    selection_size: int = DOSE_SELECTION_SIZE,
    confirmation_size: int = DOSE_CONFIRMATION_SIZE,
) -> tuple[list[PromptExample], list[PromptExample]]:
    required = selection_size + confirmation_size
    if len(examples) < required:
        raise ValueError(f"need {required} examples, received {len(examples)}")
    sampled = random.Random(seed).sample(list(examples), required)
    return sampled[:selection_size], sampled[selection_size:]


def _path_matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if len(path) != len(pattern):
        return False
    return all(expected == "*" or expected == actual for expected, actual in zip(pattern, path))


def _infer_artifact_kind(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "generic"
    if "experiment" in obj and "config" in obj:
        return "compact"
    if "causal_gate" in obj and "status" in obj:
        return "evaluation"
    if "current_stage" in obj and "profile" in obj:
        return "status"
    if "event" in obj and "stage" in obj:
        return "event"
    if "example_ids" in obj and "content_hashes" in obj:
        return "split_metadata"
    return "generic"


def _joined_gate_reason(value: str) -> bool:
    if _GATE_PASSED_RE.fullmatch(value):
        return True
    parts = [item.strip() for item in value.split(";")]
    return bool(parts) and all(bool(_GATE_REASON_RE.fullmatch(item)) for item in parts)


def _abort_reason_allowed(value: str) -> bool:
    if value in {_NO_DOSE_REASON, *_DOSE_STATUSES}:
        return True
    if value in ALLOWED_OPERATIONAL_MESSAGES:
        return True
    return _joined_gate_reason(value)


def _string_allowed(kind: str, value: str) -> bool:
    if kind == "token":
        return bool(_TOKEN_RE.fullmatch(value))
    if kind == "id":
        return bool(_ID_RE.fullmatch(value))
    if kind == "hash":
        return bool(_HASH_RE.fullmatch(value))
    if kind == "path":
        return bool(_PATH_RE.fullmatch(value))
    if kind == "iso":
        return bool(_ISO_RE.match(value))
    if kind == "status":
        return value in _DOSE_STATUSES
    if kind == "experiment":
        return value in _EXPERIMENTS
    if kind == "prefix":
        return bool(_PREFIX_RE.fullmatch(value)) and len(value.split()) <= 4
    if kind == "filename":
        return bool(_FILENAME_RE.fullmatch(value))
    if kind == "gate_reason":
        return bool(_GATE_REASON_RE.fullmatch(value))
    if kind == "joined_gate_reason":
        return _joined_gate_reason(value)
    if kind == "abort_reason":
        return _abort_reason_allowed(value)
    if kind == "catalog_message":
        return value in ALLOWED_OPERATIONAL_MESSAGES
    if kind == "abort_gate":
        return value in _ABORT_GATES
    if kind == "event_name":
        return value in _EVENT_NAMES
    if kind == "state":
        return value in _STATES
    if kind == "stage":
        return value in _STAGES
    return False


def _rule_table(kind: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    if kind == "compact":
        return _COMPACT_STRING_PATHS
    if kind == "evaluation":
        return _EVALUATION_STRING_PATHS
    if kind == "status":
        return _STATUS_STRING_PATHS
    if kind == "event":
        return _EVENT_STRING_PATHS
    if kind == "split_metadata":
        return _SPLIT_STRING_PATHS
    return (
        _EVALUATION_STRING_PATHS
        + _STATUS_STRING_PATHS
        + _EVENT_STRING_PATHS
        + _SPLIT_STRING_PATHS
    )


def _compact_structural_string(path: tuple[str, ...], value: str) -> bool:
    if len(path) >= 2 and path[-2] in {"valid_ids", "invalid_ids", "example_ids"}:
        return _string_allowed("id", value)
    if len(path) >= 3 and path[-3] == "example_ids":
        return _string_allowed("id", value)
    if len(path) >= 3 and path[-3] == "content_hashes":
        return _string_allowed("hash", value)
    return False


def _string_path_allowed(kind: str, path: tuple[str, ...], value: str) -> bool:
    if any(part.casefold() in METADATA_LEAK_KEYS for part in path):
        return False
    if kind in {"compact", "generic"} and _compact_structural_string(path, value):
        return True
    for pattern, rule in _rule_table(kind):
        if _path_matches(path, pattern):
            return _string_allowed(rule, value)
    return False


def metadata_contains_leak(
    obj: Any,
    *,
    key: str | None = None,
    kind: str | None = None,
    path: tuple[str, ...] = (),
) -> bool:
    if kind is None and not path:
        resolved_kind = _infer_artifact_kind(obj)
    else:
        resolved_kind = kind or "generic"
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            lowered = str(child_key).casefold()
            if lowered in METADATA_LEAK_KEYS:
                return True
            child_path = path + (str(child_key),)
            if metadata_contains_leak(
                value,
                key=str(child_key),
                kind=resolved_kind,
                path=child_path,
            ):
                return True
        return False
    if isinstance(obj, list):
        return any(
            metadata_contains_leak(
                item,
                key=key,
                kind=resolved_kind,
                path=path + ("*",),
            )
            for item in obj
        )
    if isinstance(obj, str):
        if (key or "").casefold() in METADATA_LEAK_KEYS:
            return True
        return not _string_path_allowed(resolved_kind, path, obj)
    return False


def compact_split_metadata(
    *,
    harmful_selection: Sequence[PromptExample],
    harmless_selection: Sequence[PromptExample],
    harmful_confirmation: Sequence[PromptExample],
    harmless_confirmation: Sequence[PromptExample],
    seed: int,
    source_revision: str,
    harmless_seed: int | None = None,
) -> dict[str, Any]:
    groups = {
        "harmful_dose_selection": harmful_selection,
        "harmless_dose_selection": harmless_selection,
        "harmful_dose_confirmation": harmful_confirmation,
        "harmless_dose_confirmation": harmless_confirmation,
    }
    payload = {
        "example_ids": {
            name: [example.id for example in items]
            for name, items in groups.items()
        },
        "content_hashes": {
            name: content_hashes_for_examples(items)
            for name, items in groups.items()
        },
        "counts": {name: len(items) for name, items in groups.items()},
        "seed": {
            "harmful": seed,
            "harmless": seed + 1 if harmless_seed is None else harmless_seed,
        },
        "source_revision": source_revision,
    }
    if metadata_contains_leak(payload):
        raise ValueError("compact split metadata must not contain prompt text")
    return payload


def _require_leftovers(
    examples: Sequence[PromptExample],
    *,
    label: str,
    required: int,
) -> None:
    if len(examples) < required:
        raise ValueError(
            f"need {required} leftover {label} examples, received {len(examples)}"
        )


def _split_groups(splits: DoseSplits) -> dict[str, Sequence[PromptExample]]:
    return {
        "harmful_dose_selection": splits.harmful_dose_selection,
        "harmless_dose_selection": splits.harmless_dose_selection,
        "harmful_dose_confirmation": splits.harmful_dose_confirmation,
        "harmless_dose_confirmation": splits.harmless_dose_confirmation,
    }


def _half_ids(examples: Sequence[PromptExample]) -> list[str]:
    return [example.id for example in examples]


def _half_hashes(examples: Sequence[PromptExample]) -> list[str]:
    return [prompt_content_hash(example.text) for example in examples]


def _assert_disjoint_splits(
    splits: DoseSplits,
    *,
    used_ids: Set[str],
    used_hashes: Set[str],
    selection_size: int,
    confirmation_size: int,
) -> None:
    groups = _split_groups(splits)
    expected_counts = {
        "harmful_dose_selection": selection_size,
        "harmless_dose_selection": selection_size,
        "harmful_dose_confirmation": confirmation_size,
        "harmless_dose_confirmation": confirmation_size,
    }
    identities: dict[str, tuple[list[str], list[str]]] = {}
    for name, items in groups.items():
        expected = expected_counts[name]
        if len(items) != expected:
            raise ValueError(
                f"{name} has {len(items)} examples, expected {expected}"
            )
        ids = _half_ids(items)
        hashes = _half_hashes(items)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} has duplicate example ids")
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{name} has duplicate content hashes")
        identities[name] = (ids, hashes)
    new_ids = {item for ids, _hashes in identities.values() for item in ids}
    new_hashes = {
        item for _ids, hashes in identities.values() for item in hashes
    }
    if not used_ids.isdisjoint(new_ids):
        raise ValueError("dose splits overlap original example ids")
    if not used_hashes.isdisjoint(new_hashes):
        raise ValueError("dose splits overlap original content hashes")
    names = list(identities)
    for index, left_name in enumerate(names):
        left_ids, left_hashes = identities[left_name]
        for right_name in names[index + 1 :]:
            right_ids, right_hashes = identities[right_name]
            if not set(left_ids).isdisjoint(right_ids):
                raise ValueError(
                    f"{left_name} and {right_name} overlap example ids"
                )
            if not set(left_hashes).isdisjoint(right_hashes):
                raise ValueError(
                    f"{left_name} and {right_name} overlap content hashes"
                )


def build_dose_splits(
    *,
    original_results_dir: Path,
    harmful_test_path: Path,
    harmless_test_path: Path,
    seed: int,
    selection_size: int = DOSE_SELECTION_SIZE,
    confirmation_size: int = DOSE_CONFIRMATION_SIZE,
) -> tuple[DoseSplits, dict[str, Any]]:
    used_ids, used_hashes = collect_original_usage(original_results_dir)
    required = selection_size + confirmation_size
    harmful_leftover = eligible_examples(
        load_local_prompt_corpus(harmful_test_path, "harmful"),
        used_ids=used_ids,
        used_hashes=used_hashes,
    )
    harmless_leftover = eligible_examples(
        load_local_prompt_corpus(harmless_test_path, "harmless"),
        used_ids=used_ids,
        used_hashes=used_hashes,
    )
    _require_leftovers(harmful_leftover, label="harmful", required=required)
    _require_leftovers(harmless_leftover, label="harmless", required=required)
    harmful_selection, harmful_confirmation = sample_dose_halves(
        harmful_leftover,
        seed=seed,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    harmless_selection, harmless_confirmation = sample_dose_halves(
        harmless_leftover,
        seed=seed + 1,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    splits = DoseSplits(
        harmful_dose_selection=tuple(harmful_selection),
        harmless_dose_selection=tuple(harmless_selection),
        harmful_dose_confirmation=tuple(harmful_confirmation),
        harmless_dose_confirmation=tuple(harmless_confirmation),
    )
    _assert_disjoint_splits(
        splits,
        used_ids=used_ids,
        used_hashes=used_hashes,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    metadata = compact_split_metadata(
        harmful_selection=splits.harmful_dose_selection,
        harmless_selection=splits.harmless_dose_selection,
        harmful_confirmation=splits.harmful_dose_confirmation,
        harmless_confirmation=splits.harmless_dose_confirmation,
        seed=seed,
        harmless_seed=seed + 1,
        source_revision=original_source_revision(original_results_dir),
    )
    return splits, metadata


def _metric_point(
    point: Mapping[str, float],
    fields: Sequence[str],
) -> dict[str, float]:
    return {field: float(point[field]) for field in fields if field in point}


def _smallest_qualifying_coefficient(
    curve: Sequence[Mapping[str, float]],
    *,
    predicate: Callable[[Mapping[str, float]], bool],
) -> float | None:
    qualifying = [
        float(point["coefficient"])
        for point in sorted(curve, key=lambda item: float(item["coefficient"]))
        if predicate(point)
    ]
    if not qualifying:
        return None
    return qualifying[0]


def select_dose_coefficients(
    *,
    ablation_curve: Sequence[Mapping[str, float]],
    addition_curve: Sequence[Mapping[str, float]],
    min_harmful_ablation_drop: float = DEFAULT_MIN_HARMFUL_ABLATION_DROP,
    max_harmless_ablation_kl: float = DEFAULT_MAX_HARMLESS_ABLATION_KL,
    min_harmless_addition_increase: float = DEFAULT_MIN_HARMLESS_ADDITION_INCREASE,
) -> dict[str, Any]:
    ablation_coefficient = _smallest_qualifying_coefficient(
        ablation_curve,
        predicate=lambda point: (
            float(point["harmful_refusal_drop"]) >= min_harmful_ablation_drop
            and float(point["harmless_kl"]) <= max_harmless_ablation_kl
        ),
    )
    addition_coefficient = _smallest_qualifying_coefficient(
        addition_curve,
        predicate=lambda point: (
            float(point["harmless_refusal_increase"])
            >= min_harmless_addition_increase
        ),
    )
    selected_pair = None
    if ablation_coefficient is not None and addition_coefficient is not None:
        selected_pair = {
            "ablation": ablation_coefficient,
            "addition": addition_coefficient,
        }
    return {
        "ablation_coefficient": ablation_coefficient,
        "addition_coefficient": addition_coefficient,
        "selected_pair": selected_pair,
        "ablation_curve": [
            _metric_point(point, ABLATION_METRIC_FIELDS)
            for point in ablation_curve
        ],
        "addition_curve": [
            _metric_point(point, ADDITION_METRIC_FIELDS)
            for point in addition_curve
        ],
    }


def build_dose_evaluation(
    *,
    status: str,
    operational_success: bool,
    run_id: str,
    implementation_commit: str | None,
    source_pilot_run_id: str,
    source_pilot_implementation_commit: str,
    source_artifact_hashes: Mapping[str, str],
    model_revision: str,
    causal_gate: GateResult | None = None,
    selected_pair: Mapping[str, float] | None = None,
    stage: str | None = None,
    exception_category: str | None = None,
    error_code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    allowed = {
        STATUS_PASS,
        STATUS_CAUSAL_GATE_FAILED,
        STATUS_OPERATIONAL_ERROR,
        STATUS_NO_DOSE_QUALIFIED,
    }
    if status not in allowed:
        raise ValueError(f"unknown dose status: {status}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "operational_success": operational_success,
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "model_revision": model_revision,
        "source_pilot_run_id": source_pilot_run_id,
        "source_pilot_implementation_commit": source_pilot_implementation_commit,
        "source_artifact_hashes": dict(source_artifact_hashes),
        "causal_gate": causal_gate.to_dict() if causal_gate is not None else None,
        "selected_pair": dict(selected_pair) if selected_pair is not None else None,
    }
    if stage is not None:
        payload["stage"] = stage
    if exception_category is not None:
        payload["exception_category"] = exception_category
    if error_code is not None:
        payload["error_code"] = error_code
    if message is not None:
        payload["message"] = message
    return payload
