"""Offline validator for causal J-lens fidelity compact results."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import load_config
from .qwen_causal_jlens import (
    BOOTSTRAP_CI_LOWER_INDEX,
    BOOTSTRAP_CI_UPPER_INDEX,
    BOOTSTRAP_DEFAULT_RESAMPLES,
    CAUSAL_JLENS_PROFILE,
    CONDITIONS,
    DATASETS,
    FROZEN_EOS_TOKEN_ID,
    LENS_VERSIONS,
    PRIMARY_LAYER,
    PRODUCTION_DATASET_SIZE,
    READOUT_LAYERS,
    READOUT_STEPS,
    TEXT_LEAK_KEYS,
    aggregate_fidelity,
    bootstrap_ci_indices,
    cluster_bootstrap_difference,
    cohort_digest,
    compact_contains_text_leak,
    observation_key,
    summarize_behavior_changes,
)
from .qwen_causal_jlens_runtime import (
    CAUSAL_STAGES,
    CHECKPOINT_FILENAME,
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
    OBSERVATION_KEYS,
    OBSERVATIONS_FILENAME,
    PanelCheckpoint,
    RESPONSES_FILENAME,
    STAGE_FILES,
    _compact_record,
    _pair_key,
    checkpoint_identity,
    expected_matrix,
    require_complete_pair_artifacts,
    require_exact_pair_matrix,
    source_context,
)
from .run_progress import EVENTS_FILENAME, STATUS_FILENAME

RESULT_FILES = (
    STAGE_FILES["prepare"],
    STAGE_FILES["pilot"],
    STAGE_FILES["panel"],
    STAGE_FILES["analysis"],
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
)
PRODUCTION_PAIR_COUNT = len(DATASETS) * PRODUCTION_DATASET_SIZE * len(CONDITIONS)
FLOAT_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-12
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMPLETED_STAGES = [*CAUSAL_STAGES]
RAW_TEXT_ALIASES = TEXT_LEAK_KEYS | frozenset(
    {
        "samples",
        "outputs",
        "messages",
        "completions",
        "prompts",
        "responses",
        "text",
        "code",
        "raw",
    }
)
FIXTURE_ALLOWED_PROFILE_FIELDS = frozenset(
    {
        "dataset_size",
        "expected_prepare_sha256",
        "expected_cohort_digest",
        "expected_v1_evaluation_sha256",
        "expected_v2_evaluation_sha256",
        "expected_v1_run_id",
        "expected_v2_run_id",
        "expected_source_dose_run_id",
    }
)
ENVELOPE_KEYS = frozenset(
    {"experiment", "config", "results", "provenance", "schema_version"}
)
PROVENANCE_KEYS = frozenset(
    {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "model_name",
        "requested_model_revision",
        "resolved_model_revision",
        "purplellama_revision",
        "jlens_upstream_commit",
        "source_dose_run_id",
        "source_artifact_hashes",
        "direction_layer",
        "direction_offset",
        "eos_token_id",
        "cohort_digest",
    }
)
PREPARE_RESULT_KEYS = frozenset({*DATASETS, "source_files"})
DATASET_BLOCK_KEYS = frozenset(
    {"dataset", "example_ids", "content_hashes", "source_indices", "selection_size"}
)
SOURCE_FILE_REQUIRED = frozenset({"repository", "revision", "relative_path"})
SOURCE_FILE_OPTIONAL = frozenset({"sha256"})
PILOT_RESULT_KEYS = frozenset(
    {
        "response_count",
        "observation_count",
        "pilot_status",
        "conditions",
        "pilot_ids",
        "terminal_reasons",
    }
)
PANEL_RESULT_KEYS = frozenset(
    {
        "response_count",
        "observation_count",
        "conditions",
        "matrix_size",
        "terminal_reasons",
    }
)
ANALYSIS_RESULT_KEYS = frozenset(
    {
        "fidelity",
        "bootstrap",
        "behavior_changes",
        "refusal_rates",
        "observation_count",
        "response_count",
    }
)
MANIFEST_RESULT_KEYS = frozenset({"status", "completed_stages", "result_files"})
EVALUATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "pilot_status",
        "response_count",
        "observation_count",
        "prepare_ids",
        "results",
        "run_id",
        "implementation_commit",
        "model_revision",
        "source_artifact_hashes",
        "config",
        "provenance",
    }
)
EVALUATION_RESULT_KEYS = frozenset(
    {"primary", "refusal_rates", "behavior_change_analysis", "fidelity"}
)


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{path.name} is missing")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            raise ValueError(f"{path.name} line {number} is not JSON") from None
        _require(isinstance(item, dict), f"{path.name} line {number} is not an object")
        rows.append(item)
    return rows


def _results_of(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    results = payload.get("results")
    _require(isinstance(results, dict), f"{name} has no results object")
    return results


def _packaged_profile() -> dict[str, Any]:
    package_path = (
        Path(__file__).resolve().parents[2] / "configs" / CAUSAL_JLENS_PROFILE
    )
    _require(package_path.is_file(), "packaged causal J-lens profile is missing")
    return dict(json.loads(package_path.read_text(encoding="utf-8")))


def _confined_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path string")
    parts = [part for part in re.split(r"[\\/]+", value) if part not in ("", ".")]
    absolute = (
        value.startswith(("/", "\\"))
        or bool(re.match(r"^[A-Za-z]:", value))
        or Path(value).is_absolute()
    )
    if absolute:
        raise ValueError(f"{label} must be relative to the run root: absolute path rejected")
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


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _close(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _close(a, b) for a, b in zip(left, right)
        )
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=FLOAT_TOLERANCE, abs_tol=ABS_TOLERANCE
        )
    return left == right


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a strict integer count")
    return value


def _check_exact_keys(obj: Any, expected: set[str], label: str) -> None:
    _require(isinstance(obj, Mapping), f"{label} is not an object")
    actual = set(obj)
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise ValueError(f"unexpected key in {label}: {sorted(extra)[0]}")
    if missing:
        raise ValueError(f"missing key in {label}: {sorted(missing)[0]}")


def _check_profile_against_packaged(
    config: Mapping[str, Any], *, allow_fixture_contract: bool
) -> None:
    package = _packaged_profile()
    for key in sorted(set(package) | set(config)):
        fixture_allowed = (
            allow_fixture_contract and key in FIXTURE_ALLOWED_PROFILE_FIELDS
        )
        if key not in config:
            if fixture_allowed:
                continue
            raise ValueError(f"production contract: root profile is missing {key}")
        if key not in package:
            if fixture_allowed:
                continue
            raise ValueError(
                f"production contract: root profile has unexpected key {key}"
            )
        if config[key] == package[key] or fixture_allowed:
            continue
        if allow_fixture_contract:
            raise ValueError(
                f"fixture contract: {key} differs from the packaged causal profile "
                "and is not in the fixture allowlist"
            )
        raise ValueError(
            f"production contract: root profile {key} {config[key]!r} "
            f"differs from the packaged causal profile {package[key]!r}"
        )


def _check_scientific_constants(config: Mapping[str, Any]) -> None:
    _require(
        list(config.get("datasets") or []) == list(DATASETS),
        "profile datasets must remain mitre and mitre_frr",
    )
    _require(
        list(config.get("conditions") or []) == list(CONDITIONS),
        "profile conditions must remain the four frozen conditions",
    )
    _require(
        list(config.get("readout_layers") or []) == list(READOUT_LAYERS),
        "profile readout_layers must remain 19, 22, 25 and 26",
    )
    _require(
        int(config.get("readout_steps", -1)) == READOUT_STEPS,
        f"profile readout_steps must remain {READOUT_STEPS}",
    )
    _require(
        int(config.get("max_new_tokens", -1)) == 64,
        "profile max_new_tokens must remain 64",
    )
    _require(
        int(config.get("eos_token_id", -1)) == FROZEN_EOS_TOKEN_ID,
        "profile eos_token_id must remain the frozen Qwen chat EOS id",
    )
    _require(
        int(config.get("bootstrap_resamples", -1)) == BOOTSTRAP_DEFAULT_RESAMPLES,
        "bootstrap resamples must remain 10000",
    )
    _require(
        int(config.get("bootstrap_seed", -1)) == 2028,
        "bootstrap seed must remain 2028",
    )
    _require(
        float(config.get("equivalence_margin", -1)) == 0.05,
        "equivalence margin must remain 0.05",
    )
    _require(
        int(config.get("primary_layer", -1)) == PRIMARY_LAYER,
        "primary layer must remain 25",
    )
    _require(int(config.get("primary_k", -1)) == 5, "primary k must remain 5")


def _contains_raw_text(obj: Any) -> bool:
    if compact_contains_text_leak(obj):
        return True
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).casefold() in RAW_TEXT_ALIASES:
                return True
            if _contains_raw_text(value):
                return True
        return False
    if isinstance(obj, (list, tuple)):
        return any(_contains_raw_text(item) for item in obj)
    return False


def _check_no_raw_text(named_payloads: Sequence[tuple[str, Any]]) -> None:
    for name, payload in named_payloads:
        if _contains_raw_text(payload):
            raise ValueError(f"raw text found in {name}")


def _check_result_schemas(
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in (
        STAGE_FILES["prepare"],
        STAGE_FILES["pilot"],
        STAGE_FILES["panel"],
        STAGE_FILES["analysis"],
        MANIFEST_FILENAME,
    ):
        _check_exact_keys(payloads[name], set(ENVELOPE_KEYS), name)
        _check_exact_keys(payloads[name]["provenance"], set(PROVENANCE_KEYS), f"{name} provenance")
    _check_exact_keys(
        _results_of(payloads[STAGE_FILES["prepare"]], STAGE_FILES["prepare"]),
        set(PREPARE_RESULT_KEYS),
        "00_prepare.json results",
    )
    prepare = _results_of(payloads[STAGE_FILES["prepare"]], STAGE_FILES["prepare"])
    for dataset in DATASETS:
        _check_exact_keys(
            prepare[dataset], set(DATASET_BLOCK_KEYS), f"00_prepare.json {dataset}"
        )
        identity = (prepare.get("source_files") or {}).get(dataset) or {}
        _require(isinstance(identity, Mapping), f"{dataset} source identity missing")
        extra = set(identity) - SOURCE_FILE_REQUIRED - SOURCE_FILE_OPTIONAL
        missing = SOURCE_FILE_REQUIRED - set(identity)
        if extra:
            raise ValueError(f"unexpected key in {dataset} source identity: {sorted(extra)[0]}")
        if missing:
            raise ValueError(f"missing key in {dataset} source identity: {sorted(missing)[0]}")
    _check_exact_keys(
        _results_of(payloads[STAGE_FILES["pilot"]], STAGE_FILES["pilot"]),
        set(PILOT_RESULT_KEYS),
        "01_pilot.json results",
    )
    _check_exact_keys(
        _results_of(payloads[STAGE_FILES["panel"]], STAGE_FILES["panel"]),
        set(PANEL_RESULT_KEYS),
        "02_panel.json results",
    )
    _check_exact_keys(
        _results_of(payloads[STAGE_FILES["analysis"]], STAGE_FILES["analysis"]),
        set(ANALYSIS_RESULT_KEYS),
        "03_analysis.json results",
    )
    _check_exact_keys(
        _results_of(payloads[MANIFEST_FILENAME], MANIFEST_FILENAME),
        set(MANIFEST_RESULT_KEYS),
        "execution_manifest.json results",
    )
    _check_exact_keys(payloads[EVALUATION_FILENAME], set(EVALUATION_KEYS), EVALUATION_FILENAME)
    _check_exact_keys(
        payloads[EVALUATION_FILENAME]["provenance"],
        set(PROVENANCE_KEYS),
        f"{EVALUATION_FILENAME} provenance",
    )
    _check_exact_keys(
        (payloads[EVALUATION_FILENAME].get("results") or {}),
        set(EVALUATION_RESULT_KEYS),
        f"{EVALUATION_FILENAME} results",
    )


def _refusal_rates(responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for condition in CONDITIONS:
            items = [
                record
                for record in responses
                if str(record["dataset"]) == dataset
                and str(record["condition"]) == condition
            ]
            refusals = [bool(record.get("refusal")) for record in items]
            output[dataset][condition] = {
                "prompt_count": len(items),
                "refusal_count": sum(refusals),
                "refusal_rate": (
                    sum(refusals) / len(refusals) if refusals else None
                ),
            }
    return output


def _primary_from_bootstrap(
    bootstrap: Mapping[str, Any], *, margin: float
) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    for key, payload in bootstrap.items():
        dataset, condition, lens_version = str(key).split("/")
        primary.setdefault(lens_version, {}).setdefault(dataset, {})[condition] = {
            "estimate": payload["estimate"],
            "ci_lower": payload["ci_lower"],
            "ci_upper": payload["ci_upper"],
            "prompt_count": payload["prompt_count"],
            "equivalent": (
                float(payload["ci_lower"]) >= -margin
                and float(payload["ci_upper"]) <= margin
            ),
        }
    return primary


def _check_generation_integrity(
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    max_new = _strict_int(config["max_new_tokens"], "max_new_tokens")
    eos_id = _strict_int(config["eos_token_id"], "eos_token_id")
    readout_limit = _strict_int(config["readout_steps"], "readout_steps")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[_pair_key(row)].append(row)
    for row in responses:
        token_ids = row.get("generated_token_ids")
        _require(
            isinstance(token_ids, list),
            "generated_token_ids must be a list of token ids",
        )
        for token in token_ids:
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError(
                    "generated_token_ids must contain strict integer token ids"
                )
        count = _strict_int(row.get("generated_token_count"), "generated_token_count")
        _require(count >= 1, "empty generation is not allowed")
        _require(
            len(token_ids) == count,
            "generated_token_ids length does not match generated_token_count",
        )
        readout = _strict_int(row.get("readout_steps"), "readout_steps")
        expected_readout = min(readout_limit, count)
        terminal = row.get("terminal_reason")
        if not isinstance(terminal, str) or not terminal.strip():
            raise ValueError("early termination marker missing")
        if terminal == "max_tokens":
            if count != max_new:
                raise ValueError(
                    "invalid terminal: max_tokens count is below max_new_tokens"
                )
            if token_ids[-1] == eos_id:
                raise ValueError(
                    "invalid terminal: max_tokens ends with the frozen EOS id"
                )
        elif terminal == "eos":
            if not 1 <= count <= max_new:
                raise ValueError(
                    "invalid terminal: eos count is outside 1..max_new_tokens"
                )
            if token_ids[-1] != eos_id:
                raise ValueError(
                    "invalid terminal: eos final token is not the frozen EOS ID"
                )
        else:
            raise ValueError("invalid terminal: unknown reason")
        if readout != expected_readout:
            raise ValueError(
                "termination readout_steps do not equal min(count, readout_steps)"
            )
        steps = {
            _strict_int(item.get("generation_step"), "generation_step")
            for item in grouped[_pair_key(row)]
        }
        expected_steps = set(range(1, expected_readout + 1))
        if steps != expected_steps:
            raise ValueError(
                "observation steps are not exactly 1..min(count, readout_steps)"
            )


def _check_unique_observation_keys(observations: Sequence[Mapping[str, Any]]) -> None:
    try:
        keys = [observation_key(row) for row in observations]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observation keys are malformed") from exc
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate observation keys")
    for row in observations:
        missing = [key for key in OBSERVATION_KEYS if key not in row]
        if missing:
            raise ValueError("observation missing keys: " + ", ".join(missing))


def _recompute_bootstrap(
    observations: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resamples = int(config["bootstrap_resamples"])
    seed = int(config["bootstrap_seed"])
    lower_index, upper_index = bootstrap_ci_indices(resamples)
    _require(
        (lower_index, upper_index)
        == (BOOTSTRAP_CI_LOWER_INDEX, BOOTSTRAP_CI_UPPER_INDEX)
        == (249, 9749),
        "bootstrap CI indices are not the frozen 249/9749 pair",
    )
    bootstrap: dict[str, Any] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            if condition == "baseline":
                continue
            for lens_version in LENS_VERSIONS:
                key = f"{dataset}/{condition}/{lens_version}"
                bootstrap[key] = cluster_bootstrap_difference(
                    observations,
                    condition=condition,
                    k=int(config["primary_k"]),
                    layer=int(config["primary_layer"]),
                    lens_version=lens_version,
                    dataset=dataset,
                    resamples=resamples,
                    seed=seed,
                )
    return bootstrap


def _check_bootstrap_block(stored: Any, expected: Mapping[str, Any]) -> None:
    _require(isinstance(stored, dict), "bootstrap block is missing")
    _require(
        int(stored.get("seed", -1)) == int(expected["seed"]) == 2028,
        "bootstrap seed is not frozen 2028",
    )
    _require(
        int(stored.get("resamples", -1)) == int(expected["resamples"]) == 10_000,
        "bootstrap resamples are not frozen at 10000",
    )
    _require(
        int(stored.get("ci_lower_index", -1)) == BOOTSTRAP_CI_LOWER_INDEX
        and int(stored.get("ci_upper_index", -1)) == BOOTSTRAP_CI_UPPER_INDEX,
        "bootstrap CI indices are not the frozen 249/9749 pair",
    )
    _require(
        stored.get("cluster_unit") == "example_id",
        "bootstrap cluster unit must remain example_id",
    )
    _require(
        _close(stored, expected),
        "bootstrap interval does not match prompt-clustered recomputation",
    )


def _check_file_identity(
    *,
    name: str,
    payload: Mapping[str, Any],
    run_id: str,
    commit: str,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    current_hashes: Mapping[str, str],
    identity: Mapping[str, Any],
    source_prepare: Mapping[str, Any],
) -> None:
    provenance = payload.get("provenance") or {}
    _require(provenance.get("run_id") == run_id, f"run_id identity drifted in {name}")
    _require(
        provenance.get("implementation_commit") == commit,
        f"implementation commit identity drifted in {name}",
    )
    _require(
        provenance.get("config_sha256") == identity["config_hash"],
        f"config hash identity drifted in {name}",
    )
    _require(
        provenance.get("source_artifact_hashes") == dict(current_hashes),
        f"source artifact hash identity drifted in {name}",
    )
    _require(
        payload.get("config") == dict(config),
        f"config in {name} differs from the causal profile",
    )
    _require(
        provenance.get("model_name") == config["model_name"],
        f"model name identity drifted in {name}",
    )
    _require(
        provenance.get("requested_model_revision") == config["model_revision"]
        and provenance.get("resolved_model_revision") == config["model_revision"],
        f"model revision identity drifted in {name}",
    )
    _require(
        provenance.get("jlens_upstream_commit") == config["jlens_upstream_commit"],
        f"jlens upstream identity drifted in {name}",
    )
    _require(
        provenance.get("source_dose_run_id") == source.get("source_dose_run_id"),
        f"source dose identity drifted in {name}",
    )
    _require(
        int(provenance.get("eos_token_id", -1)) == int(config["eos_token_id"]),
        f"eos_token_id identity drifted in {name}",
    )
    _require(
        provenance.get("cohort_digest") == cohort_digest(source_prepare),
        f"cohort digest identity drifted in {name}",
    )
    _require(
        int(provenance.get("direction_layer", -1)) == 18
        and int(provenance.get("direction_offset", 0)) == -1,
        f"direction identity drifted in {name}",
    )


def _check_shared_identity(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    prepare: Mapping[str, Any],
    source_prepare: Mapping[str, Any],
    status: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
) -> None:
    run_id = evaluation.get("run_id")
    commit = evaluation.get("implementation_commit")
    _require(isinstance(run_id, str) and bool(run_id.strip()), "evaluation run_id is missing")
    _require(
        isinstance(commit, str) and bool(COMMIT_RE.fullmatch(commit)),
        "evaluation implementation_commit is not a full commit hash",
    )
    stored_hashes = evaluation.get("source_artifact_hashes")
    current_hashes = source["source_artifact_hashes"]
    _require(
        stored_hashes == current_hashes,
        "evaluation source artifact hash identity drifted from frozen sources",
    )
    identity = checkpoint_identity(
        run_id=str(run_id),
        implementation_commit=str(commit),
        config=config,
        source=source,
        prepare=prepare,
    )
    for name, payload in payloads.items():
        _check_file_identity(
            name=name,
            payload=payload,
            run_id=str(run_id),
            commit=str(commit),
            config=config,
            source=source,
            current_hashes=current_hashes,
            identity=identity,
            source_prepare=source_prepare,
        )
    _require(
        evaluation.get("model_revision") == config["model_revision"],
        "evaluation model revision identity drifted",
    )
    expected_ids = {
        dataset: list(prepare[dataset]["example_ids"]) for dataset in DATASETS
    }
    _require(
        evaluation.get("prepare_ids") == expected_ids,
        "evaluation prepare_ids drifted from 00_prepare.json",
    )
    _require(status.get("run_id") == run_id, "run_id mismatch in status.json")
    _require(
        all(item.get("run_id") == run_id for item in events),
        "run_id mismatch in events.jsonl",
    )
    PanelCheckpoint.load(checkpoint_path, expected_identity=identity)


def _check_ids_and_matrix(
    *,
    prepare: Mapping[str, Any],
    source_prepare: Mapping[str, Any],
    config: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    allow_fixture_contract: bool,
) -> list[tuple[str, str, str]]:
    dataset_size = int(config["dataset_size"])
    if not allow_fixture_contract:
        _require(
            dataset_size == PRODUCTION_DATASET_SIZE,
            "production contract: dataset_size must be 32",
        )
    matrix = expected_matrix(prepare)
    expected_count = len(DATASETS) * dataset_size * len(CONDITIONS)
    _require(
        len(matrix) == expected_count,
        f"pair matrix size {len(matrix)} is not {expected_count}",
    )
    if not allow_fixture_contract:
        _require(
            expected_count == PRODUCTION_PAIR_COUNT == 256,
            "production contract: pair matrix must contain exactly 256 pairs",
        )
    source_results = source_prepare.get("results") or {}
    for dataset in DATASETS:
        ids = list((prepare.get(dataset) or {}).get("example_ids") or [])
        hashes = list((prepare.get(dataset) or {}).get("content_hashes") or [])
        _require(
            len(ids) == len(hashes) == dataset_size,
            f"{dataset} must have exactly {dataset_size} IDs",
        )
        _require(len(set(ids)) == len(ids), f"{dataset} duplicate example IDs")
        source_ids = list((source_results.get(dataset) or {}).get("example_ids") or [])
        source_hashes = list(
            (source_results.get(dataset) or {}).get("content_hashes") or []
        )
        _require(
            ids == source_ids and hashes == source_hashes,
            f"{dataset} IDs drifted from the frozen source prepare",
        )
    require_exact_pair_matrix(responses, matrix)
    require_complete_pair_artifacts(responses, observations, matrix)
    expected_observations = 0
    for row in responses:
        expected_observations += (
            min(int(config["readout_steps"]), int(row["generated_token_count"]))
            * len(READOUT_LAYERS)
            * len(LENS_VERSIONS)
        )
    _require(
        len(observations) == expected_observations,
        "observation count disagrees with the actual generated-step denominators",
    )
    return matrix


def _check_status_events_manifest(
    *,
    status: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    panel: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        status.get("profile") == CAUSAL_JLENS_PROFILE,
        "status.json profile is not the causal J-lens profile",
    )
    _require(
        status.get("state") == "completed" and status.get("current_stage") == "finalize",
        "status must be completed at the finalize stage",
    )
    _require(bool(events), f"{EVENTS_FILENAME} is empty")
    last = events[-1]
    _require(
        last.get("event") == "stage_completed"
        and last.get("stage") == "finalize"
        and last.get("state") == "completed",
        "events must end with the completed finalize stage",
    )
    results = _results_of(manifest, MANIFEST_FILENAME)
    _require(results.get("status") == "pass", "manifest status differs from the evaluation")
    _require(
        list(results.get("result_files") or []) == list(RESULT_FILES),
        "manifest result_files differ from the documented set",
    )
    _require(
        list(results.get("completed_stages") or []) == COMPLETED_STAGES,
        "manifest completed_stages differ from the documented stage sequence",
    )
    _require(
        evaluation.get("pilot_status") == "pass",
        "evaluation pilot_status must be pass",
    )
    _require(
        evaluation.get("response_count") == len(responses)
        and evaluation.get("observation_count") == len(observations)
        and panel.get("response_count") == len(responses)
        and panel.get("observation_count") == len(observations),
        "evaluation counts drifted from actual response and observation denominators",
    )


def validate_qwen_causal_jlens(
    root: Path, *, allow_fixture_contract: bool = False
) -> dict[str, bool]:
    """Validate causal J-lens results under ``root``.

    Strict production is the default: the run root profile must equal the
    packaged causal profile, the frozen 32+32 cohort must be intact, and the
    pair matrix must contain exactly 256 prompt-condition pairs.
    ``allow_fixture_contract=True`` may change only
    ``FIXTURE_ALLOWED_PROFILE_FIELDS``; every other packaged field, provenance
    contract, uniqueness, recomputation and raw-text rule stays strict.
    """
    try:
        return _validate(root, allow_fixture_contract=allow_fixture_contract)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(
            "malformed causal J-lens profile or artifact: "
            f"{type(exc).__name__}: {str(exc)[:120]}"
        ) from None


def _validate(root: Path, *, allow_fixture_contract: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    config = load_config(root, profile=CAUSAL_JLENS_PROFILE)
    results_dir = _confined_path(root, config.get("results_dir"), "profile results_dir")
    artifacts_dir = _confined_path(
        root, config.get("artifacts_dir"), "profile artifacts_dir"
    )
    _check_profile_against_packaged(
        config, allow_fixture_contract=allow_fixture_contract
    )
    _check_scientific_constants(config)
    checks["production_contract"] = True

    source = source_context(root, config)
    checks["frozen_cohort_and_lens_provenance"] = True

    missing = [name for name in RESULT_FILES if not (results_dir / name).is_file()]
    _require(not missing, f"required result artifacts are missing: {missing}")
    payloads = {name: _read_json(results_dir / name) for name in RESULT_FILES}
    evaluation = payloads[EVALUATION_FILENAME]
    _require(evaluation.get("status") == "pass", "evaluation status must be pass")
    _check_no_raw_text(list(payloads.items()))
    checks["no_raw_text"] = True
    _check_result_schemas(payloads)

    prepare = _results_of(payloads[STAGE_FILES["prepare"]], STAGE_FILES["prepare"])
    analysis = _results_of(payloads[STAGE_FILES["analysis"]], STAGE_FILES["analysis"])
    panel = _results_of(payloads[STAGE_FILES["panel"]], STAGE_FILES["panel"])
    source_prepare = _read_json(_confined_path(root, config["source_prepare"], "source_prepare"))
    status = _read_json(artifacts_dir / STATUS_FILENAME)
    events = _read_jsonl(artifacts_dir / EVENTS_FILENAME)
    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    _require(checkpoint_path.is_file(), f"{CHECKPOINT_FILENAME} is missing")
    _check_shared_identity(
        payloads=payloads,
        evaluation=evaluation,
        source=source,
        config=config,
        prepare=prepare,
        source_prepare=source_prepare,
        status=status,
        events=events,
        checkpoint_path=checkpoint_path,
    )
    checks["shared_identity"] = True

    responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
    observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
    _check_generation_integrity(responses, observations, config)
    checks["early_termination"] = True
    _check_unique_observation_keys(observations)
    checks["unique_observation_keys"] = True
    matrix = _check_ids_and_matrix(
        prepare=prepare,
        source_prepare=source_prepare,
        config=config,
        responses=responses,
        observations=observations,
        allow_fixture_contract=allow_fixture_contract,
    )
    checks["exact_ids_and_pair_matrix"] = True
    completed = [
        (str(item[0]), str(item[1]), str(item[2]))
        for item in (_read_json(checkpoint_path).get("completed") or [])
    ]
    _require(
        set(completed) == set(matrix) and len(completed) == len(matrix),
        "checkpoint completed pairs do not match the exact pair matrix",
    )

    compact_responses = [_compact_record(row) for row in responses]
    fidelity = aggregate_fidelity(observations)
    _require(
        _close(analysis.get("fidelity"), fidelity),
        "metric block does not match recomputation from compact observations",
    )
    _require(
        _close((evaluation.get("results") or {}).get("fidelity"), fidelity),
        "evaluation metric block does not match recomputation from compact observations",
    )
    checks["metrics_recomputed"] = True

    expected_bootstrap = _recompute_bootstrap(observations, config=config)
    stored_bootstrap = analysis.get("bootstrap")
    _require(isinstance(stored_bootstrap, dict), "analysis bootstrap is missing")
    _require(
        set(stored_bootstrap) == set(expected_bootstrap),
        "bootstrap keys drifted from the frozen dataset/condition/lens grid",
    )
    for key, expected in expected_bootstrap.items():
        _check_bootstrap_block(stored_bootstrap[key], expected)
    checks["bootstrap_recomputed"] = True

    margin = float(config["equivalence_margin"])
    expected_primary = _primary_from_bootstrap(expected_bootstrap, margin=margin)
    stored_primary = (evaluation.get("results") or {}).get("primary")
    _require(
        _close(stored_primary, expected_primary),
        "equivalence flags do not match recomputed 95% intervals",
    )
    checks["equivalence_flags"] = True

    expected_refusals = _refusal_rates(compact_responses)
    _require(
        _close(analysis.get("refusal_rates"), expected_refusals),
        "refusal rates do not match actual response denominators",
    )
    _require(
        _close((evaluation.get("results") or {}).get("refusal_rates"), expected_refusals),
        "evaluation refusal rates do not match actual response denominators",
    )
    expected_behavior = summarize_behavior_changes(compact_responses, observations)
    _require(
        _close(analysis.get("behavior_changes"), expected_behavior),
        "behavior-change summary does not match recomputation",
    )
    _require(
        _close(
            (evaluation.get("results") or {}).get("behavior_change_analysis"),
            expected_behavior,
        ),
        "evaluation behavior-change summary does not match recomputation",
    )
    checks["behavior_changes"] = True
    _require(
        int(analysis.get("observation_count", -1)) == len(observations)
        and int(analysis.get("response_count", -1)) == len(responses)
        and int(panel.get("observation_count", -1)) == len(observations)
        and int(panel.get("response_count", -1)) == len(responses)
        and int(panel.get("matrix_size", -1)) == len(matrix),
        "stored counts disagree with actual response and observation denominators",
    )

    _check_status_events_manifest(
        status=status,
        events=events,
        manifest=payloads[MANIFEST_FILENAME],
        evaluation=evaluation,
        panel=panel,
        responses=responses,
        observations=observations,
    )
    pilot = _results_of(payloads[STAGE_FILES["pilot"]], STAGE_FILES["pilot"])
    _require(
        evaluation.get("pilot_status") == pilot.get("pilot_status") == "pass",
        "evaluation pilot_status drifted from 01_pilot.json",
    )
    checks["status_events_manifest"] = True
    return checks
