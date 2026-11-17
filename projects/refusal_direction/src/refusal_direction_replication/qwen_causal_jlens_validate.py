"""Offline validator for causal J-lens fidelity compact results."""

from __future__ import annotations

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
    LENS_VERSIONS,
    PRIMARY_LAYER,
    PRODUCTION_DATASET_SIZE,
    READOUT_LAYERS,
    READOUT_STEPS,
    aggregate_fidelity,
    bootstrap_ci_indices,
    cluster_bootstrap_difference,
    compact_contains_text_leak,
    observation_key,
    summarize_behavior_changes,
)
from .qwen_causal_jlens_runtime import (
    CHECKPOINT_FILENAME,
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
    OBSERVATION_KEYS,
    OBSERVATIONS_FILENAME,
    PanelCheckpoint,
    RESPONSES_FILENAME,
    STAGE_FILES,
    _compact_record,
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


def _check_root_profile_is_production(config: Mapping[str, Any]) -> None:
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


def _check_no_raw_text(named_payloads: Sequence[tuple[str, Any]]) -> None:
    for name, payload in named_payloads:
        if compact_contains_text_leak(payload):
            raise ValueError(f"raw text found in {name}")


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


def _check_early_termination(responses: Sequence[Mapping[str, Any]]) -> None:
    for row in responses:
        try:
            generated = int(row["generated_token_count"])
            readout = int(row["readout_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("termination fields missing from a response row") from exc
        terminal = row.get("terminal_reason")
        if not isinstance(terminal, str) or not terminal.strip():
            raise ValueError("early termination marker missing")
        actual = min(READOUT_STEPS, generated)
        if readout != actual:
            raise ValueError(
                "termination readout_steps do not equal the actual generated denominator"
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


def _check_shared_identity(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    prepare: Mapping[str, Any],
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
        if name == EVALUATION_FILENAME:
            continue
        provenance = payload.get("provenance") or {}
        _require(
            provenance.get("run_id") == run_id,
            f"run_id identity drifted in {name}",
        )
        _require(
            provenance.get("implementation_commit") == commit,
            f"implementation commit identity drifted in {name}",
        )
        _require(
            provenance.get("config_sha256") == identity["config_hash"],
            f"config hash identity drifted in {name}",
        )
        _require(
            provenance.get("source_artifact_hashes") == current_hashes,
            f"source artifact hash identity drifted in {name}",
        )
        if name != MANIFEST_FILENAME:
            _require(
                payload.get("config") == dict(config),
                f"config in {name} differs from the causal profile",
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
            min(READOUT_STEPS, int(row["generated_token_count"]))
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


def validate_qwen_causal_jlens(
    root: Path, *, allow_fixture_contract: bool = False
) -> dict[str, bool]:
    """Validate causal J-lens results under ``root``.

    Strict production is the default: the run root profile must equal the
    packaged causal profile, the frozen 32+32 cohort must be intact, and the
    pair matrix must contain exactly 256 prompt-condition pairs.
    ``allow_fixture_contract=True`` is only for CPU fixtures that use a reduced
    dataset; it does not relax uniqueness, recomputation, identity or raw-text
    rules.
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
    if not allow_fixture_contract:
        _check_root_profile_is_production(config)
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
        status=status,
        events=events,
        checkpoint_path=checkpoint_path,
    )
    checks["shared_identity"] = True

    responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
    observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
    _check_early_termination(responses)
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
    )
    checks["status_events_manifest"] = True
    return checks
