"""Offline validator for compact Qwen dose follow-up artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .gates import (
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    evaluate_causal_gate,
)
from .pipeline import load_config
from .qwen_dose import (
    EXPECTED_SOURCE_LAYER,
    EXPECTED_SOURCE_OFFSET,
    STATUS_NO_DOSE_QUALIFIED,
    coefficient_checkpoint_name,
    collect_original_usage,
    select_dose_coefficients,
    source_artifact_hashes,
)
from .qwen_pilot_validate import (
    QWEN_REVISION,
    _all_finite,
    _checks_match,
    _commit_ok,
    _model_ok,
    _run_id_ok,
    _walk_contains_leak,
    validate_qwen_pilot,
)
from .run_progress import EVENTS_FILENAME, STATUS_FILENAME

DOSE_PROFILE = "qwen25_coder_7b_dose.json"
PASS_FILES = (
    "00_dose_prepare.json",
    "01_dose_baseline.json",
    "02_dose_selection.json",
    "03_dose_confirmation.json",
    "dose_followup_evaluation.json",
    "execution_manifest.json",
)
NO_DOSE_FILES = (
    "00_dose_prepare.json",
    "01_dose_baseline.json",
    "02_dose_selection.json",
    "dose_followup_evaluation.json",
    "execution_manifest.json",
)
CHECKPOINT_NAME_RE = re.compile(r"^checkpoint_coeff_[0-9]+(?:\.[0-9]+)?\.json$")
REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "implementation_commit",
    "model_name",
    "requested_model_revision",
    "resolved_model_revision",
    "source_pilot_run_id",
    "source_pilot_implementation_commit",
    "source_artifact_hashes",
)
REQUIRED_EVALUATION_FIELDS = (
    "run_id",
    "implementation_commit",
    "model_revision",
    "source_pilot_run_id",
    "source_pilot_implementation_commit",
    "source_artifact_hashes",
    "status",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_files(status: str) -> tuple[str, ...]:
    if status == STATUS_NO_DOSE_QUALIFIED:
        return NO_DOSE_FILES
    if status == STATUS_OPERATIONAL_ERROR:
        return ("dose_followup_evaluation.json",)
    return PASS_FILES


def _split_groups(prepare: dict[str, Any]) -> dict[str, list[str]]:
    splits = (prepare.get("results") or {}).get("splits") or {}
    return {
        name: [str(item) for item in (splits.get("example_ids") or {}).get(name) or []]
        for name in (
            "harmful_dose_selection",
            "harmless_dose_selection",
            "harmful_dose_confirmation",
            "harmless_dose_confirmation",
        )
    }


def _split_hashes(prepare: dict[str, Any]) -> dict[str, list[str]]:
    splits = (prepare.get("results") or {}).get("splits") or {}
    return {
        name: [str(item) for item in (splits.get("content_hashes") or {}).get(name) or []]
        for name in (
            "harmful_dose_selection",
            "harmless_dose_selection",
            "harmful_dose_confirmation",
            "harmless_dose_confirmation",
        )
    }


def _unique_and_disjoint(groups: dict[str, list[str]]) -> bool:
    seen: set[str] = set()
    for items in groups.values():
        as_set = set(items)
        if len(as_set) != len(items):
            return False
        if not as_set.isdisjoint(seen):
            return False
        seen.update(as_set)
    return True


def validate_qwen_dose(root: Path) -> dict[str, bool]:
    results_dir = root / "results" / "qwen25_coder_7b_dose"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b_dose"
    original_dir = root / "results" / "qwen25_coder_7b"
    evaluation_path = results_dir / "dose_followup_evaluation.json"
    status_path = artifacts_dir / STATUS_FILENAME
    checks = {
        "required_stages_present": False,
        "model_and_revision": True,
        "source_pilot_status": True,
        "source_layer_and_offset": True,
        "source_hashes_match": True,
        "followup_run_id_differs": True,
        "split_ids_unique_and_disjoint": True,
        "split_hashes_unique_and_disjoint": True,
        "disjoint_from_original": True,
        "split_counts": True,
        "complete_configured_grid": True,
        "metrics_are_finite": True,
        "selector_reproducible_from_curve": True,
        "confirmation_does_not_influence_selection": True,
        "confirmation_absent_when_unqualified": True,
        "gate_calculations_consistent": True,
        "no_raw_prompt_or_generation_leakage": True,
        "status_matches_result": True,
        "status_matches_gate": True,
        "manifest_agrees": True,
        "single_run_id": True,
        "code_provenance": True,
        "required_provenance": True,
        "events_present": True,
        "selected_pair_consistent": True,
        "random_control_uses_ablation_coefficient": True,
        "original_pilot_validator": True,
        "source_pilot_identity_matches": True,
        "observability_reconciles": True,
        "file_list_matches": True,
    }
    if not evaluation_path.is_file() or not status_path.is_file():
        return checks
    evaluation = _load(evaluation_path)
    status_payload = _load(status_path)
    status = evaluation.get("status")
    required = _required_files(str(status))
    present = [(results_dir / name).is_file() for name in required]
    checks["required_stages_present"] = all(present) and status_path.is_file()
    if status == STATUS_NO_DOSE_QUALIFIED:
        checks["confirmation_absent_when_unqualified"] = not (
            results_dir / "03_dose_confirmation.json"
        ).exists()
    else:
        checks["confirmation_absent_when_unqualified"] = True

    compact_files = [
        path for path in sorted(results_dir.glob("*.json")) if path.is_file()
    ]
    payloads = [_load(path) for path in compact_files]
    for payload in payloads:
        if payload.get("config") or payload.get("provenance"):
            if not _model_ok(payload):
                checks["model_and_revision"] = False
        if evaluation.get("model_revision") not in {None, QWEN_REVISION}:
            checks["model_and_revision"] = False
        if not _all_finite(payload):
            checks["metrics_are_finite"] = False
        if _walk_contains_leak(payload):
            checks["no_raw_prompt_or_generation_leakage"] = False

    try:
        config = load_config(root, profile=DOSE_PROFILE)
    except Exception:
        checks["complete_configured_grid"] = False
        return checks

    prepare_path = results_dir / "00_dose_prepare.json"
    prepare = _load(prepare_path) if prepare_path.is_file() else {}
    selected = (prepare.get("results") or {}).get("selected") or {}
    if selected.get("layer") != EXPECTED_SOURCE_LAYER or selected.get(
        "position_offset"
    ) != EXPECTED_SOURCE_OFFSET:
        checks["source_layer_and_offset"] = False
    original_eval_path = original_dir / "pilot_evaluation.json"
    if original_eval_path.is_file():
        original_eval = _load(original_eval_path)
        if original_eval.get("status") != STATUS_CAUSAL_GATE_FAILED:
            checks["source_pilot_status"] = False
        source_run = original_eval.get("run_id")
        if source_run and evaluation.get("source_pilot_run_id") not in {None, source_run}:
            checks["source_pilot_status"] = False
        source_commit = original_eval.get("implementation_commit")
        if source_commit and evaluation.get("source_pilot_implementation_commit") not in {
            None,
            source_commit,
        }:
            checks["source_pilot_status"] = False
        for payload in payloads:
            provenance = payload.get("provenance") or {}
            stored_run = provenance.get("source_pilot_run_id") or payload.get(
                "source_pilot_run_id"
            )
            stored_commit = provenance.get(
                "source_pilot_implementation_commit"
            ) or payload.get("source_pilot_implementation_commit")
            if stored_run != source_run or stored_commit != source_commit:
                checks["source_pilot_identity_matches"] = False
    else:
        checks["source_pilot_status"] = False
        checks["source_pilot_identity_matches"] = False

    try:
        current_hashes = source_artifact_hashes(root, config)
    except Exception:
        checks["source_hashes_match"] = False
        current_hashes = {}
    for payload in payloads:
        stored = (payload.get("provenance") or {}).get("source_artifact_hashes")
        if stored is None and payload is evaluation:
            stored = payload.get("source_artifact_hashes")
        if stored is not None and stored != current_hashes:
            checks["source_hashes_match"] = False
    stored_eval_hashes = evaluation.get("source_artifact_hashes")
    if stored_eval_hashes is not None and stored_eval_hashes != current_hashes:
        checks["source_hashes_match"] = False

    source_run = evaluation.get("source_pilot_run_id")
    followup_run = evaluation.get("run_id")
    if not _run_id_ok(followup_run) or followup_run == source_run:
        checks["followup_run_id_differs"] = False

    id_groups = _split_groups(prepare)
    hash_groups = _split_hashes(prepare)
    checks["split_ids_unique_and_disjoint"] = _unique_and_disjoint(id_groups)
    checks["split_hashes_unique_and_disjoint"] = _unique_and_disjoint(hash_groups)
    counts = ((prepare.get("results") or {}).get("splits") or {}).get("counts") or {}
    expected = int(config.get("dose_selection_per_class", 32))
    expected_conf = int(config.get("dose_confirmation_per_class", 32))
    expected_counts = {
        "harmful_dose_selection": expected,
        "harmless_dose_selection": expected,
        "harmful_dose_confirmation": expected_conf,
        "harmless_dose_confirmation": expected_conf,
    }
    checks["split_counts"] = counts == expected_counts and all(
        len(id_groups[name]) == expected_counts[name] for name in expected_counts
    )
    try:
        used_ids, used_hashes = collect_original_usage(original_dir)
        new_ids = {item for group in id_groups.values() for item in group}
        new_hashes = {item for group in hash_groups.values() for item in group}
        checks["disjoint_from_original"] = used_ids.isdisjoint(new_ids) and used_hashes.isdisjoint(
            new_hashes
        )
    except Exception:
        checks["disjoint_from_original"] = False

    grid = [float(item) for item in config.get("dose_grid") or []]
    expected_checkpoint_names = [
        coefficient_checkpoint_name(coefficient) for coefficient in grid
    ]
    raw_checkpoint_names = [
        path.name
        for path in results_dir.glob("checkpoint_coeff_*.json")
        if path.is_file()
    ]
    invalid_checkpoint_names = [
        name for name in raw_checkpoint_names if not CHECKPOINT_NAME_RE.fullmatch(name)
    ]
    found_checkpoint_names = sorted(
        name for name in raw_checkpoint_names if CHECKPOINT_NAME_RE.fullmatch(name)
    )
    checks["complete_configured_grid"] = (
        bool(grid)
        and not invalid_checkpoint_names
        and found_checkpoint_names == sorted(expected_checkpoint_names)
    )

    ablation_curve: list[dict[str, float]] = []
    addition_curve: list[dict[str, float]] = []
    if checks["complete_configured_grid"]:
        try:
            for coefficient in grid:
                checkpoint = _load(results_dir / coefficient_checkpoint_name(coefficient))
                results = checkpoint.get("results") or {}
                if float(results["coefficient"]) != float(coefficient):
                    checks["complete_configured_grid"] = False
                    break
                ablation_curve.append(
                    {
                        "coefficient": float(results["coefficient"]),
                        "harmful_refusal_drop": float(results["harmful_refusal_drop"]),
                        "harmless_kl": float(results["harmless_kl"]),
                    }
                )
                addition_curve.append(
                    {
                        "coefficient": float(results["coefficient"]),
                        "harmless_refusal_increase": float(
                            results["harmless_refusal_increase"]
                        ),
                    }
                )
        except (KeyError, TypeError, ValueError):
            checks["complete_configured_grid"] = False

    selection_path = results_dir / "02_dose_selection.json"
    selection_pair = None
    if selection_path.is_file():
        selection = _load(selection_path)
        results = selection.get("results") or {}
        selection_pair = results.get("selected_pair")
        dumped = str(results)
        if "confirmation" in dumped:
            checks["confirmation_does_not_influence_selection"] = False
        try:
            recomputed = select_dose_coefficients(
                ablation_curve=ablation_curve,
                addition_curve=addition_curve,
                min_harmful_ablation_drop=float(
                    config["causal_gate"]["min_harmful_ablation_drop"]
                ),
                max_harmless_ablation_kl=float(
                    config["causal_gate"]["max_harmless_ablation_kl"]
                ),
                min_harmless_addition_increase=float(
                    config["causal_gate"]["min_harmless_addition_increase"]
                ),
            )
            if recomputed["selected_pair"] != results.get("selected_pair"):
                checks["selector_reproducible_from_curve"] = False
            if recomputed["ablation_coefficient"] != results.get("ablation_coefficient"):
                checks["selector_reproducible_from_curve"] = False
            if recomputed["addition_coefficient"] != results.get("addition_coefficient"):
                checks["selector_reproducible_from_curve"] = False
        except Exception:
            checks["selector_reproducible_from_curve"] = False

    confirmation_path = results_dir / "03_dose_confirmation.json"
    confirmation_pair = None
    if confirmation_path.is_file() and evaluation.get("causal_gate") is not None:
        confirmation = _load(confirmation_path)
        conf_results = confirmation.get("results") or {}
        confirmation_pair = conf_results.get("selected_pair")
        try:
            recomputed_gate = evaluate_causal_gate(
                baseline_harmful_refusal_rate=float(
                    conf_results["baseline"]["harmful_refusal_rate"]
                ),
                baseline_harmless_refusal_rate=float(
                    conf_results["baseline"]["harmless_refusal_rate"]
                ),
                ablation_harmful_refusal_rate=float(
                    conf_results["ablation"]["harmful_refusal_rate"]
                ),
                addition_harmless_refusal_rate=float(
                    conf_results["addition"]["harmless_refusal_rate"]
                ),
                random_control_harmful_refusal_rate=float(
                    conf_results["random_control"]["harmful_refusal_rate"]
                ),
                ablation_harmless_kl=float(conf_results["ablation"]["harmless_kl"]),
                thresholds=config["causal_gate"],
            )
            if not _checks_match(evaluation["causal_gate"], recomputed_gate):
                checks["gate_calculations_consistent"] = False
            stored_passed = bool(evaluation["causal_gate"].get("passed"))
            expected_status = (
                STATUS_PASS if recomputed_gate.passed else STATUS_CAUSAL_GATE_FAILED
            )
            if status == STATUS_PASS:
                checks["status_matches_gate"] = (
                    stored_passed
                    and recomputed_gate.passed
                    and expected_status == STATUS_PASS
                )
            elif status == STATUS_CAUSAL_GATE_FAILED:
                checks["status_matches_gate"] = (
                    (not stored_passed)
                    and (not recomputed_gate.passed)
                    and expected_status == STATUS_CAUSAL_GATE_FAILED
                )
            else:
                checks["status_matches_gate"] = False
            ablation_coefficient = float((confirmation_pair or {}).get("ablation", -1))
            random_coefficient = (conf_results.get("random_control") or {}).get(
                "coefficient"
            )
            checks["random_control_uses_ablation_coefficient"] = (
                random_coefficient is not None
                and float(random_coefficient) == ablation_coefficient
            )
        except (KeyError, TypeError, ValueError):
            checks["gate_calculations_consistent"] = False
            checks["status_matches_gate"] = False
            checks["random_control_uses_ablation_coefficient"] = False
    elif status in {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED}:
        checks["gate_calculations_consistent"] = False
        checks["status_matches_gate"] = False

    evaluation_pair = evaluation.get("selected_pair")
    pairs = [item for item in (selection_pair, confirmation_pair, evaluation_pair) if item]
    if status == STATUS_NO_DOSE_QUALIFIED:
        checks["selected_pair_consistent"] = (
            selection_pair is None and evaluation_pair is None
        )
    elif status in {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED}:
        checks["selected_pair_consistent"] = bool(
            selection_pair and confirmation_pair and evaluation_pair
        ) and all(item == pairs[0] for item in pairs)
    elif pairs:
        checks["selected_pair_consistent"] = all(item == pairs[0] for item in pairs)

    run_ids: set[str] = set()
    missing_run_id = not _run_id_ok(evaluation.get("run_id"))
    commits: set[str] = set()
    missing_commit = not _commit_ok(evaluation.get("implementation_commit"))
    for payload in payloads:
        provenance = payload.get("provenance") or {}
        run_value = provenance.get("run_id") or payload.get("run_id")
        commit_value = provenance.get("implementation_commit") or payload.get(
            "implementation_commit"
        )
        if _run_id_ok(run_value):
            run_ids.add(str(run_value))
        else:
            missing_run_id = True
        if _commit_ok(commit_value):
            commits.add(str(commit_value).strip())
        else:
            missing_commit = True
    checks["single_run_id"] = (not missing_run_id) and len(run_ids) == 1
    checks["code_provenance"] = (not missing_commit) and len(commits) == 1

    state = status_payload.get("state")
    abort_gate = status_payload.get("abort_gate")
    if status == STATUS_PASS:
        checks["status_matches_result"] = state == "completed"
    elif status == STATUS_NO_DOSE_QUALIFIED:
        checks["status_matches_result"] = state == "aborted" and abort_gate == "dose"
    elif status == STATUS_CAUSAL_GATE_FAILED:
        checks["status_matches_result"] = state == "aborted" and abort_gate == "causal"
    elif status == STATUS_OPERATIONAL_ERROR:
        checks["status_matches_result"] = state == "failed"
    else:
        checks["status_matches_result"] = False

    manifest_path = results_dir / "execution_manifest.json"
    if manifest_path.is_file():
        manifest = _load(manifest_path)
        manifest_status = (manifest.get("results") or {}).get("status")
        checks["manifest_agrees"] = manifest_status == status
    else:
        checks["manifest_agrees"] = status == STATUS_OPERATIONAL_ERROR

    events_path = artifacts_dir / EVENTS_FILENAME
    checks["events_present"] = events_path.is_file() and bool(
        events_path.read_text(encoding="utf-8").strip()
    )

    def _missing(value: object) -> bool:
        return value is None or value == ""

    for payload in payloads:
        provenance = payload.get("provenance") or {}
        if provenance and any(
            _missing(provenance.get(field)) for field in REQUIRED_PROVENANCE_FIELDS
        ):
            checks["required_provenance"] = False
        if "source_artifact_hashes" in payload and "status" in payload:
            if any(_missing(payload.get(field)) for field in REQUIRED_EVALUATION_FIELDS):
                checks["required_provenance"] = False

    original_status = root / "artifacts" / "qwen25_coder_7b" / STATUS_FILENAME
    original_checks = validate_qwen_pilot(root)
    if original_status.is_file():
        checks["original_pilot_validator"] = all(original_checks.values())
    else:
        checks["original_pilot_validator"] = True

    expected_files = set(required)
    if status != STATUS_OPERATIONAL_ERROR:
        expected_files.update(expected_checkpoint_names)
    actual_files = {
        path.name for path in results_dir.iterdir() if path.is_file()
    }
    checks["file_list_matches"] = actual_files == expected_files

    expected_stage = {
        STATUS_PASS: "confirm",
        STATUS_CAUSAL_GATE_FAILED: "confirm",
        STATUS_NO_DOSE_QUALIFIED: "sweep",
        STATUS_OPERATIONAL_ERROR: status_payload.get("current_stage"),
    }.get(str(status))
    terminal_event = {
        STATUS_PASS: "stage_completed",
        STATUS_CAUSAL_GATE_FAILED: "aborted",
        STATUS_NO_DOSE_QUALIFIED: "aborted",
    }.get(str(status))
    events_ok = False
    if events_path.is_file() and terminal_event is not None:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("event") == terminal_event
                and record.get("state") == state
                and record.get("stage") == expected_stage
            ):
                events_ok = True
                break
    checks["observability_reconciles"] = (
        status_payload.get("run_id") == evaluation.get("run_id")
        and status_payload.get("profile") == DOSE_PROFILE
        and status_payload.get("current_stage") == expected_stage
        and status_payload.get("completed_units") == status_payload.get("total_units")
        and events_ok
    )

    return checks
