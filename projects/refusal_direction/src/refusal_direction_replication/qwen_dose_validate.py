"""Offline validator for compact Qwen dose follow-up artifacts."""

from __future__ import annotations

import json
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
)
from .run_progress import STATUS_FILENAME

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
        "manifest_agrees": True,
        "single_run_id": True,
        "code_provenance": True,
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
    else:
        checks["source_pilot_status"] = False

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
    checkpoint_paths = [
        results_dir / coefficient_checkpoint_name(coefficient) for coefficient in grid
    ]
    checks["complete_configured_grid"] = bool(grid) and all(
        path.is_file() for path in checkpoint_paths
    )

    selection_path = results_dir / "02_dose_selection.json"
    if selection_path.is_file():
        selection = _load(selection_path)
        results = selection.get("results") or {}
        dumped = str(results)
        if "confirmation" in dumped:
            checks["confirmation_does_not_influence_selection"] = False
        try:
            recomputed = select_dose_coefficients(
                ablation_curve=results.get("ablation_curve") or [],
                addition_curve=results.get("addition_curve") or [],
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
    if confirmation_path.is_file() and evaluation.get("causal_gate") is not None:
        confirmation = _load(confirmation_path)
        conf_results = confirmation.get("results") or {}
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
            expected_status = (
                STATUS_PASS if recomputed_gate.passed else STATUS_CAUSAL_GATE_FAILED
            )
            if status == STATUS_PASS and expected_status != STATUS_PASS:
                checks["gate_calculations_consistent"] = False
        except (KeyError, TypeError, ValueError):
            checks["gate_calculations_consistent"] = False
    elif status in {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED}:
        checks["gate_calculations_consistent"] = False

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

    return checks
