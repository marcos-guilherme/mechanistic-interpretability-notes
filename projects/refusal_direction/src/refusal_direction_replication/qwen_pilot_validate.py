"""Offline validator for compact Qwen causal-pilot artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .gates import (
    STATUS_BASELINE_GATE_FAILED,
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    evaluate_baseline_gate,
    evaluate_causal_gate,
)

QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
LEAK_KEYS = frozenset(
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
GENERIC_LEAK_KEY = re.compile(
    r"^(prompt|prompts|instruction|generation|generations|output|outputs|"
    r"response|responses|completion|completions|text)$",
    re.IGNORECASE,
)
ALLOWED_PROSE_KEYS = frozenset(
    {
        "reason",
        "abort_reason",
        "message",
        "abort_gate",
        "exception_category",
        "status",
    }
)
HEX_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
PASS_STAGE_FILES = (
    "00_preflight.json",
    "01_baseline.json",
    "02_candidates.json",
    "03_interventions.json",
    "final_evaluation.json",
    "pilot_evaluation.json",
)
BASELINE_FAIL_STAGE_FILES = (
    "00_preflight.json",
    "01_baseline.json",
    "pilot_evaluation.json",
)
REQUIRED_GATE_FIELDS = (
    "name",
    "passed",
    "threshold",
    "observed",
    "comparator",
    "reason",
)
REQUIRED_BASELINE_CHECKS = (
    "min_valid_harmful",
    "min_valid_harmless",
    "min_refusal_gap",
)
REQUIRED_CAUSAL_CHECKS = (
    "min_harmful_ablation_drop",
    "min_harmless_addition_increase",
    "max_random_control_harmful_drop",
    "max_harmless_ablation_kl",
)
SPLIT_ID_KEYS = (
    "harmful_train",
    "harmless_train",
    "harmful_val",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_files(status: str) -> tuple[str, ...]:
    if status == STATUS_BASELINE_GATE_FAILED:
        return BASELINE_FAIL_STAGE_FILES
    if status == STATUS_OPERATIONAL_ERROR:
        return ("pilot_evaluation.json",)
    return PASS_STAGE_FILES


def _looks_like_raw_text(value: str) -> bool:
    stripped = value.strip()
    if HEX_RE.fullmatch(stripped):
        return False
    if len(stripped) > 160:
        return True
    return len(stripped.split()) >= 8


def _walk_contains_leak(obj: Any, *, key: str | None = None) -> bool:
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            lowered = str(child_key).casefold()
            if lowered in LEAK_KEYS or GENERIC_LEAK_KEY.fullmatch(lowered):
                return True
            if _walk_contains_leak(value, key=str(child_key)):
                return True
        return False
    if isinstance(obj, list):
        return any(_walk_contains_leak(item, key=key) for item in obj)
    if isinstance(obj, str):
        if key is not None and str(key).casefold() in ALLOWED_PROSE_KEYS:
            return False
        return _looks_like_raw_text(obj)
    return False


def _all_finite(obj: Any) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(value) for value in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(item) for item in obj)
    return True


def _model_ok(payload: dict[str, Any]) -> bool:
    config = payload.get("config", {})
    provenance = payload.get("provenance", {})
    if not provenance and "status" in payload:
        return True
    name = config.get("model_name") or provenance.get("model_name")
    requested = config.get("model_revision") or provenance.get(
        "requested_model_revision"
    )
    resolved = provenance.get("resolved_model_revision")
    if name != QWEN_MODEL:
        return False
    if requested != QWEN_REVISION:
        return False
    if resolved != QWEN_REVISION:
        return False
    return True


def _collect_mapping(
    payloads: list[dict[str, Any]],
    field: str,
) -> dict[str, set[str]]:
    collected: dict[str, set[str]] = {}
    for payload in payloads:
        values = payload.get("results", {}).get(field) or {}
        for name, items in values.items():
            collected.setdefault(name, set()).update(str(item) for item in items)
    return collected


def _disjoint_split_groups(groups: dict[str, set[str]]) -> bool:
    train = groups.get("harmful_train", set()) | groups.get("harmless_train", set())
    val = groups.get("harmful_val", set()) | groups.get("harmless_val", set())
    test = groups.get("harmful_test", set()) | groups.get("harmless_test", set())
    return train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


def _has_required_ids_and_hashes(payloads: list[dict[str, Any]]) -> bool:
    baseline = next(
        (
            payload
            for payload in payloads
            if payload.get("experiment") == "baseline"
        ),
        None,
    )
    if baseline is None:
        return False
    results = baseline.get("results", {})
    example_ids = results.get("example_ids") or {}
    content_hashes = results.get("content_hashes") or {}
    for key in SPLIT_ID_KEYS:
        if not example_ids.get(key) or not content_hashes.get(key):
            return False
    return True


def _checks_match(stored: dict[str, Any], recomputed) -> bool:
    if stored["passed"] != recomputed.passed:
        return False
    stored_checks = stored.get("checks") or []
    if len(stored_checks) != len(recomputed.checks):
        return False
    for stored_check, check in zip(stored_checks, recomputed.checks):
        if stored_check.get("name") != check.name:
            return False
        if bool(stored_check.get("passed")) != check.passed:
            return False
        if abs(float(stored_check.get("observed")) - float(check.observed)) > 1e-9:
            return False
        if abs(float(stored_check.get("threshold")) - float(check.threshold)) > 1e-9:
            return False
    return True


def _gate_fields_complete(evaluation: dict[str, Any], status: str) -> bool:
    baseline = evaluation.get("baseline_gate")
    causal = evaluation.get("causal_gate")
    if status == STATUS_PASS:
        if not baseline or not causal:
            return False
    if status == STATUS_BASELINE_GATE_FAILED and not baseline:
        return False
    if status == STATUS_CAUSAL_GATE_FAILED and (not baseline or not causal):
        return False
    for gate, required in (
        (baseline, REQUIRED_BASELINE_CHECKS),
        (causal, REQUIRED_CAUSAL_CHECKS),
    ):
        if gate is None:
            continue
        checks = gate.get("checks") or []
        names = [item.get("name") for item in checks]
        if list(names) != list(required):
            return False
        for item in checks:
            if any(field not in item for field in REQUIRED_GATE_FIELDS):
                return False
            if item.get("comparator") not in {"gte", "lte"}:
                return False
            if not item.get("reason"):
                return False
    return True


def _unique_candidate_grid(payload: dict[str, Any]) -> bool:
    scores = payload.get("results", {}).get("candidate_scores") or []
    if len(scores) != 28:
        return False
    layers = [item.get("layer") for item in scores]
    offsets = {item.get("position_offset") for item in scores}
    return sorted(layers) == list(range(28)) and offsets == {-1}


def validate_qwen_pilot(root: Path) -> dict[str, bool]:
    results_dir = root / "results" / "qwen25_coder_7b"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b"
    evaluation_path = results_dir / "pilot_evaluation.json"
    status_path = artifacts_dir / "status.json"
    checks = {
        "required_stages_present": False,
        "model_and_revision": True,
        "disjoint_split_ids": True,
        "disjoint_content_hashes": True,
        "required_ids_and_hashes": True,
        "candidate_count_is_28": True,
        "candidate_grid_is_28_unique_layers": True,
        "metrics_are_finite": True,
        "gate_calculations_consistent": True,
        "gate_fields_complete": True,
        "no_raw_prompt_or_generation_leakage": True,
        "status_matches_result": True,
        "scientific_pass_requires_both_gates": True,
        "single_run_id": True,
        "code_provenance": True,
    }
    if not evaluation_path.is_file() or not status_path.is_file():
        return checks
    evaluation = _load(evaluation_path)
    status_payload = _load(status_path)
    status = evaluation.get("status")
    required = _required_files(status)
    present = [(results_dir / name).is_file() for name in required]
    checks["required_stages_present"] = all(present) and status_path.is_file()

    compact_files = [
        path for path in sorted(results_dir.glob("*.json")) if path.is_file()
    ]
    payloads = [_load(path) for path in compact_files]
    for payload in payloads:
        if payload.get("config") or payload.get("provenance"):
            if not _model_ok(payload):
                checks["model_and_revision"] = False
        if not _all_finite(payload):
            checks["metrics_are_finite"] = False
        if _walk_contains_leak(payload):
            checks["no_raw_prompt_or_generation_leakage"] = False

    example_ids = _collect_mapping(payloads, "example_ids")
    content_hashes = _collect_mapping(payloads, "content_hashes")
    checks["required_ids_and_hashes"] = _has_required_ids_and_hashes(payloads)
    if example_ids:
        checks["disjoint_split_ids"] = _disjoint_split_groups(example_ids)
    else:
        checks["disjoint_split_ids"] = False
    if content_hashes:
        checks["disjoint_content_hashes"] = _disjoint_split_groups(content_hashes)
    else:
        checks["disjoint_content_hashes"] = False
    if status == STATUS_OPERATIONAL_ERROR:
        checks["required_ids_and_hashes"] = True
        checks["disjoint_split_ids"] = True
        checks["disjoint_content_hashes"] = True

    candidates_path = results_dir / "02_candidates.json"
    interventions_path = results_dir / "03_interventions.json"
    if status in {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED}:
        if candidates_path.is_file():
            candidate_payload = _load(candidates_path)
            if candidate_payload.get("results", {}).get("candidate_count") != 28:
                checks["candidate_count_is_28"] = False
        else:
            checks["candidate_count_is_28"] = False
        if interventions_path.is_file():
            intervention_payload = _load(interventions_path)
            if len(intervention_payload.get("results", {}).get("candidate_scores", [])) != 28:
                checks["candidate_count_is_28"] = False
            if not _unique_candidate_grid(intervention_payload):
                checks["candidate_grid_is_28_unique_layers"] = False
        else:
            checks["candidate_count_is_28"] = False
            checks["candidate_grid_is_28_unique_layers"] = False
    else:
        checks["candidate_grid_is_28_unique_layers"] = True

    baseline_path = results_dir / "01_baseline.json"
    final_path = results_dir / "final_evaluation.json"
    try:
        if baseline_path.is_file() and evaluation.get("baseline_gate") is not None:
            baseline = _load(baseline_path)
            config = baseline["config"]
            harmful = baseline["results"]["harmful_validation"]
            harmless = baseline["results"]["harmless_validation"]
            recomputed_baseline = evaluate_baseline_gate(
                harmful_count=int(
                    harmful.get("n_valid", len(harmful["refusal_flags"]))
                ),
                harmless_count=int(
                    harmless.get("n_valid", len(harmless["refusal_flags"]))
                ),
                harmful_refusal_rate=float(harmful["refusal_rate"]),
                harmless_refusal_rate=float(harmless["refusal_rate"]),
                thresholds=config["baseline_gate"],
            )
            if not _checks_match(evaluation["baseline_gate"], recomputed_baseline):
                checks["gate_calculations_consistent"] = False
        if (
            final_path.is_file()
            and evaluation.get("causal_gate") is not None
            and status != STATUS_BASELINE_GATE_FAILED
        ):
            final = _load(final_path)
            config = final["config"]
            results = final["results"]
            recomputed_causal = evaluate_causal_gate(
                baseline_harmful_refusal_rate=float(
                    results["baseline"]["harmful_refusal_rate"]
                ),
                baseline_harmless_refusal_rate=float(
                    results["baseline"]["harmless_refusal_rate"]
                ),
                ablation_harmful_refusal_rate=float(
                    results["ablation"]["harmful_refusal_rate"]
                ),
                addition_harmless_refusal_rate=float(
                    results["addition"]["harmless_refusal_rate"]
                ),
                random_control_harmful_refusal_rate=float(
                    results["random_control"]["harmful_refusal_rate"]
                ),
                ablation_harmless_kl=float(results["ablation"]["harmless_kl"]),
                thresholds=config["causal_gate"],
            )
            if not _checks_match(evaluation["causal_gate"], recomputed_causal):
                checks["gate_calculations_consistent"] = False
    except (KeyError, TypeError, ValueError):
        checks["gate_calculations_consistent"] = False

    checks["gate_fields_complete"] = _gate_fields_complete(evaluation, status)

    baseline_passed = bool(
        evaluation.get("baseline_gate") and evaluation["baseline_gate"].get("passed")
    )
    causal_passed = bool(
        evaluation.get("causal_gate") and evaluation["causal_gate"].get("passed")
    )
    if status == STATUS_PASS:
        checks["scientific_pass_requires_both_gates"] = (
            baseline_passed and causal_passed
        )
    else:
        checks["scientific_pass_requires_both_gates"] = True

    run_ids = {
        payload.get("provenance", {}).get("run_id")
        for payload in payloads
        if payload.get("provenance")
    }
    run_ids.add(evaluation.get("run_id"))
    run_ids.discard(None)
    checks["single_run_id"] = len(run_ids) == 1

    commits = {
        payload.get("provenance", {}).get("implementation_commit")
        for payload in payloads
        if payload.get("provenance")
    }
    if evaluation.get("implementation_commit"):
        commits.add(evaluation.get("implementation_commit"))
    commits.discard(None)
    checks["code_provenance"] = len(commits) == 1 and bool(next(iter(commits), ""))
    if any(
        payload.get("provenance")
        and not payload.get("provenance", {}).get("implementation_commit")
        for payload in payloads
    ):
        checks["code_provenance"] = False

    eval_status = evaluation.get("status")
    state = status_payload.get("state")
    abort_gate = status_payload.get("abort_gate")
    if eval_status == STATUS_PASS:
        checks["status_matches_result"] = (
            state == "completed" and checks["scientific_pass_requires_both_gates"]
        )
    elif eval_status == STATUS_BASELINE_GATE_FAILED:
        checks["status_matches_result"] = (
            state == "aborted" and abort_gate == "baseline"
        )
    elif eval_status == STATUS_CAUSAL_GATE_FAILED:
        checks["status_matches_result"] = (
            state == "aborted" and abort_gate == "causal"
        )
    elif eval_status == STATUS_OPERATIONAL_ERROR:
        checks["status_matches_result"] = state == "failed"
    else:
        checks["status_matches_result"] = False
    return checks
