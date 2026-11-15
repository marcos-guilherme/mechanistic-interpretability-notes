"""Offline validator for compact Qwen causal-pilot artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .gates import (
    STATUS_BASELINE_GATE_FAILED,
    STATUS_CAUSAL_GATE_FAILED,
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
    }
)
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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_files(status: str) -> tuple[str, ...]:
    if status == STATUS_BASELINE_GATE_FAILED:
        return BASELINE_FAIL_STAGE_FILES
    return PASS_STAGE_FILES


def _walk_contains_leak(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).casefold() in LEAK_KEYS:
                return True
            if _walk_contains_leak(value):
                return True
        return False
    if isinstance(obj, list):
        return any(_walk_contains_leak(item) for item in obj)
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
    names = {
        config.get("model_name"),
        provenance.get("model_name"),
    }
    names.discard(None)
    revisions = {
        config.get("model_revision"),
        provenance.get("requested_model_revision"),
    }
    revisions.discard(None)
    return names <= {QWEN_MODEL} and revisions <= {QWEN_REVISION}


def _collect_example_ids(payloads: list[dict[str, Any]]) -> dict[str, set[str]]:
    collected: dict[str, set[str]] = {}
    for payload in payloads:
        example_ids = payload.get("results", {}).get("example_ids") or {}
        for name, values in example_ids.items():
            collected.setdefault(name, set()).update(str(item) for item in values)
    return collected


def _disjoint_split_ids(example_ids: dict[str, set[str]]) -> bool:
    train = example_ids.get("harmful_train", set()) | example_ids.get(
        "harmless_train", set()
    )
    val = example_ids.get("harmful_val", set()) | example_ids.get(
        "harmless_val", set()
    )
    test = example_ids.get("harmful_test", set()) | example_ids.get(
        "harmless_test", set()
    )
    return train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)


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


def validate_qwen_pilot(root: Path) -> dict[str, bool]:
    results_dir = root / "results" / "qwen25_coder_7b"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b"
    evaluation_path = results_dir / "pilot_evaluation.json"
    status_path = artifacts_dir / "status.json"
    checks = {
        "required_stages_present": False,
        "model_and_revision": True,
        "disjoint_split_ids": True,
        "candidate_count_is_28": True,
        "metrics_are_finite": True,
        "gate_calculations_consistent": True,
        "no_raw_prompt_or_generation_leakage": True,
        "status_matches_result": True,
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
        path
        for path in sorted(results_dir.glob("*.json"))
        if path.is_file()
    ]
    payloads = [_load(path) for path in compact_files]
    for payload in payloads:
        if not _model_ok(payload):
            checks["model_and_revision"] = False
        if not _all_finite(payload):
            checks["metrics_are_finite"] = False
        if _walk_contains_leak(payload):
            checks["no_raw_prompt_or_generation_leakage"] = False

    example_ids = _collect_example_ids(payloads)
    if example_ids:
        checks["disjoint_split_ids"] = _disjoint_split_ids(example_ids)

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
            scores = interventions_path and _load(interventions_path).get(
                "results", {}
            ).get("candidate_scores", [])
            if len(scores) != 28:
                checks["candidate_count_is_28"] = False
        else:
            checks["candidate_count_is_28"] = False

    baseline_path = results_dir / "01_baseline.json"
    final_path = results_dir / "final_evaluation.json"
    try:
        if baseline_path.is_file() and evaluation.get("baseline_gate") is not None:
            baseline = _load(baseline_path)
            config = baseline["config"]
            harmful = baseline["results"]["harmful_validation"]
            harmless = baseline["results"]["harmless_validation"]
            recomputed_baseline = evaluate_baseline_gate(
                harmful_count=len(harmful["refusal_flags"]),
                harmless_count=len(harmless["refusal_flags"]),
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

    eval_status = evaluation.get("status")
    state = status_payload.get("state")
    abort_gate = status_payload.get("abort_gate")
    if eval_status == STATUS_PASS:
        checks["status_matches_result"] = state == "completed"
    elif eval_status == STATUS_BASELINE_GATE_FAILED:
        checks["status_matches_result"] = (
            state == "aborted" and abort_gate == "baseline"
        )
    elif eval_status == STATUS_CAUSAL_GATE_FAILED:
        checks["status_matches_result"] = (
            state == "aborted" and abort_gate == "causal"
        )
    else:
        checks["status_matches_result"] = False
    return checks
