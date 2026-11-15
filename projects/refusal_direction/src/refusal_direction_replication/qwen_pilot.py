"""Resumable Qwen causal-pilot orchestration.

Stages run through ``RunProgressStore``. Completed stages for the same run ID
are skipped. Scientific gate failures abort the run. Unexpected exceptions are
marked failed and re-raised. Model references and the CUDA cache are released
between stages.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .gates import (
    STATUS_BASELINE_GATE_FAILED,
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    build_pilot_evaluation,
    classify_operational_error,
    evaluate_baseline_gate,
    evaluate_causal_gate,
)
from .pipeline import (
    implementation_commit_hash,
    load_config,
    project_paths,
    release_compute_resources,
    run_baseline_stage,
    run_evaluation_stage,
    run_extract_stage,
    run_intervention_stage,
    run_preflight_stage,
)
from .run_progress import STATUS_FILENAME, RunProgressStore

PILOT_STAGES = (
    "preflight",
    "baseline",
    "extract",
    "interventions",
    "evaluation",
)
DEFAULT_PROFILE = "qwen25_coder_7b.json"
StageRunner = Callable[..., dict[str, Any]]


def resolve_run_id(artifacts_dir: Path, requested: str | None = None) -> str:
    """Reuse the run ID from status.json unless an explicit ID is provided."""

    if requested:
        return requested
    status_path = artifacts_dir / STATUS_FILENAME
    if status_path.is_file():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        existing = payload.get("run_id")
        if existing:
            return str(existing)
    return str(uuid.uuid4())


def _default_stage_runners() -> dict[str, StageRunner]:
    return {
        "preflight": lambda root, *, profile, run_id=None: run_preflight_stage(
            root, profile=profile, run_id=run_id
        ),
        "baseline": lambda root, *, profile, run_id=None: run_baseline_stage(
            root, profile=profile, run_id=run_id
        ),
        "extract": lambda root, *, profile, run_id=None: run_extract_stage(
            root, profile=profile, run_id=run_id
        ),
        "interventions": lambda root, *, profile, run_id=None: run_intervention_stage(
            root, profile=profile, run_id=run_id
        ),
        "evaluation": lambda root, *, profile, run_id=None: run_evaluation_stage(
            root, profile=profile, run_id=run_id
        ),
    }


def _write_pilot_evaluation(results_dir: Path, payload: dict[str, Any]) -> None:
    path = results_dir / "pilot_evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_gate_from_artifact(path: Path, config: dict[str, Any]):
    artifact = _read_json(path)
    results = artifact["results"]
    harmful = results["harmful_validation"]
    harmless = results["harmless_validation"]
    return evaluate_baseline_gate(
        harmful_count=int(harmful.get("n_valid", len(harmful["refusal_flags"]))),
        harmless_count=int(harmless.get("n_valid", len(harmless["refusal_flags"]))),
        harmful_refusal_rate=float(harmful["refusal_rate"]),
        harmless_refusal_rate=float(harmless["refusal_rate"]),
        thresholds=config["baseline_gate"],
    )


def _causal_gate_from_artifact(path: Path, config: dict[str, Any]):
    artifact = _read_json(path)
    results = artifact["results"]
    return evaluate_causal_gate(
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


def _load_existing_evaluation(
    results_dir: Path,
    *,
    run_id: str,
    fallback_status: str,
) -> dict[str, Any]:
    path = results_dir / "pilot_evaluation.json"
    if path.is_file():
        return _read_json(path)
    return {
        "status": fallback_status,
        "operational_success": True,
        "baseline_gate": None,
        "causal_gate": None,
        "run_id": run_id,
    }


def run_qwen_pilot(
    root: Path,
    *,
    run_id: str | None = None,
    profile: str = DEFAULT_PROFILE,
    stage_runners: dict[str, StageRunner] | None = None,
    release_resources: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run or resume the Qwen causal pilot for ``root``."""

    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    resolved_run_id = resolve_run_id(artifacts_dir, requested=run_id)
    store = RunProgressStore(
        artifacts_dir,
        profile=profile,
        run_id=resolved_run_id,
    )
    current = store.read_status()
    if (
        current is not None
        and current.get("run_id") == resolved_run_id
        and current.get("state") == "aborted"
    ):
        gate = current.get("abort_gate")
        fallback = (
            STATUS_BASELINE_GATE_FAILED
            if gate == "baseline"
            else STATUS_CAUSAL_GATE_FAILED
        )
        return _load_existing_evaluation(
            results_dir,
            run_id=resolved_run_id,
            fallback_status=fallback,
        )

    runners = stage_runners or _default_stage_runners()
    releaser = release_resources or release_compute_resources
    last_evaluation: dict[str, Any] | None = None
    commit: str | None = None

    for stage in PILOT_STAGES:
        if store.should_skip_stage(stage):
            continue
        try:
            store.initialize(stage=stage, total_units=1)
            store.begin_running()
            if commit is None:
                commit = implementation_commit_hash()
            runners[stage](root, profile=profile, run_id=resolved_run_id)
            if stage == "baseline":
                baseline_gate = _baseline_gate_from_artifact(
                    results_dir / "01_baseline.json",
                    config,
                )
                if not baseline_gate.passed:
                    last_evaluation = build_pilot_evaluation(
                        status=STATUS_BASELINE_GATE_FAILED,
                        operational_success=True,
                        baseline_gate=baseline_gate,
                        causal_gate=None,
                        run_id=resolved_run_id,
                        implementation_commit=commit,
                    )
                    _write_pilot_evaluation(results_dir, last_evaluation)
                    store.abort(gate="baseline", reason=baseline_gate.reason)
                    return last_evaluation
            if stage == "evaluation":
                baseline_gate = _baseline_gate_from_artifact(
                    results_dir / "01_baseline.json",
                    config,
                )
                causal_gate = _causal_gate_from_artifact(
                    results_dir / "final_evaluation.json",
                    config,
                )
                status = (
                    STATUS_PASS if causal_gate.passed else STATUS_CAUSAL_GATE_FAILED
                )
                last_evaluation = build_pilot_evaluation(
                    status=status,
                    operational_success=True,
                    baseline_gate=baseline_gate,
                    causal_gate=causal_gate,
                    run_id=resolved_run_id,
                    implementation_commit=commit,
                )
                _write_pilot_evaluation(results_dir, last_evaluation)
                if not causal_gate.passed:
                    store.abort(gate="causal", reason=causal_gate.reason)
                    return last_evaluation
            store.complete_stage()
        except Exception as exc:
            category, error_code, message = classify_operational_error(exc)
            last_evaluation = build_pilot_evaluation(
                status=STATUS_OPERATIONAL_ERROR,
                operational_success=False,
                baseline_gate=None,
                causal_gate=None,
                run_id=resolved_run_id,
                stage=stage,
                exception_category=category,
                error_code=error_code,
                message=message,
                implementation_commit=commit,
            )
            try:
                _write_pilot_evaluation(results_dir, last_evaluation)
            except Exception:
                pass
            try:
                store.fail(message)
            except Exception:
                pass
            raise
        finally:
            releaser()

    if last_evaluation is None:
        last_evaluation = build_pilot_evaluation(
            status=STATUS_PASS,
            operational_success=True,
            baseline_gate=None,
            causal_gate=None,
            run_id=resolved_run_id,
            implementation_commit=commit,
        )
        existing = results_dir / "pilot_evaluation.json"
        if existing.is_file():
            last_evaluation = _read_json(existing)
        else:
            _write_pilot_evaluation(results_dir, last_evaluation)
    return last_evaluation
