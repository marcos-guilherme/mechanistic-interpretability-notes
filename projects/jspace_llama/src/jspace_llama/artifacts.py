"""Builders for the two compact, fetched result artifacts."""

from __future__ import annotations

from typing import Any

from jspace_llama.evaluation import evaluate_experiment
from jspace_llama.jsonio import document_sha256
from jspace_llama.provenance import provenance_is_complete


SCHEMA_VERSION = 1
REASON_MESSAGES = {
    "incomplete_provenance": "reproducible provenance is incomplete",
    "primary_gates_failed": "no alpha passed every primary-prompt gate",
    "cuda_required": "a CUDA runtime is required",
    "asset_load_failed": "model or lens asset loading failed",
    "resource_exhausted": "GPU resources were exhausted",
    "tokenization_failed": "intervention concepts are not valid single tokens",
    "dimension_mismatch": "model and lens dimensions do not match",
    "layer_band_invalid": "the selected layer band is invalid",
    "lens_integrity_failed": "lens integrity verification failed",
    "readout_failed": "workspace readout preparation failed",
    "intervention_preparation_failed": "intervention preparation failed",
    "random_control_failed": "random control preparation failed",
    "evaluation_data_invalid": "required aggregate evaluation data is missing",
    "unexpected_error": "unexpected programming error; see notebook traceback",
}


def compact_reason(code: str) -> dict[str, str]:
    try:
        message = REASON_MESSAGES[code]
    except KeyError as error:
        raise ValueError(f"unknown compact reason code: {code}") from error
    return {"code": code, "message": message}


def build_execution_manifest(
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    terminal_status: str,
    status: str,
    provenance: dict[str, Any],
    reason: dict[str, str] | None,
) -> dict[str, Any]:
    if reason is not None:
        code = reason.get("code")
        if not isinstance(code, str) or reason != compact_reason(code):
            raise ValueError("manifest reason must be canonical")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "execution_manifest",
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "terminal_status": terminal_status,
        "status": status,
        "provenance_complete": provenance_is_complete(provenance),
        "provenance": provenance,
        "reason": reason,
    }


def build_artifacts(
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    terminal_status: str,
    provenance: dict[str, Any],
    raw_diagnostics: dict[str, Any] | None,
    reason_code: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete = provenance_is_complete(provenance)
    evaluated = (
        evaluate_experiment(raw_diagnostics)
        if raw_diagnostics is not None
        else {
            "primary": {
                "prompt_name": "main",
                "readout": {"spider_present": False},
                "alphas": [],
                "passing_alphas": [],
                "passing_alpha_count": 0,
            },
            "paraphrases": {
                "prompt_count": 0,
                "passing_prompt_alpha_count": 0,
            },
        }
    )
    gates_pass = evaluated["primary"]["passing_alpha_count"] > 0
    if reason_code is None and not complete:
        reason_code = "incomplete_provenance"
    if reason_code is None and not gates_pass:
        reason_code = "primary_gates_failed"
    reason = None if reason_code is None else compact_reason(reason_code)
    status = "success" if complete and gates_pass and reason is None else "inconclusive"
    manifest = build_execution_manifest(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        terminal_status=terminal_status,
        status=status,
        provenance=provenance,
        reason=reason,
    )
    final = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "final_evaluation",
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "manifest_sha256": document_sha256(manifest),
        "integrity": {
            "lens_sha256": provenance["lens"]["sha256"],
            "lens_integrity_verified": (
                provenance["lens"].get("integrity_verified") is True
            ),
        },
        **evaluated,
    }
    return manifest, final
