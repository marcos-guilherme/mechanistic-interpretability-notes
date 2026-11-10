"""Builders for the two compact, fetched result artifacts."""

from __future__ import annotations

from typing import Any

from jspace_llama.evaluation import evaluate_experiment
from jspace_llama.jsonio import document_sha256
from jspace_llama.provenance import provenance_is_complete


SCHEMA_VERSION = 1


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
    reason: dict[str, str] | None = None,
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
    if reason is None and not complete:
        reason = {
            "code": "incomplete_provenance",
            "message": "success requires complete reproducible provenance",
        }
    if reason is None and not gates_pass:
        reason = {
            "code": "primary_gates_failed",
            "message": "no alpha passed every primary-prompt gate",
        }
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
        **evaluated,
    }
    return manifest, final
