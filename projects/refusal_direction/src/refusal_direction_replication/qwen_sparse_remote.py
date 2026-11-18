"""Production A100 stage execution for the sparse-refusal study.

This module owns heavy, private artifacts. The Task 4 runtime remains the
authority for compact receipts, ordering, provenance, and scientific gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


HEAVY_STAGES = (
    "prepare",
    "hardware_pilot",
    "sae_fit",
    "sae_quality",
    "decompose",
    "discover",
    "select",
    "test",
    "external_transfer",
    "code_preservation",
)
StageOperation = Callable[..., Mapping[str, Any]]


class ProductionSparseStageExecutor:
    """Dispatch approved stages to concrete Task 1–4 operations."""

    def __init__(
        self,
        root: Path,
        *,
        operations: Mapping[str, StageOperation] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.operations = dict(
            operations if operations is not None else production_operations()
        )
        required = {"prepare_environment", *HEAVY_STAGES}
        missing = required - set(self.operations)
        if missing:
            raise ValueError(
                f"sparse production operations missing: {sorted(missing)}"
            )

    def prepare_environment(self, root: Path) -> Mapping[str, str]:
        return self.operations["prepare_environment"](
            root=Path(root).resolve(),
        )

    def run_stage(
        self,
        stage: str,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        resume_from: Path | None,
    ) -> Mapping[str, Any]:
        if stage not in HEAVY_STAGES:
            raise ValueError(f"unsupported heavy stage: {stage}")
        return self.operations[stage](
            stage=stage,
            root=Path(root).resolve(),
            artifacts_dir=Path(artifacts_dir),
            store=store,
            resume_from=Path(resume_from) if resume_from else None,
        )


def production_operations() -> Mapping[str, StageOperation]:
    """Build the complete production operation table lazily."""
    from .qwen_sparse_remote_operations import production_operations as build

    return build()
