"""Integration pipeline for the non-generative observational atlas."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable
import uuid

import torch

from jspace_llama.config import ExperimentConfig
from jspace_llama.jsonio import atomic_write_json
from jspace_llama.observation_artifacts import (
    build_observation_artifacts,
)
from jspace_llama.observation_data import (
    DatasetValidationError,
    PANEL_NAMES,
    audit_tokenization,
    load_multihop_items,
)
from jspace_llama.observation_figures import write_observation_figures
from jspace_llama.observation_panel import read_qualitative_panel
from jspace_llama.observation_readout import evaluate_direct_readout
from jspace_llama.pipeline import (
    _partial_provenance,
    load_pinned_runtime,
)
from jspace_llama.provenance import (
    collect_software_provenance,
    read_source_provenance,
)
from jspace_llama.runtime import (
    AssetLoadError,
    CudaRequiredError,
    DimensionMismatchError,
    LayerBandError,
    LensIntegrityError,
    ResourceExhaustionError,
    TokenizationError,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _observation_provenance(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: provenance[key]
        for key in ("source", "model", "lens", "software", "hardware")
    }


def execute_real_observation(
    *,
    project_root: Path,
    config: ExperimentConfig,
    provenance: dict[str, Any],
) -> dict[str, object]:
    """Run direct and pursuit readouts without constructing generation requests."""
    runtime = load_pinned_runtime(config, provenance)
    bundle = runtime.bundle
    dataset_path = (
        project_root
        / "data"
        / "anthropic-jlens"
        / "evaluations"
        / "lens-eval-multihop.json"
    )
    items = load_multihop_items(dataset_path)
    audit = audit_tokenization(items, bundle.model.tokenizer)
    all_layers = tuple(
        dict.fromkeys(int(layer) for layer in bundle.lens.source_layers)
    )
    direct = evaluate_direct_readout(
        bundle.model,
        bundle.dictionary,
        items,
        audit,
        band_layers=bundle.layers,
        all_layers=all_layers,
        detail_names=frozenset(PANEL_NAMES),
        top=10,
    )
    panel = read_qualitative_panel(
        bundle.model,
        bundle.dictionary,
        items,
        direct["items"],
        layers=bundle.layers,
        k=25,
        top=10,
    )
    provenance.update(
        {
            "source": read_source_provenance(project_root),
            "software": collect_software_provenance(),
        }
    )
    return {
        "provenance": _observation_provenance(provenance),
        "tokenization_audit": list(audit),
        "direct": direct,
        "panel": panel,
        "layers": {
            "band": list(bundle.layers),
            "all_layers": list(all_layers),
        },
    }


EXPECTED_ERRORS = (
    CudaRequiredError,
    AssetLoadError,
    ResourceExhaustionError,
    TokenizationError,
    DimensionMismatchError,
    LayerBandError,
    LensIntegrityError,
    DatasetValidationError,
    torch.OutOfMemoryError,
)

ERROR_REASON_CODES = {
    CudaRequiredError: "cuda_required",
    AssetLoadError: "asset_load_failed",
    ResourceExhaustionError: "resource_exhausted",
    TokenizationError: "tokenization_failed",
    DimensionMismatchError: "dimension_mismatch",
    LayerBandError: "layer_band_invalid",
    LensIntegrityError: "lens_integrity_failed",
    DatasetValidationError: "dataset_invalid",
    torch.OutOfMemoryError: "resource_exhausted",
}


def _reason_code(error: BaseException) -> str:
    for error_type, code in ERROR_REASON_CODES.items():
        if isinstance(error, error_type):
            return code
    raise TypeError(f"unsupported expected error: {type(error).__name__}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_observation_pipeline(
    project_root: str | Path,
    *,
    config: ExperimentConfig | None = None,
    execute: Callable[[], dict[str, object]] | None = None,
    base_provenance: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Write raw, compact, manifest, and six deterministic figure artifacts."""
    root = Path(project_root)
    resolved_config = config or ExperimentConfig()
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    initial = base_provenance or _partial_provenance(root, resolved_config)
    provenance = _observation_provenance(initial)
    results_dir = root / "results"
    raw_path = root / "artifacts" / "jspace_observation_raw.json"
    manifest_path = results_dir / "jspace_observation_manifest.json"
    result_path = results_dir / "jspace_observation.json"
    figure_dir = results_dir / "figures"
    raw_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    for stem in (
        "observation_pass_at_k",
        "observation_rank_heatmap",
        "observation_panel",
    ):
        for suffix in (".png", ".svg"):
            (figure_dir / f"{stem}{suffix}").unlink(missing_ok=True)
    try:
        payload = (
            execute()
            if execute is not None
            else execute_real_observation(
                project_root=root,
                config=resolved_config,
                provenance=provenance,
            )
        )
        if not isinstance(payload, dict):
            raise TypeError("observational execute callback must return an object")
        supplied_provenance = payload.get("provenance")
        if isinstance(supplied_provenance, dict):
            provenance = _observation_provenance(supplied_provenance)
        atomic_write_json(raw_path, payload)
        raw_hash = _file_sha256(raw_path)
        manifest, result = build_observation_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            provenance=provenance,
            raw_artifact_sha256=raw_hash,
            payload=payload,
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(result_path, result)
        if result["status"] == "success":
            write_observation_figures(result, figure_dir)
        return result
    except EXPECTED_ERRORS as error:
        manifest, result = build_observation_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            provenance=provenance,
            raw_artifact_sha256=None,
            payload=None,
            reason_code=_reason_code(error),
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(result_path, result)
        return result
    except Exception:
        result_path.unlink(missing_ok=True)
        manifest, _result = build_observation_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            provenance=provenance,
            raw_artifact_sha256=None,
            payload=None,
            unexpected=True,
        )
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise
