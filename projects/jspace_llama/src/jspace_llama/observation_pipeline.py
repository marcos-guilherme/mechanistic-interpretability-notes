"""Integration pipeline for the non-generative observational atlas."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
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
    require_cuda,
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
    require_cuda()
    dataset_path = (
        project_root
        / "data"
        / "anthropic-jlens"
        / "evaluations"
        / "lens-eval-multihop.json"
    )
    try:
        items = load_multihop_items(dataset_path)
    except DatasetValidationError:
        raise
    except OSError as error:
        raise DatasetValidationError(
            "pinned multihop dataset is unavailable"
        ) from error
    runtime = load_pinned_runtime(config, provenance)
    bundle = runtime.bundle
    try:
        audit = audit_tokenization(items, bundle.model.tokenizer)
    except ValueError as error:
        if not (
            str(error) == "tokenization produced no token IDs"
            or str(error).startswith("invalid literal for int()")
        ):
            raise
        raise TokenizationError(
            "observational tokenization audit failed"
        ) from error
    if len(audit) != len(items) or any(
        not isinstance(entry, dict)
        or not isinstance(entry.get("intermediates"), (list, tuple))
        or not entry["intermediates"]
        or any(
            not isinstance(intermediate, dict)
            or not isinstance(intermediate.get("token_id"), int)
            or intermediate["token_id"] < 0
            or not isinstance(intermediate.get("token_ids"), (list, tuple))
            or not intermediate["token_ids"]
            or any(
                not isinstance(token_id, int) or token_id < 0
                for token_id in intermediate["token_ids"]
            )
            for intermediate in entry["intermediates"]
        )
        for entry in audit
    ):
        raise TokenizationError(
            "observational tokenization audit is incomplete"
        )
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


def _write_failure_raw(
    path: Path,
    *,
    provenance: dict[str, Any],
    reason_code: str,
    error_type: str,
) -> str:
    atomic_write_json(
        path,
        {
            "provenance": provenance,
            "failure": {
                "reason_code": reason_code,
                "error_type": error_type,
            },
        },
    )
    return _file_sha256(path)


def _publication_map(
    *,
    raw_path: Path,
    manifest_path: Path,
    result_path: Path,
    figure_dir: Path,
    staged_raw: Path | None,
    staged_manifest: Path,
    staged_result: Path,
    staged_figure_dir: Path | None,
) -> list[tuple[Path, Path | None]]:
    publications: list[tuple[Path, Path | None]] = [
        (raw_path, staged_raw),
        (manifest_path, staged_manifest),
        (result_path, staged_result),
    ]
    for stem in (
        "observation_pass_at_k",
        "observation_rank_heatmap",
        "observation_panel",
    ):
        for suffix in (".png", ".svg"):
            publications.append(
                (
                    figure_dir / f"{stem}{suffix}",
                    (
                        None
                        if staged_figure_dir is None
                        else staged_figure_dir / f"{stem}{suffix}"
                    ),
                )
            )
    return publications


def _publish_files(
    publications: list[tuple[Path, Path | None]],
    staging: Path,
) -> None:
    """Publish a prepared artifact set, rolling back ordinary replace failures."""
    backups = staging / "backups"
    backups.mkdir()
    originals: dict[Path, Path | None] = {}
    for index, (target, _source) in enumerate(publications):
        if target.exists():
            backup = backups / f"{index}.bak"
            shutil.copy2(target, backup)
            originals[target] = backup
        else:
            originals[target] = None
    try:
        for target, source in publications:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(source, target)
    except BaseException:
        for target, backup in originals.items():
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
        raise


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
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".observation-run-",
        dir=root,
    ) as staging_name:
        staging = Path(staging_name)
        staged_raw = staging / "jspace_observation_raw.json"
        staged_manifest = staging / "jspace_observation_manifest.json"
        staged_result = staging / "jspace_observation.json"
        staged_figure_dir = staging / "figures"
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
                raise TypeError(
                    "observational execute callback must return an object"
                )
            supplied_provenance = payload.get("provenance")
            if isinstance(supplied_provenance, dict):
                provenance = _observation_provenance(supplied_provenance)
            atomic_write_json(staged_raw, payload)
            raw_hash = _file_sha256(staged_raw)
            manifest, result = build_observation_artifacts(
                run_id=run_id,
                started_at=started_at,
                ended_at=_utc_now(),
                provenance=provenance,
                raw_artifact_sha256=raw_hash,
                payload=payload,
            )
            if (
                result["status"] == "inconclusive"
                and result["reason"]["code"] == "incomplete_observation"
            ):
                raw_hash = _write_failure_raw(
                    staged_raw,
                    provenance=provenance,
                    reason_code="incomplete_observation",
                    error_type="ObservationDataError",
                )
                manifest, result = build_observation_artifacts(
                    run_id=run_id,
                    started_at=started_at,
                    ended_at=_utc_now(),
                    provenance=provenance,
                    raw_artifact_sha256=raw_hash,
                    payload=None,
                    reason_code="incomplete_observation",
                )
            atomic_write_json(staged_manifest, manifest)
            atomic_write_json(staged_result, result)
            published_figures = None
            if result["status"] == "success":
                published_figures = write_observation_figures(
                    result,
                    staged_figure_dir,
                )
                if len(published_figures) != 6 or not all(
                    path.is_file() and path.stat().st_size > 0
                    for path in published_figures
                ):
                    raise RuntimeError(
                        "observational figure set is incomplete"
                    )
            _publish_files(
                _publication_map(
                    raw_path=raw_path,
                    manifest_path=manifest_path,
                    result_path=result_path,
                    figure_dir=figure_dir,
                    staged_raw=staged_raw,
                    staged_manifest=staged_manifest,
                    staged_result=staged_result,
                    staged_figure_dir=(
                        staged_figure_dir
                        if published_figures is not None
                        else None
                    ),
                ),
                staging,
            )
            return result
        except EXPECTED_ERRORS as error:
            reason_code = _reason_code(error)
            raw_hash = _write_failure_raw(
                staged_raw,
                provenance=provenance,
                reason_code=reason_code,
                error_type=type(error).__name__,
            )
            manifest, result = build_observation_artifacts(
                run_id=run_id,
                started_at=started_at,
                ended_at=_utc_now(),
                provenance=provenance,
                raw_artifact_sha256=raw_hash,
                payload=None,
                reason_code=reason_code,
            )
            atomic_write_json(staged_manifest, manifest)
            atomic_write_json(staged_result, result)
            _publish_files(
                _publication_map(
                    raw_path=raw_path,
                    manifest_path=manifest_path,
                    result_path=result_path,
                    figure_dir=figure_dir,
                    staged_raw=staged_raw,
                    staged_manifest=staged_manifest,
                    staged_result=staged_result,
                    staged_figure_dir=None,
                ),
                staging,
            )
            return result
        except Exception:
            if not manifest_path.exists():
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
