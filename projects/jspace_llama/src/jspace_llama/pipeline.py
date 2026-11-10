"""Notebook-callable real experiment and artifact pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable
import uuid

import torch

from jspace_llama.artifacts import (
    build_artifacts,
    build_execution_manifest,
    compact_reason,
)
from jspace_llama.config import ExperimentConfig
from jspace_llama.evaluation import EvaluationDataError
from jspace_llama.experiment import (
    ModelExperimentRunner,
    prepare_prompts,
    run_experiment,
)
from jspace_llama.interventions import (
    InterventionPreparationError,
    RandomControlError,
)
from jspace_llama.jsonio import atomic_write_json, strict_json_loads
from jspace_llama.provenance import (
    build_experiment_provenance,
    collect_software_provenance,
    file_sha256,
    read_source_provenance,
    resolve_huggingface_revisions,
    verify_lens_integrity,
)
from jspace_llama.readout import ReadoutError, read_workspace
from jspace_llama.runtime import (
    HUGGINGFACE_OPERATIONAL_ERRORS,
    AssetLoadError,
    CudaRequiredError,
    DimensionMismatchError,
    LayerBandError,
    LensIntegrityError,
    ResourceExhaustionError,
    TokenizationError,
    load_components,
    require_cuda,
    translate_huggingface_error,
    validate_intervention_tokens,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _partial_provenance(
    project_root: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    try:
        source = read_source_provenance(project_root)
    except (FileNotFoundError, ValueError):
        source = {
            "commit": None,
            "dirty": None,
            "scoped_dirty": None,
            "archive_sha256": None,
        }
    return {
        "source": source,
        "model": {
            "id": config.assets.model_id,
            "resolved_revision": None,
            "layer_count": None,
        },
        "lens": {
            "repo_id": config.assets.lens_repo_id,
            "filename": config.assets.lens_filename,
            "resolved_revision": None,
            "sha256": None,
            "size_bytes": None,
            "integrity_verified": None,
        },
        "software": collect_software_provenance(),
        "hardware": {
            "gpu_name": None,
            "cuda_version": None,
            "dtype": None,
        },
        "experiment": {
            "seed": config.seed,
            "prompts": [],
            "generation": {
                "max_new_tokens": config.generation.max_new_tokens,
                "do_sample": config.generation.do_sample,
                "decoding": "greedy",
                "first_line": False,
            },
            "layers": [],
            "alphas": list(config.sweep.alphas),
            "intervention": {
                "source_concept": config.source,
                "target_concept": config.target,
                "prefill_only": True,
                "mode": "source_to_target_clamp",
                "random_control": "norm_matched_orthogonal",
            },
        },
    }


def execute_real_experiment(
    *,
    project_root: Path,
    config: ExperimentConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, snapshot_download

    runtime_info = require_cuda()
    provenance["hardware"] = {
        "gpu_name": runtime_info.gpu_name,
        "cuda_version": str(torch.version.cuda),
        "dtype": runtime_info.dtype,
    }
    model_revision, lens_revision = resolve_huggingface_revisions(
        config.assets.model_id,
        config.assets.lens_repo_id,
    )
    provenance["model"]["resolved_revision"] = model_revision
    provenance["lens"]["resolved_revision"] = lens_revision
    try:
        model_path = snapshot_download(
            repo_id=config.assets.model_id,
            revision=model_revision,
        )
    except torch.OutOfMemoryError as error:
        raise ResourceExhaustionError("asset download exhausted memory") from error
    except HUGGINGFACE_OPERATIONAL_ERRORS as error:
        raise translate_huggingface_error(
            error,
            stage="model_download",
        ) from error
    except OSError as error:
        raise AssetLoadError(
            "failed to download pinned model",
            stage="model_download",
            category="io_error",
        ) from error
    try:
        lens_path = hf_hub_download(
            repo_id=config.assets.lens_repo_id,
            filename=config.assets.lens_filename,
            revision=lens_revision,
        )
    except torch.OutOfMemoryError as error:
        raise ResourceExhaustionError("asset download exhausted memory") from error
    except HUGGINGFACE_OPERATIONAL_ERRORS as error:
        raise translate_huggingface_error(
            error,
            stage="lens_download",
        ) from error
    except OSError as error:
        raise AssetLoadError(
            "failed to download pinned lens",
            stage="lens_download",
            category="io_error",
        ) from error
    lens_hash, lens_size = file_sha256(lens_path)
    provenance["lens"]["sha256"] = lens_hash
    provenance["lens"]["size_bytes"] = lens_size
    bundle = load_components(
        config.assets,
        lens_path,
        layers=config.sweep.layers,
        model_source=model_path,
    )
    provenance["model"]["layer_count"] = int(bundle.model.n_layers)
    provenance["lens"]["integrity_verified"] = verify_lens_integrity(
        lens_path,
        lens_hash,
        lens_size,
    )
    validate_intervention_tokens(
        bundle.model.tokenizer,
        (config.source, config.target),
    )
    prepared = prepare_prompts(config, bundle.model.tokenizer)
    readout = read_workspace(
        bundle.model,
        bundle.dictionary,
        prepared[0].token_ids,
        layers=bundle.layers,
        focus_concept=config.source,
    )
    experiment = run_experiment(
        config,
        bundle.model.tokenizer,
        layers=bundle.layers,
        generate=ModelExperimentRunner(bundle.model, bundle.dictionary),
        prepared_prompts=prepared,
    )
    provenance["experiment"] = build_experiment_provenance(
        config,
        prepared,
        bundle.layers,
    )
    provenance = {
        "source": read_source_provenance(project_root),
        "model": {
            "id": config.assets.model_id,
            "resolved_revision": model_revision,
            "layer_count": int(bundle.model.n_layers),
        },
        "lens": {
            "repo_id": config.assets.lens_repo_id,
            "filename": config.assets.lens_filename,
            "resolved_revision": lens_revision,
            "sha256": lens_hash,
            "size_bytes": lens_size,
            "integrity_verified": True,
        },
        "software": collect_software_provenance(),
        "hardware": provenance["hardware"],
        "experiment": provenance["experiment"],
    }
    return {
        "provenance": provenance,
        "readout": readout,
        "experiment": experiment,
    }


EXPECTED_ERRORS = (
    CudaRequiredError,
    AssetLoadError,
    ResourceExhaustionError,
    TokenizationError,
    DimensionMismatchError,
    LayerBandError,
    LensIntegrityError,
    ReadoutError,
    InterventionPreparationError,
    RandomControlError,
    EvaluationDataError,
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
    ReadoutError: "readout_failed",
    InterventionPreparationError: "intervention_preparation_failed",
    RandomControlError: "random_control_failed",
    EvaluationDataError: "evaluation_data_invalid",
    torch.OutOfMemoryError: "resource_exhausted",
}


def _reason_code(error: BaseException) -> str:
    for error_type, code in ERROR_REASON_CODES.items():
        if isinstance(error, error_type):
            return code
    raise TypeError(f"unsupported expected error type: {type(error).__name__}")


def _write_expected_diagnostics(
    path: Path,
    *,
    run_id: str,
    error: BaseException,
) -> None:
    def redact(value: str) -> str:
        value = re.sub(
            r"\bhf_[A-Za-z0-9]{8,}\b",
            "<credential>",
            value,
        )
        value = re.sub(
            r"(?i)\bBearer\s+\S+",
            "Bearer <credential>",
            value,
        )
        return re.sub(
            r"(?i)\b(api[_-]?key|password|secret|token)"
            r"\s*[:=]\s*\S+",
            r"\1=<credential>",
            value,
        )

    detail = redact(str(error))
    cause_detail = (
        None
        if error.__cause__ is None
        else redact(str(error.__cause__))
    )
    failure = {
        "run_id": run_id,
        "terminal_status": "expected_failure",
        "error_type": type(error).__name__,
        "detail": detail,
        "cause_type": (
            None
            if error.__cause__ is None
            else type(error.__cause__).__name__
        ),
        "cause_detail": cause_detail,
    }
    try:
        document: dict[str, Any] = failure
        if path.exists():
            existing = strict_json_loads(path.read_bytes())
            if isinstance(existing, dict):
                document = {**existing, "failure": failure}
        atomic_write_json(path, document)
    except Exception:
        pass


def run_pipeline(
    *,
    project_root: str | Path,
    config: ExperimentConfig | None = None,
    execute: Callable[[], dict[str, Any]] | None = None,
    base_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    resolved_config = config or ExperimentConfig()
    run_id = str(uuid.uuid4())
    started_at = _utc_now()
    provenance = base_provenance or _partial_provenance(
        root,
        resolved_config,
    )
    results_dir = root / "results"
    raw_path = root / "artifacts" / "readout_and_causal_swap.json"
    manifest_path = results_dir / "execution_manifest.json"
    evaluation_path = results_dir / "final_evaluation.json"
    try:
        for stale in (raw_path, manifest_path, evaluation_path):
            stale.unlink(missing_ok=True)
        payload = (
            execute()
            if execute is not None
            else execute_real_experiment(
                project_root=root,
                config=resolved_config,
                provenance=provenance,
            )
        )
        provenance = payload["provenance"]
        try:
            raw_diagnostics = {
                "readout": payload["readout"],
                "experiment": payload["experiment"],
            }
        except KeyError as error:
            raise EvaluationDataError(
                f"missing pipeline payload field: {error.args[0]}"
            ) from error
        atomic_write_json(raw_path, payload)
        manifest, final = build_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            terminal_status="completed",
            provenance=provenance,
            raw_diagnostics=raw_diagnostics,
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(evaluation_path, final)
        return final
    except EXPECTED_ERRORS as error:
        _write_expected_diagnostics(raw_path, run_id=run_id, error=error)
        manifest, final = build_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            terminal_status="expected_failure",
            provenance=provenance,
            raw_diagnostics=None,
            reason_code=_reason_code(error),
            failure_metadata=(
                {
                    "stage": error.stage,
                    "category": error.category,
                }
                if isinstance(error, AssetLoadError)
                else None
            ),
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(evaluation_path, final)
        return final
    except Exception:
        manifest = build_execution_manifest(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            terminal_status="unexpected_error",
            status="error",
            provenance=provenance,
            reason=compact_reason("unexpected_error"),
        )
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise
