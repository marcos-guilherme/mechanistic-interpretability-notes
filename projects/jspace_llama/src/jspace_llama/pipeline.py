"""Notebook-callable real experiment and artifact pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable
import uuid

import torch

from jspace_llama.artifacts import build_artifacts, build_execution_manifest
from jspace_llama.config import ExperimentConfig
from jspace_llama.evaluation import EvaluationDataError
from jspace_llama.experiment import (
    ModelExperimentRunner,
    prepare_prompts,
    run_experiment,
)
from jspace_llama.jsonio import atomic_write_json
from jspace_llama.provenance import (
    build_experiment_provenance,
    collect_software_provenance,
    file_sha256,
    read_source_provenance,
    resolve_huggingface_revisions,
)
from jspace_llama.readout import read_workspace
from jspace_llama.runtime import (
    JSpaceExperimentError,
    load_components,
    require_cuda,
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
        source = {"commit": None, "dirty": None}
    return {
        "source": source,
        "model": {
            "id": config.assets.model_id,
            "resolved_revision": None,
        },
        "lens": {
            "repo_id": config.assets.lens_repo_id,
            "filename": config.assets.lens_filename,
            "resolved_revision": None,
            "sha256": None,
            "size_bytes": None,
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
            },
            "layers": [],
            "alphas": list(config.sweep.alphas),
            "intervention": {
                "source_concept": config.source,
                "target_concept": config.target,
                "prefill_only": True,
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
    model_path = snapshot_download(
        repo_id=config.assets.model_id,
        revision=model_revision,
    )
    lens_path = hf_hub_download(
        repo_id=config.assets.lens_repo_id,
        filename=config.assets.lens_filename,
        revision=lens_revision,
    )
    lens_hash, lens_size = file_sha256(lens_path)
    provenance["lens"]["sha256"] = lens_hash
    provenance["lens"]["size_bytes"] = lens_size
    bundle = load_components(
        config.assets,
        lens_path,
        layers=config.sweep.layers,
        model_source=model_path,
    )
    validate_intervention_tokens(
        bundle.model.tokenizer,
        (config.source, config.target),
    )
    prepared = prepare_prompts(config, bundle.model.tokenizer)
    readout = read_workspace(
        bundle.model,
        bundle.dictionary,
        prepared[0].rendered,
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
        },
        "lens": {
            "repo_id": config.assets.lens_repo_id,
            "filename": config.assets.lens_filename,
            "resolved_revision": lens_revision,
            "sha256": lens_hash,
            "size_bytes": lens_size,
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


def _error_code(error: BaseException) -> str:
    if isinstance(error, torch.OutOfMemoryError):
        return "cuda_oom"
    if type(error) is JSpaceExperimentError:
        return "jspace_experiment_error"
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(error).__name__).lower()
    return name.removesuffix("_error") + "_error"


def _safe_error_message(error: BaseException) -> str:
    message = str(error)
    message = re.sub(
        r"(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s'\";,]+",
        "<local-path>",
        message,
    )
    message = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "<credential>", message)
    return message[:500] or type(error).__name__


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
    except (JSpaceExperimentError, torch.OutOfMemoryError) as error:
        reason = {
            "code": _error_code(error),
            "message": _safe_error_message(error),
        }
        manifest, final = build_artifacts(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            terminal_status="expected_failure",
            provenance=provenance,
            raw_diagnostics=None,
            reason=reason,
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(evaluation_path, final)
        return final
    except BaseException as error:
        manifest = build_execution_manifest(
            run_id=run_id,
            started_at=started_at,
            ended_at=_utc_now(),
            terminal_status="unexpected_error",
            status="error",
            provenance=provenance,
            reason={
                "code": "unexpected_error",
                "message": (
                    f"unexpected {type(error).__name__}; "
                    "see notebook traceback"
                ),
            },
        )
        atomic_write_json(manifest_path, manifest)
        raise
