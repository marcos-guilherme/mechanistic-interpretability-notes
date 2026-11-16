"""Resumable Qwen Jacobian-lens fit and holdout orchestration.

Pilot and dose namespaces stay untouched. Fitting calls upstream
``jlens.fitting.jacobian_for_prompt`` and ``jlens.from_hf`` rather than
reimplementing Jacobian math.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from .gates import STATUS_OPERATIONAL_ERROR, STATUS_PASS, classify_operational_error
from .pipeline import (
    implementation_commit_hash,
    load_config,
    project_paths,
    release_compute_resources,
)
from .qwen_jlens import (
    CHECKPOINT_FILENAME,
    DATASET_CONFIG,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DIM_BATCH_PROBE_ORDER,
    FINAL_LENS_FILENAME,
    FIT_COUNT,
    HOLDOUT_COUNT,
    JLENS_PROFILE,
    JLENS_UPSTREAM_COMMIT,
    MAX_SEQ_LEN,
    MIN_TOKENS_BEFORE_TRUNCATION,
    QWEN_REVISION,
    SKIP_FIRST,
    SOURCE_DOSE_RUN_ID,
    assert_jlens_contract,
    build_codesearchnet_corpus,
    compact_corpus_records,
    config_hash,
    corpus_hash,
    load_exclusion_hashes,
    load_fit_checkpoint,
    probe_dim_batch,
    project_fit_duration,
    rank_one_based,
    summarize_holdout_metrics,
    valid_token_positions,
    validate_checkpoint_identity,
    validate_fitted_lens,
    write_fit_checkpoint,
)
from .qwen_pilot import resolve_run_id
from .run_progress import RunProgressStore

JLENS_STAGES = ("prepare", "probe", "fit", "holdout")
STATUS_ABORTED = "aborted"
LoadRows = Callable[..., Iterable[Mapping[str, Any]]]
LoadModel = Callable[..., tuple[Any, Any]]
WrapModel = Callable[..., Any]
JacobianFn = Callable[..., tuple[dict[int, torch.Tensor], int, int]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_compact(
    path: Path,
    *,
    experiment: str,
    config: dict[str, Any],
    results: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    artifact = ArtifactEnvelope(
        experiment=experiment,
        config=config,
        results=results,
        provenance=provenance,
    )
    write_artifact(path, artifact)
    return _read_json(path)


def _jlens_provenance(
    *,
    config: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    resolved_model_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": resolved_model_revision or config["model_revision"],
        "jlens_upstream_commit": config.get(
            "jlens_upstream_commit", JLENS_UPSTREAM_COMMIT
        ),
        "dataset_repo_id": config.get("dataset_repo_id", DATASET_REPO_ID),
        "dataset_revision": config.get("dataset_revision", DATASET_REVISION),
        "source_dose_run_id": config.get("source_dose_run_id", SOURCE_DOSE_RUN_ID),
    }


def require_source_dose_pass(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    relative = config.get(
        "source_dose_evaluation",
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json",
    )
    path = root / relative
    if not path.is_file():
        raise ValueError("source dose evaluation is missing")
    payload = _read_json(path)
    if payload.get("status") != STATUS_PASS:
        raise ValueError("source dose must remain pass")
    expected_run = config.get("source_dose_run_id", SOURCE_DOSE_RUN_ID)
    if payload.get("run_id") != expected_run:
        raise ValueError("source dose run_id does not match the passing follow-up")
    return payload


def default_load_rows(config: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        config.get("dataset_repo_id", DATASET_REPO_ID),
        config.get("dataset_config", DATASET_CONFIG),
        split=config.get("dataset_split", DATASET_SPLIT),
        revision=config.get("dataset_revision", DATASET_REVISION),
        streaming=True,
    )


def default_load_model(config: Mapping[str, Any]) -> tuple[Any, Any]:
    from .pipeline import _load_model_for_config

    model, tokenizer, _dtype = _load_model_for_config(config)
    return model, tokenizer


def default_wrap_model(model: Any, tokenizer: Any) -> Any:
    import jlens

    return jlens.from_hf(model, tokenizer)


def default_jacobian_for_prompt(*args: Any, **kwargs: Any) -> tuple[dict[int, torch.Tensor], int, int]:
    from jlens.fitting import jacobian_for_prompt

    return jacobian_for_prompt(*args, **kwargs)


def _encode_length(model: Any, tokenizer: Any, text: str, *, max_seq_len: int) -> int:
    if hasattr(model, "encode"):
        encoded = model.encode(text, max_length=max_seq_len)
        return int(encoded.shape[-1])
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    return int(ids.shape[-1])


def _zero_sums(source_layers: Sequence[int], d_model: int) -> dict[int, torch.Tensor]:
    return {
        int(layer): torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }


def _write_raw_corpus(path: Path, corpus: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for split, items in (("fit", corpus.fit), ("holdout", corpus.holdout)):
            for item in items:
                handle.write(
                    json.dumps(
                        {
                            "split": split,
                            "example_id": item.example_id,
                            "content_hash": item.content_hash,
                            "fit_text": item.fit_text,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def _write_evaluation(results_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(results_dir / "jlens_evaluation.json", payload)


def _write_manifest(
    results_dir: Path,
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    status: str,
    files: Sequence[str],
) -> None:
    _write_compact(
        results_dir / "execution_manifest.json",
        experiment="execution_manifest",
        config=config,
        results={"status": status, "result_files": list(files)},
        provenance=provenance,
    )


def build_jlens_evaluation(
    *,
    status: str,
    operational_success: bool,
    run_id: str,
    implementation_commit: str,
    model_revision: str,
    source_dose_run_id: str,
    source_dose_status: str,
    selected_dim_batch: int | None = None,
    abort_gate: str | None = None,
    abort_reason: str | None = None,
    message: str | None = None,
    exception_category: str | None = None,
    error_code: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "operational_success": operational_success,
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "model_revision": model_revision,
        "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source_dose_run_id": source_dose_run_id,
        "source_dose_status": source_dose_status,
        "n_fit": FIT_COUNT,
        "n_holdout": HOLDOUT_COUNT,
    }
    if selected_dim_batch is not None:
        payload["selected_dim_batch"] = selected_dim_batch
    if abort_gate is not None:
        payload["abort_gate"] = abort_gate
    if abort_reason is not None:
        payload["abort_reason"] = abort_reason
    if message is not None:
        payload["message"] = message
    if exception_category is not None:
        payload["exception_category"] = exception_category
    if error_code is not None:
        payload["error_code"] = error_code
    if stage is not None:
        payload["stage"] = stage
    return payload


def run_holdout_readout(
    *,
    lens: Any,
    model: Any,
    tokenizer: Any,
    items: Sequence[Any],
    source_layers: Sequence[int],
    skip_first: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        seq_len = _encode_length(model, tokenizer, item.fit_text, max_seq_len=max_seq_len)
        positions = valid_token_positions(seq_len, skip_first=skip_first)
        if not positions:
            raise ValueError(f"holdout item {item.example_id} has no valid positions")
        jlens_logits, model_logits, _ids = lens.apply(
            model,
            item.fit_text,
            layers=list(source_layers),
            positions=positions,
            max_seq_len=max_seq_len,
            use_jacobian=True,
        )
        logit_logits, _, _ = lens.apply(
            model,
            item.fit_text,
            layers=list(source_layers),
            positions=positions,
            max_seq_len=max_seq_len,
            use_jacobian=False,
        )
        targets = torch.argmax(model_logits.float(), dim=-1)
        for position_index, position in enumerate(positions):
            target_id = int(targets[position_index].item())
            for layer in source_layers:
                records.append(
                    {
                        "example_id": item.example_id,
                        "position": int(position),
                        "layer": int(layer),
                        "target_token_id": target_id,
                        "jlens_rank": rank_one_based(
                            jlens_logits[int(layer)][position_index], target_id
                        ),
                        "logit_lens_rank": rank_one_based(
                            logit_logits[int(layer)][position_index], target_id
                        ),
                    }
                )
    return records


def run_qwen_jlens(
    root: Path,
    *,
    profile: str = JLENS_PROFILE,
    run_id: str | None = None,
    load_rows: LoadRows | None = None,
    load_model: LoadModel | None = None,
    wrap_model: WrapModel | None = None,
    jacobian_for_prompt_fn: JacobianFn | None = None,
    lens_cls: Any | None = None,
    release_resources: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
    probe_clock: Callable[[], float] | None = None,
    assert_contract: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source_dose = require_source_dose_pass(root, config)
    resolved_run_id = resolve_run_id(artifacts_dir, run_id)
    commit = implementation_commit_hash()
    store = RunProgressStore(
        artifacts_dir, profile=profile, run_id=resolved_run_id
    )
    provenance = _jlens_provenance(
        config=config,
        run_id=resolved_run_id,
        implementation_commit=commit,
    )
    source_layers = [int(layer) for layer in config["source_layers"]]
    target_layer = int(config["target_layer"])
    skip_first = int(config.get("skip_first", SKIP_FIRST))
    max_seq_len = int(config.get("max_seq_len", MAX_SEQ_LEN))
    d_model = int(config["d_model"])
    hashed_config = config_hash(config)
    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    lens_path = artifacts_dir / FINAL_LENS_FILENAME
    model: Any = None
    tokenizer: Any = None
    bound: Any = None
    selected_dim_batch: int | None = None

    def release_loaded() -> None:
        nonlocal model, tokenizer, bound
        model = None
        tokenizer = None
        bound = None
        if release_resources is not None:
            release_resources()
        release_compute_resources(None, None)

    def ensure_model() -> tuple[Any, Any]:
        nonlocal model, tokenizer
        if model is None or tokenizer is None:
            loader = load_model or default_load_model
            model, tokenizer = loader(config) if load_model is None else loader()
        return model, tokenizer

    rows_fn = load_rows or (lambda **_kwargs: default_load_rows(config))
    wrap = wrap_model or default_wrap_model
    jacobian_fn = jacobian_for_prompt_fn or default_jacobian_for_prompt
    Lens = lens_cls
    contract = assert_contract or assert_jlens_contract

    try:
        contract()
        store.initialize(stage="prepare", total_units=FIT_COUNT)
        store.begin_running()
        _, tokenizer = ensure_model()
        exclusion = load_exclusion_hashes(
            root / config["exclusion_hashes_path"]
            if config.get("exclusion_hashes_path")
            else None
        )
        corpus = build_codesearchnet_corpus(
            rows_fn(),
            tokenizer=tokenizer,
            fit_count=int(config.get("fit_count", FIT_COUNT)),
            holdout_count=int(config.get("holdout_count", HOLDOUT_COUNT)),
            min_tokens=int(
                config.get("min_tokens_before_truncation", MIN_TOKENS_BEFORE_TRUNCATION)
            ),
            dataset_revision=str(config.get("dataset_revision", DATASET_REVISION)),
            exclusion_hashes=exclusion.hashes,
            exclusions_available=exclusion.available,
        )
        _write_raw_corpus(artifacts_dir / "corpus.jsonl", corpus)
        compact = compact_corpus_records(corpus)
        _write_compact(
            results_dir / "00_jlens_prepare.json",
            experiment="jlens_prepare",
            config=config,
            results=compact,
            provenance=provenance,
        )
        digest = corpus_hash(corpus)
        identity = {
            "source_layers": source_layers,
            "target_layer": target_layer,
            "skip_first": skip_first,
            "corpus_hash": digest,
            "config_hash": hashed_config,
            "model_revision": str(config["model_revision"]),
            "implementation_commit": commit,
            "run_id": resolved_run_id,
        }

        resumed = False
        jacobian_sum = _zero_sums(source_layers, d_model)
        n_done = 0
        next_idx = 0
        if checkpoint_path.is_file():
            loaded = load_fit_checkpoint(checkpoint_path)
            validate_checkpoint_identity(loaded, **identity)
            jacobian_sum = {
                int(layer): tensor for layer, tensor in loaded["jacobian_sum"].items()
            }
            n_done = int(loaded["n_done"])
            next_idx = int(loaded["next_idx"])
            selected_dim_batch = (
                int(loaded["dim_batch"]) if "dim_batch" in loaded else None
            )
            resumed = True

        if not resumed:
            store.update_progress(completed_units=0, total_units=FIT_COUNT)
            hf_model, tokenizer = ensure_model()
            bound = wrap(hf_model, tokenizer)
            probe_prompt = corpus.fit[0].fit_text

            def probe_runner(dim_batch: int) -> None:
                jacobian_fn(
                    bound,
                    probe_prompt,
                    source_layers,
                    target_layer=target_layer,
                    dim_batch=dim_batch,
                    max_seq_len=max_seq_len,
                    skip_first=skip_first,
                )

            try:
                probe = probe_dim_batch(
                    probe_runner,
                    order=tuple(
                        config.get("dim_batch_probe_order", DIM_BATCH_PROBE_ORDER)
                    ),
                    clock=probe_clock or clock,
                )
            except RuntimeError as exc:
                if "dim_batch 8 OOMed" not in str(exc):
                    raise
                evaluation = build_jlens_evaluation(
                    status=STATUS_ABORTED,
                    operational_success=True,
                    run_id=resolved_run_id,
                    implementation_commit=commit,
                    model_revision=str(config["model_revision"]),
                    source_dose_run_id=source_dose["run_id"],
                    source_dose_status=STATUS_PASS,
                    abort_gate="dim_batch_oom",
                    abort_reason="dim_batch 8 OOMed",
                    stage="probe",
                )
                _write_evaluation(results_dir, evaluation)
                store.abort(gate="dim_batch_oom", reason="dim_batch 8 OOMed")
                return evaluation
            selected_dim_batch = probe.selected
            store.append_public_event(
                {
                    "event": "probe_choice",
                    "stage": "probe",
                    "state": "running",
                    "dim_batch": selected_dim_batch,
                    "attempts": list(probe.attempts),
                    "wall_seconds": probe.wall_seconds,
                }
            )
            projection = project_fit_duration(
                probe_seconds=probe.wall_seconds,
                n_prompts=len(corpus.fit),
            )
            _write_compact(
                results_dir / "01_jlens_probe.json",
                experiment="jlens_probe",
                config=config,
                results={
                    "probe_order": list(
                        config.get("dim_batch_probe_order", DIM_BATCH_PROBE_ORDER)
                    ),
                    "attempts": list(probe.attempts),
                    "selected_dim_batch": selected_dim_batch,
                    "wall_seconds": probe.wall_seconds,
                    "projected_seconds": projection.projected_seconds,
                    "projected_limit_seconds": projection.limit_seconds,
                    "exceeds_limit": projection.exceeds_limit,
                },
                provenance=provenance,
            )
            if projection.exceeds_limit:
                evaluation = build_jlens_evaluation(
                    status=STATUS_ABORTED,
                    operational_success=True,
                    run_id=resolved_run_id,
                    implementation_commit=commit,
                    model_revision=str(config["model_revision"]),
                    source_dose_run_id=source_dose["run_id"],
                    source_dose_status=STATUS_PASS,
                    selected_dim_batch=selected_dim_batch,
                    abort_gate="projected_runtime",
                    abort_reason=(
                        "projected fit duration exceeds 12 hours: "
                        f"{projection.projected_seconds:.1f}s"
                    ),
                    stage="probe",
                )
                _write_evaluation(results_dir, evaluation)
                _write_manifest(
                    results_dir,
                    config=config,
                    provenance=provenance,
                    status=STATUS_ABORTED,
                    files=["00_jlens_prepare.json", "01_jlens_probe.json", "jlens_evaluation.json"],
                )
                store.abort(
                    gate="projected_runtime",
                    reason=str(evaluation["abort_reason"]),
                )
                return evaluation
        elif selected_dim_batch is None:
            selected_dim_batch = int(
                config.get("dim_batch_probe_order", DIM_BATCH_PROBE_ORDER)[0]
            )

        if bound is None:
            hf_model, tokenizer = ensure_model()
            bound = wrap(hf_model, tokenizer)
        store.initialize(stage="fit", total_units=FIT_COUNT)
        store.begin_running()
        if n_done:
            store.update_progress(completed_units=n_done, total_units=FIT_COUNT)
        for prompt_idx, item in enumerate(corpus.fit):
            if prompt_idx < next_idx:
                continue
            per_prompt, _seq_len, _n_valid = jacobian_fn(
                bound,
                item.fit_text,
                source_layers,
                target_layer=target_layer,
                dim_batch=int(selected_dim_batch),
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
            for layer in source_layers:
                jacobian_sum[layer] = jacobian_sum[layer] + per_prompt[layer]
            n_done += 1
            next_idx = prompt_idx + 1
            write_fit_checkpoint(
                checkpoint_path,
                jacobian_sum=jacobian_sum,
                n_done=n_done,
                next_idx=next_idx,
                dim_batch=int(selected_dim_batch),
                **identity,
            )
            store.update_progress(completed_units=n_done, total_units=FIT_COUNT)
            store.append_public_event(
                {
                    "event": "checkpoint",
                    "stage": "fit",
                    "state": "running",
                    "next_idx": next_idx,
                    "n_done": n_done,
                    "completed_units": n_done,
                    "total_units": FIT_COUNT,
                }
            )
        if n_done != len(corpus.fit):
            raise ValueError(f"fit completed {n_done} prompts, expected {len(corpus.fit)}")
        jacobian_mean = {
            layer: jacobian_sum[layer] / n_done for layer in source_layers
        }
        if Lens is None:
            from jlens.lens import JacobianLens as Lens
        lens = Lens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
        validate_fitted_lens(
            lens,
            source_layers=source_layers,
            d_model=d_model,
            n_prompts=n_done,
        )
        tmp_lens = lens_path.with_name(lens_path.name + ".tmp")
        lens.save(str(tmp_lens), dtype=torch.float16)
        tmp_lens.replace(lens_path)
        _write_compact(
            results_dir / "02_jlens_fit.json",
            experiment="jlens_fit",
            config=config,
            results={
                "n_done": n_done,
                "selected_dim_batch": selected_dim_batch,
                "checkpoint": str(Path(config["artifacts_dir"]) / CHECKPOINT_FILENAME),
                "lens": str(Path(config["artifacts_dir"]) / FINAL_LENS_FILENAME),
                "source_layers": source_layers,
                "target_layer": target_layer,
                "skip_first": skip_first,
                "corpus_hash": digest,
                "config_hash": hashed_config,
            },
            provenance=provenance,
        )
        loaded_lens = lens.load(str(lens_path))
        validate_fitted_lens(
            loaded_lens,
            source_layers=source_layers,
            d_model=d_model,
            n_prompts=n_done,
        )
        records = run_holdout_readout(
            lens=loaded_lens,
            model=bound,
            tokenizer=tokenizer,
            items=corpus.holdout,
            source_layers=source_layers,
            skip_first=skip_first,
            max_seq_len=max_seq_len,
        )
        metrics = summarize_holdout_metrics(records, source_layers=source_layers)
        _write_compact(
            results_dir / "03_jlens_holdout.json",
            experiment="jlens_holdout",
            config=config,
            results=metrics,
            provenance=provenance,
        )
        evaluation = build_jlens_evaluation(
            status=STATUS_PASS,
            operational_success=True,
            run_id=resolved_run_id,
            implementation_commit=commit,
            model_revision=str(config["model_revision"]),
            source_dose_run_id=source_dose["run_id"],
            source_dose_status=STATUS_PASS,
            selected_dim_batch=selected_dim_batch,
        )
        _write_evaluation(results_dir, evaluation)
        _write_manifest(
            results_dir,
            config=config,
            provenance=provenance,
            status=STATUS_PASS,
            files=[
                "00_jlens_prepare.json",
                "01_jlens_probe.json",
                "02_jlens_fit.json",
                "03_jlens_holdout.json",
                "jlens_evaluation.json",
                "execution_manifest.json",
            ],
        )
        store.complete_stage()
        return evaluation
    except Exception as exc:
        if isinstance(exc, ValueError) and "source dose" in str(exc):
            raise
        if isinstance(exc, ValueError) and "stale checkpoint" in str(exc):
            raise
        category, error_code, message = classify_operational_error(exc)
        evaluation = build_jlens_evaluation(
            status=STATUS_OPERATIONAL_ERROR,
            operational_success=False,
            run_id=resolved_run_id,
            implementation_commit=commit,
            model_revision=str(config.get("model_revision", QWEN_REVISION)),
            source_dose_run_id=str(source_dose.get("run_id", SOURCE_DOSE_RUN_ID))
            if "source_dose" in locals()
            else SOURCE_DOSE_RUN_ID,
            source_dose_status=str(source_dose.get("status", STATUS_PASS))
            if "source_dose" in locals()
            else STATUS_PASS,
            message=message,
            exception_category=category,
            error_code=error_code,
        )
        try:
            _write_evaluation(results_dir, evaluation)
        except Exception:
            pass
        try:
            store.fail(message)
        except Exception:
            pass
        raise
    finally:
        release_loaded()
