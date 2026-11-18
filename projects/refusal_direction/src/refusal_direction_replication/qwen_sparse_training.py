"""Deterministic streaming training and atomic checkpoints for the TopK SAE."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any

import torch

from .qwen_sparse_sae import (
    TopKSparseAutoencoder,
    remove_parallel_decoder_grad_,
    sae_loss,
)


CHECKPOINT_SCHEMA_VERSION = 2
IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "corpus_sha256",
        "model_revision",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "tokens_seen",
        "batch_index",
        "activation_scale",
        "optimizer_name",
        "schedule",
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "last_activation_token",
    }
)


@dataclass(frozen=True)
class ActivationBatch:
    """One normalized token-by-hidden activation batch."""

    activations: torch.Tensor


@dataclass(frozen=True)
class SaeMicrobatchResult:
    """Outputs needed by both canonical training and the hardware pilot."""

    token_count: int
    selected_indices: torch.Tensor
    loss_finite: bool


@dataclass(frozen=True)
class SaePilotMeasurement:
    """Steady-state result for one full effective training batch."""

    elapsed_seconds: float
    tokens_per_second: float
    finite: bool
    optimizer_state_allocated: bool


@dataclass(frozen=True)
class SparseTrainConfig:
    """Frozen training identity plus optimizer and checkpoint settings."""

    run_id: str
    implementation_commit: str
    config_sha256: str
    corpus_sha256: str
    model_revision: str
    max_learning_rate: float
    min_learning_rate: float
    warmup_tokens: int
    total_fit_tokens: int
    effective_batch_tokens: int
    checkpoint_every_tokens: int
    activation_scale: float
    dead_feature_window_tokens: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        for field, value in self.identity.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        if self.max_learning_rate <= 0:
            raise ValueError("max_learning_rate must be positive")
        if not 0 < self.min_learning_rate <= self.max_learning_rate:
            raise ValueError(
                "min_learning_rate must be positive and no greater than max"
            )
        if self.warmup_tokens <= 0:
            raise ValueError("warmup_tokens must be positive")
        if self.total_fit_tokens <= self.warmup_tokens:
            raise ValueError("total_fit_tokens must exceed warmup_tokens")
        if self.effective_batch_tokens <= 0:
            raise ValueError("effective_batch_tokens must be positive")
        if self.checkpoint_every_tokens <= 0:
            raise ValueError("checkpoint_every_tokens must be positive")
        if self.activation_scale <= 0:
            raise ValueError("activation_scale must be positive")
        if self.dead_feature_window_tokens <= 0:
            raise ValueError("dead_feature_window_tokens must be positive")

    @property
    def identity(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "implementation_commit": self.implementation_commit,
            "config_sha256": self.config_sha256,
            "corpus_sha256": self.corpus_sha256,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class SparseTrainingCheckpoint:
    schema_version: int
    identity: dict[str, str]
    tokens_seen: int
    batch_index: int
    activation_scale: float
    optimizer_name: str
    schedule: dict[str, Any]
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    python_rng_state: object
    torch_rng_state: torch.Tensor
    cuda_rng_state: list[torch.Tensor]
    last_activation_token: torch.Tensor
    checkpoint_path: Path


def _validate_complete_identity(
    identity: Mapping[str, str],
    *,
    label: str,
) -> dict[str, str]:
    keys = set(identity)
    if keys != IDENTITY_FIELDS:
        missing = sorted(IDENTITY_FIELDS - keys)
        extra = sorted(keys - IDENTITY_FIELDS)
        raise ValueError(
            f"{label} must contain complete identity; "
            f"missing={missing}, extra={extra}"
        )
    normalized = dict(identity)
    for field, value in normalized.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} field {field} must be a nonempty string")
    return normalized


def load_training_checkpoint(
    path: Path,
    expected_identity: Mapping[str, str],
) -> SparseTrainingCheckpoint:
    """Load a checkpoint onto CPU and require a complete exact identity."""
    expected = _validate_complete_identity(
        expected_identity,
        label="expected_identity",
    )
    checkpoint_path = Path(path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        raise ValueError(
            "legacy checkpoint schema_version 1 requires explicit retraining; "
            "automatic optimizer migration is not permitted"
        )
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    if set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("checkpoint fields do not match schema")
    identity_value = payload["identity"]
    if not isinstance(identity_value, Mapping):
        raise ValueError("checkpoint identity must be a mapping")
    actual = _validate_complete_identity(
        identity_value,
        label="checkpoint identity",
    )
    for field in sorted(IDENTITY_FIELDS):
        if actual[field] != expected[field]:
            raise ValueError(
                f"checkpoint identity mismatch for {field}: "
                f"{actual[field]!r} != {expected[field]!r}"
            )

    model_state = payload["model_state"]
    optimizer_state = payload["optimizer_state"]
    if payload["optimizer_name"] != "AdamW":
        raise ValueError("checkpoint optimizer must be AdamW")
    schedule = payload["schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("checkpoint schedule must be a mapping")
    last_activation_token = payload["last_activation_token"]
    torch_rng_state = payload["torch_rng_state"]
    cuda_rng_state = payload["cuda_rng_state"]
    if not isinstance(model_state, dict) or not all(
        isinstance(value, torch.Tensor) for value in model_state.values()
    ):
        raise ValueError("checkpoint model_state is invalid")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint optimizer_state is invalid")
    if not isinstance(last_activation_token, torch.Tensor):
        raise ValueError("checkpoint last_activation_token is invalid")
    if not isinstance(torch_rng_state, torch.Tensor):
        raise ValueError("checkpoint torch_rng_state is invalid")
    if not isinstance(cuda_rng_state, list) or not all(
        isinstance(state, torch.Tensor) for state in cuda_rng_state
    ):
        raise ValueError("checkpoint cuda_rng_state is invalid")

    return SparseTrainingCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        identity=actual,
        tokens_seen=int(payload["tokens_seen"]),
        batch_index=int(payload["batch_index"]),
        activation_scale=float(payload["activation_scale"]),
        optimizer_name="AdamW",
        schedule=dict(schedule),
        model_state=model_state,
        optimizer_state=optimizer_state,
        python_rng_state=payload["python_rng_state"],
        torch_rng_state=torch_rng_state,
        cuda_rng_state=cuda_rng_state,
        last_activation_token=last_activation_token,
        checkpoint_path=checkpoint_path,
    )


def _cpu_model_state(
    sae: TopKSparseAutoencoder,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in sae.state_dict().items()
    }


def _checkpoint_payload(
    sae: TopKSparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: SparseTrainConfig,
    *,
    tokens_seen: int,
    batch_index: int,
    last_activation_token: torch.Tensor,
) -> dict[str, Any]:
    cuda_rng_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else []
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": config.identity,
        "tokens_seen": tokens_seen,
        "batch_index": batch_index,
        "activation_scale": config.activation_scale,
        "optimizer_name": "AdamW",
        "schedule": _schedule_contract(config),
        "model_state": _cpu_model_state(sae),
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": cuda_rng_state,
        "last_activation_token": last_activation_token.detach().cpu().clone(),
    }


def _write_checkpoint(
    sae: TopKSparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: SparseTrainConfig,
    checkpoint_dir: Path,
    *,
    tokens_seen: int,
    batch_index: int,
    last_activation_token: torch.Tensor,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / (
        f"checkpoint-{tokens_seen:012d}-{batch_index:08d}.pt"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(
            _checkpoint_payload(
                sae,
                optimizer,
                config,
                tokens_seen=tokens_seen,
                batch_index=batch_index,
                last_activation_token=last_activation_token,
            ),
            temporary,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _move_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _restore_rng_state(checkpoint: SparseTrainingCheckpoint) -> None:
    random.setstate(checkpoint.python_rng_state)
    torch.random.set_rng_state(checkpoint.torch_rng_state)
    if checkpoint.cuda_rng_state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all(checkpoint.cuda_rng_state)


def _validate_activation_batch(
    activation_batch: ActivationBatch,
    d_in: int,
) -> torch.Tensor:
    x = activation_batch.activations
    if x.ndim != 2 or x.shape[1] != d_in:
        raise ValueError(f"activation batch shape must be [tokens, {d_in}]")
    if x.shape[0] == 0:
        raise ValueError("activation batch must contain at least one token")
    return x


def _schedule_contract(config: SparseTrainConfig) -> dict[str, Any]:
    return {
        "name": "linear_warmup_cosine",
        "max_learning_rate": config.max_learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "warmup_tokens": config.warmup_tokens,
        "total_fit_tokens": config.total_fit_tokens,
        "effective_batch_tokens": config.effective_batch_tokens,
    }


def learning_rate_for_tokens(
    config: SparseTrainConfig,
    tokens_seen: int,
) -> float:
    """Return the frozen linear-warmup/cosine learning rate."""
    if tokens_seen <= 0:
        return 0.0
    if tokens_seen < config.warmup_tokens:
        return config.max_learning_rate * (
            tokens_seen / config.warmup_tokens
        )
    if tokens_seen >= config.total_fit_tokens:
        return config.min_learning_rate
    progress = (
        (tokens_seen - config.warmup_tokens)
        / (config.total_fit_tokens - config.warmup_tokens)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.max_learning_rate - config.min_learning_rate
    ) * cosine


def create_sae_optimizer(
    sae: TopKSparseAutoencoder,
) -> torch.optim.AdamW:
    """Create the frozen AdamW optimizer used by training and pilot."""
    return torch.optim.AdamW(
        sae.parameters(),
        lr=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def accumulate_sae_microbatch(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    dead_mask: torch.Tensor,
) -> SaeMicrobatchResult:
    """Accumulate one token-weighted microbatch through the canonical loss."""
    if (
        activations.ndim != 2
        or activations.shape[1] != sae.d_in
        or activations.shape[0] == 0
    ):
        raise ValueError(
            f"activation batch shape must be [tokens, {sae.d_in}]"
        )
    if dead_mask.shape != (sae.d_sae,) or dead_mask.dtype != torch.bool:
        raise ValueError("dead feature mask shape or dtype is invalid")
    loss, metrics = sae_loss(sae, activations, dead_mask)
    token_count = int(activations.shape[0])
    (loss * token_count).backward()
    return SaeMicrobatchResult(
        token_count=token_count,
        selected_indices=metrics["selected_indices"],
        loss_finite=bool(torch.isfinite(loss).item()),
    )


def complete_sae_optimizer_step(
    sae: TopKSparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    *,
    accumulated_tokens: int,
    learning_rate: float,
) -> bool:
    """Project, step, and renormalize exactly as canonical SAE training."""
    if accumulated_tokens <= 0:
        raise ValueError("accumulated_tokens must be positive")
    if learning_rate < 0:
        raise ValueError("learning_rate must be nonnegative")
    for parameter in sae.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_tokens)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    remove_parallel_decoder_grad_(sae)
    gradients = [
        parameter.grad
        for parameter in sae.parameters()
        if parameter.grad is not None
    ]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    )
    optimizer.step()
    sae.renormalize_decoder_()
    optimizer.zero_grad(set_to_none=True)
    return gradients_finite


def measure_sae_training_candidate(
    sae: TopKSparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    batch_factory: Callable[[], Iterable[ActivationBatch]],
    *,
    device: str | torch.device,
    effective_batch_tokens: int,
    learning_rate: float,
    synchronize: Callable[[], None],
    reset_peak_memory: Callable[[], None],
    clock: Callable[[], float],
) -> SaePilotMeasurement:
    """Warm optimizer state, then time the same full update used in training."""
    if effective_batch_tokens <= 0:
        raise ValueError("effective_batch_tokens must be positive")
    target_device = torch.device(device)
    sae.to(target_device)
    sae.train()

    def run_effective_batch() -> bool:
        accumulated_tokens = 0
        losses_finite = True
        optimizer.zero_grad(set_to_none=True)
        for activation_batch in batch_factory():
            x = _validate_activation_batch(activation_batch, sae.d_in).to(
                device=target_device,
                dtype=sae.decoder.weight.dtype,
            )
            result = accumulate_sae_microbatch(
                sae,
                x,
                torch.zeros(
                    sae.d_sae,
                    dtype=torch.bool,
                    device=target_device,
                ),
            )
            accumulated_tokens += result.token_count
            losses_finite = losses_finite and result.loss_finite
            if accumulated_tokens > effective_batch_tokens:
                raise ValueError(
                    "pilot microbatches exceed effective_batch_tokens"
                )
        if accumulated_tokens != effective_batch_tokens:
            raise ValueError(
                "pilot microbatches do not fill effective_batch_tokens"
            )
        gradients_finite = complete_sae_optimizer_step(
            sae,
            optimizer,
            accumulated_tokens=accumulated_tokens,
            learning_rate=learning_rate,
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in sae.parameters()
        )
        return losses_finite and gradients_finite and parameters_finite

    warmup_finite = run_effective_batch()
    synchronize()
    optimizer_state_allocated = bool(optimizer.state)
    reset_peak_memory()
    started = clock()
    measured_finite = run_effective_batch()
    synchronize()
    elapsed = clock() - started
    if elapsed <= 0:
        raise ValueError("pilot elapsed time must be positive")
    return SaePilotMeasurement(
        elapsed_seconds=elapsed,
        tokens_per_second=effective_batch_tokens / elapsed,
        finite=(
            warmup_finite
            and measured_finite
            and optimizer_state_allocated
        ),
        optimizer_state_allocated=optimizer_state_allocated,
    )


def train_sae_stream(
    sae: TopKSparseAutoencoder,
    batches: Iterable[ActivationBatch],
    config: SparseTrainConfig,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> SparseTrainingCheckpoint:
    """Train in stable stream order and return the final atomic checkpoint."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    sae.to(device)
    sae.train()
    optimizer = create_sae_optimizer(sae)
    tokens_seen = 0
    batch_index = 0
    resume_batch_index = 0
    resume_tokens_seen = 0
    last_activation_token = torch.zeros(
        sae.d_sae,
        dtype=torch.long,
        device=device,
    )

    if resume_from is not None:
        checkpoint = load_training_checkpoint(
            resume_from,
            config.identity,
        )
        if checkpoint.activation_scale != config.activation_scale:
            raise ValueError("checkpoint activation_scale mismatch")
        if checkpoint.schedule != _schedule_contract(config):
            raise ValueError("checkpoint optimizer schedule mismatch")
        if checkpoint.tokens_seen < 0 or checkpoint.batch_index < 0:
            raise ValueError("checkpoint progress must be nonnegative")
        if checkpoint.last_activation_token.shape != (sae.d_sae,):
            raise ValueError("checkpoint last_activation_token shape mismatch")
        sae.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        _move_optimizer_state_(optimizer, device)
        tokens_seen = checkpoint.tokens_seen
        batch_index = checkpoint.batch_index
        resume_batch_index = checkpoint.batch_index
        resume_tokens_seen = checkpoint.tokens_seen
        last_activation_token = checkpoint.last_activation_token.to(
            device=device,
            dtype=torch.long,
        )
        _restore_rng_state(checkpoint)

    next_checkpoint_tokens = (
        (tokens_seen // config.checkpoint_every_tokens) + 1
    ) * config.checkpoint_every_tokens
    last_checkpoint_path: Path | None = None
    skipped_batches = 0
    skipped_tokens = 0
    accumulated_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    for stream_index, activation_batch in enumerate(batches):
        x = _validate_activation_batch(activation_batch, sae.d_in)
        if stream_index < resume_batch_index:
            skipped_batches += 1
            skipped_tokens += int(x.shape[0])
            if (
                skipped_batches == resume_batch_index
                and skipped_tokens != resume_tokens_seen
            ):
                raise ValueError(
                    "resume prefix tokens_seen mismatch: "
                    f"{skipped_tokens} != {resume_tokens_seen}"
                )
            continue
        x = x.to(device=device, dtype=sae.decoder.weight.dtype)
        dead_mask = (
            tokens_seen - last_activation_token
            >= config.dead_feature_window_tokens
        )
        microbatch = accumulate_sae_microbatch(sae, x, dead_mask)
        selected_indices = microbatch.selected_indices
        batch_tokens = microbatch.token_count
        accumulated_tokens += batch_tokens
        if accumulated_tokens > config.effective_batch_tokens:
            raise ValueError(
                "activation microbatches must partition effective_batch_tokens"
            )

        batch_start_token = tokens_seen
        tokens_seen += batch_tokens
        batch_index = stream_index + 1
        token_positions = torch.arange(
            batch_start_token + 1,
            tokens_seen + 1,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1).expand_as(selected_indices)
        last_activation_token.scatter_reduce_(
            0,
            selected_indices.reshape(-1),
            token_positions.reshape(-1),
            reduce="amax",
            include_self=True,
        )
        step_complete = accumulated_tokens == config.effective_batch_tokens
        final_stream_token = tokens_seen >= config.total_fit_tokens
        if step_complete or final_stream_token:
            learning_rate = learning_rate_for_tokens(config, tokens_seen)
            complete_sae_optimizer_step(
                sae,
                optimizer,
                accumulated_tokens=accumulated_tokens,
                learning_rate=learning_rate,
            )
            accumulated_tokens = 0
        if (
            tokens_seen >= next_checkpoint_tokens
            and accumulated_tokens == 0
        ):
            last_checkpoint_path = _write_checkpoint(
                sae,
                optimizer,
                config,
                Path(checkpoint_dir),
                tokens_seen=tokens_seen,
                batch_index=batch_index,
                last_activation_token=last_activation_token,
            )
            if on_checkpoint is not None:
                on_checkpoint(last_checkpoint_path)
            while tokens_seen >= next_checkpoint_tokens:
                next_checkpoint_tokens += config.checkpoint_every_tokens

    if accumulated_tokens:
        learning_rate = learning_rate_for_tokens(config, tokens_seen)
        complete_sae_optimizer_step(
            sae,
            optimizer,
            accumulated_tokens=accumulated_tokens,
            learning_rate=learning_rate,
        )
        accumulated_tokens = 0

    if skipped_batches != resume_batch_index:
        raise ValueError(
            "truncated resume prefix: "
            f"{skipped_batches} batches != {resume_batch_index}"
        )
    if skipped_tokens != resume_tokens_seen:
        raise ValueError(
            "resume prefix tokens_seen mismatch: "
            f"{skipped_tokens} != {resume_tokens_seen}"
        )
    final_path = _write_checkpoint(
        sae,
        optimizer,
        config,
        Path(checkpoint_dir),
        tokens_seen=tokens_seen,
        batch_index=batch_index,
        last_activation_token=last_activation_token,
    )
    if on_checkpoint is not None:
        on_checkpoint(final_path)
    if last_checkpoint_path is not None and final_path == last_checkpoint_path:
        final_path = last_checkpoint_path
    return load_training_checkpoint(final_path, config.identity)
