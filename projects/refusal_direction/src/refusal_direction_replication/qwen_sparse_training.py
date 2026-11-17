"""Deterministic streaming training and atomic checkpoints for the TopK SAE."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import torch

from .qwen_sparse_sae import (
    TopKSparseAutoencoder,
    remove_parallel_decoder_grad_,
    sae_loss,
)


CHECKPOINT_SCHEMA_VERSION = 1
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
class SparseTrainConfig:
    """Frozen training identity plus optimizer and checkpoint settings."""

    run_id: str
    implementation_commit: str
    config_sha256: str
    corpus_sha256: str
    model_revision: str
    learning_rate: float
    checkpoint_every_tokens: int
    activation_scale: float
    dead_feature_window_tokens: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        for field, value in self.identity.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
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
    if set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("checkpoint fields do not match schema")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
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


def train_sae_stream(
    sae: TopKSparseAutoencoder,
    batches: Iterable[ActivationBatch],
    config: SparseTrainConfig,
    checkpoint_dir: Path,
    resume_from: Path | None = None,
) -> SparseTrainingCheckpoint:
    """Train in stable stream order and return the final atomic checkpoint."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    sae.to(device)
    sae.train()
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=config.learning_rate,
    )
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
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = sae_loss(sae, x, dead_mask)
        selected_indices = metrics["selected_indices"]
        loss.backward()
        remove_parallel_decoder_grad_(sae)
        optimizer.step()
        sae.renormalize_decoder_()

        batch_start_token = tokens_seen
        tokens_seen += int(x.shape[0])
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
        if tokens_seen >= next_checkpoint_tokens:
            last_checkpoint_path = _write_checkpoint(
                sae,
                optimizer,
                config,
                Path(checkpoint_dir),
                tokens_seen=tokens_seen,
                batch_index=batch_index,
                last_activation_token=last_activation_token,
            )
            while tokens_seen >= next_checkpoint_tokens:
                next_checkpoint_tokens += config.checkpoint_every_tokens

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
    if last_checkpoint_path is not None and final_path == last_checkpoint_path:
        final_path = last_checkpoint_path
    return load_training_checkpoint(final_path, config.identity)
