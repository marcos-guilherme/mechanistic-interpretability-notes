from __future__ import annotations

from typing import Literal

import torch


Reduction = Literal["mean", "none"]


def standard_error(values: torch.Tensor) -> torch.Tensor:
    """Estimate the standard error of a mean from per-example values."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if values.numel() < 2:
        raise ValueError("at least two observations are required")
    return values.float().std(unbiased=True) / values.numel() ** 0.5


def logits_at_end(
    logits: torch.Tensor,
    end_positions: torch.Tensor,
) -> torch.Tensor:
    """Select each example's logits at its semantic END position."""
    if logits.ndim != 3:
        raise ValueError("logits must have three dimensions: batch, position, vocab")
    if end_positions.ndim != 1 or len(end_positions) != logits.shape[0]:
        raise ValueError("end_positions must have one entry per batch example")
    if torch.any(end_positions < 0) or torch.any(
        end_positions >= logits.shape[1]
    ):
        raise ValueError("end_positions must be within the sequence dimension")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, end_positions.to(logits.device)]


def _validate_token_ids(
    token_ids: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    if token_ids.ndim != 1 or len(token_ids) != batch_size:
        raise ValueError(f"{name} must have one entry per batch example")
    return token_ids


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'mean' or 'none'")


def logit_difference(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute IO logit minus subject logit at each semantic END."""
    end_logits = logits_at_end(logits, end_positions)
    io_ids = _validate_token_ids(
        io_token_ids,
        batch_size=end_logits.shape[0],
        name="io_token_ids",
    ).to(end_logits.device)
    subject_ids = _validate_token_ids(
        s_token_ids,
        batch_size=end_logits.shape[0],
        name="s_token_ids",
    ).to(end_logits.device)
    rows = torch.arange(end_logits.shape[0], device=end_logits.device)
    differences = end_logits[rows, io_ids] - end_logits[rows, subject_ids]
    return _reduce(differences, reduction)


def io_accuracy(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Return the strict proportion (or mask) where IO logit exceeds S."""
    differences = logit_difference(
        logits,
        io_token_ids,
        s_token_ids,
        end_positions,
        reduction="none",
    )
    return _reduce((differences > 0).to(logits.dtype), reduction)


def io_probability(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Return IO probability from the full-vocabulary softmax at END."""
    end_logits = logits_at_end(logits, end_positions)
    io_ids = _validate_token_ids(
        io_token_ids,
        batch_size=end_logits.shape[0],
        name="io_token_ids",
    ).to(end_logits.device)
    probabilities = end_logits.softmax(dim=-1).gather(
        dim=-1,
        index=io_ids[:, None],
    )[:, 0]
    return _reduce(probabilities, reduction)
