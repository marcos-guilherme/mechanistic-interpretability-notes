from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch


Mode = Literal["add", "ablate"]


def _unit(direction: torch.Tensor) -> torch.Tensor:
    norm = direction.norm()
    if norm == 0:
        raise ValueError("direction must have non-zero norm")
    return direction / norm


def add_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    return hidden + coefficient * direction.to(hidden)


def ablate_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    unit = _unit(direction).to(hidden)
    projection = torch.einsum("...d,d->...", hidden, unit)
    return hidden - projection.unsqueeze(-1) * unit


def residual_intervention_hook(
    direction: torch.Tensor,
    *,
    mode: Mode,
    coefficient: float = 1.0,
) -> Callable:
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = (
            add_direction(hidden, direction, coefficient=coefficient)
            if mode == "add"
            else ablate_direction(hidden, direction)
        )
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    return hook
