from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Literal

import torch


Mode = Literal["add", "ablate"]


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / (direction.norm() + 1e-8)


def _require_finite_coefficient(coefficient: float) -> None:
    if not math.isfinite(coefficient):
        raise ValueError(f"coefficient must be finite, got {coefficient!r}")


def _require_ablation_coefficient(coefficient: float) -> None:
    _require_finite_coefficient(coefficient)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError(
            f"ablation coefficient must be in [0, 1], got {coefficient!r}"
        )


def add_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    _require_finite_coefficient(coefficient)
    return hidden + coefficient * direction.to(hidden)


def ablate_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    _require_ablation_coefficient(coefficient)
    unit = _unit(direction).to(hidden)
    projection = torch.einsum("...d,d->...", hidden, unit)
    return hidden - coefficient * projection.unsqueeze(-1) * unit


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
            else ablate_direction(hidden, direction, coefficient=coefficient)
        )
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    return hook


def residual_intervention_pre_hook(
    direction: torch.Tensor,
    *,
    mode: Mode,
    coefficient: float = 1.0,
) -> Callable:
    def hook(_module, inputs):
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        changed = (
            add_direction(hidden, direction, coefficient=coefficient)
            if mode == "add"
            else ablate_direction(hidden, direction, coefficient=coefficient)
        )
        if isinstance(inputs, tuple):
            return (changed, *inputs[1:])
        return changed

    return hook


def directional_ablation_hook_specs(
    block_modules: Sequence,
    attention_modules: Sequence,
    mlp_modules: Sequence,
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> tuple[list[tuple[object, Callable]], list[tuple[object, Callable]]]:
    if not (
        len(block_modules) == len(attention_modules) == len(mlp_modules)
    ):
        raise ValueError("block, attention, and MLP module counts must align")
    pre_hooks = [
        (
            module,
            residual_intervention_pre_hook(
                direction,
                mode="ablate",
                coefficient=coefficient,
            ),
        )
        for module in block_modules
    ]
    forward_hooks = [
        (
            module,
            residual_intervention_hook(
                direction,
                mode="ablate",
                coefficient=coefficient,
            ),
        )
        for module in [*attention_modules, *mlp_modules]
    ]
    return pre_hooks, forward_hooks
