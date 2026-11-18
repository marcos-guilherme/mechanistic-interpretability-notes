from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from transformers import Qwen2ForCausalLM

from .configuration_qwen2_refusal_ablated import Qwen2RefusalAblatedConfig


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / (direction.norm() + 1e-8)


def ablate_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    coefficient: float = 1.0,
) -> torch.Tensor:
    unit = _unit(direction).to(hidden)
    projection = torch.einsum("...d,d->...", hidden, unit)
    return hidden - coefficient * projection.unsqueeze(-1) * unit


def residual_intervention_hook(
    direction: torch.Tensor,
    *,
    coefficient: float,
) -> Callable:
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = ablate_direction(hidden, direction, coefficient=coefficient)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    return hook


def residual_intervention_pre_hook(
    direction: torch.Tensor,
    *,
    coefficient: float,
) -> Callable:
    def hook(_module, inputs):
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        changed = ablate_direction(hidden, direction, coefficient=coefficient)
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
    coefficient: float,
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
                coefficient=coefficient,
            ),
        )
        for module in [*attention_modules, *mlp_modules]
    ]
    return pre_hooks, forward_hooks


class Qwen2RefusalAblatedForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen2RefusalAblatedConfig

    def __init__(self, config: Qwen2RefusalAblatedConfig) -> None:
        super().__init__(config)
        self.register_buffer(
            "refusal_direction",
            torch.zeros(config.hidden_size),
            persistent=True,
        )
        self.ablation_pre_hooks: list[tuple[object, Callable]] = []
        self.ablation_forward_hooks: list[tuple[object, Callable]] = []
        self._ablation_hooks_installed = False
        self.install_ablation_hooks()

    def install_ablation_hooks(self) -> None:
        if self._ablation_hooks_installed:
            return
        blocks = list(self.model.layers)
        attentions = [block.self_attn for block in blocks]
        mlps = [block.mlp for block in blocks]
        pre_hooks, forward_hooks = directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            self.refusal_direction,
            coefficient=float(self.config.ablation_coefficient),
        )
        for module, hook in pre_hooks:
            module.register_forward_pre_hook(hook)
        for module, hook in forward_hooks:
            module.register_forward_hook(hook)
        self.ablation_pre_hooks = pre_hooks
        self.ablation_forward_hooks = forward_hooks
        self._ablation_hooks_installed = True
