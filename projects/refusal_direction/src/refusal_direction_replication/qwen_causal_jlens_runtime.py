"""Synchronized dual-lens readout during one-pass causal generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .model import format_chat_instruction
from .pipeline import temporary_hooks, transformer_module_lists
from .qwen_causal_jlens import (
    LENS_VERSIONS,
    READOUT_LAYERS,
    READOUT_STEPS,
    compact_contains_text_leak,
)
from .qwen_jlens import rank_one_based
from .qwen_lab_readout import incremental_logits_from_residual
from .qwen_lab_runtime import condition_hook_specs
from .runtime import seed_everything


@dataclass(frozen=True)
class GenerationResult:
    generated_token_ids: list[int]
    response_text: str
    observations: list[dict[str, Any]]
    terminal_reason: str


class DualLensReadout:
    def __init__(self, lenses: Mapping[str, Any], bound_model: Any) -> None:
        if tuple(lenses) != LENS_VERSIONS:
            raise ValueError("lenses must be ordered as v1, v2")
        self._lenses = dict(lenses)
        self._bound_model = bound_model

    def rank_target(
        self,
        residual: torch.Tensor,
        *,
        layer: int,
        target_token_id: int,
    ) -> dict[str, int]:
        return {
            version: rank_one_based(
                incremental_logits_from_residual(
                    lens, self._bound_model, residual, layer=layer
                ),
                target_token_id,
            )
            for version, lens in self._lenses.items()
        }


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError, AttributeError):
        return torch.device("cpu")


def _encode_prompt(
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(input_ids)
    else:
        mask = mask.to(device)
    return input_ids, mask


def _tensor_direction(direction: Any) -> torch.Tensor:
    vector = getattr(direction, "vector", direction)
    if not isinstance(vector, torch.Tensor):
        raise TypeError("direction must be a tensor or expose .vector")
    return vector


def _eos_token_id(model: Any, tokenizer: Any) -> int | None:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is not None:
        return int(token_id)
    config = getattr(model, "config", None)
    config_id = getattr(config, "eos_token_id", None)
    if config_id is None:
        return None
    return int(config_id)


def _decode_response(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _capture_hook(captured: dict[int, torch.Tensor], layer: int):
    def capture_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer] = hidden[:, -1, :].detach().clone()

    return capture_hook


def stream_causal_generation(
    *,
    model: Any,
    tokenizer: Any,
    readout: DualLensReadout,
    layers: Sequence[int] = READOUT_LAYERS,
    readout_steps: int = READOUT_STEPS,
    max_new_tokens: int = 64,
    prompt: str,
    condition: str,
    coefficient: float,
    direction: Any,
    random_control: torch.Tensor,
    dataset: str = "mitre",
    example_id: str = "example",
    seed: int | None = None,
) -> GenerationResult:
    if seed is not None:
        seed_everything(int(seed))
    device = _model_device(model)
    blocks, attentions, mlps = transformer_module_lists(model)
    pre_hooks, forward_hooks = condition_hook_specs(
        condition=str(condition),
        coefficient=float(coefficient),
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=_tensor_direction(direction),
        random_control=random_control,
    )
    requested_layers = tuple(int(layer) for layer in layers)
    captured: dict[int, torch.Tensor] = {}
    capture_hooks = [
        (blocks[layer], _capture_hook(captured, layer))
        for layer in requested_layers
    ]
    forward_hooks = [*forward_hooks, *capture_hooks]
    rendered = format_chat_instruction(tokenizer, prompt)
    input_ids, attention_mask = _encode_prompt(tokenizer, rendered, device)
    eos_id = _eos_token_id(model, tokenizer)
    past = None
    next_ids = input_ids
    cache_position = torch.arange(input_ids.shape[1], device=device)
    generated_ids: list[int] = []
    observations: list[dict[str, Any]] = []
    terminal_reason = "max_tokens"
    max_steps = int(max_new_tokens)
    readout_limit = int(readout_steps)

    try:
        with temporary_hooks(pre_hooks=pre_hooks, forward_hooks=forward_hooks):
            with torch.inference_mode():
                for step in range(1, max_steps + 1):
                    outputs = model(
                        input_ids=next_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    past = getattr(outputs, "past_key_values", None)
                    logits = outputs.logits[:, -1, :].float()
                    token_id = int(torch.argmax(logits.reshape(-1)).item())
                    if step <= readout_limit:
                        for layer in requested_layers:
                            residual = captured[layer].reshape(-1)
                            ranks = readout.rank_target(
                                residual,
                                layer=layer,
                                target_token_id=token_id,
                            )
                            for version, rank in ranks.items():
                                observations.append(
                                    {
                                        "dataset": dataset,
                                        "example_id": example_id,
                                        "condition": condition,
                                        "generation_step": step,
                                        "layer": int(layer),
                                        "lens_version": version,
                                        "rank": int(rank),
                                        "target_token_id": int(token_id),
                                    }
                                )
                    del outputs
                    del logits
                    captured.clear()
                    generated_ids.append(int(token_id))
                    if eos_id is not None and token_id == int(eos_id):
                        terminal_reason = "eos"
                        break
                    if step >= max_steps:
                        terminal_reason = "max_tokens"
                        break
                    next_ids = torch.tensor(
                        [[token_id]],
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    ones = torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask, ones], dim=1)
                    cache_position = torch.tensor(
                        [input_ids.shape[1] + step - 1],
                        device=device,
                    )
    finally:
        captured.clear()
        past = None

    if compact_contains_text_leak(observations):
        raise ValueError("observations contain raw text")
    return GenerationResult(
        generated_token_ids=generated_ids,
        response_text=_decode_response(tokenizer, generated_ids),
        observations=observations,
        terminal_reason=terminal_reason,
    )
