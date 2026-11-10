from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GEMMA_CHAT_TEMPLATE = (
    "<start_of_turn>user\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
GEMMA_REFUSAL_TOKEN_IDS = (235285,)


def format_instruction(instruction: str) -> str:
    return GEMMA_CHAT_TEMPLATE.format(instruction=instruction)


def load_gemma(
    model_name: str,
    *,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    return model, tokenizer


def tokenize_instructions(
    tokenizer: Any,
    instructions: list[str],
) -> dict[str, torch.Tensor]:
    return tokenizer(
        [format_instruction(item) for item in instructions],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def end_of_instruction_offsets(tokenizer: Any) -> tuple[int, ...]:
    suffix = GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1]
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    return tuple(range(-len(token_ids), 0))


def resolve_post_instruction_positions(
    attention_mask: torch.Tensor,
    *,
    offsets: tuple[int, ...],
    padding_side: str,
) -> torch.Tensor:
    if padding_side == "left":
        bases = torch.full(
            (attention_mask.shape[0],),
            attention_mask.shape[1],
            device=attention_mask.device,
            dtype=torch.long,
        )
    elif padding_side == "right":
        bases = attention_mask.sum(dim=-1).to(torch.long)
    else:
        raise ValueError("padding_side must be 'left' or 'right'")
    return torch.stack([bases + offset for offset in offsets], dim=-1)


@torch.inference_mode()
def capture_residual_grid(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    block_modules: Iterable[Any],
    position_offsets: tuple[int, ...],
    padding_side: str,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handles = []
    positions = resolve_post_instruction_positions(
        batch["attention_mask"],
        offsets=position_offsets,
        padding_side=padding_side,
    )

    def hook(_module: Any, inputs: Any) -> None:
        hidden = inputs[0]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        captured.append(
            hidden[rows[:, None], positions].detach().float().cpu()
        )

    try:
        for module in block_modules:
            handles.append(module.register_forward_pre_hook(hook))
        model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(captured, dim=1)
