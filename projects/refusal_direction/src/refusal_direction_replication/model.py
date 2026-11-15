from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FORMAT_STYLE_GEMMA = "gemma"
FORMAT_STYLE_CHAT_TEMPLATE = "chat_template"

GEMMA_CHAT_TEMPLATE = (
    "<start_of_turn>user\n{instruction}<end_of_turn>\n"
    "<start_of_turn>model\n"
)
GEMMA_REFUSAL_TOKEN_IDS = (235285,)


def format_instruction(instruction: str) -> str:
    return GEMMA_CHAT_TEMPLATE.format(instruction=instruction)


def _ensure_pad_token(tokenizer: Any) -> bool:
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return False
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return True


def format_chat_instruction(tokenizer: Any, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_causal_lm(
    model_name: str,
    *,
    dtype: torch.dtype,
    revision: str | None = None,
) -> tuple[Any, Any]:
    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "cuda",
    }
    if revision is not None:
        tokenizer_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    added_pad_token = _ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).eval()
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.requires_grad_(False)
    return model, tokenizer


def load_gemma(
    model_name: str,
    *,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    return load_causal_lm(model_name, dtype=dtype)


def tokenize_instructions(
    tokenizer: Any,
    instructions: list[str],
    *,
    format_style: str = FORMAT_STYLE_GEMMA,
) -> dict[str, torch.Tensor]:
    if format_style == FORMAT_STYLE_GEMMA:
        texts = [format_instruction(item) for item in instructions]
    elif format_style == FORMAT_STYLE_CHAT_TEMPLATE:
        texts = [
            format_chat_instruction(tokenizer, item) for item in instructions
        ]
    else:
        raise ValueError(f"unsupported format_style: {format_style}")
    return tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def resolve_refusal_token_ids(
    tokenizer: Any,
    prefixes: Sequence[str],
) -> tuple[int, ...]:
    resolved: list[int] = []
    seen: set[int] = set()
    for prefix in prefixes:
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"prefix {prefix!r} did not tokenize")
        first = token_ids[0]
        if first not in seen:
            seen.add(first)
            resolved.append(first)
    return tuple(resolved)


def end_of_instruction_offsets(
    tokenizer: Any,
    *,
    format_style: str = FORMAT_STYLE_GEMMA,
) -> tuple[int, ...]:
    if format_style == FORMAT_STYLE_CHAT_TEMPLATE:
        messages = [{"role": "user", "content": "probe"}]
        with_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        without_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        suffix_len = len(with_prompt) - len(without_prompt)
        if suffix_len <= 0:
            return (-1,)
        return tuple(range(-suffix_len, 0))
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
