from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from torch import nn

from refusal_direction_replication.pipeline import random_direction_control
from refusal_direction_replication.qwen_causal_jlens import READOUT_LAYERS
from refusal_direction_replication.qwen_causal_jlens_runtime import (
    DualLensReadout,
    stream_causal_generation,
)
from refusal_direction_replication.qwen_jlens import rank_one_based


READOUT_D_MODEL = 8
READOUT_VOCAB = 32
RANDOM_CONTROL_SEED = 1042


class CausalTokenizer:
    def __init__(self, *, vocab_size: int = READOUT_VOCAB) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.add_bos_token = False

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        content = messages[0]["content"]
        text = f"<user>{content}</user>"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = [2 + (ord(character) % 20) for character in text]
        return ids or [2]

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(97 + (int(token) % 26)) for token in ids)

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        if isinstance(text, list):
            text = text[0]
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class ThinBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden: torch.Tensor, *args, **kwargs):
        del args, kwargs
        return hidden


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class TinyLensModel(nn.Module):
    def __init__(
        self,
        *,
        n_layers: int = 4,
        d_model: int = 8,
        vocab_size: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [ResidualBlock(d_model) for _ in range(n_layers)]
        )
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        ids = [1 + (ord(character) % 30) for character in text][:max_length]
        return torch.tensor([ids or [1]], dtype=torch.long)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual.float()))


class GreedyFixture(nn.Module):
    def __init__(
        self,
        *,
        tokens: int = 10,
        n_layers: int = 27,
        d_model: int = READOUT_D_MODEL,
        vocab_size: int = READOUT_VOCAB,
        terminal: str = "eos",
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ThinBlock() for _ in range(n_layers)])
        self.config = SimpleNamespace(eos_token_id=0)
        self.vocab_size = vocab_size
        planned = [2 + index for index in range(max(tokens - 1, 0))]
        planned.append(0)
        self._planned = planned[:tokens]
        self._emitted = 0
        self._terminal = terminal
        self.prompt_length: int | None = None
        self.past_seen: list[object] = []
        self.use_cache_flags: list[bool] = []
        self.input_widths: list[int] = []
        self.active_hook_count = 0
        self._patch_hook_counters()

    def _patch_hook_counters(self) -> None:
        for module in self.modules():
            original_forward = module.register_forward_hook
            original_pre = module.register_forward_pre_hook

            def register_forward_hook(
                hook,
                *args,
                _original=original_forward,
                **kwargs,
            ):
                handle = _original(hook, *args, **kwargs)
                return _CountingHandle(self, handle)

            def register_forward_pre_hook(
                hook,
                *args,
                _original=original_pre,
                **kwargs,
            ):
                handle = _original(hook, *args, **kwargs)
                return _CountingHandle(self, handle)

            module.register_forward_hook = register_forward_hook  # type: ignore[method-assign]
            module.register_forward_pre_hook = register_forward_pre_hook  # type: ignore[method-assign]

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if self._terminal == "error":
            raise RuntimeError("causal readout failed")
        if self.prompt_length is None:
            self.prompt_length = int(input_ids.shape[1])
        self.past_seen.append(past_key_values)
        self.use_cache_flags.append(bool(use_cache))
        self.input_widths.append(int(input_ids.shape[1]))
        hidden = self.embed(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            dtype=torch.float32,
        )
        index = min(self._emitted, len(self._planned) - 1)
        logits[:, -1, int(self._planned[index])] = 10.0
        self._emitted += 1
        cache = past_key_values
        if use_cache:
            previous = 0 if past_key_values is None else int(past_key_values)
            cache = previous + int(input_ids.shape[1])
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _CountingHandle:
    def __init__(self, owner: GreedyFixture, inner: Any) -> None:
        self._owner = owner
        self._inner = inner
        self._removed = False
        owner.active_hook_count += 1

    def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        self._owner.active_hook_count -= 1
        self._inner.remove()


def _lens(
    offset: float = 0.0,
    *,
    d_model: int = 2,
    layers: tuple[int, ...] = (25,),
) -> JacobianLens:
    jacobian = torch.eye(d_model)
    jacobian[0, 0] = 1.0 + 2.0 * float(offset)
    return JacobianLens(
        {int(layer): jacobian.clone() for layer in layers},
        n_prompts=1,
        d_model=d_model,
    )


def _bound_model(*, d_model: int = 2, vocab_size: int = 5) -> SimpleNamespace:
    weight = torch.zeros(vocab_size, d_model)
    weight[0, 0] = 1.0
    weight[3, 1] = 1.0

    def unembed(residual: torch.Tensor) -> torch.Tensor:
        return residual.float().reshape(-1, d_model) @ weight.T

    return SimpleNamespace(unembed=unembed)


def _tokenizer() -> CausalTokenizer:
    return CausalTokenizer()


def _dual_readout(
    *,
    d_model: int = READOUT_D_MODEL,
    layers: tuple[int, ...] = READOUT_LAYERS,
) -> DualLensReadout:
    weight = torch.zeros(READOUT_VOCAB, d_model)
    weight[: min(READOUT_VOCAB, d_model), :d_model] = torch.eye(
        min(READOUT_VOCAB, d_model)
    )
    bound = SimpleNamespace(
        unembed=lambda residual: residual.float().reshape(-1, d_model) @ weight.T
    )
    return DualLensReadout(
        lenses={
            "v1": _lens(0.0, d_model=d_model, layers=layers),
            "v2": _lens(1.0, d_model=d_model, layers=layers),
        },
        bound_model=bound,
    )


def _intervention_inputs(condition: str) -> dict[str, Any]:
    direction = torch.nn.functional.normalize(
        torch.arange(READOUT_D_MODEL, dtype=torch.float32),
        dim=0,
    )
    coefficient = 0.0 if condition == "baseline" else 0.2
    return {
        "prompt": "hello",
        "condition": condition,
        "coefficient": coefficient,
        "direction": direction,
        "random_control": random_direction_control(
            direction,
            seed=RANDOM_CONTROL_SEED,
        ),
        "dataset": "mitre",
        "example_id": "example-1",
    }


def _greedy_fixture(tokens: int = 10, **kwargs) -> GreedyFixture:
    return GreedyFixture(tokens=tokens, **kwargs)


def _hook_counting_model(*, terminal: str) -> GreedyFixture:
    tokens = 1 if terminal == "eos" else 2
    return GreedyFixture(tokens=tokens, terminal=terminal)


def _fixture_args() -> dict[str, Any]:
    return {
        "tokenizer": _tokenizer(),
        "readout": _dual_readout(),
        "layers": READOUT_LAYERS,
        "readout_steps": 8,
        "max_new_tokens": 64,
        **_intervention_inputs("baseline"),
    }


def test_dual_readout_ranks_the_same_target_from_the_same_residual() -> None:
    readout = DualLensReadout(
        lenses={"v1": _lens(offset=0.0), "v2": _lens(offset=1.0)},
        bound_model=_bound_model(),
    )
    ranks = readout.rank_target(
        torch.tensor([1.0, 2.0]),
        layer=25,
        target_token_id=3,
    )
    assert ranks == {"v1": 1, "v2": 2}


def test_stream_records_four_layers_two_lenses_for_first_eight_steps() -> None:
    result = stream_causal_generation(
        model=_greedy_fixture(tokens=10),
        tokenizer=_tokenizer(),
        readout=_dual_readout(),
        layers=(19, 22, 25, 26),
        readout_steps=8,
        max_new_tokens=64,
        **_intervention_inputs("baseline"),
    )
    assert len(result.observations) == 8 * 4 * 2
    assert {row["target_token_id"] for row in result.observations} == set(
        result.generated_token_ids[:8]
    )
    assert len(result.generated_token_ids) == 10
    assert {row["layer"] for row in result.observations} == {19, 22, 25, 26}
    assert {row["lens_version"] for row in result.observations} == {"v1", "v2"}
    assert {row["generation_step"] for row in result.observations} == set(range(1, 9))
    assert result.terminal_reason == "eos"
    assert result.response_text


@pytest.mark.parametrize("terminal", ["eos", "error"])
def test_stream_always_removes_hooks(terminal: str) -> None:
    model = _hook_counting_model(terminal=terminal)
    with contextlib.suppress(RuntimeError):
        stream_causal_generation(model=model, **_fixture_args())
    assert model.active_hook_count == 0


def test_rank_target_matches_canonical_apply_for_both_lenses() -> None:
    bound = TinyLensModel(n_layers=4, d_model=8, vocab_size=32, seed=1)
    layer = 2
    lenses = {
        "v1": JacobianLens(
            {index: torch.eye(8) for index in range(3)},
            n_prompts=1,
            d_model=8,
        ),
        "v2": JacobianLens(
            {
                index: torch.eye(8) + 0.25 * torch.ones(8, 8)
                for index in range(3)
            },
            n_prompts=1,
            d_model=8,
        ),
    }
    readout = DualLensReadout(lenses=lenses, bound_model=bound)
    prompt = "ab"
    canonical: dict[str, torch.Tensor] = {}
    model_logits = None
    for version, lens in lenses.items():
        lens_logits, model_logits, _ids = lens.apply(
            bound,
            prompt,
            layers=[layer],
            positions=[-1],
            max_seq_len=128,
        )
        canonical[version] = lens_logits[layer][0]
    assert model_logits is not None
    target_token_id = int(model_logits[0].argmax().item())
    encoded = bound.encode(prompt, max_length=128)
    with ActivationRecorder(bound.layers, at=[layer]) as recorder:
        bound.forward(encoded)
        residual = recorder.activations[layer][0, -1].detach()
    ranks = readout.rank_target(
        residual,
        layer=layer,
        target_token_id=target_token_id,
    )
    assert ranks == {
        version: rank_one_based(logits, target_token_id)
        for version, logits in canonical.items()
    }


def test_stream_threads_kv_cache_and_stops_on_eos() -> None:
    model = _greedy_fixture(tokens=3)
    result = stream_causal_generation(
        model=model,
        tokenizer=_tokenizer(),
        readout=_dual_readout(),
        layers=(19, 22, 25, 26),
        readout_steps=8,
        max_new_tokens=64,
        **_intervention_inputs("baseline"),
    )
    assert result.terminal_reason == "eos"
    assert len(result.generated_token_ids) == 3
    assert result.generated_token_ids[-1] == 0
    assert len(result.observations) == 3 * 4 * 2
    assert model.use_cache_flags == [True, True, True]
    assert model.past_seen[0] is None
    assert model.past_seen[1] == model.prompt_length
    assert model.past_seen[2] == model.prompt_length + 1
    assert model.input_widths[0] == model.prompt_length
    assert model.input_widths[1:] == [1, 1]
