from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from torch import nn

from refusal_direction_replication.artifacts import atomic_write_json
from refusal_direction_replication.data import prompt_content_hash
from refusal_direction_replication.pipeline import random_direction_control
from refusal_direction_replication.qwen_causal_jlens import (
    CAUSAL_JLENS_PROFILE,
    CONDITIONS,
    DATASETS,
    READOUT_LAYERS,
    compact_contains_text_leak,
)
from refusal_direction_replication.qwen_causal_jlens_runtime import (
    DualLensReadout,
    stream_causal_generation,
)
from refusal_direction_replication.qwen_jlens import rank_one_based
from refusal_direction_replication.run_progress import RunProgressStore


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
    def __init__(self, *, layer_index: int = 0, tuple_output: bool = False) -> None:
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()
        self.layer_index = int(layer_index)
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor, *args, **kwargs):
        del args, kwargs
        stamped = hidden.clone()
        stamped[:, -1, 1] = float(self.layer_index)
        if self.tuple_output:
            present = {"layer": self.layer_index}
            return (stamped, present, {"aux": True})
        return stamped


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
        tuple_output: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                ThinBlock(layer_index=index, tuple_output=tuple_output)
                for index in range(n_layers)
            ]
        )
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
        self.attention_masks: list[torch.Tensor | None] = []
        self.cache_positions: list[torch.Tensor | None] = []
        self.forward_ids: list[int] = []
        self._forward_id = 0
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
        cache_position=None,
        **kwargs,
    ):
        del kwargs
        if self._terminal == "error":
            raise RuntimeError("causal readout failed")
        if past_key_values is None:
            self._emitted = 0
            self.prompt_length = int(input_ids.shape[1])
        elif self.prompt_length is None:
            self.prompt_length = int(input_ids.shape[1])
        self.past_seen.append(past_key_values)
        self.use_cache_flags.append(bool(use_cache))
        self.input_widths.append(int(input_ids.shape[1]))
        self.attention_masks.append(
            None if attention_mask is None else attention_mask.detach().clone()
        )
        self.cache_positions.append(
            None if cache_position is None else cache_position.detach().clone()
        )
        self._forward_id += 1
        self.forward_ids.append(self._forward_id)
        hidden = self.embed(input_ids).clone()
        hidden[:, -1, 0] = float(self._forward_id)
        for block in self.model.layers:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
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


def _assert_each_observation_targets_its_generated_token(result) -> None:
    generated = result.generated_token_ids
    for row in result.observations:
        step = int(row["generation_step"])
        assert 1 <= step <= len(generated)
        assert int(row["target_token_id"]) == int(generated[step - 1])


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
    assert len(result.generated_token_ids) == 10
    _assert_each_observation_targets_its_generated_token(result)
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
    _assert_each_observation_targets_its_generated_token(result)
    assert model.use_cache_flags == [True, True, True]
    assert model.past_seen[0] is None
    assert model.past_seen[1] == model.prompt_length
    assert model.past_seen[2] == model.prompt_length + 1
    assert model.input_widths[0] == model.prompt_length
    assert model.input_widths[1:] == [1, 1]
    _assert_exact_mask_and_cache_position(model)


def test_tuple_block_outputs_are_captured_without_replacing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_causal_jlens_runtime as module

    model = _greedy_fixture(tokens=2, tuple_output=True)
    layers = (19, 22, 25, 26)
    replacements: list[object] = []
    original_outputs: list[object] = []
    extracted: list[torch.Tensor] = []
    ranked: list[torch.Tensor] = []
    originals = {
        layer: model.model.layers[layer].register_forward_hook
        for layer in layers
    }
    original_readout = module.incremental_logits_from_residual

    def spy(lens, bound_model, residual, *, layer):
        ranked.append(residual.detach().float().reshape(-1).cpu().clone())
        return original_readout(lens, bound_model, residual, layer=layer)

    def bind(layer: int):
        original = originals[layer]

        def tracking_register(hook, *args, **kwargs):
            def wrapped(module, inputs, output):
                original_outputs.append(output)
                hidden = output[0]
                extracted.append(
                    hidden[:, -1, :].detach().float().reshape(-1).cpu().clone()
                )
                result = hook(module, inputs, output)
                replacements.append(result)
                return result

            return original(wrapped, *args, **kwargs)

        return tracking_register

    for layer in layers:
        model.model.layers[layer].register_forward_hook = bind(layer)  # type: ignore[method-assign]

    monkeypatch.setattr(module, "incremental_logits_from_residual", spy)
    result = stream_causal_generation(
        model=model,
        tokenizer=_tokenizer(),
        readout=_dual_readout(),
        layers=layers,
        readout_steps=8,
        max_new_tokens=64,
        **_intervention_inputs("baseline"),
    )
    assert result.generated_token_ids
    assert replacements
    assert all(item is None for item in replacements)
    assert original_outputs
    assert all(
        isinstance(item, tuple) and len(item) >= 2 for item in original_outputs
    )
    assert all(torch.is_tensor(item[0]) for item in original_outputs)
    assert extracted
    assert len(ranked) == len(extracted) * 2
    for index, residual in enumerate(extracted):
        pair = ranked[index * 2 : index * 2 + 2]
        assert torch.equal(pair[0], residual)
        assert torch.equal(pair[1], residual)
    assert model.past_seen[0] is None
    assert model.past_seen[1] == model.prompt_length


def test_dual_lenses_see_the_same_captured_residual_from_one_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_causal_jlens_runtime as module

    model = _greedy_fixture(tokens=3)
    readout = _dual_readout()
    layers = (19, 22, 25, 26)
    seen: list[dict[str, Any]] = []
    original = module.incremental_logits_from_residual

    def spy(lens, bound_model, residual, *, layer):
        version = next(
            name for name, item in readout._lenses.items() if item is lens
        )
        vector = residual.detach().float().reshape(-1).cpu().clone()
        seen.append(
            {
                "version": version,
                "layer": int(layer),
                "residual": vector,
                "forward_id": int(round(float(vector[0].item()))),
            }
        )
        return original(lens, bound_model, residual, layer=layer)

    monkeypatch.setattr(module, "incremental_logits_from_residual", spy)
    result = stream_causal_generation(
        model=model,
        tokenizer=_tokenizer(),
        readout=readout,
        layers=layers,
        readout_steps=8,
        max_new_tokens=64,
        **_intervention_inputs("baseline"),
    )
    readout_steps = min(8, len(result.generated_token_ids))
    assert len(seen) == readout_steps * len(layers) * 2
    _assert_each_observation_targets_its_generated_token(result)
    for step in range(1, readout_steps + 1):
        expected_forward = model.forward_ids[step - 1]
        for layer in layers:
            pair = [
                item
                for item in seen
                if item["forward_id"] == expected_forward and item["layer"] == layer
            ]
            assert {item["version"] for item in pair} == {"v1", "v2"}
            assert torch.equal(pair[0]["residual"], pair[1]["residual"])
            assert int(round(float(pair[0]["residual"][1].item()))) == layer


def _assert_exact_mask_and_cache_position(model: GreedyFixture) -> None:
    prompt_len = int(model.prompt_length)
    assert len(model.attention_masks) == len(model.cache_positions) == 3
    first_mask = model.attention_masks[0]
    first_position = model.cache_positions[0]
    assert first_mask is not None and first_position is not None
    assert tuple(first_mask.shape) == (1, prompt_len)
    assert torch.equal(first_mask, torch.ones_like(first_mask))
    assert torch.equal(
        first_position.cpu(),
        torch.arange(prompt_len, dtype=first_position.dtype),
    )
    for index, (mask, position) in enumerate(
        zip(model.attention_masks[1:], model.cache_positions[1:], strict=True),
        start=1,
    ):
        assert mask is not None and position is not None
        expected_width = prompt_len + index
        assert tuple(mask.shape) == (1, expected_width)
        assert torch.equal(mask, torch.ones_like(mask))
        assert tuple(position.shape) == (1,)
        assert int(position.item()) == prompt_len + index - 1


def _config(root: Path, *, dataset_size: int = 32) -> dict[str, Any]:
    del root
    return {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "dtype": "bfloat16",
        "batch_size": 1,
        "format_style": "chat_template",
        "seed": 42,
        "max_new_tokens": 64,
        "readout_steps": 8,
        "readout_layers": [19, 22, 25, 26],
        "datasets": ["mitre", "mitre_frr"],
        "dataset_size": int(dataset_size),
        "conditions": ["baseline", "ablation", "addition", "random_control"],
        "ablation_coefficient": 0.2,
        "addition_coefficient": 0.5,
        "random_control_coefficient": 0.2,
        "random_control_seed": 1042,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 2028,
        "equivalence_margin": 0.05,
        "primary_layer": 25,
        "primary_k": 5,
        "purplellama_repository": "meta-llama/PurpleLlama",
        "purplellama_revision": "0654501399e7cbd560db9501965dc09c567a5563",
        "mitre_relative_path": (
            "CybersecurityBenchmarks/datasets/mitre/"
            "mitre_benchmark_100_per_category_with_augmentation.json"
        ),
        "mitre_frr_relative_path": (
            "CybersecurityBenchmarks/datasets/mitre_frr/mitre_frr.json"
        ),
        "jlens_upstream": "anthropics/jacobian-lens",
        "jlens_upstream_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "source_prepare": "results/qwen25_coder_7b_code_eval/00_prepare.json",
        "source_direction": "artifacts/qwen25_coder_7b/selected_direction.pt",
        "source_dose_evaluation": (
            "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
        ),
        "source_v1_lens": (
            "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
        ),
        "source_v2_lens": (
            "artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt"
        ),
        "results_dir": "results/qwen25_coder_7b_causal_jlens",
        "artifacts_dir": "artifacts/qwen25_coder_7b_causal_jlens",
    }


def _seed_sources(root: Path) -> None:
    config = _config(root)
    direction = root / str(config["source_direction"])
    v1_lens = root / str(config["source_v1_lens"])
    v2_lens = root / str(config["source_v2_lens"])
    prepare = root / str(config["source_prepare"])
    dose = root / str(config["source_dose_evaluation"])
    for path in (direction, v1_lens, v2_lens, prepare, dose):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer": 18,
            "position_offset": -1,
            "vector": torch.ones(READOUT_D_MODEL),
        },
        direction,
    )
    v1_lens.write_bytes(b"v1-lens")
    v2_lens.write_bytes(b"v2-lens")
    atomic_write_json(
        prepare,
        {
            "results": {
                "mitre": {
                    "example_ids": ["id-1", "id-2"],
                    "content_hashes": ["h1", "h2"],
                    "source_indices": [0, 1],
                },
                "mitre_frr": {
                    "example_ids": ["id-3", "id-4"],
                    "content_hashes": ["h3", "h4"],
                    "source_indices": [0, 1],
                },
            }
        },
    )
    atomic_write_json(
        dose,
        {
            "status": "pass",
            "run_id": "dose-run",
            "selected_pair": {"ablation": 0.2, "addition": 0.5},
        },
    )


def _rewrite_dose(root: Path, *, addition: float) -> None:
    path = root / "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_pair"]["addition"] = float(addition)
    atomic_write_json(path, payload)


def _identity(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_commit": "a" * 40,
        "config_hash": "b" * 64,
        "source_artifact_hashes": {
            "prepare": "c" * 64,
            "direction": "d" * 64,
            "dose_evaluation": "e" * 64,
            "v1_lens": "f" * 64,
            "v2_lens": "1" * 64,
        },
        "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "dataset_ids": {
            "mitre": ["id-1", "id-2"],
            "mitre_frr": ["id-3", "id-4"],
        },
        "dataset_hashes": {
            "mitre": ["h1", "h2"],
            "mitre_frr": ["h3", "h4"],
        },
    }


def _matrix() -> list[tuple[str, str, str]]:
    return [
        ("mitre", "id-1", "baseline"),
        ("mitre", "id-1", "ablation"),
        ("mitre", "id-2", "baseline"),
        ("mitre_frr", "id-3", "baseline"),
    ]


def test_source_context_requires_exact_frozen_inputs(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import source_context

    _seed_sources(tmp_path)
    context = source_context(tmp_path, _config(tmp_path))
    assert context["direction_layer"] == 18
    assert context["direction_offset"] == -1
    assert set(context["source_artifact_hashes"]) == {
        "prepare",
        "direction",
        "dose_evaluation",
        "v1_lens",
        "v2_lens",
    }
    _rewrite_dose(tmp_path, addition=0.4)
    with pytest.raises(ValueError, match="coefficient"):
        source_context(tmp_path, _config(tmp_path))


def test_resume_skips_only_complete_matching_pairs(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import PanelCheckpoint

    checkpoint = PanelCheckpoint.create(_identity("run-a"))
    checkpoint.mark_complete("mitre", "id-1", "baseline")
    checkpoint.write(tmp_path / "checkpoint.json")
    loaded = PanelCheckpoint.load(
        tmp_path / "checkpoint.json", expected_identity=_identity("run-a")
    )
    assert loaded.pending(_matrix()) == _matrix()[1:]
    with pytest.raises(ValueError, match="identity"):
        PanelCheckpoint.load(
            tmp_path / "checkpoint.json", expected_identity=_identity("run-b")
        )


def _record(dataset: str, index: int, text: str, category: str) -> dict[str, Any]:
    digest = prompt_content_hash(text)
    return {
        "dataset": dataset,
        "source_index": index,
        "text": text,
        "category": category,
        "content_hash": digest,
        "example_id": f"{dataset}:{index}:{digest[:16]}",
    }


def _write_prepare_source(root: Path, records: dict[str, list[dict[str, Any]]]) -> None:
    config = _config(root)
    prepare = root / str(config["source_prepare"])
    prepare.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": {}}
    for dataset, items in records.items():
        payload["results"][dataset] = {
            "dataset": dataset,
            "example_ids": [item["example_id"] for item in items],
            "content_hashes": [item["content_hash"] for item in items],
            "source_indices": [item["source_index"] for item in items],
        }
    atomic_write_json(prepare, payload)


def _write_purplellama(root: Path, records: dict[str, list[dict[str, Any]]]) -> Path:
    config = _config(root)
    source_root = root / "sources" / "PurpleLlama"
    specifications = {
        "mitre": (config["mitre_relative_path"], "mitre_category"),
        "mitre_frr": (config["mitre_frr_relative_path"], "attack_type"),
    }
    for dataset, items in records.items():
        relative, category_field = specifications[dataset]
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        max_index = max(item["source_index"] for item in items)
        rows: list[dict[str, Any]] = [
            {"mutated_prompt": f"pad-{dataset}-{index}", category_field: "pad"}
            for index in range(max_index + 1)
        ]
        for item in items:
            rows[item["source_index"]] = {
                "mutated_prompt": item["text"],
                category_field: item["category"],
            }
        path.write_text(json.dumps(rows), encoding="utf-8")
    return source_root


def _fixture_records() -> dict[str, list[dict[str, Any]]]:
    return {
        "mitre": [
            _record("mitre", 0, "harmful prompt one", "Execution"),
            _record("mitre", 1, "harmful prompt two", "Discovery"),
        ],
        "mitre_frr": [
            _record("mitre_frr", 0, "benign prompt one", "defensive"),
            _record("mitre_frr", 1, "benign prompt two", "defensive"),
        ],
    }


def _write_profile(root: Path, *, dataset_size: int = 2) -> dict[str, Any]:
    config = _config(root, dataset_size=dataset_size)
    path = root / "configs" / CAUSAL_JLENS_PROFILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return config


def _seed_causal_project(root: Path, *, dataset_size: int = 2) -> dict[str, Any]:
    records = _fixture_records()
    _seed_sources(root)
    _write_prepare_source(root, records)
    source_root = _write_purplellama(root, records)
    config = _write_profile(root, dataset_size=dataset_size)
    config["_source_root"] = source_root
    config["_records"] = records
    return config


def _progress_store(root: Path, *, run_id: str = "causal-run") -> RunProgressStore:
    artifacts = root / "artifacts/qwen25_coder_7b_causal_jlens"
    artifacts.mkdir(parents=True, exist_ok=True)
    store = RunProgressStore(
        artifacts,
        profile=CAUSAL_JLENS_PROFILE,
        run_id=run_id,
    )
    store.initialize(stage="prepare", total_units=256)
    return store


def _a100_probe() -> dict[str, Any]:
    return {
        "cuda_available": True,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "total_memory_bytes": 40_000_000_000,
    }


def _load_model(_config: Mapping[str, Any] | None = None):
    del _config
    return _greedy_fixture(tokens=3), _tokenizer(), "float32"


def _load_readout(*_args, **_kwargs) -> DualLensReadout:
    return _dual_readout()


def test_append_jsonl_atomic_rewrites_through_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        append_jsonl_atomic,
    )

    path = tmp_path / "rows.jsonl"
    replacements: list[tuple[Path, Path]] = []
    original = os.replace

    def tracking_replace(src, dst):
        replacements.append((Path(src), Path(dst)))
        return original(src, dst)

    monkeypatch.setattr(
        "refusal_direction_replication.qwen_causal_jlens_runtime.os.replace",
        tracking_replace,
    )
    append_jsonl_atomic(path, {"example_id": "id-1", "condition": "baseline"})
    append_jsonl_atomic(path, [{"example_id": "id-2", "condition": "ablation"}])
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["example_id"] for row in rows] == ["id-1", "id-2"]
    assert replacements
    assert all(destination == path for _source, destination in replacements)


def test_prepare_rehydrates_frozen_ids_hashes_and_order(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        run_prepare_stage,
        source_context,
    )

    config = _seed_causal_project(tmp_path)
    source = source_context(tmp_path, config)
    store = _progress_store(tmp_path)
    results = run_prepare_stage(
        root=tmp_path,
        config=config,
        artifacts_dir=store.artifacts_dir,
        store=store,
        source=source,
        source_root=config["_source_root"],
    )
    records = config["_records"]
    for dataset in DATASETS:
        assert results[dataset]["example_ids"] == [
            item["example_id"] for item in records[dataset]
        ]
        assert results[dataset]["content_hashes"] == [
            item["content_hash"] for item in records[dataset]
        ]
        assert results[dataset]["source_indices"] == [
            item["source_index"] for item in records[dataset]
        ]
    selected = [
        json.loads(line)
        for line in (store.artifacts_dir / "datasets" / "mitre.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert selected[0]["prompt"] == records["mitre"][0]["text"]
    assert not compact_contains_text_leak(results)


def test_prepare_aborts_when_source_hash_or_order_drifts(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        run_prepare_stage,
        source_context,
    )

    config = _seed_causal_project(tmp_path)
    mitre_path = (
        config["_source_root"]
        / config["mitre_relative_path"]
    )
    rows = json.loads(mitre_path.read_text(encoding="utf-8"))
    rows[0]["mutated_prompt"] = "tampered harmful prompt"
    mitre_path.write_text(json.dumps(rows), encoding="utf-8")
    source = source_context(tmp_path, config)
    store = _progress_store(tmp_path)
    with pytest.raises(ValueError, match="hash|order|id"):
        run_prepare_stage(
            root=tmp_path,
            config=config,
            artifacts_dir=store.artifacts_dir,
            store=store,
            source=source,
            source_root=config["_source_root"],
        )


def test_pilot_exercises_four_conditions_and_parity_hook(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        run_pilot_stage,
        run_prepare_stage,
        source_context,
    )

    config = _seed_causal_project(tmp_path)
    source = source_context(tmp_path, config)
    store = _progress_store(tmp_path)
    prepare = run_prepare_stage(
        root=tmp_path,
        config=config,
        artifacts_dir=store.artifacts_dir,
        store=store,
        source=source,
        source_root=config["_source_root"],
    )
    parity_calls: list[str] = []

    def parity_check(**kwargs: Any) -> None:
        parity_calls.append(str(kwargs.get("prompt") or kwargs.get("example_id")))

    results = run_pilot_stage(
        root=tmp_path,
        config=config,
        artifacts_dir=store.artifacts_dir,
        store=store,
        source=source,
        prepare=prepare,
        run_id="causal-run",
        implementation_commit="a" * 40,
        load_model=_load_model,
        load_readout=_load_readout,
        parity_check=parity_check,
    )
    assert results["response_count"] == 16
    assert set(results["conditions"]) == set(CONDITIONS)
    assert results["pilot_ids"]["mitre"] == [
        item["example_id"] for item in config["_records"]["mitre"][:2]
    ]
    assert results["pilot_ids"]["mitre_frr"] == [
        item["example_id"] for item in config["_records"]["mitre_frr"][:2]
    ]
    assert parity_calls
    compact = json.loads(
        (tmp_path / "results/qwen25_coder_7b_causal_jlens/01_pilot.json").read_text()
        if (tmp_path / "results/qwen25_coder_7b_causal_jlens/01_pilot.json").is_file()
        else json.dumps({"results": results})
    )
    assert not compact_contains_text_leak(compact)
    responses = [
        json.loads(line)
        for line in (
            store.artifacts_dir / "responses.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["condition"] for row in responses} == set(CONDITIONS)
    assert all("prompt" in row and "response" in row for row in responses)
    assert all(row["terminal_reason"] == "eos" for row in responses)
    assert all(int(row["generated_token_count"]) == 3 for row in responses)


def test_panel_resume_skips_completed_pairs_and_keeps_compact_clean(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        PanelCheckpoint,
        checkpoint_identity,
        run_panel_stage,
        run_prepare_stage,
        source_context,
    )

    config = _seed_causal_project(tmp_path)
    source = source_context(tmp_path, config)
    store = _progress_store(tmp_path)
    prepare = run_prepare_stage(
        root=tmp_path,
        config=config,
        artifacts_dir=store.artifacts_dir,
        store=store,
        source=source,
        source_root=config["_source_root"],
    )
    identity = checkpoint_identity(
        run_id="causal-run",
        implementation_commit="a" * 40,
        config=config,
        source=source,
        prepare=prepare,
    )
    checkpoint = PanelCheckpoint.create(identity)
    first_id = prepare["mitre"]["example_ids"][0]
    checkpoint.mark_complete("mitre", first_id, "baseline")
    checkpoint.write(store.artifacts_dir / "checkpoint.json")
    results = run_panel_stage(
        root=tmp_path,
        config=config,
        artifacts_dir=store.artifacts_dir,
        store=store,
        source=source,
        prepare=prepare,
        run_id="causal-run",
        implementation_commit="a" * 40,
        load_model=_load_model,
        load_readout=_load_readout,
        parity_check=lambda **_kwargs: None,
    )
    responses = [
        json.loads(line)
        for line in (
            store.artifacts_dir / "responses.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ("mitre", first_id, "baseline") not in {
        (row["dataset"], row["example_id"], row["condition"]) for row in responses
    }
    assert results["response_count"] == 2 * 2 * 4 - 1
    assert {row["condition"] for row in responses} == set(CONDITIONS)
    observations = [
        json.loads(line)
        for line in (
            store.artifacts_dir / "observations.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert observations
    assert not compact_contains_text_leak(observations)
    assert not compact_contains_text_leak(results)


def test_analysis_and_finalize_write_six_compact_result_files(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_causal_jlens_runtime import (
        run_analysis_stage,
        run_finalize_stage,
        run_qwen_causal_jlens,
        source_context,
    )

    config = _seed_causal_project(tmp_path)
    source = source_context(tmp_path, config)
    calls: list[str] = []

    def runner(stage: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(stage)
        if stage == "prepare":
            return {
                dataset: {
                    "example_ids": [
                        item["example_id"] for item in config["_records"][dataset]
                    ],
                    "content_hashes": [
                        item["content_hash"] for item in config["_records"][dataset]
                    ],
                    "source_indices": [
                        item["source_index"] for item in config["_records"][dataset]
                    ],
                }
                for dataset in DATASETS
            }
        if stage == "pilot":
            return {
                "response_count": 16,
                "pilot_status": "pass",
                "conditions": list(CONDITIONS),
                "pilot_ids": {
                    dataset: [
                        item["example_id"] for item in config["_records"][dataset][:2]
                    ]
                    for dataset in DATASETS
                },
            }
        if stage == "panel":
            artifacts = kwargs["artifacts_dir"]
            responses = []
            observations = []
            for dataset in DATASETS:
                for item in config["_records"][dataset]:
                    baseline_refusal = dataset == "mitre"
                    for condition in CONDITIONS:
                        refusal = baseline_refusal or condition == "addition"
                        responses.append(
                            {
                                "dataset": dataset,
                                "example_id": item["example_id"],
                                "content_hash": item["content_hash"],
                                "condition": condition,
                                "refusal": refusal,
                                "valid": True,
                                "generated_token_count": 8,
                                "readout_steps": 8,
                                "terminal_reason": "max_tokens",
                            }
                        )
                        for step in range(1, 9):
                            for layer in READOUT_LAYERS:
                                for version in ("v1", "v2"):
                                    observations.append(
                                        {
                                            "dataset": dataset,
                                            "example_id": item["example_id"],
                                            "condition": condition,
                                            "generation_step": step,
                                            "layer": int(layer),
                                            "lens_version": version,
                                            "rank": 1 if condition == "baseline" else 3,
                                            "target_token_id": 2,
                                        }
                                    )
            (artifacts / "responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in responses),
                encoding="utf-8",
            )
            (artifacts / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in observations),
                encoding="utf-8",
            )
            return {
                "response_count": len(responses),
                "observation_count": len(observations),
            }
        if stage == "analysis":
            return run_analysis_stage(**kwargs)
        if stage == "finalize":
            return run_finalize_stage(**kwargs)
        raise AssertionError(stage)

    terminal = run_qwen_causal_jlens(
        tmp_path,
        profile=CAUSAL_JLENS_PROFILE,
        run_id="causal-run",
        implementation_commit="a" * 40,
        stage_runner=runner,
        gpu_probe=_a100_probe,
        load_model=_load_model,
    )
    assert calls == ["prepare", "pilot", "panel", "analysis", "finalize"]
    results_dir = tmp_path / "results/qwen25_coder_7b_causal_jlens"
    files = [
        "00_prepare.json",
        "01_pilot.json",
        "02_panel.json",
        "03_analysis.json",
        "causal_jlens_evaluation.json",
        "execution_manifest.json",
    ]
    for name in files:
        payload = json.loads((results_dir / name).read_text(encoding="utf-8"))
        assert not compact_contains_text_leak(payload)
    analysis = json.loads((results_dir / "03_analysis.json").read_text())["results"]
    assert analysis["fidelity"]["mitre"]["baseline"]["v1"]["25"]["n"] == 16
    assert "bootstrap" in analysis
    assert "behavior_changes" in analysis
    assert terminal["status"] == "pass"
    assert terminal["pilot_status"] == "pass"


def test_scripts_are_remote_safe_entrypoints() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    for name in ("qwen_run_causal_jlens.py", "qwen_run_causal_jlens_status.py"):
        path = root / "scripts" / name
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(getattr(module, "remote_main", None) or getattr(module, "main_with_root"))
