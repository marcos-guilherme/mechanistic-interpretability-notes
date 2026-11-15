import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refusal_direction_replication.directions import candidate_grid
from refusal_direction_replication.model import (
    GEMMA_REFUSAL_TOKEN_IDS,
    end_of_instruction_offsets,
    format_instruction,
    resolve_refusal_token_ids,
    tokenize_instructions,
)
from refusal_direction_replication.pipeline import (
    load_config,
    project_paths,
    transformer_module_lists,
)

ROOT = Path(__file__).parents[1]
QWEN_CONFIG = ROOT / "configs" / "qwen25_coder_7b.json"


class ChatTemplateTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.chat_calls: list[tuple[list[dict[str, str]], bool, bool]] = []
        self.prefix_token_ids = {
            "I": [10],
            "I'm": [11, 12],
            "Sorry": [13],
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        content = messages[0]["content"]
        base = [100, 101, *self.encode(content, add_special_tokens=False)]
        if add_generation_prompt:
            base.extend([200, 201])
        if tokenize:
            return base
        return f"<user>{content}</user><assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text in self.prefix_token_ids:
            return self.prefix_token_ids[text]
        return [ord(character) for character in text]

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding and not truncation and return_tensors == "pt"
        ids = [
            torch.tensor(self.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            ))
            for text in texts
        ]
        max_len = max(item.shape[0] for item in ids)
        batch = torch.full((len(ids), max_len), self.pad_token_id)
        for row, item in enumerate(ids):
            batch[row, -item.shape[0] :] = item
        attention = (batch != self.pad_token_id).to(torch.long)
        return {"input_ids": batch, "attention_mask": attention}


def test_qwen_config_matches_required_profile() -> None:
    config = json.loads(QWEN_CONFIG.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["model_revision"] == (
        "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
    assert config["dtype"] == "bfloat16"
    assert config["batch_size"] == 1
    assert config["seed"] == 42
    assert config["train_per_class"] == 128
    assert config["validation_per_class"] == 32
    assert config["max_new_tokens"] == 128
    assert config["ce_tolerance"] == 0.1
    assert config["upstream_commit"] == (
        "9d852fae1a9121c78b29142de733cb1340770cc3"
    )
    assert config["results_dir"] == "results/qwen25_coder_7b"
    assert config["artifacts_dir"] == "artifacts/qwen25_coder_7b"
    assert config["position_offsets"] == [-1]


def test_load_config_defaults_to_gemma_profile() -> None:
    config = load_config(ROOT)
    assert config["model_name"] == "google/gemma-2b-it"


def test_load_config_accepts_explicit_qwen_profile() -> None:
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    assert config["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_project_paths_keep_gemma_defaults() -> None:
    config = load_config(ROOT)
    results, artifacts = project_paths(ROOT, config)
    assert results == ROOT / "results"
    assert artifacts == ROOT / "artifacts"


def test_project_paths_use_qwen_namespaces() -> None:
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    results, artifacts = project_paths(ROOT, config)
    assert results == ROOT / "results" / "qwen25_coder_7b"
    assert artifacts == ROOT / "artifacts" / "qwen25_coder_7b"


def test_tokenize_instructions_use_chat_template_for_qwen() -> None:
    tokenizer = ChatTemplateTokenizer()
    batch = tokenize_instructions(
        tokenizer,
        ["alpha", "beta"],
        format_style="chat_template",
    )
    assert batch["input_ids"].shape[0] == 2
    assert tokenizer.chat_calls[:2] == [
        ([{"role": "user", "content": "alpha"}], False, True),
        ([{"role": "user", "content": "beta"}], False, True),
    ]


def test_tokenize_instructions_preserve_gemma_default() -> None:
    class GemmaTokenizer:
        padding_side = "left"

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
            }

    tokenizer = GemmaTokenizer()
    tokenize_instructions(tokenizer, ["hello", "world"])
    assert "hello" in format_instruction("hello")


def test_resolve_refusal_token_ids_use_prefix_first_tokens() -> None:
    tokenizer = ChatTemplateTokenizer()
    resolved = resolve_refusal_token_ids(
        tokenizer,
        ("I", "I'm", "Sorry", "I"),
    )
    assert resolved == (10, 11, 13)


def test_assistant_boundary_offsets_are_generic_for_chat_template() -> None:
    tokenizer = ChatTemplateTokenizer()
    offsets = end_of_instruction_offsets(
        tokenizer,
        format_style="chat_template",
    )
    assert offsets == (-2, -1)


def test_qwen_candidate_grid_has_one_direction_per_layer() -> None:
    layers = 28
    examples = 4
    hidden = 8
    harmful = torch.randn(examples, layers, 1, hidden)
    harmless = torch.randn(examples, layers, 1, hidden)
    candidates = candidate_grid(
        harmful,
        harmless,
        position_offsets=(-1,),
    )
    assert len(candidates) == layers
    assert {item.position_offset for item in candidates} == {-1}


def test_transformer_module_lists_work_for_qwen_like_layout() -> None:
    layer = SimpleNamespace(
        self_attn=object(),
        mlp=object(),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer] * 28))
    blocks, attentions, mlps = transformer_module_lists(model)
    assert len(blocks) == len(attentions) == len(mlps) == 28


def test_gemma_refusal_token_ids_remain_default_proxy() -> None:
    assert GEMMA_REFUSAL_TOKEN_IDS == (235285,)