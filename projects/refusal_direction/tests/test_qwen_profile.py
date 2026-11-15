import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from refusal_direction_replication import model as model_module
from refusal_direction_replication.directions import candidate_grid
from refusal_direction_replication.model import (
    GEMMA_REFUSAL_TOKEN_IDS,
    end_of_instruction_offsets,
    format_instruction,
    load_causal_lm,
    resolve_refusal_token_ids,
    tokenize_instructions,
)
from refusal_direction_replication.pipeline import (
    _format_style,
    _position_offsets,
    _provenance,
    _refusal_token_ids,
    load_config,
    project_paths,
    transformer_module_lists,
    validate_profile_config,
)

ROOT = Path(__file__).parents[1]
QWEN_CONFIG = ROOT / "configs" / "qwen25_coder_7b.json"


class ChatTemplateTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.chat_calls: list[tuple[list[dict[str, str]], bool, bool]] = []
        self.batch_inputs: list[str] = []
        self.prefix_token_ids = {
            "I": [10],
            "I'm": [11, 12],
            "Sorry": [13],
            "I am": [10, 14],
            "As an": [15, 16],
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
            return list(self.prefix_token_ids[text])
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
        self.batch_inputs = list(texts)
        ids = [
            torch.tensor(self.encode(text, add_special_tokens=False))
            for text in texts
        ]
        max_len = max(item.shape[0] for item in ids)
        batch = torch.full((len(ids), max_len), self.pad_token_id)
        for row, item in enumerate(ids):
            batch[row, -item.shape[0] :] = item
        attention = (batch != self.pad_token_id).to(torch.long)
        return {"input_ids": batch, "attention_mask": attention}


class RecordingGemmaTokenizer:
    padding_side = "left"
    recorded_texts: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        RecordingGemmaTokenizer.recorded_texts.append(list(texts))
        return {
            "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
        }


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
    assert config["format_style"] == "chat_template"
    assert config["refusal_prefixes"] == [
        "I",
        "I'm",
        "I am",
        "Sorry",
        "As an",
    ]


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


def test_chat_template_batch_receives_rendered_strings() -> None:
    tokenizer = ChatTemplateTokenizer()
    tokenize_instructions(tokenizer, ["alpha"], format_style="chat_template")
    assert tokenizer.batch_inputs == ["<user>alpha</user><assistant>"]
    assert len(tokenizer.chat_calls) == 1


def test_tokenize_instructions_default_records_gemma_formatted_texts() -> None:
    RecordingGemmaTokenizer.recorded_texts.clear()
    tokenizer = RecordingGemmaTokenizer()
    tokenize_instructions(tokenizer, ["hello", "world"])
    assert RecordingGemmaTokenizer.recorded_texts[-1] == [
        format_instruction("hello"),
        format_instruction("world"),
    ]


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


def test_position_offsets_use_qwen_config_not_generic_suffix() -> None:
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    tokenizer = ChatTemplateTokenizer()
    assert _position_offsets(tokenizer, config) == (-1,)
    assert end_of_instruction_offsets(tokenizer, format_style="chat_template") == (
        -2,
        -1,
    )


def test_format_style_and_refusal_token_ids_for_qwen_profile() -> None:
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    tokenizer = ChatTemplateTokenizer()
    assert _format_style(config) == "chat_template"
    assert _refusal_token_ids(tokenizer, config) == (10, 11, 13, 15)


def test_qwen_candidate_grid_has_one_direction_per_layer() -> None:
    layers = 28
    examples = 4
    hidden = 8
    harmful = torch.randn(examples, layers, 1, hidden)
    harmless = torch.randn(examples, layers, 1, hidden)
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    offsets = _position_offsets(ChatTemplateTokenizer(), config)
    candidates = candidate_grid(
        harmful,
        harmless,
        position_offsets=offsets,
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


def test_validate_profile_rejects_missing_qwen_fields(tmp_path: Path) -> None:
    incomplete = {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "abc",
    }
    with pytest.raises(ValueError, match="missing required profile fields"):
        validate_profile_config(incomplete, profile="incomplete.json")


def test_load_config_rejects_incomplete_non_gemma_profile(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    profile = configs / "broken.json"
    profile.write_text(
        json.dumps({"model_name": "custom/model"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required profile fields"):
        load_config(tmp_path, profile="broken.json")


def test_provenance_records_requested_model_revision() -> None:
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="resolved-revision")
    )
    payload = _provenance(model, torch.bfloat16, config)
    assert payload["requested_model_revision"] == config["model_revision"]
    assert payload["resolved_model_revision"] == "resolved-revision"


def test_load_causal_lm_passes_revision_to_pretrained(monkeypatch) -> None:
    tokenizer_calls: list[dict] = []
    model_calls: list[dict] = []

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 5
            self.eos_token_id = 6
            self.padding_side = "right"

    def fake_from_pretrained_tokenizer(name: str, **kwargs):
        tokenizer_calls.append({"name": name, **kwargs})
        return FakeTokenizer()

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.requires_grad_.return_value = None

    def fake_from_pretrained_model(name: str, **kwargs):
        model_calls.append({"name": name, **kwargs})
        return fake_model

    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained_tokenizer,
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained_model,
    )

    load_causal_lm(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        dtype=torch.bfloat16,
        revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
    )

    assert tokenizer_calls[0]["revision"] == (
        "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
    assert model_calls[0]["revision"] == (
        "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )


def test_load_causal_lm_uses_eos_pad_when_pad_missing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.eos_token_id = 99
            self.pad_token = None
            self.eos_token = "</s>"
            self.padding_side = "right"

    def fake_from_pretrained_tokenizer(name: str, **kwargs):
        return FakeTokenizer()

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.requires_grad_.return_value = None

    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained_tokenizer,
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_model,
    )

    _model, tokenizer = load_causal_lm("model", dtype=torch.float32)
    captured["pad_token"] = tokenizer.pad_token
    captured["pad_token_id"] = tokenizer.pad_token_id
    assert captured["pad_token"] == "</s>"
    assert captured["pad_token_id"] == 99
    fake_model.resize_token_embeddings.assert_not_called()


def test_load_causal_lm_resizes_embeddings_when_pad_added(monkeypatch) -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.eos_token_id = None
            self.padding_side = "right"
            self.vocab_size = 128

        def add_special_tokens(self, tokens: dict[str, str]) -> int:
            self.pad_token = tokens["pad_token"]
            self.pad_token_id = 127
            self.vocab_size = 129
            return 1

        def __len__(self) -> int:
            return self.vocab_size

    def fake_from_pretrained_tokenizer(name: str, **kwargs):
        return FakeTokenizer()

    fake_model = MagicMock()
    fake_model.eval.return_value = fake_model
    fake_model.requires_grad_.return_value = None

    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained_tokenizer,
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_model,
    )

    load_causal_lm("model", dtype=torch.float32)
    fake_model.resize_token_embeddings.assert_called_once_with(129)