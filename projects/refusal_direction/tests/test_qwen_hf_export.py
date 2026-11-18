from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from refusal_direction_replication.directions import CandidateDirection, save_direction
from refusal_direction_replication.qwen_dose import sha256_file
from refusal_direction_replication.qwen_hf_export import (
    EXPECTED_ABLATION_COEFFICIENT,
    EXPECTED_DIRECTION_SHA256,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SOURCE_LAYER,
    EXPORT_MANIFEST_FILENAME,
    HF_ABLATED_PROFILE,
    HuggingFaceAblatedExport,
    export_qwen_hf_ablated,
    is_export_allowlisted,
    load_hf_ablated_config,
    tensor_sha256,
)


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "export_qwen_hf_ablated.py"
PACKAGED_PROFILE = PROJECT / "configs" / HF_ABLATED_PROFILE
FROZEN_DIRECTION = "artifacts/qwen25_coder_7b/selected_direction.pt"
REMOTE_CODE = (
    "configuration_qwen2_refusal_ablated.py",
    "modeling_qwen2_refusal_ablated.py",
)


class FakeTokenizer:
    def __init__(self, revision: str = EXPECTED_MODEL_REVISION) -> None:
        self.revision = revision
        self._commit_hash = revision
        self.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "tokenizer.json").write_text("{}", encoding="utf-8")
        (output / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": self.chat_template}),
            encoding="utf-8",
        )
        (output / "vocab.json").write_text("{}", encoding="utf-8")
        (output / "merges.txt").write_text("", encoding="utf-8")


def _tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.config._commit_hash = EXPECTED_MODEL_REVISION
    return model


def _write_direction(path: Path, *, hidden_size: int = 8) -> tuple[str, torch.Tensor]:
    vector = torch.zeros(hidden_size)
    vector[0] = 1.0
    save_direction(
        path,
        CandidateDirection(
            layer=EXPECTED_SOURCE_LAYER,
            position_offset=-1,
            vector=vector,
        ),
    )
    return sha256_file(path), vector


def _export_config(direction_sha256: str, **overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name": EXPECTED_MODEL_NAME,
        "model_revision": EXPECTED_MODEL_REVISION,
        "dtype": "bfloat16",
        "format_style": "chat_template",
        "ablation_coefficient": EXPECTED_ABLATION_COEFFICIENT,
        "source_layer": EXPECTED_SOURCE_LAYER,
        "source_direction": FROZEN_DIRECTION,
        "direction_sha256": direction_sha256,
        "artifacts_dir": "artifacts/qwen25_coder_7b_hf_ablated",
    }
    payload.update(overrides)
    return payload


def _export(
    root: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    load_pretrained: Any | None = None,
) -> HuggingFaceAblatedExport:
    def _load(model_name: str, *, dtype: torch.dtype, revision: str):
        assert model_name == EXPECTED_MODEL_NAME
        assert revision == EXPECTED_MODEL_REVISION
        assert dtype is torch.bfloat16
        return _tiny_qwen(), FakeTokenizer(revision=revision)

    return export_qwen_hf_ablated(
        root,
        output_dir=output_dir,
        config=config,
        load_pretrained=load_pretrained or _load,
    )


def test_packaged_config_pins_revision_coefficient_and_direction_hash() -> None:
    assert PACKAGED_PROFILE.is_file()
    config = load_hf_ablated_config(PROJECT)
    assert config["model_name"] == EXPECTED_MODEL_NAME
    assert config["model_revision"] == EXPECTED_MODEL_REVISION
    assert config["ablation_coefficient"] == pytest.approx(0.2)
    assert config["source_layer"] == EXPECTED_SOURCE_LAYER
    assert config["source_direction"] == FROZEN_DIRECTION
    assert config["direction_sha256"] == EXPECTED_DIRECTION_SHA256
    blob = json.dumps(config)
    assert "prompt" not in blob
    assert "generation" not in blob
    assert "credential" not in blob


def test_export_rejects_wrong_revision(tmp_path: Path) -> None:
    direction = tmp_path / FROZEN_DIRECTION
    digest, _vector = _write_direction(direction)
    config = _export_config(digest, model_revision="deadbeef")
    with pytest.raises(ValueError, match="revision"):
        _export(tmp_path, tmp_path / "out", config)


def test_export_rejects_coefficient_other_than_0_2(tmp_path: Path) -> None:
    direction = tmp_path / FROZEN_DIRECTION
    digest, _vector = _write_direction(direction)
    config = _export_config(digest, ablation_coefficient=1.0)
    with pytest.raises(ValueError, match="coefficient"):
        _export(tmp_path, tmp_path / "out", config)


def test_export_rejects_direction_hash_mismatch(tmp_path: Path) -> None:
    direction = tmp_path / FROZEN_DIRECTION
    _write_direction(direction)
    config = _export_config("0" * 64)
    with pytest.raises(ValueError, match="direction"):
        _export(tmp_path, tmp_path / "out", config)


def test_export_refuses_incomplete_output_directory(tmp_path: Path) -> None:
    direction = tmp_path / FROZEN_DIRECTION
    digest, _vector = _write_direction(direction)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="incomplete"):
        _export(tmp_path, output_dir, _export_config(digest))


def test_export_writes_allowlisted_checkpoint_and_manifest(tmp_path: Path) -> None:
    direction = tmp_path / FROZEN_DIRECTION
    digest, vector = _write_direction(direction)
    output_dir = tmp_path / "out"
    result = _export(tmp_path, output_dir, _export_config(digest))
    names = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    assert EXPORT_MANIFEST_FILENAME in names
    assert "config.json" in names
    assert "model.safetensors" in names
    for filename in REMOTE_CODE:
        assert filename in names
    assert "tokenizer.json" in names
    assert "tokenizer_config.json" in names
    assert all(is_export_allowlisted(name) for name in names)
    assert "prompts.json" not in names
    assert "generations.json" not in names
    manifest = json.loads(
        (output_dir / EXPORT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert result.manifest == manifest
    assert result.output_dir == output_dir
    assert manifest["base_model_revision"] == EXPECTED_MODEL_REVISION
    assert manifest["ablation_coefficient"] == pytest.approx(0.2)
    assert manifest["source_layer"] == EXPECTED_SOURCE_LAYER
    assert manifest["direction_sha256"] == digest
    assert manifest["direction_vector_sha256"] == tensor_sha256(vector)
    assert set(manifest["files"]) == set(names) - {EXPORT_MANIFEST_FILENAME}
    saved = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["ablation_coefficient"] == pytest.approx(0.2)
    assert saved["auto_map"]["AutoModelForCausalLM"].endswith(
        "Qwen2RefusalAblatedForCausalLM"
    )
    assert saved["direction_sha256"] == digest
    from safetensors.torch import load_file

    state = load_file(output_dir / "model.safetensors")
    assert "refusal_direction" in state
    assert torch.equal(state["refusal_direction"], vector)


def test_cli_forwards_root_and_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("export_qwen_hf_ablated_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called: dict[str, Any] = {}

    def fake_export(root: Path, **kwargs: Any) -> HuggingFaceAblatedExport:
        called["root"] = Path(root)
        called.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        return HuggingFaceAblatedExport(
            output_dir=output_dir,
            manifest_path=output_dir / EXPORT_MANIFEST_FILENAME,
            manifest={"ok": True},
        )

    monkeypatch.setattr(module, "export_qwen_hf_ablated", fake_export)
    output_dir = tmp_path / "checkpoint"
    assert module.main(["--root", str(tmp_path), "--output-dir", str(output_dir)]) == 0
    assert called["root"] == tmp_path
    assert called["output_dir"] == output_dir
