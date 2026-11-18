from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

from refusal_direction_replication.interventions import (
    ablate_direction,
    directional_ablation_hook_specs,
)
from refusal_direction_replication.pipeline import transformer_module_lists

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hf_qwen_ablated.configuration_qwen2_refusal_ablated import (  # noqa: E402
    Qwen2RefusalAblatedConfig,
)
from hf_qwen_ablated.modeling_qwen2_refusal_ablated import (  # noqa: E402
    Qwen2RefusalAblatedForCausalLM,
)


BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DEFAULT_COEFFICIENT = 0.2
SOURCE_LAYER = 18
DIRECTION_SHA256 = "ab" * 32


def _tiny_config(**overrides: object) -> Qwen2RefusalAblatedConfig:
    values: dict[str, object] = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "direction_sha256": DIRECTION_SHA256,
        "source_layer": SOURCE_LAYER,
        "base_model_name": BASE_MODEL_NAME,
        "base_model_revision": BASE_MODEL_REVISION,
    }
    values.update(overrides)
    return Qwen2RefusalAblatedConfig(**values)


def _tiny_model(**overrides: object) -> Qwen2RefusalAblatedForCausalLM:
    torch.manual_seed(0)
    return Qwen2RefusalAblatedForCausalLM(_tiny_config(**overrides)).eval()


def _axis_direction(hidden_size: int = 8) -> torch.Tensor:
    direction = torch.zeros(hidden_size)
    direction[0] = 1.0
    return direction


def test_config_registers_auto_map_provenance_and_default_coefficient() -> None:
    config = _tiny_config()
    assert config.ablation_coefficient == pytest.approx(DEFAULT_COEFFICIENT)
    assert config.direction_sha256 == DIRECTION_SHA256
    assert config.source_layer == SOURCE_LAYER
    assert config.base_model_name == BASE_MODEL_NAME
    assert config.base_model_revision == BASE_MODEL_REVISION
    assert config.auto_map == {
        "AutoConfig": (
            "configuration_qwen2_refusal_ablated.Qwen2RefusalAblatedConfig"
        ),
        "AutoModelForCausalLM": (
            "modeling_qwen2_refusal_ablated.Qwen2RefusalAblatedForCausalLM"
        ),
    }


def test_config_rejects_coefficient_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="ablation coefficient"):
        _tiny_config(ablation_coefficient=1.1)
    with pytest.raises(ValueError, match="ablation coefficient"):
        _tiny_config(ablation_coefficient=-0.01)
    with pytest.raises(ValueError, match="finite"):
        _tiny_config(ablation_coefficient=math.nan)
    with pytest.raises(ValueError, match="finite"):
        _tiny_config(ablation_coefficient=math.inf)


def test_model_stores_direction_as_persistent_buffer() -> None:
    model = _tiny_model()
    assert "refusal_direction" in model._buffers
    assert "refusal_direction" not in model._non_persistent_buffers_set
    assert "refusal_direction" in model.state_dict()
    assert tuple(model.refusal_direction.shape) == (model.config.hidden_size,)


def test_hooks_install_once_on_the_same_modules_as_runtime_specs() -> None:
    model = _tiny_model()
    blocks, attentions, mlps = transformer_module_lists(model)
    expected_pre, expected_forward = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        model.refusal_direction,
        coefficient=DEFAULT_COEFFICIENT,
    )
    assert [module for module, _hook in model.ablation_pre_hooks] == [
        module for module, _hook in expected_pre
    ]
    assert [module for module, _hook in model.ablation_forward_hooks] == [
        module for module, _hook in expected_forward
    ]
    first_pre = [len(module._forward_pre_hooks) for module in blocks]
    first_forward = [
        len(module._forward_hooks) for module in [*attentions, *mlps]
    ]
    model.install_ablation_hooks()
    assert [len(module._forward_pre_hooks) for module in blocks] == first_pre
    assert [
        len(module._forward_hooks) for module in [*attentions, *mlps]
    ] == first_forward
    assert first_pre == [1] * len(blocks)
    assert first_forward == [1] * (len(attentions) + len(mlps))


def test_installed_hooks_match_ablate_direction_at_coefficient_0_2() -> None:
    model = _tiny_model()
    direction = _axis_direction()
    model.refusal_direction.copy_(direction)
    hidden = torch.tensor([[[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    expected = ablate_direction(hidden, direction, coefficient=DEFAULT_COEFFICIENT)
    pre_changed, = model.ablation_pre_hooks[0][1](None, (hidden,))
    forward_changed = model.ablation_forward_hooks[0][1](None, None, hidden)
    assert torch.equal(pre_changed, expected)
    assert torch.equal(forward_changed, expected)
    assert torch.equal(pre_changed, torch.tensor([[[2.4, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]))
