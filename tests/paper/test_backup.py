from __future__ import annotations

import pytest
import torch

import ioi_replication.paper.backup as backup_module
from ioi_replication.paper.backup import (
    measure_backup_heads_from_cache,
    scaled_head_logit_attribution,
)
from ioi_replication.paper.circuit import (
    PUBLISHED_CIRCUIT,
    CircuitClass,
)


def test_scaled_head_dla_centers_and_uses_example_ln_final_scale() -> None:
    result = torch.zeros(2, 3, 2, 2)
    result[0, 2, 1] = torch.tensor([4.0, 2.0])
    result[1, 1, 1] = torch.tensor([8.0, 4.0])
    scale = torch.ones(2, 3, 1)
    scale[0, 2] = 2.0
    scale[1, 1] = 4.0
    unembed = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    values = scaled_head_logit_attribution(
        result,
        scale,
        unembed,
        io_token_ids=torch.tensor([0, 0]),
        s_token_ids=torch.tensor([1, 1]),
        end_positions=torch.tensor([2, 1]),
        head=1,
    )

    assert values.shape == (2,)
    assert torch.allclose(values, torch.tensor([1.0, 1.0]))


def test_dla_delta_decomposes_result_change_from_scale_change() -> None:
    baseline_result = torch.zeros(2, 3, 2, 2)
    knockout_result = torch.zeros_like(baseline_result)
    baseline_result[:, 2, 1] = torch.tensor([4.0, 2.0])
    knockout_result[:, 2, 1] = torch.tensor([6.0, 2.0])
    baseline_scale = torch.full((2, 3, 1), 2.0)
    knockout_scale = torch.full((2, 3, 1), 4.0)

    decomposition = backup_module.decompose_head_logit_attribution(
        baseline_result,
        knockout_result,
        baseline_scale,
        knockout_scale,
        torch.eye(2),
        io_token_ids=torch.tensor([0, 0]),
        s_token_ids=torch.tensor([1, 1]),
        end_positions=torch.tensor([2, 2]),
        head=1,
    )

    assert torch.allclose(
        decomposition.delta_result_fixed_scale,
        torch.ones(2),
    )
    assert torch.allclose(decomposition.delta_scale, -torch.ones(2))
    assert torch.allclose(decomposition.delta_total, torch.zeros(2))
    assert torch.allclose(
        decomposition.delta_result_fixed_scale + decomposition.delta_scale,
        decomposition.delta_total,
    )


def _cache(
    *,
    result_vector: tuple[float, float],
    attention_io: float,
    attention_s1: float,
    attention_s2: float,
    scale: float = 2.0,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {
        "ln_final.hook_scale": torch.full((2, 3, 1), scale),
    }
    for layer, head in ((9, 0), (10, 1)):
        result = torch.zeros(2, 3, 12, 2)
        result[:, 2, head] = torch.tensor(result_vector)
        pattern = torch.zeros(2, 12, 3, 3)
        pattern[:, head, 2, 0] = attention_io
        pattern[:, head, 2, 1] = attention_s1
        pattern[:, head, 2, 2] = attention_s2
        cache[f"blocks.{layer}.attn.hook_result"] = result
        cache[f"blocks.{layer}.attn.hook_pattern"] = pattern
    return cache


def test_backup_measurements_separate_baseline_from_intervention_changes() -> None:
    baseline_logits = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [3.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [5.0, 2.0]],
        ]
    )
    knockout_logits = baseline_logits.clone()
    knockout_logits[:, 2, 0] += 1.0
    baseline_cache = _cache(
        result_vector=(4.0, 2.0),
        attention_io=0.2,
        attention_s1=0.7,
        attention_s2=0.4,
    )
    knockout_cache = _cache(
        result_vector=(6.0, 2.0),
        attention_io=0.8,
        attention_s1=0.1,
        attention_s2=0.3,
    )
    knockout_cache["blocks.9.attn.hook_result"] = baseline_cache[
        "blocks.9.attn.hook_result"
    ].clone()
    knockout_cache["blocks.9.attn.hook_pattern"] = baseline_cache[
        "blocks.9.attn.hook_pattern"
    ].clone()
    heads = ((9, 0), (10, 1))

    result = measure_backup_heads_from_cache(
        baseline_logits=baseline_logits,
        knockout_logits=knockout_logits,
        baseline_cache=baseline_cache,
        knockout_cache=knockout_cache,
        unembed=torch.eye(2),
        io_token_ids=torch.tensor([0, 0]),
        s_token_ids=torch.tensor([1, 1]),
        end_positions=torch.tensor([2, 2]),
        io_positions=torch.tensor([0, 0]),
        s1_positions=torch.tensor([1, 1]),
        s2_positions=torch.tensor([2, 2]),
        heads=heads,
        causally_downstream_heads=((10, 1),),
    )

    assert set(result.heads) == set(heads)
    assert result.baseline_logit_difference.mean == pytest.approx(2.5)
    assert result.knockout_logit_difference.mean == pytest.approx(3.5)
    same_layer = result.heads[(9, 0)]
    assert same_layer.delta_result_fixed_scale.mean == pytest.approx(0.0)
    assert same_layer.delta_attention_io.mean == pytest.approx(0.0)
    assert same_layer.delta_attention_s1.mean == pytest.approx(0.0)
    assert same_layer.delta_attention_s2.mean == pytest.approx(0.0)
    assert same_layer.is_causally_downstream is False

    posterior = result.heads[(10, 1)]
    assert posterior.baseline_dla.mean == pytest.approx(1.0)
    assert posterior.knockout_dla.mean == pytest.approx(2.0)
    assert posterior.delta_dla.mean == pytest.approx(1.0)
    assert posterior.delta_result_fixed_scale.mean == pytest.approx(1.0)
    assert posterior.delta_scale.mean == pytest.approx(0.0)
    assert posterior.baseline_attention_io.mean == pytest.approx(0.2)
    assert posterior.knockout_attention_io.mean == pytest.approx(0.8)
    assert posterior.delta_attention_io.mean == pytest.approx(0.6)
    assert posterior.baseline_attention_s1.mean == pytest.approx(0.7)
    assert posterior.knockout_attention_s1.mean == pytest.approx(0.1)
    assert posterior.delta_attention_s1.mean == pytest.approx(-0.6)
    assert posterior.baseline_attention_s2.mean == pytest.approx(0.4)
    assert posterior.knockout_attention_s2.mean == pytest.approx(0.3)
    assert posterior.delta_attention_s2.mean == pytest.approx(-0.1)
    assert posterior.delta_dla.standard_error == pytest.approx(0.0)
    assert posterior.is_causally_downstream is True


def test_compensation_fraction_is_ratio_of_means_with_bootstrap_limits() -> None:
    fraction = backup_module.compute_compensation_fraction(
        backup_delta_result=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        removed_primary_baseline=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        n_bootstrap=200,
        seed=7,
    )

    assert fraction.value == pytest.approx(0.5)
    assert fraction.standard_error >= 0
    assert fraction.ci_lower <= fraction.value <= fraction.ci_upper
    assert "ratio of means" in fraction.definition
    assert "not" in fraction.final_logit_delta_warning.lower()


def test_published_backup_head_set_has_eight_entries() -> None:
    backup_nodes = PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]

    assert len(backup_nodes) == 8
    assert len({(node.layer, node.head) for node in backup_nodes}) == 8
