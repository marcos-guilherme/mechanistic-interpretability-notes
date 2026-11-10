from __future__ import annotations

import pytest
import torch

from ioi_replication.paper import (
    HeadPositionNode,
    PathReceiver,
    SemanticPosition,
    activation_patch_node,
    build_paired_dataset,
    cache_grouped_mean_activations,
    logit_difference,
    measure_backup_name_movers,
    paper_path_patch,
)
from ioi_replication.runtime import load_gpt2_small, select_device


@pytest.fixture(scope="module")
def real_model_and_dataset():
    try:
        model = load_gpt2_small(select_device())
    except (OSError, RuntimeError) as error:
        pytest.skip(f"cached GPT-2 Small is unavailable: {error}")
    dataset = build_paired_dataset(model.tokenizer, n=2, seed=71)
    return model, dataset


def _metric(dataset, device):
    return lambda logits: logit_difference(
        logits,
        dataset.io_token_ids.to(device),
        dataset.s_token_ids.to(device),
        dataset.positions["END"].to(device),
    )


def test_real_gpt2_activation_patch_result_endpoint_has_finite_signal(
    real_model_and_dataset,
) -> None:
    model, dataset = real_model_and_dataset
    device = torch.device(model.cfg.device)
    ioi_tokens = dataset.ioi_tokens.to(device)
    abc_tokens = dataset.abc_tokens.to(device)
    metric = _metric(dataset, device)
    node = HeadPositionNode(9, 9, SemanticPosition.END)

    model.set_use_attn_result(True)
    try:
        _, source_cache = model.run_with_cache(
            abc_tokens,
            names_filter=["blocks.9.attn.hook_result"],
        )
    finally:
        model.set_use_attn_result(False)

    baseline = metric(model(ioi_tokens))
    patched = activation_patch_node(
        model=model,
        target_tokens=ioi_tokens,
        source_cache=source_cache,
        metric=metric,
        node=node,
        target_positions=dataset.positions,
        source_positions=dataset.positions,
        activation="result",
    )

    assert patched.shape == baseline.shape == torch.Size([])
    assert torch.isfinite(patched)
    assert not torch.allclose(patched, baseline, atol=1e-7, rtol=0)
    assert model.cfg.use_attn_result is False


def test_real_gpt2_paper_path_patch_has_finite_nonzero_effect(
    real_model_and_dataset,
) -> None:
    model, dataset = real_model_and_dataset
    device = torch.device(model.cfg.device)
    metric = _metric(dataset, device)

    result = paper_path_patch(
        model,
        dataset.ioi_tokens.to(device),
        dataset.abc_tokens.to(device),
        dataset.positions,
        dataset.positions,
        senders=(
            HeadPositionNode(3, 0, SemanticPosition.S2),
        ),
        receivers=(
            PathReceiver(7, 3, "v", SemanticPosition.S2),
        ),
        metric=metric,
    )

    assert result.baseline.shape == result.patched.shape == torch.Size([])
    assert torch.isfinite(result.baseline)
    assert torch.isfinite(result.patched)
    assert result.effect.abs().item() > 1e-8
    assert model.cfg.use_attn_result is False


def test_real_gpt2_backup_knockout_has_finite_structured_signals(
    real_model_and_dataset,
) -> None:
    model, dataset = real_model_and_dataset
    device = torch.device(model.cfg.device)
    means = cache_grouped_mean_activations(
        model,
        dataset.abc_tokens.to(device),
        dataset.groups,
        activation="result",
    )

    result = measure_backup_name_movers(
        model=model,
        dataset=dataset,
        group_means=means,
    )

    assert len(result.heads) == 8
    assert torch.isfinite(result.baseline_logit_difference.values).all()
    assert torch.isfinite(result.knockout_logit_difference.values).all()
    for metrics in result.heads.values():
        assert torch.isfinite(metrics.baseline_dla.values).all()
        assert torch.isfinite(metrics.knockout_dla.values).all()
        assert torch.isfinite(metrics.baseline_attention_io.values).all()
        assert torch.isfinite(metrics.knockout_attention_io.values).all()
        assert torch.isfinite(metrics.baseline_attention_s.values).all()
        assert torch.isfinite(metrics.knockout_attention_s.values).all()
    assert model.cfg.use_attn_result is False
