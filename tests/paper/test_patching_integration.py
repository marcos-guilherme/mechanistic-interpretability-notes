from __future__ import annotations

import pytest
import torch

import ioi_replication.paper as paper_api
from ioi_replication.paper import (
    PUBLISHED_CIRCUIT,
    CircuitClass,
    HeadPositionNode,
    PathReceiver,
    SemanticPosition,
    activation_patch_node,
    build_paired_dataset,
    cache_grouped_mean_activations,
    evaluate_completeness,
    evaluate_faithfulness,
    evaluate_minimality,
    logit_difference,
    measure_backup_name_movers,
    paper_path_patch,
)
from ioi_replication.paper.validation import (
    HISTORICAL_MINIMALITY_CONTEXTS,
    MeanAblationEvaluator,
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
    reference = paper_api.build_balanced_paired_dataset(
        model.tokenizer,
        examples_per_group=1,
        seed=72,
    )
    means = cache_grouped_mean_activations(
        model,
        reference.abc_tokens.to(device),
        reference.groups,
        activation="result",
        batch_size=8,
        min_group_size=1,
        reference_config={"kind": "balanced", "n": len(reference)},
        reference_sha256=reference.dataset_sha256,
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
        assert torch.isfinite(metrics.baseline_attention_s1.values).all()
        assert torch.isfinite(metrics.knockout_attention_s1.values).all()
        assert torch.isfinite(metrics.baseline_attention_s2.values).all()
        assert torch.isfinite(metrics.knockout_attention_s2.values).all()
    assert len(result.removed_primary_baseline_dla) == 3
    assert torch.isfinite(
        result.compensation_fraction.value
        * torch.ones(())
    )
    assert model.cfg.use_attn_result is False


def test_real_gpt2_knockout_replaces_only_primary_end_nodes(
    real_model_and_dataset,
) -> None:
    model, target = real_model_and_dataset
    device = torch.device(model.cfg.device)
    reference = paper_api.build_balanced_paired_dataset(
        model.tokenizer,
        examples_per_group=1,
        seed=73,
    )
    means = cache_grouped_mean_activations(
        model,
        reference.abc_tokens.to(device),
        reference.groups,
        batch_size=8,
        min_group_size=1,
    )
    primary_nodes = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    hook_names = sorted(
        {
            f"blocks.{node.layer}.attn.hook_result"
            for node in primary_nodes
        }
    )
    tokens = target.ioi_tokens.to(device)
    model.set_use_attn_result(True)
    try:
        _, baseline_cache = model.run_with_cache(
            tokens,
            names_filter=hook_names,
        )
    finally:
        model.set_use_attn_result(False)

    pre_ablation_cache = {}
    _, knockout_cache = paper_api.knockout_nodes_with_cache(
        model,
        tokens,
        target.positions,
        target.groups,
        means,
        primary_nodes,
        names_filter=hook_names,
        before_cache=pre_ablation_cache,
    )
    row_groups = [None] * len(target)
    for key, indices in target.groups.items():
        for row in indices.tolist():
            row_groups[row] = key

    rows = torch.arange(len(target), device=device)
    for hook_name in hook_names:
        baseline = baseline_cache[hook_name]
        knocked_out = knockout_cache[hook_name]
        layer = int(hook_name.split(".")[1])
        if layer == 9:
            assert torch.allclose(
                pre_ablation_cache[hook_name],
                baseline,
                atol=1e-6,
                rtol=1e-6,
            )
        unchanged = torch.ones(
            baseline.shape[:3],
            dtype=torch.bool,
            device=device,
        )
        for node in primary_nodes:
            if node.layer != layer:
                continue
            end = target.positions["END"].to(device)
            expected = torch.stack(
                [
                    means.activations[hook_name][key][
                        target.positions["END"][row],
                        node.head,
                    ]
                    for row, key in enumerate(row_groups)
                ]
            ).to(device)
            assert torch.allclose(
                knocked_out[rows, end, node.head],
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
            unchanged[rows, end, node.head] = False
        assert torch.allclose(
            knocked_out[unchanged],
            pre_ablation_cache[hook_name][unchanged],
            atol=1e-6,
            rtol=1e-6,
        )
    assert model.cfg.use_attn_result is False


def test_real_gpt2_validation_smoke_has_finite_cached_results(
    real_model_and_dataset,
) -> None:
    model, target = real_model_and_dataset
    device = torch.device(model.cfg.device)
    reference = paper_api.build_balanced_paired_dataset(
        model.tokenizer,
        examples_per_group=1,
        seed=74,
    )
    means = cache_grouped_mean_activations(
        model,
        reference.abc_tokens.to(device),
        reference.groups,
        positions=reference.positions,
        batch_size=8,
        min_group_size=1,
    )
    evaluator = MeanAblationEvaluator(
        model=model,
        target_tokens=target.ioi_tokens.to(device),
        positions=target.positions,
        groups=target.groups,
        group_means=means,
        io_token_ids=target.io_token_ids,
        s_token_ids=target.s_token_ids,
        batch_size=2,
    )

    faithfulness = evaluate_faithfulness(evaluator)
    removed = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    completeness = evaluate_completeness(evaluator, removed)
    node = removed[0]
    minimality = evaluate_minimality(
        evaluator,
        node=node,
        context=HISTORICAL_MINIMALITY_CONTEXTS[node],
    )
    cached_before_repeat = evaluator.cached_evaluations
    assert evaluate_faithfulness(evaluator) == faithfulness

    values = (
        faithfulness.model.mean,
        faithfulness.circuit.mean,
        completeness.incompleteness,
        minimality.raw_effect,
        minimality.normalized_effect,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in values)
    assert evaluator.cached_evaluations == cached_before_repeat
