from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ioi_replication.paper.ablation import (
    GroupedActivationMeans,
    _temporary_attn_result,
    knockout_nodes_with_cache,
)
from ioi_replication.paper.circuit import (
    PUBLISHED_CIRCUIT,
    CircuitClass,
)
from ioi_replication.paper.dataset import PairedIOIDataset
from ioi_replication.paper.metrics import logit_difference


@dataclass(frozen=True)
class Estimate:
    """Per-example values plus their mean and standard error."""

    values: torch.Tensor
    mean: float
    standard_error: float


@dataclass(frozen=True)
class DLADecomposition:
    baseline: torch.Tensor
    knockout: torch.Tensor
    delta_total: torch.Tensor
    delta_result_fixed_scale: torch.Tensor
    delta_scale: torch.Tensor


@dataclass(frozen=True)
class CompensationFraction:
    value: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    definition: str
    final_logit_delta_warning: str


@dataclass(frozen=True)
class BackupHeadMetrics:
    baseline_dla: Estimate
    knockout_dla: Estimate
    delta_dla: Estimate
    delta_result_fixed_scale: Estimate
    delta_scale: Estimate
    baseline_attention_io: Estimate
    knockout_attention_io: Estimate
    delta_attention_io: Estimate
    baseline_attention_s1: Estimate
    knockout_attention_s1: Estimate
    delta_attention_s1: Estimate
    baseline_attention_s2: Estimate
    knockout_attention_s2: Estimate
    delta_attention_s2: Estimate
    is_causally_downstream: bool

    @property
    def baseline_attention_s(self) -> Estimate:
        """Backward-compatible alias for the explicitly reported S1 position."""
        return self.baseline_attention_s1

    @property
    def knockout_attention_s(self) -> Estimate:
        return self.knockout_attention_s1

    @property
    def delta_attention_s(self) -> Estimate:
        return self.delta_attention_s1


@dataclass(frozen=True)
class BackupExperimentResult:
    baseline_logit_difference: Estimate
    knockout_logit_difference: Estimate
    delta_logit_difference: Estimate
    heads: Mapping[tuple[int, int], BackupHeadMetrics]
    removed_primary_baseline_dla: Mapping[tuple[int, int], Estimate]
    primary_substitution_dla: Mapping[tuple[int, int], Estimate]
    effectively_removed_primary_dla: Mapping[tuple[int, int], Estimate]
    compensation_fraction: CompensationFraction | None


def _estimate(values: torch.Tensor) -> Estimate:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("estimate values must be a non-empty vector")
    detached = values.detach()
    standard_error = (
        detached.float().std(unbiased=True) / detached.numel() ** 0.5
        if detached.numel() > 1
        else torch.zeros((), device=detached.device)
    )
    return Estimate(
        values=detached,
        mean=float(detached.float().mean().item()),
        standard_error=float(standard_error.item()),
    )


def _indices(
    values: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(values)
    if (
        values.ndim != 1
        or len(values) != batch_size
        or values.dtype == torch.bool
        or values.is_floating_point()
    ):
        raise ValueError(f"{name} must contain one integer per example")
    values = values.long()
    if torch.any(values < 0) or torch.any(values >= sequence_length):
        raise ValueError(f"{name} must be within the sequence")
    return values


def scaled_head_logit_attribution(
    result: torch.Tensor,
    ln_final_scale: torch.Tensor,
    unembed: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    *,
    head: int,
) -> torch.Tensor:
    """Return per-example IO−S DLA for one result head at semantic END."""
    if result.ndim != 4:
        raise ValueError("result must have [batch, position, head, d_model]")
    batch_size, sequence_length, n_heads, d_model = result.shape
    if not 0 <= head < n_heads:
        raise ValueError("head must fit the result head axis")
    if ln_final_scale.shape != (batch_size, sequence_length, 1):
        raise ValueError("ln_final_scale must have [batch, position, 1]")
    if unembed.ndim != 2 or unembed.shape[0] != d_model:
        raise ValueError("unembed must have [d_model, vocabulary]")
    end = _indices(
        end_positions,
        batch_size=batch_size,
        sequence_length=sequence_length,
        name="end_positions",
    ).to(result.device)
    io_ids = torch.as_tensor(io_token_ids)
    s_ids = torch.as_tensor(s_token_ids)
    if io_ids.shape != (batch_size,) or s_ids.shape != (batch_size,):
        raise ValueError("token ids must have one entry per example")
    io_ids = io_ids.long().to(unembed.device)
    s_ids = s_ids.long().to(unembed.device)
    if (
        torch.any(io_ids < 0)
        or torch.any(s_ids < 0)
        or torch.any(io_ids >= unembed.shape[1])
        or torch.any(s_ids >= unembed.shape[1])
    ):
        raise ValueError("token ids must fit the unembedding vocabulary")

    rows = torch.arange(batch_size, device=result.device)
    selected = result[rows, end, head]
    centered = selected - selected.mean(dim=-1, keepdim=True)
    scale = ln_final_scale.to(result.device)[rows, end]
    if torch.any(scale == 0):
        raise ValueError("ln_final_scale must be nonzero")
    direction = (
        unembed[:, io_ids].T - unembed[:, s_ids].T
    ).to(device=result.device, dtype=result.dtype)
    return ((centered / scale) * direction).sum(dim=-1)


def decompose_head_logit_attribution(
    baseline_result: torch.Tensor,
    knockout_result: torch.Tensor,
    baseline_scale: torch.Tensor,
    knockout_scale: torch.Tensor,
    unembed: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    *,
    head: int,
) -> DLADecomposition:
    """Split total DLA change into result change and final-LN rescaling."""
    baseline = scaled_head_logit_attribution(
        baseline_result,
        baseline_scale,
        unembed,
        io_token_ids,
        s_token_ids,
        end_positions,
        head=head,
    )
    knockout = scaled_head_logit_attribution(
        knockout_result,
        knockout_scale,
        unembed,
        io_token_ids,
        s_token_ids,
        end_positions,
        head=head,
    )
    knockout_at_baseline_scale = scaled_head_logit_attribution(
        knockout_result,
        baseline_scale,
        unembed,
        io_token_ids,
        s_token_ids,
        end_positions,
        head=head,
    )
    delta_result = knockout_at_baseline_scale - baseline
    delta_scale = knockout - knockout_at_baseline_scale
    return DLADecomposition(
        baseline=baseline,
        knockout=knockout,
        delta_total=knockout - baseline,
        delta_result_fixed_scale=delta_result,
        delta_scale=delta_scale,
    )


def compute_compensation_fraction(
    *,
    backup_delta_result: torch.Tensor,
    removed_primary_effective: torch.Tensor,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> CompensationFraction:
    """Estimate ratio-of-means compensation with paired bootstrap limits."""
    if (
        backup_delta_result.ndim != 1
        or removed_primary_effective.ndim != 1
        or backup_delta_result.shape != removed_primary_effective.shape
        or backup_delta_result.numel() < 2
    ):
        raise ValueError("fraction inputs must be equal vectors with at least two rows")
    if n_bootstrap <= 1:
        raise ValueError("n_bootstrap must exceed one")
    numerator = backup_delta_result.detach().float().cpu()
    denominator = removed_primary_effective.detach().float().cpu()
    denominator_mean = denominator.mean()
    if denominator_mean.abs() < 1e-8:
        raise ValueError("removed primary baseline mean is too close to zero")
    value = numerator.mean() / denominator_mean
    generator = torch.Generator().manual_seed(seed)
    sample_rows = torch.randint(
        numerator.numel(),
        (n_bootstrap, numerator.numel()),
        generator=generator,
    )
    bootstrap_denominator = denominator[sample_rows].mean(dim=1)
    valid = bootstrap_denominator.abs() >= 1e-8
    bootstrap = (
        numerator[sample_rows].mean(dim=1)[valid]
        / bootstrap_denominator[valid]
    )
    if bootstrap.numel() < 2:
        raise ValueError("bootstrap denominators are numerically unstable")
    return CompensationFraction(
        value=float(value.item()),
        standard_error=float(bootstrap.std(unbiased=True).item()),
        ci_lower=float(torch.quantile(bootstrap, 0.025).item()),
        ci_upper=float(torch.quantile(bootstrap, 0.975).item()),
        definition=(
            "ratio of means: summed posterior-backup Δresult at fixed "
            "baseline scale divided by effectively removed contribution "
            "(baseline primary DLA minus pABC substitution DLA, both at "
            "baseline scale)"
        ),
        final_logit_delta_warning=(
            "This diagnostic is not an identity with the final logit-difference "
            "delta; downstream nonlinear and residual effects remain."
        ),
    )


def _head_attention(
    pattern: torch.Tensor,
    *,
    head: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    if pattern.ndim != 4:
        raise ValueError("pattern must have [batch, head, query, key]")
    batch_size, n_heads, query_length, key_length = pattern.shape
    if not 0 <= head < n_heads:
        raise ValueError("head must fit the attention pattern")
    query = _indices(
        query_positions,
        batch_size=batch_size,
        sequence_length=query_length,
        name="query_positions",
    ).to(pattern.device)
    key = _indices(
        key_positions,
        batch_size=batch_size,
        sequence_length=key_length,
        name="key_positions",
    ).to(pattern.device)
    rows = torch.arange(batch_size, device=pattern.device)
    return pattern[rows, head, query, key]


def measure_backup_heads_from_cache(
    *,
    baseline_logits: torch.Tensor,
    knockout_logits: torch.Tensor,
    baseline_cache: Mapping[str, torch.Tensor],
    knockout_cache: Mapping[str, torch.Tensor],
    unembed: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    end_positions: torch.Tensor,
    io_positions: torch.Tensor,
    s1_positions: torch.Tensor,
    s2_positions: torch.Tensor,
    heads: Sequence[tuple[int, int]],
    causally_downstream_heads: Sequence[tuple[int, int]] | None = None,
    primary_heads: Sequence[tuple[int, int]] = (),
    same_layer_tolerance: float = 1e-6,
    bootstrap_seed: int = 0,
) -> BackupExperimentResult:
    """Summarize baseline and post-knockout signals for backup heads."""
    heads = tuple(heads)
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("heads must be a non-empty unique sequence")
    downstream = (
        set(heads)
        if causally_downstream_heads is None
        else set(causally_downstream_heads)
    )
    if not downstream.issubset(set(heads)):
        raise ValueError("causally_downstream_heads must be a subset of heads")
    batch_size = baseline_logits.shape[0]
    if knockout_logits.shape != baseline_logits.shape:
        raise ValueError("baseline and knockout logits must have equal shape")
    baseline_logit = logit_difference(
        baseline_logits,
        io_token_ids,
        s_token_ids,
        end_positions,
        reduction="none",
    )
    knockout_logit = logit_difference(
        knockout_logits,
        io_token_ids,
        s_token_ids,
        end_positions,
        reduction="none",
    )
    head_metrics: dict[tuple[int, int], BackupHeadMetrics] = {}
    for layer, head in heads:
        result_name = f"blocks.{layer}.attn.hook_result"
        pattern_name = f"blocks.{layer}.attn.hook_pattern"
        required = (
            (baseline_cache, result_name),
            (baseline_cache, pattern_name),
            (knockout_cache, result_name),
            (knockout_cache, pattern_name),
            (baseline_cache, "ln_final.hook_scale"),
            (knockout_cache, "ln_final.hook_scale"),
        )
        for cache, name in required:
            if name not in cache:
                raise KeyError(f"cache did not contain {name!r}")

        decomposition = decompose_head_logit_attribution(
            baseline_cache[result_name],
            knockout_cache[result_name],
            baseline_cache["ln_final.hook_scale"],
            knockout_cache["ln_final.hook_scale"],
            unembed,
            io_token_ids,
            s_token_ids,
            end_positions,
            head=head,
        )
        baseline_io = _head_attention(
            baseline_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=io_positions,
        )
        knockout_io = _head_attention(
            knockout_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=io_positions,
        )
        baseline_s1 = _head_attention(
            baseline_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s1_positions,
        )
        knockout_s1 = _head_attention(
            knockout_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s1_positions,
        )
        baseline_s2 = _head_attention(
            baseline_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s2_positions,
        )
        knockout_s2 = _head_attention(
            knockout_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s2_positions,
        )
        if any(
            values.shape != (batch_size,)
            for values in (
                decomposition.baseline,
                decomposition.knockout,
                baseline_io,
                knockout_io,
                baseline_s1,
                knockout_s1,
                baseline_s2,
                knockout_s2,
            )
        ):
            raise ValueError("all head measurements must have one value per example")
        is_downstream = (layer, head) in downstream
        if (
            not is_downstream
            and decomposition.delta_result_fixed_scale.abs().max().item()
            > same_layer_tolerance
        ):
            raise RuntimeError(
                f"non-downstream head {layer}.{head} changed result beyond "
                f"tolerance {same_layer_tolerance}"
            )
        head_metrics[(layer, head)] = BackupHeadMetrics(
            baseline_dla=_estimate(decomposition.baseline),
            knockout_dla=_estimate(decomposition.knockout),
            delta_dla=_estimate(decomposition.delta_total),
            delta_result_fixed_scale=_estimate(
                decomposition.delta_result_fixed_scale
            ),
            delta_scale=_estimate(decomposition.delta_scale),
            baseline_attention_io=_estimate(baseline_io),
            knockout_attention_io=_estimate(knockout_io),
            delta_attention_io=_estimate(knockout_io - baseline_io),
            baseline_attention_s1=_estimate(baseline_s1),
            knockout_attention_s1=_estimate(knockout_s1),
            delta_attention_s1=_estimate(knockout_s1 - baseline_s1),
            baseline_attention_s2=_estimate(baseline_s2),
            knockout_attention_s2=_estimate(knockout_s2),
            delta_attention_s2=_estimate(knockout_s2 - baseline_s2),
            is_causally_downstream=is_downstream,
        )

    primary_metrics: dict[tuple[int, int], Estimate] = {}
    substitution_metrics: dict[tuple[int, int], Estimate] = {}
    effective_metrics: dict[tuple[int, int], Estimate] = {}
    for layer, head in primary_heads:
        result_name = f"blocks.{layer}.attn.hook_result"
        if result_name not in baseline_cache or result_name not in knockout_cache:
            raise KeyError(f"cache did not contain {result_name!r}")
        baseline_primary = scaled_head_logit_attribution(
            baseline_cache[result_name],
            baseline_cache["ln_final.hook_scale"],
            unembed,
            io_token_ids,
            s_token_ids,
            end_positions,
            head=head,
        )
        substitution = scaled_head_logit_attribution(
            knockout_cache[result_name],
            baseline_cache["ln_final.hook_scale"],
            unembed,
            io_token_ids,
            s_token_ids,
            end_positions,
            head=head,
        )
        primary_metrics[(layer, head)] = _estimate(baseline_primary)
        substitution_metrics[(layer, head)] = _estimate(substitution)
        effective_metrics[(layer, head)] = _estimate(
            baseline_primary - substitution
        )

    compensation_fraction = None
    if primary_metrics:
        numerator = torch.stack(
            [
                metrics.delta_result_fixed_scale.values
                for head, metrics in head_metrics.items()
                if head in downstream
            ]
        ).sum(dim=0)
        denominator = torch.stack(
            [metrics.values for metrics in effective_metrics.values()]
        ).sum(dim=0)
        compensation_fraction = compute_compensation_fraction(
            backup_delta_result=numerator,
            removed_primary_effective=denominator,
            seed=bootstrap_seed,
        )

    return BackupExperimentResult(
        baseline_logit_difference=_estimate(baseline_logit),
        knockout_logit_difference=_estimate(knockout_logit),
        delta_logit_difference=_estimate(knockout_logit - baseline_logit),
        heads=head_metrics,
        removed_primary_baseline_dla=primary_metrics,
        primary_substitution_dla=substitution_metrics,
        effectively_removed_primary_dla=effective_metrics,
        compensation_fraction=compensation_fraction,
    )


def measure_backup_name_movers(
    *,
    model: Any,
    dataset: PairedIOIDataset,
    group_means: GroupedActivationMeans,
) -> BackupExperimentResult:
    """Knock out primary Name Movers and measure the eight backup heads."""
    if group_means.activation != "result":
        raise ValueError("backup experiment requires result-space group means")
    device = torch.device(model.cfg.device)
    tokens = dataset.ioi_tokens.to(device)
    backup_nodes = PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]
    primary_nodes = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    heads = tuple((node.layer, node.head) for node in backup_nodes)
    layers = sorted({layer for layer, _ in heads})
    cache_names = [
        name
        for layer in layers
        for name in (
            f"blocks.{layer}.attn.hook_result",
            f"blocks.{layer}.attn.hook_pattern",
        )
    ]
    cache_names.append("ln_final.hook_scale")

    with _temporary_attn_result(model, enabled=True):
        with torch.inference_mode():
            baseline_logits, baseline_cache = model.run_with_cache(
                tokens,
                names_filter=cache_names,
            )

    knockout_logits, knockout_cache = knockout_nodes_with_cache(
        model,
        tokens,
        dataset.positions,
        dataset.groups,
        group_means,
        primary_nodes,
        names_filter=cache_names,
    )
    downstream_heads = tuple(
        head
        for head in heads
        if head[0] > min(node.layer for node in primary_nodes)
    )

    return measure_backup_heads_from_cache(
        baseline_logits=baseline_logits,
        knockout_logits=knockout_logits,
        baseline_cache=baseline_cache,
        knockout_cache=knockout_cache,
        unembed=model.W_U,
        io_token_ids=dataset.io_token_ids.to(device),
        s_token_ids=dataset.s_token_ids.to(device),
        end_positions=dataset.positions["END"].to(device),
        io_positions=dataset.positions["IO"].to(device),
        s1_positions=dataset.positions["S1"].to(device),
        s2_positions=dataset.positions["S2"].to(device),
        heads=heads,
        causally_downstream_heads=downstream_heads,
        primary_heads=tuple(
            (node.layer, node.head) for node in primary_nodes
        ),
        bootstrap_seed=dataset.seed,
    )
