from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ioi_replication.paper.ablation import (
    GroupedActivationMeans,
    _run_ablation,
    _temporary_attn_result,
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
class BackupHeadMetrics:
    baseline_dla: Estimate
    knockout_dla: Estimate
    delta_dla: Estimate
    baseline_attention_io: Estimate
    knockout_attention_io: Estimate
    delta_attention_io: Estimate
    baseline_attention_s: Estimate
    knockout_attention_s: Estimate
    delta_attention_s: Estimate


@dataclass(frozen=True)
class BackupExperimentResult:
    baseline_logit_difference: Estimate
    knockout_logit_difference: Estimate
    delta_logit_difference: Estimate
    heads: Mapping[tuple[int, int], BackupHeadMetrics]


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
    s_positions: torch.Tensor,
    heads: Sequence[tuple[int, int]],
) -> BackupExperimentResult:
    """Summarize baseline and post-knockout signals for backup heads."""
    heads = tuple(heads)
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("heads must be a non-empty unique sequence")
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

        baseline_dla = scaled_head_logit_attribution(
            baseline_cache[result_name],
            baseline_cache["ln_final.hook_scale"],
            unembed,
            io_token_ids,
            s_token_ids,
            end_positions,
            head=head,
        )
        knockout_dla = scaled_head_logit_attribution(
            knockout_cache[result_name],
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
        baseline_s = _head_attention(
            baseline_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s_positions,
        )
        knockout_s = _head_attention(
            knockout_cache[pattern_name],
            head=head,
            query_positions=end_positions,
            key_positions=s_positions,
        )
        if any(
            values.shape != (batch_size,)
            for values in (
                baseline_dla,
                knockout_dla,
                baseline_io,
                knockout_io,
                baseline_s,
                knockout_s,
            )
        ):
            raise ValueError("all head measurements must have one value per example")
        head_metrics[(layer, head)] = BackupHeadMetrics(
            baseline_dla=_estimate(baseline_dla),
            knockout_dla=_estimate(knockout_dla),
            delta_dla=_estimate(knockout_dla - baseline_dla),
            baseline_attention_io=_estimate(baseline_io),
            knockout_attention_io=_estimate(knockout_io),
            delta_attention_io=_estimate(knockout_io - baseline_io),
            baseline_attention_s=_estimate(baseline_s),
            knockout_attention_s=_estimate(knockout_s),
            delta_attention_s=_estimate(knockout_s - baseline_s),
        )

    return BackupExperimentResult(
        baseline_logit_difference=_estimate(baseline_logit),
        knockout_logit_difference=_estimate(knockout_logit),
        delta_logit_difference=_estimate(knockout_logit - baseline_logit),
        heads=head_metrics,
    )


def _capture_hooks(
    names: Sequence[str],
    destination: dict[str, torch.Tensor],
) -> list[tuple[str, Any]]:
    hooks: list[tuple[str, Any]] = []
    for hook_name in names:
        def capture(
            value: torch.Tensor,
            hook: Any,
            *,
            name: str = hook_name,
        ) -> torch.Tensor:
            destination[name] = value.detach().clone()
            return value

        hooks.append((hook_name, capture))
    return hooks


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

    knockout_cache: dict[str, torch.Tensor] = {}
    knockout_logits = _run_ablation(
        model,
        tokens,
        dataset.positions,
        dataset.groups,
        group_means,
        primary_nodes,
        keep_mode=False,
        additional_hooks=_capture_hooks(cache_names, knockout_cache),
    )
    missing = set(cache_names) - set(knockout_cache)
    if missing:
        raise RuntimeError(f"knockout cache hooks did not fire: {sorted(missing)}")

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
        s_positions=dataset.positions["S1"].to(device),
        heads=heads,
    )
