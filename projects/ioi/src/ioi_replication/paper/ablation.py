from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

import torch

from ioi_replication.paper.circuit import (
    HeadPositionNode,
    SemanticPosition,
)


ActivationName = Literal["result", "z"]
GroupKey = tuple[str, int]


@dataclass(frozen=True)
class GroupedActivationMeans:
    """Absolute-position pABC means indexed by exact dataset group."""

    activation: ActivationName
    activations: Mapping[str, Mapping[GroupKey, torch.Tensor]]
    sequence_length: int
    n_heads: int
    semantic_positions: Mapping[
        GroupKey,
        Mapping[SemanticPosition, int],
    ] = field(default_factory=dict)
    group_counts: Mapping[GroupKey, int] = field(default_factory=dict)
    min_group_size: int = 1
    reference_config: Mapping[str, Any] | None = None
    reference_sha256: str | None = None


def _validate_activation(activation: str) -> ActivationName:
    if activation not in {"result", "z"}:
        raise ValueError("activation must be 'result' or 'z'")
    return activation


def _group_for_each_example(
    groups: Mapping[GroupKey, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[GroupKey, ...]:
    if not groups:
        raise ValueError("groups must not be empty")
    membership = torch.zeros(batch_size, dtype=torch.long)
    group_for_row: list[GroupKey | None] = [None] * batch_size
    for key, raw_indices in groups.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not isinstance(key[1], int)
        ):
            raise ValueError("group keys must be (pattern, template_id)")
        indices = torch.as_tensor(raw_indices)
        if (
            indices.ndim != 1
            or indices.dtype == torch.bool
            or indices.is_floating_point()
        ):
            raise ValueError("group indices must be one-dimensional integers")
        indices = indices.long().cpu()
        if torch.any(indices < 0) or torch.any(indices >= batch_size):
            raise ValueError("group indices must be within the batch")
        membership[indices] += 1
        for row in indices.tolist():
            group_for_row[row] = key
    if torch.any(membership != 1):
        raise ValueError("each example must belong to exactly one reference group")
    return tuple(key for key in group_for_row if key is not None)


def compute_grouped_mean_activations(
    activation_cache: Mapping[str, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    *,
    activation: ActivationName = "result",
    positions: Mapping[Any, torch.Tensor] | None = None,
    min_group_size: int = 1,
    reference_config: Mapping[str, Any] | None = None,
    reference_sha256: str | None = None,
) -> GroupedActivationMeans:
    """Average pABC head activations within exact pattern/template groups.

    Sequence positions are never pooled: every output has shape
    ``[absolute_position, head, width]``.
    """
    activation = _validate_activation(activation)
    if not activation_cache:
        raise ValueError("activation_cache must not be empty")
    expected_suffix = f".attn.hook_{activation}"
    first = next(iter(activation_cache.values()))
    if first.ndim != 4:
        raise ValueError("head activations must have [batch, position, head, width]")
    batch_size, sequence_length, n_heads, trailing = first.shape
    _group_for_each_example(groups, batch_size=batch_size)
    group_counts = _validate_group_counts(
        groups,
        min_group_size=min_group_size,
    )

    grouped: dict[str, dict[GroupKey, torch.Tensor]] = {}
    for hook_name, values in activation_cache.items():
        if not hook_name.endswith(expected_suffix):
            raise ValueError(
                f"cache hook {hook_name!r} does not match activation {activation!r}"
            )
        if values.ndim != 4 or values.shape != (
            batch_size,
            sequence_length,
            n_heads,
            trailing,
        ):
            raise ValueError("all cached head activations must have equal shape")
        grouped[hook_name] = {}
        for key, raw_indices in groups.items():
            indices = torch.as_tensor(raw_indices, device=values.device).long()
            grouped[hook_name][key] = (
                values.index_select(0, indices).mean(dim=0).detach()
            )
    return GroupedActivationMeans(
        activation=activation,
        activations=grouped,
        sequence_length=sequence_length,
        n_heads=n_heads,
        semantic_positions=_reference_semantic_positions(
            positions,
            groups,
            batch_size=batch_size,
            sequence_length=sequence_length,
        ),
        group_counts=group_counts,
        min_group_size=min_group_size,
        reference_config=(
            dict(reference_config) if reference_config is not None else None
        ),
        reference_sha256=reference_sha256,
    )


def _validate_group_counts(
    groups: Mapping[GroupKey, torch.Tensor],
    *,
    min_group_size: int,
) -> dict[GroupKey, int]:
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    counts = {
        key: int(torch.as_tensor(indices).numel())
        for key, indices in groups.items()
    }
    insufficient = {
        key: count
        for key, count in counts.items()
        if count < min_group_size
    }
    if insufficient:
        details = ", ".join(
            f"{pattern}/{template_id}: {count}"
            for (pattern, template_id), count in sorted(insufficient.items())
        )
        raise ValueError(
            f"reference groups below minimum {min_group_size}: {details}"
        )
    return counts


@contextmanager
def _temporary_attn_result(
    model: Any,
    *,
    enabled: bool,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    cfg = getattr(model, "cfg", None)
    previous = bool(getattr(cfg, "use_attn_result", False))
    setter = getattr(model, "set_use_attn_result", None)
    if not previous:
        if setter is None:
            raise TypeError(
                "model must expose set_use_attn_result for result ablation"
            )
        setter(True)
    try:
        yield
    finally:
        if not previous:
            setter(False)


def cache_grouped_mean_activations(
    model: Any,
    reference_tokens: torch.Tensor,
    groups: Mapping[GroupKey, torch.Tensor],
    *,
    activation: ActivationName = "result",
    positions: Mapping[Any, torch.Tensor] | None = None,
    batch_size: int = 32,
    min_group_size: int = 1,
    reference_config: Mapping[str, Any] | None = None,
    reference_sha256: str | None = None,
) -> GroupedActivationMeans:
    """Stream pABC minibatches into exact-group, absolute-position means."""
    activation = _validate_activation(activation)
    if reference_tokens.ndim != 2:
        raise ValueError("reference_tokens must have shape [batch, position]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    reference_size, sequence_length = reference_tokens.shape
    _group_for_each_example(groups, batch_size=reference_size)
    group_counts = _validate_group_counts(
        groups,
        min_group_size=min_group_size,
    )
    hook_names = [
        f"blocks.{layer}.attn.hook_{activation}"
        for layer in range(int(model.cfg.n_layers))
    ]
    sums: dict[str, dict[GroupKey, torch.Tensor]] = {
        name: {} for name in hook_names
    }
    n_heads: int | None = None
    with _temporary_attn_result(model, enabled=activation == "result"):
        with torch.inference_mode():
            for start in range(0, reference_size, batch_size):
                stop = min(start + batch_size, reference_size)
                _, cache = model.run_with_cache(
                    reference_tokens[start:stop],
                    names_filter=hook_names,
                )
                missing = set(hook_names) - set(cache)
                if missing:
                    raise KeyError(
                        f"model cache did not contain hooks: {sorted(missing)}"
                    )
                for hook_name in hook_names:
                    values = cache[hook_name]
                    if values.ndim != 4:
                        raise ValueError(
                            "head activations must have "
                            "[batch, position, head, width]"
                        )
                    if values.shape[:2] != (stop - start, sequence_length):
                        raise ValueError(
                            "cached activations must align with reference tokens"
                        )
                    if n_heads is None:
                        n_heads = values.shape[2]
                    elif values.shape[2] != n_heads:
                        raise ValueError(
                            "all cached activations must share a head count"
                        )
                    for key, raw_indices in groups.items():
                        global_indices = torch.as_tensor(raw_indices).long()
                        selected = global_indices[
                            (global_indices >= start) & (global_indices < stop)
                        ]
                        if selected.numel() == 0:
                            continue
                        local_indices = (selected - start).to(values.device)
                        chunk_sum = (
                            values.index_select(0, local_indices)
                            .sum(dim=0)
                            .detach()
                            .cpu()
                        )
                        if key in sums[hook_name]:
                            sums[hook_name][key] += chunk_sum
                        else:
                            sums[hook_name][key] = chunk_sum
                del cache
    if n_heads is None:
        raise RuntimeError("no reference activations were cached")
    means = {
        hook_name: {
            key: grouped_sums[key] / group_counts[key]
            for key in groups
        }
        for hook_name, grouped_sums in sums.items()
    }
    return GroupedActivationMeans(
        activation=activation,
        activations=means,
        sequence_length=sequence_length,
        n_heads=n_heads,
        semantic_positions=_reference_semantic_positions(
            positions,
            groups,
            batch_size=reference_size,
            sequence_length=sequence_length,
        ),
        group_counts=group_counts,
        min_group_size=min_group_size,
        reference_config=(
            dict(reference_config) if reference_config is not None else None
        ),
        reference_sha256=reference_sha256,
    )


def _position_indices(
    positions: Mapping[Any, torch.Tensor],
    semantic_position: SemanticPosition,
    *,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    values = positions.get(semantic_position)
    if values is None:
        values = positions.get(semantic_position.value)
    if values is None:
        raise ValueError(
            f"positions has no {semantic_position.value!r} indices"
        )
    values = torch.as_tensor(values)
    if (
        values.ndim != 1
        or len(values) != batch_size
        or values.dtype == torch.bool
        or values.is_floating_point()
    ):
        raise ValueError(
            f"positions[{semantic_position.value!r}] must contain "
            "one integer per example"
        )
    values = values.long()
    if torch.any(values < 0) or torch.any(values >= sequence_length):
        raise ValueError("semantic positions must be within the sequence")
    return values


def _reference_semantic_positions(
    positions: Mapping[Any, torch.Tensor] | None,
    groups: Mapping[GroupKey, torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[GroupKey, dict[SemanticPosition, int]]:
    if positions is None:
        return {}
    result: dict[GroupKey, dict[SemanticPosition, int]] = {
        key: {} for key in groups
    }
    for semantic_position in SemanticPosition:
        if (
            semantic_position not in positions
            and semantic_position.value not in positions
        ):
            continue
        indices = _position_indices(
            positions,
            semantic_position,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        for key, raw_rows in groups.items():
            rows = torch.as_tensor(raw_rows).long()
            values = indices.index_select(0, rows)
            unique = values.unique()
            if unique.numel() != 1:
                raise ValueError(
                    "reference semantic positions must be constant "
                    f"within exact group {key!r}"
                )
            result[key][semantic_position] = int(unique.item())
    return result


def _validate_run(
    model: Any,
    target_tokens: torch.Tensor,
    positions: Mapping[Any, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    group_means: GroupedActivationMeans,
    nodes: Sequence[HeadPositionNode],
) -> tuple[tuple[GroupKey, ...], tuple[HeadPositionNode, ...]]:
    if target_tokens.ndim != 2:
        raise ValueError("target_tokens must have shape [batch, position]")
    batch_size, sequence_length = target_tokens.shape
    if sequence_length > group_means.sequence_length:
        raise ValueError(
            "reference activations must cover the target sequence length"
        )
    if int(model.cfg.n_heads) != group_means.n_heads:
        raise ValueError("group means head count must match the model")
    group_for_row = _group_for_each_example(groups, batch_size=batch_size)
    nodes = tuple(nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    for node in nodes:
        if node.layer >= int(model.cfg.n_layers):
            raise ValueError("node layer must fit the model")
        if node.head >= int(model.cfg.n_heads):
            raise ValueError("node head must fit the model")
        _position_indices(
            positions,
            node.position,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
    reference_groups = {
        key
        for values in group_means.activations.values()
        for key in values
    }
    missing = set(group_for_row) - reference_groups
    if missing:
        raise ValueError(
            f"target examples have no reference means for groups: {sorted(missing)}"
        )
    if group_means.semantic_positions:
        for node in nodes:
            missing_positions = {
                key
                for key in group_for_row
                if node.position
                not in group_means.semantic_positions.get(key, {})
            }
            if missing_positions:
                raise ValueError(
                    "reference semantic positions are missing "
                    f"{node.position.value!r} for groups: "
                    f"{sorted(missing_positions)}"
                )
    return group_for_row, nodes


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


def _run_ablation(
    model: Any,
    target_tokens: torch.Tensor,
    positions: Mapping[Any, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    group_means: GroupedActivationMeans,
    nodes: Sequence[HeadPositionNode],
    *,
    keep_mode: bool,
    before_hooks: Sequence[tuple[str, Any]] = (),
    additional_hooks: Sequence[tuple[str, Any]] = (),
) -> torch.Tensor:
    group_for_row, nodes = _validate_run(
        model,
        target_tokens,
        positions,
        groups,
        group_means,
        nodes,
    )
    nodes_by_layer: dict[int, list[HeadPositionNode]] = {}
    for node in nodes:
        nodes_by_layer.setdefault(node.layer, []).append(node)

    layers = (
        range(int(model.cfg.n_layers))
        if keep_mode
        else sorted(nodes_by_layer)
    )
    hooks: list[tuple[str, Any]] = []
    for layer in layers:
        hook_name = f"blocks.{layer}.attn.hook_{group_means.activation}"
        if hook_name not in group_means.activations:
            raise KeyError(f"group means do not contain {hook_name!r}")
        means_by_group = group_means.activations[hook_name]
        absent = set(group_for_row) - set(means_by_group)
        if absent:
            raise ValueError(
                f"{hook_name} has no reference means for groups: {sorted(absent)}"
            )

        def ablate(
            activation_value: torch.Tensor,
            hook: Any,
            *,
            layer_index: int = layer,
            grouped_means: Mapping[GroupKey, torch.Tensor] = means_by_group,
        ) -> torch.Tensor:
            if activation_value.ndim != 4:
                raise ValueError(
                    "head activations must have [batch, position, head, width]"
                )
            replacement = torch.stack(
                [
                    grouped_means[key][: activation_value.shape[1]]
                    for key in group_for_row
                ],
                dim=0,
            )
            if group_means.semantic_positions:
                rows = torch.arange(activation_value.shape[0])
                for semantic_position in SemanticPosition:
                    if not all(
                        semantic_position
                        in group_means.semantic_positions.get(key, {})
                        for key in group_for_row
                    ):
                        continue
                    target_indices = _position_indices(
                        positions,
                        semantic_position,
                        batch_size=activation_value.shape[0],
                        sequence_length=activation_value.shape[1],
                    )
                    reference_indices = torch.tensor(
                        [
                            group_means.semantic_positions[key][
                                semantic_position
                            ]
                            for key in group_for_row
                        ]
                    )
                    reference_values = torch.stack(
                        [
                            grouped_means[key][reference_index]
                            for key, reference_index in zip(
                                group_for_row,
                                reference_indices.tolist(),
                            )
                        ]
                    )
                    replacement[rows, target_indices] = reference_values
            replacement = replacement.to(
                device=activation_value.device,
                dtype=activation_value.dtype,
            )
            if replacement.shape != activation_value.shape:
                raise ValueError(
                    "group means shape must match target head activations"
                )
            mask = torch.full(
                activation_value.shape[:3],
                keep_mode,
                dtype=torch.bool,
                device=activation_value.device,
            )
            rows = torch.arange(
                activation_value.shape[0],
                device=activation_value.device,
            )
            for node in nodes_by_layer.get(layer_index, ()):
                semantic_indices = _position_indices(
                    positions,
                    node.position,
                    batch_size=activation_value.shape[0],
                    sequence_length=activation_value.shape[1],
                ).to(activation_value.device)
                mask[rows, semantic_indices, node.head] = not keep_mode
            return torch.where(
                mask[..., None],
                replacement,
                activation_value,
            )

        hooks.append((hook_name, ablate))

    with _temporary_attn_result(
        model,
        enabled=group_means.activation == "result",
    ):
        with torch.inference_mode():
            return model.run_with_hooks(
                target_tokens,
                fwd_hooks=[*before_hooks, *hooks, *additional_hooks],
            )


def knockout_nodes(
    model: Any,
    target_tokens: torch.Tensor,
    positions: Mapping[Any, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    group_means: GroupedActivationMeans,
    nodes: Sequence[HeadPositionNode],
) -> torch.Tensor:
    """Mean-ablate only the requested head/semantic-position nodes."""
    return _run_ablation(
        model,
        target_tokens,
        positions,
        groups,
        group_means,
        nodes,
        keep_mode=False,
    )


def knockout_nodes_with_cache(
    model: Any,
    target_tokens: torch.Tensor,
    positions: Mapping[Any, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    group_means: GroupedActivationMeans,
    nodes: Sequence[HeadPositionNode],
    *,
    names_filter: Sequence[str],
    before_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Mean-ablate nodes while capturing selected post-hook activations."""
    names = tuple(names_filter)
    if not names:
        raise ValueError("names_filter must not be empty")
    cache: dict[str, torch.Tensor] = {}
    before_hooks = (
        _capture_hooks(names, before_cache)
        if before_cache is not None
        else ()
    )
    logits = _run_ablation(
        model,
        target_tokens,
        positions,
        groups,
        group_means,
        nodes,
        keep_mode=False,
        before_hooks=before_hooks,
        additional_hooks=_capture_hooks(names, cache),
    )
    missing = set(names) - set(cache)
    if missing:
        raise RuntimeError(f"cache hooks did not fire: {sorted(missing)}")
    return logits, cache


def run_circuit(
    model: Any,
    target_tokens: torch.Tensor,
    positions: Mapping[Any, torch.Tensor],
    groups: Mapping[GroupKey, torch.Tensor],
    group_means: GroupedActivationMeans,
    nodes_to_keep: Sequence[HeadPositionNode],
) -> torch.Tensor:
    """Run circuit extraction, ablating every head-position except kept nodes."""
    return _run_ablation(
        model,
        target_tokens,
        positions,
        groups,
        group_means,
        nodes_to_keep,
        keep_mode=True,
    )
