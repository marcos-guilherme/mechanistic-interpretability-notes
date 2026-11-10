"""Semantic activation patching and the historical IOI path-patching algorithm.

``paper_path_patch`` ports the four-pass procedure in the original
Easy-Transformer ``ioi_utils.path_patching``:

1. cache sender results on the new distribution;
2. cache original Q/K/V inputs (and optionally MLP outputs);
3. insert only the new sender results while freezing every other Q/K/V input,
   then capture the selected receiver activations before their freeze hooks;
4. patch those captured receiver activations into an otherwise original run.

This is intentionally different from TransformerLens 4.3
``get_act_patch_direct_path``.  The modern helper adds a source head's
clean-minus-corrupted residual write to one pre-Q/K/V fork at *all* token
positions while every other reader keeps the original source write, thereby
isolating that direct edge.  The paper algorithm recomputes the historical
multi-pass intervention, can leave MLPs as the Appendix B confounder, and
patches each example at its own semantic position.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import torch

from ioi_replication.paper.circuit import (
    HeadPositionNode,
    ReceiverComponent,
    SemanticPosition,
)


ActivationName = Literal["z", "result"]
Metric = Callable[[torch.Tensor], torch.Tensor]
PositionMap = Mapping[SemanticPosition | str, torch.Tensor]


def _as_positions(
    values: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    name: str,
) -> torch.Tensor:
    values = torch.as_tensor(values)
    if values.ndim != 1 or len(values) != batch_size:
        raise ValueError(f"{name} must have one entry per batch example")
    if values.dtype == torch.bool or values.is_floating_point():
        raise ValueError(f"{name} must contain integer indices")
    values = values.long()
    if torch.any(values < 0) or torch.any(values >= sequence_length):
        raise ValueError(f"{name} must be within the sequence dimension")
    return values


def _lookup_positions(
    positions: PositionMap,
    semantic_position: SemanticPosition,
    *,
    batch_size: int,
    sequence_length: int,
    name: str,
) -> torch.Tensor:
    value = positions.get(semantic_position)
    if value is None:
        value = positions.get(semantic_position.value)
    if value is None:
        raise ValueError(
            f"{name} has no indices for semantic position "
            f"{semantic_position.value!r}"
        )
    return _as_positions(
        value,
        batch_size=batch_size,
        sequence_length=sequence_length,
        name=f"{name}[{semantic_position.value!r}]",
    )


def patch_semantic_positions(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    source_positions: torch.Tensor | None = None,
    head: int | None = None,
) -> torch.Tensor:
    """Clone ``target`` and patch one independently indexed position per row.

    Activations are expected to be ``[batch, position, ...]``.  When ``head``
    is given, dimension 2 is interpreted as the head axis.
    """
    if target.ndim < 2 or source.ndim != target.ndim:
        raise ValueError("target and source must have equal rank of at least two")
    if target.shape[0] != source.shape[0]:
        raise ValueError("target and source must have the same batch size")
    if target.shape[2:] != source.shape[2:]:
        raise ValueError("target and source trailing activation shapes must match")

    batch_size = target.shape[0]
    target_index = _as_positions(
        target_positions,
        batch_size=batch_size,
        sequence_length=target.shape[1],
        name="target_positions",
    )
    source_index = _as_positions(
        target_positions if source_positions is None else source_positions,
        batch_size=batch_size,
        sequence_length=source.shape[1],
        name="source_positions",
    )

    if head is not None:
        if target.ndim != 4:
            raise ValueError("head patching requires rank four activations")
        if not 0 <= head < target.shape[2]:
            raise ValueError("head must be within the activation head dimension")

    patched = target.clone()
    target_rows = torch.arange(batch_size, device=target.device)
    source_rows = torch.arange(batch_size, device=source.device)
    target_index = target_index.to(target.device)
    source_index = source_index.to(source.device)
    if head is None:
        replacement = source[source_rows, source_index].to(target.device)
        patched[target_rows, target_index] = replacement
    else:
        replacement = source[source_rows, source_index, head].to(target.device)
        patched[target_rows, target_index, head] = replacement
    return patched


def _activation_hook_name(layer: int, activation: ActivationName) -> str:
    if activation not in {"z", "result"}:
        raise ValueError("activation must be 'z' or 'result'")
    return f"blocks.{layer}.attn.hook_{activation}"


def activation_patch_node(
    *,
    model: Any,
    target_tokens: torch.Tensor,
    source_cache: Mapping[str, torch.Tensor],
    metric: Metric,
    node: HeadPositionNode,
    target_positions: PositionMap,
    source_positions: PositionMap | None = None,
    activation: ActivationName = "result",
) -> torch.Tensor:
    """Patch one head-position node from ``source_cache`` into a target run."""
    if target_tokens.ndim != 2:
        raise ValueError("target_tokens must have shape [batch, position]")
    hook_name = _activation_hook_name(node.layer, activation)
    if hook_name not in source_cache:
        raise KeyError(f"source_cache does not contain {hook_name!r}")
    source = source_cache[hook_name]
    target_index = _lookup_positions(
        target_positions,
        node.position,
        batch_size=target_tokens.shape[0],
        sequence_length=target_tokens.shape[1],
        name="target_positions",
    )
    source_map = target_positions if source_positions is None else source_positions
    source_index = _lookup_positions(
        source_map,
        node.position,
        batch_size=source.shape[0],
        sequence_length=source.shape[1],
        name="source_positions",
    )

    def patch_hook(activation_value: torch.Tensor, hook: Any) -> torch.Tensor:
        return patch_semantic_positions(
            activation_value,
            source,
            target_positions=target_index,
            source_positions=source_index,
            head=node.head,
        )

    with _temporary_attn_result(model, enabled=activation == "result"):
        with torch.inference_mode():
            logits = model.run_with_hooks(
                target_tokens,
                fwd_hooks=[(hook_name, patch_hook)],
            )
    return torch.as_tensor(metric(logits))


@dataclass(frozen=True)
class ActivationPatchMatrix:
    """Layer × head × semantic-position activation-patching values."""

    values: torch.Tensor
    semantic_positions: tuple[SemanticPosition, ...]
    activation: ActivationName


def activation_patch_matrix(
    *,
    model: Any,
    target_tokens: torch.Tensor,
    source_cache: Mapping[str, torch.Tensor],
    metric: Metric,
    target_positions: PositionMap,
    source_positions: PositionMap | None = None,
    semantic_positions: Sequence[SemanticPosition | str] = tuple(
        SemanticPosition
    ),
    activation: ActivationName = "result",
) -> ActivationPatchMatrix:
    """Evaluate every layer/head at selected semantic positions.

    ``metric`` must return one scalar per intervention.  The resulting tensor
    has shape ``[n_layers, n_heads, n_semantic_positions]``.
    """
    positions = tuple(SemanticPosition(position) for position in semantic_positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("semantic_positions must be a non-empty unique sequence")
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    if n_layers > 12 or n_heads > 12:
        raise ValueError("HeadPositionNode currently targets GPT-2 Small dimensions")

    values: list[torch.Tensor] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            for position in positions:
                value = activation_patch_node(
                    model=model,
                    target_tokens=target_tokens,
                    source_cache=source_cache,
                    metric=metric,
                    node=HeadPositionNode(layer, head, position),
                    target_positions=target_positions,
                    source_positions=source_positions,
                    activation=activation,
                )
                if value.ndim != 0:
                    raise ValueError("metric must return a scalar for matrix sweeps")
                values.append(value.detach())
    matrix = torch.stack(values).reshape(n_layers, n_heads, len(positions))
    return ActivationPatchMatrix(matrix, positions, activation)


@dataclass(frozen=True)
class PathReceiver:
    """One Q/K/V receiver head at a semantic position."""

    layer: int
    head: int
    component: ReceiverComponent
    position: SemanticPosition

    def __post_init__(self) -> None:
        if not 0 <= self.layer < 12:
            raise ValueError("layer must be in [0, 12) for GPT-2 Small")
        if not 0 <= self.head < 12:
            raise ValueError("head must be in [0, 12) for GPT-2 Small")
        try:
            component = ReceiverComponent(self.component)
        except ValueError as error:
            raise ValueError(f"unknown receiver component: {self.component!r}") from error
        try:
            position = SemanticPosition(self.position)
        except ValueError as error:
            raise ValueError(f"unknown semantic position: {self.position!r}") from error
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class PathPatchResult:
    """Metric endpoints for one historical path-patching intervention."""

    baseline: torch.Tensor
    patched: torch.Tensor

    @property
    def effect(self) -> torch.Tensor:
        return self.patched - self.baseline


@contextmanager
def _temporary_attn_result(model: Any, *, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    cfg = getattr(model, "cfg", None)
    was_enabled = bool(getattr(cfg, "use_attn_result", False))
    setter = getattr(model, "set_use_attn_result", None)
    if not was_enabled:
        if setter is None:
            raise TypeError("model must expose set_use_attn_result for result patching")
        setter(True)
    try:
        yield
    finally:
        if not was_enabled:
            setter(False)


def _cache_tensor(cache: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        return cache[name]
    except KeyError as error:
        raise KeyError(f"model cache did not contain required hook {name!r}") from error


def _group_senders(
    senders: Sequence[HeadPositionNode],
) -> dict[str, list[HeadPositionNode]]:
    grouped: dict[str, list[HeadPositionNode]] = {}
    for sender in senders:
        grouped.setdefault(
            _activation_hook_name(sender.layer, "result"),
            [],
        ).append(sender)
    return grouped


def _group_receivers(
    receivers: Sequence[PathReceiver],
) -> dict[str, list[PathReceiver]]:
    grouped: dict[str, list[PathReceiver]] = {}
    for receiver in receivers:
        name = (
            f"blocks.{receiver.layer}.attn."
            f"hook_{receiver.component.value}"
        )
        grouped.setdefault(name, []).append(receiver)
    return grouped


def _validate_path_request(
    model: Any,
    original_tokens: torch.Tensor,
    new_tokens: torch.Tensor,
    original_positions: PositionMap,
    new_positions: PositionMap,
    senders: Sequence[HeadPositionNode],
    receivers: Sequence[PathReceiver],
) -> None:
    if not senders:
        raise ValueError("at least one sender is required")
    if not receivers:
        raise ValueError("at least one receiver is required")
    if len(set(senders)) != len(senders):
        raise ValueError("senders must be unique")
    if len(set(receivers)) != len(receivers):
        raise ValueError("receivers must be unique")
    if original_tokens.ndim != 2 or new_tokens.ndim != 2:
        raise ValueError("original_tokens and new_tokens must be [batch, position]")
    if original_tokens.shape[0] != new_tokens.shape[0]:
        raise ValueError("original_tokens and new_tokens must share a batch size")
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    for endpoint in (*senders, *receivers):
        if endpoint.layer >= n_layers or endpoint.head >= n_heads:
            raise ValueError("sender and receiver coordinates must fit the model")
    last_sender_layer = max(sender.layer for sender in senders)
    if any(receiver.layer <= last_sender_layer for receiver in receivers):
        raise ValueError("every receiver must be in a later layer than every sender")

    for sender in senders:
        _lookup_positions(
            original_positions,
            sender.position,
            batch_size=original_tokens.shape[0],
            sequence_length=original_tokens.shape[1],
            name="original_positions",
        )
        _lookup_positions(
            new_positions,
            sender.position,
            batch_size=new_tokens.shape[0],
            sequence_length=new_tokens.shape[1],
            name="new_positions",
        )
    for receiver in receivers:
        _lookup_positions(
            original_positions,
            receiver.position,
            batch_size=original_tokens.shape[0],
            sequence_length=original_tokens.shape[1],
            name="original_positions",
        )


def paper_path_patch(
    model: Any,
    original_tokens: torch.Tensor,
    new_tokens: torch.Tensor,
    original_positions: PositionMap,
    new_positions: PositionMap,
    *,
    senders: Sequence[HeadPositionNode],
    receivers: Sequence[PathReceiver],
    metric: Metric,
    freeze_mlps: bool = False,
    allow_internal_interactions: bool = False,
) -> PathPatchResult:
    """Run the paper's four-pass sender-to-receiver path patch.

    Q/K/V activations other than the receiver value being captured are frozen
    to the original run.  As in the paper and historical implementation, MLPs
    are recomputed by default and are therefore possible confounders; pass
    ``freeze_mlps=True`` for the stricter diagnostic.  Setting
    ``allow_internal_interactions=True`` reproduces the historical
    ``have_internal_interactions`` option by not freezing receiver hook names
    during pass C.

    This implementation accepts attention-head senders only.  Every receiver
    must be in a layer later than every sender, which keeps the four-pass
    causal ordering unambiguous; MLP senders and interleaved sender/receiver
    layers are intentionally unsupported.
    """
    senders = tuple(senders)
    receivers = tuple(receivers)
    _validate_path_request(
        model,
        original_tokens,
        new_tokens,
        original_positions,
        new_positions,
        senders,
        receivers,
    )
    sender_groups = _group_senders(senders)
    receiver_groups = _group_receivers(receivers)
    qkv_names = [
        f"blocks.{layer}.attn.hook_{component}"
        for layer in range(int(model.cfg.n_layers))
        for component in ("q", "k", "v")
    ]
    original_names = list(qkv_names)
    if freeze_mlps:
        original_names.extend(
            f"blocks.{layer}.hook_mlp_out"
            for layer in range(int(model.cfg.n_layers))
        )

    with _temporary_attn_result(model):
        with torch.inference_mode():
            _, new_cache = model.run_with_cache(
                new_tokens,
                names_filter=list(sender_groups),
            )
            original_logits, original_cache = model.run_with_cache(
                original_tokens,
                names_filter=original_names,
            )

            receiver_cache: dict[str, torch.Tensor] = {}
            capture_hooks = []
            for hook_name in receiver_groups:
                def capture(
                    activation: torch.Tensor,
                    hook: Any,
                    *,
                    name: str = hook_name,
                ) -> torch.Tensor:
                    receiver_cache[name] = activation.detach().clone()
                    return activation

                capture_hooks.append((hook_name, capture))

            freeze_hooks = []
            for hook_name in qkv_names:
                if allow_internal_interactions and hook_name in receiver_groups:
                    continue
                original_value = _cache_tensor(original_cache, hook_name)

                def freeze(
                    activation: torch.Tensor,
                    hook: Any,
                    *,
                    value: torch.Tensor = original_value,
                ) -> torch.Tensor:
                    return value.to(activation.device)

                freeze_hooks.append((hook_name, freeze))

            if freeze_mlps:
                for layer in range(int(model.cfg.n_layers)):
                    hook_name = f"blocks.{layer}.hook_mlp_out"
                    original_value = _cache_tensor(original_cache, hook_name)

                    def freeze_mlp(
                        activation: torch.Tensor,
                        hook: Any,
                        *,
                        value: torch.Tensor = original_value,
                    ) -> torch.Tensor:
                        return value.to(activation.device)

                    freeze_hooks.append((hook_name, freeze_mlp))

            sender_hooks = []
            for hook_name, hook_senders in sender_groups.items():
                source = _cache_tensor(new_cache, hook_name)

                def patch_senders(
                    activation: torch.Tensor,
                    hook: Any,
                    *,
                    nodes: Sequence[HeadPositionNode] = tuple(hook_senders),
                    source_value: torch.Tensor = source,
                ) -> torch.Tensor:
                    patched = activation
                    for node in nodes:
                        patched = patch_semantic_positions(
                            patched,
                            source_value,
                            target_positions=_lookup_positions(
                                original_positions,
                                node.position,
                                batch_size=original_tokens.shape[0],
                                sequence_length=original_tokens.shape[1],
                                name="original_positions",
                            ),
                            source_positions=_lookup_positions(
                                new_positions,
                                node.position,
                                batch_size=new_tokens.shape[0],
                                sequence_length=new_tokens.shape[1],
                                name="new_positions",
                            ),
                            head=node.head,
                        )
                    return patched

                sender_hooks.append((hook_name, patch_senders))

            model.run_with_hooks(
                original_tokens,
                fwd_hooks=[*capture_hooks, *freeze_hooks, *sender_hooks],
            )
            missing = set(receiver_groups) - set(receiver_cache)
            if missing:
                raise RuntimeError(
                    f"receiver hooks did not fire: {sorted(missing)}"
                )

            receiver_patch_hooks = []
            for hook_name, hook_receivers in receiver_groups.items():
                source = receiver_cache[hook_name]

                def patch_receivers(
                    activation: torch.Tensor,
                    hook: Any,
                    *,
                    endpoints: Sequence[PathReceiver] = tuple(hook_receivers),
                    source_value: torch.Tensor = source,
                ) -> torch.Tensor:
                    patched = activation
                    for receiver in endpoints:
                        positions = _lookup_positions(
                            original_positions,
                            receiver.position,
                            batch_size=original_tokens.shape[0],
                            sequence_length=original_tokens.shape[1],
                            name="original_positions",
                        )
                        patched = patch_semantic_positions(
                            patched,
                            source_value,
                            target_positions=positions,
                            source_positions=positions,
                            head=receiver.head,
                        )
                    return patched

                receiver_patch_hooks.append((hook_name, patch_receivers))

            patched_logits = model.run_with_hooks(
                original_tokens,
                fwd_hooks=receiver_patch_hooks,
            )
    baseline = torch.as_tensor(metric(original_logits)).detach()
    patched = torch.as_tensor(metric(patched_logits)).detach()
    if baseline.shape != patched.shape:
        raise ValueError("metric must return the same shape for both endpoints")
    return PathPatchResult(baseline=baseline, patched=patched)

