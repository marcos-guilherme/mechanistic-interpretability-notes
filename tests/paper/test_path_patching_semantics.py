from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch

from ioi_replication.paper.circuit import HeadPositionNode, SemanticPosition
from ioi_replication.paper.patching import PathReceiver, paper_path_patch


@dataclass
class _Event:
    name: str
    before: torch.Tensor
    after: torch.Tensor


@dataclass
class _Cfg:
    n_layers: int = 4
    n_heads: int = 3
    device: str = "cpu"
    use_attn_result: bool = False


class _HookModel:
    """Small deterministic residual network with real ordered hook execution."""

    def __init__(self, *, use_attn_result: bool = False) -> None:
        self.cfg = _Cfg(use_attn_result=use_attn_result)
        self.hook_runs: list[list[_Event]] = []
        self.cache_runs: list[dict[str, torch.Tensor]] = []

    def set_use_attn_result(self, enabled: bool) -> None:
        self.cfg.use_attn_result = enabled

    def run_with_cache(self, tokens, names_filter):
        logits, activations = self._forward(tokens, ())
        if callable(names_filter):
            cache = {
                name: value
                for name, value in activations.items()
                if names_filter(name)
            }
        else:
            wanted = set(names_filter)
            cache = {
                name: value
                for name, value in activations.items()
                if name in wanted
            }
        self.cache_runs.append(cache)
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks):
        events: list[_Event] = []
        self.hook_runs.append(events)
        logits, _ = self._forward(tokens, tuple(fwd_hooks), events)
        return logits

    def _forward(
        self,
        tokens: torch.Tensor,
        hooks: tuple[tuple[str, Callable], ...],
        events: list[_Event] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, positions = tokens.shape
        state = tokens.float()
        activations: dict[str, torch.Tensor] = {}
        heads = torch.arange(self.cfg.n_heads).view(1, 1, -1, 1).float()

        def apply(name: str, value: torch.Tensor) -> torch.Tensor:
            for hook_name, hook_fn in hooks:
                if hook_name != name:
                    continue
                before = value.detach().clone()
                value = hook_fn(value, SimpleNamespace(name=name))
                if events is not None:
                    events.append(
                        _Event(name, before, value.detach().clone())
                    )
            activations[name] = value.detach().clone()
            return value

        for layer in range(self.cfg.n_layers):
            qkv = {}
            for component_index, component in enumerate(("q", "k", "v"), 1):
                name = f"blocks.{layer}.attn.hook_{component}"
                value = (
                    state[:, :, None, None]
                    + 100 * layer
                    + 10 * component_index
                    + heads
                )
                qkv[component] = apply(name, value)

            result_name = f"blocks.{layer}.attn.hook_result"
            result = (
                state[:, :, None, None]
                + 1000 * layer
                + heads
            )
            result = apply(result_name, result)

            mlp_name = f"blocks.{layer}.hook_mlp_out"
            mlp = apply(
                mlp_name,
                state[:, :, None] + 10000 * layer + 7,
            )
            state = (
                state
                + result.sum(dim=(2, 3))
                + 0.01
                * sum(value.sum(dim=(2, 3)) for value in qkv.values())
                + mlp[..., 0]
            )

        logits = torch.stack((state, -state), dim=-1)
        return logits, activations


ORIGINAL = torch.zeros(2, 4, dtype=torch.long)
NEW = torch.tensor(
    [
        [11, 12, 13, 14],
        [21, 22, 23, 24],
    ],
    dtype=torch.long,
)
ORIGINAL_POSITIONS = {
    SemanticPosition.S2: torch.tensor([1, 2]),
    SemanticPosition.END: torch.tensor([3, 3]),
}
NEW_POSITIONS = {
    SemanticPosition.S2: torch.tensor([2, 0]),
    SemanticPosition.END: torch.tensor([3, 2]),
}


def _metric(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., 0].sum()


def _run(
    *,
    senders: tuple[HeadPositionNode, ...] = (
        HeadPositionNode(0, 0, SemanticPosition.S2),
    ),
    receivers: tuple[PathReceiver, ...] = (
        PathReceiver(3, 1, "v", SemanticPosition.S2),
    ),
    freeze_mlps: bool = False,
    allow_internal_interactions: bool = False,
    use_attn_result: bool = False,
):
    model = _HookModel(use_attn_result=use_attn_result)
    result = paper_path_patch(
        model,
        ORIGINAL,
        NEW,
        ORIGINAL_POSITIONS,
        NEW_POSITIONS,
        senders=senders,
        receivers=receivers,
        metric=_metric,
        freeze_mlps=freeze_mlps,
        allow_internal_interactions=allow_internal_interactions,
    )
    assert len(model.cache_runs) == 2
    assert len(model.hook_runs) == 2
    return model, result


def _events_for(events: list[_Event], name: str) -> list[_Event]:
    return [event for event in events if event.name == name]


def test_pass_c_captures_receiver_before_freeze_and_freezes_other_qkv() -> None:
    model, result = _run()
    pass_c = model.hook_runs[0]
    original_cache = model.cache_runs[1]
    receiver_name = "blocks.3.attn.hook_v"

    receiver_events = _events_for(pass_c, receiver_name)
    assert len(receiver_events) == 2
    assert torch.equal(receiver_events[0].after, receiver_events[0].before)
    assert torch.equal(receiver_events[1].before, receiver_events[0].after)
    assert torch.equal(receiver_events[1].after, original_cache[receiver_name])
    assert not torch.equal(receiver_events[0].after, receiver_events[1].after)

    non_receiver_name = "blocks.2.attn.hook_k"
    non_receiver_events = _events_for(pass_c, non_receiver_name)
    assert len(non_receiver_events) == 1
    assert torch.equal(
        non_receiver_events[0].after,
        original_cache[non_receiver_name],
    )
    assert torch.isfinite(result.effect)


def test_pass_d_patches_only_receiver_head_and_semantic_positions() -> None:
    model, _ = _run()
    pass_d = model.hook_runs[1]
    event = _events_for(pass_d, "blocks.3.attn.hook_v")[0]
    changed = (event.after != event.before).squeeze(-1)
    expected = torch.zeros_like(changed)
    expected[
        torch.arange(2),
        ORIGINAL_POSITIONS[SemanticPosition.S2],
        1,
    ] = True

    assert torch.equal(changed, expected)


def test_multiple_senders_and_receivers_patch_every_requested_endpoint() -> None:
    senders = (
        HeadPositionNode(1, 0, SemanticPosition.S2),
        HeadPositionNode(1, 2, SemanticPosition.END),
    )
    receivers = (
        PathReceiver(3, 1, "v", SemanticPosition.S2),
        PathReceiver(3, 2, "v", SemanticPosition.END),
    )
    model, _ = _run(senders=senders, receivers=receivers)
    pass_c, pass_d = model.hook_runs

    sender_events = _events_for(
        pass_c,
        "blocks.1.attn.hook_result",
    )
    assert len(sender_events) == 1
    sender_changed = (sender_events[0].after != sender_events[0].before).squeeze(-1)
    sender_expected = torch.zeros_like(sender_changed)
    for sender in senders:
        sender_expected[
            torch.arange(2),
            ORIGINAL_POSITIONS[sender.position],
            sender.head,
        ] = True
    assert torch.equal(sender_changed, sender_expected)

    receiver_events = _events_for(pass_d, "blocks.3.attn.hook_v")
    assert len(receiver_events) == 1
    receiver_changed = (
        receiver_events[0].after != receiver_events[0].before
    ).squeeze(-1)
    receiver_expected = torch.zeros_like(receiver_changed)
    for receiver in receivers:
        receiver_expected[
            torch.arange(2),
            ORIGINAL_POSITIONS[receiver.position],
            receiver.head,
        ] = True
    assert torch.equal(receiver_changed, receiver_expected)


def test_freeze_mlps_true_freezes_every_mlp_while_false_recomputes() -> None:
    frozen_model, _ = _run(freeze_mlps=True)
    recomputed_model, _ = _run(freeze_mlps=False)
    frozen_pass_c = frozen_model.hook_runs[0]
    original_cache = frozen_model.cache_runs[1]

    for layer in range(frozen_model.cfg.n_layers):
        name = f"blocks.{layer}.hook_mlp_out"
        events = _events_for(frozen_pass_c, name)
        assert len(events) == 1
        assert torch.equal(events[0].after, original_cache[name])
        assert _events_for(recomputed_model.hook_runs[0], name) == []


def test_allow_internal_interactions_skips_only_receiver_hook_freeze() -> None:
    model, _ = _run(
        allow_internal_interactions=True,
        use_attn_result=True,
    )
    pass_c = model.hook_runs[0]
    receiver_events = _events_for(pass_c, "blocks.3.attn.hook_v")
    non_receiver_events = _events_for(pass_c, "blocks.3.attn.hook_k")

    assert len(receiver_events) == 1
    assert torch.equal(receiver_events[0].after, receiver_events[0].before)
    assert len(non_receiver_events) == 1
    assert torch.equal(
        non_receiver_events[0].after,
        model.cache_runs[1]["blocks.3.attn.hook_k"],
    )
    assert model.cfg.use_attn_result is True
