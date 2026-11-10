from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from ioi_replication.paper.ablation import (
    GroupedActivationMeans,
    cache_grouped_mean_activations,
    compute_grouped_mean_activations,
    knockout_nodes,
    run_circuit,
)
from ioi_replication.paper.circuit import HeadPositionNode, SemanticPosition


GROUPS = {
    ("ABBA", 0): torch.tensor([0, 2]),
    ("BABA", 0): torch.tensor([1, 3]),
}


def test_group_means_are_conditioned_on_exact_pattern_and_template() -> None:
    values = torch.zeros(4, 3, 2, 1, dtype=torch.float64)
    values[GROUPS[("ABBA", 0)]] = 100.0
    values[GROUPS[("BABA", 0)]] = 10.0

    means = compute_grouped_mean_activations(
        {"blocks.0.attn.hook_result": values},
        GROUPS,
        activation="result",
    )

    by_group = means.activations["blocks.0.attn.hook_result"]
    assert torch.equal(
        by_group[("ABBA", 0)],
        torch.full((3, 2, 1), 100.0, dtype=torch.float64),
    )
    assert torch.equal(
        by_group[("BABA", 0)],
        torch.full((3, 2, 1), 10.0, dtype=torch.float64),
    )
    assert by_group[("ABBA", 0)].dtype is values.dtype
    assert by_group[("ABBA", 0)].device == values.device


@pytest.mark.parametrize(
    "groups, message",
    [
        ({("ABBA", 0): torch.tensor([0])}, "exactly one"),
        (
            {
                ("ABBA", 0): torch.tensor([0, 1]),
                ("BABA", 0): torch.tensor([1]),
            },
            "exactly one",
        ),
        ({("ABBA", 0): torch.tensor([0, 2])}, "within"),
    ],
)
def test_group_means_validate_complete_nonoverlapping_membership(
    groups: dict[tuple[str, int], torch.Tensor],
    message: str,
) -> None:
    cache = {"blocks.0.attn.hook_result": torch.zeros(2, 3, 2, 1)}

    with pytest.raises(ValueError, match=message):
        compute_grouped_mean_activations(cache, groups, activation="result")


def test_group_means_record_reference_metadata_and_enforce_minimum_size() -> None:
    cache = {
        "blocks.0.attn.hook_result": torch.arange(
            4 * 3 * 2,
            dtype=torch.float64,
        ).reshape(4, 3, 2, 1)
    }

    means = compute_grouped_mean_activations(
        cache,
        GROUPS,
        activation="result",
        min_group_size=2,
        reference_config={"seed": 17, "kind": "balanced_pABC"},
        reference_sha256="a" * 64,
    )

    assert means.group_counts == {("ABBA", 0): 2, ("BABA", 0): 2}
    assert means.min_group_size == 2
    assert means.reference_config == {"seed": 17, "kind": "balanced_pABC"}
    assert means.reference_sha256 == "a" * 64

    with pytest.raises(ValueError, match=r"BABA.*2.*3|minimum.*3"):
        compute_grouped_mean_activations(
            cache,
            GROUPS,
            activation="result",
            min_group_size=3,
        )


class _StreamingCacheModel:
    def __init__(self) -> None:
        self.cfg = _Cfg()
        self.max_batch_seen = 0

    def set_use_attn_result(self, enabled: bool) -> None:
        self.cfg.use_attn_result = enabled

    def run_with_cache(self, tokens, names_filter):
        self.max_batch_seen = max(self.max_batch_seen, len(tokens))
        heads = torch.arange(self.cfg.n_heads, dtype=torch.float64).view(
            1, 1, -1, 1
        )
        cache = {}
        for layer in range(self.cfg.n_layers):
            name = f"blocks.{layer}.attn.hook_result"
            if name in names_filter:
                cache[name] = (
                    tokens.to(torch.float64)[:, :, None, None]
                    + 100 * layer
                    + heads
                )
        return torch.zeros(len(tokens), tokens.shape[1], 1), cache


def test_streaming_group_means_equal_small_in_memory_computation() -> None:
    tokens = torch.arange(18).reshape(6, 3)
    groups = {
        ("ABBA", 0): torch.tensor([0, 2, 4]),
        ("BABA", 0): torch.tensor([1, 3, 5]),
    }
    full_model = _StreamingCacheModel()
    _, full_cache = full_model.run_with_cache(
        tokens,
        [
            "blocks.0.attn.hook_result",
            "blocks.1.attn.hook_result",
        ],
    )
    expected = compute_grouped_mean_activations(
        full_cache,
        groups,
        activation="result",
        min_group_size=3,
    )
    streaming_model = _StreamingCacheModel()

    actual = cache_grouped_mean_activations(
        streaming_model,
        tokens,
        groups,
        activation="result",
        batch_size=2,
        min_group_size=3,
        reference_config={"n": 6},
        reference_sha256="b" * 64,
    )

    assert streaming_model.max_batch_seen == 2
    assert actual.group_counts == expected.group_counts
    assert actual.reference_config == {"n": 6}
    assert actual.reference_sha256 == "b" * 64
    for hook_name in expected.activations:
        for key in groups:
            assert torch.equal(
                actual.activations[hook_name][key],
                expected.activations[hook_name][key],
            )


@dataclass
class _Cfg:
    n_layers: int = 2
    n_heads: int = 2
    use_attn_result: bool = False
    device: str = "cpu"


class _AblationModel:
    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        use_attn_result: bool = False,
        fail: bool = False,
        activation: str = "result",
    ) -> None:
        self.cfg = _Cfg(use_attn_result=use_attn_result)
        self.dtype = dtype
        self.fail = fail
        self.activation = activation
        self.events: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_use_attn_result(self, enabled: bool) -> None:
        self.cfg.use_attn_result = enabled

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        fwd_hooks: list[tuple[str, Callable]],
    ) -> torch.Tensor:
        batch, sequence = tokens.shape
        state = torch.zeros(
            batch,
            sequence,
            self.cfg.n_heads,
            1,
            dtype=self.dtype,
            device=tokens.device,
        )
        for layer in range(self.cfg.n_layers):
            name = f"blocks.{layer}.attn.hook_{self.activation}"
            before = state + 100.0 * layer
            after = before
            for hook_name, hook_fn in fwd_hooks:
                if hook_name == name:
                    after = hook_fn(after, SimpleNamespace(name=name))
            self.events[name] = (before.detach().clone(), after.detach().clone())
            state = after
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        return state.sum(dim=(2, 3))


def _constant_means(
    value: float = 7.0,
    *,
    dtype: torch.dtype = torch.float64,
    activation: str = "result",
) -> GroupedActivationMeans:
    activations = {}
    for layer in range(2):
        name = f"blocks.{layer}.attn.hook_{activation}"
        activations[name] = {
            key: torch.full((3, 2, 1), value, dtype=dtype)
            for key in GROUPS
        }
    return GroupedActivationMeans(
        activation=activation,
        activations=activations,
        sequence_length=3,
        n_heads=2,
    )


POSITIONS = {
    SemanticPosition.END: torch.tensor([2, 1, 0, 2]),
    SemanticPosition.S2: torch.tensor([1, 0, 2, 1]),
}
TOKENS = torch.zeros(4, 3, dtype=torch.long)


def _changed(event: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    before, after = event
    return (before != after).squeeze(-1)


def test_knockout_changes_only_requested_heads_and_variable_positions() -> None:
    model = _AblationModel()
    nodes = (
        HeadPositionNode(0, 1, SemanticPosition.END),
        HeadPositionNode(1, 0, SemanticPosition.S2),
    )

    logits = knockout_nodes(
        model,
        TOKENS,
        POSITIONS,
        GROUPS,
        _constant_means(),
        nodes,
    )

    for layer, node in enumerate(nodes):
        changed = _changed(
            model.events[f"blocks.{layer}.attn.hook_result"]
        )
        expected = torch.zeros_like(changed)
        expected[
            torch.arange(len(TOKENS)),
            POSITIONS[node.position],
            node.head,
        ] = True
        assert torch.equal(changed, expected)
    assert logits.shape == TOKENS.shape
    assert logits.dtype is torch.float64
    assert logits.device == TOKENS.device
    assert model.cfg.use_attn_result is False


def test_run_circuit_ablates_everything_except_kept_head_position_nodes() -> None:
    model = _AblationModel()
    kept = (HeadPositionNode(0, 1, SemanticPosition.END),)

    run_circuit(
        model,
        TOKENS,
        POSITIONS,
        GROUPS,
        _constant_means(),
        kept,
    )

    layer_zero_changed = _changed(
        model.events["blocks.0.attn.hook_result"]
    )
    expected_zero = torch.ones_like(layer_zero_changed)
    expected_zero[
        torch.arange(len(TOKENS)),
        POSITIONS[SemanticPosition.END],
        1,
    ] = False
    assert torch.equal(layer_zero_changed, expected_zero)
    assert torch.all(
        _changed(model.events["blocks.1.attn.hook_result"])
    )


def test_ablation_rejects_target_example_without_reference_group() -> None:
    model = _AblationModel()
    incomplete_groups = {("ABBA", 0): torch.tensor([0, 2])}

    with pytest.raises(ValueError, match="exactly one"):
        knockout_nodes(
            model,
            TOKENS,
            POSITIONS,
            incomplete_groups,
            _constant_means(),
            (HeadPositionNode(0, 0, SemanticPosition.END),),
        )


def test_result_ablation_restores_model_config_when_forward_raises() -> None:
    model = _AblationModel(use_attn_result=False, fail=True)

    with pytest.raises(RuntimeError, match="synthetic"):
        knockout_nodes(
            model,
            TOKENS,
            POSITIONS,
            GROUPS,
            _constant_means(),
            (HeadPositionNode(0, 0, SemanticPosition.END),),
        )

    assert model.cfg.use_attn_result is False


def test_result_ablation_preserves_pre_enabled_model_config() -> None:
    model = _AblationModel(use_attn_result=True)

    knockout_nodes(
        model,
        TOKENS,
        POSITIONS,
        GROUPS,
        _constant_means(),
        (HeadPositionNode(0, 0, SemanticPosition.END),),
    )

    assert model.cfg.use_attn_result is True


def test_z_ablation_uses_head_width_without_enabling_attn_result() -> None:
    model = _AblationModel(activation="z")

    knockout_nodes(
        model,
        TOKENS,
        POSITIONS,
        GROUPS,
        _constant_means(activation="z"),
        (HeadPositionNode(0, 0, SemanticPosition.END),),
    )

    changed = _changed(model.events["blocks.0.attn.hook_z"])
    expected = torch.zeros_like(changed)
    expected[
        torch.arange(len(TOKENS)),
        POSITIONS[SemanticPosition.END],
        0,
    ] = True
    assert torch.equal(changed, expected)
    assert model.cfg.use_attn_result is False
