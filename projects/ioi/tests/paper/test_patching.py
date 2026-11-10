from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from ioi_replication.paper.circuit import HeadPositionNode, SemanticPosition
from ioi_replication.paper.patching import (
    PathReceiver,
    activation_patch_matrix,
    activation_patch_node,
    paper_path_patch,
    patch_semantic_positions,
)


def test_patch_semantic_positions_uses_individual_source_and_target_rows() -> None:
    target = torch.zeros(3, 4, 2, 1)
    source = torch.arange(3 * 4 * 2).reshape(3, 4, 2, 1).float()

    patched = patch_semantic_positions(
        target,
        source,
        target_positions=torch.tensor([0, 2, 3]),
        source_positions=torch.tensor([1, 1, 0]),
        head=1,
    )

    rows = torch.arange(3)
    assert torch.equal(
        patched[rows, torch.tensor([0, 2, 3]), 1],
        source[rows, torch.tensor([1, 1, 0]), 1],
    )
    assert torch.count_nonzero(patched[:, :, 0]) == 0
    assert torch.count_nonzero(target) == 0


def test_patch_semantic_positions_can_patch_all_heads_at_variable_positions() -> None:
    target = torch.zeros(2, 3, 2, 2)
    source = torch.ones_like(target)

    patched = patch_semantic_positions(
        target,
        source,
        target_positions=torch.tensor([2, 0]),
    )

    assert torch.equal(patched[0, 2], torch.ones(2, 2))
    assert torch.equal(patched[1, 0], torch.ones(2, 2))
    assert patched.sum().item() == 8


def test_patch_semantic_positions_requires_rank_four_when_head_is_given() -> None:
    with pytest.raises(ValueError, match="rank four"):
        patch_semantic_positions(
            torch.zeros(2, 3, 2),
            torch.ones(2, 3, 2),
            target_positions=torch.tensor([0, 1]),
            head=1,
        )


@pytest.mark.parametrize(
    ("target_positions", "source_positions", "head", "message"),
    [
        (torch.tensor([0]), None, None, "batch"),
        (torch.tensor([0, 3]), None, None, "target_positions"),
        (torch.tensor([0, 1]), torch.tensor([0]), None, "batch"),
        (torch.tensor([0, 1]), None, 2, "head"),
    ],
)
def test_patch_semantic_positions_validates_arguments(
    target_positions: torch.Tensor,
    source_positions: torch.Tensor | None,
    head: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        patch_semantic_positions(
            torch.zeros(2, 3, 2, 1),
            torch.ones(2, 3, 2, 1),
            target_positions=target_positions,
            source_positions=source_positions,
            head=head,
        )


@dataclass
class _Cfg:
    n_layers: int = 2
    n_heads: int = 2
    device: str = "cpu"


class _ActivationModel:
    cfg = _Cfg()

    def __init__(self) -> None:
        self.inference_modes: list[bool] = []

    def run_with_hooks(self, tokens, fwd_hooks):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        activation = torch.zeros(tokens.shape[0], tokens.shape[1], 2, 1)
        for _, hook_fn in fwd_hooks:
            activation = hook_fn(activation, None)
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], 2)
        logits[..., 0] = activation.sum(dim=(2, 3))
        return logits


def _source_cache() -> dict[str, torch.Tensor]:
    cache = {}
    for layer in range(2):
        value = torch.arange(2 * 3 * 2).reshape(2, 3, 2, 1).float()
        cache[f"blocks.{layer}.attn.hook_z"] = value + 100 * layer
    return cache


def test_activation_patch_node_patches_only_the_nodes_semantic_position() -> None:
    model = _ActivationModel()
    node = HeadPositionNode(1, 0, SemanticPosition.END)

    value = activation_patch_node(
        model=model,
        target_tokens=torch.zeros(2, 3, dtype=torch.long),
        source_cache=_source_cache(),
        metric=lambda logits: logits[..., 0].sum(),
        node=node,
        target_positions={SemanticPosition.END: torch.tensor([0, 2])},
        source_positions={SemanticPosition.END: torch.tensor([1, 0])},
        activation="z",
    )

    assert value.item() == pytest.approx(208.0)
    assert model.inference_modes == [True]


def test_activation_patch_matrix_has_layer_head_position_axes() -> None:
    model = _ActivationModel()
    positions = {
        SemanticPosition.S2: torch.tensor([1, 0]),
        SemanticPosition.END: torch.tensor([0, 2]),
    }

    result = activation_patch_matrix(
        model=model,
        target_tokens=torch.zeros(2, 3, dtype=torch.long),
        source_cache=_source_cache(),
        metric=lambda logits: logits[..., 0].sum(),
        target_positions=positions,
        source_positions=positions,
        semantic_positions=(SemanticPosition.S2, SemanticPosition.END),
        activation="z",
    )

    assert result.values.shape == (2, 2, 2)
    assert result.semantic_positions == (
        SemanticPosition.S2,
        SemanticPosition.END,
    )
    assert torch.isfinite(result.values).all()


def test_path_patching_rejects_empty_or_noncausal_endpoints() -> None:
    model = _ActivationModel()
    tokens = torch.zeros(2, 3, dtype=torch.long)
    positions = {SemanticPosition.END: torch.tensor([2, 1])}
    sender = HeadPositionNode(1, 0, SemanticPosition.END)
    receiver = PathReceiver(1, 1, "q", SemanticPosition.END)

    with pytest.raises(ValueError, match="sender"):
        paper_path_patch(
            model,
            tokens,
            tokens,
            positions,
            positions,
            senders=(),
            receivers=(receiver,),
            metric=lambda logits: logits.sum(),
        )
    with pytest.raises(ValueError, match="receiver"):
        paper_path_patch(
            model,
            tokens,
            tokens,
            positions,
            positions,
            senders=(sender,),
            receivers=(),
            metric=lambda logits: logits.sum(),
        )
    with pytest.raises(ValueError, match="later"):
        paper_path_patch(
            model,
            tokens,
            tokens,
            positions,
            positions,
            senders=(sender,),
            receivers=(receiver,),
            metric=lambda logits: logits.sum(),
        )


def test_path_receiver_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="component"):
        PathReceiver(7, 3, "result", SemanticPosition.END)
