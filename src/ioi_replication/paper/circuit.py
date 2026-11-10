"""Published GPT-2 Small IOI circuit at semantic token positions.

The head lists and positions are ported from
``easy_transformer/ioi_circuit_extraction.py``.  That source groups all
eleven positive Name Movers together; this module exposes the three primary
Name Movers separately from the eight Backup Name Movers while preserving the
published 26-node circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class SemanticPosition(str, Enum):
    IO = "IO"
    S1 = "S1"
    S2 = "S2"
    S1_PLUS_1 = "S1+1"
    END = "END"


class CircuitClass(str, Enum):
    NAME_MOVERS = "name_movers"
    BACKUP_NAME_MOVERS = "backup_name_movers"
    NEGATIVE_NAME_MOVERS = "negative_name_movers"
    S_INHIBITION = "s_inhibition"
    INDUCTION = "induction"
    DUPLICATE_TOKEN = "duplicate_token"
    PREVIOUS_TOKEN = "previous_token"


class ReceiverComponent(str, Enum):
    Q = "q"
    K = "k"
    V = "v"


@dataclass(frozen=True, order=True)
class HeadPositionNode:
    """One attention head evaluated at one semantic token position."""

    layer: int
    head: int
    position: SemanticPosition

    def __post_init__(self) -> None:
        if not 0 <= self.layer < 12:
            raise ValueError("layer must be in [0, 12) for GPT-2 Small")
        if not 0 <= self.head < 12:
            raise ValueError("head must be in [0, 12) for GPT-2 Small")
        try:
            position = SemanticPosition(self.position)
        except ValueError as error:
            raise ValueError(f"unknown semantic position: {self.position!r}") from error
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class CircuitClassEdge:
    """A published class-to-class information-flow relation."""

    sender: CircuitClass
    receiver: CircuitClass
    component: ReceiverComponent
    receiver_position: SemanticPosition

    def __post_init__(self) -> None:
        object.__setattr__(self, "sender", CircuitClass(self.sender))
        object.__setattr__(self, "receiver", CircuitClass(self.receiver))
        object.__setattr__(self, "component", ReceiverComponent(self.component))
        object.__setattr__(
            self,
            "receiver_position",
            SemanticPosition(self.receiver_position),
        )
        if self.sender == self.receiver:
            raise ValueError("a circuit class edge cannot be a self-edge")


def _nodes(
    heads: Sequence[tuple[int, int]],
    position: SemanticPosition,
) -> tuple[HeadPositionNode, ...]:
    return tuple(
        HeadPositionNode(layer, head, position)
        for layer, head in heads
    )


PUBLISHED_CIRCUIT: dict[CircuitClass, tuple[HeadPositionNode, ...]] = {
    CircuitClass.NAME_MOVERS: _nodes(
        ((9, 9), (10, 0), (9, 6)),
        SemanticPosition.END,
    ),
    CircuitClass.BACKUP_NAME_MOVERS: _nodes(
        (
            (10, 10),
            (10, 6),
            (10, 2),
            (10, 1),
            (11, 2),
            (9, 7),
            (9, 0),
            (11, 9),
        ),
        SemanticPosition.END,
    ),
    CircuitClass.NEGATIVE_NAME_MOVERS: _nodes(
        ((10, 7), (11, 10)),
        SemanticPosition.END,
    ),
    CircuitClass.S_INHIBITION: _nodes(
        ((7, 3), (7, 9), (8, 6), (8, 10)),
        SemanticPosition.END,
    ),
    CircuitClass.INDUCTION: _nodes(
        ((5, 5), (5, 8), (5, 9), (6, 9)),
        SemanticPosition.S2,
    ),
    CircuitClass.DUPLICATE_TOKEN: _nodes(
        ((0, 1), (0, 10), (3, 0)),
        SemanticPosition.S2,
    ),
    CircuitClass.PREVIOUS_TOKEN: _nodes(
        ((2, 2), (4, 11)),
        SemanticPosition.S1_PLUS_1,
    ),
}


PUBLISHED_CLASS_EDGES = (
    CircuitClassEdge(
        CircuitClass.PREVIOUS_TOKEN,
        CircuitClass.INDUCTION,
        ReceiverComponent.K,
        SemanticPosition.S1_PLUS_1,
    ),
    CircuitClassEdge(
        CircuitClass.DUPLICATE_TOKEN,
        CircuitClass.S_INHIBITION,
        ReceiverComponent.V,
        SemanticPosition.S2,
    ),
    CircuitClassEdge(
        CircuitClass.INDUCTION,
        CircuitClass.S_INHIBITION,
        ReceiverComponent.V,
        SemanticPosition.S2,
    ),
    CircuitClassEdge(
        CircuitClass.S_INHIBITION,
        CircuitClass.NAME_MOVERS,
        ReceiverComponent.Q,
        SemanticPosition.END,
    ),
    CircuitClassEdge(
        CircuitClass.S_INHIBITION,
        CircuitClass.BACKUP_NAME_MOVERS,
        ReceiverComponent.Q,
        SemanticPosition.END,
    ),
    CircuitClassEdge(
        CircuitClass.S_INHIBITION,
        CircuitClass.NEGATIVE_NAME_MOVERS,
        ReceiverComponent.Q,
        SemanticPosition.END,
    ),
)


def published_nodes() -> tuple[HeadPositionNode, ...]:
    """Return all published nodes in class/list order."""
    return tuple(
        node
        for circuit_class in CircuitClass
        for node in PUBLISHED_CIRCUIT[circuit_class]
    )


def validate_circuit(
    circuit: Mapping[CircuitClass, Sequence[HeadPositionNode]],
    *,
    n_layers: int = 12,
    n_heads: int = 12,
    expected_total: int = 26,
) -> None:
    """Validate class coverage, GPT dimensions, uniqueness and node count."""
    if set(circuit) != set(CircuitClass):
        raise ValueError("circuit must contain exactly the seven published classes")
    nodes = tuple(node for group in circuit.values() for node in group)
    if len(nodes) != expected_total:
        raise ValueError(
            f"circuit must contain {expected_total} nodes, got {len(nodes)}"
        )
    if len(set(nodes)) != len(nodes):
        raise ValueError("circuit nodes must be unique")
    for node in nodes:
        if not 0 <= node.layer < n_layers:
            raise ValueError(f"node layer {node.layer} is outside model range")
        if not 0 <= node.head < n_heads:
            raise ValueError(f"node head {node.head} is outside model range")


validate_circuit(PUBLISHED_CIRCUIT)

