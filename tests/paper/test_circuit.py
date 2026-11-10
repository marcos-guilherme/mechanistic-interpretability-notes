import pytest

import ioi_replication.paper as paper
from ioi_replication.paper.circuit import (
    PUBLISHED_CIRCUIT,
    PUBLISHED_CLASS_EDGES,
    CircuitClass,
    HeadPositionNode,
    ReceiverComponent,
    SemanticPosition,
    published_nodes,
    validate_circuit,
)


EXPECTED_HEADS = {
    CircuitClass.NAME_MOVERS: ((9, 9), (10, 0), (9, 6)),
    CircuitClass.BACKUP_NAME_MOVERS: (
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ),
    CircuitClass.NEGATIVE_NAME_MOVERS: ((10, 7), (11, 10)),
    CircuitClass.S_INHIBITION: ((7, 3), (7, 9), (8, 6), (8, 10)),
    CircuitClass.INDUCTION: ((5, 5), (5, 8), (5, 9), (6, 9)),
    CircuitClass.DUPLICATE_TOKEN: ((0, 1), (0, 10), (3, 0)),
    CircuitClass.PREVIOUS_TOKEN: ((2, 2), (4, 11)),
}

EXPECTED_POSITIONS = {
    CircuitClass.NAME_MOVERS: SemanticPosition.END,
    CircuitClass.BACKUP_NAME_MOVERS: SemanticPosition.END,
    CircuitClass.NEGATIVE_NAME_MOVERS: SemanticPosition.END,
    CircuitClass.S_INHIBITION: SemanticPosition.END,
    CircuitClass.INDUCTION: SemanticPosition.S2,
    CircuitClass.DUPLICATE_TOKEN: SemanticPosition.S2,
    CircuitClass.PREVIOUS_TOKEN: SemanticPosition.S1_PLUS_1,
}


def test_published_circuit_has_exactly_seven_classes_and_26_nodes() -> None:
    assert set(PUBLISHED_CIRCUIT) == set(CircuitClass)
    assert len(published_nodes()) == 26

    for circuit_class, expected_heads in EXPECTED_HEADS.items():
        nodes = PUBLISHED_CIRCUIT[circuit_class]
        assert tuple((node.layer, node.head) for node in nodes) == expected_heads
        assert {node.position for node in nodes} == {
            EXPECTED_POSITIONS[circuit_class]
        }


def test_primary_and_backup_name_movers_are_separate_but_preserve_11_heads() -> None:
    primary = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    backup = PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]

    assert len(primary) == 3
    assert len(backup) == 8
    assert len(primary + backup) == 11
    assert not set(primary) & set(backup)


def test_published_nodes_are_unique_and_in_gpt2_small_range() -> None:
    nodes = published_nodes()

    assert len(nodes) == len(set(nodes))
    assert all(0 <= node.layer < 12 for node in nodes)
    assert all(0 <= node.head < 12 for node in nodes)
    validate_circuit(PUBLISHED_CIRCUIT)


def test_node_rejects_invalid_head_coordinates_or_position() -> None:
    with pytest.raises(ValueError, match="layer"):
        HeadPositionNode(12, 0, SemanticPosition.END)
    with pytest.raises(ValueError, match="head"):
        HeadPositionNode(0, -1, SemanticPosition.END)
    with pytest.raises(ValueError, match="semantic position"):
        HeadPositionNode(0, 0, "not-a-position")


def test_class_edges_cover_the_published_information_flow() -> None:
    edge_specs = {
        (
            edge.sender,
            edge.receiver,
            edge.component,
            edge.receiver_position,
        )
        for edge in PUBLISHED_CLASS_EDGES
    }

    assert edge_specs == {
        (
            CircuitClass.PREVIOUS_TOKEN,
            CircuitClass.INDUCTION,
            ReceiverComponent.K,
            SemanticPosition.S1_PLUS_1,
        ),
        (
            CircuitClass.DUPLICATE_TOKEN,
            CircuitClass.S_INHIBITION,
            ReceiverComponent.V,
            SemanticPosition.S2,
        ),
        (
            CircuitClass.INDUCTION,
            CircuitClass.S_INHIBITION,
            ReceiverComponent.V,
            SemanticPosition.S2,
        ),
        (
            CircuitClass.S_INHIBITION,
            CircuitClass.NAME_MOVERS,
            ReceiverComponent.Q,
            SemanticPosition.END,
        ),
        (
            CircuitClass.S_INHIBITION,
            CircuitClass.BACKUP_NAME_MOVERS,
            ReceiverComponent.Q,
            SemanticPosition.END,
        ),
        (
            CircuitClass.S_INHIBITION,
            CircuitClass.NEGATIVE_NAME_MOVERS,
            ReceiverComponent.Q,
            SemanticPosition.END,
        ),
    }


def test_paper_namespace_exports_circuit_and_patching_apis() -> None:
    assert paper.PUBLISHED_CIRCUIT is PUBLISHED_CIRCUIT
    assert paper.HeadPositionNode is HeadPositionNode
    assert callable(paper.activation_patch_node)
    assert callable(paper.paper_path_patch)
