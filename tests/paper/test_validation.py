from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import pytest

from ioi_replication.paper.circuit import (
    PUBLISHED_CIRCUIT,
    CircuitClass,
    HeadPositionNode,
    published_nodes,
)
from ioi_replication.paper.validation import (
    HISTORICAL_MINIMALITY_CONTEXTS,
    PerformanceEstimate,
    evaluate_completeness,
    evaluate_faithfulness,
    evaluate_minimality,
    greedy_completeness,
    random_completeness,
    validate_minimality_contexts,
)


Scope = Literal["model", "circuit"]


@dataclass
class _SyntheticEvaluator:
    values: dict[tuple[Scope, frozenset[HeadPositionNode]], float]
    calls: list[tuple[Scope, frozenset[HeadPositionNode]]] = field(
        default_factory=list
    )

    def evaluate(
        self,
        scope: Scope,
        removed: Iterable[HeadPositionNode] = (),
    ) -> PerformanceEstimate:
        key = (scope, frozenset(removed))
        self.calls.append(key)
        return PerformanceEstimate(mean=self.values[key], standard_error=0.0, n=4)


def _node(index: int) -> HeadPositionNode:
    return published_nodes()[index]


def test_faithfulness_reports_paper_definition_and_targets() -> None:
    empty = frozenset()
    evaluator = _SyntheticEvaluator(
        {
            ("model", empty): 4.0,
            ("circuit", empty): 3.0,
        }
    )

    result = evaluate_faithfulness(evaluator)

    assert result.model.mean == 4.0
    assert result.circuit.mean == 3.0
    assert result.absolute_gap == 1.0
    assert result.retained_fraction == 0.75
    assert result.paper.model_performance == 3.56
    assert result.paper.absolute_gap == 0.46
    assert result.paper.retained_fraction == pytest.approx(0.87)


def test_completeness_removes_same_k_from_model_and_circuit() -> None:
    removed = frozenset({_node(0), _node(1)})
    evaluator = _SyntheticEvaluator(
        {
            ("model", removed): 2.5,
            ("circuit", removed): 1.25,
        }
    )

    result = evaluate_completeness(evaluator, removed)

    assert result.removed == tuple(sorted(removed))
    assert result.incompleteness == 1.25
    assert evaluator.calls == [
        ("model", removed),
        ("circuit", removed),
    ]


def test_random_completeness_is_reproducible_and_records_sets() -> None:
    nodes = (_node(0), _node(1), _node(2), _node(3))
    all_sets = [
        frozenset(nodes[index] for index in range(len(nodes)) if mask >> index & 1)
        for mask in range(2 ** len(nodes))
    ]
    values = {
        (scope, removed): float(len(removed)) + (scope == "model")
        for scope in ("model", "circuit")
        for removed in all_sets
    }

    first = random_completeness(
        _SyntheticEvaluator(values),
        nodes=nodes,
        count=10,
        seed=42,
    )
    second = random_completeness(
        _SyntheticEvaluator(values),
        nodes=nodes,
        count=10,
        seed=42,
    )

    assert [item.removed for item in first] == [
        item.removed for item in second
    ]
    assert len(first) == 10
    assert all(item.incompleteness == 1.0 for item in first)


def test_greedy_completeness_selects_best_increment_and_records_curve() -> None:
    nodes = (_node(0), _node(1), _node(2))

    def gap(removed: frozenset[HeadPositionNode]) -> float:
        score = sum(
            {
                nodes[0]: 1.0,
                nodes[1]: 4.0,
                nodes[2]: 2.0,
            }[node]
            for node in removed
        )
        if {nodes[1], nodes[2]} <= removed:
            score += 3.0
        return score

    sets = [
        frozenset(nodes[index] for index in range(len(nodes)) if mask >> index & 1)
        for mask in range(2 ** len(nodes))
    ]
    values = {
        **{("circuit", removed): 0.0 for removed in sets},
        **{("model", removed): gap(removed) for removed in sets},
    }
    run = greedy_completeness(
        _SyntheticEvaluator(values),
        nodes=nodes,
        steps=2,
        candidate_limit=None,
        seed=7,
    )

    assert [step.selected for step in run.steps] == [nodes[1], nodes[2]]
    assert [step.result.incompleteness for step in run.steps] == [4.0, 9.0]
    assert run.steps[0].result.removed == (nodes[1],)
    assert run.steps[1].result.removed == tuple(sorted((nodes[1], nodes[2])))


def test_greedy_completeness_rejects_non_improving_candidate() -> None:
    nodes = (_node(0), _node(1))
    sets = (frozenset(), frozenset({nodes[0]}), frozenset({nodes[1]}))
    gaps = {
        frozenset(): 3.0,
        frozenset({nodes[0]}): 2.0,
        frozenset({nodes[1]}): 1.0,
    }
    values = {
        **{("circuit", removed): 0.0 for removed in sets},
        **{("model", removed): gaps[removed] for removed in sets},
    }

    run = greedy_completeness(
        _SyntheticEvaluator(values),
        nodes=nodes,
        steps=1,
        candidate_limit=None,
        seed=3,
    )

    assert run.steps[0].accepted is False
    assert run.steps[0].candidate_result.incompleteness == 2.0
    assert run.steps[0].result == run.initial
    assert run.steps[0].result.removed == ()


def test_historical_minimality_contexts_are_exact_and_consistent() -> None:
    primary = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    backup = PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]
    negative = PUBLISHED_CIRCUIT[CircuitClass.NEGATIVE_NAME_MOVERS]
    induction = PUBLISHED_CIRCUIT[CircuitClass.INDUCTION]

    validate_minimality_contexts(HISTORICAL_MINIMALITY_CONTEXTS)

    assert set(HISTORICAL_MINIMALITY_CONTEXTS) == set(published_nodes())
    assert HISTORICAL_MINIMALITY_CONTEXTS[primary[0]] == ()
    assert HISTORICAL_MINIMALITY_CONTEXTS[primary[1]] == (primary[0],)
    assert HISTORICAL_MINIMALITY_CONTEXTS[primary[2]] == primary[:2]
    assert set(HISTORICAL_MINIMALITY_CONTEXTS[induction[1]]) == set(negative)
    assert set(HISTORICAL_MINIMALITY_CONTEXTS[induction[2]]) == set(negative)
    assert set(HISTORICAL_MINIMALITY_CONTEXTS[backup[-2]]) == (
        set(primary + backup + negative) - {backup[-2]}
    )
    assert set(HISTORICAL_MINIMALITY_CONTEXTS[backup[-1]]) == (
        set(primary + backup + negative) - {backup[-1]}
    )
    for node, context in HISTORICAL_MINIMALITY_CONTEXTS.items():
        assert node not in context
        assert set(context) <= set(published_nodes())


def test_minimality_uses_increment_given_context_and_one_percent_threshold() -> None:
    node = _node(0)
    context = (_node(1),)
    without_node = frozenset((*context, node))
    with_node = frozenset(context)
    evaluator = _SyntheticEvaluator(
        {
            ("model", frozenset()): 4.0,
            ("circuit", without_node): 1.0,
            ("circuit", with_node): 1.03,
        }
    )

    result = evaluate_minimality(evaluator, node=node, context=context)

    assert result.without_node.mean == 1.0
    assert result.with_node.mean == 1.03
    assert result.raw_effect == pytest.approx(0.03)
    assert result.normalized_effect == pytest.approx(0.0075)
    assert result.threshold_fraction == 0.01
    assert result.passes_threshold is False


def test_minimality_context_validation_rejects_non_circuit_nodes() -> None:
    valid = dict(HISTORICAL_MINIMALITY_CONTEXTS)
    alien = HeadPositionNode(1, 1, published_nodes()[0].position)
    valid[_node(0)] = (alien,)

    with pytest.raises(ValueError, match="outside the published circuit"):
        validate_minimality_contexts(valid)
