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
    evaluate_completeness_study,
    evaluate_faithfulness,
    evaluate_all_minimality,
    evaluate_minimality,
    greedy_completeness,
    random_completeness,
    validate_minimality_contexts,
)


Scope = Literal["model", "circuit"]


@dataclass
class _SyntheticEvaluator:
    values: dict[
        tuple[Scope, frozenset[HeadPositionNode]],
        tuple[float, ...],
    ]
    calls: list[tuple[Scope, frozenset[HeadPositionNode]]] = field(
        default_factory=list
    )
    circuit_nodes: tuple[HeadPositionNode, ...] = field(
        default_factory=published_nodes
    )

    def evaluate(
        self,
        scope: Scope,
        removed: Iterable[HeadPositionNode] = (),
    ) -> PerformanceEstimate:
        key = (scope, frozenset(removed))
        self.calls.append(key)
        return PerformanceEstimate(values=self.values[key])


def _node(index: int) -> HeadPositionNode:
    return published_nodes()[index]


def test_faithfulness_reports_paper_definition_and_targets() -> None:
    empty = frozenset()
    evaluator = _SyntheticEvaluator(
        {
            ("model", empty): (4.0, 10.0, 6.0, 8.0),
            ("circuit", empty): (3.0, 9.0, 5.0, 7.0),
        }
    )

    result = evaluate_faithfulness(evaluator)

    assert result.model.values == (4.0, 10.0, 6.0, 8.0)
    assert result.model.mean == 7.0
    assert result.circuit.mean == 6.0
    assert result.absolute_gap == 1.0
    assert result.retained_fraction == pytest.approx(6 / 7)
    assert result.absolute_gap_standard_error == 0.0
    assert result.absolute_gap_ci95.low == 1.0
    assert result.absolute_gap_ci95.high == 1.0
    assert result.paper.model_performance == 3.56
    assert result.paper.absolute_gap == 0.46
    assert result.paper.retained_fraction == pytest.approx(0.87)


def test_completeness_removes_same_k_from_model_and_circuit() -> None:
    removed = frozenset({_node(0), _node(1)})
    evaluator = _SyntheticEvaluator(
        {
            ("model", removed): (2.5, 5.5, 8.5, 11.5),
            ("circuit", removed): (1.25, 4.25, 7.25, 10.25),
        }
    )

    result = evaluate_completeness(evaluator, removed)

    assert result.removed == tuple(sorted(removed))
    assert result.incompleteness == 1.25
    assert result.incompleteness_standard_error == 0.0
    assert result.incompleteness_ci95.low == 1.25
    assert result.incompleteness_ci95.high == 1.25
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
        (scope, removed): (
            float(len(removed)) + (scope == "model"),
        ) * 4
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
        **{("circuit", removed): (0.0,) * 4 for removed in sets},
        **{("model", removed): (gap(removed),) * 4 for removed in sets},
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
        **{("circuit", removed): (0.0,) * 4 for removed in sets},
        **{("model", removed): (gaps[removed],) * 4 for removed in sets},
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


def test_historical_minimality_contexts_match_literal_figure_20_sets() -> None:
    def n(layer: int, head: int, position: str) -> HeadPositionNode:
        return HeadPositionNode(layer, head, position)

    positive = (
        n(9, 9, "END"), n(10, 0, "END"), n(9, 6, "END"),
        n(10, 10, "END"), n(10, 6, "END"), n(10, 2, "END"),
        n(10, 1, "END"), n(11, 2, "END"), n(9, 7, "END"),
        n(9, 0, "END"), n(11, 9, "END"),
    )
    negative = (n(10, 7, "END"), n(11, 10, "END"))
    s_inhibition = (
        n(7, 3, "END"), n(7, 9, "END"),
        n(8, 6, "END"), n(8, 10, "END"),
    )
    induction = (
        n(5, 5, "S2"), n(5, 8, "S2"),
        n(5, 9, "S2"), n(6, 9, "S2"),
    )
    duplicate = (n(0, 1, "S2"), n(0, 10, "S2"), n(3, 0, "S2"))
    previous = (n(2, 2, "S1+1"), n(4, 11, "S1+1"))
    expected = {
        positive[0]: (),
        positive[1]: (positive[0],),
        positive[2]: positive[:2],
        positive[3]: positive[:3],
        positive[4]: positive[:4],
        positive[5]: positive[:5],
        positive[6]: positive[:6],
        positive[7]: positive[:7],
        positive[8]: positive[:8],
        positive[9]: tuple(
            node for node in positive + negative if node != positive[9]
        ),
        positive[10]: tuple(
            node for node in positive + negative if node != positive[10]
        ),
        negative[0]: (negative[1],),
        negative[1]: (negative[0],),
        s_inhibition[0]: s_inhibition[1:],
        s_inhibition[1]: (s_inhibition[0], *s_inhibition[2:]),
        s_inhibition[2]: (*s_inhibition[:2], s_inhibition[3]),
        s_inhibition[3]: s_inhibition[:3],
        induction[0]: induction[1:],
        induction[1]: negative,
        induction[2]: negative,
        induction[3]: induction[:3],
        duplicate[0]: duplicate[1:],
        duplicate[1]: (duplicate[0], duplicate[2]),
        duplicate[2]: duplicate[:2],
        previous[0]: (previous[1],),
        previous[1]: (previous[0],),
    }

    validate_minimality_contexts(HISTORICAL_MINIMALITY_CONTEXTS)
    assert HISTORICAL_MINIMALITY_CONTEXTS == expected


def test_minimality_uses_increment_given_context_and_one_percent_threshold() -> None:
    node = _node(0)
    context = (_node(1),)
    without_node = frozenset((*context, node))
    with_node = frozenset(context)
    evaluator = _SyntheticEvaluator(
        {
            ("model", frozenset()): (4.0,) * 4,
            ("circuit", without_node): (1.0, 2.0, 3.0, 4.0),
            ("circuit", with_node): (1.2, 1.9, 3.2, 4.1),
        }
    )

    result = evaluate_minimality(evaluator, node=node, context=context)

    assert result.without_node.mean == 2.5
    assert result.with_node.mean == pytest.approx(2.6)
    assert result.raw_effect == pytest.approx(0.1)
    assert result.raw_effect_standard_error > 0
    assert result.normalized_effect == pytest.approx(0.025)
    assert result.normalized_effect_ci95.low == 0.0
    assert result.threshold_fraction == 0.01
    assert result.passes_threshold is True
    assert result.ci_supports_threshold is False


def test_minimality_validates_against_evaluator_circuit_nodes() -> None:
    evaluator = _SyntheticEvaluator(
        {},
        circuit_nodes=(_node(0), _node(1)),
    )

    with pytest.raises(ValueError, match="evaluator circuit"):
        evaluate_minimality(
            evaluator,
            node=_node(0),
            context=(_node(2),),
        )


def test_all_minimality_respects_evaluator_circuit_nodes() -> None:
    nodes = (_node(0), _node(1))
    contexts = {
        nodes[0]: (nodes[1],),
        nodes[1]: (nodes[0],),
    }
    removed_sets = {
        frozenset(),
        frozenset({nodes[0]}),
        frozenset({nodes[1]}),
        frozenset(nodes),
    }
    evaluator = _SyntheticEvaluator(
        {
            **{
                ("circuit", removed): (
                    3.0 - float(len(removed)),
                ) * 4
                for removed in removed_sets
            },
            ("model", frozenset()): (4.0,) * 4,
        },
        circuit_nodes=nodes,
    )

    results = evaluate_all_minimality(
        evaluator,
        contexts=contexts,
    )

    assert tuple(result.node for result in results) == nodes


def test_historical_completeness_uses_six_classes_and_eleven_name_movers() -> None:
    class _FunctionEvaluator:
        circuit_nodes = published_nodes()

        def evaluate(self, scope, removed=()):
            value = float(len(frozenset(removed))) + (scope == "model")
            return PerformanceEstimate(values=(value,) * 4)

    study = evaluate_completeness_study(
        _FunctionEvaluator(),
        random_count=0,
        greedy_runs=0,
        greedy_steps=0,
        greedy_candidate_limit=10,
        seed=1,
    )

    assert [item.class_name for item in study.by_class] == [
        "name_movers",
        "negative_name_movers",
        "s_inhibition",
        "induction",
        "duplicate_token",
        "previous_token",
    ]
    assert len(study.by_class[0].result.removed) == 11


def test_ten_greedy_runs_record_top_five_accepted_intermediate_sets() -> None:
    nodes = published_nodes()

    class _FunctionEvaluator:
        circuit_nodes = nodes

        def evaluate(self, scope, removed=()):
            removed_set = frozenset(removed)
            gap = float(sum(nodes.index(node) + 1 for node in removed_set))
            value = gap if scope == "model" else 0.0
            return PerformanceEstimate(values=(value,) * 4)

    study = evaluate_completeness_study(
        _FunctionEvaluator(),
        random_count=0,
        greedy_runs=10,
        greedy_steps=10,
        greedy_candidate_limit=10,
        seed=123,
    )

    assert len(study.greedy) == 10
    assert all(len(run.steps) == 10 for run in study.greedy)
    assert all(len(step.candidates) == 10 for run in study.greedy for step in run.steps)
    accepted = [
        (run_index, step_index, step.result)
        for run_index, run in enumerate(study.greedy)
        for step_index, step in enumerate(run.steps, start=1)
        if step.accepted
    ]
    expected = sorted(
        accepted,
        key=lambda item: item[2].incompleteness,
        reverse=True,
    )
    unique_expected = []
    seen_sets = set()
    for item in expected:
        if item[2].removed in seen_sets:
            continue
        seen_sets.add(item[2].removed)
        unique_expected.append(item)
        if len(unique_expected) == 5:
            break
    assert [
        (item.run_index, item.step_index, item.result)
        for item in study.top_greedy
    ] == unique_expected
    assert len({item.result.removed for item in study.top_greedy}) == 5


def test_minimality_context_validation_rejects_non_circuit_nodes() -> None:
    valid = dict(HISTORICAL_MINIMALITY_CONTEXTS)
    alien = HeadPositionNode(1, 1, published_nodes()[0].position)
    valid[_node(0)] = (alien,)

    with pytest.raises(ValueError, match="outside the published circuit"):
        validate_minimality_contexts(valid)
