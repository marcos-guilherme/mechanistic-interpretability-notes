"""Paper-faithful faithfulness, completeness, and minimality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable, Literal, Mapping, Protocol, Sequence

import torch

from ioi_replication.paper.ablation import (
    GroupKey,
    GroupedActivationMeans,
    knockout_nodes,
    run_circuit,
)
from ioi_replication.paper.circuit import (
    PUBLISHED_CIRCUIT,
    CircuitClass,
    HeadPositionNode,
    published_nodes,
)
from ioi_replication.paper.metrics import logit_difference


Scope = Literal["model", "circuit"]
PAPER_MODEL_PERFORMANCE = 3.56
PAPER_FAITHFULNESS_GAP = 0.46
PAPER_RETAINED_FRACTION = 0.87
PAPER_MINIMALITY_THRESHOLD = 0.01
CI95_Z = 1.959963984540054


@dataclass(frozen=True)
class PerformanceEstimate:
    """Per-example IO-S values with derived mean and standard error."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError("values must not be empty")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("performance values must be finite")
        object.__setattr__(self, "values", values)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values) / self.n

    @property
    def standard_error(self) -> float:
        if self.n < 2:
            return 0.0
        centered = sum((value - self.mean) ** 2 for value in self.values)
        sample_variance = centered / (self.n - 1)
        return math.sqrt(sample_variance / self.n)


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("confidence interval bounds must be finite")
        if self.low < 0 or self.high < self.low:
            raise ValueError("confidence interval bounds are invalid")


class PerformanceEvaluator(Protocol):
    """Evaluate F for a model or circuit under a shared knockout set."""

    def evaluate(
        self,
        scope: Scope,
        removed: Iterable[HeadPositionNode] = (),
    ) -> PerformanceEstimate: ...


@dataclass(frozen=True)
class PaperFaithfulnessTargets:
    model_performance: float = PAPER_MODEL_PERFORMANCE
    absolute_gap: float = PAPER_FAITHFULNESS_GAP
    retained_fraction: float = PAPER_RETAINED_FRACTION


@dataclass(frozen=True)
class FaithfulnessResult:
    model: PerformanceEstimate
    circuit: PerformanceEstimate
    absolute_gap: float
    retained_fraction: float
    absolute_gap_standard_error: float
    absolute_gap_ci95: ConfidenceInterval
    paper: PaperFaithfulnessTargets = field(
        default_factory=PaperFaithfulnessTargets
    )


@dataclass(frozen=True)
class CompletenessResult:
    removed: tuple[HeadPositionNode, ...]
    model_without: PerformanceEstimate
    circuit_without: PerformanceEstimate
    incompleteness: float
    incompleteness_standard_error: float
    incompleteness_ci95: ConfidenceInterval


@dataclass(frozen=True)
class ClassCompletenessResult:
    class_name: str
    result: CompletenessResult


@dataclass(frozen=True)
class GreedyStep:
    selected: HeadPositionNode
    candidates: tuple[HeadPositionNode, ...]
    candidate_result: CompletenessResult
    accepted: bool
    result: CompletenessResult


@dataclass(frozen=True)
class GreedyCompletenessRun:
    seed: int
    initial: CompletenessResult
    steps: tuple[GreedyStep, ...]


@dataclass(frozen=True)
class RankedGreedyResult:
    run_index: int
    step_index: int
    result: CompletenessResult


@dataclass(frozen=True)
class CompletenessStudy:
    empty: CompletenessResult
    by_class: tuple[ClassCompletenessResult, ...]
    random: tuple[CompletenessResult, ...]
    greedy: tuple[GreedyCompletenessRun, ...]
    top_greedy: tuple[RankedGreedyResult, ...]


@dataclass(frozen=True)
class MinimalityResult:
    node: HeadPositionNode
    context: tuple[HeadPositionNode, ...]
    without_node: PerformanceEstimate
    with_node: PerformanceEstimate
    raw_effect: float
    raw_effect_standard_error: float
    raw_effect_ci95: ConfidenceInterval
    normalized_effect: float
    normalized_effect_ci95: ConfidenceInterval
    threshold_fraction: float
    passes_threshold: bool
    ci_supports_threshold: bool


def _ordered(nodes: Iterable[HeadPositionNode]) -> tuple[HeadPositionNode, ...]:
    return tuple(sorted(set(nodes)))


def _paired_absolute_effect(
    first: PerformanceEstimate,
    second: PerformanceEstimate,
) -> tuple[float, float, ConfidenceInterval]:
    if first.n != second.n:
        raise ValueError("paired estimates must contain the same number of prompts")
    differences = tuple(
        first_value - second_value
        for first_value, second_value in zip(first.values, second.values)
    )
    signed_mean = sum(differences) / len(differences)
    effect = abs(signed_mean)
    if len(differences) < 2:
        standard_error = 0.0
    else:
        centered = sum(
            (difference - signed_mean) ** 2
            for difference in differences
        )
        standard_error = math.sqrt(
            centered / (len(differences) - 1) / len(differences)
        )
    margin = CI95_Z * standard_error
    return (
        effect,
        standard_error,
        ConfidenceInterval(
            low=max(0.0, effect - margin),
            high=effect + margin,
        ),
    )


HISTORICAL_COMPLETENESS_CLASSES: dict[
    str,
    tuple[HeadPositionNode, ...],
] = {
    "name_movers": (
        PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
        + PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]
    ),
    "negative_name_movers": PUBLISHED_CIRCUIT[
        CircuitClass.NEGATIVE_NAME_MOVERS
    ],
    "s_inhibition": PUBLISHED_CIRCUIT[CircuitClass.S_INHIBITION],
    "induction": PUBLISHED_CIRCUIT[CircuitClass.INDUCTION],
    "duplicate_token": PUBLISHED_CIRCUIT[CircuitClass.DUPLICATE_TOKEN],
    "previous_token": PUBLISHED_CIRCUIT[CircuitClass.PREVIOUS_TOKEN],
}


def f_model(evaluator: PerformanceEvaluator) -> PerformanceEstimate:
    """Return F(M)."""
    return evaluator.evaluate("model", ())


def f_circuit(evaluator: PerformanceEvaluator) -> PerformanceEstimate:
    """Return F(C)."""
    return evaluator.evaluate("circuit", ())


def f_model_without(
    evaluator: PerformanceEvaluator,
    removed: Iterable[HeadPositionNode],
) -> PerformanceEstimate:
    """Return F(M\\K) using the evaluator's mean ablation."""
    return evaluator.evaluate("model", removed)


def f_circuit_without(
    evaluator: PerformanceEvaluator,
    removed: Iterable[HeadPositionNode],
) -> PerformanceEstimate:
    """Return F(C\\K) using the same mean ablation as F(M\\K)."""
    return evaluator.evaluate("circuit", removed)


def evaluate_faithfulness(
    evaluator: PerformanceEvaluator,
) -> FaithfulnessResult:
    model = f_model(evaluator)
    circuit = f_circuit(evaluator)
    if model.mean == 0:
        raise ZeroDivisionError("F(M) must be non-zero to compute retained fraction")
    gap, gap_standard_error, gap_ci95 = _paired_absolute_effect(
        model,
        circuit,
    )
    return FaithfulnessResult(
        model=model,
        circuit=circuit,
        absolute_gap=gap,
        retained_fraction=circuit.mean / model.mean,
        absolute_gap_standard_error=gap_standard_error,
        absolute_gap_ci95=gap_ci95,
    )


def evaluate_completeness(
    evaluator: PerformanceEvaluator,
    removed: Iterable[HeadPositionNode],
) -> CompletenessResult:
    removed_nodes = _ordered(removed)
    model = f_model_without(evaluator, removed_nodes)
    circuit = f_circuit_without(evaluator, removed_nodes)
    incompleteness, standard_error, ci95 = _paired_absolute_effect(
        model,
        circuit,
    )
    return CompletenessResult(
        removed=removed_nodes,
        model_without=model,
        circuit_without=circuit,
        incompleteness=incompleteness,
        incompleteness_standard_error=standard_error,
        incompleteness_ci95=ci95,
    )


def random_completeness(
    evaluator: PerformanceEvaluator,
    *,
    nodes: Sequence[HeadPositionNode] = published_nodes(),
    count: int,
    seed: int,
) -> tuple[CompletenessResult, ...]:
    """Evaluate uniformly sampled subsets of circuit nodes."""
    if count < 0:
        raise ValueError("count must be non-negative")
    nodes = tuple(nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    rng = random.Random(seed)
    return tuple(
        evaluate_completeness(
            evaluator,
            (node for node in nodes if rng.randrange(2)),
        )
        for _ in range(count)
    )


def greedy_completeness(
    evaluator: PerformanceEvaluator,
    *,
    nodes: Sequence[HeadPositionNode] = published_nodes(),
    steps: int,
    candidate_limit: int | None,
    seed: int,
) -> GreedyCompletenessRun:
    """Follow ``completeness.py`` greedy sampling and acceptance.

    Each iteration samples up to ``candidate_limit`` remaining nodes and only
    adds the best candidate when it improves incompleteness. This follows the
    executable historical source, not Algorithm 3's literal line 4, whose
    pseudocode compares two circuit scores instead of the completeness metric.
    """
    nodes = tuple(nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if candidate_limit is not None and candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive or None")
    rng = random.Random(seed)
    removed: set[HeadPositionNode] = set()
    initial = evaluate_completeness(evaluator, removed)
    current = initial
    curve: list[GreedyStep] = []

    for _ in range(min(steps, len(nodes))):
        remaining = [node for node in nodes if node not in removed]
        if not remaining:
            break
        sample_size = (
            len(remaining)
            if candidate_limit is None
            else min(candidate_limit, len(remaining))
        )
        candidates = tuple(rng.sample(remaining, sample_size))
        scored = [
            (
                evaluate_completeness(evaluator, (*removed, candidate)),
                candidate,
            )
            for candidate in candidates
        ]
        best_result, best_node = max(
            scored,
            key=lambda item: item[0].incompleteness,
        )
        accepted = best_result.incompleteness > current.incompleteness
        if accepted:
            removed.add(best_node)
            current = best_result
        curve.append(
            GreedyStep(
                selected=best_node,
                candidates=candidates,
                candidate_result=best_result,
                accepted=accepted,
                result=current,
            )
        )

    return GreedyCompletenessRun(
        seed=seed,
        initial=initial,
        steps=tuple(curve),
    )


def evaluate_completeness_study(
    evaluator: PerformanceEvaluator,
    *,
    classes: Mapping[
        str,
        Sequence[HeadPositionNode],
    ] = HISTORICAL_COMPLETENESS_CLASSES,
    random_count: int,
    greedy_runs: int,
    greedy_steps: int,
    greedy_candidate_limit: int | None,
    seed: int,
) -> CompletenessStudy:
    if greedy_runs < 0:
        raise ValueError("greedy_runs must be non-negative")
    nodes = tuple(
        getattr(evaluator, "circuit_nodes", published_nodes())
    )
    class_results = tuple(
        ClassCompletenessResult(
            class_name=class_name,
            result=evaluate_completeness(evaluator, class_nodes),
        )
        for class_name, class_nodes in classes.items()
    )
    greedy_runs_results = tuple(
        greedy_completeness(
            evaluator,
            nodes=nodes,
            steps=greedy_steps,
            candidate_limit=greedy_candidate_limit,
            seed=seed + run,
        )
        for run in range(greedy_runs)
    )
    accepted_intermediates = [
        RankedGreedyResult(
            run_index=run_index,
            step_index=step_index,
            result=step.result,
        )
        for run_index, run in enumerate(greedy_runs_results)
        for step_index, step in enumerate(run.steps, start=1)
        if step.accepted
    ]
    ranked_intermediates = sorted(
        accepted_intermediates,
        key=lambda item: item.result.incompleteness,
        reverse=True,
    )
    top_greedy_list: list[RankedGreedyResult] = []
    seen_sets: set[tuple[HeadPositionNode, ...]] = set()
    for item in ranked_intermediates:
        if item.result.removed in seen_sets:
            continue
        seen_sets.add(item.result.removed)
        top_greedy_list.append(item)
        if len(top_greedy_list) == 5:
            break
    top_greedy = tuple(top_greedy_list)
    return CompletenessStudy(
        empty=evaluate_completeness(evaluator, ()),
        by_class=class_results,
        random=random_completeness(
            evaluator,
            nodes=nodes,
            count=random_count,
            seed=seed,
        ),
        greedy=greedy_runs_results,
        top_greedy=top_greedy,
    )


def _historical_minimality_contexts() -> dict[
    HeadPositionNode,
    tuple[HeadPositionNode, ...],
]:
    primary = PUBLISHED_CIRCUIT[CircuitClass.NAME_MOVERS]
    backup = PUBLISHED_CIRCUIT[CircuitClass.BACKUP_NAME_MOVERS]
    positive = primary + backup
    negative = PUBLISHED_CIRCUIT[CircuitClass.NEGATIVE_NAME_MOVERS]
    contexts: dict[HeadPositionNode, tuple[HeadPositionNode, ...]] = {
        node: positive[:index]
        for index, node in enumerate(positive)
    }
    for node in backup[-2:]:
        contexts[node] = tuple(
            other for other in positive + negative if other != node
        )

    for circuit_class in (
        CircuitClass.NEGATIVE_NAME_MOVERS,
        CircuitClass.S_INHIBITION,
        CircuitClass.INDUCTION,
        CircuitClass.DUPLICATE_TOKEN,
        CircuitClass.PREVIOUS_TOKEN,
    ):
        class_nodes = PUBLISHED_CIRCUIT[circuit_class]
        for node in class_nodes:
            contexts[node] = tuple(other for other in class_nodes if other != node)

    induction = PUBLISHED_CIRCUIT[CircuitClass.INDUCTION]
    contexts[induction[1]] = negative
    contexts[induction[2]] = negative
    return contexts


HISTORICAL_MINIMALITY_CONTEXTS = _historical_minimality_contexts()


def validate_minimality_contexts(
    contexts: Mapping[
        HeadPositionNode,
        Sequence[HeadPositionNode],
    ],
    *,
    circuit_nodes: Sequence[HeadPositionNode] = published_nodes(),
) -> None:
    circuit = set(circuit_nodes)
    if set(contexts) != circuit:
        raise ValueError("minimality contexts must cover every circuit node exactly")
    for node, raw_context in contexts.items():
        context = tuple(raw_context)
        if len(set(context)) != len(context):
            raise ValueError(f"minimality context for {node} contains duplicates")
        if node in context:
            raise ValueError(f"minimality context for {node} must exclude the node")
        if not set(context) <= circuit:
            raise ValueError(
                f"minimality context for {node} contains nodes "
                "outside the published circuit"
            )


validate_minimality_contexts(HISTORICAL_MINIMALITY_CONTEXTS)


def evaluate_minimality(
    evaluator: PerformanceEvaluator,
    *,
    node: HeadPositionNode,
    context: Sequence[HeadPositionNode],
    threshold_fraction: float = PAPER_MINIMALITY_THRESHOLD,
) -> MinimalityResult:
    if threshold_fraction < 0:
        raise ValueError("threshold_fraction must be non-negative")
    circuit_nodes = tuple(
        getattr(evaluator, "circuit_nodes", published_nodes())
    )
    circuit = set(circuit_nodes)
    if node not in circuit or not set(context) <= circuit:
        raise ValueError(
            "node and context nodes must belong to the evaluator circuit"
        )
    if node in context:
        raise ValueError("minimality context must exclude the evaluated node")
    context = _ordered(context)
    without_node = f_circuit_without(evaluator, (*context, node))
    with_node = f_circuit_without(evaluator, context)
    raw_effect, raw_standard_error, raw_ci95 = _paired_absolute_effect(
        with_node,
        without_node,
    )
    model = f_model(evaluator)
    if model.mean == 0:
        raise ZeroDivisionError("F(M) must be non-zero to normalize minimality")
    normalized = raw_effect / abs(model.mean)
    normalized_ci95 = ConfidenceInterval(
        low=raw_ci95.low / abs(model.mean),
        high=raw_ci95.high / abs(model.mean),
    )
    return MinimalityResult(
        node=node,
        context=context,
        without_node=without_node,
        with_node=with_node,
        raw_effect=raw_effect,
        raw_effect_standard_error=raw_standard_error,
        raw_effect_ci95=raw_ci95,
        normalized_effect=normalized,
        normalized_effect_ci95=normalized_ci95,
        threshold_fraction=threshold_fraction,
        passes_threshold=normalized >= threshold_fraction,
        ci_supports_threshold=normalized_ci95.low > threshold_fraction,
    )


def evaluate_all_minimality(
    evaluator: PerformanceEvaluator,
    *,
    contexts: Mapping[
        HeadPositionNode,
        Sequence[HeadPositionNode],
    ] = HISTORICAL_MINIMALITY_CONTEXTS,
    threshold_fraction: float = PAPER_MINIMALITY_THRESHOLD,
) -> tuple[MinimalityResult, ...]:
    circuit_nodes = tuple(
        getattr(evaluator, "circuit_nodes", published_nodes())
    )
    if contexts is HISTORICAL_MINIMALITY_CONTEXTS and set(contexts) != set(
        circuit_nodes
    ):
        circuit = set(circuit_nodes)
        contexts = {
            node: tuple(
                context_node
                for context_node in HISTORICAL_MINIMALITY_CONTEXTS[node]
                if context_node in circuit
            )
            for node in circuit_nodes
        }
    validate_minimality_contexts(
        contexts,
        circuit_nodes=circuit_nodes,
    )
    return tuple(
        evaluate_minimality(
            evaluator,
            node=node,
            context=contexts[node],
            threshold_fraction=threshold_fraction,
        )
        for node in circuit_nodes
    )


class MeanAblationEvaluator:
    """Cached F evaluator backed by the shared grouped pABC mean ablation."""

    def __init__(
        self,
        *,
        model: object,
        target_tokens: torch.Tensor,
        positions: Mapping[object, torch.Tensor],
        groups: Mapping[GroupKey, torch.Tensor],
        group_means: GroupedActivationMeans,
        io_token_ids: torch.Tensor,
        s_token_ids: torch.Tensor,
        circuit_nodes: Sequence[HeadPositionNode] = published_nodes(),
        batch_size: int | None = None,
    ) -> None:
        if target_tokens.ndim != 2 or target_tokens.shape[0] == 0:
            raise ValueError("target_tokens must be a non-empty batch")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        self.model = model
        self.target_tokens = target_tokens
        self.positions = positions
        self.groups = groups
        self.group_means = group_means
        self.io_token_ids = io_token_ids
        self.s_token_ids = s_token_ids
        self.circuit_nodes = tuple(circuit_nodes)
        if len(set(self.circuit_nodes)) != len(self.circuit_nodes):
            raise ValueError("circuit_nodes must be unique")
        self.batch_size = batch_size or len(target_tokens)
        self._cache: dict[
            tuple[Scope, frozenset[HeadPositionNode]],
            PerformanceEstimate,
        ] = {}

    @property
    def cached_evaluations(self) -> int:
        return len(self._cache)

    def _batch_groups(self, start: int, stop: int) -> dict[GroupKey, torch.Tensor]:
        result: dict[GroupKey, torch.Tensor] = {}
        for key, raw_indices in self.groups.items():
            indices = torch.as_tensor(raw_indices).long()
            selected = indices[(indices >= start) & (indices < stop)] - start
            if selected.numel():
                result[key] = selected
        return result

    def _run_batch(
        self,
        scope: Scope,
        removed: tuple[HeadPositionNode, ...],
        start: int,
        stop: int,
    ) -> torch.Tensor:
        tokens = self.target_tokens[start:stop]
        positions = {
            key: torch.as_tensor(value)[start:stop]
            for key, value in self.positions.items()
        }
        groups = self._batch_groups(start, stop)
        if scope == "model":
            if removed:
                logits = knockout_nodes(
                    self.model,
                    tokens,
                    positions,
                    groups,
                    self.group_means,
                    removed,
                )
            else:
                with torch.inference_mode():
                    logits = self.model(tokens)
        elif scope == "circuit":
            removed_set = set(removed)
            kept = tuple(
                node for node in self.circuit_nodes if node not in removed_set
            )
            logits = run_circuit(
                self.model,
                tokens,
                positions,
                groups,
                self.group_means,
                kept,
            )
        else:
            raise ValueError("scope must be 'model' or 'circuit'")
        return logit_difference(
            logits,
            self.io_token_ids[start:stop],
            self.s_token_ids[start:stop],
            torch.as_tensor(self.positions["END"])[start:stop],
            reduction="none",
        )

    def evaluate(
        self,
        scope: Scope,
        removed: Iterable[HeadPositionNode] = (),
    ) -> PerformanceEstimate:
        removed_nodes = _ordered(removed)
        if not set(removed_nodes) <= set(self.circuit_nodes):
            raise ValueError("removed nodes must belong to the circuit")
        key = (scope, frozenset(removed_nodes))
        if key not in self._cache:
            values = torch.cat(
                [
                    self._run_batch(
                        scope,
                        removed_nodes,
                        start,
                        min(start + self.batch_size, len(self.target_tokens)),
                    )
                    .detach()
                    .float()
                    .cpu()
                    for start in range(
                        0,
                        len(self.target_tokens),
                        self.batch_size,
                    )
                ]
            )
            self._cache[key] = PerformanceEstimate(
                values=tuple(float(value) for value in values),
            )
        return self._cache[key]
