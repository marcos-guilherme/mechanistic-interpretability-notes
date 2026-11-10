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


@dataclass(frozen=True)
class PerformanceEstimate:
    """Mean IO-S logit difference and its standard error."""

    mean: float
    standard_error: float
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not math.isfinite(self.mean) or not math.isfinite(self.standard_error):
            raise ValueError("performance estimates must be finite")
        if self.standard_error < 0:
            raise ValueError("standard_error must be non-negative")


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


@dataclass(frozen=True)
class ClassCompletenessResult:
    circuit_class: CircuitClass
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
class CompletenessStudy:
    empty: CompletenessResult
    by_class: tuple[ClassCompletenessResult, ...]
    random: tuple[CompletenessResult, ...]
    greedy: tuple[GreedyCompletenessRun, ...]


@dataclass(frozen=True)
class MinimalityResult:
    node: HeadPositionNode
    context: tuple[HeadPositionNode, ...]
    without_node: PerformanceEstimate
    with_node: PerformanceEstimate
    raw_effect: float
    raw_effect_standard_error: float
    normalized_effect: float
    threshold_fraction: float
    passes_threshold: bool


def _ordered(nodes: Iterable[HeadPositionNode]) -> tuple[HeadPositionNode, ...]:
    return tuple(sorted(set(nodes)))


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
    return FaithfulnessResult(
        model=model,
        circuit=circuit,
        absolute_gap=abs(model.mean - circuit.mean),
        retained_fraction=circuit.mean / model.mean,
        absolute_gap_standard_error=math.hypot(
            model.standard_error,
            circuit.standard_error,
        ),
    )


def evaluate_completeness(
    evaluator: PerformanceEvaluator,
    removed: Iterable[HeadPositionNode],
) -> CompletenessResult:
    removed_nodes = _ordered(removed)
    model = f_model_without(evaluator, removed_nodes)
    circuit = f_circuit_without(evaluator, removed_nodes)
    return CompletenessResult(
        removed=removed_nodes,
        model_without=model,
        circuit_without=circuit,
        incompleteness=abs(model.mean - circuit.mean),
        incompleteness_standard_error=math.hypot(
            model.standard_error,
            circuit.standard_error,
        ),
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
    """Greedily add the candidate maximizing incompleteness at each step."""
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
    circuit: Mapping[
        CircuitClass,
        Sequence[HeadPositionNode],
    ] = PUBLISHED_CIRCUIT,
    random_count: int,
    greedy_runs: int,
    greedy_steps: int,
    greedy_candidate_limit: int | None,
    seed: int,
) -> CompletenessStudy:
    if greedy_runs < 0:
        raise ValueError("greedy_runs must be non-negative")
    nodes = tuple(node for group in circuit.values() for node in group)
    return CompletenessStudy(
        empty=evaluate_completeness(evaluator, ()),
        by_class=tuple(
            ClassCompletenessResult(
                circuit_class=circuit_class,
                result=evaluate_completeness(evaluator, class_nodes),
            )
            for circuit_class, class_nodes in circuit.items()
        ),
        random=random_completeness(
            evaluator,
            nodes=nodes,
            count=random_count,
            seed=seed,
        ),
        greedy=tuple(
            greedy_completeness(
                evaluator,
                nodes=nodes,
                steps=greedy_steps,
                candidate_limit=greedy_candidate_limit,
                seed=seed + run,
            )
            for run in range(greedy_runs)
        ),
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
    circuit = set(published_nodes())
    if node not in circuit or not set(context) <= circuit:
        raise ValueError("node and context nodes must belong to the circuit")
    if node in context:
        raise ValueError("minimality context must exclude the evaluated node")
    context = _ordered(context)
    without_node = f_circuit_without(evaluator, (*context, node))
    with_node = f_circuit_without(evaluator, context)
    raw_effect = abs(without_node.mean - with_node.mean)
    model = f_model(evaluator)
    if model.mean == 0:
        raise ZeroDivisionError("F(M) must be non-zero to normalize minimality")
    normalized = raw_effect / abs(model.mean)
    return MinimalityResult(
        node=node,
        context=context,
        without_node=without_node,
        with_node=with_node,
        raw_effect=raw_effect,
        raw_effect_standard_error=math.hypot(
            without_node.standard_error,
            with_node.standard_error,
        ),
        normalized_effect=normalized,
        threshold_fraction=threshold_fraction,
        passes_threshold=normalized >= threshold_fraction,
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
    validate_minimality_contexts(contexts)
    return tuple(
        evaluate_minimality(
            evaluator,
            node=node,
            context=contexts[node],
            threshold_fraction=threshold_fraction,
        )
        for node in published_nodes()
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
            standard_error = (
                float(values.std(unbiased=True) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            self._cache[key] = PerformanceEstimate(
                mean=float(values.mean()),
                standard_error=standard_error,
                n=len(values),
            )
        return self._cache[key]
