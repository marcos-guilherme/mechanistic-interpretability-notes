"""Pair-level bootstrap intervals and frozen sparse-steering Pareto gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
import statistics
from typing import Any


@dataclass(frozen=True)
class PairedBehaviorRow:
    """One response side with both compared conditions kept together."""

    pair_id: str
    side: str
    baseline_metrics: Mapping[str, float]
    intervention_metrics: Mapping[str, float]


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    estimate: float
    ci_lower: float
    ci_upper: float
    pair_count: int
    resamples: int
    seed: int

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric must be nonempty")
        if not all(
            math.isfinite(value)
            for value in (self.estimate, self.ci_lower, self.ci_upper)
        ):
            raise ValueError("bootstrap values must be finite")
        if self.ci_lower > self.ci_upper:
            raise ValueError("bootstrap interval is reversed")
        if self.pair_count <= 0 or self.resamples <= 0:
            raise ValueError("bootstrap counts must be positive")


@dataclass(frozen=True)
class ParetoGateResult:
    status: str
    primary: BootstrapResult
    improvement: BootstrapResult


def _percentile_bounds(draws: list[float]) -> tuple[float, float]:
    draws.sort()
    count = len(draws)
    lower_index = max(0, int(0.025 * count) - 1)
    upper_index = min(count - 1, int(0.975 * count) - 1)
    return draws[lower_index], draws[upper_index]


def paired_bootstrap(
    rows: Sequence[PairedBehaviorRow],
    metric: str,
    seed: int = 2029,
    resamples: int = 10_000,
) -> BootstrapResult:
    """Bootstrap pair IDs, carrying every side and condition in each unit."""
    if not metric:
        raise ValueError("metric must be nonempty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped_deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.pair_id:
            raise ValueError("pair_id must be nonempty")
        if not row.side:
            raise ValueError("side must be nonempty")
        if metric not in row.baseline_metrics:
            raise ValueError(f"baseline metric is missing: {metric}")
        if metric not in row.intervention_metrics:
            raise ValueError(f"intervention metric is missing: {metric}")
        baseline = float(row.baseline_metrics[metric])
        intervention = float(row.intervention_metrics[metric])
        if not math.isfinite(baseline) or not math.isfinite(intervention):
            raise ValueError("behavior values must be finite")
        grouped_deltas[row.pair_id].append(intervention - baseline)
    if not grouped_deltas:
        raise ValueError("rows must contain at least one pair")

    pair_ids = tuple(sorted(grouped_deltas))
    pair_means = {
        pair_id: statistics.fmean(grouped_deltas[pair_id])
        for pair_id in pair_ids
    }
    estimate = statistics.fmean(pair_means.values())
    rng = random.Random(seed)
    draws = [
        statistics.fmean(
            pair_means[pair_id]
            for pair_id in rng.choices(pair_ids, k=len(pair_ids))
        )
        for _ in range(resamples)
    ]
    ci_lower, ci_upper = _percentile_bounds(draws)
    return BootstrapResult(
        metric=metric,
        estimate=float(estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        pair_count=len(pair_ids),
        resamples=resamples,
        seed=seed,
    )


def evaluate_addition_gate(
    *,
    offensive_delta: BootstrapResult,
    defensive_improvement: BootstrapResult,
) -> ParetoGateResult:
    passed = (
        offensive_delta.ci_lower >= -0.05
        and defensive_improvement.ci_lower >= 0.20
    )
    return ParetoGateResult(
        status=(
            "pass_addition"
            if passed
            else "fail_selectivity_addition"
        ),
        primary=offensive_delta,
        improvement=defensive_improvement,
    )


def evaluate_ablation_gate(
    *,
    defensive_recovery_delta: BootstrapResult,
    offensive_retention_improvement: BootstrapResult,
) -> ParetoGateResult:
    passed = (
        defensive_recovery_delta.ci_lower >= -0.05
        and offensive_retention_improvement.ci_lower >= 0.20
    )
    return ParetoGateResult(
        status=(
            "pass_ablation"
            if passed
            else "fail_selectivity_ablation"
        ),
        primary=defensive_recovery_delta,
        improvement=offensive_retention_improvement,
    )


def evaluate_pareto_gates(
    addition: Mapping[str, BootstrapResult],
    ablation: Mapping[str, BootstrapResult],
) -> dict[str, Any]:
    """Evaluate both frozen interval gates and their joint Pareto status."""
    required_addition = {
        "offensive_delta",
        "defensive_improvement",
    }
    required_ablation = {
        "defensive_recovery_delta",
        "offensive_retention_improvement",
    }
    missing_addition = required_addition - set(addition)
    missing_ablation = required_ablation - set(ablation)
    if missing_addition or missing_ablation:
        raise ValueError(
            "missing Pareto metrics: "
            f"addition={sorted(missing_addition)}, "
            f"ablation={sorted(missing_ablation)}"
        )
    addition_result = evaluate_addition_gate(
        offensive_delta=addition["offensive_delta"],
        defensive_improvement=addition["defensive_improvement"],
    )
    ablation_result = evaluate_ablation_gate(
        defensive_recovery_delta=ablation[
            "defensive_recovery_delta"
        ],
        offensive_retention_improvement=ablation[
            "offensive_retention_improvement"
        ],
    )
    return {
        "status": (
            "pass_pareto"
            if addition_result.status == "pass_addition"
            and ablation_result.status == "pass_ablation"
            else "fail_pareto"
        ),
        "addition": addition_result,
        "ablation": ablation_result,
    }
