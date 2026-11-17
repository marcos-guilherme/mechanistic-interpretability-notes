import random

import pytest

from refusal_direction_replication.qwen_sparse_statistics import (
    BootstrapResult,
    PairedBehaviorRow,
    _percentile_bounds,
    evaluate_ablation_gate,
    evaluate_addition_gate,
    evaluate_pareto_gates,
    paired_bootstrap,
)


def interval(
    *,
    lower: float,
    estimate: float,
    upper: float,
) -> BootstrapResult:
    return BootstrapResult(
        metric="score",
        estimate=estimate,
        ci_lower=lower,
        ci_upper=upper,
        pair_count=4,
        resamples=10_000,
        seed=2029,
    )


def test_paired_bootstrap_resamples_pair_ids_not_individual_sides() -> None:
    rows = (
        PairedBehaviorRow("p1", "offensive", {"score": 0.0}, {"score": 2.0}),
        PairedBehaviorRow("p1", "defensive", {"score": 1.0}, {"score": 5.0}),
        PairedBehaviorRow("p2", "offensive", {"score": 2.0}, {"score": 8.0}),
        PairedBehaviorRow("p2", "defensive", {"score": 4.0}, {"score": 12.0}),
    )
    seed = 17
    resamples = 8
    rng = random.Random(seed)
    pair_means = {"p1": 3.0, "p2": 7.0}
    expected_draws = sorted(
        sum(
            pair_means[pair_id]
            for pair_id in rng.choices(("p1", "p2"), k=2)
        )
        / 2
        for _ in range(resamples)
    )

    result = paired_bootstrap(
        rows,
        "score",
        seed=seed,
        resamples=resamples,
    )

    assert result.estimate == 5.0
    assert result.ci_lower == expected_draws[0]
    assert result.ci_upper == expected_draws[6]
    assert result.pair_count == 2


def test_paired_bootstrap_has_exact_local_seed_semantics() -> None:
    rows = (
        PairedBehaviorRow("p1", "offensive", {"m": 0.0}, {"m": 1.0}),
        PairedBehaviorRow("p2", "offensive", {"m": 0.0}, {"m": 3.0}),
    )
    random.seed(81)
    state = random.getstate()

    first = paired_bootstrap(rows, "m", seed=2029, resamples=50)
    second = paired_bootstrap(rows, "m", seed=2029, resamples=50)

    assert first == second
    assert random.getstate() == state


def test_bootstrap_uses_frozen_10000_percentile_indices() -> None:
    draws = [float(value) for value in reversed(range(10_000))]

    assert _percentile_bounds(draws) == (249.0, 9749.0)


def test_paired_bootstrap_rejects_nonfinite_values() -> None:
    rows = (
        PairedBehaviorRow(
            "p1",
            "offensive",
            {"score": 0.0},
            {"score": float("nan")},
        ),
    )

    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap(rows, "score", resamples=4)


def test_addition_gate_requires_both_bounds() -> None:
    result = evaluate_addition_gate(
        offensive_delta=interval(lower=-0.04, estimate=-0.01, upper=0.02),
        defensive_improvement=interval(
            lower=0.21,
            estimate=0.30,
            upper=0.38,
        ),
    )

    assert result.status == "pass_addition"


def test_addition_gate_fails_point_gain_with_weak_interval() -> None:
    result = evaluate_addition_gate(
        offensive_delta=interval(lower=-0.04, estimate=-0.01, upper=0.02),
        defensive_improvement=interval(
            lower=0.18,
            estimate=0.30,
            upper=0.39,
        ),
    )

    assert result.status == "fail_selectivity_addition"


def test_ablation_gate_uses_strict_interval_bounds() -> None:
    passing = evaluate_ablation_gate(
        defensive_recovery_delta=interval(
            lower=-0.05,
            estimate=0.0,
            upper=0.05,
        ),
        offensive_retention_improvement=interval(
            lower=0.20,
            estimate=0.25,
            upper=0.30,
        ),
    )
    failing = evaluate_ablation_gate(
        defensive_recovery_delta=interval(
            lower=-0.051,
            estimate=0.0,
            upper=0.05,
        ),
        offensive_retention_improvement=interval(
            lower=0.30,
            estimate=0.35,
            upper=0.40,
        ),
    )

    assert passing.status == "pass_ablation"
    assert failing.status == "fail_selectivity_ablation"


def test_pareto_gates_require_named_metrics_and_both_passes() -> None:
    result = evaluate_pareto_gates(
        addition={
            "offensive_delta": interval(
                lower=-0.05,
                estimate=-0.01,
                upper=0.01,
            ),
            "defensive_improvement": interval(
                lower=0.20,
                estimate=0.30,
                upper=0.40,
            ),
        },
        ablation={
            "defensive_recovery_delta": interval(
                lower=-0.05,
                estimate=0.0,
                upper=0.05,
            ),
            "offensive_retention_improvement": interval(
                lower=0.20,
                estimate=0.25,
                upper=0.30,
            ),
        },
    )

    assert result["status"] == "pass_pareto"
    assert result["addition"].status == "pass_addition"
    assert result["ablation"].status == "pass_ablation"
