from dataclasses import replace
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

FROZEN_PAIR_IDS = tuple(f"pair-{index:02d}" for index in range(64))


def interval(
    *,
    metric: str,
    lower: float,
    estimate: float,
    upper: float,
    pair_ids: tuple[str, ...] = FROZEN_PAIR_IDS,
) -> BootstrapResult:
    return BootstrapResult(
        metric=metric,
        estimate=estimate,
        ci_lower=lower,
        ci_upper=upper,
        pair_count=len(pair_ids),
        resamples=10_000,
        seed=2029,
        pair_ids=pair_ids,
    )


def test_paired_bootstrap_resamples_pair_ids_not_individual_sides() -> None:
    rows = (
        PairedBehaviorRow("p1", "offensive", "dense", False),
        PairedBehaviorRow("p1", "offensive", "sparse_filtered", True),
        PairedBehaviorRow("p1", "defensive", "dense", True),
        PairedBehaviorRow("p1", "defensive", "sparse_filtered", False),
        PairedBehaviorRow("p2", "offensive", "dense", False),
        PairedBehaviorRow("p2", "offensive", "sparse_filtered", False),
        PairedBehaviorRow("p2", "defensive", "dense", False),
        PairedBehaviorRow("p2", "defensive", "sparse_filtered", False),
    )
    seed = 17
    resamples = 8
    rng = random.Random(seed)
    pair_contrasts = {"p1": 1.0, "p2": 0.0}
    expected_draws = sorted(
        sum(
            pair_contrasts[pair_id]
            for pair_id in rng.choices(("p1", "p2"), k=2)
        )
        / 2
        for _ in range(resamples)
    )

    result = paired_bootstrap(
        rows,
        "offensive_delta",
        seed=seed,
        resamples=resamples,
    )
    defensive = paired_bootstrap(
        rows,
        "defensive_improvement",
        seed=seed,
        resamples=resamples,
    )
    recovery = paired_bootstrap(
        rows,
        "defensive_recovery_delta",
        seed=seed,
        resamples=resamples,
    )
    retention = paired_bootstrap(
        rows,
        "offensive_retention_improvement",
        seed=seed,
        resamples=resamples,
    )

    assert result.estimate == 0.5
    assert result.ci_lower == expected_draws[0]
    assert result.ci_upper == expected_draws[6]
    assert result.pair_count == 2
    assert result.pair_ids == ("p1", "p2")
    assert defensive.estimate == 0.5
    assert defensive.ci_lower == result.ci_lower
    assert defensive.ci_upper == result.ci_upper
    assert recovery.estimate == defensive.estimate
    assert retention.estimate == result.estimate


def test_paired_bootstrap_has_exact_local_seed_semantics() -> None:
    rows = (
        PairedBehaviorRow("p1", "offensive", "dense", False),
        PairedBehaviorRow("p1", "offensive", "sparse_filtered", True),
        PairedBehaviorRow("p1", "defensive", "dense", True),
        PairedBehaviorRow("p1", "defensive", "sparse_filtered", False),
    )
    random.seed(81)
    state = random.getstate()

    first = paired_bootstrap(
        rows,
        "offensive_delta",
        seed=2029,
        resamples=50,
    )
    second = paired_bootstrap(
        rows,
        "offensive_delta",
        seed=2029,
        resamples=50,
    )

    assert first == second
    assert random.getstate() == state


def test_bootstrap_uses_frozen_10000_percentile_indices() -> None:
    draws = [float(value) for value in reversed(range(10_000))]

    assert _percentile_bounds(draws) == (249.0, 9749.0)


@pytest.mark.parametrize(
    "rows",
    (
        (
            PairedBehaviorRow("p1", "offensive", "dense", False),
            PairedBehaviorRow("p1", "offensive", "sparse_filtered", True),
            PairedBehaviorRow("p1", "defensive", "dense", True),
        ),
        (
            PairedBehaviorRow("p1", "offensive", "dense", False),
            PairedBehaviorRow("p1", "offensive", "dense", True),
            PairedBehaviorRow("p1", "defensive", "dense", True),
            PairedBehaviorRow("p1", "defensive", "sparse_filtered", False),
        ),
    ),
)
def test_paired_bootstrap_requires_complete_unique_pair_structure(
    rows: tuple[PairedBehaviorRow, ...],
) -> None:
    with pytest.raises(ValueError, match="pair structure"):
        paired_bootstrap(rows, "offensive_delta", resamples=4)


def test_paired_bootstrap_rejects_nonboolean_outcome() -> None:
    rows = (
        PairedBehaviorRow("p1", "offensive", "dense", 1),
        PairedBehaviorRow("p1", "offensive", "sparse_filtered", True),
        PairedBehaviorRow("p1", "defensive", "dense", True),
        PairedBehaviorRow("p1", "defensive", "sparse_filtered", False),
    )

    with pytest.raises(ValueError, match="boolean"):
        paired_bootstrap(rows, "offensive_delta", resamples=4)


def test_addition_gate_requires_both_bounds() -> None:
    result = evaluate_addition_gate(
        offensive_delta=interval(
            metric="offensive_delta",
            lower=-0.04,
            estimate=-0.01,
            upper=0.02,
        ),
        defensive_improvement=interval(
            metric="defensive_improvement",
            lower=0.21,
            estimate=0.30,
            upper=0.38,
        ),
    )

    assert result.status == "pass_addition"


def test_addition_gate_fails_point_gain_with_weak_interval() -> None:
    result = evaluate_addition_gate(
        offensive_delta=interval(
            metric="offensive_delta",
            lower=-0.04,
            estimate=-0.01,
            upper=0.02,
        ),
        defensive_improvement=interval(
            metric="defensive_improvement",
            lower=0.18,
            estimate=0.30,
            upper=0.39,
        ),
    )

    assert result.status == "fail_selectivity_addition"


def test_ablation_gate_uses_strict_interval_bounds() -> None:
    passing = evaluate_ablation_gate(
        defensive_recovery_delta=interval(
            metric="defensive_recovery_delta",
            lower=-0.05,
            estimate=0.0,
            upper=0.05,
        ),
        offensive_retention_improvement=interval(
            metric="offensive_retention_improvement",
            lower=0.20,
            estimate=0.25,
            upper=0.30,
        ),
    )
    failing = evaluate_ablation_gate(
        defensive_recovery_delta=interval(
            metric="defensive_recovery_delta",
            lower=-0.051,
            estimate=0.0,
            upper=0.05,
        ),
        offensive_retention_improvement=interval(
            metric="offensive_retention_improvement",
            lower=0.30,
            estimate=0.35,
            upper=0.40,
        ),
    )

    assert passing.status == "pass_ablation"
    assert failing.status == "fail_selectivity_ablation"


def test_pareto_gates_keep_addition_and_ablation_separate() -> None:
    result = evaluate_pareto_gates(
        addition={
            "offensive_delta": interval(
                metric="offensive_delta",
                lower=-0.05,
                estimate=-0.01,
                upper=0.01,
            ),
            "defensive_improvement": interval(
                metric="defensive_improvement",
                lower=0.20,
                estimate=0.30,
                upper=0.40,
            ),
        },
        ablation={
            "defensive_recovery_delta": interval(
                metric="defensive_recovery_delta",
                lower=-0.05,
                estimate=0.0,
                upper=0.05,
            ),
            "offensive_retention_improvement": interval(
                metric="offensive_retention_improvement",
                lower=0.20,
                estimate=0.25,
                upper=0.30,
            ),
        },
    )

    assert set(result) == {"addition", "ablation"}
    assert result["addition"].status == "pass_addition"
    assert result["ablation"].status == "pass_ablation"


@pytest.mark.parametrize(
    "mutation",
    ("metric", "seed", "resamples", "pair_count"),
)
def test_addition_gate_rejects_noncanonical_bootstrap_results(
    mutation: str,
) -> None:
    offensive = interval(
        metric="offensive_delta",
        lower=-0.04,
        estimate=-0.01,
        upper=0.02,
    )
    if mutation == "metric":
        bad_result = replace(offensive, metric="wrong_metric")
    elif mutation == "seed":
        bad_result = replace(offensive, seed=1)
    elif mutation == "resamples":
        bad_result = replace(offensive, resamples=9999)
    else:
        bad_result = replace(
            offensive,
            pair_count=63,
            pair_ids=FROZEN_PAIR_IDS[:-1],
        )
    defensive = interval(
        metric="defensive_improvement",
        lower=0.21,
        estimate=0.30,
        upper=0.38,
    )

    with pytest.raises(ValueError, match="canonical"):
        evaluate_addition_gate(
            offensive_delta=bad_result,
            defensive_improvement=defensive,
        )


def test_gate_rejects_mismatched_pair_coverage() -> None:
    changed_coverage = (*FROZEN_PAIR_IDS[:-1], "pair-99")

    with pytest.raises(ValueError, match="coverage"):
        evaluate_addition_gate(
            offensive_delta=interval(
                metric="offensive_delta",
                lower=-0.04,
                estimate=-0.01,
                upper=0.02,
            ),
            defensive_improvement=interval(
                metric="defensive_improvement",
                lower=0.21,
                estimate=0.30,
                upper=0.38,
                pair_ids=changed_coverage,
            ),
        )
