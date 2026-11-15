"""Pure baseline and causal abort gates for the Qwen refusal pilot.

Each function returns a structured pass/fail result with the threshold,
observed value, comparator, and a human-readable reason for every check.
These helpers have no CUDA, Transformers, or filesystem dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

RATE_LIMIT_DENOMINATOR = 10_000

STATUS_PASS = "pass"
STATUS_BASELINE_GATE_FAILED = "baseline_gate_failed"
STATUS_CAUSAL_GATE_FAILED = "causal_gate_failed"
STATUS_OPERATIONAL_ERROR = "operational_error"


@dataclass(frozen=True)
class GateCheck:
    """One numeric comparison used by a gate."""

    name: str
    passed: bool
    threshold: float
    observed: float
    comparator: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    """Aggregate result for a named abort gate."""

    name: str
    passed: bool
    checks: tuple[GateCheck, ...]

    @property
    def reason(self) -> str:
        failed = [check.reason for check in self.checks if not check.passed]
        if not failed:
            return f"{self.name} passed"
        return "; ".join(failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
        }


def to_fraction(value: float | int | Fraction) -> Fraction:
    """Convert a rate or count into a rational for exact boundary checks.

    Binary floats such as ``0.30 - 0.10`` are not equal to ``0.20``. Gate
    math therefore uses ``Fraction.limit_denominator`` so comparisons are
    exact for counts over finite samples.
    """

    if isinstance(value, Fraction):
        return value
    return Fraction(value).limit_denominator(RATE_LIMIT_DENOMINATOR)


def _compare(
    name: str,
    observed: float | Fraction,
    threshold: float | Fraction,
    comparator: str,
) -> GateCheck:
    observed_fraction = to_fraction(observed)
    threshold_fraction = to_fraction(threshold)
    if comparator == "gte":
        passed = observed_fraction >= threshold_fraction
        relation = ">="
    elif comparator == "lte":
        passed = observed_fraction <= threshold_fraction
        relation = "<="
    else:
        raise ValueError(f"unsupported comparator: {comparator}")
    if passed:
        reason = (
            f"{name}: observed {float(observed_fraction)} {relation} "
            f"{float(threshold_fraction)}"
        )
    else:
        reason = (
            f"{name}: observed {float(observed_fraction)} not {relation} "
            f"{float(threshold_fraction)}"
        )
    return GateCheck(
        name=name,
        passed=passed,
        threshold=float(threshold_fraction),
        observed=float(observed_fraction),
        comparator=comparator,
        reason=reason,
    )


def evaluate_baseline_gate(
    *,
    harmful_count: int,
    harmless_count: int,
    harmful_refusal_rate: float,
    harmless_refusal_rate: float,
    thresholds: dict[str, float],
) -> GateResult:
    """Require enough unfiltered items and a minimum generated refusal gap."""

    gap = to_fraction(harmful_refusal_rate) - to_fraction(harmless_refusal_rate)
    checks = (
        _compare(
            "min_valid_harmful",
            float(harmful_count),
            float(thresholds["min_valid_per_class"]),
            "gte",
        ),
        _compare(
            "min_valid_harmless",
            float(harmless_count),
            float(thresholds["min_valid_per_class"]),
            "gte",
        ),
        _compare(
            "min_refusal_gap",
            gap,
            float(thresholds["min_refusal_gap"]),
            "gte",
        ),
    )
    return GateResult(
        name="baseline",
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def evaluate_causal_gate(
    *,
    baseline_harmful_refusal_rate: float,
    baseline_harmless_refusal_rate: float,
    ablation_harmful_refusal_rate: float,
    addition_harmless_refusal_rate: float,
    random_control_harmful_refusal_rate: float,
    ablation_harmless_kl: float,
    thresholds: dict[str, float],
) -> GateResult:
    """Require directional causal effects on disjoint generated conditions."""

    harmful_drop = to_fraction(baseline_harmful_refusal_rate) - to_fraction(
        ablation_harmful_refusal_rate
    )
    harmless_increase = to_fraction(addition_harmless_refusal_rate) - to_fraction(
        baseline_harmless_refusal_rate
    )
    control_drop = to_fraction(baseline_harmful_refusal_rate) - to_fraction(
        random_control_harmful_refusal_rate
    )
    checks = (
        _compare(
            "min_harmful_ablation_drop",
            harmful_drop,
            float(thresholds["min_harmful_ablation_drop"]),
            "gte",
        ),
        _compare(
            "min_harmless_addition_increase",
            harmless_increase,
            float(thresholds["min_harmless_addition_increase"]),
            "gte",
        ),
        _compare(
            "max_random_control_harmful_drop",
            control_drop,
            float(thresholds["max_random_control_harmful_drop"]),
            "lte",
        ),
        _compare(
            "max_harmless_ablation_kl",
            to_fraction(ablation_harmless_kl),
            to_fraction(thresholds["max_harmless_ablation_kl"]),
            "lte",
        ),
    )
    return GateResult(
        name="causal",
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def build_pilot_evaluation(
    *,
    status: str,
    operational_success: bool,
    baseline_gate: GateResult | None,
    causal_gate: GateResult | None,
    run_id: str,
    stage: str | None = None,
    exception_category: str | None = None,
    message: str | None = None,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """Compact evaluation payload distinguishing pass, gate failure, and ops."""

    allowed = {
        STATUS_PASS,
        STATUS_BASELINE_GATE_FAILED,
        STATUS_CAUSAL_GATE_FAILED,
        STATUS_OPERATIONAL_ERROR,
    }
    if status not in allowed:
        raise ValueError(f"unknown pilot status: {status}")
    payload: dict[str, Any] = {
        "status": status,
        "operational_success": operational_success,
        "baseline_gate": baseline_gate.to_dict() if baseline_gate is not None else None,
        "causal_gate": causal_gate.to_dict() if causal_gate is not None else None,
        "run_id": run_id,
    }
    if stage is not None:
        payload["stage"] = stage
    if exception_category is not None:
        payload["exception_category"] = exception_category
    if message is not None:
        payload["message"] = message
    if implementation_commit is not None:
        payload["implementation_commit"] = implementation_commit
    return payload
