"""Compact, deterministic evaluation of aggregate experiment classifications."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from jspace_llama.runtime import JSpaceExperimentError


PRIMARY_PROMPT = "main"
CONDITIONS = ("baseline", "noop", "random", "target")


class EvaluationDataError(JSpaceExperimentError):
    pass


def _degraded(classification: dict[str, Any]) -> bool:
    degradation = classification["degradation"]
    return bool(degradation["empty"] or degradation["repetition"])


def evaluate_conditions(
    classifications: dict[str, dict[str, Any]],
    *,
    spider_present: bool,
) -> tuple[dict[str, bool], list[str]]:
    baseline = classifications["baseline"]
    noop = classifications["noop"]
    random = classifications["random"]
    target = classifications["target"]
    gates = {
        "readout_spider_present": spider_present,
        "baseline_eight": bool(baseline["answer_evidence"]["eight"]),
        "baseline_not_degraded": not _degraded(baseline),
        "noop_eight": bool(noop["answer_evidence"]["eight"]),
        "noop_not_six": not bool(noop["answer_evidence"]["six"]),
        "random_not_comparable_six_reroute": not (
            random["answer_evidence"]["six"]
            and not random["answer_evidence"]["eight"]
        ),
        "random_not_degraded": not _degraded(random),
        "target_six": bool(target["answer_evidence"]["six"]),
        "target_not_eight": not bool(target["answer_evidence"]["eight"]),
        "target_no_ant": not bool(target["target_entity_leakage"]),
        "target_nonempty": not bool(target["degradation"]["empty"]),
        "target_nonrepetitive": not bool(
            target["degradation"]["repetition"]
        ),
    }
    reason_by_gate = {
        "readout_spider_present": "readout_spider_missing",
        "baseline_eight": "baseline_missing_eight",
        "baseline_not_degraded": "baseline_degraded",
        "noop_eight": "noop_missing_eight",
        "noop_not_six": "noop_contains_six",
        "random_not_comparable_six_reroute": (
            "random_comparable_six_reroute"
        ),
        "random_not_degraded": "random_degraded",
        "target_six": "target_missing_six",
        "target_not_eight": "target_contains_eight",
        "target_no_ant": "target_contains_ant",
        "target_nonempty": "target_empty",
        "target_nonrepetitive": "target_repetitive",
    }
    reasons = [
        reason_by_gate[name] for name, passed in gates.items() if not passed
    ]
    return gates, reasons


def _compact_classification(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_evidence": {
            "eight": bool(classification["answer_evidence"]["eight"]),
            "six": bool(classification["answer_evidence"]["six"]),
        },
        "target_entity_leakage": bool(
            classification["target_entity_leakage"]
        ),
        "degradation": {
            "empty": bool(classification["degradation"]["empty"]),
            "repetition": bool(
                classification["degradation"]["repetition"]
            ),
        },
    }


def evaluate_experiment(raw_diagnostics: dict[str, Any]) -> dict[str, Any]:
    try:
        observations = raw_diagnostics["experiment"]["observations"]
        spider_present = bool(
            raw_diagnostics["readout"]["spider"]["present"]
        )
    except (KeyError, TypeError) as error:
        raise EvaluationDataError(
            "missing required readout or experiment aggregate"
        ) from error
    grouped: dict[
        tuple[str, float], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for item in observations:
        key = (str(item["prompt"]["name"]), float(item["parameters"]["alpha"]))
        condition = str(item["condition"])
        if condition in grouped[key]:
            raise EvaluationDataError(
                f"duplicate {condition} observation for {key[0]} alpha {key[1]}"
            )
        grouped[key][condition] = _compact_classification(
            item["classification"]
        )

    primary_alphas: list[dict[str, Any]] = []
    passing_alphas: list[float] = []
    for (prompt, alpha), conditions in sorted(grouped.items()):
        if prompt != PRIMARY_PROMPT:
            continue
        missing = [name for name in CONDITIONS if name not in conditions]
        if missing:
            raise EvaluationDataError(
                f"missing primary condition(s) at alpha {alpha}: "
                + ", ".join(missing)
            )
        else:
            gates, reasons = evaluate_conditions(
                conditions,
                spider_present=spider_present,
            )
            passed = all(gates.values())
        primary_alphas.append(
            {
                "alpha": alpha,
                "conditions": conditions,
                "gates": gates,
                "passed": passed,
                "reasons": reasons,
            }
        )
        if passed:
            passing_alphas.append(alpha)

    if not primary_alphas:
        raise EvaluationDataError("missing primary main-prompt observations")

    secondary_passing = 0
    for (prompt, _alpha), conditions in grouped.items():
        if prompt == PRIMARY_PROMPT:
            continue
        if all(name in conditions for name in CONDITIONS):
            gates, _ = evaluate_conditions(
                conditions,
                spider_present=spider_present,
            )
            secondary_passing += int(all(gates.values()))

    return {
        "primary": {
            "prompt_name": PRIMARY_PROMPT,
            "readout": {"spider_present": spider_present},
            "alphas": primary_alphas,
            "passing_alphas": passing_alphas,
            "passing_alpha_count": len(passing_alphas),
        },
        "paraphrases": {
            "prompt_count": len(
                {
                    prompt
                    for prompt, _alpha in grouped
                    if prompt != PRIMARY_PROMPT
                }
            ),
            "passing_prompt_alpha_count": secondary_passing,
        },
    }
