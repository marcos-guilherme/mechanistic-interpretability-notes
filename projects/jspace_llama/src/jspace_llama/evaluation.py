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
    return bool(
        degradation["empty"]
        or degradation["repetition"]
        or degradation["incoherent"]
    )


def _uniquely_parsed_as(
    classification: dict[str, Any],
    expected: str,
) -> bool:
    answer = classification["answer_parse"]
    return answer["status"] == "unique" and answer["value"] == expected


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
        "baseline_unique_eight": _uniquely_parsed_as(baseline, "eight"),
        "baseline_not_degraded": not _degraded(baseline),
        "noop_unique_eight": _uniquely_parsed_as(noop, "eight"),
        "noop_not_degraded": not _degraded(noop),
        "random_unique_non_six": (
            random["answer_parse"]["status"] == "unique"
            and random["answer_parse"]["value"] != "six"
        ),
        "random_not_degraded": not _degraded(random),
        "target_unique_six": _uniquely_parsed_as(target, "six"),
        "target_no_ant": not bool(target["target_entity_leakage"]),
        "target_not_degraded": not _degraded(target),
    }
    reason_by_gate = {
        "readout_spider_present": "readout_spider_missing",
        "baseline_unique_eight": "baseline_answer_invalid",
        "baseline_not_degraded": "baseline_degraded",
        "noop_unique_eight": "noop_answer_invalid",
        "noop_not_degraded": "noop_degraded",
        "random_unique_non_six": "random_answer_invalid",
        "random_not_degraded": "random_degraded",
        "target_unique_six": "target_answer_invalid",
        "target_no_ant": "target_contains_ant",
        "target_not_degraded": "target_degraded",
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
        "answer_parse": {
            "status": str(classification["answer_parse"]["status"]),
            "value": classification["answer_parse"]["value"],
        },
        "target_entity_leakage": bool(
            classification["target_entity_leakage"]
        ),
        "degradation": {
            "empty": bool(classification["degradation"]["empty"]),
            "repetition": bool(
                classification["degradation"]["repetition"]
            ),
            "incoherent": bool(
                classification["degradation"]["incoherent"]
            ),
        },
    }


def evaluate_experiment(
    raw_diagnostics: dict[str, Any],
    *,
    expected_prompt_names: list[str] | None = None,
    expected_alphas: list[float] | None = None,
) -> dict[str, Any]:
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

    prompt_order = (
        list(expected_prompt_names)
        if expected_prompt_names is not None
        else sorted(
            {prompt for prompt, _alpha in grouped},
            key=lambda prompt: (prompt != PRIMARY_PROMPT, prompt),
        )
    )
    alpha_order = (
        [float(alpha) for alpha in expected_alphas]
        if expected_alphas is not None
        else sorted({alpha for _prompt, alpha in grouped})
    )
    if (
        PRIMARY_PROMPT not in prompt_order
        or len(set(prompt_order)) != len(prompt_order)
        or not alpha_order
        or len(set(alpha_order)) != len(alpha_order)
    ):
        raise EvaluationDataError(
            "expected prompt and alpha plan is invalid"
        )
    expected_pairs = {
        (prompt, alpha)
        for prompt in prompt_order
        for alpha in alpha_order
    }
    if set(grouped) != expected_pairs:
        raise EvaluationDataError(
            "observations do not match the expected prompt and alpha plan"
        )

    primary_alphas: list[dict[str, Any]] = []
    passing_alphas: list[float] = []
    for alpha in alpha_order:
        conditions = grouped[(PRIMARY_PROMPT, alpha)]
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

    secondary_prompt_alphas: list[dict[str, Any]] = []
    for prompt in prompt_order:
        if prompt != PRIMARY_PROMPT:
            for alpha in alpha_order:
                conditions = grouped[(prompt, alpha)]
                missing = [
                    name for name in CONDITIONS if name not in conditions
                ]
                if missing:
                    raise EvaluationDataError(
                        f"missing paraphrase condition(s) for {prompt} "
                        f"at alpha {alpha}: " + ", ".join(missing)
                    )
                gates, _ = evaluate_conditions(
                    conditions,
                    spider_present=spider_present,
                )
                secondary_prompt_alphas.append(
                    {
                        "prompt_name": prompt,
                        "alpha": alpha,
                        "conditions": conditions,
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )

    secondary_names = [
        prompt for prompt in prompt_order if prompt != PRIMARY_PROMPT
    ]
    secondary_passing = sum(
        int(item["passed"]) for item in secondary_prompt_alphas
    )

    return {
        "primary": {
            "prompt_name": PRIMARY_PROMPT,
            "readout": {
                "evaluated": True,
                "spider_present": spider_present,
            },
            "alphas": primary_alphas,
            "passing_alphas": passing_alphas,
            "passing_alpha_count": len(passing_alphas),
        },
        "paraphrases": {
            "prompt_count": len(secondary_names),
            "passing_prompt_alpha_count": secondary_passing,
            "prompt_alphas": secondary_prompt_alphas,
        },
    }
