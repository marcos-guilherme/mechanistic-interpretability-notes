"""Pure contracts and statistics for causal J-lens fidelity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import random
from statistics import mean
from typing import Any

CAUSAL_JLENS_PROFILE = "qwen25_coder_7b_causal_jlens.json"
READOUT_LAYERS = (19, 22, 25, 26)
LENS_VERSIONS = ("v1", "v2")
READOUT_STEPS = 8
CONDITIONS = ("baseline", "ablation", "addition", "random_control")
DATASETS = ("mitre", "mitre_frr")
PRIMARY_LAYER = 25
TEXT_LEAK_KEYS = frozenset(
    {
        "prompt",
        "response",
        "response_text",
        "generated_text",
        "instruction",
        "content",
    }
)


def observation_key(record: Mapping[str, Any]) -> tuple[str, str, str, int, int, str]:
    return (
        str(record["dataset"]),
        str(record["example_id"]),
        str(record["condition"]),
        int(record["generation_step"]),
        int(record["layer"]),
        str(record["lens_version"]),
    )


def metric_block(ranks: Sequence[int]) -> dict[str, float | int | None]:
    if not ranks:
        return {
            "n": 0,
            "pass_at_1": None,
            "pass_at_5": None,
            "pass_at_10": None,
            "mean_rank": None,
        }
    return {
        "n": len(ranks),
        "pass_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "pass_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "pass_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }


def _pass_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(int(rank) <= k for rank in ranks) / len(ranks)


def _filter_observations(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    condition: str,
    layer: int,
    lens_version: str,
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in records
        if str(record["dataset"]) == dataset
        and str(record["condition"]) == condition
        and int(record["layer"]) == int(layer)
        and str(record["lens_version"]) == lens_version
    ]


def _ranks_by_prompt(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[str(record["example_id"])].append(int(record["rank"]))
    return dict(grouped)


def aggregate_fidelity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    prompts: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for record in records:
        key = (
            str(record["dataset"]),
            str(record["condition"]),
            str(record["lens_version"]),
            int(record["layer"]),
        )
        buckets[key].append(int(record["rank"]))
        prompts[key].add(str(record["example_id"]))
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for condition in CONDITIONS:
            output[dataset][condition] = {}
            for lens_version in LENS_VERSIONS:
                output[dataset][condition][lens_version] = {}
                for layer in READOUT_LAYERS:
                    key = (dataset, condition, lens_version, int(layer))
                    block = dict(metric_block(buckets.get(key, [])))
                    block["prompt_count"] = len(prompts.get(key, set()))
                    output[dataset][condition][lens_version][str(int(layer))] = block
    if compact_contains_text_leak(output):
        raise ValueError("compact aggregate contains raw text")
    return output


def cluster_bootstrap_difference(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    k: int,
    layer: int,
    lens_version: str,
    dataset: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if condition == "baseline":
        raise ValueError("condition must differ from baseline")
    baseline_records = _filter_observations(
        records,
        dataset=dataset,
        condition="baseline",
        layer=layer,
        lens_version=lens_version,
    )
    condition_records = _filter_observations(
        records,
        dataset=dataset,
        condition=condition,
        layer=layer,
        lens_version=lens_version,
    )
    baseline_by_prompt = _ranks_by_prompt(baseline_records)
    condition_by_prompt = _ranks_by_prompt(condition_records)
    prompt_ids = sorted(set(baseline_by_prompt) & set(condition_by_prompt))
    if not prompt_ids:
        raise ValueError("no paired prompts for bootstrap")

    def pooled_pass_at_k(selected: Sequence[str], by_prompt: dict[str, list[int]]) -> float:
        ranks = [
            rank
            for prompt_id in selected
            for rank in by_prompt[prompt_id]
        ]
        return _pass_at_k(ranks, k)

    point_baseline = _pass_at_k(
        [rank for ranks in baseline_by_prompt.values() for rank in ranks],
        k,
    )
    point_condition = _pass_at_k(
        [rank for ranks in condition_by_prompt.values() for rank in ranks],
        k,
    )
    estimate = point_condition - point_baseline

    generator = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        sampled = [generator.choice(prompt_ids) for _ in prompt_ids]
        baseline_rate = pooled_pass_at_k(sampled, baseline_by_prompt)
        condition_rate = pooled_pass_at_k(sampled, condition_by_prompt)
        diffs.append(condition_rate - baseline_rate)
    diffs.sort()
    lower_index = max(int(0.025 * resamples) - 1, 0)
    upper_index = min(int(0.975 * resamples), resamples - 1)
    return {
        "cluster_unit": "example_id",
        "dataset": dataset,
        "condition": condition,
        "baseline": "baseline",
        "layer": int(layer),
        "lens_version": lens_version,
        "metric": f"pass_at_{k}",
        "k": int(k),
        "resamples": int(resamples),
        "seed": int(seed),
        "prompt_count": len(prompt_ids),
        "estimate": estimate,
        "ci_lower": diffs[lower_index],
        "ci_upper": diffs[upper_index],
    }


def _baseline_refusal_labels(
    responses: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    labels: dict[tuple[str, str], bool] = {}
    for record in responses:
        if str(record["condition"]) != "baseline":
            continue
        labels[(str(record["dataset"]), str(record["example_id"]))] = bool(
            record["refusal"]
        )
    return labels


def _mean_rank_by_prompt(
    observations: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    condition: str,
    lens_version: str,
    layer: int,
) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in observations:
        if str(record["dataset"]) != dataset:
            continue
        if str(record["condition"]) != condition:
            continue
        if str(record["lens_version"]) != lens_version:
            continue
        if int(record["layer"]) != int(layer):
            continue
        grouped[str(record["example_id"])].append(int(record["rank"]))
    return {
        example_id: mean(ranks)
        for example_id, ranks in grouped.items()
        if ranks
    }


def summarize_behavior_changes(
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    layer: int = PRIMARY_LAYER,
) -> dict[str, Any]:
    baseline_labels = _baseline_refusal_labels(responses)
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for condition in CONDITIONS:
            if condition == "baseline":
                continue
            output[dataset][condition] = {"changed": {}, "unchanged": {}}
            changed_ids: list[str] = []
            unchanged_ids: list[str] = []
            for record in responses:
                if str(record["dataset"]) != dataset:
                    continue
                if str(record["condition"]) != condition:
                    continue
                example_id = str(record["example_id"])
                baseline_refusal = baseline_labels.get((dataset, example_id))
                if baseline_refusal is None:
                    continue
                if bool(record["refusal"]) != baseline_refusal:
                    changed_ids.append(example_id)
                else:
                    unchanged_ids.append(example_id)
            for group_name, prompt_ids in (
                ("changed", changed_ids),
                ("unchanged", unchanged_ids),
            ):
                output[dataset][condition][group_name] = {}
                for lens_version in LENS_VERSIONS:
                    baseline_means = _mean_rank_by_prompt(
                        observations,
                        dataset=dataset,
                        condition="baseline",
                        lens_version=lens_version,
                        layer=layer,
                    )
                    condition_means = _mean_rank_by_prompt(
                        observations,
                        dataset=dataset,
                        condition=condition,
                        lens_version=lens_version,
                        layer=layer,
                    )
                    deltas = [
                        condition_means[prompt_id] - baseline_means[prompt_id]
                        for prompt_id in prompt_ids
                        if prompt_id in baseline_means and prompt_id in condition_means
                    ]
                    block: dict[str, Any] = {
                        "prompt_count": len(prompt_ids),
                        "layer": int(layer),
                        "mean_rank_delta": (
                            mean(deltas) if deltas else None
                        ),
                    }
                    if deltas:
                        baseline_ranks = _filter_observations(
                            observations,
                            dataset=dataset,
                            condition="baseline",
                            layer=layer,
                            lens_version=lens_version,
                        )
                        condition_ranks = _filter_observations(
                            observations,
                            dataset=dataset,
                            condition=condition,
                            layer=layer,
                            lens_version=lens_version,
                        )
                        block["pass_at_5"] = {
                            "baseline": _pass_at_k(
                                [
                                    int(item["rank"])
                                    for item in baseline_ranks
                                    if str(item["example_id"]) in prompt_ids
                                ],
                                5,
                            ),
                            "condition": _pass_at_k(
                                [
                                    int(item["rank"])
                                    for item in condition_ranks
                                    if str(item["example_id"]) in prompt_ids
                                ],
                                5,
                            ),
                        }
                    output[dataset][condition][group_name][lens_version] = block
    if compact_contains_text_leak(output):
        raise ValueError("compact behavior summary contains raw text")
    return output


def compact_contains_text_leak(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return any(
            str(key).casefold() in TEXT_LEAK_KEYS
            or compact_contains_text_leak(value)
            for key, value in obj.items()
        )
    if isinstance(obj, (list, tuple)):
        return any(compact_contains_text_leak(item) for item in obj)
    return False