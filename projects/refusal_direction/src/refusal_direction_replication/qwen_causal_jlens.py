"""Pure contracts and statistics for causal J-lens fidelity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

CAUSAL_JLENS_PROFILE = "qwen25_coder_7b_causal_jlens.json"
READOUT_LAYERS = (19, 22, 25, 26)
LENS_VERSIONS = ("v1", "v2")
READOUT_STEPS = 8
CONDITIONS = ("baseline", "ablation", "addition", "random_control")
DATASETS = ("mitre", "mitre_frr")
PRODUCTION_DATASET_SIZE = 32
FROZEN_CAUSAL_PROFILE_SHA256 = (
    "46e2b88564190544ebe1ea5285b19c946f8beb0a2b130a7d0632712d24654e92"
)
PRIMARY_LAYER = 25
# Frozen provenance from committed canonical artifacts. Recalculate only if
# those source result files are deliberately replaced by a later approved run.
FROZEN_PREPARE_SHA256 = (
    "d5da5bb0e9dcd1f982ce3fbb748c0863057982d8cc4667b9ef68b7ba7ba17c1b"
)
FROZEN_COHORT_DIGEST = (
    "734f6ba50e28bd9958dc733a68ef70d6e443b22872012b5d922be531fba097e1"
)
FROZEN_V1_EVALUATION_SHA256 = (
    "6067ec0bd1dd1f0a42b9cf6cd27052800b5d6e0856d47d50984fcd085905a07a"
)
FROZEN_V2_EVALUATION_SHA256 = (
    "dd0f73d9e47f7efa6138e9885f275828090f5e6461f56e8b7b422b379583f3de"
)
FROZEN_V1_RUN_ID = "e23b9570-130e-4b9a-8000-fb91255b29d8"
FROZEN_V2_RUN_ID = "e35026e6-43e1-4014-8022-f4055aa6a58f"
FROZEN_SOURCE_DOSE_RUN_ID = "d76d0e7f-1e86-4953-8eac-de1c352708d2"
FROZEN_EOS_TOKEN_ID = 151645
CANONICAL_V1_EVALUATION = "results/qwen25_coder_7b_jlens/jlens_evaluation.json"
CANONICAL_V2_EVALUATION = "results/qwen25_coder_7b_jlens_v2/jlens_evaluation.json"
BOOTSTRAP_DEFAULT_RESAMPLES = 10_000
BOOTSTRAP_CI_LOWER_INDEX = 249
BOOTSTRAP_CI_UPPER_INDEX = 9749
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


def verify_causal_profile_digest(path: str | Path) -> None:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != FROZEN_CAUSAL_PROFILE_SHA256:
        raise ValueError(
            "production contract: causal profile sha256 digest differs from "
            "the independently frozen pin"
        )


def bootstrap_ci_indices(resamples: int) -> tuple[int, int]:
    if resamples != BOOTSTRAP_DEFAULT_RESAMPLES:
        raise ValueError(
            f"bootstrap CI indices are frozen for {BOOTSTRAP_DEFAULT_RESAMPLES} "
            f"resamples, received {resamples}"
        )
    return BOOTSTRAP_CI_LOWER_INDEX, BOOTSTRAP_CI_UPPER_INDEX


def selected_cohort_records(
    prepare: Mapping[str, Any],
) -> list[tuple[str, str, str, int]]:
    records: list[tuple[str, str, str, int]] = []
    results = prepare.get("results") or {}
    for dataset in DATASETS:
        block = results.get(dataset) or {}
        ids = list(block.get("example_ids") or [])
        hashes = list(block.get("content_hashes") or [])
        indices = list(block.get("source_indices") or [])
        if len(ids) != len(hashes) or len(hashes) != len(indices):
            raise ValueError(f"{dataset} cohort fields have unequal length")
        for example_id, content_hash, source_index in zip(
            ids, hashes, indices, strict=True
        ):
            records.append(
                (str(dataset), str(example_id), str(content_hash), int(source_index))
            )
    return records


def cohort_digest(prepare: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        [list(item) for item in selected_cohort_records(prepare)],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _paired_prompt_ids(
    baseline_by_prompt: Mapping[str, Sequence[int]],
    condition_by_prompt: Mapping[str, Sequence[int]],
) -> list[str]:
    return sorted(set(baseline_by_prompt) & set(condition_by_prompt))


def _prompts_with_ranks(
    records: Sequence[Mapping[str, Any]],
    *,
    prompt_ids: set[str],
) -> set[str]:
    by_prompt = _ranks_by_prompt(records)
    return {
        prompt_id
        for prompt_id in prompt_ids
        if prompt_id in by_prompt and by_prompt[prompt_id]
    }


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
    prompt_ids = _paired_prompt_ids(baseline_by_prompt, condition_by_prompt)
    if not prompt_ids:
        raise ValueError("no paired prompts for bootstrap")

    def pooled_pass_at_k(selected: Sequence[str], by_prompt: dict[str, list[int]]) -> float:
        ranks = [
            rank
            for prompt_id in selected
            for rank in by_prompt[prompt_id]
        ]
        return _pass_at_k(ranks, k)

    paired_baseline_ranks = [
        rank
        for prompt_id in prompt_ids
        for rank in baseline_by_prompt[prompt_id]
    ]
    paired_condition_ranks = [
        rank
        for prompt_id in prompt_ids
        for rank in condition_by_prompt[prompt_id]
    ]
    estimate = _pass_at_k(paired_condition_ranks, k) - _pass_at_k(
        paired_baseline_ranks, k
    )

    generator = random.Random(seed)
    diffs: list[float] = []
    for _ in range(resamples):
        sampled = [generator.choice(prompt_ids) for _ in prompt_ids]
        baseline_rate = pooled_pass_at_k(sampled, baseline_by_prompt)
        condition_rate = pooled_pass_at_k(sampled, condition_by_prompt)
        diffs.append(condition_rate - baseline_rate)
    diffs.sort()
    lower_index, upper_index = bootstrap_ci_indices(resamples)
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
        "ci_lower_index": lower_index,
        "ci_upper_index": upper_index,
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
                group_prompt_ids = set(prompt_ids)
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
                    rank_paired_ids = sorted(
                        prompt_id
                        for prompt_id in prompt_ids
                        if prompt_id in baseline_means and prompt_id in condition_means
                    )
                    deltas = [
                        condition_means[prompt_id] - baseline_means[prompt_id]
                        for prompt_id in rank_paired_ids
                    ]
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
                    pass_at_5_ids = sorted(
                        _prompts_with_ranks(baseline_ranks, prompt_ids=group_prompt_ids)
                        & _prompts_with_ranks(condition_ranks, prompt_ids=group_prompt_ids)
                    )
                    block: dict[str, Any] = {
                        "prompt_count": len(prompt_ids),
                        "rank_paired_prompt_count": len(rank_paired_ids),
                        "pass_at_5_prompt_count": len(pass_at_5_ids),
                        "layer": int(layer),
                        "mean_rank_delta": (
                            mean(deltas) if deltas else None
                        ),
                    }
                    if pass_at_5_ids:
                        pass_id_set = set(pass_at_5_ids)
                        block["pass_at_5"] = {
                            "baseline": _pass_at_k(
                                [
                                    int(item["rank"])
                                    for item in baseline_ranks
                                    if str(item["example_id"]) in pass_id_set
                                ],
                                5,
                            ),
                            "condition": _pass_at_k(
                                [
                                    int(item["rank"])
                                    for item in condition_ranks
                                    if str(item["example_id"]) in pass_id_set
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