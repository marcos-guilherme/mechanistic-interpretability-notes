"""Model-independent helpers for the Qwen dose follow-up."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from .data import (
    PromptExample,
    content_hashes_for_examples,
    load_local_prompt_corpus,
    prompt_content_hash,
)


DOSE_SELECTION_SIZE = 32
DOSE_CONFIRMATION_SIZE = 32
DEFAULT_MIN_HARMFUL_ABLATION_DROP = 0.20
DEFAULT_MAX_HARMLESS_ABLATION_KL = 0.10
DEFAULT_MIN_HARMLESS_ADDITION_INCREASE = 0.20


@dataclass(frozen=True)
class DoseSplits:
    harmful_dose_selection: tuple[PromptExample, ...]
    harmless_dose_selection: tuple[PromptExample, ...]
    harmful_dose_confirmation: tuple[PromptExample, ...]
    harmless_dose_confirmation: tuple[PromptExample, ...]


def collect_original_usage(
    results_dir: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        results = payload.get("results") or {}
        if not isinstance(results, dict):
            continue
        for group in (results.get("example_ids") or {}).values():
            ids.update(str(item) for item in group)
        for group in (results.get("content_hashes") or {}).values():
            hashes.update(str(item) for item in group)
    return frozenset(ids), frozenset(hashes)


def original_source_revision(results_dir: Path) -> str:
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        provenance = payload.get("provenance") or {}
        revision = provenance.get("upstream_commit")
        if revision:
            return str(revision)
    raise ValueError("original results do not record a source revision")


def eligible_examples(
    examples: Sequence[PromptExample],
    *,
    used_ids: Set[str],
    used_hashes: Set[str],
) -> list[PromptExample]:
    leftover: list[PromptExample] = []
    for example in examples:
        if example.id in used_ids:
            continue
        if prompt_content_hash(example.text) in used_hashes:
            continue
        leftover.append(example)
    return leftover


def sample_dose_halves(
    examples: Sequence[PromptExample],
    *,
    seed: int,
    selection_size: int = DOSE_SELECTION_SIZE,
    confirmation_size: int = DOSE_CONFIRMATION_SIZE,
) -> tuple[list[PromptExample], list[PromptExample]]:
    required = selection_size + confirmation_size
    if len(examples) < required:
        raise ValueError(f"need {required} examples, received {len(examples)}")
    sampled = random.Random(seed).sample(list(examples), required)
    return sampled[:selection_size], sampled[selection_size:]


def compact_split_metadata(
    *,
    harmful_selection: Sequence[PromptExample],
    harmless_selection: Sequence[PromptExample],
    harmful_confirmation: Sequence[PromptExample],
    harmless_confirmation: Sequence[PromptExample],
    seed: int,
    source_revision: str,
) -> dict[str, Any]:
    groups = {
        "harmful_dose_selection": harmful_selection,
        "harmless_dose_selection": harmless_selection,
        "harmful_dose_confirmation": harmful_confirmation,
        "harmless_dose_confirmation": harmless_confirmation,
    }
    return {
        "example_ids": {
            name: [example.id for example in items]
            for name, items in groups.items()
        },
        "content_hashes": {
            name: content_hashes_for_examples(items)
            for name, items in groups.items()
        },
        "counts": {name: len(items) for name, items in groups.items()},
        "seed": seed,
        "source_revision": source_revision,
    }


def _require_leftovers(
    examples: Sequence[PromptExample],
    *,
    label: str,
    required: int,
) -> None:
    if len(examples) < required:
        raise ValueError(
            f"need {required} leftover {label} examples, received {len(examples)}"
        )


def _split_groups(splits: DoseSplits) -> dict[str, Sequence[PromptExample]]:
    return {
        "harmful_dose_selection": splits.harmful_dose_selection,
        "harmless_dose_selection": splits.harmless_dose_selection,
        "harmful_dose_confirmation": splits.harmful_dose_confirmation,
        "harmless_dose_confirmation": splits.harmless_dose_confirmation,
    }


def _assert_disjoint_splits(
    splits: DoseSplits,
    *,
    used_ids: Set[str],
    used_hashes: Set[str],
    selection_size: int,
    confirmation_size: int,
) -> None:
    groups = _split_groups(splits)
    new_ids = {example.id for items in groups.values() for example in items}
    new_hashes = {
        prompt_content_hash(example.text)
        for items in groups.values()
        for example in items
    }
    if not used_ids.isdisjoint(new_ids):
        raise ValueError("dose splits overlap original example ids")
    if not used_hashes.isdisjoint(new_hashes):
        raise ValueError("dose splits overlap original content hashes")
    selection_ids = {
        example.id
        for example in (
            *splits.harmful_dose_selection,
            *splits.harmless_dose_selection,
        )
    }
    confirmation_ids = {
        example.id
        for example in (
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    selection_hashes = {
        prompt_content_hash(example.text)
        for example in (
            *splits.harmful_dose_selection,
            *splits.harmless_dose_selection,
        )
    }
    confirmation_hashes = {
        prompt_content_hash(example.text)
        for example in (
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    if not selection_ids.isdisjoint(confirmation_ids):
        raise ValueError("dose selection and confirmation ids overlap")
    if not selection_hashes.isdisjoint(confirmation_hashes):
        raise ValueError("dose selection and confirmation hashes overlap")
    expected_counts = {
        "harmful_dose_selection": selection_size,
        "harmless_dose_selection": selection_size,
        "harmful_dose_confirmation": confirmation_size,
        "harmless_dose_confirmation": confirmation_size,
    }
    for name, items in groups.items():
        expected = expected_counts[name]
        if len(items) != expected:
            raise ValueError(
                f"{name} has {len(items)} examples, expected {expected}"
            )


def build_dose_splits(
    *,
    original_results_dir: Path,
    harmful_test_path: Path,
    harmless_test_path: Path,
    seed: int,
    selection_size: int = DOSE_SELECTION_SIZE,
    confirmation_size: int = DOSE_CONFIRMATION_SIZE,
) -> tuple[DoseSplits, dict[str, Any]]:
    used_ids, used_hashes = collect_original_usage(original_results_dir)
    required = selection_size + confirmation_size
    harmful_leftover = eligible_examples(
        load_local_prompt_corpus(harmful_test_path, "harmful"),
        used_ids=used_ids,
        used_hashes=used_hashes,
    )
    harmless_leftover = eligible_examples(
        load_local_prompt_corpus(harmless_test_path, "harmless"),
        used_ids=used_ids,
        used_hashes=used_hashes,
    )
    _require_leftovers(harmful_leftover, label="harmful", required=required)
    _require_leftovers(harmless_leftover, label="harmless", required=required)
    harmful_selection, harmful_confirmation = sample_dose_halves(
        harmful_leftover,
        seed=seed,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    harmless_selection, harmless_confirmation = sample_dose_halves(
        harmless_leftover,
        seed=seed + 1,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    splits = DoseSplits(
        harmful_dose_selection=tuple(harmful_selection),
        harmless_dose_selection=tuple(harmless_selection),
        harmful_dose_confirmation=tuple(harmful_confirmation),
        harmless_dose_confirmation=tuple(harmless_confirmation),
    )
    _assert_disjoint_splits(
        splits,
        used_ids=used_ids,
        used_hashes=used_hashes,
        selection_size=selection_size,
        confirmation_size=confirmation_size,
    )
    metadata = compact_split_metadata(
        harmful_selection=splits.harmful_dose_selection,
        harmless_selection=splits.harmless_dose_selection,
        harmful_confirmation=splits.harmful_dose_confirmation,
        harmless_confirmation=splits.harmless_dose_confirmation,
        seed=seed,
        source_revision=original_source_revision(original_results_dir),
    )
    return splits, metadata


def _smallest_qualifying_coefficient(
    curve: Sequence[Mapping[str, float]],
    *,
    predicate: Callable[[Mapping[str, float]], bool],
) -> float | None:
    qualifying = [
        float(point["coefficient"])
        for point in sorted(curve, key=lambda item: float(item["coefficient"]))
        if predicate(point)
    ]
    if not qualifying:
        return None
    return qualifying[0]


def select_dose_coefficients(
    *,
    ablation_curve: Sequence[Mapping[str, float]],
    addition_curve: Sequence[Mapping[str, float]],
    min_harmful_ablation_drop: float = DEFAULT_MIN_HARMFUL_ABLATION_DROP,
    max_harmless_ablation_kl: float = DEFAULT_MAX_HARMLESS_ABLATION_KL,
    min_harmless_addition_increase: float = DEFAULT_MIN_HARMLESS_ADDITION_INCREASE,
) -> dict[str, Any]:
    ablation_coefficient = _smallest_qualifying_coefficient(
        ablation_curve,
        predicate=lambda point: (
            float(point["harmful_refusal_drop"]) >= min_harmful_ablation_drop
            and float(point["harmless_kl"]) <= max_harmless_ablation_kl
        ),
    )
    addition_coefficient = _smallest_qualifying_coefficient(
        addition_curve,
        predicate=lambda point: (
            float(point["harmless_refusal_increase"])
            >= min_harmless_addition_increase
        ),
    )
    selected_pair = None
    if ablation_coefficient is not None and addition_coefficient is not None:
        selected_pair = {
            "ablation": ablation_coefficient,
            "addition": addition_coefficient,
        }
    return {
        "ablation_coefficient": ablation_coefficient,
        "addition_coefficient": addition_coefficient,
        "selected_pair": selected_pair,
        "ablation_curve": [dict(point) for point in ablation_curve],
        "addition_curve": [dict(point) for point in addition_curve],
    }
