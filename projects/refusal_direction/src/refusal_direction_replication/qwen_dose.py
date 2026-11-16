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
ORIGINAL_SPLIT_KEYS = (
    "harmful_train",
    "harmless_train",
    "harmful_val",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)
ABLATION_METRIC_FIELDS = (
    "coefficient",
    "harmful_refusal_drop",
    "harmless_kl",
)
ADDITION_METRIC_FIELDS = (
    "coefficient",
    "harmless_refusal_increase",
)
METADATA_LEAK_KEYS = frozenset(
    {
        "text",
        "instruction",
        "prompt",
        "prompts",
        "generation",
        "generations",
        "outputs",
        "responses",
        "raw",
        "generated",
        "completion",
        "completions",
    }
)


@dataclass(frozen=True)
class DoseSplits:
    harmful_dose_selection: tuple[PromptExample, ...]
    harmless_dose_selection: tuple[PromptExample, ...]
    harmful_dose_confirmation: tuple[PromptExample, ...]
    harmless_dose_confirmation: tuple[PromptExample, ...]


def _require_usage_list(value: object, *, field: str, key: str) -> list:
    if not isinstance(value, list):
        raise ValueError(
            f"{field}[{key!r}] must be a list, got {type(value).__name__}"
        )
    return value


def _extend_usage_groups(
    groups: dict[str, list[str]],
    mapping: object,
    *,
    field: str,
) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{field} must be an object mapping split names to lists")
    for key, value in mapping.items():
        items = _require_usage_list(value, field=field, key=str(key))
        groups.setdefault(str(key), []).extend(str(item) for item in items)


def collect_original_usage(
    results_dir: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    id_groups: dict[str, list[str]] = {}
    hash_groups: dict[str, list[str]] = {}
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        results = payload.get("results") or {}
        if not isinstance(results, dict):
            continue
        example_ids = results.get("example_ids")
        content_hashes = results.get("content_hashes")
        if example_ids is None and content_hashes is None:
            continue
        if example_ids is None or content_hashes is None:
            raise ValueError(
                f"{path.name} must record both example_ids and content_hashes"
            )
        _extend_usage_groups(id_groups, example_ids, field="example_ids")
        _extend_usage_groups(hash_groups, content_hashes, field="content_hashes")
    missing = [
        key
        for key in ORIGINAL_SPLIT_KEYS
        if not id_groups.get(key) or not hash_groups.get(key)
    ]
    if missing:
        raise ValueError(
            "original usage missing required split keys: " + ", ".join(missing)
        )
    ids = {item for group in id_groups.values() for item in group}
    hashes = {item for group in hash_groups.values() for item in group}
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


def metadata_contains_leak(obj: Any, *, key: str | None = None) -> bool:
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            lowered = str(child_key).casefold()
            if lowered in METADATA_LEAK_KEYS:
                return True
            if metadata_contains_leak(value, key=str(child_key)):
                return True
        return False
    if isinstance(obj, list):
        return any(metadata_contains_leak(item, key=key) for item in obj)
    if isinstance(obj, str):
        stripped = obj.strip()
        if len(stripped.split()) >= 8:
            return True
        return False
    return False


def compact_split_metadata(
    *,
    harmful_selection: Sequence[PromptExample],
    harmless_selection: Sequence[PromptExample],
    harmful_confirmation: Sequence[PromptExample],
    harmless_confirmation: Sequence[PromptExample],
    seed: int,
    source_revision: str,
    harmless_seed: int | None = None,
) -> dict[str, Any]:
    groups = {
        "harmful_dose_selection": harmful_selection,
        "harmless_dose_selection": harmless_selection,
        "harmful_dose_confirmation": harmful_confirmation,
        "harmless_dose_confirmation": harmless_confirmation,
    }
    payload = {
        "example_ids": {
            name: [example.id for example in items]
            for name, items in groups.items()
        },
        "content_hashes": {
            name: content_hashes_for_examples(items)
            for name, items in groups.items()
        },
        "counts": {name: len(items) for name, items in groups.items()},
        "seed": {
            "harmful": seed,
            "harmless": seed + 1 if harmless_seed is None else harmless_seed,
        },
        "source_revision": source_revision,
    }
    if metadata_contains_leak(payload):
        raise ValueError("compact split metadata must not contain prompt text")
    return payload


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


def _half_ids(examples: Sequence[PromptExample]) -> list[str]:
    return [example.id for example in examples]


def _half_hashes(examples: Sequence[PromptExample]) -> list[str]:
    return [prompt_content_hash(example.text) for example in examples]


def _assert_disjoint_splits(
    splits: DoseSplits,
    *,
    used_ids: Set[str],
    used_hashes: Set[str],
    selection_size: int,
    confirmation_size: int,
) -> None:
    groups = _split_groups(splits)
    expected_counts = {
        "harmful_dose_selection": selection_size,
        "harmless_dose_selection": selection_size,
        "harmful_dose_confirmation": confirmation_size,
        "harmless_dose_confirmation": confirmation_size,
    }
    identities: dict[str, tuple[list[str], list[str]]] = {}
    for name, items in groups.items():
        expected = expected_counts[name]
        if len(items) != expected:
            raise ValueError(
                f"{name} has {len(items)} examples, expected {expected}"
            )
        ids = _half_ids(items)
        hashes = _half_hashes(items)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} has duplicate example ids")
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{name} has duplicate content hashes")
        identities[name] = (ids, hashes)
    new_ids = {item for ids, _hashes in identities.values() for item in ids}
    new_hashes = {
        item for _ids, hashes in identities.values() for item in hashes
    }
    if not used_ids.isdisjoint(new_ids):
        raise ValueError("dose splits overlap original example ids")
    if not used_hashes.isdisjoint(new_hashes):
        raise ValueError("dose splits overlap original content hashes")
    names = list(identities)
    for index, left_name in enumerate(names):
        left_ids, left_hashes = identities[left_name]
        for right_name in names[index + 1 :]:
            right_ids, right_hashes = identities[right_name]
            if not set(left_ids).isdisjoint(right_ids):
                raise ValueError(
                    f"{left_name} and {right_name} overlap example ids"
                )
            if not set(left_hashes).isdisjoint(right_hashes):
                raise ValueError(
                    f"{left_name} and {right_name} overlap content hashes"
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
        harmless_seed=seed + 1,
        source_revision=original_source_revision(original_results_dir),
    )
    return splits, metadata


def _metric_point(
    point: Mapping[str, float],
    fields: Sequence[str],
) -> dict[str, float]:
    return {field: float(point[field]) for field in fields if field in point}


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
        "ablation_curve": [
            _metric_point(point, ABLATION_METRIC_FIELDS)
            for point in ablation_curve
        ],
        "addition_curve": [
            _metric_point(point, ADDITION_METRIC_FIELDS)
            for point in addition_curve
        ],
    }
