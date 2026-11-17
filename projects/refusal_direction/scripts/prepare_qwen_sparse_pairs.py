"""Prepare text-only offensive seeds and validate drafted defensive pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any
from urllib.request import urlopen


_SCRIPT = Path(__file__).resolve()
_PROJECT = _SCRIPT.parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from refusal_direction_replication.data import prompt_content_hash  # noqa: E402
from refusal_direction_replication.qwen_sparse_data import PairRecord  # noqa: E402
from refusal_direction_replication.qwen_sparse_protocol import (  # noqa: E402
    PAIR_SPLIT_SEED,
    PURPLELLAMA_REVISION,
)


PAIR_COUNT = 256
PURPLELLAMA_REPOSITORY = "meta-llama/PurpleLlama"
MITRE_RELATIVE_PATH = (
    "CybersecurityBenchmarks/datasets/mitre/"
    "mitre_benchmark_100_per_category_with_augmentation.json"
)


def _walk_named_lists(value: Any) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"example_ids", "content_hashes"} and isinstance(child, list):
                yield str(key), child
            yield from _walk_named_lists(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_named_lists(child)


def collect_frozen_usage(root: Path) -> tuple[set[str], set[str]]:
    """Collect every selected ID and hash from existing prepare artifacts."""
    ids: set[str] = set()
    hashes: set[str] = set()
    for path in sorted((Path(root) / "results").glob("**/00_prepare.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, values in _walk_named_lists(payload):
            destination = ids if key == "example_ids" else hashes
            destination.update(str(value) for value in values)
    return ids, hashes


def mitre_example_id(source_index: int, text: str) -> str:
    digest = prompt_content_hash(text)
    return f"mitre:{int(source_index)}:{digest[:16]}"


def _round_robin_indices(
    groups: Mapping[str, Sequence[int]],
    *,
    size: int,
    seed: int,
) -> list[int]:
    shuffled: dict[str, list[int]] = {}
    for category in sorted(groups):
        values = list(groups[category])
        random.Random(f"{seed}:{category}").shuffle(values)
        shuffled[category] = values
    selected: list[int] = []
    offset = 0
    while len(selected) < size:
        made_progress = False
        for category in sorted(shuffled):
            values = shuffled[category]
            if offset < len(values):
                selected.append(values[offset])
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            break
        offset += 1
    return selected


def select_offensive_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    forbidden_ids: set[str],
    forbidden_hashes: set[str],
    size: int = PAIR_COUNT,
    seed: int = PAIR_SPLIT_SEED,
) -> list[PairRecord]:
    """Select unused MITRE prompts deterministically and by category."""
    candidates: list[tuple[str, str, str]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    seen_hashes: set[str] = set()
    for source_index, row in enumerate(rows):
        text = row.get("mutated_prompt")
        category_value = row.get("mitre_category")
        if not isinstance(text, str) or not text.strip():
            continue
        if category_value is None or not str(category_value).strip():
            continue
        category = str(category_value)
        digest = prompt_content_hash(text)
        identity = mitre_example_id(source_index, text)
        if identity in forbidden_ids or digest in forbidden_hashes:
            continue
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        candidate_index = len(candidates)
        candidates.append((identity, category, text))
        groups[category].append(candidate_index)

    selected_indices = _round_robin_indices(groups, size=size, seed=seed)
    if len(selected_indices) != size:
        raise ValueError(
            f"need {size} unused unique MITRE items, received "
            f"{len(selected_indices)}"
        )
    return [
        PairRecord(
            pair_id=f"pair-{pair_index:04d}",
            category=candidates[candidate_index][1],
            source_identity=candidates[candidate_index][0],
            offensive_prompt=candidates[candidate_index][2],
        )
        for pair_index, candidate_index in enumerate(selected_indices)
    ]


def _require_draft_text(
    draft: Mapping[str, Any],
    field: str,
    pair_id: str,
) -> str:
    value = draft.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{pair_id}: {field} must be nonempty")
    return value


def validate_drafts(
    seeds: Sequence[PairRecord],
    drafts: Sequence[Mapping[str, Any]],
    *,
    forbidden_hashes: set[str],
) -> list[dict[str, Any]]:
    """Validate complete defensive drafts against their frozen seeds."""
    if len(seeds) != PAIR_COUNT:
        raise ValueError(f"expected {PAIR_COUNT} seeds, got {len(seeds)}")
    source_identities = [str(draft.get("source_identity", "")) for draft in drafts]
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("each source identity must appear exactly once")
    seed_by_identity = {seed.source_identity: seed for seed in seeds}
    if len(drafts) != len(seeds) or set(source_identities) != set(seed_by_identity):
        raise ValueError("each source identity must appear exactly once")

    draft_by_identity = {
        str(draft["source_identity"]): draft for draft in drafts
    }
    candidates: list[dict[str, Any]] = []
    all_hashes: list[str] = []
    for seed in seeds:
        draft = draft_by_identity[seed.source_identity]
        pair_id = _require_draft_text(draft, "pair_id", seed.pair_id)
        category = _require_draft_text(draft, "category", seed.pair_id)
        offensive = _require_draft_text(
            draft, "offensive_prompt", seed.pair_id
        )
        defensive = _require_draft_text(
            draft, "defensive_prompt", seed.pair_id
        )
        rationale = _require_draft_text(
            draft, "intent_rationale", seed.pair_id
        )
        if pair_id != seed.pair_id:
            raise ValueError(f"{seed.pair_id}: pair_id changed in draft")
        if category != seed.category:
            raise ValueError(f"{seed.pair_id}: category differs from seed manifest")
        if offensive != seed.offensive_prompt:
            raise ValueError(f"{seed.pair_id}: offensive prompt differs from seed")
        offensive_hash = prompt_content_hash(offensive)
        defensive_hash = prompt_content_hash(defensive)
        if offensive_hash == defensive_hash:
            raise ValueError(f"{seed.pair_id}: same normalized text")
        if offensive_hash in forbidden_hashes or defensive_hash in forbidden_hashes:
            raise ValueError(f"{seed.pair_id}: forbidden frozen hash")
        all_hashes.extend((offensive_hash, defensive_hash))
        candidates.append(
            {
                "pair_id": seed.pair_id,
                "category": seed.category,
                "source_identity": seed.source_identity,
                "offensive_prompt": offensive,
                "defensive_prompt": defensive,
                "intent_rationale": rationale,
                "offensive_sha256": offensive_hash,
                "defensive_sha256": defensive_hash,
            }
        )
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("prompt hash reused across candidate pairs")
    return candidates


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_purplellama_mitre() -> list[dict[str, Any]]:
    url = (
        "https://raw.githubusercontent.com/"
        f"{PURPLELLAMA_REPOSITORY}/{PURPLELLAMA_REVISION}/{MITRE_RELATIVE_PATH}"
    )
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("PurpleLlama MITRE source must be a JSON list")
    return [dict(row) for row in payload]


def _pair_dir(root: Path) -> Path:
    return (
        Path(root)
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "pairs"
    )


def _seed_offensive(root: Path) -> Path:
    forbidden_ids, forbidden_hashes = collect_frozen_usage(root)
    records = select_offensive_seeds(
        _load_purplellama_mitre(),
        forbidden_ids=forbidden_ids,
        forbidden_hashes=forbidden_hashes,
    )
    destination = _pair_dir(root) / "seeds.jsonl"
    _write_jsonl(
        destination,
        [
            {
                "pair_id": record.pair_id,
                "category": record.category,
                "source_identity": record.source_identity,
                "offensive_prompt": record.offensive_prompt,
                "offensive_sha256": prompt_content_hash(
                    record.offensive_prompt
                ),
                "source_repository": PURPLELLAMA_REPOSITORY,
                "source_revision": PURPLELLAMA_REVISION,
                "source_relative_path": MITRE_RELATIVE_PATH,
            }
            for record in records
        ],
    )
    return destination


def _validate_draft_file(root: Path) -> Path:
    pair_dir = _pair_dir(root)
    seed_rows = _read_jsonl(pair_dir / "seeds.jsonl")
    seeds = [
        PairRecord(
            pair_id=_require_draft_text(row, "pair_id", "<seed>"),
            category=_require_draft_text(row, "category", "<seed>"),
            source_identity=_require_draft_text(
                row, "source_identity", "<seed>"
            ),
            offensive_prompt=_require_draft_text(
                row, "offensive_prompt", "<seed>"
            ),
        )
        for row in seed_rows
    ]
    _forbidden_ids, forbidden_hashes = collect_frozen_usage(root)
    candidates = validate_drafts(
        seeds,
        _read_jsonl(pair_dir / "drafts.jsonl"),
        forbidden_hashes=forbidden_hashes,
    )
    destination = pair_dir / "candidates.jsonl"
    _write_jsonl(destination, candidates)
    return destination


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sparse-refusal counterfactual pairs."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seed-offensive", action="store_true")
    action.add_argument("--validate-drafts", action="store_true")
    parser.add_argument("--root", type=Path, default=_PROJECT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.root).resolve()
    destination = (
        _seed_offensive(root)
        if args.seed_offensive
        else _validate_draft_file(root)
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
