"""Human review interface for sparse-refusal counterfactual pairs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


_SCRIPT = Path(__file__).resolve()
_PROJECT = _SCRIPT.parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from refusal_direction_replication.data import prompt_content_hash  # noqa: E402
from refusal_direction_replication.qwen_sparse_data import (  # noqa: E402
    ReviewedPair,
    validate_reviewed_pairs,
)


REVIEW_COMMANDS = frozenset({"a", "r", "e", "b", "q"})
EXPECTED_PAIR_COUNT = 256


def _required_text(
    record: Mapping[str, Any],
    field: str,
    identity: str,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{identity}: {field} must be nonempty")
    return value


def build_review_record(
    candidate: Mapping[str, Any],
    *,
    decision: str,
    defensive_prompt: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one auditable review record from explicitly allowed fields."""
    if decision not in {"approved", "rejected"}:
        raise ValueError("reviewer decision must be approved or rejected")
    pair_id = _required_text(candidate, "pair_id", "<candidate>")
    offensive = _required_text(candidate, "offensive_prompt", pair_id)
    original_defensive = _required_text(
        candidate, "defensive_prompt", pair_id
    )
    edited_defensive = (
        original_defensive
        if defensive_prompt is None
        else str(defensive_prompt)
    )
    if not edited_defensive.strip():
        raise ValueError(f"{pair_id}: defensive_prompt must be nonempty")

    offensive_hash = prompt_content_hash(offensive)
    original_defensive_hash = prompt_content_hash(original_defensive)
    declared_offensive_hash = str(candidate.get("offensive_sha256", ""))
    declared_defensive_hash = str(candidate.get("defensive_sha256", ""))
    if declared_offensive_hash != offensive_hash:
        raise ValueError(f"{pair_id}: original offensive hash mismatch")
    if declared_defensive_hash != original_defensive_hash:
        raise ValueError(f"{pair_id}: original defensive hash mismatch")
    edited_defensive_hash = prompt_content_hash(edited_defensive)
    if edited_defensive_hash == offensive_hash:
        raise ValueError(f"{pair_id}: same normalized text")

    return {
        "pair_id": pair_id,
        "category": _required_text(candidate, "category", pair_id),
        "source_identity": _required_text(
            candidate, "source_identity", pair_id
        ),
        "offensive_prompt": offensive,
        "defensive_prompt": edited_defensive,
        "intent_rationale": _required_text(
            candidate, "intent_rationale", pair_id
        ),
        "reviewer_decision": decision,
        "reviewed_at": timestamp
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_hashes": {
            "offensive_sha256": offensive_hash,
            "defensive_sha256": original_defensive_hash,
        },
        "edited_hashes": {
            "offensive_sha256": offensive_hash,
            "defensive_sha256": edited_defensive_hash,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def freeze_approved_reviews(
    review_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return frozen reviewed rows only when all 256 are approved."""
    approved = [
        record
        for record in review_records
        if record.get("reviewer_decision") == "approved"
    ]
    if len(approved) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            "refusing to freeze: expected exactly 256 approved records, "
            f"got {len(approved)}"
        )
    reviewed = [
        ReviewedPair(
            pair_id=_required_text(record, "pair_id", "<review>"),
            category=_required_text(
                record, "category", str(record.get("pair_id", "<review>"))
            ),
            source_identity=_required_text(
                record,
                "source_identity",
                str(record.get("pair_id", "<review>")),
            ),
            offensive_prompt=_required_text(
                record,
                "offensive_prompt",
                str(record.get("pair_id", "<review>")),
            ),
            defensive_prompt=_required_text(
                record,
                "defensive_prompt",
                str(record.get("pair_id", "<review>")),
            ),
            intent_rationale=_required_text(
                record,
                "intent_rationale",
                str(record.get("pair_id", "<review>")),
            ),
            approved=True,
        )
        for record in approved
    ]
    validate_reviewed_pairs(reviewed)
    return [
        {
            **record.compact(),
            "offensive_prompt": record.offensive_prompt,
            "defensive_prompt": record.defensive_prompt,
            "intent_rationale": record.intent_rationale,
            "approved": True,
            "reviewer_decision": "approved",
            "reviewed_at": source["reviewed_at"],
            "original_hashes": dict(source["original_hashes"]),
            "edited_hashes": dict(source["edited_hashes"]),
        }
        for record, source in zip(reviewed, approved, strict=True)
    ]


def _display_candidate(
    candidate: Mapping[str, Any],
    *,
    index: int,
    total: int,
    edited_defensive: str,
) -> None:
    print(f"\n[{index + 1}/{total}] {candidate['pair_id']}")
    print(f"Category: {candidate['category']}")
    print(f"Source: {candidate['source_identity']}")
    print("\nOFFENSIVE:")
    print(candidate["offensive_prompt"])
    print("\nDEFENSIVE:")
    print(edited_defensive)
    print("\nRATIONALE:")
    print(candidate["intent_rationale"])
    print("\na approve | r reject | e edit defensive prompt | b back | q save and quit")


def review_candidates(pair_dir: Path) -> bool:
    """Review candidates interactively; return whether a freeze was written."""
    candidates = _read_jsonl(pair_dir / "candidates.jsonl")
    if len(candidates) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"expected {EXPECTED_PAIR_COUNT} candidates, got {len(candidates)}"
        )
    candidate_ids = [str(record.get("pair_id", "")) for record in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate pair_id")

    review_path = pair_dir / "reviews.jsonl"
    saved_rows = _read_jsonl(review_path)
    reviews = {str(record["pair_id"]): record for record in saved_rows}
    if not set(reviews).issubset(set(candidate_ids)):
        raise ValueError("saved review references an unknown candidate")
    edits = {
        pair_id: str(record["defensive_prompt"])
        for pair_id, record in reviews.items()
    }
    index = next(
        (
            offset
            for offset, candidate in enumerate(candidates)
            if str(candidate["pair_id"]) not in reviews
        ),
        0,
    )

    while 0 <= index < len(candidates):
        candidate = candidates[index]
        pair_id = str(candidate["pair_id"])
        defensive = edits.get(pair_id, str(candidate["defensive_prompt"]))
        _display_candidate(
            candidate,
            index=index,
            total=len(candidates),
            edited_defensive=defensive,
        )
        command = input("> ").strip().casefold()
        if command not in REVIEW_COMMANDS:
            print("Unknown command. Use a, r, e, b, or q.")
            continue
        if command == "q":
            break
        if command == "b":
            index = max(0, index - 1)
            continue
        if command == "e":
            edited = input("New defensive prompt: ").strip()
            if not edited:
                print("Defensive prompt cannot be empty.")
                continue
            edits[pair_id] = edited
            continue

        decision = "approved" if command == "a" else "rejected"
        reviews[pair_id] = build_review_record(
            candidate,
            decision=decision,
            defensive_prompt=defensive,
        )
        ordered_reviews = [
            reviews[item_id] for item_id in candidate_ids if item_id in reviews
        ]
        _write_jsonl(review_path, ordered_reviews)
        index += 1

    ordered_reviews = [
        reviews[item_id] for item_id in candidate_ids if item_id in reviews
    ]
    _write_jsonl(review_path, ordered_reviews)
    approved_count = sum(
        row["reviewer_decision"] == "approved" for row in ordered_reviews
    )
    if approved_count != EXPECTED_PAIR_COUNT:
        print(
            "Not frozen: "
            f"{approved_count}/{EXPECTED_PAIR_COUNT} records are approved."
        )
        return False

    frozen = freeze_approved_reviews(ordered_reviews)
    _write_jsonl(pair_dir / "reviewed.jsonl", frozen)
    print(f"Frozen {len(frozen)} approved pair identities.")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human review for sparse-refusal pairs."
    )
    parser.add_argument("--root", type=Path, default=_PROJECT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pair_dir = (
        Path(args.root).resolve()
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "pairs"
    )
    review_candidates(pair_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
