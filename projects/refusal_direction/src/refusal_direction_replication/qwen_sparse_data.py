"""Text-only data contracts for the sparse-refusal steering study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import AbstractSet, Any

from .data import prompt_content_hash
from .qwen_sparse_protocol import (
    CORPUS_SOURCES,
    PAIR_SPLIT_SEED,
    PAIR_SPLIT_SIZES,
    SPARSE_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class PairRecord:
    """An offensive seed selected for defensive-counterpart drafting."""

    pair_id: str
    category: str
    source_identity: str
    offensive_prompt: str

    def compact(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "category": self.category,
            "source_identity": self.source_identity,
            "offensive_sha256": prompt_content_hash(self.offensive_prompt),
        }


@dataclass(frozen=True)
class ReviewedPair:
    """A human-reviewed counterfactual pair eligible for freezing."""

    pair_id: str
    category: str
    source_identity: str
    offensive_prompt: str
    defensive_prompt: str
    intent_rationale: str
    approved: bool

    def compact(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "category": self.category,
            "source_identity": self.source_identity,
            "offensive_sha256": prompt_content_hash(self.offensive_prompt),
            "defensive_sha256": prompt_content_hash(self.defensive_prompt),
        }


@dataclass(frozen=True)
class PairSplit:
    """Compact identity-only representation of one frozen pair split."""

    name: str
    pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusRecord:
    """A text record with a fully resolved dataset identity."""

    source: str
    revision: str
    source_identity: str
    text: str


def _require_text(value: str, field: str, identity: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{identity}: {field} must be nonempty")


def validate_reviewed_pairs(records: Sequence[ReviewedPair]) -> None:
    """Enforce the complete, approved, disjoint 256-pair contract."""
    if len(records) != 256:
        raise ValueError(f"expected 256 approved pairs, got {len(records)}")
    if not all(record.approved for record in records):
        raise ValueError("all frozen pairs must be approved")

    pair_ids = [record.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate pair_id")
    source_identities = [record.source_identity for record in records]
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("source identity reused")

    all_hashes: list[str] = []
    for record in records:
        _require_text(record.pair_id, "pair_id", "<pair>")
        _require_text(record.category, "category", record.pair_id)
        _require_text(record.source_identity, "source_identity", record.pair_id)
        _require_text(record.offensive_prompt, "offensive_prompt", record.pair_id)
        _require_text(record.defensive_prompt, "defensive_prompt", record.pair_id)
        _require_text(record.intent_rationale, "intent_rationale", record.pair_id)
        offensive = prompt_content_hash(record.offensive_prompt)
        defensive = prompt_content_hash(record.defensive_prompt)
        if offensive == defensive:
            raise ValueError(f"{record.pair_id}: same normalized text")
        all_hashes.extend((offensive, defensive))
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("prompt hash reused across pairs")


def split_reviewed_pairs(
    records: Sequence[ReviewedPair],
) -> dict[str, tuple[ReviewedPair, ...]]:
    """Create deterministic 128/64/64 splits stratified by category."""
    validate_reviewed_pairs(records)
    by_category: dict[str, list[ReviewedPair]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    splits: dict[str, list[ReviewedPair]] = {
        name: [] for name in PAIR_SPLIT_SIZES
    }
    remaining = dict(PAIR_SPLIT_SIZES)
    for category in sorted(by_category):
        rows = sorted(
            by_category[category],
            key=lambda row: hashlib.sha256(
                f"{PAIR_SPLIT_SEED}:{row.pair_id}".encode()
            ).hexdigest(),
        )
        denominator = sum(remaining.values())
        exact = {
            name: len(rows) * capacity / denominator
            for name, capacity in remaining.items()
        }
        allocation = {
            name: min(remaining[name], math.floor(exact[name]))
            for name in remaining
        }
        unassigned = len(rows) - sum(allocation.values())
        order = sorted(
            remaining,
            key=lambda name: (
                -(exact[name] - math.floor(exact[name])),
                name,
            ),
        )
        while unassigned:
            assigned_this_pass = False
            for name in order:
                if allocation[name] < remaining[name]:
                    allocation[name] += 1
                    unassigned -= 1
                    assigned_this_pass = True
                    if not unassigned:
                        break
            if not assigned_this_pass:
                raise AssertionError("split allocation exceeded remaining capacity")

        cursor = 0
        for name in ("discovery", "selection", "test"):
            count = allocation[name]
            splits[name].extend(rows[cursor : cursor + count])
            cursor += count
            remaining[name] -= count

    if any(remaining.values()):
        raise AssertionError(f"split allocation incomplete: {remaining}")
    return {name: tuple(rows) for name, rows in splits.items()}


def _known_source_revisions() -> dict[str, str]:
    return {
        str(source["repo_id"]): str(source["revision"])
        for source in CORPUS_SOURCES
    }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError("tokenizer.encode must return a token sequence")
    return [int(token) for token in encoded]


def _manifest_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_corpus_manifest(
    records: Sequence[CorpusRecord],
    exclusion_hashes: AbstractSet[str],
    tokenizer: Any,
) -> dict[str, Any]:
    """Deduplicate and tokenize corpus records without retaining raw text."""
    known_revisions = _known_source_revisions()
    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_identities: set[str] = set()
    excluded_records = 0
    duplicate_records = 0

    for record in records:
        _require_text(record.source, "source", record.source_identity)
        _require_text(record.revision, "revision", record.source_identity)
        _require_text(record.source_identity, "source_identity", "<corpus>")
        _require_text(record.text, "text", record.source_identity)
        expected_revision = known_revisions.get(record.source)
        if expected_revision is None:
            raise ValueError(f"unknown corpus source: {record.source}")
        if record.revision != expected_revision:
            raise ValueError(
                f"{record.source_identity}: unresolved corpus revision "
                f"{record.revision}"
            )
        if record.source_identity in seen_identities:
            raise ValueError(
                f"duplicate corpus source identity: {record.source_identity}"
            )
        seen_identities.add(record.source_identity)

        content_hash = prompt_content_hash(record.text)
        if content_hash in exclusion_hashes:
            excluded_records += 1
            continue
        if content_hash in seen_hashes:
            duplicate_records += 1
            continue
        seen_hashes.add(content_hash)
        token_count = len(_token_ids(tokenizer, record.text))
        if token_count == 0:
            raise ValueError(f"{record.source_identity}: tokenized to zero tokens")
        accepted.append(
            {
                "source": record.source,
                "revision": record.revision,
                "source_identity": record.source_identity,
                "content_sha256": content_hash,
                "token_count": token_count,
            }
        )

    source_stats: dict[str, dict[str, Any]] = {}
    for source in CORPUS_SOURCES:
        repo_id = str(source["repo_id"])
        source_rows = [row for row in accepted if row["source"] == repo_id]
        source_stats[repo_id] = {
            "config": source["config"],
            "split": source["split"],
            "revision": source["revision"],
            "target_tokens": source["target_tokens"],
            "accepted_records": len(source_rows),
            "accepted_tokens": sum(row["token_count"] for row in source_rows),
        }

    return {
        "schema_version": SPARSE_SCHEMA_VERSION,
        "records": accepted,
        "records_sha256": _manifest_digest(accepted),
        "accepted_records": len(accepted),
        "accepted_tokens": sum(row["token_count"] for row in accepted),
        "excluded_records": excluded_records,
        "duplicate_records": duplicate_records,
        "sources": source_stats,
    }
