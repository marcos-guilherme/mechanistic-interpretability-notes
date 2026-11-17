from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from refusal_direction_replication.data import prompt_content_hash
from refusal_direction_replication.qwen_sparse_data import (
    CorpusRecord,
    PairRecord,
    PairSplit,
    ReviewedPair,
    build_corpus_manifest,
    split_reviewed_pairs,
    validate_reviewed_pairs,
)


def reviewed_pair(
    *,
    pair_id: str,
    category: str = "category-0",
    offensive_prompt: str | None = None,
    defensive_prompt: str | None = None,
    approved: bool = True,
) -> ReviewedPair:
    return ReviewedPair(
        pair_id=pair_id,
        category=category,
        source_identity=f"purplellama:{pair_id}",
        offensive_prompt=offensive_prompt or f"Offensive request {pair_id}.",
        defensive_prompt=defensive_prompt or f"Defensive request {pair_id}.",
        intent_rationale=f"Intent differs for {pair_id}.",
        approved=approved,
    )


def synthetic_reviewed_pairs(count: int, *, categories: int) -> list[ReviewedPair]:
    return [
        reviewed_pair(
            pair_id=f"pair-{index:04d}",
            category=f"category-{index % categories}",
        )
        for index in range(count)
    ]


def test_pair_contracts_are_immutable_and_compact() -> None:
    seed = PairRecord(
        pair_id="pair-0001",
        category="credential-access",
        source_identity="purplellama:mitre:17",
        offensive_prompt="Perform the offensive action.",
    )
    split = PairSplit(name="discovery", pair_ids=("pair-0001",))
    assert seed.pair_id == "pair-0001"
    assert split.pair_ids == ("pair-0001",)
    with pytest.raises(AttributeError):
        seed.pair_id = "changed"  # type: ignore[misc]

    pair = reviewed_pair(pair_id="pair-0001")
    assert pair.compact() == {
        "pair_id": "pair-0001",
        "category": "category-0",
        "source_identity": "purplellama:pair-0001",
        "offensive_sha256": prompt_content_hash(pair.offensive_prompt),
        "defensive_sha256": prompt_content_hash(pair.defensive_prompt),
    }
    assert "prompt" not in " ".join(pair.compact())


def test_reviewed_pairs_are_disjoint_deterministic_and_stratified() -> None:
    records = synthetic_reviewed_pairs(256, categories=8)
    first = split_reviewed_pairs(records)
    second = split_reviewed_pairs(list(reversed(records)))
    assert {name: len(rows) for name, rows in first.items()} == {
        "discovery": 128,
        "selection": 64,
        "test": 64,
    }
    assert {
        name: tuple(row.pair_id for row in rows)
        for name, rows in first.items()
    } == {
        name: tuple(row.pair_id for row in rows)
        for name, rows in second.items()
    }
    assert {
        name: hashlib.sha256(
            "\n".join(row.pair_id for row in rows).encode()
        ).hexdigest()
        for name, rows in first.items()
    } == {
        "discovery": (
            "1412fc7f40f4ed5c81270fceb1f456190a4207f3ee799632f909869c0312feac"
        ),
        "selection": (
            "f86a14f49b89defecdd8c93e56007438188378b3ce4be7cdf495b5d3e7e9b878"
        ),
        "test": (
            "8c11a4a24b2d53614e1c90af90a8b6f55a42eb3a4462aaa2c2f54da46253952a"
        ),
    }
    ids = [row.pair_id for rows in first.values() for row in rows]
    assert len(ids) == len(set(ids))
    for category in {row.category for row in records}:
        counts = Counter(
            name
            for name, rows in first.items()
            for row in rows
            if row.category == category
        )
        assert counts == {"discovery": 16, "selection": 8, "test": 8}
    validate_reviewed_pairs(records)


def test_reviewed_pairs_reject_same_text_on_both_sides() -> None:
    records = synthetic_reviewed_pairs(256, categories=8)
    records[0] = reviewed_pair(
        pair_id="pair-0000",
        offensive_prompt="Inspect this service.",
        defensive_prompt="  inspect THIS\nservice. ",
    )
    with pytest.raises(ValueError, match="same normalized text"):
        validate_reviewed_pairs(records)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"approved": False}, "all frozen pairs must be approved"),
        ({"pair_id": "pair-0001"}, "duplicate pair_id"),
        (
            {"offensive_prompt": "Offensive request pair-0001."},
            "prompt hash reused across pairs",
        ),
        ({"source_identity": "purplellama:pair-0001"}, "source identity reused"),
    ],
)
def test_reviewed_pairs_reject_non_disjoint_records(
    replacement: dict[str, object],
    message: str,
) -> None:
    records = synthetic_reviewed_pairs(256, categories=8)
    records[0] = replace(records[0], **replacement)
    with pytest.raises(ValueError, match=message):
        validate_reviewed_pairs(records)


def test_reviewed_pairs_require_exactly_256_records() -> None:
    with pytest.raises(ValueError, match="expected 256 approved pairs, got 1"):
        validate_reviewed_pairs([reviewed_pair(pair_id="pair-0001")])


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return list(range(len(text.split())))


def test_corpus_manifest_excludes_deduplicates_and_omits_raw_text() -> None:
    excluded_text = "Frozen pair text"
    records = [
        CorpusRecord(
            source="code-search-net/code_search_net",
            revision="bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
            source_identity="code:1",
            text="def alpha(): pass",
        ),
        CorpusRecord(
            source="HuggingFaceH4/ultrachat_200k",
            revision="8049631c405ae6576f93f445c6b8166f76f5505a",
            source_identity="chat:1",
            text=excluded_text,
        ),
        CorpusRecord(
            source="rufimelo/DeltaSecommits",
            revision="40de1664860b8e37fb2322ab2b2fbd0b248f3598",
            source_identity="commit:1",
            text="Fix the parser safely",
        ),
        CorpusRecord(
            source="rufimelo/DeltaSecommits",
            revision="40de1664860b8e37fb2322ab2b2fbd0b248f3598",
            source_identity="commit:2",
            text="  fix THE parser\nsafely ",
        ),
    ]
    tokenizer = RecordingTokenizer()
    manifest = build_corpus_manifest(
        records,
        {prompt_content_hash(excluded_text)},
        tokenizer,
    )

    assert manifest["schema_version"] == 1
    assert manifest["accepted_records"] == 2
    assert manifest["excluded_records"] == 1
    assert manifest["duplicate_records"] == 1
    assert manifest["accepted_tokens"] == 7
    assert [row["source_identity"] for row in manifest["records"]] == [
        "code:1",
        "commit:1",
    ]
    assert all("text" not in row for row in manifest["records"])
    assert excluded_text not in repr(manifest)
    assert tokenizer.calls == [
        ("def alpha(): pass", False),
        ("Fix the parser safely", False),
    ]


PROJECT_ROOT = Path(__file__).parents[1]


def load_script(name: str) -> Any:
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offensive_seed_selection_is_round_robin_and_excludes_frozen_items() -> None:
    prepare = load_script("prepare_qwen_sparse_pairs")
    rows = [
        {
            "mutated_prompt": f"Offensive source text {index}",
            "mitre_category": f"category-{index % 8}",
        }
        for index in range(320)
    ]
    forbidden_id = prepare.mitre_example_id(0, rows[0]["mutated_prompt"])
    forbidden_hash = prompt_content_hash(rows[1]["mutated_prompt"])
    seeds = prepare.select_offensive_seeds(
        rows,
        forbidden_ids={forbidden_id},
        forbidden_hashes={forbidden_hash},
    )

    assert len(seeds) == 256
    assert forbidden_id not in {seed.source_identity for seed in seeds}
    assert forbidden_hash not in {
        prompt_content_hash(seed.offensive_prompt) for seed in seeds
    }
    counts = Counter(seed.category for seed in seeds)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert seeds == prepare.select_offensive_seeds(
        rows,
        forbidden_ids={forbidden_id},
        forbidden_hashes={forbidden_hash},
    )


def test_draft_validation_requires_complete_matching_counterparts() -> None:
    prepare = load_script("prepare_qwen_sparse_pairs")
    seeds = [
        PairRecord(
            pair_id=f"pair-{index:04d}",
            category=f"category-{index % 8}",
            source_identity=f"mitre:{index}:source",
            offensive_prompt=f"Offensive request {index}",
        )
        for index in range(256)
    ]
    drafts = [
        {
            "pair_id": seed.pair_id,
            "category": seed.category,
            "source_identity": seed.source_identity,
            "offensive_prompt": seed.offensive_prompt,
            "defensive_prompt": f"Detect and mitigate mechanism {index}",
            "intent_rationale": "Changes action from attack to mitigation.",
        }
        for index, seed in enumerate(seeds)
    ]

    candidates = prepare.validate_drafts(
        seeds,
        drafts,
        forbidden_hashes=set(),
    )
    assert len(candidates) == 256
    assert candidates[0]["offensive_sha256"] == prompt_content_hash(
        seeds[0].offensive_prompt
    )
    assert candidates[0]["defensive_sha256"] == prompt_content_hash(
        drafts[0]["defensive_prompt"]
    )

    with pytest.raises(ValueError, match="exactly once"):
        prepare.validate_drafts(
            seeds,
            [*drafts[:-1], drafts[0]],
            forbidden_hashes=set(),
        )
    with pytest.raises(ValueError, match="forbidden frozen hash"):
        prepare.validate_drafts(
            seeds,
            drafts,
            forbidden_hashes={prompt_content_hash(drafts[0]["defensive_prompt"])},
        )


def test_review_records_preserve_original_and_edited_hashes_without_outputs() -> None:
    review = load_script("review_qwen_sparse_pairs")
    candidate = {
        "pair_id": "pair-0001",
        "category": "credential-access",
        "source_identity": "mitre:1:source",
        "offensive_prompt": "Perform an offensive action.",
        "defensive_prompt": "Detect that offensive action.",
        "intent_rationale": "The counterpart changes the requested action.",
        "offensive_sha256": prompt_content_hash("Perform an offensive action."),
        "defensive_sha256": prompt_content_hash("Detect that offensive action."),
    }
    saved = review.build_review_record(
        candidate,
        decision="approved",
        defensive_prompt="Audit and detect that offensive action.",
        timestamp="2026-11-17T22:00:00+00:00",
    )

    assert review.REVIEW_COMMANDS == frozenset({"a", "r", "e", "b", "q"})
    assert saved["reviewer_decision"] == "approved"
    assert saved["reviewed_at"] == "2026-11-17T22:00:00+00:00"
    assert saved["original_hashes"] == {
        "offensive_sha256": candidate["offensive_sha256"],
        "defensive_sha256": candidate["defensive_sha256"],
    }
    assert saved["edited_hashes"]["defensive_sha256"] == prompt_content_hash(
        saved["defensive_prompt"]
    )
    assert not any("output" in key.casefold() for key in saved)


def test_review_freeze_keeps_approved_text_under_artifacts() -> None:
    review = load_script("review_qwen_sparse_pairs")
    saved = []
    for index in range(256):
        offensive = f"Offensive action {index}."
        defensive = f"Detect and mitigate action {index}."
        candidate = {
            "pair_id": f"pair-{index:04d}",
            "category": f"category-{index % 8}",
            "source_identity": f"mitre:{index}:source",
            "offensive_prompt": offensive,
            "defensive_prompt": defensive,
            "intent_rationale": "Changes the requested action to mitigation.",
            "offensive_sha256": prompt_content_hash(offensive),
            "defensive_sha256": prompt_content_hash(defensive),
        }
        saved.append(
            review.build_review_record(
                candidate,
                decision="approved",
                timestamp="2026-11-17T22:00:00+00:00",
            )
        )

    frozen = review.freeze_approved_reviews(saved)
    assert frozen[0]["offensive_prompt"] == "Offensive action 0."
    assert frozen[0]["defensive_prompt"] == "Detect and mitigate action 0."
    assert frozen[0]["intent_rationale"]
    assert frozen[0]["approved"] is True
    with pytest.raises(ValueError, match="expected exactly 256 approved"):
        review.freeze_approved_reviews(saved[:-1])
