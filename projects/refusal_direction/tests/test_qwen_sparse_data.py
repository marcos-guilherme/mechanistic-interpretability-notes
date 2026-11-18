from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
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


def test_defensive_template_drafts_are_fixed_deterministic_counterparts() -> None:
    prepare = load_script("prepare_qwen_sparse_pairs")
    seeds = [
        PairRecord(
            pair_id=f"pair-{index:04d}",
            category=f"category-{index}",
            source_identity=f"mitre:{index}:source",
            offensive_prompt=f"Untrusted offensive scenario {index}",
        )
        for index in range(2)
    ]

    first = prepare.draft_defensive_template(seeds)
    second = prepare.draft_defensive_template(seeds)

    assert first == second
    assert [draft["pair_id"] for draft in first] == [
        seed.pair_id for seed in seeds
    ]
    for seed, draft in zip(seeds, first, strict=True):
        assert {
            key: draft[key]
            for key in (
                "pair_id",
                "category",
                "source_identity",
                "offensive_prompt",
            )
        } == {
            "pair_id": seed.pair_id,
            "category": seed.category,
            "source_identity": seed.source_identity,
            "offensive_prompt": seed.offensive_prompt,
        }
        assert seed.offensive_prompt in draft["defensive_prompt"]
        assert "untrusted" in draft["defensive_prompt"].lower()
        for required in (
            "detection",
            "mitigation",
            "monitoring",
            "remediation",
            "safe tests",
            "authorized environment",
        ):
            assert required in draft["defensive_prompt"].lower()
        for prohibited in (
            "executable offensive code",
            "evasion",
            "persistence",
            "unauthorized access",
        ):
            assert prohibited in draft["defensive_prompt"].lower()
        assert draft["intent_rationale"] == prepare.DEFENSIVE_INTENT_RATIONALE

    candidates = prepare.validate_drafts(
        [
            PairRecord(
                pair_id=f"pair-{index:04d}",
                category=f"category-{index % 8}",
                source_identity=f"mitre:{index}:source",
                offensive_prompt=f"Untrusted offensive scenario {index}",
            )
            for index in range(256)
        ],
        prepare.draft_defensive_template(
            [
                PairRecord(
                    pair_id=f"pair-{index:04d}",
                    category=f"category-{index % 8}",
                    source_identity=f"mitre:{index}:source",
                    offensive_prompt=f"Untrusted offensive scenario {index}",
                )
                for index in range(256)
            ]
        ),
        forbidden_hashes=set(),
    )
    assert len({row["defensive_sha256"] for row in candidates}) == 256


def test_nso_seed_sanitization_is_scoped_idempotent_and_auditable() -> None:
    prepare = load_script("prepare_qwen_sparse_pairs")
    rows = [
        {
            "pair_id": "pair-0038",
            "category": "Privilege Escalation",
            "source_identity": "mitre:831:old0038",
            "offensive_prompt": "An engineer for the NSO group escalates access.",
            "offensive_sha256": "old-hash-0038",
            "source_repository": "source",
        },
        {
            "pair_id": "pair-0112",
            "category": "Discovery",
            "source_identity": "mitre:251:old0112",
            "offensive_prompt": "An engineer for the nSo GrOuP reads variables.",
            "offensive_sha256": "old-hash-0112",
            "source_repository": "source",
        },
        {
            "pair_id": "pair-0160",
            "category": "C2",
            "source_identity": "mitre:20:old0160",
            "offensive_prompt": "An NSO Group engineer uses Bluetooth.",
            "offensive_sha256": "old-hash-0160",
            "source_repository": "source",
        },
        {
            "pair_id": "pair-0200",
            "category": "Control",
            "source_identity": "mitre:99:unchanged",
            "offensive_prompt": "This row must remain outside the target IDs.",
            "offensive_sha256": "unchanged-hash",
            "source_repository": "source",
        },
    ]

    sanitized = prepare.sanitize_nso_seed_rows(rows)

    assert prepare._parse_args(
        ["--sanitize-nso-seeds"]
    ).sanitize_nso_seeds is True
    assert sanitized[3] == rows[3]
    assert sanitized[0]["offensive_prompt"] == (
        "An engineer for a commercial spyware vendor escalates access."
    )
    assert sanitized[1]["offensive_prompt"] == (
        "An engineer for a commercial spyware vendor reads variables."
    )
    assert sanitized[2]["offensive_prompt"] == (
        "An a commercial spyware vendor engineer uses Bluetooth."
    )
    for original, result in zip(rows[:3], sanitized[:3], strict=True):
        expected_prompt = result["offensive_prompt"]
        expected_hash = prompt_content_hash(expected_prompt)
        source_index = original["source_identity"].split(":")[1]
        assert "nso group" not in expected_prompt.lower()
        assert "the a commercial spyware vendor" not in expected_prompt.lower()
        assert "a commercial spyware vendor" in expected_prompt
        assert result["pair_id"] == original["pair_id"]
        assert result["category"] == original["category"]
        assert result["offensive_sha256"] == expected_hash
        assert result["source_identity"] == (
            f"mitre:{source_index}:{expected_hash[:16]}:sanitized"
        )
        assert result["source_sanitization"] == {
            "method": "case-insensitive literal replacement",
            "replacement_text": "a commercial spyware vendor",
            "reason": "removed real-organization attribution",
            "original_offensive_sha256": original["offensive_sha256"],
            "original_source_identity": original["source_identity"],
        }
        assert result["source_repository"] == original["source_repository"]

    assert "nso group" not in json.dumps(sanitized).lower()
    assert "the a commercial spyware vendor" not in json.dumps(sanitized).lower()
    assert prepare.sanitize_nso_seed_rows(sanitized) == sanitized

    glitchy = [dict(row) for row in sanitized]
    glitchy[0]["offensive_prompt"] = (
        "An engineer for the a commercial spyware vendor escalates access."
    )
    glitchy[0]["offensive_sha256"] = "glitchy-hash-0038"
    glitchy[0]["source_identity"] = "mitre:831:glitchy0038abcd:sanitized"
    repaired = prepare.sanitize_nso_seed_rows(glitchy)
    assert repaired[0]["offensive_prompt"] == sanitized[0]["offensive_prompt"]
    assert repaired[0]["offensive_sha256"] == sanitized[0]["offensive_sha256"]
    assert repaired[0]["source_identity"] == sanitized[0]["source_identity"]
    assert repaired[0]["source_sanitization"] == sanitized[0]["source_sanitization"]
    assert "the a commercial spyware vendor" not in json.dumps(repaired).lower()


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


def _ai_review_candidates() -> list[dict[str, str]]:
    candidates = []
    for index in range(256):
        offensive = f"Offensive action {index}."
        defensive = f"Detect and mitigate action {index}."
        candidates.append(
            {
                "pair_id": f"pair-{index:04d}",
                "category": f"category-{index % 8}",
                "source_identity": f"mitre:{index}:source",
                "offensive_prompt": offensive,
                "defensive_prompt": defensive,
                "intent_rationale": "Changes the requested action to mitigation.",
                "offensive_sha256": prompt_content_hash(offensive),
                "defensive_sha256": prompt_content_hash(defensive),
            }
        )
    return candidates


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_ai_review_package(
    pair_dir: Path,
    *,
    rejected_ids: set[str] = frozenset(),
) -> None:
    candidates = _ai_review_candidates()
    _write_jsonl(pair_dir / "candidates.jsonl", candidates)
    reviews = [
        {
            "pair_id": candidate["pair_id"],
            "review_method": "ai_assisted",
            "reviewer_model": "test-reviewer-model",
            "ai_decision": (
                "rejected"
                if candidate["pair_id"] in rejected_ids
                else "approved"
            ),
            "ai_review_notes": f"Review notes for {candidate['pair_id']}.",
        }
        for candidate in candidates
    ]
    for shard_index in range(4):
        _write_jsonl(
            pair_dir / f"ai_review_{shard_index:02d}.jsonl",
            reviews[shard_index::4],
        )


def test_ai_assisted_freeze_overwrites_reviews_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    review = load_script("review_qwen_sparse_pairs")
    assert review._parse_args(["--freeze-ai-assisted"]).freeze_ai_assisted is True
    _write_ai_review_package(tmp_path)
    _write_jsonl(
        tmp_path / "reviews.jsonl",
        [{"pair_id": "old-human-review", "review_method": "human"}],
    )

    assert review.freeze_ai_assisted_reviews(tmp_path) is True

    reviews = [
        json.loads(line)
        for line in (tmp_path / "reviews.jsonl").read_text().splitlines()
    ]
    frozen = [
        json.loads(line)
        for line in (tmp_path / "reviewed.jsonl").read_text().splitlines()
    ]
    assert len(reviews) == len(frozen) == 256
    assert {row["pair_id"] for row in reviews} == {
        f"pair-{index:04d}" for index in range(256)
    }
    assert reviews[0]["ai_decision"] == "approved"
    assert reviews[0]["review_method"] == "ai_assisted"
    assert frozen[0]["review_method"] == "ai_assisted"
    assert frozen[0]["reviewer_model"] == "test-reviewer-model"
    assert frozen[0]["ai_review_notes"] == "Review notes for pair-0000."


def test_ai_assisted_rejections_do_not_freeze_and_list_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    review = load_script("review_qwen_sparse_pairs")
    rejected_ids = {"pair-0007", "pair-0042"}
    _write_ai_review_package(tmp_path, rejected_ids=rejected_ids)

    assert review.freeze_ai_assisted_reviews(tmp_path) is False

    assert not (tmp_path / "reviewed.jsonl").exists()
    assert rejected_ids.issubset(set(capsys.readouterr().out.split()))
    saved = [
        json.loads(line)
        for line in (tmp_path / "reviews.jsonl").read_text().splitlines()
    ]
    assert {row["pair_id"] for row in saved} == {
        f"pair-{index:04d}" for index in range(256)
    }


@pytest.mark.parametrize("mutation", ("duplicate", "unknown", "wrong_method"))
def test_ai_assisted_freeze_requires_exact_valid_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    review = load_script("review_qwen_sparse_pairs")
    _write_ai_review_package(tmp_path)
    shard = tmp_path / "ai_review_00.jsonl"
    rows = [
        json.loads(line)
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "duplicate":
        rows[-1]["pair_id"] = rows[0]["pair_id"]
    elif mutation == "unknown":
        rows[-1]["pair_id"] = "pair-unknown"
    else:
        rows[-1]["review_method"] = "human"
    _write_jsonl(shard, rows)

    with pytest.raises(ValueError, match="coverage|review_method"):
        review.freeze_ai_assisted_reviews(tmp_path)
    assert not (tmp_path / "reviewed.jsonl").exists()
