from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _rows(categories: tuple[str, ...], per_category: int = 12) -> list[dict]:
    return [
        {
            "mutated_prompt": f"prompt-{category}-{index}",
            "mitre_category": category,
        }
        for category in categories
        for index in range(per_category)
    ]


def test_stratified_selection_is_deterministic_balanced_and_text_free() -> None:
    from refusal_direction_replication.qwen_code_eval import select_records

    rows = _rows(("C2", "Discovery", "Execution", "Persistence"))
    first, first_meta = select_records(
        rows,
        dataset="mitre",
        text_field="mutated_prompt",
        category_field="mitre_category",
        size=32,
        seed=2026,
    )
    second, second_meta = select_records(
        rows,
        dataset="mitre",
        text_field="mutated_prompt",
        category_field="mitre_category",
        size=32,
        seed=2026,
    )

    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert first_meta == second_meta
    assert first_meta["counts_by_category"] == {
        "C2": 8,
        "Discovery": 8,
        "Execution": 8,
        "Persistence": 8,
    }
    assert len(first_meta["example_ids"]) == 32
    assert len(first_meta["content_hashes"]) == 32
    assert "prompt" not in json.dumps(first_meta).casefold()


def test_selection_rejects_short_or_duplicate_corpora() -> None:
    from refusal_direction_replication.qwen_code_eval import select_records

    rows = [{"prompt": "same", "category": "a"} for _ in range(40)]
    with pytest.raises(ValueError, match="unique"):
        select_records(
            rows,
            dataset="mitre-frr",
            text_field="prompt",
            category_field="category",
            size=32,
            seed=7,
        )


def test_compact_aggregation_recomputes_refusal_kl_and_pass_at_one() -> None:
    from refusal_direction_replication.qwen_code_eval import aggregate_records

    records = []
    for condition in ("baseline", "ablation", "addition", "random_control"):
        for dataset in ("mitre", "mitre_frr", "humaneval_plus"):
            for index in range(32):
                records.append(
                    {
                        "condition": condition,
                        "dataset": dataset,
                        "example_id": f"{dataset}:{index}",
                        "refusal": index % 4 == 0,
                        "valid": True,
                        "kl_from_baseline": (
                            None if condition == "baseline" else float(index) / 100
                        ),
                        "evalplus_pass": (
                            index % 2 == 0 if dataset == "humaneval_plus" else None
                        ),
                    }
                )

    compact = aggregate_records(records)

    assert compact["counts"]["mitre"]["baseline"] == 32
    assert compact["metrics"]["mitre"]["baseline"]["refusal_rate"] == 0.25
    assert compact["metrics"]["mitre_frr"]["ablation"]["mean_kl"] == pytest.approx(
        0.155
    )
    assert (
        compact["metrics"]["humaneval_plus"]["addition"]["pass_at_1"] == 0.5
    )
    assert all(
        key not in json.dumps(compact).casefold()
        for key in ('"prompt"', '"response"', '"generation"', '"code"', '"solution"')
    )


class _RoundTripTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {
            "rendered": [10, 11],
            "renderedX": [10, 11, 99],
            "renderedY": [10, 12, 98],
        }[text]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return {99: "X", 98: "Y"}[ids[0]]


def test_first_response_position_requires_exact_round_trip() -> None:
    from refusal_direction_replication.qwen_code_eval import first_response_position

    tokenizer = _RoundTripTokenizer()
    assert first_response_position(tokenizer, "rendered", 99) == {
        "available": True,
        "position": 2,
        "token_id": 99,
    }
    assert first_response_position(tokenizer, "rendered", 98) == {
        "available": False,
        "reason": "round_trip_mismatch",
    }


def test_jlens_summary_uses_minimum_lexicon_rank_by_layer_and_position() -> None:
    from refusal_direction_replication.qwen_code_eval import summarize_refusal_ranks

    records = [
        {
            "dataset": "mitre",
            "condition": "baseline",
            "example_id": "m:0",
            "position_kind": position,
            "layer": layer,
            "token_ranks": {101: 9 + layer, 202: 3 + layer},
        }
        for position in ("instruction_last", "response_first")
        for layer in (0, 1)
    ]
    summary = summarize_refusal_ranks(records, lexicon_token_ids=(101, 202))

    layer_zero = summary["mitre"]["baseline"]["instruction_last"]["0"]
    assert layer_zero == {
        "n_available": 1,
        "n_unavailable": 0,
        "mean_min_rank": 3.0,
        "median_min_rank": 3.0,
        "hit_at_1": 0.0,
        "hit_at_5": 1.0,
        "min_rank": 3,
    }


def test_blind_review_sample_is_deterministic_and_keeps_key_separate(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval import write_blind_review_sample

    records = [
        {
            "dataset": "mitre_frr",
            "condition": "baseline",
            "example_id": f"frr:{index}",
            "response": f"answer-{index}",
            "refusal": index % 2 == 0,
        }
        for index in range(20)
    ]
    public_path = tmp_path / "blind.jsonl"
    key_path = tmp_path / "key.json"
    compact = write_blind_review_sample(
        records,
        sample_path=public_path,
        key_path=key_path,
        sample_size=8,
        seed=2026,
    )

    sample = [
        json.loads(line)
        for line in public_path.read_text(encoding="utf-8").splitlines()
    ]
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert compact["human_review_status"] == "pending"
    assert compact["sample_size"] == 8
    assert all("refusal" not in item for item in sample)
    assert all("response" in item for item in sample)
    assert {item["blind_id"] for item in sample} == set(key)
    assert all("example_id" in value and "automatic_refusal" in value for value in key.values())


def test_source_identity_hashes_files_and_requires_pinned_revisions(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_code_eval import build_source_identity

    source = tmp_path / "dataset.json"
    source.write_text("[]\n", encoding="utf-8")
    identity = build_source_identity(
        source,
        repository="meta-llama/PurpleLlama",
        revision="0654501399e7cbd560db9501965dc09c567a5563",
        relative_path="datasets/example.json",
    )
    assert identity["revision"] == "0654501399e7cbd560db9501965dc09c567a5563"
    assert len(identity["sha256"]) == 64
    assert identity["bytes"] == source.stat().st_size


def test_rank_tokens_returns_one_based_ranks() -> None:
    from refusal_direction_replication.qwen_code_eval import rank_token_ids

    logits = torch.tensor([0.0, 4.0, 2.0, 3.0])
    assert rank_token_ids(logits, (1, 2, 3)) == {1: 1, 2: 3, 3: 2}
