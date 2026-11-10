from __future__ import annotations

from dataclasses import dataclass
import re

import pytest
import torch

from ioi_replication.paper.dataset import (
    ABBA_TEMPLATES,
    BABA_TEMPLATES,
    HISTORICAL_NAMES,
    HISTORICAL_OBJECTS,
    HISTORICAL_PLACES,
    build_paired_dataset,
    filter_single_token_vocabulary,
)


@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class WhitespaceTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, split_words: set[str] | None = None) -> None:
        self.split_words = split_words or set()
        self._ids: dict[str, int] = {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        assert add_special_tokens is False
        word = text.strip()
        if word in self.split_words:
            return [998, 999]
        return [self._token_id(word)]

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> _Batch:
        assert padding is True
        assert add_special_tokens is False
        assert return_tensors == "pt"
        rows = [
            [
                self._token_id(piece)
                for piece in re.findall(r"[A-Za-z]+|[^\w\s]", text)
            ]
            for text in texts
        ]
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_token_id)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        return _Batch(input_ids=input_ids, attention_mask=attention_mask)

    def _token_id(self, piece: str) -> int:
        if piece not in self._ids:
            self._ids[piece] = len(self._ids) + 1
        return self._ids[piece]


def test_historical_templates_and_vocabularies_are_available() -> None:
    assert len(BABA_TEMPLATES) == len(ABBA_TEMPLATES) == 15
    assert len(HISTORICAL_NAMES) == 99
    assert HISTORICAL_PLACES == (
        "store",
        "garden",
        "restaurant",
        "school",
        "hospital",
        "office",
        "house",
        "station",
    )
    assert HISTORICAL_OBJECTS == (
        "ring",
        "kiss",
        "bone",
        "basketball",
        "computer",
        "necklace",
        "drink",
        "snack",
    )


def test_single_token_filter_removes_split_items() -> None:
    tokenizer = WhitespaceTokenizer(split_words={"Mary", "garden"})

    filtered = filter_single_token_vocabulary(
        tokenizer,
        names=("John", "Mary", "Alice"),
        places=("store", "garden"),
        objects=("ring", "snack"),
    )

    assert filtered.names == ("John", "Alice")
    assert filtered.places == ("store",)
    assert filtered.objects == ("ring", "snack")


def test_dataset_is_deterministic_and_supports_small_n() -> None:
    tokenizer = WhitespaceTokenizer()

    first = build_paired_dataset(tokenizer, n=4, seed=17)
    second = build_paired_dataset(tokenizer, n=4, seed=17)
    different = build_paired_dataset(tokenizer, n=4, seed=18)

    assert first.ioi_prompts == second.ioi_prompts
    assert first.abc_prompts == second.abc_prompts
    assert torch.equal(first.ioi_tokens, second.ioi_tokens)
    assert first.ioi_prompts != different.ioi_prompts
    assert len(first) == 4
    assert first.prepend_bos is False


def test_pioi_pabc_pairs_are_matched_and_grouped_by_template() -> None:
    dataset = build_paired_dataset(
        WhitespaceTokenizer(),
        n=31,
        seed=5,
        pattern="mixed",
    )

    covered_rows = torch.cat(tuple(dataset.groups.values())).sort().values
    assert torch.equal(covered_rows, torch.arange(len(dataset)))
    assert set(dataset.groups) == set(dataset.template_ids.tolist())
    assert torch.equal(dataset.ioi_attention_mask, dataset.abc_attention_mask)
    assert dataset.ioi_tokens.shape == dataset.abc_tokens.shape

    for index, metadata in enumerate(dataset.metadata):
        assert len(
            {metadata.io_name, metadata.subject_name, metadata.abc_name}
        ) == 3
        assert metadata.template_id in dataset.groups
        assert index in dataset.groups[metadata.template_id].tolist()
        assert dataset.ioi_prompts[index] != dataset.abc_prompts[index]


@pytest.mark.parametrize("pattern", ["BABA", "ABBA", "mixed"])
def test_semantic_positions_point_to_the_expected_tokens(pattern: str) -> None:
    dataset = build_paired_dataset(
        WhitespaceTokenizer(),
        n=8,
        seed=9,
        pattern=pattern,
    )

    rows = torch.arange(len(dataset))
    assert torch.equal(
        dataset.ioi_tokens[rows, dataset.positions["IO"]],
        dataset.io_token_ids,
    )
    assert torch.equal(
        dataset.ioi_tokens[rows, dataset.positions["S1"]],
        dataset.s_token_ids,
    )
    assert torch.equal(
        dataset.ioi_tokens[rows, dataset.positions["S2"]],
        dataset.s_token_ids,
    )
    assert torch.equal(
        dataset.positions["S1+1"],
        dataset.positions["S1"] + 1,
    )
    assert torch.equal(
        dataset.ioi_tokens[rows, dataset.positions["END"] + 1],
        dataset.io_token_ids,
    )
    assert torch.equal(
        dataset.abc_tokens[rows, dataset.positions["S2"]],
        dataset.abc_token_ids,
    )


def test_dataset_rejects_invalid_configuration() -> None:
    tokenizer = WhitespaceTokenizer()

    with pytest.raises(ValueError, match="n must be positive"):
        build_paired_dataset(tokenizer, n=0)
    with pytest.raises(ValueError, match="pattern"):
        build_paired_dataset(tokenizer, n=2, pattern="invalid")
