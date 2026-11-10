from __future__ import annotations

from dataclasses import dataclass
import re

import pytest
import torch
from transformers import AutoTokenizer

import ioi_replication.paper.dataset as paper_dataset
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
    padding_side = "right"

    def __init__(
        self,
        split_words: set[str] | None = None,
        padding_side: str = "right",
    ) -> None:
        self.split_words = split_words or set()
        self.padding_side = padding_side
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
            if self.padding_side == "right":
                input_ids[row_index, : len(row)] = torch.tensor(row)
                attention_mask[row_index, : len(row)] = 1
            else:
                input_ids[row_index, -len(row) :] = torch.tensor(row)
                attention_mask[row_index, -len(row) :] = 1
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
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.dataset_sha256 != different.dataset_sha256


def test_pabc_replaces_all_three_roles_with_unrelated_names() -> None:
    dataset = build_paired_dataset(
        WhitespaceTokenizer(),
        n=31,
        seed=5,
        pattern="mixed",
    )

    assert torch.equal(dataset.ioi_attention_mask, dataset.abc_attention_mask)
    assert dataset.ioi_tokens.shape == dataset.abc_tokens.shape

    for index, metadata in enumerate(dataset.metadata):
        original_names = {metadata.io_name, metadata.subject_name}
        abc_names = {
            metadata.abc_io_name,
            metadata.abc_s1_name,
            metadata.abc_s2_name,
        }
        assert len(abc_names) == 3
        assert original_names.isdisjoint(abc_names)

        positions = dataset.positions
        assert (
            dataset.abc_tokens[index, positions["IO"][index]]
            == dataset.abc_io_token_ids[index]
        )
        assert (
            dataset.abc_tokens[index, positions["S1"][index]]
            == dataset.abc_s1_token_ids[index]
        )
        assert (
            dataset.abc_tokens[index, positions["S2"][index]]
            == dataset.abc_s2_token_ids[index]
        )
        assert (
            dataset.abc_tokens[index, positions["END"][index] + 1]
            == dataset.abc_io_token_ids[index]
        )


def test_groups_distinguish_pattern_and_template() -> None:
    dataset = build_paired_dataset(
        WhitespaceTokenizer(),
        n=200,
        seed=5,
        pattern="mixed",
    )

    covered_rows = torch.cat(tuple(dataset.groups.values())).sort().values
    assert torch.equal(covered_rows, torch.arange(len(dataset)))
    assert len(dataset.groups) <= 30
    for key, indices in dataset.groups.items():
        pattern, template_id = key
        assert pattern in {"ABBA", "BABA"}
        assert 0 <= template_id < 15
        assert all(
            dataset.metadata[index].pattern == pattern
            and dataset.metadata[index].template_id == template_id
            for index in indices.tolist()
        )


def test_balanced_dataset_covers_every_pattern_template_group_equally() -> None:
    dataset = paper_dataset.build_balanced_paired_dataset(
        WhitespaceTokenizer(),
        examples_per_group=3,
        seed=23,
    )

    assert len(dataset) == 90
    assert len(dataset.groups) == 30
    assert set(dataset.groups) == {
        (pattern, template_id)
        for pattern in ("ABBA", "BABA")
        for template_id in range(15)
    }
    assert {len(indices) for indices in dataset.groups.values()} == {3}
    repeated = paper_dataset.build_balanced_paired_dataset(
        WhitespaceTokenizer(),
        examples_per_group=3,
        seed=23,
    )
    assert dataset.dataset_sha256 == repeated.dataset_sha256


def test_balanced_dataset_rejects_nonpositive_group_size() -> None:
    with pytest.raises(ValueError, match="examples_per_group"):
        paper_dataset.build_balanced_paired_dataset(
            WhitespaceTokenizer(),
            examples_per_group=0,
        )


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
        dataset.abc_s2_token_ids,
    )


def test_abba_and_baba_preserve_expected_io_subject_order() -> None:
    tokenizer = WhitespaceTokenizer()
    abba = build_paired_dataset(tokenizer, n=8, seed=4, pattern="ABBA")
    baba = build_paired_dataset(tokenizer, n=8, seed=4, pattern="BABA")

    assert all(
        io < s1 < s2 < end
        for io, s1, s2, end in zip(
            abba.positions["IO"],
            abba.positions["S1"],
            abba.positions["S2"],
            abba.positions["END"],
        )
    )
    assert all(
        s1 < io < s2 < end
        for io, s1, s2, end in zip(
            baba.positions["IO"],
            baba.positions["S1"],
            baba.positions["S2"],
            baba.positions["END"],
        )
    )
    assert ABBA_TEMPLATES[0].startswith(
        "Then, [A] and [B] went to the [PLACE]. [B]"
    )


def test_end_positions_follow_each_unpadded_length() -> None:
    dataset = build_paired_dataset(
        WhitespaceTokenizer(),
        n=40,
        seed=12,
    )
    lengths = dataset.ioi_attention_mask.sum(dim=1)

    assert len(lengths.unique()) > 1
    assert torch.equal(dataset.positions["END"], lengths - 2)


def test_dataset_rejects_left_padding() -> None:
    with pytest.raises(ValueError, match="right padding"):
        build_paired_dataset(
            WhitespaceTokenizer(padding_side="left"),
            n=2,
        )


def test_dataset_requires_explicit_pad_token_without_mutating_tokenizer() -> None:
    tokenizer = WhitespaceTokenizer()
    tokenizer.pad_token_id = None

    with pytest.raises(ValueError, match="pad token explicitly"):
        build_paired_dataset(tokenizer, n=2)
    assert tokenizer.pad_token_id is None


def test_real_gpt2_tokenizer_smoke() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("GPT-2 tokenizer is not available in the local cache")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_paired_dataset(tokenizer, n=4, seed=3)

    assert dataset.ioi_tokens.shape == dataset.abc_tokens.shape
    assert torch.equal(
        dataset.positions["END"],
        dataset.ioi_attention_mask.sum(dim=1) - 2,
    )


def test_dataset_rejects_invalid_configuration() -> None:
    tokenizer = WhitespaceTokenizer()

    with pytest.raises(ValueError, match="n must be positive"):
        build_paired_dataset(tokenizer, n=0)
    with pytest.raises(ValueError, match="pattern"):
        build_paired_dataset(tokenizer, n=2, pattern="invalid")
