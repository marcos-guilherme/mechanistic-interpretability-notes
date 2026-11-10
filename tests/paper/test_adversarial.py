from __future__ import annotations

from dataclasses import dataclass
import random
import re

import pytest
import torch

from ioi_replication.paper.adversarial import (
    ADVERSARIAL_CLAUSE_TEMPLATES,
    DOUBLE_ADVERSARIAL_TEMPLATES,
    EASY_TRANSFORMER_SOURCE_COMMIT,
    PAPER_ADVERSARIAL_TARGETS,
    PAPER_FIGURE_8_TARGETS,
    build_adversarial_dataset,
    insert_adversarial_clauses,
    paired_adversarial_metrics,
)
from ioi_replication.runtime import load_gpt2_small, select_device


@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class WhitespaceTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def __init__(self, padding_side: str = "right") -> None:
        self.padding_side = padding_side
        self._ids: dict[str, int] = {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        assert add_special_tokens is False
        return [self._token_id(text.strip())]

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


def test_adversarial_api_is_exported_from_paper_package() -> None:
    import ioi_replication.paper as paper

    assert paper.build_adversarial_dataset is build_adversarial_dataset
    assert paper.paired_adversarial_metrics is paired_adversarial_metrics
    assert paper.PAPER_ADVERSARIAL_TARGETS is PAPER_ADVERSARIAL_TARGETS
    assert paper.PAPER_FIGURE_8_TARGETS is PAPER_FIGURE_8_TARGETS


def test_historical_adversarial_templates_and_source_are_literal() -> None:
    assert ADVERSARIAL_CLAUSE_TEMPLATES == (
        " [A] had a good day.",
        " [A] was enjoying the situation.",
        " [A] was tired.",
        " [A] enjoyed being with a friend.",
        " [A] was an enthusiast person.",
    )
    assert DOUBLE_ADVERSARIAL_TEMPLATES == tuple(
        first + second
        for first in ADVERSARIAL_CLAUSE_TEMPLATES
        for second in ADVERSARIAL_CLAUSE_TEMPLATES
    )
    assert len(DOUBLE_ADVERSARIAL_TEMPLATES) == 25
    assert EASY_TRANSFORMER_SOURCE_COMMIT == (
        "ea15315dd24481e9e2ac5c3ef335d82907a1dc34"
    )
    assert PAPER_ADVERSARIAL_TARGETS == {
        "adversarial_logit_difference": 1.23,
        "adversarial_s_wins_error_rate": 0.234,
    }
    assert PAPER_FIGURE_8_TARGETS == {
        "baseline": {
            "logit_difference": 3.55,
            "io_probability": 0.49,
            "s_wins_error_rate": 0.007,
        },
        "paper_s_single": {
            "logit_difference": 3.64,
            "io_probability": 0.59,
            "s_wins_error_rate": 0.004,
        },
        "paper_io_single": {
            "logit_difference": 1.23,
            "io_probability": 0.36,
            "s_wins_error_rate": 0.234,
        },
    }


def test_insertion_matches_advex_middle_punctuation_semantics() -> None:
    prompt_with_period = (
        "Then, Kyle and Jessica were thinking about going to the office. "
        "Kyle wanted to give a computer to Jessica"
    )
    clauses = DOUBLE_ADVERSARIAL_TEMPLATES[1]
    assert insert_adversarial_clauses(
        prompt_with_period,
        io_name="Jessica",
        template=clauses,
    ) == (
        "Then, Kyle and Jessica were thinking about going to the office."
        " Jessica had a good day. Jessica was enjoying the situation. "
        "Kyle wanted to give a computer to Jessica"
    )

    prompt_with_commas = (
        "Then, Kyle and Jessica had a long argument, and afterwards "
        "Kyle said to Jessica"
    )
    assert insert_adversarial_clauses(
        prompt_with_commas,
        io_name="Jessica",
        template=DOUBLE_ADVERSARIAL_TEMPLATES[0],
    ) == (
        "Then, Kyle and Jessica had a long argument,"
        " Jessica had a good day. Jessica had a good day. and afterwards "
        "Kyle said to Jessica"
    )


def test_dataset_is_deterministic_and_uses_historical_double_choice_order() -> None:
    tokenizer = WhitespaceTokenizer()
    first = build_adversarial_dataset(
        tokenizer,
        n=12,
        base_seed=7,
        attack_seed=17,
        clause_count="double",
    )
    repeated = build_adversarial_dataset(
        tokenizer,
        n=12,
        base_seed=7,
        attack_seed=17,
        clause_count="double",
    )
    different = build_adversarial_dataset(
        tokenizer,
        n=12,
        base_seed=7,
        attack_seed=18,
        clause_count="double",
    )

    assert first.base_prompts == repeated.base_prompts
    assert first.adversarial_prompts == repeated.adversarial_prompts
    assert first.dataset_sha256 == repeated.dataset_sha256
    assert first.dataset_sha256 != different.dataset_sha256
    expected_first_template = random.Random(17).choice(
        DOUBLE_ADVERSARIAL_TEMPLATES
    )
    assert first.metadata[0].adversarial_template == expected_first_template
    assert first.prepend_bos is False
    assert first.base_seed == 7
    assert first.attack_seed == 17


def test_base_and_attack_seeds_control_only_their_own_sampling() -> None:
    tokenizer = WhitespaceTokenizer()
    first = build_adversarial_dataset(
        tokenizer,
        n=40,
        base_seed=101,
        attack_seed=201,
        role="IO",
        clause_count="single",
        allowed_template_ids=range(7),
    )
    changed_attack = build_adversarial_dataset(
        tokenizer,
        n=40,
        base_seed=101,
        attack_seed=202,
        role="IO",
        clause_count="single",
        allowed_template_ids=range(7),
    )
    changed_base = build_adversarial_dataset(
        tokenizer,
        n=40,
        base_seed=102,
        attack_seed=201,
        role="IO",
        clause_count="single",
        allowed_template_ids=range(7),
    )

    assert first.base_prompts == changed_attack.base_prompts
    assert first.adversarial_prompts != changed_attack.adversarial_prompts
    assert [
        item.adversarial_template for item in first.metadata
    ] == [
        item.adversarial_template for item in changed_base.metadata
    ]
    assert first.base_prompts != changed_base.base_prompts


@pytest.mark.parametrize(
    ("role", "role_occurrences", "other_occurrences"),
    [("IO", 3, 2), ("S", 3, 2)],
)
def test_single_clause_attack_tracks_inserted_role_occurrence(
    role: str,
    role_occurrences: int,
    other_occurrences: int,
) -> None:
    dataset = build_adversarial_dataset(
        WhitespaceTokenizer(),
        n=80,
        base_seed=9,
        attack_seed=19,
        role=role,
        clause_count="single",
        allowed_template_ids=range(7),
    )

    assert dataset.attack_role == role
    assert dataset.clause_count == "single"
    assert dataset.inserted_positions.shape == (len(dataset), 1)
    assert set(dataset.template_ids.tolist()) == set(range(7))
    assert {item.pattern for item in dataset.metadata} == {"ABBA", "BABA"}
    assert dataset.metadata[0].adversarial_template == random.Random(
        19
    ).choice(ADVERSARIAL_CLAUSE_TEMPLATES)
    assert torch.equal(
        dataset.adversarial_positions["END"],
        dataset.adversarial_attention_mask.sum(dim=1) - 2,
    )
    assert not torch.equal(
        dataset.base_positions["S2"],
        dataset.adversarial_positions["S2"],
    )
    for row in range(len(dataset)):
        length = int(dataset.adversarial_attention_mask[row].sum())
        io_matches = (
            dataset.adversarial_tokens[row, :length]
            == dataset.io_token_ids[row]
        ).nonzero().flatten()
        s_matches = (
            dataset.adversarial_tokens[row, :length]
            == dataset.s_token_ids[row]
        ).nonzero().flatten()
        attacked_matches = io_matches if role == "IO" else s_matches
        other_matches = s_matches if role == "IO" else io_matches
        assert len(attacked_matches) == role_occurrences
        assert len(other_matches) == other_occurrences
        assert dataset.inserted_positions[row, 0] in attacked_matches
        assert dataset.inserted_positions[row, 0] not in {
            int(dataset.adversarial_positions["IO"][row]),
            int(dataset.adversarial_positions["S1"][row]),
            int(dataset.adversarial_positions["S2"][row]),
        }


def test_names_positions_and_inserted_occurrences_are_distinct() -> None:
    dataset = build_adversarial_dataset(
        WhitespaceTokenizer(),
        n=40,
        base_seed=9,
        attack_seed=10,
        role="IO",
        clause_count="double",
        allowed_template_ids=range(7),
    )
    rows = torch.arange(len(dataset))

    assert torch.equal(
        dataset.base_tokens[rows, dataset.base_positions["IO"]],
        dataset.io_token_ids,
    )
    assert torch.equal(
        dataset.base_tokens[rows, dataset.base_positions["S1"]],
        dataset.s_token_ids,
    )
    assert torch.equal(
        dataset.base_tokens[rows, dataset.base_positions["S2"]],
        dataset.s_token_ids,
    )
    assert torch.equal(
        dataset.adversarial_tokens[
            rows, dataset.adversarial_positions["IO"]
        ],
        dataset.io_token_ids,
    )
    assert torch.equal(
        dataset.adversarial_tokens[
            rows, dataset.adversarial_positions["S1"]
        ],
        dataset.s_token_ids,
    )
    assert torch.equal(
        dataset.adversarial_tokens[
            rows, dataset.adversarial_positions["S2"]
        ],
        dataset.s_token_ids,
    )
    assert dataset.inserted_positions.shape == (len(dataset), 2)
    assert torch.equal(dataset.extra_io_positions, dataset.inserted_positions)
    assert torch.all(
        dataset.adversarial_positions["IO"][:, None]
        < dataset.inserted_positions
    )
    assert torch.all(
        dataset.inserted_positions
        < dataset.adversarial_positions["S2"][:, None]
    )
    assert not torch.equal(
        dataset.base_positions["S2"],
        dataset.adversarial_positions["S2"],
    )
    for row in range(len(dataset)):
        length = int(dataset.adversarial_attention_mask[row].sum())
        matches = (
            dataset.adversarial_tokens[row, :length]
            == dataset.io_token_ids[row]
        ).nonzero().flatten()
        assert len(matches) == 4
        assert torch.equal(matches[1:3], dataset.inserted_positions[row])


def test_end_positions_follow_each_prompts_own_unpadded_length() -> None:
    dataset = build_adversarial_dataset(
        WhitespaceTokenizer(),
        n=80,
        seed=12,
    )
    base_lengths = dataset.base_attention_mask.sum(dim=1)
    adversarial_lengths = dataset.adversarial_attention_mask.sum(dim=1)

    assert len(base_lengths.unique()) > 1
    assert len(adversarial_lengths.unique()) > 1
    assert torch.equal(dataset.base_positions["END"], base_lengths - 2)
    assert torch.equal(
        dataset.adversarial_positions["END"],
        adversarial_lengths - 2,
    )
    assert torch.all(
        dataset.adversarial_positions["END"] > dataset.base_positions["END"]
    )


def test_dataset_validates_configuration() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        build_adversarial_dataset(WhitespaceTokenizer(), n=0)
    with pytest.raises(ValueError, match="right padding"):
        build_adversarial_dataset(
            WhitespaceTokenizer(padding_side="left"),
            n=2,
        )
    with pytest.raises(ValueError, match="role"):
        build_adversarial_dataset(
            WhitespaceTokenizer(),
            n=2,
            role="invalid",
        )
    with pytest.raises(ValueError, match="clause_count"):
        build_adversarial_dataset(
            WhitespaceTokenizer(),
            n=2,
            clause_count="invalid",
        )


def test_paired_metrics_use_per_example_deltas_and_strict_s_wins() -> None:
    dataset = build_adversarial_dataset(
        WhitespaceTokenizer(),
        n=3,
        seed=21,
    )
    vocab_size = int(
        max(dataset.base_tokens.max(), dataset.adversarial_tokens.max()).item()
        + 3
    )
    base_logits = torch.zeros(3, dataset.base_tokens.shape[1], vocab_size)
    adversarial_logits = torch.zeros(
        3,
        dataset.adversarial_tokens.shape[1],
        vocab_size,
    )
    base_differences = torch.tensor([3.0, 1.0, -1.0])
    adversarial_differences = torch.tensor([1.0, -2.0, 0.0])
    for row in range(3):
        base_end = dataset.base_positions["END"][row]
        adversarial_end = dataset.adversarial_positions["END"][row]
        io_id = dataset.io_token_ids[row]
        s_id = dataset.s_token_ids[row]
        base_logits[row, base_end, io_id] = base_differences[row]
        adversarial_logits[row, adversarial_end, io_id] = (
            adversarial_differences[row]
        )
        base_logits[row, base_end, s_id] = 0
        adversarial_logits[row, adversarial_end, s_id] = 0

    result = paired_adversarial_metrics(
        base_logits,
        adversarial_logits,
        dataset,
    )

    assert result.logit_difference.baseline.values.tolist() == pytest.approx(
        [3.0, 1.0, -1.0]
    )
    assert result.logit_difference.adversarial.values.tolist() == pytest.approx(
        [1.0, -2.0, 0.0]
    )
    assert result.logit_difference.delta.values.tolist() == pytest.approx(
        [-2.0, -3.0, 1.0]
    )
    assert result.s_wins_error_rate.baseline.mean == pytest.approx(1 / 3)
    assert result.s_wins_error_rate.adversarial.mean == pytest.approx(1 / 3)
    assert result.s_wins_error_rate.adversarial.values.tolist() == [
        0.0,
        1.0,
        0.0,
    ]
    assert result.s_wins_error_rate.baseline.ci_method == "wilson"
    assert result.s_wins_error_rate.adversarial.ci_method == "wilson"
    assert result.s_wins_error_rate.delta.ci_method == (
        "normal_paired_1.96_se"
    )
    assert 0 <= result.s_wins_error_rate.baseline.ci95[0]
    assert result.s_wins_error_rate.baseline.ci95[1] <= 1
    assert 0 <= result.s_wins_error_rate.adversarial.ci95[0]
    assert result.s_wins_error_rate.adversarial.ci95[1] <= 1
    expected_se = torch.tensor([-2.0, -3.0, 1.0]).std(
        unbiased=True
    ).item() / 3**0.5
    assert result.logit_difference.delta.standard_error == pytest.approx(
        expected_se
    )
    assert result.logit_difference.delta.ci95 == pytest.approx(
        (
            -4 / 3 - 1.959963984540054 * expected_se,
            -4 / 3 + 1.959963984540054 * expected_se,
        )
    )
    assert (
        result.io_probability.delta.values
        == result.io_probability.adversarial.values
        - result.io_probability.baseline.values
    ).all()


def test_real_gpt2_small_adversarial_smoke() -> None:
    try:
        model = load_gpt2_small(select_device())
    except (OSError, RuntimeError) as error:
        pytest.skip(f"cached GPT-2 Small is unavailable: {error}")
    dataset = build_adversarial_dataset(model.tokenizer, n=2, seed=31)
    device = torch.device(model.cfg.device)

    with torch.inference_mode():
        result = paired_adversarial_metrics(
            model(dataset.base_tokens.to(device)),
            model(dataset.adversarial_tokens.to(device)),
            dataset,
        )

    assert torch.isfinite(result.logit_difference.baseline.values).all()
    assert torch.isfinite(result.logit_difference.adversarial.values).all()
    assert (result.io_probability.baseline.values >= 0).all()
    assert (result.io_probability.baseline.values <= 1).all()
