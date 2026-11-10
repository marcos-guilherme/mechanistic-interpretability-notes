import pytest
import torch

from ioi_replication.paper.metrics import (
    io_accuracy,
    io_probability,
    logit_difference,
    logits_at_end,
)


def _example_logits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.zeros(3, 5, 7)
    end_positions = torch.tensor([1, 3, 2])
    io_tokens = torch.tensor([1, 2, 3])
    s_tokens = torch.tensor([4, 5, 6])

    logits[0, 1, [1, 4]] = torch.tensor([4.0, 1.0])
    logits[1, 3, [2, 5]] = torch.tensor([2.0, 3.0])
    logits[2, 2, [3, 6]] = torch.tensor([0.5, 0.5])
    logits[:, -1, :] = 100.0
    return logits, io_tokens, s_tokens, end_positions


def test_logits_at_end_gathers_each_examples_semantic_end() -> None:
    logits, _, _, end_positions = _example_logits()

    selected = logits_at_end(logits, end_positions)

    assert selected.shape == (3, 7)
    assert selected[0, 1].item() == 4.0
    assert selected[1, 2].item() == 2.0
    assert selected[2, 3].item() == 0.5
    assert selected.max().item() < 100.0


def test_logit_difference_supports_per_example_and_mean() -> None:
    logits, io_tokens, s_tokens, end_positions = _example_logits()

    per_example = logit_difference(
        logits,
        io_tokens,
        s_tokens,
        end_positions,
        reduction="none",
    )

    assert torch.equal(per_example, torch.tensor([3.0, -1.0, 0.0]))
    assert logit_difference(
        logits,
        io_tokens,
        s_tokens,
        end_positions,
    ).item() == pytest.approx(2 / 3)


def test_accuracy_counts_strict_io_over_subject_wins() -> None:
    logits, io_tokens, s_tokens, end_positions = _example_logits()

    accuracy = io_accuracy(logits, io_tokens, s_tokens, end_positions)

    assert accuracy.item() == pytest.approx(1 / 3)


def test_io_probability_uses_full_vocabulary_softmax_at_end() -> None:
    logits, io_tokens, _, end_positions = _example_logits()

    per_example = io_probability(
        logits,
        io_tokens,
        end_positions,
        reduction="none",
    )
    expected = logits_at_end(logits, end_positions).softmax(dim=-1).gather(
        dim=-1,
        index=io_tokens[:, None],
    )[:, 0]

    assert torch.allclose(per_example, expected)
    assert io_probability(
        logits,
        io_tokens,
        end_positions,
    ).item() == pytest.approx(expected.mean().item())


@pytest.mark.parametrize(
    ("logits", "end_positions", "message"),
    [
        (torch.zeros(2, 3), torch.tensor([1, 1]), "three dimensions"),
        (torch.zeros(2, 3, 4), torch.tensor([1]), "batch"),
        (torch.zeros(2, 3, 4), torch.tensor([1, 3]), "within"),
    ],
)
def test_logits_at_end_validates_shapes(
    logits: torch.Tensor,
    end_positions: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        logits_at_end(logits, end_positions)
