import torch

from refusal_direction_replication.model import (
    format_instruction,
    resolve_post_instruction_positions,
)


def test_format_instruction_matches_upstream_gemma_template() -> None:
    assert format_instruction("hello") == (
        "<start_of_turn>user\nhello<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_post_instruction_positions_are_relative_to_last_token() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    positions = resolve_post_instruction_positions(
        mask,
        offsets=(-1, -2),
        padding_side="right",
    )
    assert positions.tolist() == [[2, 1], [3, 2]]


def test_left_padding_uses_the_physical_sequence_end() -> None:
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    positions = resolve_post_instruction_positions(
        mask,
        offsets=(-1, -2),
        padding_side="left",
    )
    assert positions.tolist() == [[3, 2], [3, 2]]
