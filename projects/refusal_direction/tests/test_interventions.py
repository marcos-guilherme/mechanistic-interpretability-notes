import torch

from refusal_direction_replication.interventions import (
    ablate_direction,
    add_direction,
)


def test_add_direction_broadcasts_over_batch_and_position() -> None:
    hidden = torch.zeros(2, 3, 2)
    result = add_direction(hidden, torch.tensor([1.0, -1.0]), coefficient=2.0)
    assert torch.equal(result[0, 0], torch.tensor([2.0, -2.0]))


def test_ablation_removes_projection() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    direction = torch.tensor([1.0, 0.0])
    result = ablate_direction(hidden, direction)
    assert torch.equal(result, torch.tensor([[[0.0, 4.0], [0.0, 2.0]]]))
