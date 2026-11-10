import torch

from refusal_direction_replication.directions import (
    CandidateDirection,
    difference_in_means,
    load_direction,
    save_direction,
)


def test_difference_in_means_uses_harmful_minus_harmless() -> None:
    harmful = torch.tensor([[3.0, 5.0], [5.0, 7.0]])
    harmless = torch.tensor([[1.0, 1.0], [1.0, 3.0]])
    assert torch.equal(
        difference_in_means(harmful, harmless),
        torch.tensor([3.0, 4.0]),
    )


def test_direction_round_trip(tmp_path) -> None:
    candidate = CandidateDirection(2, -1, torch.tensor([3.0, 4.0]))
    path = tmp_path / "direction.pt"
    save_direction(path, candidate)
    loaded = load_direction(path)
    assert loaded.layer == 2
    assert loaded.position_offset == -1
    assert torch.allclose(loaded.unit, torch.tensor([0.6, 0.8]))
