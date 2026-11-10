import pytest
import torch

from ioi_replication.metrics import logits_to_logit_diff, normalized_recovery


def test_logit_diff_returns_per_prompt_and_mean() -> None:
    logits = torch.zeros(2, 3, 5)
    answers = torch.tensor([[1, 2], [3, 4]])
    logits[0, -1, 1], logits[0, -1, 2] = 5.0, 2.0
    logits[1, -1, 3], logits[1, -1, 4] = 1.0, 2.0

    per_prompt = logits_to_logit_diff(logits, answers, per_prompt=True)

    assert torch.equal(per_prompt, torch.tensor([3.0, -1.0]))
    assert logits_to_logit_diff(logits, answers).item() == 1.0


def test_normalized_recovery_maps_corrupted_and_clean_endpoints() -> None:
    clean = torch.tensor(4.0)
    corrupted = torch.tensor(-2.0)

    assert normalized_recovery(corrupted, clean, corrupted).item() == 0.0
    assert normalized_recovery(clean, clean, corrupted).item() == 1.0
    assert normalized_recovery(torch.tensor(1.0), clean, corrupted).item() == 0.5


def test_normalized_recovery_rejects_equal_baselines() -> None:
    baseline = torch.tensor(1.0)

    with pytest.raises(
        ValueError,
        match="clean and corrupted logit differences must differ",
    ):
        normalized_recovery(baseline, baseline, baseline)
