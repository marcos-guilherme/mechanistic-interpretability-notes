import torch

from refusal_direction_replication.metrics import (
    contains_refusal,
    false_refusal_rate,
    kl_divergence,
    refusal_log_odds,
    refusal_rate,
)


def test_refusal_matching_is_case_insensitive() -> None:
    assert contains_refusal("I CANNOT assist with that.")
    assert not contains_refusal("Here is a safe historical overview.")


def test_rates_use_the_expected_population() -> None:
    outputs = ["I cannot help.", "Here is the answer."]
    assert refusal_rate(outputs) == 0.5
    assert false_refusal_rate(outputs) == 0.5


def test_refusal_log_odds_uses_first_token_logits() -> None:
    logits = torch.tensor([[2.0, 0.0]])
    score = refusal_log_odds(logits, refusal_token_ids=(0,))
    assert torch.allclose(score, torch.tensor([2.0], dtype=torch.float64))


def test_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[2.0, 0.0]])
    assert torch.allclose(kl_divergence(logits, logits), torch.zeros(1))
