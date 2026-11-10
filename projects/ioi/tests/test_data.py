import pytest
import torch

from ioi_replication.data import build_ioi_examples, tokenize_answers


def test_examples_are_paired_with_swapped_answers() -> None:
    prompts, answers = build_ioi_examples()

    assert len(prompts) == len(answers) == 8
    assert answers[0] == (" Mary", " John")
    assert answers[1] == (" John", " Mary")
    assert "John gave" in prompts[0]
    assert "Mary gave" in prompts[1]


def test_tokenize_answers_returns_two_ids_per_prompt() -> None:
    class FakeModel:
        def to_tokens(self, names, prepend_bos):
            assert prepend_bos is False
            return torch.tensor([[10], [20]])

    tokens = tokenize_answers(
        FakeModel(),
        [(" Mary", " John"), (" John", " Mary")],
        torch.device("cpu"),
    )

    assert torch.equal(tokens, torch.tensor([[10, 20], [10, 20]]))


def test_tokenize_answers_rejects_multi_token_names() -> None:
    class FakeModel:
        def to_tokens(self, names, prepend_bos):
            return torch.tensor([[10, 11], [20, 21]])

    with pytest.raises(ValueError, match="expected one token per answer"):
        tokenize_answers(
            FakeModel(),
            [(" Mary Jane", " John Doe")],
            torch.device("cpu"),
        )
