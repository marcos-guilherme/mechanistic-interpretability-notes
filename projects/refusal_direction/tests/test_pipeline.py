import torch

from refusal_direction_replication.data import PromptExample
from refusal_direction_replication.directions import CandidateDirection
from refusal_direction_replication.metrics import CandidateScore
from refusal_direction_replication.pipeline import (
    dtype_from_name,
    filter_examples_by_score,
    load_candidate_artifact,
    random_direction_control,
    sample_examples,
    save_candidate_artifact,
    select_candidate,
    summarize_generated_condition,
    temporary_hooks,
)


def test_select_candidate_filters_then_minimizes_ablation_score() -> None:
    candidates = [
        CandidateDirection(0, -1, torch.tensor([1.0, 0.0])),
        CandidateDirection(1, -1, torch.tensor([0.0, 1.0])),
    ]
    scores = [
        CandidateScore(
            ablation_refusal_score=-4.0,
            addition_refusal_score=1.0,
            kl_divergence=0.2,
            layer=0,
            n_layers=10,
        ),
        CandidateScore(
            ablation_refusal_score=-2.0,
            addition_refusal_score=1.0,
            kl_divergence=0.05,
            layer=1,
            n_layers=10,
        ),
    ]
    assert select_candidate(candidates, scores).layer == 1


def test_random_control_matches_norm_and_is_orthogonal() -> None:
    direction = torch.tensor([3.0, 4.0, 0.0])
    control = random_direction_control(direction, seed=7)
    assert torch.allclose(control.norm(), direction.norm())
    assert abs(torch.dot(control, direction).item()) < 1e-5


def test_temporary_hooks_are_removed_after_context() -> None:
    module = torch.nn.Identity()

    def add_one(_module, _inputs, output):
        return output + 1

    baseline = module(torch.zeros(1))
    with temporary_hooks(forward_hooks=[(module, add_one)]):
        changed = module(torch.zeros(1))
    restored = module(torch.zeros(1))
    assert torch.equal(baseline, restored)
    assert torch.equal(changed, torch.ones(1))


def test_filter_examples_keeps_expected_refusal_sign() -> None:
    examples = [
        PromptExample("a", "first", "harmful", "fixture"),
        PromptExample("b", "second", "harmful", "fixture"),
    ]
    assert filter_examples_by_score(
        examples,
        torch.tensor([1.0, -1.0]),
        keep_positive=True,
    ) == [examples[0]]


def test_generated_condition_summary_keeps_per_example_flags() -> None:
    summary = summarize_generated_condition(
        ["I cannot help.", "Here is a safe answer."]
    )
    assert summary["refusal_rate"] == 0.5
    assert summary["refusal_flags"] == [True, False]


def test_dtype_name_is_explicit() -> None:
    assert dtype_from_name("bfloat16") is torch.bfloat16


def test_sampling_is_deterministic_without_mutating_input() -> None:
    examples = [
        PromptExample(str(index), str(index), "harmless", "fixture")
        for index in range(5)
    ]
    original = list(examples)
    assert sample_examples(examples, size=3, seed=11) == sample_examples(
        examples,
        size=3,
        seed=11,
    )
    assert examples == original


def test_candidate_artifact_round_trip(tmp_path) -> None:
    candidates = [
        CandidateDirection(0, -2, torch.tensor([1.0, 0.0])),
        CandidateDirection(1, -1, torch.tensor([0.0, 1.0])),
    ]
    path = tmp_path / "candidates.pt"
    save_candidate_artifact(path, candidates)
    loaded = load_candidate_artifact(path)
    assert [(item.layer, item.position_offset) for item in loaded] == [
        (0, -2),
        (1, -1),
    ]
    assert torch.equal(loaded[1].vector, candidates[1].vector)
