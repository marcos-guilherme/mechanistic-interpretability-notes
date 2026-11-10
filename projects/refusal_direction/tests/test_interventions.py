import torch

from refusal_direction_replication.interventions import (
    ablate_direction,
    add_direction,
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
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


def test_ablation_with_zero_direction_is_a_noop() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    result = ablate_direction(hidden, torch.zeros(2))
    assert torch.equal(result, hidden)


def test_pre_hook_preserves_auxiliary_inputs() -> None:
    hook = residual_intervention_pre_hook(
        torch.tensor([1.0, 0.0]),
        mode="ablate",
    )
    changed, mask = hook(
        None,
        (torch.tensor([[[3.0, 4.0]]]), torch.tensor([[1]])),
    )
    assert torch.equal(changed, torch.tensor([[[0.0, 4.0]]]))
    assert torch.equal(mask, torch.tensor([[1]]))


def test_ablation_specs_cover_residual_attention_and_mlp_paths() -> None:
    blocks = [object(), object()]
    attentions = [object(), object()]
    mlps = [object(), object()]
    pre_hooks, forward_hooks = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        torch.tensor([1.0, 0.0]),
    )
    assert [module for module, _hook in pre_hooks] == blocks
    assert [module for module, _hook in forward_hooks] == attentions + mlps
