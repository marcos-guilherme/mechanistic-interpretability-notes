import math

import pytest
import torch

from refusal_direction_replication.interventions import (
    ablate_direction,
    add_direction,
    directional_ablation_hook_specs,
    residual_intervention_hook,
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


def test_ablation_default_coefficient_matches_full_removal() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    direction = torch.tensor([1.0, 0.0])
    assert torch.equal(
        ablate_direction(hidden, direction),
        ablate_direction(hidden, direction, coefficient=1.0),
    )
    assert torch.equal(
        ablate_direction(hidden, direction, coefficient=1.0),
        torch.tensor([[[0.0, 4.0], [0.0, 2.0]]]),
    )


def test_zero_ablation_coefficient_is_a_noop() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    result = ablate_direction(hidden, torch.tensor([1.0, 0.0]), coefficient=0.0)
    assert torch.equal(result, hidden)


def test_half_ablation_removes_half_the_projection() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    result = ablate_direction(hidden, torch.tensor([1.0, 0.0]), coefficient=0.5)
    assert torch.equal(result, torch.tensor([[[1.5, 4.0], [0.5, 2.0]]]))


def test_ablate_rejects_coefficient_outside_unit_interval() -> None:
    hidden = torch.tensor([[[3.0, 4.0]]])
    direction = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="ablation coefficient"):
        ablate_direction(hidden, direction, coefficient=1.1)
    with pytest.raises(ValueError, match="ablation coefficient"):
        ablate_direction(hidden, direction, coefficient=-0.01)


def test_ablate_rejects_nonfinite_coefficient() -> None:
    hidden = torch.tensor([[[3.0, 4.0]]])
    direction = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        ablate_direction(hidden, direction, coefficient=math.nan)
    with pytest.raises(ValueError, match="finite"):
        ablate_direction(hidden, direction, coefficient=math.inf)


def test_add_rejects_nonfinite_coefficient_but_keeps_unbounded_math() -> None:
    hidden = torch.zeros(1, 1, 2)
    direction = torch.tensor([1.0, -1.0])
    result = add_direction(hidden, direction, coefficient=2.0)
    assert torch.equal(result[0, 0], torch.tensor([2.0, -2.0]))
    with pytest.raises(ValueError, match="finite"):
        add_direction(hidden, direction, coefficient=math.nan)
    with pytest.raises(ValueError, match="finite"):
        add_direction(hidden, direction, coefficient=-math.inf)


def test_pre_hook_threads_ablation_coefficient() -> None:
    hook = residual_intervention_pre_hook(
        torch.tensor([1.0, 0.0]),
        mode="ablate",
        coefficient=0.5,
    )
    changed, = hook(None, (torch.tensor([[[3.0, 4.0]]]),))
    assert torch.equal(changed, torch.tensor([[[1.5, 4.0]]]))


def test_forward_hook_threads_ablation_coefficient() -> None:
    hook = residual_intervention_hook(
        torch.tensor([1.0, 0.0]),
        mode="ablate",
        coefficient=0.5,
    )
    changed = hook(None, None, torch.tensor([[[3.0, 4.0]]]))
    assert torch.equal(changed, torch.tensor([[[1.5, 4.0]]]))


def test_ablation_specs_thread_fractional_coefficient() -> None:
    pre_hooks, forward_hooks = directional_ablation_hook_specs(
        [object()],
        [object()],
        [object()],
        torch.tensor([1.0, 0.0]),
        coefficient=0.5,
    )
    hidden = torch.tensor([[[3.0, 4.0]]])
    pre_changed, = pre_hooks[0][1](None, (hidden,))
    forward_changed = forward_hooks[0][1](None, None, hidden)
    assert torch.equal(pre_changed, torch.tensor([[[1.5, 4.0]]]))
    assert torch.equal(forward_changed, torch.tensor([[[1.5, 4.0]]]))


def test_ablation_specs_default_to_full_coefficient() -> None:
    pre_hooks, forward_hooks = directional_ablation_hook_specs(
        [object()],
        [object()],
        [object()],
        torch.tensor([1.0, 0.0]),
    )
    hidden = torch.tensor([[[3.0, 4.0]]])
    pre_changed, = pre_hooks[0][1](None, (hidden,))
    forward_changed = forward_hooks[0][1](None, None, hidden)
    assert torch.equal(pre_changed, torch.tensor([[[0.0, 4.0]]]))
    assert torch.equal(forward_changed, torch.tensor([[[0.0, 4.0]]]))
