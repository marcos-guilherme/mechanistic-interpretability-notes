import random

import pytest
import torch

from refusal_direction_replication.qwen_sparse_direction import (
    DirectionCandidate,
    FeatureActivationRow,
    FeatureAssociation,
    PairedFeatureRow,
    associate_features,
    elastic_net_objective,
    filter_direction,
    fit_direction_grid,
    random_feature_control,
    solve_elastic_net,
    top_feature_examples,
)


def candidate_with_indices(
    indices: list[int],
    coefficients: list[float],
) -> DirectionCandidate:
    columns = torch.eye(4)[:, indices]
    return DirectionCandidate.from_components(
        feature_indices=indices,
        coefficients=torch.tensor(coefficients),
        decoder_columns=columns,
        target_norm=4.0,
        kind="sparse_unfiltered",
    )


def association(feature_index: int, label: str) -> FeatureAssociation:
    return FeatureAssociation(
        feature_index=feature_index,
        label=label,
        offensive_activation_frequency=0.5,
        defensive_activation_frequency=0.5,
        standardized_paired_difference=0.0,
        ci_lower=-1.0,
        ci_upper=1.0,
        positive_fraction=0.5,
        coefficient_sign=1,
        decoder_cosine=0.0,
        basal_refusal_association=0.0,
    )


def test_sparse_solver_recovers_known_signed_components() -> None:
    decoder = torch.eye(4)
    direction = torch.tensor([2.0, 0.0, -1.0, 0.0])

    result = solve_elastic_net(
        decoder,
        direction,
        l1=0.0,
        l2=0.0,
        steps=500,
    )

    torch.testing.assert_close(
        result.reconstruction,
        direction,
        atol=1e-4,
        rtol=0,
    )
    assert result.coefficients[0] > 0
    assert result.coefficients[1] < 0


def test_sparse_solver_detaches_autograd_inputs_and_outputs() -> None:
    decoder = torch.eye(2, requires_grad=True)
    direction = torch.tensor([1.0, -1.0], requires_grad=True)

    result = solve_elastic_net(
        decoder,
        direction,
        l1=0.0,
        l2=0.0,
        steps=50,
    )

    assert decoder.grad is None
    assert direction.grad is None
    for tensor in (
        result.coefficients,
        result.decoder_columns,
        result.reconstruction,
        result.target,
    ):
        assert tensor is not None
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_elastic_net_rejects_nonfinite_inputs_and_objective() -> None:
    with pytest.raises(ValueError, match="finite"):
        elastic_net_objective(
            torch.tensor([[float("nan")]]),
            torch.ones(1),
            torch.ones(1),
            l1=0.0,
            l2=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        solve_elastic_net(
            torch.eye(2),
            torch.tensor([1.0, float("inf")]),
            l1=0.0,
            l2=0.0,
            steps=1,
        )


def test_fit_grid_breaks_exact_ties_deterministically() -> None:
    decoder = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    direction = torch.tensor([1.0, 0.0])

    first = fit_direction_grid(decoder, direction, cardinalities=[1])
    second = fit_direction_grid(decoder, direction, cardinalities=[1])

    assert first[0].feature_indices == second[0].feature_indices
    assert first[0].requested_cardinality == 1
    assert torch.isfinite(first[0].reconstruction).all()
    assert first[0].reconstruction.norm().item() == pytest.approx(1.0)


def test_filter_removes_only_defensive_components_and_rescales() -> None:
    candidate = candidate_with_indices(
        [0, 1, 2],
        coefficients=[1.0, 2.0, 3.0],
    )
    associations = (
        association(0, "offensive_associated"),
        association(1, "defensive_associated"),
        association(2, "unresolved"),
    )

    filtered = filter_direction(candidate, associations)

    assert filtered.feature_indices == (0, 2)
    assert filtered.removed_feature_indices == (1,)
    assert filtered.kind == "sparse_filtered"
    assert filtered.reconstruction.norm().item() == pytest.approx(4.0)
    assert filtered.original_norm == pytest.approx(10.0**0.5)


@pytest.mark.parametrize(
    "associations",
    (
        (association(0, "unresolved"),),
        (
            association(0, "unresolved"),
            association(1, "unresolved"),
            association(2, "unresolved"),
        ),
        (
            association(0, "unresolved"),
            association(0, "offensive_associated"),
        ),
        (
            association(0, "invalid_label"),
            association(1, "unresolved"),
        ),
    ),
)
def test_filter_requires_association_bijection_and_valid_labels(
    associations: tuple[FeatureAssociation, ...],
) -> None:
    candidate = candidate_with_indices([0, 1], [1.0, 2.0])

    with pytest.raises(ValueError, match="association"):
        filter_direction(candidate, associations)


def test_filter_rejects_removing_every_feature() -> None:
    candidate = candidate_with_indices([0, 1], [1.0, 2.0])

    with pytest.raises(ValueError, match="all features"):
        filter_direction(
            candidate,
            (
                association(0, "defensive_associated"),
                association(1, "defensive_associated"),
            ),
        )


def test_candidate_rejects_zero_or_nonpositive_norms() -> None:
    with pytest.raises(ValueError, match="target_norm"):
        DirectionCandidate.from_components(
            feature_indices=[0],
            coefficients=torch.ones(1),
            decoder_columns=torch.ones(2, 1),
            target_norm=0.0,
            kind="sparse_unfiltered",
        )
    with pytest.raises(ValueError, match="original norm"):
        DirectionCandidate.from_components(
            feature_indices=[0],
            coefficients=torch.zeros(1),
            decoder_columns=torch.ones(2, 1),
            target_norm=1.0,
            kind="sparse_unfiltered",
        )


def test_random_control_uses_exact_local_seed_semantics() -> None:
    candidate = candidate_with_indices(
        [0, 1],
        coefficients=[-2.0, 1.0],
    )
    decoder = torch.eye(4)
    expected_indices = tuple(
        sorted(random.Random(2030).sample([0, 1, 2, 3], 2))
    )
    expected_coefficients = [-2.0, 1.0]
    random.Random(2030).shuffle(expected_coefficients)
    scale = 4.0 / (5.0**0.5)
    expected_coefficients = [
        value * scale for value in expected_coefficients
    ]
    random.seed(77)
    state = random.getstate()

    control = random_feature_control(
        candidate,
        [0, 1, 2, 3],
        decoder,
        seed=2030,
    )

    assert control.feature_indices == expected_indices
    assert control.coefficients.tolist() == pytest.approx(
        expected_coefficients
    )
    assert control.reconstruction.norm().item() == pytest.approx(4.0)
    assert random.getstate() == state


def paired_feature_rows() -> tuple[PairedFeatureRow, ...]:
    return (
        PairedFeatureRow(
            "p1", "offensive", 1, (99.0, 4.0), 1, 0.8, 0.3
        ),
        PairedFeatureRow(
            "p1", "defensive", 1, (99.0, 1.0), 1, 0.8, 0.3
        ),
        PairedFeatureRow(
            "p2", "offensive", 1, (99.0, 5.0), 1, 0.8, 0.3
        ),
        PairedFeatureRow(
            "p2", "defensive", 1, (99.0, 2.0), 1, 0.8, 0.3
        ),
        PairedFeatureRow(
            "p1", "offensive", 2, (99.0, 0.0), -1, -0.6, -0.2
        ),
        PairedFeatureRow(
            "p1", "defensive", 2, (99.0, 2.0), -1, -0.6, -0.2
        ),
        PairedFeatureRow(
            "p2", "offensive", 2, (99.0, 1.0), -1, -0.6, -0.2
        ),
        PairedFeatureRow(
            "p2", "defensive", 2, (99.0, 3.0), -1, -0.6, -0.2
        ),
    )


def test_association_uses_last_token_paired_differences_and_seed() -> None:
    first = associate_features(paired_feature_rows(), [1, 2], seed=2029)
    second = associate_features(paired_feature_rows(), [1, 2], seed=2029)

    assert first == second
    assert [item.label for item in first] == [
        "offensive_associated",
        "defensive_associated",
    ]
    assert first[0].positive_fraction == 1.0
    assert first[0].offensive_activation_frequency == 1.0
    assert first[1].offensive_activation_frequency == 0.5
    assert all(
        all(
            torch.isfinite(torch.tensor(value))
            for value in (
                item.standardized_paired_difference,
                item.ci_lower,
                item.ci_upper,
            )
        )
        for item in first
    )


def test_association_sign_fraction_uses_bootstrap_draws() -> None:
    rows = (
        PairedFeatureRow("p1", "offensive", 7, (0.0,), 1, 0.1, 0.2),
        PairedFeatureRow("p1", "defensive", 7, (1.0,), 1, 0.1, 0.2),
        PairedFeatureRow("p2", "offensive", 7, (1.0,), 1, 0.1, 0.2),
        PairedFeatureRow("p2", "defensive", 7, (0.0,), 1, 0.1, 0.2),
        PairedFeatureRow("p3", "offensive", 7, (1.0,), 1, 0.1, 0.2),
        PairedFeatureRow("p3", "defensive", 7, (0.0,), 1, 0.1, 0.2),
    )
    rng = random.Random(2029)
    draws = [
        sum(rng.choices((-1.0, 1.0, 1.0), k=3)) / 3
        for _ in range(10_000)
    ]
    expected = sum(draw > 0 for draw in draws) / len(draws)

    result = associate_features(rows, [7], seed=2029)[0]

    assert result.positive_fraction == expected
    assert result.positive_fraction != pytest.approx(2 / 3)


def test_top_examples_return_ids_only_with_deterministic_ties() -> None:
    rows = (
        FeatureActivationRow("z", "discovery", 1, 3.0),
        FeatureActivationRow("a", "discovery", 1, 3.0),
        FeatureActivationRow("c", "unlabeled_corpus", 1, 2.0),
    )

    result = top_feature_examples(rows, [1], k=2)

    assert result == {1: ("a", "z")}
    assert all(isinstance(value, str) for value in result[1])


def test_top_examples_reject_labeled_or_nondiscovery_sources() -> None:
    rows = (FeatureActivationRow("private-id", "selection", 1, 3.0),)

    with pytest.raises(ValueError, match="source"):
        top_feature_examples(rows, [1])
