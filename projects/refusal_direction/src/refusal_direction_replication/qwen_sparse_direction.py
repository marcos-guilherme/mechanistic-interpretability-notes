"""Signed sparse refusal-direction decomposition and feature controls."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
import statistics

import torch
from torch.nn import functional as F


_ASSOCIATION_RESAMPLES = 10_000
_ALLOWED_EXAMPLE_SOURCES = frozenset({"discovery", "unlabeled_corpus"})
_ASSOCIATION_LABELS = frozenset(
    {
        "offensive_associated",
        "defensive_associated",
        "unresolved",
    }
)


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _bootstrap_bounds(draws: list[float]) -> tuple[float, float]:
    draws.sort()
    count = len(draws)
    lower_index = max(0, int(0.025 * count) - 1)
    upper_index = min(count - 1, int(0.975 * count) - 1)
    return draws[lower_index], draws[upper_index]


@dataclass(frozen=True)
class DirectionCandidate:
    """Compact sparse coefficients plus only their selected decoder columns."""

    kind: str
    feature_indices: tuple[int, ...]
    coefficients: torch.Tensor
    decoder_columns: torch.Tensor
    reconstruction: torch.Tensor
    target_norm: float
    original_norm: float
    cosine: float
    relative_l2: float
    requested_cardinality: int | None = None
    removed_feature_indices: tuple[int, ...] = ()
    target: torch.Tensor | None = None
    l1: float | None = None
    l2: float | None = None

    @property
    def cardinality(self) -> int:
        return len(self.feature_indices)

    @classmethod
    def from_components(
        cls,
        *,
        feature_indices: Sequence[int],
        coefficients: torch.Tensor,
        decoder_columns: torch.Tensor,
        target_norm: float,
        kind: str,
        target: torch.Tensor | None = None,
        requested_cardinality: int | None = None,
        removed_feature_indices: Sequence[int] = (),
        l1: float | None = None,
        l2: float | None = None,
        rescale: bool = False,
    ) -> DirectionCandidate:
        indices = tuple(int(index) for index in feature_indices)
        compact_coefficients = coefficients.detach().float().reshape(-1)
        columns = decoder_columns.detach().float()
        if not indices:
            raise ValueError("feature_indices must be nonempty")
        if len(indices) != len(set(indices)):
            raise ValueError("feature_indices must be unique")
        if compact_coefficients.shape != (len(indices),):
            raise ValueError("coefficients must match feature_indices")
        if columns.ndim != 2 or columns.shape[1] != len(indices):
            raise ValueError("decoder_columns must be [hidden, features]")
        if not math.isfinite(target_norm) or target_norm <= 0:
            raise ValueError("target_norm must be finite and strictly positive")
        _finite_tensor(compact_coefficients, "coefficients")
        _finite_tensor(columns, "decoder_columns")
        reconstruction = columns @ compact_coefficients
        original_norm = float(torch.linalg.vector_norm(reconstruction).item())
        if not math.isfinite(original_norm) or original_norm <= 0:
            raise ValueError("original norm must be finite and strictly positive")
        if rescale:
            scale = target_norm / original_norm
            compact_coefficients = compact_coefficients * scale
            reconstruction = reconstruction * scale
        target_value: torch.Tensor | None = None
        if target is not None:
            target_value = target.detach().float().reshape(-1)
            if target_value.shape != reconstruction.shape:
                raise ValueError("target must match reconstruction")
            _finite_tensor(target_value, "target")
            actual_target_norm = float(target_value.norm().item())
            if actual_target_norm <= 0:
                raise ValueError(
                    "target norm must be finite and strictly positive"
                )
            if not math.isclose(
                actual_target_norm,
                target_norm,
                rel_tol=1e-5,
                abs_tol=1e-7,
            ):
                raise ValueError("target_norm does not match target")
            cosine = (
                float(
                    F.cosine_similarity(
                        reconstruction.unsqueeze(0),
                        target_value.unsqueeze(0),
                    ).item()
                )
                if reconstruction.norm().item() > 0 else 0.0
            )
            denominator = actual_target_norm
            relative_l2 = float(
                torch.linalg.vector_norm(reconstruction - target_value).item()
                / denominator
            )
        else:
            cosine = 0.0
            relative_l2 = 0.0
        if not all(
            math.isfinite(value)
            for value in (original_norm, cosine, relative_l2)
        ):
            raise ValueError("candidate metrics must be finite")
        return cls(
            kind=kind,
            feature_indices=indices,
            coefficients=compact_coefficients.detach(),
            decoder_columns=columns.detach(),
            reconstruction=reconstruction.detach(),
            target_norm=float(target_norm),
            original_norm=original_norm,
            cosine=cosine,
            relative_l2=relative_l2,
            requested_cardinality=requested_cardinality,
            removed_feature_indices=tuple(
                int(index) for index in removed_feature_indices
            ),
            target=target_value.detach() if target_value is not None else None,
            l1=l1,
            l2=l2,
        )

    @classmethod
    def from_tensors(
        cls,
        coefficients: torch.Tensor,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        *,
        decoder: torch.Tensor | None = None,
        l1: float | None = None,
        l2: float | None = None,
    ) -> DirectionCandidate:
        full_coefficients = coefficients.detach().float().reshape(-1)
        selected = torch.nonzero(
            full_coefficients != 0,
            as_tuple=False,
        ).flatten()
        if decoder is None:
            if selected.numel() != full_coefficients.numel():
                raise ValueError(
                    "decoder is required for sparse full coefficients"
                )
            columns = torch.eye(
                reconstruction.numel(),
                dtype=torch.float32,
                device=reconstruction.device,
            )
        else:
            columns = decoder.detach()[:, selected]
        candidate = cls.from_components(
            feature_indices=selected.tolist(),
            coefficients=full_coefficients[selected],
            decoder_columns=columns,
            target_norm=float(
                target.detach().float().reshape(-1).norm().item()
            ),
            target=target.detach(),
            kind="sparse_unfiltered",
            l1=l1,
            l2=l2,
        )
        expected = reconstruction.detach().float().reshape(-1)
        if not torch.allclose(
            candidate.reconstruction,
            expected,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("reconstruction does not match components")
        return candidate

    def restrict_to(
        self,
        feature_indices: Sequence[int],
        *,
        kind: str,
    ) -> DirectionCandidate:
        keep = tuple(int(index) for index in feature_indices)
        if not keep:
            raise ValueError("restricted feature set must be nonempty")
        if len(keep) != len(set(keep)):
            raise ValueError("restricted feature set must be unique")
        positions = {index: position for position, index in enumerate(self.feature_indices)}
        if any(index not in positions for index in keep):
            raise ValueError("restricted feature not present in candidate")
        compact_positions = torch.tensor(
            [positions[index] for index in keep],
            dtype=torch.long,
            device=self.coefficients.device,
        )
        columns = self.decoder_columns.index_select(1, compact_positions)
        coefficients = self.coefficients.index_select(0, compact_positions)
        removed = tuple(
            index for index in self.feature_indices if index not in set(keep)
        )
        return DirectionCandidate.from_components(
            feature_indices=keep,
            coefficients=coefficients,
            decoder_columns=columns,
            target_norm=self.target_norm,
            target=self.target,
            kind=kind,
            requested_cardinality=self.requested_cardinality,
            removed_feature_indices=removed,
            l1=self.l1,
            l2=self.l2,
            rescale=True,
        )


@dataclass(frozen=True)
class FeatureAssociation:
    feature_index: int
    label: str
    offensive_activation_frequency: float
    defensive_activation_frequency: float
    standardized_paired_difference: float
    ci_lower: float
    ci_upper: float
    positive_fraction: float
    coefficient_sign: int
    decoder_cosine: float
    basal_refusal_association: float


@dataclass(frozen=True)
class PairedFeatureRow:
    pair_id: str
    side: str
    feature_index: int
    token_activations: tuple[float, ...]
    coefficient_sign: int
    decoder_cosine: float
    basal_refusal_association: float


@dataclass(frozen=True)
class FeatureActivationRow:
    """Identity-only example row; raw text is deliberately not represented."""

    example_id: str
    source: str
    feature_index: int
    activation: float


def elastic_net_objective(
    matrix: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    l1: float,
    l2: float,
) -> float:
    """Return a validated finite Elastic Net objective."""
    if l1 < 0 or l2 < 0 or not math.isfinite(l1) or not math.isfinite(l2):
        raise ValueError("regularization strengths must be finite and nonnegative")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    vector = target.reshape(-1)
    compact = coefficients.reshape(-1)
    if matrix.shape != (vector.numel(), compact.numel()):
        raise ValueError("Elastic Net shapes do not align")
    _finite_tensor(matrix, "matrix")
    _finite_tensor(vector, "target")
    _finite_tensor(compact, "coefficients")
    return _elastic_net_objective_value(
        matrix,
        vector,
        compact,
        l1,
        l2,
    )


def _elastic_net_objective_value(
    matrix: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    l1: float,
    l2: float,
) -> float:
    residual = matrix @ coefficients - target
    objective = (
        residual.square().sum()
        + l1 * coefficients.abs().sum()
        + l2 * coefficients.square().sum()
    )
    if not torch.isfinite(objective):
        raise ValueError("Elastic Net objective must be finite")
    return float(objective.item())


@torch.no_grad()
def solve_elastic_net(
    decoder: torch.Tensor,
    direction: torch.Tensor,
    *,
    l1: float,
    l2: float,
    steps: int = 2_000,
) -> DirectionCandidate:
    """Fit signed coefficients with deterministic proximal gradient descent."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    matrix = decoder.detach().float()
    target = direction.detach().float().reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != target.numel()
        or matrix.shape[1] == 0
    ):
        raise ValueError("decoder and direction shapes do not align")
    _finite_tensor(matrix, "decoder")
    _finite_tensor(target, "direction")
    if float(target.norm().item()) <= 0:
        raise ValueError("direction norm must be strictly positive")
    coefficients = torch.zeros(
        matrix.shape[1],
        dtype=torch.float32,
        device=matrix.device,
    )
    learning_rate = 1e-2
    previous = elastic_net_objective(
        matrix,
        target,
        coefficients,
        l1,
        l2,
    )
    for _ in range(steps):
        residual = matrix @ coefficients - target
        gradient = 2.0 * (matrix.T @ residual) + 2.0 * l2 * coefficients
        accepted = False
        converged = False
        for _line_search in range(24):
            proposal = coefficients - learning_rate * gradient
            proposal = torch.sign(proposal) * torch.relu(
                proposal.abs() - learning_rate * l1
            )
            objective = _elastic_net_objective_value(
                matrix,
                target,
                proposal,
                l1,
                l2,
            )
            tolerance = 1e-12 * max(1.0, abs(previous))
            if objective <= previous + tolerance:
                converged = bool(
                    torch.max(torch.abs(proposal - coefficients)).item()
                    <= 1e-7
                )
                coefficients = proposal
                previous = objective
                learning_rate *= 1.02
                accepted = True
                break
            learning_rate *= 0.5
        if not accepted:
            raise RuntimeError("Elastic Net line search did not descend")
        if converged:
            break
    reconstruction = matrix @ coefficients
    return DirectionCandidate.from_tensors(
        coefficients,
        reconstruction,
        target,
        decoder=matrix,
        l1=l1,
        l2=l2,
    )


@torch.no_grad()
def fit_direction_grid(
    decoder: torch.Tensor,
    direction: torch.Tensor,
    cardinalities: Sequence[int],
) -> tuple[DirectionCandidate, ...]:
    """Fit the frozen regularization grid and select deterministic candidates."""
    matrix = decoder.detach().float()
    target = direction.detach().float().reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != target.numel()
        or matrix.shape[1] == 0
    ):
        raise ValueError("decoder and direction shapes do not align")
    requested = tuple(int(value) for value in cardinalities)
    if not requested or any(
        value <= 0 or value > matrix.shape[1] for value in requested
    ):
        raise ValueError("cardinalities must be within decoder width")
    _finite_tensor(matrix, "decoder")
    _finite_tensor(target, "direction")
    if float(target.norm().item()) <= 0:
        raise ValueError("direction norm must be strictly positive")
    l1_max = float(2.0 * (matrix.T @ target).abs().max().item())
    candidates = [
        solve_elastic_net(
            matrix,
            target,
            l1=fraction * l1_max,
            l2=1e-4,
        )
        for fraction in (0.50, 0.25, 0.10, 0.05, 0.01)
    ]
    selected: list[DirectionCandidate] = []
    for cardinality in requested:
        best = min(
            enumerate(candidates),
            key=lambda item: (
                abs(item[1].cardinality - cardinality),
                -item[1].cosine,
                item[1].relative_l2,
                item[0],
            ),
        )[1]
        selected.append(
            DirectionCandidate.from_components(
                feature_indices=best.feature_indices,
                coefficients=best.coefficients,
                decoder_columns=best.decoder_columns,
                target_norm=float(target.norm().item()),
                target=target,
                kind="sparse_unfiltered",
                requested_cardinality=cardinality,
                l1=best.l1,
                l2=best.l2,
                rescale=True,
            )
        )
    return tuple(selected)


def association_label(
    ci_lower: float,
    ci_upper: float,
    positive_fraction: float,
) -> str:
    if ci_lower > 0 and positive_fraction >= 0.70:
        return "offensive_associated"
    if ci_upper < 0 and positive_fraction <= 0.30:
        return "defensive_associated"
    return "unresolved"


def _constant_metadata(
    rows: Sequence[PairedFeatureRow],
) -> tuple[int, float, float]:
    values = {
        (
            int(row.coefficient_sign),
            float(row.decoder_cosine),
            float(row.basal_refusal_association),
        )
        for row in rows
    }
    if len(values) != 1:
        raise ValueError("feature metadata must be constant across pair rows")
    coefficient_sign, decoder_cosine, basal = values.pop()
    if coefficient_sign not in {-1, 0, 1}:
        raise ValueError("coefficient_sign must be -1, 0, or 1")
    if not math.isfinite(decoder_cosine) or not math.isfinite(basal):
        raise ValueError("feature metadata must be finite")
    return coefficient_sign, decoder_cosine, basal


def associate_features(
    rows: Sequence[PairedFeatureRow],
    feature_indices: Sequence[int],
    seed: int = 2029,
) -> tuple[FeatureAssociation, ...]:
    """Associate last-template-token activations within discovery pairs."""
    output: list[FeatureAssociation] = []
    for feature_index in feature_indices:
        feature_rows = [
            row for row in rows if row.feature_index == int(feature_index)
        ]
        if not feature_rows:
            raise ValueError(f"feature {feature_index} has no rows")
        metadata = _constant_metadata(feature_rows)
        paired: dict[str, dict[str, float]] = defaultdict(dict)
        for row in feature_rows:
            if not row.pair_id:
                raise ValueError("pair_id must be nonempty")
            if row.side not in {"offensive", "defensive"}:
                raise ValueError("side must be offensive or defensive")
            if not row.token_activations:
                raise ValueError("token_activations must be nonempty")
            activation = float(row.token_activations[-1])
            if not math.isfinite(activation):
                raise ValueError("activation must be finite")
            if row.side in paired[row.pair_id]:
                raise ValueError("duplicate feature side within pair")
            paired[row.pair_id][row.side] = activation
        pair_ids = tuple(sorted(paired))
        if any(
            set(paired[pair_id]) != {"offensive", "defensive"}
            for pair_id in pair_ids
        ):
            raise ValueError("each pair must contain both sides")
        offensive = [paired[pair_id]["offensive"] for pair_id in pair_ids]
        defensive = [paired[pair_id]["defensive"] for pair_id in pair_ids]
        differences = [
            offensive_value - defensive_value
            for offensive_value, defensive_value in zip(
                offensive,
                defensive,
                strict=True,
            )
        ]
        mean_difference = statistics.fmean(differences)
        standard_deviation = (
            statistics.stdev(differences) if len(differences) > 1 else 0.0
        )
        standardized = (
            mean_difference / standard_deviation
            if standard_deviation > 1e-12
            else mean_difference / 1e-12
        )
        rng = random.Random(seed)
        draws = [
            statistics.fmean(
                rng.choices(differences, k=len(differences))
            )
            for _ in range(_ASSOCIATION_RESAMPLES)
        ]
        ci_lower, ci_upper = _bootstrap_bounds(draws)
        positive_fraction = sum(value > 0 for value in draws) / len(draws)
        output.append(
            FeatureAssociation(
                feature_index=int(feature_index),
                label=association_label(
                    ci_lower,
                    ci_upper,
                    positive_fraction,
                ),
                offensive_activation_frequency=sum(
                    value > 0 for value in offensive
                )
                / len(offensive),
                defensive_activation_frequency=sum(
                    value > 0 for value in defensive
                )
                / len(defensive),
                standardized_paired_difference=float(standardized),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                positive_fraction=float(positive_fraction),
                coefficient_sign=metadata[0],
                decoder_cosine=metadata[1],
                basal_refusal_association=metadata[2],
            )
        )
    return tuple(output)


def top_feature_examples(
    rows: Sequence[FeatureActivationRow],
    feature_indices: Sequence[int],
    k: int = 20,
) -> dict[int, tuple[str, ...]]:
    """Return private IDs only, never example text."""
    if k <= 0:
        raise ValueError("k must be positive")
    for row in rows:
        if row.source not in _ALLOWED_EXAMPLE_SOURCES:
            raise ValueError(f"example source is not permitted: {row.source}")
        if not row.example_id:
            raise ValueError("example_id must be nonempty")
        if not math.isfinite(row.activation):
            raise ValueError("activation must be finite")
    return {
        int(feature_index): tuple(
            row.example_id
            for row in sorted(
                (
                    row
                    for row in rows
                    if row.feature_index == int(feature_index)
                ),
                key=lambda row: (-row.activation, row.example_id),
            )[:k]
        )
        for feature_index in feature_indices
    }


def filter_direction(
    candidate: DirectionCandidate,
    associations: Sequence[FeatureAssociation],
) -> DirectionCandidate:
    if (
        not math.isfinite(candidate.target_norm)
        or candidate.target_norm <= 0
        or not math.isfinite(candidate.original_norm)
        or candidate.original_norm <= 0
    ):
        raise ValueError("candidate norms must be finite and strictly positive")
    reconstruction_norm = float(candidate.reconstruction.norm().item())
    if not math.isfinite(reconstruction_norm) or reconstruction_norm <= 0:
        raise ValueError(
            "candidate reconstruction norm must be strictly positive"
        )
    association_indices = tuple(
        item.feature_index for item in associations
    )
    if (
        len(association_indices) != len(set(association_indices))
        or set(association_indices) != set(candidate.feature_indices)
    ):
        raise ValueError(
            "association features must bijectively match candidate features"
        )
    if any(item.label not in _ASSOCIATION_LABELS for item in associations):
        raise ValueError("association label is invalid")
    labels = {item.feature_index: item.label for item in associations}
    keep = tuple(
        index
        for index in candidate.feature_indices
        if labels.get(index) != "defensive_associated"
    )
    if not keep:
        raise ValueError("filtering removed all features")
    return candidate.restrict_to(keep, kind="sparse_filtered")


def permute_signed_coefficients(
    coefficients: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    values = coefficients.detach().cpu().tolist()
    random.Random(seed).shuffle(values)
    return torch.tensor(
        values,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )


def random_feature_control(
    candidate: DirectionCandidate,
    eligible_indices: Sequence[int],
    decoder: torch.Tensor,
    *,
    seed: int = 2030,
) -> DirectionCandidate:
    eligible = [int(index) for index in eligible_indices]
    if len(eligible) != len(set(eligible)):
        raise ValueError("eligible_indices must be unique")
    if len(eligible) < candidate.cardinality:
        raise ValueError("not enough eligible features")
    matrix = decoder.float()
    if matrix.ndim != 2 or any(
        index < 0 or index >= matrix.shape[1] for index in eligible
    ):
        raise ValueError("eligible feature index is outside decoder")
    _finite_tensor(matrix, "decoder")
    rng = random.Random(seed)
    selected = tuple(
        sorted(rng.sample(eligible, candidate.cardinality))
    )
    coefficients = permute_signed_coefficients(
        candidate.coefficients,
        seed=seed,
    )
    return DirectionCandidate.from_components(
        feature_indices=selected,
        coefficients=coefficients,
        decoder_columns=matrix[:, selected],
        target_norm=candidate.target_norm,
        target=candidate.target,
        kind="random_control",
        requested_cardinality=candidate.requested_cardinality,
        rescale=True,
    )
