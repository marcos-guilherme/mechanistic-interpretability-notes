"""Memory-bounded causal and norm-matched control interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jspace.hooks import PatchHook
from jspace_llama.runtime import JSpaceExperimentError


class RandomControlError(JSpaceExperimentError):
    pass


class InterventionPreparationError(JSpaceExperimentError):
    pass


@dataclass(frozen=True)
class SwapLayer:
    vectors: torch.Tensor
    pseudoinverse: torch.Tensor
    noop: bool


def prepare_swap_layers(
    dictionary: Any,
    *,
    layers: Sequence[int],
    source: str,
    target: str,
) -> dict[int, SwapLayer]:
    """Retain only two small vectors per layer, always evicting full matrices."""
    prepared: dict[int, SwapLayer] = {}
    for layer in layers:
        try:
            source_id, source_vector = dictionary.token_vector(layer, source)
            if source == target:
                target_id, target_vector = source_id, source_vector
            else:
                target_id, target_vector = dictionary.token_vector(layer, target)
            vectors = torch.stack(
                [
                    source_vector.detach().float().cpu(),
                    target_vector.detach().float().cpu(),
                ]
            )
            prepared[int(layer)] = SwapLayer(
                vectors=vectors,
                pseudoinverse=torch.linalg.pinv(vectors.T),
                noop=int(source_id) == int(target_id),
            )
        except torch.OutOfMemoryError:
            raise
        except (KeyError, torch.linalg.LinAlgError) as exc:
            raise InterventionPreparationError(
                f"swap-vector preparation failed at layer {layer}"
            ) from exc
        finally:
            dictionary.evict(layer)
    return prepared


def apply_causal_clamp(
    hidden: torch.Tensor,
    layer: SwapLayer,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply upstream LensSwap clamp math to arbitrary leading dimensions."""
    if layer.noop:
        return hidden.clone()
    vectors = layer.vectors.to(device=hidden.device, dtype=torch.float32)
    pseudoinverse = layer.pseudoinverse.to(
        device=hidden.device, dtype=torch.float32
    )
    values = hidden.float()
    coordinates = values @ pseudoinverse.T
    mass = (
        coordinates[..., 0] + coordinates[..., 1].clamp_min(0.0)
    ).clamp_min(0.0)
    clamped = torch.stack(
        [torch.zeros_like(mass), float(alpha) * mass], dim=-1
    )
    return values + (clamped - coordinates) @ vectors


def _orthogonal_random_direction(
    vectors: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    try:
        vectors = vectors.detach().float().cpu()
        left, singular, _ = torch.linalg.svd(
            vectors.T, full_matrices=False
        )
        if singular.numel():
            keep = singular > singular[0] * 1e-6
            basis = left[:, keep]
        else:
            basis = left[:, :0]
        for _ in range(8):
            candidate = torch.randn(
                vectors.shape[1], generator=generator, dtype=torch.float32
            )
            if basis.numel():
                candidate = candidate - basis @ (basis.T @ candidate)
            norm = candidate.norm()
            if norm > 1e-6:
                return candidate / norm
    except torch.linalg.LinAlgError as error:
        raise RandomControlError(
            "random-control projection preparation failed"
        ) from error
    raise RandomControlError(
        "could not construct a random direction outside source/target span"
    )


def make_random_directions(
    prepared: Mapping[int, SwapLayer], *, seed: int
) -> dict[int, torch.Tensor]:
    """Create one deterministic source/target-orthogonal unit vector per layer."""
    directions: dict[int, torch.Tensor] = {}
    for layer, swap_layer in prepared.items():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(layer) * 1_000_003)
        directions[layer] = _orthogonal_random_direction(
            swap_layer.vectors, generator=generator
        )
    return directions


def apply_norm_matched_random(
    hidden: torch.Tensor,
    layer: SwapLayer,
    direction: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    causal_delta = apply_causal_clamp(hidden, layer, alpha=alpha) - hidden.float()
    norms = causal_delta.norm(dim=-1, keepdim=True)
    unit = direction.to(device=hidden.device, dtype=torch.float32)
    return hidden.float() + norms * unit


class MemoryBoundedLensSwap(PatchHook):
    """Prompt-prefill-only clamp swap with bounded dictionary preparation."""

    def __init__(
        self,
        model: Any,
        dictionary: Any,
        source: str,
        target: str,
        *,
        layers: Sequence[int],
        positions: Sequence[int] | None = None,
        alpha: float = 4.0,
    ) -> None:
        super().__init__(model, layers, patch_generation=False)
        self._positions = None if positions is None else list(positions)
        self._alpha = float(alpha)
        self._prepared = prepare_swap_layers(
            dictionary, layers=self._layers, source=source, target=target
        )

    def patch(
        self, hidden: torch.Tensor, layer: int, is_prefill: bool
    ) -> torch.Tensor:
        output = hidden.float().clone()
        selection = (
            self._positions
            if is_prefill and self._positions is not None
            else slice(None)
        )
        output[:, selection] = apply_causal_clamp(
            output[:, selection],
            self._prepared[layer],
            alpha=self._alpha,
        )
        return output


class NormMatchedRandomControl(PatchHook):
    """Prompt-prefill-only random control with exact slotwise delta norms."""

    def __init__(
        self,
        model: Any,
        dictionary: Any,
        source: str,
        target: str,
        *,
        layers: Sequence[int],
        positions: Sequence[int] | None = None,
        alpha: float = 4.0,
        seed: int = 42,
    ) -> None:
        super().__init__(model, layers, patch_generation=False)
        self._positions = None if positions is None else list(positions)
        self._alpha = float(alpha)
        self._prepared = prepare_swap_layers(
            dictionary, layers=self._layers, source=source, target=target
        )
        self._directions = make_random_directions(
            self._prepared, seed=int(seed)
        )

    def patch(
        self, hidden: torch.Tensor, layer: int, is_prefill: bool
    ) -> torch.Tensor:
        output = hidden.float().clone()
        selection = (
            self._positions
            if is_prefill and self._positions is not None
            else slice(None)
        )
        output[:, selection] = apply_norm_matched_random(
            output[:, selection],
            self._prepared[layer],
            self._directions[layer],
            alpha=self._alpha,
        )
        return output
