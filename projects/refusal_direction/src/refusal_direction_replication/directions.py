from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CandidateDirection:
    layer: int
    position_offset: int
    vector: torch.Tensor

    @property
    def unit(self) -> torch.Tensor:
        norm = self.vector.norm()
        if norm == 0:
            raise ValueError("candidate direction has zero norm")
        return self.vector / norm


def difference_in_means(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
) -> torch.Tensor:
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise ValueError("expected [examples, d_model] tensors")
    return harmful.float().mean(0) - harmless.float().mean(0)


def candidate_grid(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    *,
    position_offsets: tuple[int, ...],
) -> tuple[CandidateDirection, ...]:
    if harmful.shape[1:] != harmless.shape[1:]:
        raise ValueError("harmful and harmless activation grids must align")
    if harmful.ndim != 4:
        raise ValueError("expected [examples, layers, positions, d_model]")
    return tuple(
        CandidateDirection(
            layer,
            position_offsets[position],
            difference_in_means(
                harmful[:, layer, position],
                harmless[:, layer, position],
            ),
        )
        for layer in range(harmful.shape[1])
        for position in range(harmful.shape[2])
    )


def save_direction(path: Path, candidate: CandidateDirection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer": candidate.layer,
            "position_offset": candidate.position_offset,
            "vector": candidate.vector.cpu(),
        },
        path,
    )


def load_direction(path: Path) -> CandidateDirection:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return CandidateDirection(
        int(payload["layer"]),
        int(payload["position_offset"]),
        payload["vector"],
    )
