"""TopK sparse autoencoder and frozen quality gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class TopKSparseAutoencoder(nn.Module):
    """Sparse autoencoder with nonnegative, exact-TopK latent selection."""

    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__()
        if d_in <= 0 or d_sae <= 0:
            raise ValueError("d_in and d_sae must be positive")
        if not 0 < k <= d_sae:
            raise ValueError("k must be within d_sae")
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=True)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.renormalize_decoder_()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.decoder.bias.zero_()
            self.encoder.bias.zero_()

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.decoder.bias
        return F.relu(self.encoder(centered))

    def encode_with_support(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and expose indices/support using a deterministic tie policy.

        Features are ordered by descending pre-activation. Equal values prefer
        the lower feature index, including all-zero rows. Support records the
        selected TopK positions even when a selected activation value is zero.
        """
        pre = self.pre_activations(x)
        indices = torch.argsort(
            pre,
            dim=-1,
            descending=True,
            stable=True,
        )[..., : self.k]
        values = pre.gather(-1, indices)
        encoded = torch.zeros_like(pre)
        encoded.scatter_(-1, indices, values)
        support = torch.zeros_like(pre, dtype=torch.bool)
        support.scatter_(-1, indices, True)
        return encoded, indices, support

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _, _ = self.encode_with_support(x)
        return encoded

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        return self.decode(encoded), encoded

    @torch.no_grad()
    def renormalize_decoder_(self) -> None:
        weight = self.decoder.weight
        norms = torch.linalg.vector_norm(weight, dim=0).clamp_min(1e-12)
        weight.div_(norms)


def sae_loss(
    sae: TopKSparseAutoencoder,
    x: torch.Tensor,
    dead_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the reconstruction plus dead-latent auxiliary loss."""
    if dead_mask.shape != (sae.d_sae,) or dead_mask.dtype != torch.bool:
        raise ValueError("dead_mask must be a boolean tensor of shape [d_sae]")
    reconstructed, encoded = sae(x)
    residual = x - reconstructed
    mse = residual.square().mean()
    dead_count = int(dead_mask.sum().item())
    if dead_count == 0:
        aux = mse.new_zeros(())
    else:
        pre = sae.pre_activations(x)
        dead_pre = pre.masked_fill(
            ~dead_mask.unsqueeze(0),
            float("-inf"),
        )
        aux_values, aux_indices = torch.topk(
            dead_pre,
            k=min(sae.k, dead_count),
            dim=-1,
        )
        aux_code = torch.zeros_like(pre)
        aux_code.scatter_(
            -1,
            aux_indices,
            aux_values.nan_to_num(),
        )
        aux_reconstruction = F.linear(aux_code, sae.decoder.weight)
        aux = (residual.detach() - aux_reconstruction).square().mean()
    total = mse + aux / 32.0
    return total, {
        "mse": mse.detach(),
        "aux": aux.detach(),
        "l0": encoded.ne(0).sum(-1).float().mean(),
    }


@torch.no_grad()
def remove_parallel_decoder_grad_(sae: TopKSparseAutoencoder) -> None:
    """Project decoder gradients onto each unit column's tangent plane."""
    weight = sae.decoder.weight
    grad = weight.grad
    if grad is None:
        return
    parallel = (grad * weight).sum(dim=0, keepdim=True) * weight
    grad.sub_(parallel)


@dataclass(frozen=True)
class SAEQuality:
    explained_variance: float
    mean_cosine: float
    nonfinite_count: int
    mean_l0: float
    dead_feature_fraction: float
    norm_drift: float

    @property
    def status(self) -> str:
        passed = (
            self.explained_variance >= 0.80
            and self.mean_cosine >= 0.90
            and self.nonfinite_count == 0
            and self.mean_l0 == 64.0
            and self.dead_feature_fraction <= 0.20
            and self.norm_drift <= 0.20
        )
        return "pass" if passed else "sae_quality_failed"


@torch.no_grad()
def evaluate_sae_quality(
    sae: TopKSparseAutoencoder,
    heldout_batches: Iterable[torch.Tensor],
    fit_mean_norm: float,
) -> SAEQuality:
    """Evaluate token-weighted SAE metrics without retaining held-out batches."""
    if not math.isfinite(fit_mean_norm) or fit_mean_norm <= 0:
        raise ValueError("fit_mean_norm must be finite and positive")

    was_training = sae.training
    sae.eval()
    device = sae.decoder.weight.device
    total_tokens = 0
    input_sum = torch.zeros(sae.d_in, dtype=torch.float64)
    input_square_sum = 0.0
    residual_square_sum = 0.0
    cosine_sum = 0.0
    l0_sum = 0.0
    norm_sum = 0.0
    nonfinite_count = 0
    active_features = torch.zeros(sae.d_sae, dtype=torch.bool)

    try:
        for batch in heldout_batches:
            if batch.ndim != 2 or batch.shape[1] != sae.d_in:
                raise ValueError(
                    f"held-out batch shape must be [tokens, {sae.d_in}]"
                )
            if batch.shape[0] == 0:
                continue
            x = batch.to(device=device)
            encoded, _, support = sae.encode_with_support(x)
            reconstructed = sae.decode(encoded)
            nonfinite_count += int((~torch.isfinite(x)).sum().item())
            nonfinite_count += int(
                (~torch.isfinite(reconstructed)).sum().item()
            )
            clean_x = torch.nan_to_num(x).to(dtype=torch.float64)
            clean_reconstructed = torch.nan_to_num(
                reconstructed
            ).to(dtype=torch.float64)
            residual = clean_x - clean_reconstructed
            token_count = int(x.shape[0])
            total_tokens += token_count
            input_sum += clean_x.sum(dim=0).cpu()
            input_square_sum += float(clean_x.square().sum().item())
            residual_square_sum += float(residual.square().sum().item())
            cosine_sum += float(
                F.cosine_similarity(
                    clean_x,
                    clean_reconstructed,
                    dim=-1,
                ).sum().item()
            )
            l0_sum += float(encoded.ne(0).sum().item())
            norm_sum += float(
                torch.linalg.vector_norm(clean_x, dim=-1).sum().item()
            )
            active_features |= support.any(dim=0).cpu()
    finally:
        sae.train(was_training)

    if total_tokens == 0:
        raise ValueError("heldout_batches must contain at least one token")
    centered_square_sum = input_square_sum - float(
        input_sum.square().sum().item()
    ) / total_tokens
    if centered_square_sum <= 0:
        explained_variance = (
            1.0 if residual_square_sum == 0 else float("-inf")
        )
    else:
        explained_variance = 1.0 - (
            residual_square_sum / centered_square_sum
        )
    heldout_mean_norm = norm_sum / total_tokens
    return SAEQuality(
        explained_variance=float(explained_variance),
        mean_cosine=float(cosine_sum / total_tokens),
        nonfinite_count=nonfinite_count,
        mean_l0=float(l0_sum / total_tokens),
        dead_feature_fraction=float(
            (~active_features).sum().item() / sae.d_sae
        ),
        norm_drift=float(
            abs(heldout_mean_norm / fit_mean_norm - 1.0)
        ),
    )
