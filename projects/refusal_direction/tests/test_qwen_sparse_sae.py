import pytest
import torch

from refusal_direction_replication.qwen_sparse_sae import (
    SAEQuality,
    TopKSparseAutoencoder,
    evaluate_sae_quality,
    remove_parallel_decoder_grad_,
    sae_loss,
)


def test_topk_sae_keeps_exactly_k_positive_features() -> None:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(
            torch.arange(1, 33, dtype=torch.float32).reshape(8, 4)
        )
        sae.encoder.bias.zero_()
        sae.decoder.bias.zero_()

    encoded = sae.encode(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    assert encoded.shape == (1, 8)
    assert torch.count_nonzero(encoded, dim=-1).tolist() == [2]
    assert torch.all(encoded >= 0)


def test_decoder_columns_remain_unit_norm() -> None:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    with torch.no_grad():
        sae.decoder.weight.mul_(torch.arange(1, 9, dtype=torch.float32))

    sae.renormalize_decoder_()

    norms = torch.linalg.vector_norm(sae.decoder.weight, dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_sae_loss_handles_no_dead_features() -> None:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    total, metrics = sae_loss(
        sae,
        torch.randn(3, 4),
        torch.zeros(8, dtype=torch.bool),
    )

    assert torch.isfinite(total)
    assert metrics["aux"].item() == 0.0
    assert metrics["l0"].item() <= 2.0


def test_decoder_gradient_projection_removes_parallel_component() -> None:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    sae.decoder.weight.grad = sae.decoder.weight.detach().clone()

    remove_parallel_decoder_grad_(sae)

    parallel_component = (
        sae.decoder.weight.grad * sae.decoder.weight
    ).sum(dim=0)
    torch.testing.assert_close(
        parallel_component,
        torch.zeros_like(parallel_component),
        atol=1e-6,
        rtol=0.0,
    )


def test_quality_gate_rejects_low_explained_variance() -> None:
    quality = SAEQuality(
        explained_variance=0.79,
        mean_cosine=0.95,
        nonfinite_count=0,
        mean_l0=64.0,
        dead_feature_fraction=0.10,
        norm_drift=0.05,
    )
    assert quality.status == "sae_quality_failed"


def test_quality_gate_accepts_all_threshold_boundaries() -> None:
    quality = SAEQuality(
        explained_variance=0.80,
        mean_cosine=0.90,
        nonfinite_count=0,
        mean_l0=64.0,
        dead_feature_fraction=0.20,
        norm_drift=0.20,
    )
    assert quality.status == "pass"


def test_quality_uses_global_explained_variance_across_uneven_batches() -> None:
    sae = TopKSparseAutoencoder(d_in=2, d_sae=2, k=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(2))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(0.5 * torch.eye(2))
        sae.decoder.bias.zero_()
    batches = [
        torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
        torch.tensor([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]),
    ]

    quality = evaluate_sae_quality(
        sae,
        batches,
        fit_mean_norm=torch.cat(batches).norm(dim=-1).mean().item(),
    )

    assert quality.explained_variance == pytest.approx(-0.8125)
    assert quality.mean_cosine == pytest.approx(1.0)
    assert quality.nonfinite_count == 0
    assert quality.mean_l0 == 2.0
    assert quality.dead_feature_fraction == 0.0
    assert quality.norm_drift < 1e-6
