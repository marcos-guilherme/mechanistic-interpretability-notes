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


def test_topk_indices_track_selected_zero_values_compactly() -> None:
    sae = TopKSparseAutoencoder(d_in=2, d_sae=4, k=2)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.decoder.bias.zero_()

    encoded, indices = sae.encode_with_indices(torch.zeros(3, 2))

    assert indices.shape == (3, 2)
    assert indices.tolist() == [[0, 1], [0, 1], [0, 1]]
    assert torch.count_nonzero(encoded).item() == 0


def test_topk_boundary_tie_prefers_lower_feature_index() -> None:
    sae = TopKSparseAutoencoder(d_in=1, d_sae=4, k=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(
            torch.tensor([[3.0], [2.0], [2.0], [1.0]])
        )
        sae.encoder.bias.zero_()
        sae.decoder.bias.zero_()

    encoded, indices = sae.encode_with_indices(torch.ones(1, 1))

    assert indices.tolist() == [[0, 1]]
    assert encoded.tolist() == [[3.0, 2.0, 0.0, 0.0]]


def test_ordinary_encode_does_not_request_tracking_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sae = TopKSparseAutoencoder(d_in=2, d_sae=4, k=2)

    def fail_if_requested(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del x
        raise AssertionError("ordinary encode requested tracking indices")

    monkeypatch.setattr(sae, "encode_with_indices", fail_if_requested)

    assert sae.encode(torch.ones(2, 2)).shape == (2, 4)


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


def test_sae_loss_computes_preactivations_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    original = sae.pre_activations
    calls = 0

    def count_calls(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(x)

    monkeypatch.setattr(sae, "pre_activations", count_calls)
    sae_loss(
        sae,
        torch.randn(3, 4),
        torch.ones(8, dtype=torch.bool),
    )

    assert calls == 1


def test_dead_feature_auxiliary_loss_is_residual_energy_normalized() -> None:
    sae = TopKSparseAutoencoder(d_in=1, d_sae=2, k=1)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.copy_(torch.tensor([2.0, 1.0]))
        sae.decoder.weight.copy_(torch.tensor([[1.0, -1.0]]))
        sae.decoder.bias.zero_()

    total, metrics = sae_loss(
        sae,
        torch.zeros(1, 1),
        torch.tensor([False, True]),
    )

    assert metrics["mse"].item() == pytest.approx(4.0)
    assert metrics["aux"].item() == pytest.approx(0.25)
    assert total.item() == pytest.approx(4.0 + 0.25 / 32)


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
        last_activation_token=torch.tensor([100, 100]),
        fit_tokens_seen=100,
        dead_feature_window_tokens=10,
    )

    assert quality.explained_variance == pytest.approx(-0.8125)
    assert quality.mean_cosine == pytest.approx(1.0)
    assert quality.nonfinite_count == 0
    assert quality.mean_l0 == 2.0
    assert quality.dead_feature_fraction == 0.0
    assert quality.norm_drift < 1e-6


def test_quality_combines_recent_fit_and_holdout_feature_activity() -> None:
    sae = TopKSparseAutoencoder(d_in=1, d_sae=2, k=1)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.copy_(torch.tensor([2.0, 1.0]))
        sae.decoder.weight.copy_(torch.tensor([[1.0, -1.0]]))
        sae.decoder.bias.zero_()

    quality = evaluate_sae_quality(
        sae,
        [torch.zeros(2, 1)],
        fit_mean_norm=1.0,
        last_activation_token=torch.tensor([100, 95]),
        fit_tokens_seen=100,
        dead_feature_window_tokens=10,
    )

    assert quality.dead_feature_fraction == 0.0
