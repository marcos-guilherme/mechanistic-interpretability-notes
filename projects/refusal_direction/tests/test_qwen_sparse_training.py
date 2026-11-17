import random
from pathlib import Path

import pytest
import torch

from refusal_direction_replication.qwen_sparse_sae import TopKSparseAutoencoder
from refusal_direction_replication.qwen_sparse_training import (
    ActivationBatch,
    SparseTrainConfig,
    load_training_checkpoint,
    train_sae_stream,
)


IDENTITY = {
    "run_id": "tiny-run",
    "implementation_commit": "1" * 40,
    "config_sha256": "2" * 64,
    "corpus_sha256": "3" * 64,
    "model_revision": "4" * 40,
}


def deterministic_batches(
    *, seed: int, count: int, d_in: int
) -> list[ActivationBatch]:
    generator = torch.Generator().manual_seed(seed)
    return [
        ActivationBatch(torch.randn(5, d_in, generator=generator))
        for _ in range(count)
    ]


def tiny_config() -> SparseTrainConfig:
    return SparseTrainConfig(
        **IDENTITY,
        learning_rate=1e-3,
        checkpoint_every_tokens=10,
        activation_scale=2.5,
        dead_feature_window_tokens=10,
        device="cpu",
    )


def fresh_sae(state: dict[str, torch.Tensor]) -> TopKSparseAutoencoder:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    sae.load_state_dict(state)
    return sae


def test_checkpoint_resume_reproduces_uninterrupted_weights(
    tmp_path: Path,
) -> None:
    batches = deterministic_batches(seed=42, count=8, d_in=4)
    torch.manual_seed(7)
    initial = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2).state_dict()
    initial = {name: tensor.clone() for name, tensor in initial.items()}

    random.seed(11)
    torch.manual_seed(13)
    full = train_sae_stream(
        fresh_sae(initial),
        batches,
        tiny_config(),
        tmp_path / "full",
    )
    full_python_state = random.getstate()
    full_torch_state = torch.random.get_rng_state()

    random.seed(11)
    torch.manual_seed(13)
    partial = train_sae_stream(
        fresh_sae(initial),
        batches[:4],
        tiny_config(),
        tmp_path / "partial",
    )
    random.random()
    torch.rand(3)
    resumed = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        batches,
        tiny_config(),
        tmp_path / "resumed",
        resume_from=partial.checkpoint_path,
    )

    assert resumed.tokens_seen == full.tokens_seen == 40
    assert resumed.batch_index == full.batch_index == 8
    for name, expected in full.model_state.items():
        torch.testing.assert_close(resumed.model_state[name], expected, rtol=0, atol=0)
    assert random.getstate() == full_python_state
    torch.testing.assert_close(torch.random.get_rng_state(), full_torch_state)


def test_checkpoint_rejects_changed_and_partial_identity(tmp_path: Path) -> None:
    checkpoint = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        deterministic_batches(seed=2, count=1, d_in=4),
        tiny_config(),
        tmp_path,
    )
    changed = {**IDENTITY, "corpus_sha256": "b" * 64}

    with pytest.raises(ValueError, match="corpus_sha256"):
        load_training_checkpoint(checkpoint.checkpoint_path, changed)
    with pytest.raises(ValueError, match="complete identity"):
        load_training_checkpoint(
            checkpoint.checkpoint_path,
            {"corpus_sha256": IDENTITY["corpus_sha256"]},
        )


def test_checkpoint_contains_resume_state_and_is_atomically_replaced(
    tmp_path: Path,
) -> None:
    result = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        deterministic_batches(seed=5, count=2, d_in=4),
        tiny_config(),
        tmp_path,
    )

    payload = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert set(payload) == {
        "schema_version",
        "identity",
        "tokens_seen",
        "batch_index",
        "activation_scale",
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "last_activation_token",
    }
    assert payload["identity"] == IDENTITY
    assert payload["activation_scale"] == 2.5
    assert payload["last_activation_token"].shape == (8,)
    assert not list(tmp_path.glob("*.tmp"))


def test_training_rejects_wrong_activation_width(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            [ActivationBatch(torch.ones(3, 5))],
            tiny_config(),
            tmp_path,
        )
