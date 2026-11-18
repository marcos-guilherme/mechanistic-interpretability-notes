import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from refusal_direction_replication.qwen_sparse_sae import TopKSparseAutoencoder
from refusal_direction_replication.qwen_sparse_training import (
    ActivationBatch,
    SparseTrainConfig,
    create_sae_optimizer,
    load_training_checkpoint,
    measure_sae_training_candidate,
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
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_tokens=10,
        total_fit_tokens=40,
        effective_batch_tokens=10,
        checkpoint_every_tokens=10,
        activation_scale=2.5,
        dead_feature_window_tokens=10,
        device="cpu",
    )


def fresh_sae(state: dict[str, torch.Tensor]) -> TopKSparseAutoencoder:
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    sae.load_state_dict(state)
    return sae


def test_training_publishes_each_valid_token_checkpoint(
    tmp_path: Path,
) -> None:
    published: list[Path] = []
    result = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        deterministic_batches(seed=31, count=4, d_in=4),
        tiny_config(),
        tmp_path / "published",
        on_checkpoint=lambda path: published.append(Path(path)),
    )

    assert len(published) >= 2
    assert published[-1] == result.checkpoint_path
    assert all(path.is_file() for path in published)


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


def test_resume_validates_skipped_prefix_batch_shape(tmp_path: Path) -> None:
    batches = deterministic_batches(seed=17, count=3, d_in=4)
    partial = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        batches[:2],
        tiny_config(),
        tmp_path / "partial-shape",
    )
    mismatched = [
        ActivationBatch(torch.ones(5, 5)),
        *batches[1:],
    ]

    with pytest.raises(ValueError, match="shape"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            mismatched,
            tiny_config(),
            tmp_path / "resume-shape",
            resume_from=partial.checkpoint_path,
        )


def test_resume_rejects_truncated_prefix(tmp_path: Path) -> None:
    batches = deterministic_batches(seed=23, count=3, d_in=4)
    partial = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        batches[:2],
        tiny_config(),
        tmp_path / "partial-truncated",
    )

    with pytest.raises(ValueError, match="truncated resume prefix"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            batches[:1],
            tiny_config(),
            tmp_path / "resume-truncated",
            resume_from=partial.checkpoint_path,
        )


def test_resume_rejects_prefix_token_mismatch(tmp_path: Path) -> None:
    batches = deterministic_batches(seed=29, count=3, d_in=4)
    partial = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        batches[:2],
        tiny_config(),
        tmp_path / "partial-tokens",
    )
    mismatched = [
        ActivationBatch(torch.ones(4, 4)),
        batches[1],
        batches[2],
    ]

    with pytest.raises(ValueError, match="tokens_seen"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            mismatched,
            tiny_config(),
            tmp_path / "resume-tokens",
            resume_from=partial.checkpoint_path,
        )


def test_dead_feature_timestamps_use_last_selected_token(tmp_path: Path) -> None:
    sae = TopKSparseAutoencoder(d_in=1, d_sae=4, k=2)
    with torch.no_grad():
        sae.encoder.weight.copy_(
            torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
        )
        sae.encoder.bias.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0]))
        sae.decoder.bias.zero_()
    result = train_sae_stream(
        sae,
        [ActivationBatch(torch.tensor([[2.0], [-2.0], [0.0]]))],
        tiny_config(),
        tmp_path / "timestamps",
    )

    assert result.last_activation_token.tolist() == [3, 2, 3, 0]


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
        "optimizer_name",
        "schedule",
        "model_state",
        "optimizer_state",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "last_activation_token",
    }
    assert payload["identity"] == IDENTITY
    assert payload["optimizer_name"] == "AdamW"
    assert payload["schedule"] == {
        "name": "linear_warmup_cosine",
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_tokens": 10,
        "total_fit_tokens": 40,
        "effective_batch_tokens": 10,
    }
    assert payload["optimizer_state"]["param_groups"][0]["betas"] == (
        0.9,
        0.999,
    )
    assert payload["optimizer_state"]["param_groups"][0]["eps"] == 1e-8
    assert payload["optimizer_state"]["param_groups"][0]["weight_decay"] == 0
    assert payload["activation_scale"] == 2.5
    assert payload["last_activation_token"].shape == (8,)
    assert not list(tmp_path.glob("*.tmp"))


def test_learning_rate_uses_spec_warmup_and_cosine() -> None:
    from refusal_direction_replication.qwen_sparse_training import (
        learning_rate_for_tokens,
    )

    config = tiny_config()
    assert learning_rate_for_tokens(config, 0) == 0
    assert learning_rate_for_tokens(config, 10) == pytest.approx(1e-3)
    assert learning_rate_for_tokens(config, 25) == pytest.approx(5.5e-4)
    assert learning_rate_for_tokens(config, 40) == pytest.approx(1e-4)
    assert learning_rate_for_tokens(config, 100) == pytest.approx(1e-4)


def test_checkpoint_v1_is_rejected_with_explicit_migration_error(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save({"schema_version": 1}, legacy)
    with pytest.raises(ValueError, match="legacy checkpoint schema_version 1"):
        load_training_checkpoint(legacy, IDENTITY)


def test_resume_rejects_optimizer_schedule_change(tmp_path: Path) -> None:
    partial = train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        deterministic_batches(seed=41, count=2, d_in=4),
        tiny_config(),
        tmp_path / "partial-schedule",
    )
    changed = replace(tiny_config(), min_learning_rate=5e-5)
    with pytest.raises(ValueError, match="schedule"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            deterministic_batches(seed=41, count=4, d_in=4),
            changed,
            tmp_path / "changed-schedule",
            resume_from=partial.checkpoint_path,
        )


def test_optimizer_steps_only_after_effective_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = 0
    original = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    train_sae_stream(
        TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
        deterministic_batches(seed=43, count=4, d_in=4),
        tiny_config(),
        tmp_path / "effective-batch",
    )
    assert steps == 2


def test_shared_effective_batch_path_allocates_adamw_state_and_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import refusal_direction_replication.qwen_sparse_training as training

    torch.manual_seed(47)
    sae = TopKSparseAutoencoder(d_in=4, d_sae=8, k=2)
    optimizer = create_sae_optimizer(sae)
    projection_calls = 0
    original_projection = training.remove_parallel_decoder_grad_

    def counted_projection(model):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(model)

    monkeypatch.setattr(
        training,
        "remove_parallel_decoder_grad_",
        counted_projection,
    )
    assert optimizer.state == {}
    before = sae.decoder.weight.detach().clone()
    factory_calls = 0

    def batch_factory():
        nonlocal factory_calls
        factory_calls += 1
        generator = torch.Generator().manual_seed(100 + factory_calls)
        return [
            ActivationBatch(torch.randn(2, 4, generator=generator))
            for _ in range(4)
        ]

    clock_values = iter((10.0, 12.0))
    resets = 0

    def reset_peak_memory() -> None:
        nonlocal resets
        resets += 1

    measurement = measure_sae_training_candidate(
        sae,
        optimizer,
        batch_factory,
        device="cpu",
        effective_batch_tokens=8,
        learning_rate=1e-3,
        synchronize=lambda: None,
        reset_peak_memory=reset_peak_memory,
        clock=lambda: next(clock_values),
    )

    assert factory_calls == 2
    assert projection_calls == 2
    assert resets == 1
    assert measurement.elapsed_seconds == 2.0
    assert measurement.tokens_per_second == 4.0
    assert measurement.finite
    assert measurement.optimizer_state_allocated
    assert optimizer.state
    assert not torch.equal(before, sae.decoder.weight)
    assert torch.allclose(
        torch.linalg.vector_norm(sae.decoder.weight, dim=0),
        torch.ones(sae.d_sae),
        atol=1e-5,
    )


def test_training_rejects_wrong_activation_width(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            [ActivationBatch(torch.ones(3, 5))],
            tiny_config(),
            tmp_path,
        )


def test_atomic_checkpoint_cleanup_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "failed-save"

    def fail_after_partial_write(payload: object, path: Path) -> None:
        del payload
        Path(path).write_bytes(b"partial")
        raise OSError("forced save failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="forced save failure"):
        train_sae_stream(
            TopKSparseAutoencoder(d_in=4, d_sae=8, k=2),
            deterministic_batches(seed=31, count=1, d_in=4),
            tiny_config(),
            checkpoint_dir,
        )

    assert list(checkpoint_dir.iterdir()) == []
