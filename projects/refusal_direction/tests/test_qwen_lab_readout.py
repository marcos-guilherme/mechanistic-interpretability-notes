from __future__ import annotations

from pathlib import Path

import pytest
import torch
from jlens.lens import JacobianLens

from refusal_direction_replication.qwen_dose import sha256_file
from refusal_direction_replication.qwen_lab_readout import (
    FROZEN_LENS_SHA256,
    LiveJLensReadout,
    assert_incremental_readout_parity,
    build_qwen_sized_lens,
    validate_qwen_lab_lens,
    verify_frozen_lens_hash,
)
from jlens_tiny import TinyDecoder


def test_frozen_lens_hash_matches_evaluation_record() -> None:
    assert (
        FROZEN_LENS_SHA256
        == "b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e"
    )


def test_validate_qwen_lab_lens_rejects_wrong_shape_and_hash(tmp_path: Path) -> None:
    small = JacobianLens({0: torch.eye(4)}, n_prompts=1, d_model=4)
    with pytest.raises(ValueError, match="source_layers"):
        validate_qwen_lab_lens(small)
    wrong_width = build_qwen_sized_lens(d_model=4)
    with pytest.raises(ValueError, match="d_model"):
        validate_qwen_lab_lens(wrong_width)
    small_path = tmp_path / "tiny.pt"
    small.save(str(small_path))
    with pytest.raises(ValueError, match="lens_sha256"):
        verify_frozen_lens_hash(str(small_path), "0" * 64)
    assert sha256_file(small_path) == sha256_file(small_path)


def test_live_readout_never_calls_lens_apply() -> None:
    decoder = TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=0)
    jacobians = {layer: torch.eye(8) for layer in range(3)}
    lens = JacobianLens(jacobians, n_prompts=1, d_model=8)

    class SpyLens(JacobianLens):
        def apply(self, *args, **kwargs):
            raise AssertionError("lens.apply must not run on the live readout path")

    spy = SpyLens(lens.jacobians, n_prompts=lens.n_prompts, d_model=lens.d_model)
    readout = LiveJLensReadout(
        spy,
        decoder,
        decode_token=lambda token_id: decoder.tokenizer.decode([token_id]),
        _skip_qwen_contract=True,
    )
    residual = decoder.forward(decoder.encode("ab")).last_hidden_state[0, -1]
    readout.predict(residual, layer=1, top_k=3)


def test_incremental_readout_matches_apply_on_tiny_decoder() -> None:
    decoder = TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=1)
    jacobians = {layer: torch.eye(8) for layer in range(3)}
    lens = JacobianLens(jacobians, n_prompts=2, d_model=8)

    assert_incremental_readout_parity(
        lens,
        decoder,
        "cat",
        layer=2,
        position=-1,
        top_k=4,
        atol=1e-5,
    )


def test_readout_tracks_rank_deltas_across_steps() -> None:
    decoder = TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=2)
    jacobians = {layer: torch.eye(8) for layer in range(3)}
    lens = JacobianLens(jacobians, n_prompts=1, d_model=8)
    readout = LiveJLensReadout(
        lens,
        decoder,
        decode_token=lambda token_id: decoder.tokenizer.decode([int(token_id)]),
        _skip_qwen_contract=True,
    )
    residual_a = torch.randn(8)
    residual_b = residual_a + 0.05
    first = readout.predict(residual_a, layer=1, top_k=3)
    second = readout.predict(residual_b, layer=1, top_k=3)
    assert all(1 <= prediction.rank <= 3 for prediction in first)
    overlap = {p.token_id for p in first} & {p.token_id for p in second}
    for token_id in overlap:
        first_pred = next(p for p in first if p.token_id == token_id)
        second_pred = next(p for p in second if p.token_id == token_id)
        assert second_pred.previous_rank_delta == first_pred.rank - second_pred.rank


def test_parity_honors_intervention_hooks() -> None:
    decoder = TinyDecoder(n_layers=4, d_model=8, vocab_size=32, seed=3)
    jacobians = {layer: torch.eye(8) for layer in range(3)}
    lens = JacobianLens(jacobians, n_prompts=1, d_model=8)
    module = decoder.layers[2].linear

    def scale(_module, inputs):
        return (inputs[0] * 2.0,)

    assert_incremental_readout_parity(
        lens,
        decoder,
        "hook",
        layer=2,
        position=-1,
        top_k=3,
        atol=1e-5,
        pre_hooks=[(module, scale)],
    )