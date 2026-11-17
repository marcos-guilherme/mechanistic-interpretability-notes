"""Incremental J-lens readout for the Qwen intervention lab."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from .pipeline import temporary_hooks
from .qwen_code_eval_runtime import apply_lens_with_hooks
from .qwen_dose import sha256_file
from .qwen_jlens import EXPECTED_D_MODEL, EXPECTED_SOURCE_LAYERS, validate_fitted_lens
from .qwen_lab import TokenPrediction, escape_token_display


FROZEN_LENS_SHA256 = (
    "b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e"
)
PARITY_LOGITS_ATOL = 1e-4
PARITY_RANK_ATOL = 0.0


DecodeToken = Callable[[int], str]
LensContract = Literal["qwen", "fixture"]
LENS_CONTRACT_QWEN = "qwen"
LENS_CONTRACT_FIXTURE = "fixture"


def verify_frozen_lens_hash(path: str | Path, expected_lens_sha256: str) -> None:
    actual = sha256_file(Path(path))
    if actual != expected_lens_sha256:
        raise ValueError(
            f"lens_sha256 {actual!r} != expected {expected_lens_sha256!r}"
        )


def validate_qwen_lab_lens(
    lens: Any,
    *,
    lens_path: str | None = None,
    expected_lens_sha256: str | None = FROZEN_LENS_SHA256,
    n_prompts: int | None = None,
) -> None:
    fitted_prompts = int(n_prompts if n_prompts is not None else lens.n_prompts)
    validate_fitted_lens(
        lens,
        source_layers=EXPECTED_SOURCE_LAYERS,
        d_model=EXPECTED_D_MODEL,
        n_prompts=fitted_prompts,
    )
    if lens_path is not None and expected_lens_sha256 is not None:
        verify_frozen_lens_hash(lens_path, expected_lens_sha256)


def build_qwen_sized_lens(*, d_model: int, n_prompts: int = 1) -> Any:
    from jlens.lens import JacobianLens

    jacobians = {
        layer: torch.eye(d_model)
        for layer in EXPECTED_SOURCE_LAYERS
    }
    return JacobianLens(jacobians, n_prompts=n_prompts, d_model=d_model)


def logits_to_predictions(
    logits: torch.Tensor,
    *,
    top_k: int,
    prior_ranks: dict[int, int],
    decode_token: DecodeToken,
) -> tuple[TokenPrediction, ...]:
    vector = logits.reshape(-1).float()
    k = min(int(top_k), int(vector.numel()))
    scores, indices = torch.topk(vector, k=k)
    maximum = float(scores[0].item()) if k else 0.0
    predictions: list[TokenPrediction] = []
    for rank, (token_id, score) in enumerate(
        zip(indices.tolist(), scores.tolist(), strict=True),
        start=1,
    ):
        token_id = int(token_id)
        decoded = decode_token(token_id)
        previous = prior_ranks.get(token_id)
        predictions.append(
            TokenPrediction(
                token_id=token_id,
                display_text=escape_token_display(token_id=token_id, text=decoded),
                score=float(score),
                relative_bar=(float(score) / maximum) if maximum else 0.0,
                rank=rank,
                previous_rank_delta=(
                    None if previous is None else int(previous - rank)
                ),
            )
        )
    return tuple(predictions)


class LiveJLensReadout:
    """Top-k J-lens predictions from a single captured block residual."""

    def __init__(
        self,
        lens: Any,
        bound_model: Any,
        *,
        decode_token: DecodeToken,
        lens_path: str | None = None,
        expected_lens_sha256: str | None = FROZEN_LENS_SHA256,
        lens_contract: LensContract = LENS_CONTRACT_QWEN,
    ) -> None:
        if lens_contract not in {LENS_CONTRACT_QWEN, LENS_CONTRACT_FIXTURE}:
            raise ValueError(f"unknown lens_contract: {lens_contract!r}")
        self.lens_contract = lens_contract
        self.lens_path = None if lens_path is None else str(lens_path)
        self.lens_validated = False
        if lens_contract == LENS_CONTRACT_QWEN:
            if self.lens_path is None:
                raise ValueError("lens_path is required for Qwen lab lens validation")
            validate_qwen_lab_lens(
                lens,
                lens_path=self.lens_path,
                expected_lens_sha256=expected_lens_sha256,
            )
            self.lens_validated = True
        self._lens = lens
        self._bound_model = bound_model
        self._decode_token = decode_token
        self._prior_ranks: dict[int, int] = {}

    def predict(
        self,
        residual: torch.Tensor,
        *,
        layer: int,
        top_k: int,
    ) -> tuple[TokenPrediction, ...]:
        layer_index = int(layer)
        if layer_index not in self._lens.jacobians:
            raise ValueError(f"layer {layer_index} is not available in the fitted lens")
        transported = self._lens.transport(residual.float(), layer_index)
        logits = self._bound_model.unembed(transported).float().reshape(-1)
        predictions = logits_to_predictions(
            logits,
            top_k=top_k,
            prior_ranks=self._prior_ranks,
            decode_token=self._decode_token,
        )
        self._prior_ranks = {prediction.token_id: prediction.rank for prediction in predictions}
        return predictions


def incremental_logits_from_residual(
    lens: Any,
    bound_model: Any,
    residual: torch.Tensor,
    *,
    layer: int,
) -> torch.Tensor:
    transported = lens.transport(residual.float(), int(layer))
    return bound_model.unembed(transported).float().reshape(-1)


def assert_incremental_readout_parity(
    lens: Any,
    bound_model: Any,
    prompt: str,
    *,
    layer: int,
    position: int,
    top_k: int,
    atol: float = PARITY_LOGITS_ATOL,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
    return_canonical_logits: bool = False,
) -> None | torch.Tensor:
    """Compare transport+unembed with canonical ``JacobianLens.apply``."""

    def _canonical_logits() -> torch.Tensor:
        lens_logits, _model_logits, _input_ids = apply_lens_with_hooks(
            lens,
            bound_model,
            prompt,
            layers=[int(layer)],
            positions=[int(position)],
            max_seq_len=128,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        return lens_logits[int(layer)][0].float().reshape(-1)

    canonical = _canonical_logits()
    from jlens.hooks import ActivationRecorder

    record_layers = sorted({int(layer), bound_model.n_layers - 1})
    with temporary_hooks(pre_hooks=pre_hooks, forward_hooks=forward_hooks):
        input_ids = bound_model.encode(prompt, max_length=128)
        with ActivationRecorder(bound_model.layers, at=record_layers) as recorder:
            bound_model.forward(input_ids)
            activations = {
                index: recorder.activations[index].detach()
                for index in record_layers
            }
    residual = activations[int(layer)][0, int(position)].float()
    incremental = incremental_logits_from_residual(
        lens,
        bound_model,
        residual,
        layer=layer,
    )
    compare_device = incremental.device
    canonical_compare = canonical.detach().to(
        device=compare_device,
        dtype=torch.float32,
    )
    incremental_compare = incremental.detach().to(
        device=compare_device,
        dtype=torch.float32,
    )
    if not torch.allclose(
        canonical_compare,
        incremental_compare,
        atol=atol,
        rtol=0.0,
    ):
        raise AssertionError("incremental logits diverge from JacobianLens.apply")
    canonical_top = torch.topk(
        canonical_compare,
        k=min(top_k, canonical_compare.numel()),
    ).indices
    incremental_top = torch.topk(
        incremental_compare,
        k=min(top_k, incremental_compare.numel()),
    ).indices
    if not torch.equal(canonical_top, incremental_top):
        raise AssertionError("top-k token ids/order differ between readout paths")
    if return_canonical_logits:
        return canonical
    return None