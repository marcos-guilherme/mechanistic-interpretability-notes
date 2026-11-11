"""Direct J-lens and logit-lens ranking for observational prompts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from jspace_llama.observation_data import (
    MultihopItem,
    PANEL_NAMES,
    normalize_prompt_token_ids,
)
from jspace_llama.runtime import TokenizationError


class ObservationReadoutError(ValueError):
    """Expected malformed or non-finite numerical readout output."""


def rank_one_based(logits: torch.Tensor, token_id: int) -> int:
    """Return a one-based competition rank for ``token_id``."""
    logits = _logit_vector(logits)
    if (
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        or token_id >= logits.numel()
    ):
        raise ObservationReadoutError("target token ID is outside logits")
    score = logits[int(token_id)]
    return 1 + int((logits > score).sum().item())


def aggregate_pass_at_k(
    records: list[dict[str, object]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    """Average each item's fraction of intermediates recovered by rank ``k``."""
    totals = {k: 0.0 for k in ks}
    n_intermediates = 0
    for record in records:
        intermediates = record["intermediates"]
        if not isinstance(intermediates, list) or not intermediates:
            raise ValueError("each record must contain intermediates")
        n_intermediates += len(intermediates)
        for k in ks:
            totals[k] += (
                sum(
                    int(intermediate["best_rank"]) <= k
                    for intermediate in intermediates
                )
                / len(intermediates)
            )
    n_items = len(records)
    return {
        "n_items": n_items,
        "n_intermediates": n_intermediates,
        **{
            f"pass@{k}": totals[k] / n_items if n_items else 0.0
            for k in ks
        },
    }


def _logit_vector(logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ObservationReadoutError("lens logits must be a tensor")
    if logits.ndim == 1:
        vector = logits
    elif logits.ndim == 2 and logits.shape[0] == 1:
        vector = logits[0]
    else:
        raise ObservationReadoutError(
            "lens logits must contain exactly one selected position"
        )
    if (
        vector.numel() == 0
        or not vector.dtype.is_floating_point
        or not torch.isfinite(vector).all().item()
    ):
        raise ObservationReadoutError(
            "lens logits must be non-empty and finite"
        )
    return vector


def _best_rank(
    ranks: list[dict[str, int]],
    selected_layers: frozenset[int],
) -> tuple[int, int]:
    selected = [entry for entry in ranks if entry["layer"] in selected_layers]
    if not selected:
        raise ValueError("rank scope must contain at least one layer")
    best = min(selected, key=lambda entry: entry["rank"])
    return best["rank"], best["layer"]


def _validate_logits_by_layer(
    logits_by_layer: Any,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    if not isinstance(logits_by_layer, dict):
        raise ObservationReadoutError("lens output must map layers to logits")
    vectors: dict[int, torch.Tensor] = {}
    vocabulary_size: int | None = None
    for layer in layers:
        if layer not in logits_by_layer:
            raise ObservationReadoutError("lens output is missing a layer")
        vector = _logit_vector(logits_by_layer[layer])
        if vocabulary_size is None:
            vocabulary_size = vector.numel()
        elif vector.numel() != vocabulary_size:
            raise ObservationReadoutError(
                "lens vocabulary dimension differs across layers"
            )
        vectors[layer] = vector
    return vectors


def _top_tokens(
    logits_by_layer: dict[int, torch.Tensor],
    layers: list[int],
    tokenizer: Any,
    top: int,
) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for layer in layers:
        logits = _logit_vector(logits_by_layer[layer])
        count = min(top, int(logits.numel()))
        token_ids = torch.argsort(
            logits,
            descending=True,
            stable=True,
        )[:count].tolist()
        slots.append(
            {
                "layer": layer,
                "tokens": [
                    {
                        "token_id": int(token_id),
                        "token": str(tokenizer.decode([int(token_id)])),
                        "rank": rank_one_based(logits, int(token_id)),
                    }
                    for token_id in token_ids
                ],
            }
        )
    return slots


def _audit_by_name(
    audit: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for entry in audit:
        name = entry.get("name")
        if not isinstance(name, str) or name in indexed:
            raise ValueError("tokenization audit names must be unique strings")
        indexed[name] = entry
    return indexed


def _metric_records(
    records: list[dict[str, object]],
    method: str,
    scope: str,
    *,
    single_token_only: bool,
) -> list[dict[str, object]]:
    projected: list[dict[str, object]] = []
    rank_key = (
        "best_rank" if scope == "band" else "all_layers_best_rank"
    )
    for record in records:
        intermediates = [
            {"best_rank": intermediate[method][rank_key]}
            for intermediate in record["intermediates"]
            if not single_token_only or intermediate["single_token"]
        ]
        if intermediates:
            projected.append({"intermediates": intermediates})
    return projected


def _metrics(
    records: list[dict[str, object]],
    *,
    single_token_only: bool,
) -> dict[str, object]:
    return {
        method: {
            scope: aggregate_pass_at_k(
                _metric_records(
                    records,
                    method,
                    scope,
                    single_token_only=single_token_only,
                )
            )
            for scope in ("band", "all_layers")
        }
        for method in ("jlens", "logit_lens")
    }


@torch.no_grad()
def evaluate_direct_readout(
    model: Any,
    dictionary: Any,
    items: Sequence[MultihopItem],
    audit: Sequence[dict[str, object]],
    *,
    band_layers: Sequence[int],
    all_layers: Sequence[int],
    detail_names: frozenset[str] = frozenset(PANEL_NAMES),
    top: int = 10,
) -> dict[str, object]:
    """Rank audited intermediate tokens without generation or intervention."""
    layers = list(dict.fromkeys(int(layer) for layer in all_layers))
    band_layer_list = list(
        dict.fromkeys(int(layer) for layer in band_layers)
    )
    band = frozenset(band_layer_list)
    if not layers:
        raise ValueError("all_layers must not be empty")
    if not band or not band <= set(layers):
        raise ValueError("band_layers must be a non-empty subset of all_layers")
    if top < 0:
        raise ValueError("top must not be negative")

    audited = _audit_by_name(audit)
    detailed = detail_names & frozenset(PANEL_NAMES)
    tokenizer = dictionary.model.tokenizer
    records: list[dict[str, object]] = []

    for item in items:
        audit_entry = audited.get(item.name)
        if audit_entry is None:
            raise ValueError(f"missing tokenization audit for {item.name}")
        audit_intermediates = audit_entry.get("intermediates")
        prompt_ids = audit_entry.get("prompt_token_ids")
        if (
            audit_entry.get("prompt") != item.prompt
            or not isinstance(prompt_ids, (list, tuple))
            or not prompt_ids
        ):
            raise ValueError(f"prompt tokenization audit mismatch for {item.name}")
        try:
            lens_prompt_ids = normalize_prompt_token_ids(
                model.encode(item.prompt, max_length=512)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise TokenizationError(
                f"lens prompt tokenization failed for {item.name}"
            ) from error
        if tuple(prompt_ids) != lens_prompt_ids:
            raise TokenizationError(
                f"lens prompt tokens differ from audit for {item.name}"
            )
        if not isinstance(audit_intermediates, (list, tuple)) or len(
            audit_intermediates
        ) != len(item.intermediates):
            raise ValueError(f"tokenization audit mismatch for {item.name}")

        intermediates: list[dict[str, object]] = []
        for text, tokenized in zip(item.intermediates, audit_intermediates):
            if not isinstance(tokenized, dict) or tokenized.get("text") != text:
                raise ValueError(f"tokenization audit mismatch for {item.name}")
            intermediates.append(
                {
                    "text": text,
                    "token_id": int(tokenized["token_id"]),
                    "single_token": bool(tokenized["single_token"]),
                }
            )

        record: dict[str, object] = {
            "name": item.name,
            "prompt": item.prompt,
            "target": item.target,
            "intermediates": intermediates,
        }
        top_tokens: dict[str, object] = {}
        for method, use_jacobian in (("jlens", True), ("logit_lens", False)):
            logits_by_layer, _, _ = dictionary.lens.apply(
                model,
                item.prompt,
                layers=layers,
                positions=[-1],
                max_seq_len=512,
                use_jacobian=use_jacobian,
            )
            logits_by_layer = _validate_logits_by_layer(
                logits_by_layer,
                layers,
            )
            for intermediate in intermediates:
                token_id = int(intermediate["token_id"])
                ranks = [
                    {
                        "layer": layer,
                        "rank": rank_one_based(
                            _logit_vector(logits_by_layer[layer]),
                            token_id,
                        ),
                    }
                    for layer in layers
                ]
                best_rank, best_layer = _best_rank(ranks, band)
                all_layers_best_rank, all_layers_best_layer = _best_rank(
                    ranks,
                    frozenset(layers),
                )
                intermediate[method] = {
                    "ranks": ranks,
                    "best_rank": best_rank,
                    "best_layer": best_layer,
                    "all_layers_best_rank": all_layers_best_rank,
                    "all_layers_best_layer": all_layers_best_layer,
                }
            if item.name in detailed:
                top_tokens[method] = _top_tokens(
                    logits_by_layer,
                    band_layer_list,
                    tokenizer,
                    top,
                )
        if top_tokens:
            record["top_tokens"] = top_tokens
        records.append(record)

    return {
        "items": records,
        "metrics": _metrics(records, single_token_only=False),
        "single_token_metrics": _metrics(records, single_token_only=True),
    }
