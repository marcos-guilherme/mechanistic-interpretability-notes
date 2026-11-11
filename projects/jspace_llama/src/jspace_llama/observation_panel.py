"""Memory-bounded qualitative readout for the registered prompt panel."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
import math
from typing import Any

import torch

from jspace_llama.observation_data import MultihopItem, PANEL_NAMES
from jspace_llama.observation_readout import ObservationReadoutError


FINITE_CHECK_CHUNK_ELEMENTS = 1_000_000


def _tensor_chunks_bounded(
    tensor: torch.Tensor,
    *,
    max_elements: int,
) -> Iterator[torch.Tensor]:
    """Yield tensor views containing at most ``max_elements`` values."""
    if tensor.numel() <= max_elements:
        yield tensor
        return
    if tensor.is_contiguous():
        flattened = tensor.view(-1)
        for start in range(0, flattened.numel(), max_elements):
            yield flattened.narrow(
                0,
                start,
                min(max_elements, flattened.numel() - start),
            )
        return
    if tensor.ndim == 1:
        for start in range(0, tensor.numel(), max_elements):
            yield tensor.narrow(
                0,
                start,
                min(max_elements, tensor.numel() - start),
            )
        return
    trailing_elements = tensor[0].numel()
    if trailing_elements <= max_elements:
        rows_per_chunk = max(1, max_elements // trailing_elements)
        for start in range(0, tensor.shape[0], rows_per_chunk):
            yield tensor.narrow(
                0,
                start,
                min(rows_per_chunk, tensor.shape[0] - start),
            )
        return
    for subtensor in tensor.unbind(0):
        yield from _tensor_chunks_bounded(
            subtensor,
            max_elements=max_elements,
        )


def _tensor_is_finite_bounded(tensor: torch.Tensor) -> bool:
    """Check finiteness with at most one 1M-element mask resident at a time."""
    for chunk in _tensor_chunks_bounded(
        tensor,
        max_elements=FINITE_CHECK_CHUNK_ELEMENTS,
    ):
        finite_mask = torch.isfinite(chunk)
        chunk_is_finite = bool(finite_mask.all().item())
        del finite_mask
        if not chunk_is_finite:
            return False
    return True


def select_panel(
    items: Sequence[MultihopItem],
) -> tuple[MultihopItem, ...]:
    """Select the complete registered panel in its fixed display order."""
    indexed: dict[str, MultihopItem] = {}
    for item in items:
        if item.name in indexed:
            raise ValueError(f"duplicate observational item {item.name}")
        indexed[item.name] = item
    missing = [name for name in PANEL_NAMES if name not in indexed]
    if missing:
        raise ValueError(f"missing registered panel items: {missing}")
    return tuple(indexed[name] for name in PANEL_NAMES)


def _unique_layers(layers: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(layer) for layer in layers))


@torch.no_grad()
def read_final_activations(
    model: Any,
    items: Sequence[MultihopItem],
    layers: Sequence[int],
    audit: Sequence[dict[str, object]],
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture each prompt's final-position residuals and move them to CPU."""
    from jspace.hooks import ActivationRecorder

    selected_layers = _unique_layers(layers)
    audited = {
        entry.get("name"): entry
        for entry in audit
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if len(audited) != len(audit):
        raise ValueError("prompt tokenization audit names are invalid")
    activations: dict[str, dict[int, torch.Tensor]] = {}
    for item in items:
        entry = audited.get(item.name)
        token_ids = None if entry is None else entry.get("prompt_token_ids")
        if (
            not isinstance(token_ids, (list, tuple))
            or not token_ids
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError(f"missing prompt token audit for {item.name}")
        input_ids = torch.tensor(
            [list(token_ids)],
            dtype=torch.long,
            device=model.input_device,
        )
        with ActivationRecorder(
            model.layers,
            at=selected_layers,
        ) as recorder:
            model.forward(input_ids)
        by_layer: dict[int, torch.Tensor] = {}
        for layer in selected_layers:
            activation = recorder.activations[layer]
            if (
                not isinstance(activation, torch.Tensor)
                or activation.ndim != 3
                or activation.shape[0] != 1
                or activation.shape[1] != len(token_ids)
                or activation.shape[2] <= 0
            ):
                raise ObservationReadoutError(
                    "captured activation has invalid shape"
                )
            final = activation[0, -1].detach().float().cpu()
            if not torch.isfinite(final).all().item():
                raise ObservationReadoutError(
                    "captured activation is non-finite"
                )
            by_layer[layer] = final
        activations[item.name] = by_layer
    return activations


def _direct_slots(
    record: dict[str, object],
    method: str,
    layers: tuple[int, ...],
    top: int,
) -> list[dict[str, object]]:
    top_tokens = record.get("top_tokens")
    if not isinstance(top_tokens, dict):
        raise ValueError(f"direct record {record.get('name')} has no top tokens")
    method_slots = top_tokens.get(method)
    if not isinstance(method_slots, (list, tuple)):
        raise ValueError(
            f"direct record {record.get('name')} has no {method} top tokens"
        )
    by_layer = {
        int(slot["layer"]): slot
        for slot in method_slots
        if isinstance(slot, dict) and "layer" in slot
    }
    missing = [layer for layer in layers if layer not in by_layer]
    if missing:
        raise ValueError(
            f"direct record {record.get('name')} lacks layers {missing}"
        )
    slots: list[dict[str, object]] = []
    for layer in layers:
        tokens = by_layer[layer].get("tokens")
        if not isinstance(tokens, (list, tuple)):
            raise ValueError(
                f"direct record {record.get('name')} has invalid tokens"
            )
        slots.append(
            {
                "layer": layer,
                "tokens": [
                    dict(token)
                    for token in tokens[:top]
                    if isinstance(token, dict)
                ],
            }
        )
    return slots


def index_direct_top_tokens(
    direct_records: Sequence[dict[str, object]],
    selected: Sequence[MultihopItem],
    layers: Sequence[int],
    top: int,
) -> dict[str, object]:
    """Copy precomputed direct top-token detail for the selected band."""
    if top < 0:
        raise ValueError("top must not be negative")
    indexed: dict[str, dict[str, object]] = {}
    for record in direct_records:
        name = record.get("name")
        if not isinstance(name, str) or name in indexed:
            raise ValueError("direct record names must be unique strings")
        indexed[name] = record
    selected_layers = _unique_layers(layers)
    output_items: list[dict[str, object]] = []
    for item in selected:
        record = indexed.get(item.name)
        if record is None:
            raise ValueError(f"missing direct record for {item.name}")
        output_items.append(
            {
                "name": item.name,
                "prompt": item.prompt,
                "target": item.target,
                "intermediates": deepcopy(record.get("intermediates", [])),
                "top_tokens": {
                    method: _direct_slots(
                        record,
                        method,
                        selected_layers,
                        top,
                    )
                    for method in ("jlens", "logit_lens")
                },
                "pursuit": [],
            }
        )
    return {"items": output_items}


def append_pursuit_slot(
    output: dict[str, object],
    item: MultihopItem,
    layer: int,
    pursuit: Any,
    dictionary: Any,
) -> None:
    """Append one JSON-ready pursuit result without mixing direct rankings."""
    output_items = output["items"]
    if not isinstance(output_items, list):
        raise ValueError("panel output items must be a list")
    output_item = next(
        candidate
        for candidate in output_items
        if candidate["name"] == item.name
    )
    try:
        indices = pursuit.indices
        coeffs = pursuit.coeffs
        variance = pursuit.var_explained
    except AttributeError as error:
        raise ObservationReadoutError(
            "pursuit output is missing required fields"
        ) from error
    if (
        not isinstance(indices, torch.Tensor)
        or not isinstance(coeffs, torch.Tensor)
        or indices.ndim != 1
        or coeffs.ndim != 1
        or indices.numel() != coeffs.numel()
        or indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or not coeffs.dtype.is_floating_point
        or not torch.isfinite(coeffs).all().item()
        or (indices < 0).any().item()
    ):
        raise ObservationReadoutError(
            "pursuit indices or coefficients are invalid"
        )
    if isinstance(variance, torch.Tensor):
        if variance.numel() != 1:
            raise ObservationReadoutError("pursuit variance has invalid shape")
        variance_value = float(variance.item())
    else:
        try:
            variance_value = float(variance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ObservationReadoutError(
                "pursuit variance is invalid"
            ) from error
    if not math.isfinite(variance_value):
        raise ObservationReadoutError("pursuit variance must be finite")
    order = torch.argsort(
        pursuit.coeffs,
        descending=True,
        stable=True,
    ).tolist()
    tokenizer = dictionary.model.tokenizer
    tokens = [
        {
            "token_id": int(pursuit.indices[index]),
            "token": str(
                tokenizer.decode([int(pursuit.indices[index])])
            ),
            "rank": rank,
            "coefficient": float(pursuit.coeffs[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]
    output_item["pursuit"].append(
        {
            "layer": int(layer),
            "tokens": tokens,
            "variance_explained": variance_value,
        }
    )


@torch.no_grad()
def read_qualitative_panel(
    model: Any,
    dictionary: Any,
    items: Sequence[MultihopItem],
    direct_records: Sequence[dict[str, object]],
    audit: Sequence[dict[str, object]],
    *,
    layers: Sequence[int],
    k: int = 25,
    top: int = 10,
    activation_reader: Callable[..., dict[str, dict[int, torch.Tensor]]]
    | None = None,
    pursuit_fn: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Read pursuit features while reusing precomputed direct lens detail."""
    selected = select_panel(items)
    selected_layers = _unique_layers(layers)
    reader = activation_reader or read_final_activations
    activations = reader(model, selected, selected_layers, audit)
    output = index_direct_top_tokens(
        direct_records,
        selected,
        selected_layers,
        top,
    )
    if pursuit_fn is None:
        from jspace import gradient_pursuit

        pursuit_fn = gradient_pursuit

    for layer in selected_layers:
        vectors = None
        try:
            vectors = dictionary.unit(layer)
            if (
                not isinstance(vectors, torch.Tensor)
                or vectors.ndim != 2
                or min(vectors.shape) <= 0
                or not vectors.dtype.is_floating_point
                or not _tensor_is_finite_bounded(vectors)
            ):
                raise ObservationReadoutError(
                    "dictionary vectors must be a finite matrix"
                )
            for item in selected:
                try:
                    hidden = activations[item.name].pop(layer).to(
                        vectors.device
                    )
                except (KeyError, AttributeError) as error:
                    raise ObservationReadoutError(
                        "panel activation output is incomplete"
                    ) from error
                if (
                    hidden.ndim != 1
                    or hidden.shape[0] != vectors.shape[1]
                    or not hidden.dtype.is_floating_point
                    or not torch.isfinite(hidden).all().item()
                ):
                    raise ObservationReadoutError(
                        "panel activation is malformed or non-finite"
                    )
                try:
                    pursuit = pursuit_fn(hidden, vectors, k=k)
                except torch.linalg.LinAlgError as error:
                    raise ObservationReadoutError(
                        "pursuit numerical decomposition failed"
                    ) from error
                if (
                    isinstance(getattr(pursuit, "indices", None), torch.Tensor)
                    and pursuit.indices.numel() > 0
                    and (pursuit.indices >= vectors.shape[0]).any().item()
                ):
                    raise ObservationReadoutError(
                        "pursuit token ID is outside dictionary vectors"
                    )
                append_pursuit_slot(
                    output,
                    item,
                    layer,
                    pursuit,
                    dictionary,
                )
        finally:
            dictionary.evict(layer)
            del vectors
    return output
