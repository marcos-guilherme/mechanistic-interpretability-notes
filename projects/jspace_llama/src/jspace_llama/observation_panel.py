"""Memory-bounded qualitative readout for the registered prompt panel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import torch

from jspace_llama.observation_data import MultihopItem, PANEL_NAMES


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
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture each prompt's final-position residuals and move them to CPU."""
    from jspace.hooks import ActivationRecorder

    selected_layers = _unique_layers(layers)
    activations: dict[str, dict[int, torch.Tensor]] = {}
    for item in items:
        input_ids = model.encode(item.prompt, max_length=512)
        with ActivationRecorder(
            model.layers,
            at=selected_layers,
        ) as recorder:
            model.forward(input_ids)
        activations[item.name] = {
            layer: recorder.activations[layer][0, -1]
            .detach()
            .float()
            .cpu()
            for layer in selected_layers
        }
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
            "variance_explained": float(pursuit.var_explained),
        }
    )


@torch.no_grad()
def read_qualitative_panel(
    model: Any,
    dictionary: Any,
    items: Sequence[MultihopItem],
    direct_records: Sequence[dict[str, object]],
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
    activations = reader(model, selected, selected_layers)
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
            for item in selected:
                hidden = activations[item.name].pop(layer).to(vectors.device)
                pursuit = pursuit_fn(hidden, vectors, k=k)
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
