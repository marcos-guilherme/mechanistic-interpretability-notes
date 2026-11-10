"""Memory-bounded J-space workspace readout."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


class ReadoutError(RuntimeError):
    pass


def _read_activations(
    model: Any,
    prompt: str,
    layers: Sequence[int],
    max_seq_len: int,
) -> tuple[list[int], dict[int, torch.Tensor]]:
    from jspace.hooks import ActivationRecorder

    input_ids = model.encode(prompt, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=sorted(set(layers))) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in set(layers)
        }
    return input_ids[0].tolist(), activations


def _concept_summary(
    slots: list[dict[str, Any]], concept: str
) -> dict[str, Any]:
    peaks: dict[str, float] = {}
    for slot in slots:
        for entry in slot["entries"]:
            normalized = entry["token"].strip().casefold()
            peaks[normalized] = max(
                peaks.get(normalized, 0.0), entry["coefficient"]
            )
    ranked = sorted(peaks.items(), key=lambda item: item[1], reverse=True)
    wanted = concept.strip().casefold()
    rank = next(
        (index for index, (token, _) in enumerate(ranked, start=1) if token == wanted),
        None,
    )
    return {
        "concept": concept,
        "present": rank is not None,
        "rank": rank,
        "peak_coefficient": peaks.get(wanted),
    }


@torch.no_grad()
def read_workspace(
    model: Any,
    dictionary: Any,
    prompt: str,
    *,
    layers: Sequence[int],
    positions: Sequence[int] = (-1,),
    k: int = 25,
    max_seq_len: int = 512,
    activation_reader: Callable[..., tuple[list[int], dict[int, torch.Tensor]]]
    | None = None,
    pursuit_fn: Callable[..., Any] | None = None,
    focus_concept: str = "spider",
) -> dict[str, Any]:
    """Read selected layers one at a time and evict each full dictionary."""
    if activation_reader is None:
        activation_reader = _read_activations
    if pursuit_fn is None:
        from jspace import gradient_pursuit

        pursuit_fn = gradient_pursuit

    sequence, activations = activation_reader(
        model, prompt, list(layers), max_seq_len
    )
    tokenizer = dictionary.model.tokenizer
    slots: list[dict[str, Any]] = []
    for layer in layers:
        try:
            vectors = dictionary.unit(layer)
            for position in positions:
                hidden = activations[layer][0, position].float().to(vectors.device)
                result = pursuit_fn(hidden, vectors, k=k)
                order = torch.argsort(result.coeffs, descending=True).tolist()
                entries = [
                    {
                        "token_id": int(result.indices[index]),
                        "token": tokenizer.decode([int(result.indices[index])]),
                        "coefficient": float(result.coeffs[index]),
                    }
                    for index in order
                ]
                slots.append(
                    {
                        "layer": int(layer),
                        "position": int(position),
                        "position_token": tokenizer.decode([sequence[position]]),
                        "entries": entries,
                        "variance_explained": float(result.var_explained),
                    }
                )
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            raise ReadoutError(
                f"workspace readout failed at layer {layer}: {exc}"
            ) from exc
        finally:
            dictionary.evict(layer)
    return {
        "slots": slots,
        "spider": _concept_summary(slots, focus_concept),
    }
