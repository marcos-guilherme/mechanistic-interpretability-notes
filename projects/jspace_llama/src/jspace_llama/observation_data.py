"""Validation and tokenization audit for the observational multihop dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from jspace_llama.jsonio import strict_json_loads


DATASET_SHA256 = (
    "2cd5a47aa3d7627708843bc61f03ab1d"
    "10eda4184f357c5cbf93c97184e32c9c"
)
EXPECTED_ITEM_COUNT = 93


class DatasetValidationError(ValueError):
    pass


PANEL_NAMES = (
    "carnival-ocean",
    "amazon-language",
    "mars-color",
    "spider-legs",
    "basketball-players",
    "atomic-79-symbol",
    "planet-3-moons",
    "rhyme-rain-neighbor",
    "etym-wargod-month",
    "chem-photosynthesis-Z",
    "nhop-primary-planet",
    "inv-sunrise-opposite",
)


@dataclass(frozen=True)
class MultihopItem:
    name: str
    prompt: str
    target: str
    intermediates: tuple[str, ...]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    canonical = "[" + ",".join(str(token_id) for token_id in token_ids) + "]"
    return _text_sha256(canonical)


def load_multihop_items(path: Path) -> tuple[MultihopItem, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise DatasetValidationError("multihop dataset SHA-256 mismatch")
    document = strict_json_loads(payload)
    if not isinstance(document, dict) or set(document) != {"items"}:
        raise DatasetValidationError(
            "multihop dataset root must contain only items"
        )
    raw_items = document["items"]
    if not isinstance(raw_items, list) or len(raw_items) != EXPECTED_ITEM_COUNT:
        raise DatasetValidationError("multihop dataset must contain 93 items")

    items: list[MultihopItem] = []
    names: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "prompt",
            "target",
            "intermediates",
        }:
            raise DatasetValidationError("multihop item has unexpected fields")
        if not all(
            isinstance(raw[field], str) and bool(raw[field])
            for field in ("name", "prompt", "target")
        ):
            raise DatasetValidationError(
                "multihop name, prompt, and target must be non-empty strings"
            )
        if raw["name"] in names:
            raise DatasetValidationError("multihop item name is duplicated")
        if not raw["prompt"].endswith((" ", ' "')):
            raise DatasetValidationError(
                "multihop prompt must retain final space"
            )
        if (
            not isinstance(raw["intermediates"], list)
            or not raw["intermediates"]
            or not all(
                isinstance(value, str) and value
                for value in raw["intermediates"]
            )
        ):
            raise DatasetValidationError(
                "intermediates must be non-empty strings"
            )
        names.add(raw["name"])
        items.append(
            MultihopItem(
                name=raw["name"],
                prompt=raw["prompt"],
                target=raw["target"],
                intermediates=tuple(raw["intermediates"]),
            )
        )
    return tuple(items)


def canonical_first_token(
    tokenizer: Any,
    text: str,
) -> tuple[int, tuple[int, ...], str]:
    variants = (
        ("leading_space", " " + text.lstrip()),
        ("raw", text),
    )
    for variant, prompt in variants:
        token_ids = tuple(
            int(token_id)
            for token_id in tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )
        if token_ids:
            return token_ids[0], token_ids, variant
    raise ValueError("tokenization produced no token IDs")


def audit_tokenization(
    items: tuple[MultihopItem, ...],
    tokenizer: Any,
) -> tuple[dict[str, object], ...]:
    audit: list[dict[str, object]] = []
    for item in items:
        intermediates: list[dict[str, object]] = []
        for text in item.intermediates:
            token_id, token_ids, variant = canonical_first_token(
                tokenizer,
                text,
            )
            prompt = " " + text.lstrip() if variant == "leading_space" else text
            intermediates.append(
                {
                    "text": text,
                    "token_id": token_id,
                    "token_ids": token_ids,
                    "variant": variant,
                    "single_token": len(token_ids) == 1,
                    "prompt_sha256": _text_sha256(prompt),
                    "token_ids_sha256": _token_ids_sha256(token_ids),
                }
            )
        audit.append(
            {
                "name": item.name,
                "intermediates": intermediates,
            }
        )
    return tuple(audit)
