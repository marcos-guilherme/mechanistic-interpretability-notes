from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Literal, Sequence


Label = Literal["harmful", "harmless"]


@dataclass(frozen=True)
class PromptExample:
    id: str
    text: str
    label: Label
    source: str


@dataclass(frozen=True)
class PromptSplits:
    harmful_train: tuple[PromptExample, ...]
    harmful_val: tuple[PromptExample, ...]
    harmless_train: tuple[PromptExample, ...]
    harmless_val: tuple[PromptExample, ...]


def _split(
    examples: Sequence[PromptExample],
    train_size: int,
    val_size: int,
    rng: random.Random,
) -> tuple[tuple[PromptExample, ...], tuple[PromptExample, ...]]:
    ordered = list(examples)
    rng.shuffle(ordered)
    required = train_size + val_size
    if len(ordered) < required:
        raise ValueError(f"need {required} examples, received {len(ordered)}")
    return tuple(ordered[:train_size]), tuple(ordered[train_size:required])


def split_prompts(
    harmful: Sequence[PromptExample],
    harmless: Sequence[PromptExample],
    *,
    train_size: int,
    val_size: int,
    seed: int,
) -> PromptSplits:
    harmful_train, harmful_val = _split(
        harmful, train_size, val_size, random.Random(seed)
    )
    harmless_train, harmless_val = _split(
        harmless, train_size, val_size, random.Random(seed + 1)
    )
    return PromptSplits(
        harmful_train,
        harmful_val,
        harmless_train,
        harmless_val,
    )


def hash_examples(examples: Sequence[PromptExample]) -> str:
    payload = [example.__dict__ for example in examples]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def prompt_content_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt_text(text).encode("utf-8")).hexdigest()


def content_hashes_for_examples(examples: Sequence[PromptExample]) -> list[str]:
    return [prompt_content_hash(example.text) for example in examples]


def load_local_prompt_corpus(path: Path, label: Label) -> list[PromptExample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        PromptExample(
            id=str(row["id"]),
            text=str(row["text"]),
            label=label,
            source=str(row["source"]),
        )
        for row in rows
    ]
