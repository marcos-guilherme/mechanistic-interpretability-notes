from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Literal, Sequence

import torch


BABA_TEMPLATES = (
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
)


def _to_abba(template: str) -> str:
    """Swap A and B only in the introductory clause."""
    first_b = template.index("[B]")
    first_a = template.index("[A]", first_b)
    return (
        template[:first_b]
        + "[A]"
        + template[first_b + 3 : first_a]
        + "[B]"
        + template[first_a + 3 :]
    )


ABBA_TEMPLATES = tuple(_to_abba(template) for template in BABA_TEMPLATES)

HISTORICAL_NAMES = (
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
)

HISTORICAL_PLACES = (
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
)

HISTORICAL_OBJECTS = (
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
)


@dataclass(frozen=True)
class FilteredVocabulary:
    names: tuple[str, ...]
    places: tuple[str, ...]
    objects: tuple[str, ...]


@dataclass(frozen=True)
class PromptMetadata:
    template_id: int
    pattern: Literal["BABA", "ABBA"]
    io_name: str
    subject_name: str
    abc_io_name: str
    abc_s1_name: str
    abc_s2_name: str
    place: str
    object_name: str


@dataclass
class PairedIOIDataset:
    """Matched pIOI/pABC prompts with shared templates and token positions."""

    ioi_prompts: list[str]
    abc_prompts: list[str]
    ioi_tokens: torch.Tensor
    abc_tokens: torch.Tensor
    ioi_attention_mask: torch.Tensor
    abc_attention_mask: torch.Tensor
    positions: dict[str, torch.Tensor]
    io_token_ids: torch.Tensor
    s_token_ids: torch.Tensor
    abc_io_token_ids: torch.Tensor
    abc_s1_token_ids: torch.Tensor
    abc_s2_token_ids: torch.Tensor
    template_ids: torch.Tensor
    groups: dict[tuple[str, int], torch.Tensor]
    metadata: list[PromptMetadata]
    seed: int
    prepend_bos: bool = False

    def __len__(self) -> int:
        return len(self.ioi_prompts)

    @property
    def n(self) -> int:
        return len(self)

    @property
    def dataset_sha256(self) -> str:
        """Hash prompt content and grouping metadata for result provenance."""
        payload = {
            "ioi_prompts": self.ioi_prompts,
            "abc_prompts": self.abc_prompts,
            "groups": [
                [item.pattern, item.template_id] for item in self.metadata
            ],
            "prepend_bos": self.prepend_bos,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _single_token_items(
    tokenizer: Any,
    items: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        item
        for item in items
        if len(
            tokenizer.encode(
                " " + item,
                add_special_tokens=False,
            )
        )
        == 1
    )


def filter_single_token_vocabulary(
    tokenizer: Any,
    *,
    names: Sequence[str] = HISTORICAL_NAMES,
    places: Sequence[str] = HISTORICAL_PLACES,
    objects: Sequence[str] = HISTORICAL_OBJECTS,
) -> FilteredVocabulary:
    """Keep historical words represented by one GPT-2 token after a space."""
    return FilteredVocabulary(
        names=_single_token_items(tokenizer, names),
        places=_single_token_items(tokenizer, places),
        objects=_single_token_items(tokenizer, objects),
    )


def _fill_template(
    template: str,
    *,
    io_name: str,
    subject_name: str,
    place: str,
    object_name: str,
) -> str:
    return (
        template.replace("[A]", io_name)
        .replace("[B]", subject_name)
        .replace("[PLACE]", place)
        .replace("[OBJECT]", object_name)
    )


def _fill_abc_template(
    template: str,
    *,
    abc_io_name: str,
    abc_s1_name: str,
    abc_s2_name: str,
    place: str,
    object_name: str,
) -> str:
    """Apply historical IO→RAND, S→RAND, S1→RAND role replacements."""
    return (
        template.replace("[A]", abc_io_name)
        .replace("[B]", abc_s1_name, 1)
        .replace("[B]", abc_s2_name, 1)
        .replace("[PLACE]", place)
        .replace("[OBJECT]", object_name)
    )


def _tokenize(tokenizer: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("paper datasets require right padding")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError(
            "configure a pad token explicitly before building the dataset"
        )
    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encoded.input_ids.long(), encoded.attention_mask.long()


def _find_positions(
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    positions = {"IO": [], "S1": [], "S2": [], "S1+1": [], "END": []}
    for row in range(tokens.shape[0]):
        length = int(attention_mask[row].sum().item())
        row_tokens = tokens[row, :length]
        io_matches = (row_tokens == io_token_ids[row]).nonzero().flatten()
        s_matches = (row_tokens == s_token_ids[row]).nonzero().flatten()
        if len(io_matches) != 2 or len(s_matches) != 2:
            raise ValueError(
                "single-token names must occur exactly twice in every IOI prompt"
            )
        if row_tokens[-1] != io_token_ids[row]:
            raise ValueError("the final token must be the IO answer")
        s1 = int(s_matches[0].item())
        positions["IO"].append(int(io_matches[0].item()))
        positions["S1"].append(s1)
        positions["S2"].append(int(s_matches[1].item()))
        positions["S1+1"].append(s1 + 1)
        positions["END"].append(length - 2)
    return {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in positions.items()
    }


def build_paired_dataset(
    tokenizer: Any,
    *,
    n: int = 500,
    seed: int = 0,
    pattern: Literal["BABA", "ABBA", "mixed"] = "mixed",
) -> PairedIOIDataset:
    """Generate deterministic, token-aligned pIOI/pABC examples.

    pABC applies the historical IO→RAND, S→RAND, S1→RAND chain, yielding
    unrelated names at the IO, S1 and S2 roles while retaining all positions.
    The tokenizer must be explicitly configured for right padding with a pad
    token; this function never mutates tokenizer configuration.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if pattern not in {"BABA", "ABBA", "mixed"}:
        raise ValueError("pattern must be 'BABA', 'ABBA', or 'mixed'")

    vocabulary = filter_single_token_vocabulary(tokenizer)
    if len(vocabulary.names) < 5:
        raise ValueError("at least five single-token names are required")
    if not vocabulary.places:
        raise ValueError("at least one single-token place is required")
    if not vocabulary.objects:
        raise ValueError("at least one single-token object is required")

    rng = random.Random(seed)
    ioi_prompts: list[str] = []
    abc_prompts: list[str] = []
    metadata: list[PromptMetadata] = []

    for _ in range(n):
        template_id = rng.randrange(len(BABA_TEMPLATES))
        selected_pattern = (
            rng.choice(("BABA", "ABBA")) if pattern == "mixed" else pattern
        )
        templates = (
            BABA_TEMPLATES if selected_pattern == "BABA" else ABBA_TEMPLATES
        )
        template = templates[template_id]
        (
            io_name,
            subject_name,
            abc_io_name,
            abc_s1_name,
            abc_s2_name,
        ) = rng.sample(vocabulary.names, 5)
        place = rng.choice(vocabulary.places)
        object_name = rng.choice(vocabulary.objects)

        ioi_prompts.append(
            _fill_template(
                template,
                io_name=io_name,
                subject_name=subject_name,
                place=place,
                object_name=object_name,
            )
        )
        abc_prompts.append(
            _fill_abc_template(
                template,
                abc_io_name=abc_io_name,
                abc_s1_name=abc_s1_name,
                abc_s2_name=abc_s2_name,
                place=place,
                object_name=object_name,
            )
        )
        metadata.append(
            PromptMetadata(
                template_id=template_id,
                pattern=selected_pattern,
                io_name=io_name,
                subject_name=subject_name,
                abc_io_name=abc_io_name,
                abc_s1_name=abc_s1_name,
                abc_s2_name=abc_s2_name,
                place=place,
                object_name=object_name,
            )
        )

    ioi_tokens, ioi_attention_mask = _tokenize(tokenizer, ioi_prompts)
    abc_tokens, abc_attention_mask = _tokenize(tokenizer, abc_prompts)
    if not torch.equal(ioi_attention_mask, abc_attention_mask):
        raise ValueError("matched pIOI/pABC prompts must have equal token lengths")

    io_token_ids = torch.tensor(
        [
            tokenizer.encode(" " + item.io_name, add_special_tokens=False)[0]
            for item in metadata
        ],
        dtype=torch.long,
    )
    s_token_ids = torch.tensor(
        [
            tokenizer.encode(
                " " + item.subject_name,
                add_special_tokens=False,
            )[0]
            for item in metadata
        ],
        dtype=torch.long,
    )
    abc_io_token_ids = torch.tensor(
        [
            tokenizer.encode(
                " " + item.abc_io_name,
                add_special_tokens=False,
            )[0]
            for item in metadata
        ],
        dtype=torch.long,
    )
    abc_s1_token_ids = torch.tensor(
        [
            tokenizer.encode(
                " " + item.abc_s1_name,
                add_special_tokens=False,
            )[0]
            for item in metadata
        ],
        dtype=torch.long,
    )
    abc_s2_token_ids = torch.tensor(
        [
            tokenizer.encode(
                " " + item.abc_s2_name,
                add_special_tokens=False,
            )[0]
            for item in metadata
        ],
        dtype=torch.long,
    )
    positions = _find_positions(
        ioi_tokens,
        ioi_attention_mask,
        io_token_ids,
        s_token_ids,
    )
    rows = torch.arange(n)
    expected_abc_roles = {
        "IO": abc_io_token_ids,
        "S1": abc_s1_token_ids,
        "S2": abc_s2_token_ids,
    }
    for role, token_ids in expected_abc_roles.items():
        if not torch.equal(abc_tokens[rows, positions[role]], token_ids):
            raise ValueError(f"pABC {role} role must align with pIOI")
    if not torch.equal(
        abc_tokens[rows, positions["END"] + 1],
        abc_io_token_ids,
    ):
        raise ValueError("pABC final answer must repeat its IO role")

    template_ids = torch.tensor(
        [item.template_id for item in metadata],
        dtype=torch.long,
    )
    group_keys = [(item.pattern, item.template_id) for item in metadata]
    groups = {
        key: torch.tensor(
            [index for index, item_key in enumerate(group_keys) if item_key == key],
            dtype=torch.long,
        )
        for key in sorted(set(group_keys))
    }
    return PairedIOIDataset(
        ioi_prompts=ioi_prompts,
        abc_prompts=abc_prompts,
        ioi_tokens=ioi_tokens,
        abc_tokens=abc_tokens,
        ioi_attention_mask=ioi_attention_mask,
        abc_attention_mask=abc_attention_mask,
        positions=positions,
        io_token_ids=io_token_ids,
        s_token_ids=s_token_ids,
        abc_io_token_ids=abc_io_token_ids,
        abc_s1_token_ids=abc_s1_token_ids,
        abc_s2_token_ids=abc_s2_token_ids,
        template_ids=template_ids,
        groups=groups,
        metadata=metadata,
        seed=seed,
    )
