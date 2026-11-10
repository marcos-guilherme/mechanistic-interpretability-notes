"""Matched adversarial IOI examples from the historical ``advex.py``.

The literal templates and insertion algorithm are ported from
``Easy-Transformer/easy_transformer/advex.py`` at commit
``ea15315dd24481e9e2ac5c3ef335d82907a1dc34``.  The historical experiment
sampled one of all 25 ordered pairs of natural clauses for every prompt and
inserted it immediately after the template's middle punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Literal

import torch

from ioi_replication.paper.dataset import (
    PromptMetadata,
    _tokenize,
    build_paired_dataset,
)
from ioi_replication.paper.metrics import io_probability, logit_difference


EASY_TRANSFORMER_SOURCE_COMMIT = (
    "ea15315dd24481e9e2ac5c3ef335d82907a1dc34"
)
EASY_TRANSFORMER_ADVEX_URL = (
    "https://github.com/redwoodresearch/Easy-Transformer/blob/"
    f"{EASY_TRANSFORMER_SOURCE_COMMIT}/easy_transformer/advex.py"
)
ADVERSARIAL_CLAUSE_TEMPLATES = (
    " [A] had a good day.",
    " [A] was enjoying the situation.",
    " [A] was tired.",
    " [A] enjoyed being with a friend.",
    " [A] was an enthusiast person.",
)
DOUBLE_ADVERSARIAL_TEMPLATES = tuple(
    first + second
    for first in ADVERSARIAL_CLAUSE_TEMPLATES
    for second in ADVERSARIAL_CLAUSE_TEMPLATES
)
PAPER_ADVERSARIAL_TARGETS = {
    "adversarial_logit_difference": 1.23,
    "adversarial_s_wins_error_rate": 0.234,
}
CI95_Z = 1.959963984540054


@dataclass(frozen=True)
class AdversarialPromptMetadata:
    base: PromptMetadata
    adversarial_template: str

    @property
    def io_name(self) -> str:
        return self.base.io_name

    @property
    def subject_name(self) -> str:
        return self.base.subject_name

    @property
    def pattern(self) -> Literal["BABA", "ABBA"]:
        return self.base.pattern

    @property
    def template_id(self) -> int:
        return self.base.template_id


@dataclass
class MatchedAdversarialDataset:
    """Matched base/adversarial prompts with independently computed positions."""

    base_prompts: list[str]
    adversarial_prompts: list[str]
    base_tokens: torch.Tensor
    adversarial_tokens: torch.Tensor
    base_attention_mask: torch.Tensor
    adversarial_attention_mask: torch.Tensor
    base_positions: dict[str, torch.Tensor]
    adversarial_positions: dict[str, torch.Tensor]
    extra_io_positions: torch.Tensor
    io_token_ids: torch.Tensor
    s_token_ids: torch.Tensor
    metadata: list[AdversarialPromptMetadata]
    seed: int
    prepend_bos: bool = False

    def __len__(self) -> int:
        return len(self.base_prompts)

    @property
    def n(self) -> int:
        return len(self)

    @property
    def dataset_sha256(self) -> str:
        payload = {
            "base_prompts": self.base_prompts,
            "adversarial_prompts": self.adversarial_prompts,
            "templates": [
                item.adversarial_template for item in self.metadata
            ],
            "seed": self.seed,
            "prepend_bos": self.prepend_bos,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class MeanEstimate:
    values: torch.Tensor
    mean: float
    standard_error: float
    ci95: tuple[float, float]


@dataclass(frozen=True)
class PairedEstimate:
    baseline: MeanEstimate
    adversarial: MeanEstimate
    delta: MeanEstimate


@dataclass(frozen=True)
class PairedAdversarialMetrics:
    logit_difference: PairedEstimate
    s_wins_error_rate: PairedEstimate
    io_probability: PairedEstimate


def insert_adversarial_clauses(
    prompt: str,
    *,
    io_name: str,
    template: str,
) -> str:
    """Insert a filled double clause at the historical middle punctuation."""
    if template.count("[A]") != 2:
        raise ValueError("adversarial template must contain exactly two [A] slots")
    punctuation_index = (
        prompt.rfind(".") if "." in prompt else prompt.rfind(",")
    )
    if punctuation_index < 0:
        raise ValueError("base prompt must contain middle punctuation")
    clauses = template.replace("[A]", io_name)
    return (
        prompt[: punctuation_index + 1]
        + clauses
        + prompt[punctuation_index + 1 :]
    )


def _find_adversarial_positions(
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    positions = {"IO": [], "S1": [], "S2": [], "S1+1": [], "END": []}
    extras: list[list[int]] = []
    for row in range(tokens.shape[0]):
        length = int(attention_mask[row].sum().item())
        row_tokens = tokens[row, :length]
        io_matches = (row_tokens == io_token_ids[row]).nonzero().flatten()
        s_matches = (row_tokens == s_token_ids[row]).nonzero().flatten()
        if len(io_matches) != 4:
            raise ValueError(
                "the IO name must occur twice in the base prompt and twice "
                "in the inserted clauses"
            )
        if len(s_matches) != 2:
            raise ValueError(
                "the subject name must occur exactly twice in every prompt"
            )
        if row_tokens[-1] != io_token_ids[row]:
            raise ValueError("the final token must be the IO answer")
        io = int(io_matches[0].item())
        s1 = int(s_matches[0].item())
        s2 = int(s_matches[1].item())
        row_extras = [
            int(io_matches[1].item()),
            int(io_matches[2].item()),
        ]
        if not (io < row_extras[0] < row_extras[1] < s2):
            raise ValueError(
                "inserted IO occurrences must be between the introductory "
                "IO and S2"
            )
        positions["IO"].append(io)
        positions["S1"].append(s1)
        positions["S2"].append(s2)
        positions["S1+1"].append(s1 + 1)
        positions["END"].append(length - 2)
        extras.append(row_extras)
    return (
        {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in positions.items()
        },
        torch.tensor(extras, dtype=torch.long),
    )


def build_adversarial_dataset(
    tokenizer: Any,
    *,
    n: int = 500,
    seed: int = 0,
    pattern: Literal["BABA", "ABBA", "mixed"] = "mixed",
) -> MatchedAdversarialDataset:
    """Build deterministic matched IOI and historical adversarial prompts.

    The base IOI examples use the project's paper-faithful generator.  A
    dedicated seeded RNG then reproduces ``rd.choice(DOUBLE_ADX_TEMPLATE)`` for
    each row.  This makes the originally unseeded attack sampling reproducible
    without changing its uniform distribution or ordered-pair support.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    base = build_paired_dataset(
        tokenizer,
        n=n,
        seed=seed,
        pattern=pattern,
    )
    rng = random.Random(seed)
    metadata: list[AdversarialPromptMetadata] = []
    adversarial_prompts: list[str] = []
    for prompt, base_metadata in zip(base.ioi_prompts, base.metadata):
        adversarial_template = rng.choice(DOUBLE_ADVERSARIAL_TEMPLATES)
        adversarial_prompts.append(
            insert_adversarial_clauses(
                prompt,
                io_name=base_metadata.io_name,
                template=adversarial_template,
            )
        )
        metadata.append(
            AdversarialPromptMetadata(
                base=base_metadata,
                adversarial_template=adversarial_template,
            )
        )

    adversarial_tokens, adversarial_attention_mask = _tokenize(
        tokenizer,
        adversarial_prompts,
    )
    adversarial_positions, extra_io_positions = (
        _find_adversarial_positions(
            adversarial_tokens,
            adversarial_attention_mask,
            base.io_token_ids,
            base.s_token_ids,
        )
    )
    rows = torch.arange(n)
    if not torch.equal(
        adversarial_tokens[
            rows,
            adversarial_positions["END"] + 1,
        ],
        base.io_token_ids,
    ):
        raise ValueError("adversarial final answer must repeat its IO role")

    return MatchedAdversarialDataset(
        base_prompts=base.ioi_prompts,
        adversarial_prompts=adversarial_prompts,
        base_tokens=base.ioi_tokens,
        adversarial_tokens=adversarial_tokens,
        base_attention_mask=base.ioi_attention_mask,
        adversarial_attention_mask=adversarial_attention_mask,
        base_positions=base.positions,
        adversarial_positions=adversarial_positions,
        extra_io_positions=extra_io_positions,
        io_token_ids=base.io_token_ids,
        s_token_ids=base.s_token_ids,
        metadata=metadata,
        seed=seed,
    )


def _estimate(values: torch.Tensor) -> MeanEstimate:
    values = torch.as_tensor(values).detach().float().cpu()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("metric values must be a non-empty vector")
    if not torch.isfinite(values).all():
        raise ValueError("metric values must be finite")
    mean = float(values.mean().item())
    standard_error = (
        float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
        if values.numel() > 1
        else 0.0
    )
    return MeanEstimate(
        values=values,
        mean=mean,
        standard_error=standard_error,
        ci95=(
            mean - CI95_Z * standard_error,
            mean + CI95_Z * standard_error,
        ),
    )


def _paired(
    baseline: torch.Tensor,
    adversarial: torch.Tensor,
) -> PairedEstimate:
    baseline = torch.as_tensor(baseline).detach().float().cpu()
    adversarial = torch.as_tensor(adversarial).detach().float().cpu()
    if baseline.shape != adversarial.shape:
        raise ValueError("paired metric vectors must have equal shapes")
    return PairedEstimate(
        baseline=_estimate(baseline),
        adversarial=_estimate(adversarial),
        delta=_estimate(adversarial - baseline),
    )


def summarize_adversarial_values(
    *,
    baseline_logit_difference: torch.Tensor,
    adversarial_logit_difference: torch.Tensor,
    baseline_io_probability: torch.Tensor,
    adversarial_io_probability: torch.Tensor,
) -> PairedAdversarialMetrics:
    """Summarize paired values, including paired delta SEs and 95% CIs."""
    baseline_logit_difference = torch.as_tensor(
        baseline_logit_difference
    ).float()
    adversarial_logit_difference = torch.as_tensor(
        adversarial_logit_difference
    ).float()
    return PairedAdversarialMetrics(
        logit_difference=_paired(
            baseline_logit_difference,
            adversarial_logit_difference,
        ),
        s_wins_error_rate=_paired(
            (baseline_logit_difference < 0).float(),
            (adversarial_logit_difference < 0).float(),
        ),
        io_probability=_paired(
            baseline_io_probability,
            adversarial_io_probability,
        ),
    )


def paired_adversarial_metrics(
    baseline_logits: torch.Tensor,
    adversarial_logits: torch.Tensor,
    dataset: MatchedAdversarialDataset,
) -> PairedAdversarialMetrics:
    """Compute paired behavioral metrics at each prompt's own semantic END."""
    if baseline_logits.shape[0] != len(dataset):
        raise ValueError("baseline logits must have one row per example")
    if adversarial_logits.shape[0] != len(dataset):
        raise ValueError("adversarial logits must have one row per example")
    baseline_logit_difference = logit_difference(
        baseline_logits,
        dataset.io_token_ids,
        dataset.s_token_ids,
        dataset.base_positions["END"],
        reduction="none",
    )
    adversarial_logit_difference = logit_difference(
        adversarial_logits,
        dataset.io_token_ids,
        dataset.s_token_ids,
        dataset.adversarial_positions["END"],
        reduction="none",
    )
    baseline_io_probability = io_probability(
        baseline_logits,
        dataset.io_token_ids,
        dataset.base_positions["END"],
        reduction="none",
    )
    adversarial_io_probability = io_probability(
        adversarial_logits,
        dataset.io_token_ids,
        dataset.adversarial_positions["END"],
        reduction="none",
    )
    return summarize_adversarial_values(
        baseline_logit_difference=baseline_logit_difference,
        adversarial_logit_difference=adversarial_logit_difference,
        baseline_io_probability=baseline_io_probability,
        adversarial_io_probability=adversarial_io_probability,
    )
