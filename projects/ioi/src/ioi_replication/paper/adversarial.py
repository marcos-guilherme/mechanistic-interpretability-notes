"""Matched Figure 8 and historical ``advex.py`` IOI examples.

The literal templates are shared by Appendix L and
``Easy-Transformer/easy_transformer/advex.py`` at commit
``ea15315dd24481e9e2ac5c3ef335d82907a1dc34``. Figure 8 inserts one of five
clauses; ``advex.py`` samples one of their 25 ordered pairs. Both insert at
the template's middle punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Literal, Sequence

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
PAPER_FIGURE_8_TARGETS = {
    "baseline": {
        "logit_difference": 3.55,
        "io_probability": 0.49,
        "s_wins_error_rate": 0.007,
    },
    "paper_s_single": {
        "logit_difference": 3.64,
        "io_probability": 0.59,
        "s_wins_error_rate": 0.004,
    },
    "paper_io_single": {
        "logit_difference": 1.23,
        "io_probability": 0.36,
        "s_wins_error_rate": 0.234,
    },
}
# Backward-compatible projection of the Figure 8 paper-IO-single row.
PAPER_ADVERSARIAL_TARGETS = {
    "adversarial_logit_difference": 1.23,
    "adversarial_s_wins_error_rate": 0.234,
}
CI95_Z = 1.959963984540054


@dataclass(frozen=True)
class AdversarialPromptMetadata:
    base: PromptMetadata
    adversarial_template: str
    attack_role: Literal["IO", "S"]
    clause_count: Literal["single", "double"]

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
    inserted_positions: torch.Tensor
    io_token_ids: torch.Tensor
    s_token_ids: torch.Tensor
    template_ids: torch.Tensor
    metadata: list[AdversarialPromptMetadata]
    base_seed: int
    attack_seed: int
    attack_role: Literal["IO", "S"]
    clause_count: Literal["single", "double"]
    allowed_template_ids: tuple[int, ...]
    prepend_bos: bool = False

    def __len__(self) -> int:
        return len(self.base_prompts)

    @property
    def n(self) -> int:
        return len(self)

    @property
    def seed(self) -> int:
        """Backward-compatible alias for the base generation seed."""
        return self.base_seed

    @property
    def extra_io_positions(self) -> torch.Tensor:
        """Backward-compatible IO insertion positions."""
        if self.attack_role == "IO":
            return self.inserted_positions
        return torch.empty((len(self), 0), dtype=torch.long)

    @property
    def extra_s_positions(self) -> torch.Tensor:
        if self.attack_role == "S":
            return self.inserted_positions
        return torch.empty((len(self), 0), dtype=torch.long)

    @property
    def dataset_sha256(self) -> str:
        payload = {
            "base_prompts": self.base_prompts,
            "adversarial_prompts": self.adversarial_prompts,
            "templates": [
                item.adversarial_template for item in self.metadata
            ],
            "base_seed": self.base_seed,
            "attack_seed": self.attack_seed,
            "attack_role": self.attack_role,
            "clause_count": self.clause_count,
            "allowed_template_ids": self.allowed_template_ids,
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
    ci_method: str


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
    io_name: str | None = None,
    role_name: str | None = None,
    template: str,
) -> str:
    """Insert one or two filled clauses at the historical middle punctuation."""
    if (io_name is None) == (role_name is None):
        raise ValueError("provide exactly one of io_name or role_name")
    if template.count("[A]") not in {1, 2}:
        raise ValueError(
            "adversarial template must contain one or two [A] slots"
        )
    punctuation_index = (
        prompt.rfind(".") if "." in prompt else prompt.rfind(",")
    )
    if punctuation_index < 0:
        raise ValueError("base prompt must contain middle punctuation")
    clauses = template.replace(
        "[A]",
        io_name if io_name is not None else role_name,
    )
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
    *,
    attack_role: Literal["IO", "S"],
    inserted_count: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    positions = {"IO": [], "S1": [], "S2": [], "S1+1": [], "END": []}
    extras: list[list[int]] = []
    for row in range(tokens.shape[0]):
        length = int(attention_mask[row].sum().item())
        row_tokens = tokens[row, :length]
        io_matches = (row_tokens == io_token_ids[row]).nonzero().flatten()
        s_matches = (row_tokens == s_token_ids[row]).nonzero().flatten()
        expected_io_count = 2 + (
            inserted_count if attack_role == "IO" else 0
        )
        expected_s_count = 2 + (
            inserted_count if attack_role == "S" else 0
        )
        if len(io_matches) != expected_io_count:
            raise ValueError(
                f"the IO name must occur exactly {expected_io_count} times"
            )
        if len(s_matches) != expected_s_count:
            raise ValueError(
                f"the subject name must occur exactly {expected_s_count} times"
            )
        if row_tokens[-1] != io_token_ids[row]:
            raise ValueError("the final token must be the IO answer")
        io = int(io_matches[0].item())
        s1 = int(s_matches[0].item())
        s2 = int(s_matches[-1].item())
        attacked_matches = io_matches if attack_role == "IO" else s_matches
        row_extras = [
            int(position.item()) for position in attacked_matches[1:-1]
        ]
        if len(row_extras) != inserted_count or not all(
            max(io, s1) < position < s2 for position in row_extras
        ):
            raise ValueError(
                "inserted role occurrences must be distinct from IO/S1/S2 "
                "and lie before S2"
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
    seed: int | None = None,
    base_seed: int = 0,
    attack_seed: int | None = None,
    role: Literal["IO", "S"] = "IO",
    clause_count: Literal["single", "double"] = "double",
    pattern: Literal["BABA", "ABBA", "mixed"] = "mixed",
    allowed_template_ids: Sequence[int] | None = None,
) -> MatchedAdversarialDataset:
    """Build deterministic matched base and role-duplication prompts.

    ``single`` samples one of the five Figure 8 clauses. ``double`` samples one
    of the 25 ordered pairs from historical ``advex.py``. Base prompt sampling
    and clause sampling use independent RNGs and distinct default seeds.
    ``seed`` remains as a legacy alias which explicitly sets both seeds.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if role not in {"IO", "S"}:
        raise ValueError("role must be 'IO' or 'S'")
    if clause_count not in {"single", "double"}:
        raise ValueError("clause_count must be 'single' or 'double'")
    if seed is not None:
        if base_seed != 0 or attack_seed is not None:
            raise ValueError(
                "seed cannot be combined with base_seed or attack_seed"
            )
        base_seed = seed
        attack_seed = seed
    elif attack_seed is None:
        attack_seed = 1
    selected_template_ids = (
        tuple(range(15))
        if allowed_template_ids is None
        else tuple(allowed_template_ids)
    )
    base = build_paired_dataset(
        tokenizer,
        n=n,
        seed=base_seed,
        pattern=pattern,
        allowed_template_ids=selected_template_ids,
    )
    attack_templates = (
        ADVERSARIAL_CLAUSE_TEMPLATES
        if clause_count == "single"
        else DOUBLE_ADVERSARIAL_TEMPLATES
    )
    inserted_count = 1 if clause_count == "single" else 2
    rng = random.Random(attack_seed)
    metadata: list[AdversarialPromptMetadata] = []
    adversarial_prompts: list[str] = []
    for prompt, base_metadata in zip(base.ioi_prompts, base.metadata):
        adversarial_template = rng.choice(attack_templates)
        role_name = (
            base_metadata.io_name
            if role == "IO"
            else base_metadata.subject_name
        )
        adversarial_prompts.append(
            insert_adversarial_clauses(
                prompt,
                role_name=role_name,
                template=adversarial_template,
            )
        )
        metadata.append(
            AdversarialPromptMetadata(
                base=base_metadata,
                adversarial_template=adversarial_template,
                attack_role=role,
                clause_count=clause_count,
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
            attack_role=role,
            inserted_count=inserted_count,
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
        inserted_positions=extra_io_positions,
        io_token_ids=base.io_token_ids,
        s_token_ids=base.s_token_ids,
        template_ids=base.template_ids,
        metadata=metadata,
        base_seed=base_seed,
        attack_seed=attack_seed,
        attack_role=role,
        clause_count=clause_count,
        allowed_template_ids=selected_template_ids,
    )


def _estimate(
    values: torch.Tensor,
    *,
    ci_method: str = "normal_1.96_se",
) -> MeanEstimate:
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
        ci_method=ci_method,
    )


def _proportion_estimate(values: torch.Tensor) -> MeanEstimate:
    estimate = _estimate(values, ci_method="wilson")
    n = estimate.values.numel()
    successes = float(estimate.values.sum().item())
    denominator = 1 + CI95_Z**2 / n
    center = (
        successes / n + CI95_Z**2 / (2 * n)
    ) / denominator
    half_width = (
        CI95_Z
        * math.sqrt(
            (successes / n) * (1 - successes / n) / n
            + CI95_Z**2 / (4 * n**2)
        )
        / denominator
    )
    return MeanEstimate(
        values=estimate.values,
        mean=estimate.mean,
        standard_error=estimate.standard_error,
        ci95=(center - half_width, center + half_width),
        ci_method="wilson",
    )


def _paired(
    baseline: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    proportion: bool = False,
) -> PairedEstimate:
    baseline = torch.as_tensor(baseline).detach().float().cpu()
    adversarial = torch.as_tensor(adversarial).detach().float().cpu()
    if baseline.shape != adversarial.shape:
        raise ValueError("paired metric vectors must have equal shapes")
    arm_estimate = _proportion_estimate if proportion else _estimate
    return PairedEstimate(
        baseline=arm_estimate(baseline),
        adversarial=arm_estimate(adversarial),
        delta=_estimate(
            adversarial - baseline,
            ci_method="normal_paired_1.96_se",
        ),
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
            proportion=True,
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
