"""Structured orchestration for the approved causal minimum."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from jspace_llama.config import ExperimentConfig
from jspace_llama.interventions import (
    MemoryBoundedLensSwap,
    NormMatchedRandomControl,
)
from jspace_llama.runtime import render_chat_prompt


CONDITIONS = ("baseline", "noop", "random", "target")


@dataclass(frozen=True)
class PreparedPrompt:
    name: str
    instruction: str
    rendered: str


@dataclass(frozen=True)
class GenerationRequest:
    prompt_name: str
    instruction: str
    rendered_prompt: str
    condition: str
    layers: tuple[int, ...]
    alpha: float
    seed: int
    source: str
    target: str
    max_new_tokens: int


def prepare_prompts(
    config: ExperimentConfig, tokenizer: Any
) -> tuple[PreparedPrompt, ...]:
    """Render each case once for reuse by every condition and sweep point."""
    return tuple(
        PreparedPrompt(
            name=case.name,
            instruction=case.text,
            rendered=render_chat_prompt(tokenizer, case.text),
        )
        for case in config.prompts.cases
    )


def _contains_evidence(text: str, numeral: str, word: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w)(?:{re.escape(numeral)}|{re.escape(word)})(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_basic_repetition(text: str) -> bool:
    tokens = re.findall(r"\b[\w'-]+\b", text.casefold())
    if len(tokens) < 4:
        return False
    most_common = max(tokens.count(token) for token in set(tokens))
    return most_common >= 3 and most_common / len(tokens) >= 0.6


def classify_continuation(continuation: str) -> dict[str, Any]:
    """Record evidence and obvious failure modes without judging success."""
    return {
        "answer_evidence": {
            "eight": _contains_evidence(continuation, "8", "eight"),
            "six": _contains_evidence(continuation, "6", "six"),
        },
        "target_entity_leakage": bool(
            re.search(r"(?<!\w)ant(?!\w)", continuation, flags=re.IGNORECASE)
        ),
        "degradation": {
            "empty": not continuation.strip(),
            "repetition": _has_basic_repetition(continuation),
        },
    }


class ModelExperimentRunner:
    """Translate generation requests into model calls and bounded hooks."""

    def __init__(self, model: Any, dictionary: Any) -> None:
        self.model = model
        self.dictionary = dictionary

    def __call__(self, request: GenerationRequest) -> str:
        hook = None
        if request.condition == "noop":
            hook = MemoryBoundedLensSwap(
                self.model,
                self.dictionary,
                request.source,
                request.source,
                layers=request.layers,
                alpha=request.alpha,
            )
        elif request.condition == "random":
            hook = NormMatchedRandomControl(
                self.model,
                self.dictionary,
                request.source,
                request.target,
                layers=request.layers,
                alpha=request.alpha,
                seed=request.seed,
            )
        elif request.condition == "target":
            hook = MemoryBoundedLensSwap(
                self.model,
                self.dictionary,
                request.source,
                request.target,
                layers=request.layers,
                alpha=request.alpha,
            )
        elif request.condition != "baseline":
            raise ValueError(f"unknown condition {request.condition!r}")
        return self.model.generate(
            request.rendered_prompt,
            max_new_tokens=request.max_new_tokens,
            hook=hook,
            first_line=False,
        )


def run_experiment(
    config: ExperimentConfig,
    tokenizer: Any,
    *,
    layers: Sequence[int],
    generate: Callable[[GenerationRequest], str],
    prepared_prompts: Sequence[PreparedPrompt] | None = None,
) -> dict[str, Any]:
    if config.generation.do_sample:
        raise ValueError("the approved experiment requires greedy generation")
    prepared = (
        tuple(prepared_prompts)
        if prepared_prompts is not None
        else prepare_prompts(config, tokenizer)
    )
    resolved_layers = tuple(int(layer) for layer in layers)
    observations: list[dict[str, Any]] = []
    for prompt in prepared:
        for alpha in config.sweep.alphas:
            for condition in CONDITIONS:
                request = GenerationRequest(
                    prompt_name=prompt.name,
                    instruction=prompt.instruction,
                    rendered_prompt=prompt.rendered,
                    condition=condition,
                    layers=resolved_layers,
                    alpha=float(alpha),
                    seed=config.seed,
                    source=config.source,
                    target=config.target,
                    max_new_tokens=config.generation.max_new_tokens,
                )
                continuation = generate(request)
                observations.append(
                    {
                        "prompt": {
                            "name": prompt.name,
                            "instruction": prompt.instruction,
                            "rendered": prompt.rendered,
                        },
                        "parameters": {
                            "layers": list(resolved_layers),
                            "alpha": float(alpha),
                            "seed": config.seed,
                        },
                        "condition": condition,
                        "intervention": {
                            "source": (
                                None if condition == "baseline" else config.source
                            ),
                            "target": {
                                "baseline": None,
                                "noop": config.source,
                                "random": "seeded_random_direction",
                                "target": config.target,
                            }[condition],
                            "prefill_only": condition != "baseline",
                        },
                        "continuation": continuation,
                        "classification": classify_continuation(continuation),
                    }
                )
    return {
        "provenance": {
            "assets": asdict(config.assets),
            "generation": asdict(config.generation),
            "seed": config.seed,
            "source": config.source,
            "target": config.target,
        },
        "observations": observations,
    }
