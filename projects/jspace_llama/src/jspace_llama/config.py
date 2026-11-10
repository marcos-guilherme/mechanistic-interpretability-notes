"""Pinned configuration for the approved J-space causal experiment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetsConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    lens_repo_id: str = "neuronpedia/jacobian-lens"
    lens_filename: str = (
        "llama3.1-8b-it/jlens/Salesforce-wikitext/"
        "Llama-3.1-8B-Instruct_jacobian_lens.pt"
    )
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 12
    do_sample: bool = False


@dataclass(frozen=True)
class PromptCase:
    name: str
    text: str


def _default_prompt_cases() -> tuple[PromptCase, ...]:
    return (
        PromptCase(
            name="main",
            text="The animal that spins webs has this many legs:",
        ),
        PromptCase(
            name="paraphrase_animal",
            text="An animal that spins webs has how many legs?",
        ),
        PromptCase(
            name="paraphrase_creature",
            text="How many legs does the creature that spins webs have?",
        ),
    )


@dataclass(frozen=True)
class PromptConfig:
    cases: tuple[PromptCase, ...] = field(default_factory=_default_prompt_cases)


@dataclass(frozen=True)
class SweepConfig:
    layers: tuple[int, ...] | None = None
    alphas: tuple[float, ...] = (4.0,)


@dataclass(frozen=True)
class ExperimentConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    seed: int = 42
    source: str = "spider"
    target: str = "ant"
