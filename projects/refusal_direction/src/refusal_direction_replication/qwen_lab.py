"""CPU-testable contracts for the Qwen intervention lab runtime and UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Literal

from .qwen_code_eval import CONDITIONS, COEFFICIENTS
from .qwen_dose import EXPECTED_SOURCE_LAYER, EXPECTED_SOURCE_OFFSET
from .qwen_jlens import (
    EXPECTED_D_MODEL,
    EXPECTED_SOURCE_LAYERS,
    EXPECTED_TARGET_LAYER,
    QWEN_MODEL,
    QWEN_REVISION,
)


LabCondition = Literal["baseline", "ablation", "addition", "random_control"]

LAB_MODEL_NAME = QWEN_MODEL
LAB_MODEL_REVISION = QWEN_REVISION
LAB_D_MODEL = EXPECTED_D_MODEL
LAB_JLENS_SOURCE_LAYERS = EXPECTED_SOURCE_LAYERS
LAB_JLENS_TARGET_LAYER = EXPECTED_TARGET_LAYER
LAB_DIRECTION_LAYER = EXPECTED_SOURCE_LAYER
LAB_DIRECTION_OFFSET = EXPECTED_SOURCE_OFFSET
LAB_MIN_JLENS_LAYER = 0
LAB_MAX_JLENS_LAYER = LAB_JLENS_SOURCE_LAYERS[-1]

MIN_TOP_K = 1
MAX_TOP_K = 20
DEFAULT_TOP_K = 10
MAX_NEW_TOKENS_LIMIT = 512


@dataclass(frozen=True)
class LabPreset:
    condition: LabCondition
    coefficient: float
    label: str


LAB_PRESETS: tuple[LabPreset, ...] = tuple(
    LabPreset(
        condition=condition,
        coefficient=float(COEFFICIENTS[condition]),
        label=condition.replace("_", " ").title(),
    )
    for condition in CONDITIONS
)


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    condition: LabCondition
    coefficient: float
    j_lens_layer: int
    top_k: int
    temperature: float
    top_p: float
    seed: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        validate_generation_request(self)


@dataclass(frozen=True)
class TokenPrediction:
    token_id: int
    display_text: str
    score: float
    relative_bar: float
    rank: int
    previous_rank_delta: int | None = None


@dataclass(frozen=True)
class GenerationEvent:
    step: int
    generated_token_id: int
    generated_token_text: str
    accumulated_text: str
    effective_condition: LabCondition
    effective_coefficient: float
    j_lens_layer: int
    predictions: tuple[TokenPrediction, ...]
    step_time_seconds: float
    gpu_memory_bytes: int | None
    status: str | None = None


def validate_generation_request(request: GenerationRequest) -> None:
    if not request.prompt.strip():
        raise ValueError("prompt must be non-empty")
    if not LAB_MIN_JLENS_LAYER <= int(request.j_lens_layer) <= LAB_MAX_JLENS_LAYER:
        raise ValueError(
            f"layer must be between {LAB_MIN_JLENS_LAYER} and {LAB_MAX_JLENS_LAYER}"
        )
    coefficient = float(request.coefficient)
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must be between 0 and 1")
    if request.condition == "baseline" and coefficient != 0.0:
        raise ValueError("baseline coefficient must remain zero")
    if not MIN_TOP_K <= int(request.top_k) <= MAX_TOP_K:
        raise ValueError(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
    temperature = float(request.temperature)
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    top_p = float(request.top_p)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if int(request.max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if int(request.max_new_tokens) > MAX_NEW_TOKENS_LIMIT:
        raise ValueError("max_new_tokens exceeds the configured limit")


def escape_token_display(*, token_id: int, text: str) -> str:
    del token_id  # deterministic in token id + text; id reserved for callers/tests
    if text == "":
        return "<empty>"
    parts: list[str] = []
    for character in text:
        if character == " ":
            parts.append("␠")
        elif character == "\t":
            parts.append("\\t")
        elif character == "\n":
            parts.append("\\n")
        elif character == "\r":
            parts.append("\\r")
        elif ord(character) < 32 or ord(character) == 127:
            parts.append(f"\\x{ord(character):02x}")
        else:
            parts.append(character)
    return "".join(parts)


def generation_request_to_dict(request: GenerationRequest) -> dict[str, Any]:
    return asdict(request)


def generation_request_from_dict(payload: Mapping[str, Any]) -> GenerationRequest:
    allowed = {field.name for field in fields(GenerationRequest)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return GenerationRequest(**filtered)


def token_prediction_to_dict(prediction: TokenPrediction) -> dict[str, Any]:
    return asdict(prediction)


def generation_event_to_dict(event: GenerationEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["predictions"] = [
        token_prediction_to_dict(prediction) for prediction in event.predictions
    ]
    return payload