from __future__ import annotations

import json
import math

import pytest

from refusal_direction_replication.qwen_code_eval import COEFFICIENTS, CONDITIONS
from refusal_direction_replication.qwen_dose import EXPECTED_SOURCE_LAYER, EXPECTED_SOURCE_OFFSET
from refusal_direction_replication.qwen_jlens import (
    EXPECTED_D_MODEL,
    EXPECTED_SOURCE_LAYERS,
    EXPECTED_TARGET_LAYER,
    QWEN_MODEL,
    QWEN_REVISION,
)
from refusal_direction_replication.qwen_lab import (
    DEFAULT_TOP_K,
    LAB_MAX_JLENS_LAYER,
    LAB_MIN_JLENS_LAYER,
    LAB_PRESETS,
    MAX_NEW_TOKENS_LIMIT,
    MAX_TOP_K,
    MIN_TOP_K,
    GenerationEvent,
    GenerationRequest,
    LabCondition,
    LabPreset,
    escape_token_display,
    generation_event_to_dict,
    generation_request_from_dict,
    generation_request_to_dict,
    token_prediction_to_dict,
    validate_generation_request,
)


def _valid_request(**overrides) -> GenerationRequest:
    base = dict(
        prompt="hello",
        condition="baseline",
        coefficient=0.0,
        j_lens_layer=12,
        top_k=DEFAULT_TOP_K,
        temperature=1.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=32,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def test_lab_presets_match_study_coefficients_in_order() -> None:
    assert tuple(p.condition for p in LAB_PRESETS) == CONDITIONS
    assert [p.coefficient for p in LAB_PRESETS] == [
        COEFFICIENTS[name] for name in CONDITIONS
    ]
    assert [p.coefficient for p in LAB_PRESETS] == [0.0, 0.2, 0.5, 0.2]


def test_frozen_scientific_constants_are_reexported() -> None:
    from refusal_direction_replication import qwen_lab as module

    assert module.LAB_MODEL_NAME == QWEN_MODEL
    assert module.LAB_MODEL_REVISION == QWEN_REVISION
    assert module.LAB_D_MODEL == EXPECTED_D_MODEL == 3584
    assert module.LAB_JLENS_SOURCE_LAYERS == EXPECTED_SOURCE_LAYERS == list(range(27))
    assert module.LAB_JLENS_TARGET_LAYER == EXPECTED_TARGET_LAYER == 27
    assert module.LAB_DIRECTION_LAYER == EXPECTED_SOURCE_LAYER == 18
    assert module.LAB_DIRECTION_OFFSET == EXPECTED_SOURCE_OFFSET == -1
    assert LAB_MIN_JLENS_LAYER == 0
    assert LAB_MAX_JLENS_LAYER == 26


def test_escape_token_display_reveals_whitespace_and_controls() -> None:
    assert escape_token_display(token_id=1, text="") == "<empty>"
    assert escape_token_display(token_id=2, text=" ") == "␠"
    assert escape_token_display(token_id=3, text="a\tb") == "a\\tb"
    assert escape_token_display(token_id=4, text="a\nb") == "a\\nb"
    assert escape_token_display(token_id=5, text="a\rb") == "a\\rb"
    assert escape_token_display(token_id=6, text="\x07") == "\\x07"


@pytest.mark.parametrize(
    "field, value, pattern",
    [
        ("prompt", "", "prompt"),
        ("prompt", "   ", "prompt"),
        ("j_lens_layer", -1, "layer"),
        ("j_lens_layer", 27, "layer"),
        ("coefficient", -0.01, "coefficient"),
        ("coefficient", 1.01, "coefficient"),
        ("top_k", MIN_TOP_K - 1, "top_k"),
        ("top_k", MAX_TOP_K + 1, "top_k"),
        ("temperature", float("nan"), "temperature"),
        ("temperature", float("inf"), "temperature"),
        ("temperature", -0.1, "temperature"),
        ("top_p", 0.0, "top_p"),
        ("top_p", 1.01, "top_p"),
        ("max_new_tokens", 0, "max_new_tokens"),
        ("max_new_tokens", MAX_NEW_TOKENS_LIMIT + 1, "max_new_tokens"),
    ],
)
def test_validate_generation_request_rejects_invalid_fields(
    field: str, value: object, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        _valid_request(**{field: value})


def test_baseline_coefficient_must_remain_zero() -> None:
    with pytest.raises(ValueError, match="baseline"):
        validate_generation_request(_valid_request(condition="baseline", coefficient=0.2))
    validate_generation_request(_valid_request(condition="ablation", coefficient=0.2))


def test_generation_request_round_trip_serialization() -> None:
    request = _valid_request(condition="ablation", coefficient=0.2, j_lens_layer=4)
    payload = generation_request_to_dict(request)
    restored = generation_request_from_dict(payload)
    assert restored == request
    assert json.loads(json.dumps(payload)) == payload


def test_event_and_prediction_serialization_are_json_ready() -> None:
    from refusal_direction_replication.qwen_lab import TokenPrediction

    event = GenerationEvent(
        step=1,
        generated_token_id=9,
        generated_token_text="␠hi",
        accumulated_text="␠hi",
        effective_condition="ablation",
        effective_coefficient=0.2,
        j_lens_layer=3,
        predictions=(
            TokenPrediction(
                token_id=1,
                display_text="␠",
                score=1.0,
                relative_bar=1.0,
                rank=1,
                previous_rank_delta=None,
            ),
        ),
        step_time_seconds=0.01,
        gpu_memory_bytes=1_000,
        status=None,
    )
    payload = generation_event_to_dict(event)
    assert payload["step"] == 1
    assert payload["status"] is None
    assert token_prediction_to_dict(event.predictions[0])["previous_rank_delta"] is None


def test_lab_preset_is_immutable() -> None:
    preset = LabPreset(condition="addition", coefficient=0.5, label="Addition 50%")
    with pytest.raises(Exception):
        preset.coefficient = 0.1  # type: ignore[misc]


def test_valid_conditions_are_typed_literals() -> None:
    assert LabCondition.__args__ == CONDITIONS  # type: ignore[attr-defined]


def test_default_jlens_version_is_v2() -> None:
    from refusal_direction_replication.qwen_lab import DEFAULT_JLENS_VERSION

    request = _valid_request()
    assert DEFAULT_JLENS_VERSION == "v2"
    assert request.j_lens_version == "v2"


def test_explicit_v1_jlens_version_is_accepted_and_serialized() -> None:
    request = _valid_request(j_lens_version="v1")
    assert request.j_lens_version == "v1"
    payload = generation_request_to_dict(request)
    assert payload["j_lens_version"] == "v1"
    restored = generation_request_from_dict(payload)
    assert restored.j_lens_version == "v1"


def test_legacy_request_payload_without_version_defaults_to_v2() -> None:
    request = _valid_request()
    payload = generation_request_to_dict(request)
    payload.pop("j_lens_version", None)
    restored = generation_request_from_dict(payload)
    assert restored.j_lens_version == "v2"


def test_invalid_jlens_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="j_lens_version"):
        _valid_request(j_lens_version="v3")


def test_jlens_selector_labels_match_design() -> None:
    from refusal_direction_replication.qwen_lab import JLENS_VERSION_LABELS

    assert JLENS_VERSION_LABELS["v2"] == "v2 (contexto longo)"
    assert JLENS_VERSION_LABELS["v1"] == "v1 (100 amostras)"