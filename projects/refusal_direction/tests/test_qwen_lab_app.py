from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from refusal_direction_replication.qwen_code_eval import COEFFICIENTS, CONDITIONS
from refusal_direction_replication.qwen_lab import (
    DEFAULT_TOP_K,
    LAB_MAX_JLENS_LAYER,
    LAB_MIN_JLENS_LAYER,
    LAB_PRESETS,
    GenerationEvent,
    GenerationRequest,
    TokenPrediction,
    escape_token_display,
    generation_event_to_dict,
    generation_request_to_dict,
)
from refusal_direction_replication.qwen_lab_readout import LENS_CONTRACT_QWEN


def _prediction(
    token_id: int,
    *,
    rank: int = 1,
    score: float = 1.0,
    bar: float = 1.0,
    delta: int | None = None,
) -> TokenPrediction:
    text = chr(97 + (token_id % 26))
    return TokenPrediction(
        token_id=token_id,
        display_text=escape_token_display(token_id=token_id, text=text),
        score=score,
        relative_bar=bar,
        rank=rank,
        previous_rank_delta=delta,
    )


def _event(
    step: int,
    *,
    token_id: int = 7,
    text: str = "abc",
    predictions: tuple[TokenPrediction, ...] | None = None,
    status: str | None = None,
) -> GenerationEvent:
    preds = predictions or (
        _prediction(7, rank=1),
        _prediction(3, rank=2, score=0.4, bar=0.4, delta=1),
    )
    display = escape_token_display(token_id=token_id, text=text[step - 1 : step] if step else "x")
    return GenerationEvent(
        step=step,
        generated_token_id=token_id,
        generated_token_text=display,
        accumulated_text=text[:step],
        effective_condition="baseline",
        effective_coefficient=0.0,
        j_lens_layer=12,
        predictions=preds,
        step_time_seconds=0.01,
        gpu_memory_bytes=1024,
        status=status,
    )


class FakeEngine:
    def __init__(self, events: list[GenerationEvent] | None = None) -> None:
        self._events = list(events or [_event(1), _event(2, text="abcd")])
        self.terminal_status: str | None = None
        self.active = False

    def generate(
        self,
        request: GenerationRequest,
        cancelled: Any = None,
    ) -> Iterator[GenerationEvent]:
        del request
        self.active = True
        try:
            for index, event in enumerate(self._events):
                if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                    self.terminal_status = "cancelled"
                    return
                yield event
                if cancelled is not None and getattr(cancelled, "is_set", lambda: False)():
                    self.terminal_status = "cancelled"
                    return
            self.terminal_status = self._events[-1].status or "completed"
        finally:
            self.active = False


def test_module_import_avoids_gradio_dependency() -> None:
    blocked = ModuleNotFoundError("gradio blocked")
    with patch.dict(sys.modules, {"gradio": blocked}):
        module = importlib.import_module("refusal_direction_replication.qwen_lab_app")
        assert module is not None
    source = inspect.getsource(module.build_gradio_app)
    assert "import gradio" in source


def test_preset_resolution_matches_study_coefficients() -> None:
    from refusal_direction_replication.qwen_lab_app import LabUIController

    controller = LabUIController()
    for preset, condition in zip(LAB_PRESETS, CONDITIONS, strict=True):
        resolved = controller.resolve_preset(preset.condition)
        assert resolved.condition == condition
        assert resolved.coefficient == float(COEFFICIENTS[condition])
    baseline = controller.resolve_preset("baseline")
    assert baseline.manual_coefficient_enabled is False
    assert baseline.coefficient == 0.0
    addition = controller.resolve_preset("addition")
    assert addition.manual_coefficient_enabled is True


def test_build_request_applies_manual_coefficient_and_validates() -> None:
    from refusal_direction_replication.qwen_lab_app import LabUIController

    controller = LabUIController()
    request = controller.build_generation_request(
        prompt="olá",
        condition="ablation",
        manual_mode=True,
        coefficient=0.35,
        j_lens_layer=10,
        top_k=DEFAULT_TOP_K,
        temperature=0.7,
        top_p=0.95,
        seed=5,
        max_new_tokens=16,
    )
    assert request.coefficient == pytest.approx(0.35)
    assert request.condition == "ablation"
    with pytest.raises(ValueError, match="baseline"):
        controller.build_generation_request(
            prompt="x",
            condition="baseline",
            manual_mode=True,
            coefficient=0.2,
            j_lens_layer=10,
            top_k=DEFAULT_TOP_K,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_new_tokens=8,
        )


def test_streaming_updates_text_and_predictions_in_sync() -> None:
    from refusal_direction_replication.qwen_lab_app import (
        GenerationSession,
        LabUIController,
        LabViewState,
    )

    engine = FakeEngine([_event(1, text="xy"), _event(2, text="xy")])
    controller = LabUIController()
    session = GenerationSession()
    state = LabViewState()
    request = controller.build_generation_request(
        prompt="pergunta",
        condition="baseline",
        manual_mode=False,
        coefficient=0.0,
        j_lens_layer=12,
        top_k=2,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=4,
    )
    texts: list[str] = []
    for tick in controller.stream_generation(
        engine,
        request,
        state=state,
        session=session,
    ):
        texts.append(tick.view_state.response_text)
    assert texts == ["", "x", "xy"]
    assert len(state.prediction_rows) == 2
    produced = state.prediction_caption
    assert "id 7" in produced.lower() or "ID 7" in produced
    assert state.events[-1].step == 2


def test_generation_lock_and_unlock_descriptors() -> None:
    from refusal_direction_replication.qwen_lab_app import LabUIController

    controller = LabUIController()
    locked = {item.component: item for item in controller.generation_lock_updates()}
    unlocked = {
        item.component: item
        for item in controller.generation_unlock_updates(
            condition="addition",
            manual_mode=True,
        )
    }
    assert locked["condition_dropdown"].interactive is False
    assert locked["layer_slider"].interactive is False
    assert locked["generate_button"].interactive is False
    assert unlocked["condition_dropdown"].interactive is True
    assert unlocked["manual_coefficient"].interactive is True


def test_baseline_manual_slider_forced_disabled() -> None:
    from refusal_direction_replication.qwen_lab_app import LabUIController

    manual = LabUIController().manual_coefficient_update(
        condition="baseline",
        manual_mode=True,
        coefficient=0.5,
    )
    assert manual.value == 0.0
    assert manual.interactive is False


def test_cancel_preserves_partial_response_and_unlocks() -> None:
    from refusal_direction_replication.qwen_lab_app import (
        GenerationSession,
        LabUIController,
        LabViewState,
    )

    engine = FakeEngine(
        [
            _event(1, text="parcial"),
            _event(2, text="parcial!"),
            _event(3, text="parcial!x"),
        ]
    )
    controller = LabUIController()
    session = GenerationSession()
    state = LabViewState()
    request = controller.build_generation_request(
        prompt="p",
        condition="baseline",
        manual_mode=False,
        coefficient=0.0,
        j_lens_layer=5,
        top_k=DEFAULT_TOP_K,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=8,
    )
    stream = controller.stream_generation(engine, request, state=state, session=session)
    next(stream)
    next(stream)
    next(stream)
    session.cancel_event.set()
    rest = list(stream)
    assert not rest
    assert state.response_text == "pa"
    assert engine.terminal_status == "cancelled"
    unlock = controller.generation_unlock_updates(condition="baseline", manual_mode=False)
    manual = next(item for item in unlock if item.component == "manual_coefficient")
    assert manual.interactive is False


def test_clear_resets_state_and_removes_export_file(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LabUIController,
        LabViewState,
        remove_secure_temp_file,
        write_secure_export_json,
    )

    export_path = write_secure_export_json(tmp_path / "holder", {"ok": True})
    assert export_path.is_file()
    state = LabViewState(response_text="texto", export_path=str(export_path))
    state.events.append(_event(1))
    controller = LabUIController()
    cleared = controller.clear_session(state)
    assert cleared.response_text == ""
    assert cleared.events == []
    assert not export_path.exists()
    remove_secure_temp_file(str(export_path))


def test_export_only_on_explicit_action_and_round_trip(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        EXPORT_SCHEMA_VERSION,
        LabUIController,
        LabViewState,
        assert_export_document_safe,
        load_export_document,
    )

    controller = LabUIController()
    state = LabViewState()
    request = controller.build_generation_request(
        prompt="exportável",
        condition="addition",
        manual_mode=False,
        coefficient=0.5,
        j_lens_layer=8,
        top_k=5,
        temperature=1.0,
        top_p=1.0,
        seed=9,
        max_new_tokens=3,
    )
    state.last_request = request
    state.events = [_event(1, status="max_tokens")]
    with pytest.raises(ValueError, match="explicit"):
        controller.export_session(state, explicit=False, holder=tmp_path / "h")
    path = controller.export_session(state, explicit=True, holder=tmp_path / "h")
    document = load_export_document(path)
    assert document["schema_version"] == EXPORT_SCHEMA_VERSION
    assert document["request"] == generation_request_to_dict(request)
    assert len(document["events"]) == 1
    assert document["terminal_status"] == "max_tokens"
    assert_export_document_safe(document)
    roundtrip = generation_request_to_dict(
        GenerationRequest(**document["request"])
    )
    assert roundtrip == document["request"]


def test_export_redacts_secrets_and_paths(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LabUIController,
        LabViewState,
        assert_export_document_safe,
    )

    controller = LabUIController()
    state = LabViewState()
    state.last_request = controller.build_generation_request(
        prompt="p",
        condition="baseline",
        manual_mode=False,
        coefficient=0.0,
        j_lens_layer=1,
        top_k=3,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=2,
    )
    state.events = [_event(1)]
    with pytest.raises(ValueError):
        assert_export_document_safe(
            {
                "schema_version": 1,
                "request": generation_request_to_dict(state.last_request),
                "events": [generation_event_to_dict(state.events[0])],
                "direction_path": "C:\\artifacts\\lens.pt",
            }
        )
    path = controller.export_session(state, explicit=True, holder=tmp_path / "h")
    raw = json.loads(path.read_text(encoding="utf-8"))
    dumped = json.dumps(raw, ensure_ascii=False)
    assert "selected_direction" not in dumped
    assert "jacobian_lens" not in dumped
    assert "HF_TOKEN" not in dumped
    assert "password" not in dumped.lower()


def test_operational_logging_rejects_prompt_like_fields() -> None:
    from refusal_direction_replication.qwen_lab_app import lab_operational_log

    payload = lab_operational_log(status="running", step=1, condition="baseline")
    assert payload["status"] == "running"
    with pytest.raises(ValueError):
        lab_operational_log(prompt="não deve aparecer")
    with pytest.raises(ValueError):
        lab_operational_log(accumulated_text="vazamento")


def test_preflight_fail_closed_without_a100() -> None:
    from refusal_direction_replication.qwen_lab_app import bootstrap_intervention_lab

    probe = lambda: {
        "cuda_available": False,
        "device_name": "",
        "total_memory_bytes": 0,
    }
    with pytest.raises(ValueError, match="A100"):
        bootstrap_intervention_lab(
            Path("."),
            gpu_probe=probe,
            load_stack=lambda **_kwargs: SimpleNamespace(),
        )


def test_bootstrap_production_uses_qwen_lens_contract(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import bootstrap_intervention_lab

    captured: dict[str, Any] = {}

    def fake_load_stack(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        engine = MagicMock()
        return SimpleNamespace(engine=engine, preflight_summary={"device_name": "A100"})

    probe = lambda: {
        "cuda_available": True,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "total_memory_bytes": 40_000_000_000,
    }
    direction = tmp_path / "direction.pt"
    lens = tmp_path / "lens.pt"
    direction.write_bytes(b"dir")
    lens.write_bytes(b"lens")
    with patch(
        "refusal_direction_replication.qwen_lab_app.verify_direction_artifact",
        return_value="0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086",
    ), patch("refusal_direction_replication.qwen_lab_app.verify_frozen_lens_hash"):
        bundle = bootstrap_intervention_lab(
            tmp_path,
            direction_path=direction,
            lens_path=lens,
            gpu_probe=probe,
            load_stack=fake_load_stack,
        )
        assert bundle.engine is not None
        assert captured.get("lens_contract") == LENS_CONTRACT_QWEN


def test_build_gradio_app_returns_blocks_with_portuguese_layout() -> None:
    pytest.importorskip("gradio")
    from refusal_direction_replication.qwen_lab_app import (
        LAB_DISCLAIMER,
        build_gradio_app,
    )

    engine = FakeEngine([])
    demo = build_gradio_app(
        engine,
        preflight_summary={
            "device_name": "NVIDIA A100",
            "model_revision": "c03e6d3",
            "direction_sha256": "0b1cff5e",
            "lens_sha256": "b700a78c",
        },
    )
    assert demo.__class__.__name__ == "Blocks"
    assert "previsões da j-lens" in LAB_DISCLAIMER.casefold()
    assert "pensamento" in LAB_DISCLAIMER.casefold()


def test_gradio_html_excludes_secret_markers() -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LAB_CSS,
        LAB_DISCLAIMER,
        assert_ui_strings_safe,
    )

    assert_ui_strings_safe(LAB_CSS)
    assert_ui_strings_safe(LAB_DISCLAIMER)


def test_gradio_pin_version_is_documented() -> None:
    from refusal_direction_replication.qwen_lab_app import GRADIO_PIN_VERSION

    assert GRADIO_PIN_VERSION == "5.49.1"


def test_gradio_output_slider_uses_backend_token_limit() -> None:
    from refusal_direction_replication.qwen_lab import MAX_NEW_TOKENS_LIMIT
    from refusal_direction_replication.qwen_lab_app import build_gradio_app

    source = inspect.getsource(build_gradio_app)
    assert "gr.Slider(1, MAX_NEW_TOKENS_LIMIT, value=64" in source


def test_pack_generation_outputs_has_fixed_cardinality() -> None:
    pytest.importorskip("gradio")
    from refusal_direction_replication.qwen_lab_app import (
        GENERATION_OUTPUT_COUNT,
        LabUIController,
        LabViewState,
        pack_generation_outputs,
    )

    state = LabViewState(response_text="ok", status_message="Estado: ok")
    row = pack_generation_outputs(
        state,
        LabUIController(),
        condition="baseline",
        manual_mode=False,
        lock=LabUIController().generation_lock_updates(),
    )
    assert len(row) == GENERATION_OUTPUT_COUNT == 15


def test_run_generation_handler_yield_shape_matches_gradio_outputs(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from refusal_direction_replication.qwen_lab_app import (
        GENERATION_OUTPUT_COUNT,
        LabSessionRegistry,
        LabUIController,
        create_run_generation_handler,
    )

    registry = LabSessionRegistry(export_root=tmp_path / "sessions")
    handler = create_run_generation_handler(
        FakeEngine([_event(1, text="ab"), _event(2, text="ab", status="completed")]),
        registry,
        LabUIController(),
    )
    rows = list(
        handler(
            "sess-a",
            "pergunta",
            "baseline",
            False,
            0.0,
            10.0,
            float(DEFAULT_TOP_K),
            0.0,
            1.0,
            0.0,
            4.0,
        )
    )
    assert len(rows) >= 2
    assert all(len(row) == GENERATION_OUTPUT_COUNT for row in rows)
    assert all(not hasattr(item, "elem_id") for row in rows for item in row)


def test_sync_preset_updates_return_two_gradio_values() -> None:
    pytest.importorskip("gradio")
    import gradio as gr

    from refusal_direction_replication.qwen_lab_app import (
        LabUIController,
        sync_preset_updates,
    )

    pair = sync_preset_updates(LabUIController(), "ablation")
    assert len(pair) == 2
    assert all(isinstance(item, dict) and item.get("__type__") == "update" for item in pair)


def test_session_registry_isolates_state_and_cancel(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LabSessionRegistry,
        LabUIController,
    )

    registry = LabSessionRegistry(export_root=tmp_path / "reg")
    first = registry.get_or_create("one")
    second = registry.get_or_create("two")
    first.view_state.response_text = "alpha"
    second.generation.cancel_event.set()
    assert second.view_state.response_text == ""
    assert not first.generation.cancel_event.is_set()
    assert first.export_dir != second.export_dir


def test_partial_auth_credentials_fail_closed() -> None:
    from refusal_direction_replication.qwen_lab_app import resolve_auth_credentials

    with pytest.raises(ValueError, match="both"):
        resolve_auth_credentials(user="only-user", password="")
    with pytest.raises(ValueError, match="both"):
        resolve_auth_credentials(user="", password="secret")


def test_public_launch_requires_complete_auth() -> None:
    from refusal_direction_replication.qwen_lab_app import require_launch_authentication

    with pytest.raises(ValueError, match="authentication"):
        require_launch_authentication(share=True, server_name="0.0.0.0", auth=None)
    with pytest.raises(ValueError, match="authentication"):
        require_launch_authentication(
            share=False,
            server_name="0.0.0.0",
            auth=None,
        )
    auth = require_launch_authentication(
        share=False,
        server_name="127.0.0.1",
        auth=None,
    )
    assert auth == ("", "")


def test_model_revision_missing_or_mismatch_fails() -> None:
    from refusal_direction_replication.qwen_lab_app import (
        _assert_model_revision,
        _assert_tokenizer_revision,
    )

    with pytest.raises(ValueError, match="missing"):
        _assert_model_revision(None)
    with pytest.raises(ValueError, match="!="):
        _assert_model_revision("deadbeef")

    tokenizer = SimpleNamespace(revision="deadbeef")
    with pytest.raises(ValueError, match="tokenizer"):
        _assert_tokenizer_revision(tokenizer, "c03e6d358207e414f1eca0bb1891e29f1db0e242")


def test_export_allows_prompt_words_without_blocking(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LabUIController,
        LabViewState,
        assert_export_document_safe,
    )

    controller = LabUIController()
    state = LabViewState()
    state.last_request = controller.build_generation_request(
        prompt="Explique password managers e token buckets",
        condition="baseline",
        manual_mode=False,
        coefficient=0.0,
        j_lens_layer=3,
        top_k=3,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=4,
    )
    state.events = [_event(1)]
    document = controller.build_export_document(state)
    assert "password" in document["request"]["prompt"]
    assert_export_document_safe(document)


def test_run_generation_unlocks_after_engine_error(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        GENERATION_OUTPUT_COUNT,
        LabSessionRegistry,
        LabUIController,
        create_run_generation_handler,
    )

    class ErrorEngine:
        def generate(self, request: GenerationRequest, cancelled: Any = None):
            del request, cancelled
            if False:
                yield _event(1)
            raise ValueError("campo inválido")

    registry = LabSessionRegistry(export_root=tmp_path / "err")
    handler = create_run_generation_handler(ErrorEngine(), registry, LabUIController())
    rows = list(
        handler(
            "err",
            "x",
            "baseline",
            False,
            0.0,
            1.0,
            3.0,
            0.0,
            1.0,
            0.0,
            2.0,
        )
    )
    client = registry.get_or_create("err")
    assert len(rows) == 2
    assert len(rows[-1]) == GENERATION_OUTPUT_COUNT
    assert "campo inválido" in client.view_state.status_message.lower()


def test_new_lab_session_hash_factory_is_not_constant() -> None:
    from refusal_direction_replication.qwen_lab_app import new_lab_session_hash

    first = new_lab_session_hash()
    second = new_lab_session_hash()
    assert first != second
    assert len(first) == 32


def test_session_delete_callback_releases_registry_and_export(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_app import (
        LabSessionRegistry,
        make_lab_session_delete_callback,
        write_secure_export_json,
    )

    registry = LabSessionRegistry(export_root=tmp_path / "reg")
    callback = make_lab_session_delete_callback(registry)
    client = registry.get_or_create("hash-1")
    client.view_state.response_text = "resposta"
    export = write_secure_export_json(client.export_dir, {"schema_version": 1})
    client.view_state.export_path = str(export)
    callback("hash-1")
    refreshed = registry.get_or_create("hash-1")
    assert refreshed.view_state.response_text == ""
    assert refreshed.view_state.export_path is None
    assert not export.exists()


def test_gradio_state_uses_callable_ttl_and_independent_sessions(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    import gradio as gr
    from gradio.state_holder import StateHolder

    from refusal_direction_replication.qwen_lab_app import (
        STATE_TTL_SECONDS,
        LabSessionRegistry,
        build_gradio_app,
        new_lab_session_hash,
    )

    registry = LabSessionRegistry(export_root=tmp_path / "gradio_sessions")
    demo = build_gradio_app(
        FakeEngine([]),
        preflight_summary={"device_name": "A100"},
        session_registry=registry,
    )
    state = demo.lab_session_state
    assert callable(state.value)
    assert state.value is new_lab_session_hash
    assert state.time_to_live == STATE_TTL_SECONDS
    assert state.delete_callback is demo.lab_session_delete_callback

    holder = StateHolder()
    holder.set_blocks(demo)
    state_id = state._id

    def bootstrap(browser_id: str) -> str:
        session = holder[browser_id]
        if state_id not in session.state_data:
            session.state_data[state_id] = state.value()
        return str(session.state_data[state_id])

    hash_a = bootstrap("browser-a")
    hash_b = bootstrap("browser-b")
    assert hash_a != hash_b

    registry.get_or_create(hash_a).view_state.response_text = "only-a"
    assert registry.get_or_create(hash_b).view_state.response_text == ""

    state.delete_callback(hash_a)
    assert registry.get_or_create(hash_a).view_state.response_text == ""

    holder.delete_state("browser-a")


def test_qwen_run_lab_main_rejects_share_without_auth() -> None:
    from importlib.machinery import SourceFileLoader

    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_run_lab.py"
    module = SourceFileLoader("qwen_run_lab", str(script)).load_module()
    with pytest.raises(ValueError, match="authentication"):
        module.main(["--share"])


def test_qwen_run_lab_main_does_not_print_password(capsys: pytest.CaptureFixture[str]) -> None:
    from importlib.machinery import SourceFileLoader

    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_run_lab.py"
    module = SourceFileLoader("qwen_run_lab", str(script)).load_module()
    launch = MagicMock()
    with patch.object(
        module,
        "bootstrap_intervention_lab",
        return_value=SimpleNamespace(
            engine=FakeEngine([]),
            preflight_summary={"device_name": "A100"},
        ),
    ), patch.object(
        module,
        "build_gradio_app",
        return_value=SimpleNamespace(launch=launch, auth=("local-user", "secret")),
    ):
        env = {
            "QWEN_LAB_AUTH_USER": "local-user",
            "QWEN_LAB_AUTH_PASSWORD": "SUPER-SECRET-PASS",
        }
        with patch.dict(os.environ, env, clear=False):
            module.main(["--no-share", "--server-name", "127.0.0.1"])
    captured = capsys.readouterr().out + capsys.readouterr().err
    assert "SUPER-SECRET-PASS" not in captured
    launch.assert_called_once()