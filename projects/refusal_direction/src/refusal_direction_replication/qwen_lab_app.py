"""Gradio-facing controller and bootstrap for the Qwen intervention lab."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .directions import load_direction
from .pipeline import load_causal_lm, transformer_module_lists
from .qwen_code_eval_runtime import _default_gpu_probe, require_a100_preflight
from .qwen_dose import METADATA_LEAK_KEYS, sha256_file
from .qwen_jlens import QWEN_MODEL, QWEN_REVISION, assert_jlens_contract, wrap_hf_model_for_jlens
from .qwen_lab import (
    DEFAULT_TOP_K,
    LAB_MAX_JLENS_LAYER,
    LAB_MIN_JLENS_LAYER,
    LAB_MODEL_REVISION,
    LAB_PRESETS,
    GenerationEvent,
    GenerationRequest,
    LabCondition,
    TokenPrediction,
    generation_event_to_dict,
    generation_request_to_dict,
)
from .qwen_lab_readout import (
    FROZEN_LENS_SHA256,
    LENS_CONTRACT_QWEN,
    LiveJLensReadout,
    verify_frozen_lens_hash,
)
from .qwen_lab_runtime import (
    build_random_control_direction,
    operational_status_log,
    wrap_lab_model_for_jlens,
    InterventionEngine,
)

FROZEN_DIRECTION_SHA256 = (
    "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
)
DEFAULT_DIRECTION_RELATIVE = Path("artifacts/qwen25_coder_7b/selected_direction.pt")
DEFAULT_LENS_RELATIVE = Path(
    "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
)
EXPORT_SCHEMA_VERSION = 1

LAB_DISCLAIMER = (
    "Previsões da J-lens: estes tokens são os maiores scores da leitura incremental "
    "na camada escolhida. Não são pensamentos literais do modelo nem explicação causal."
)

LAB_CSS = """
.lab-root { font-family: system-ui, sans-serif; }
.lab-header h1 { font-size: 1.35rem; margin-bottom: 0.25rem; }
.lab-disclaimer {
    font-size: 0.85rem;
    color: #444;
    border-left: 3px solid #8a8a8a;
    padding: 0.35rem 0.65rem;
    margin: 0.5rem 0 1rem 0;
    background: #f7f7f7;
}
.lab-preflight { font-size: 0.8rem; color: #555; }
.lab-response textarea { font-family: ui-monospace, monospace; min-height: 320px; }
"""

EXPORT_FORBIDDEN_KEYS = METADATA_LEAK_KEYS | {
    "password",
    "auth",
    "hf_token",
    "token",
    "lens",
    "direction",
    "jacobians",
    "weights",
    "state_dict",
    "lens_path",
    "direction_path",
    "source_lens",
    "source_direction",
    "artifacts",
    "credentials",
    "secret",
}

UI_FORBIDDEN_SUBSTRINGS = (
    "hf_token",
    "huggingface",
    "selected_direction.pt",
    "jacobian_lens.pt",
    "password",
    "artifacts/",
    "artifacts\\",
)

_PREDICTION_HEADERS = (
    "Rank",
    "ID",
    "Previsão",
    "Score",
    "Barra rel.",
    "Δ rank",
)


@dataclass(frozen=True)
class PresetResolution:
    condition: LabCondition
    coefficient: float
    manual_coefficient_enabled: bool


@dataclass(frozen=True)
class ComponentUpdate:
    component: str
    value: Any | None = None
    interactive: bool | None = None


@dataclass
class LabViewState:
    response_text: str = ""
    prediction_rows: list[list[Any]] = field(default_factory=list)
    prediction_caption: str = ""
    status_message: str = ""
    events: list[GenerationEvent] = field(default_factory=list)
    last_request: GenerationRequest | None = None
    export_path: str | None = None
    terminal_status: str | None = None


@dataclass
class GenerationSession:
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def reset_cancel(self) -> None:
        self.cancel_event = threading.Event()


@dataclass(frozen=True)
class StreamTick:
    view_state: LabViewState
    lock: tuple[ComponentUpdate, ...] = ()
    unlock: tuple[ComponentUpdate, ...] = ()


@dataclass(frozen=True)
class LabRuntimeBundle:
    engine: InterventionEngine
    preflight_summary: dict[str, Any]


def lab_operational_log(**fields: Any) -> dict[str, Any]:
    return operational_status_log(**fields)


def _short_hash(value: str) -> str:
    cleaned = str(value).strip()
    if len(cleaned) <= 12:
        return cleaned
    return cleaned[:8]


def resolve_lab_artifact_paths(
    root: Path,
    *,
    direction_path: str | Path | None = None,
    lens_path: str | Path | None = None,
) -> tuple[Path, Path]:
    root = root.resolve()
    direction = (
        Path(direction_path)
        if direction_path is not None
        else root / DEFAULT_DIRECTION_RELATIVE
    )
    lens = Path(lens_path) if lens_path is not None else root / DEFAULT_LENS_RELATIVE
    if not direction.is_file():
        raise FileNotFoundError(f"direction artifact missing: {direction}")
    if not lens.is_file():
        raise FileNotFoundError(f"lens artifact missing: {lens}")
    return direction.resolve(), lens.resolve()


def verify_direction_artifact(
    path: str | Path,
    *,
    expected_sha256: str = FROZEN_DIRECTION_SHA256,
) -> str:
    actual = sha256_file(Path(path))
    if actual != expected_sha256:
        raise ValueError(
            f"direction_sha256 {actual!r} != expected {expected_sha256!r}"
        )
    return actual


def _assert_model_revision(resolved_revision: str | None) -> str:
    observed = str(resolved_revision or "").strip()
    if observed != LAB_MODEL_REVISION:
        raise ValueError(
            f"resolved revision {observed!r} != expected {LAB_MODEL_REVISION!r}"
        )
    return observed


def assert_ui_strings_safe(text: str) -> None:
    lowered = text.casefold()
    if any(marker.casefold() in lowered for marker in UI_FORBIDDEN_SUBSTRINGS):
        raise ValueError("ui strings must not include secrets or local artifact paths")
    if re.search(r"[A-Za-z]:\\\\", text):
        raise ValueError("ui strings must not include windows paths")
    if "/content/" in lowered:
        raise ValueError("ui strings must not include colab paths")


def _walk_values(payload: Any) -> Iterator[Any]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield key
            yield from _walk_values(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _walk_values(item)


def assert_export_document_safe(document: Mapping[str, Any]) -> None:
    for key in document:
        if str(key).casefold() in EXPORT_FORBIDDEN_KEYS:
            raise ValueError(f"export must not include forbidden key {key!r}")
    for value in _walk_values(document):
        if isinstance(value, str):
            lowered = value.casefold()
            if any(marker in lowered for marker in ("selected_direction", "jacobian_lens")):
                raise ValueError("export must not include artifact file names")
            if "hf_token" in lowered or "password" in lowered:
                raise ValueError("export must not include credentials")
            if re.search(r"^[A-Za-z]:\\\\", value) or value.startswith("/content/"):
                raise ValueError("export must not include local paths")


def remove_secure_temp_file(path: str | None) -> None:
    if not path:
        return
    candidate = Path(path)
    if candidate.is_file():
        candidate.unlink(missing_ok=True)


def write_secure_export_json(holder: Path, payload: Mapping[str, Any]) -> Path:
    holder.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="qwen-lab-export-",
        suffix=".json",
        dir=holder,
    )
    os.close(fd)
    path = Path(tmp_name)
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        if os.name != "nt":
            os.chmod(path, 0o600)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def load_export_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_production_stack(
    *,
    model: Any,
    tokenizer: Any,
    direction_path: Path,
    lens_path: Path,
    lens_contract: Literal["qwen", "fixture"] = LENS_CONTRACT_QWEN,
) -> tuple[InterventionEngine, LiveJLensReadout]:
    assert_jlens_contract()
    direction = load_direction(direction_path)
    blocks, attentions, mlps = transformer_module_lists(model)
    bound = wrap_lab_model_for_jlens(model, tokenizer)
    from jlens.lens import JacobianLens

    lens = JacobianLens.load(str(lens_path))
    decode_token = lambda token_id: tokenizer.decode(  # noqa: E731
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    readout = LiveJLensReadout(
        lens,
        bound,
        decode_token=decode_token,
        lens_path=str(lens_path),
        expected_lens_sha256=FROZEN_LENS_SHA256,
        lens_contract=lens_contract,
    )
    random_control = build_random_control_direction(direction.vector)
    engine = InterventionEngine(
        model,
        tokenizer,
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=direction,
        random_control=random_control,
        readout=readout,
        lens_path=str(lens_path),
        expected_lens_sha256=FROZEN_LENS_SHA256,
    )
    return engine, readout


def bootstrap_intervention_lab(
    root: Path,
    *,
    direction_path: str | Path | None = None,
    lens_path: str | Path | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] | None = None,
    load_model: Callable[..., tuple[Any, Any, Any]] | None = None,
    load_stack: Callable[..., Any] | None = None,
) -> LabRuntimeBundle:
    probe = gpu_probe or _default_gpu_probe
    preflight = require_a100_preflight(probe)
    root = root.resolve()
    direction_file, lens_file = resolve_lab_artifact_paths(
        root,
        direction_path=direction_path or os.environ.get("QWEN_LAB_DIRECTION"),
        lens_path=lens_path or os.environ.get("QWEN_LAB_LENS"),
    )
    direction_digest = verify_direction_artifact(direction_file)
    verify_frozen_lens_hash(lens_file, FROZEN_LENS_SHA256)
    if load_stack is not None:
        built = load_stack(
            root=root,
            direction_path=direction_file,
            lens_path=lens_file,
            lens_contract=LENS_CONTRACT_QWEN,
        )
        engine = built.engine if hasattr(built, "engine") else built
        summary = dict(getattr(built, "preflight_summary", {}))
        return LabRuntimeBundle(engine=engine, preflight_summary=summary)

    loader = load_model or _load_bf16_model
    model, tokenizer, _dtype = loader(
        model_name=QWEN_MODEL,
        revision=LAB_MODEL_REVISION,
    )
    resolved = getattr(model.config, "_commit_hash", None) or LAB_MODEL_REVISION
    revision = _assert_model_revision(resolved)
    engine, _readout = build_production_stack(
        model=model,
        tokenizer=tokenizer,
        direction_path=direction_file,
        lens_path=lens_file,
        lens_contract=LENS_CONTRACT_QWEN,
    )
    summary = {
        "device_name": preflight.get("device_name"),
        "model_revision": _short_hash(revision),
        "direction_sha256": _short_hash(direction_digest),
        "lens_sha256": _short_hash(FROZEN_LENS_SHA256),
    }
    return LabRuntimeBundle(engine=engine, preflight_summary=summary)


def _load_bf16_model(*, model_name: str, revision: str) -> tuple[Any, Any, Any]:
    import torch

    model, tokenizer = load_causal_lm(model_name, dtype=torch.bfloat16, revision=revision)
    return model, tokenizer, torch.bfloat16


class LabUIController:
    """Presentation logic without Gradio imports."""

    CONDITIONS: tuple[LabCondition, ...] = tuple(preset.condition for preset in LAB_PRESETS)

    def resolve_preset(self, condition: LabCondition) -> PresetResolution:
        match = next(
            (preset for preset in LAB_PRESETS if preset.condition == condition),
            None,
        )
        if match is None:
            raise ValueError(f"unknown preset condition: {condition!r}")
        manual = condition != "baseline"
        return PresetResolution(
            condition=match.condition,
            coefficient=float(match.coefficient),
            manual_coefficient_enabled=manual,
        )

    def manual_coefficient_update(
        self,
        *,
        condition: LabCondition,
        manual_mode: bool,
        coefficient: float,
    ) -> ComponentUpdate:
        if condition == "baseline":
            return ComponentUpdate(
                component="manual_coefficient",
                value=0.0,
                interactive=False,
            )
        enabled = manual_mode and condition != "baseline"
        value = float(coefficient) if enabled else self.resolve_preset(condition).coefficient
        return ComponentUpdate(
            component="manual_coefficient",
            value=value,
            interactive=enabled,
        )

    def build_generation_request(
        self,
        *,
        prompt: str,
        condition: LabCondition,
        manual_mode: bool,
        coefficient: float,
        j_lens_layer: int,
        top_k: int,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
    ) -> GenerationRequest:
        resolved = self.resolve_preset(condition)
        effective = float(coefficient) if manual_mode else resolved.coefficient
        if condition == "baseline":
            if manual_mode and effective != 0.0:
                raise ValueError("baseline coefficient must remain zero")
            effective = 0.0
        return GenerationRequest(
            prompt=prompt,
            condition=condition,
            coefficient=effective,
            j_lens_layer=int(j_lens_layer),
            top_k=int(top_k),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed),
            max_new_tokens=int(max_new_tokens),
        )

    def generation_lock_updates(self) -> tuple[ComponentUpdate, ...]:
        targets = (
            "condition_dropdown",
            "layer_slider",
            "preset_radio",
            "manual_mode",
            "manual_coefficient",
            "top_k",
            "temperature",
            "top_p",
            "seed",
            "max_new_tokens",
            "generate_button",
        )
        return tuple(
            ComponentUpdate(component=name, interactive=False) for name in targets
        )

    def generation_unlock_updates(
        self,
        *,
        condition: LabCondition = "baseline",
        manual_mode: bool = False,
    ) -> tuple[ComponentUpdate, ...]:
        manual = self.manual_coefficient_update(
            condition=condition,
            manual_mode=manual_mode,
            coefficient=0.0,
        )
        base = (
            ComponentUpdate(component="condition_dropdown", interactive=True),
            ComponentUpdate(component="layer_slider", interactive=True),
            ComponentUpdate(component="preset_radio", interactive=True),
            ComponentUpdate(component="manual_mode", interactive=True),
            ComponentUpdate(component="top_k", interactive=True),
            ComponentUpdate(component="temperature", interactive=True),
            ComponentUpdate(component="top_p", interactive=True),
            ComponentUpdate(component="seed", interactive=True),
            ComponentUpdate(component="max_new_tokens", interactive=True),
            ComponentUpdate(component="generate_button", interactive=True),
            manual,
        )
        return base

    @staticmethod
    def predictions_to_rows(predictions: Sequence[TokenPrediction]) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for prediction in predictions:
            delta = (
                "—"
                if prediction.previous_rank_delta is None
                else str(prediction.previous_rank_delta)
            )
            rows.append(
                [
                    prediction.rank,
                    prediction.token_id,
                    prediction.display_text,
                    f"{prediction.score:.4f}",
                    f"{prediction.relative_bar:.2f}",
                    delta,
                ]
            )
        return rows

    def apply_event(self, state: LabViewState, event: GenerationEvent) -> None:
        state.response_text = event.accumulated_text
        state.prediction_rows = self.predictions_to_rows(event.predictions)
        state.prediction_caption = (
            f"Camada {event.j_lens_layer} · passo {event.step} · "
            f"token produzido: {event.generated_token_text} (id {event.generated_token_id})"
        )
        state.events.append(event)
        if event.status:
            state.terminal_status = event.status
            state.status_message = f"Estado: {event.status}"

    def stream_generation(
        self,
        engine: Any,
        request: GenerationRequest,
        *,
        state: LabViewState,
        session: GenerationSession,
    ) -> Iterator[StreamTick]:
        session.reset_cancel()
        state.last_request = request
        state.events = []
        state.response_text = ""
        state.prediction_rows = []
        state.prediction_caption = ""
        state.status_message = "Gerando…"
        state.terminal_status = None
        lab_operational_log(status="started", condition=request.condition, layer=request.j_lens_layer)
        lock = self.generation_lock_updates()
        yield StreamTick(view_state=state, lock=lock)
        iterator = engine.generate(request, cancelled=session.cancel_event)
        try:
            while True:
                if session.cancel_event.is_set():
                    if getattr(engine, "terminal_status", None) is None:
                        engine.terminal_status = "cancelled"
                    state.terminal_status = "cancelled"
                    break
                try:
                    event = next(iterator)
                except StopIteration:
                    break
                self.apply_event(state, event)
                yield StreamTick(view_state=state)
                if session.cancel_event.is_set():
                    if getattr(engine, "terminal_status", None) is None:
                        engine.terminal_status = "cancelled"
                    state.terminal_status = "cancelled"
                    break
        finally:
            iterator.close()
            terminal = engine.terminal_status or state.terminal_status
            state.terminal_status = terminal
            if terminal:
                state.status_message = f"Estado: {terminal}"
            lab_operational_log(status=terminal or "idle", condition=request.condition)

    def clear_session(self, state: LabViewState) -> LabViewState:
        remove_secure_temp_file(state.export_path)
        state.response_text = ""
        state.prediction_rows = []
        state.prediction_caption = ""
        state.status_message = ""
        state.events = []
        state.last_request = None
        state.export_path = None
        state.terminal_status = None
        lab_operational_log(status="cleared")
        return state

    def build_export_document(self, state: LabViewState) -> dict[str, Any]:
        if state.last_request is None or not state.events:
            raise ValueError("nothing to export")
        terminal = state.terminal_status
        if terminal is None and state.events:
            terminal = state.events[-1].status
        document = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "terminal_status": terminal,
            "request": generation_request_to_dict(state.last_request),
            "events": [generation_event_to_dict(event) for event in state.events],
            "parameters": {
                "model_revision": LAB_MODEL_REVISION,
                "direction_sha256": FROZEN_DIRECTION_SHA256,
                "lens_sha256": FROZEN_LENS_SHA256,
            },
        }
        assert_export_document_safe(document)
        return document

    def export_session(
        self,
        state: LabViewState,
        *,
        explicit: bool,
        holder: Path,
    ) -> Path:
        if not explicit:
            raise ValueError("export requires an explicit user action")
        document = self.build_export_document(state)
        remove_secure_temp_file(state.export_path)
        path = write_secure_export_json(holder, document)
        state.export_path = str(path)
        lab_operational_log(status="exported", event_count=len(state.events))
        return path


def build_gradio_app(
    engine: InterventionEngine,
    *,
    preflight_summary: Mapping[str, Any],
    auth: tuple[str, str] | None = None,
    export_dir: Path | None = None,
) -> Any:
    import gradio as gr

    controller = LabUIController()
    session = GenerationSession()
    view_state = LabViewState()
    export_holder = export_dir or Path(tempfile.gettempdir()) / "qwen_lab_exports"
    export_holder.mkdir(parents=True, exist_ok=True)

    preflight_line = (
        f"GPU: {preflight_summary.get('device_name', '—')} · "
        f"revisão {preflight_summary.get('model_revision', '—')} · "
        f"direção {preflight_summary.get('direction_sha256', '—')} · "
        f"lente {preflight_summary.get('lens_sha256', '—')}"
    )

    def _apply_updates(updates: Sequence[ComponentUpdate], components: dict[str, Any]) -> list[Any]:
        ordered: list[Any] = []
        for update in updates:
            component = components.get(update.component)
            if component is None:
                continue
            values: list[Any] = [component]
            if update.value is not None:
                values.append(gr.update(value=update.value))
            elif update.interactive is not None:
                values.append(gr.update(interactive=update.interactive))
            else:
                values.append(gr.update())
            ordered.extend(values)
        return ordered

    with gr.Blocks(css=LAB_CSS, title="Laboratório de intervenção Qwen") as demo:
        gr.Markdown(
            '<div class="lab-header"><h1>Laboratório de intervenção</h1></div>',
        )
        gr.Markdown(f'<div class="lab-disclaimer">{LAB_DISCLAIMER}</div>')
        gr.Markdown(f'<div class="lab-preflight">{preflight_line}</div>')

        components: dict[str, Any] = {}

        with gr.Row(elem_classes=["lab-root"]):
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Escreva o prompt do usuário…",
                    lines=6,
                )
                preset_radio = gr.Radio(
                    choices=[(preset.label, preset.condition) for preset in LAB_PRESETS],
                    value="baseline",
                    label="Preset de intervenção",
                )
                condition_dropdown = gr.Dropdown(
                    choices=list(controller.CONDITIONS),
                    value="baseline",
                    label="Condição",
                )
                manual_mode = gr.Checkbox(label="Usar coeficiente manual", value=False)
                manual_coefficient = gr.Slider(
                    0.0,
                    1.0,
                    value=0.0,
                    step=0.05,
                    label="Coeficiente manual",
                    interactive=False,
                )
                layer_slider = gr.Slider(
                    LAB_MIN_JLENS_LAYER,
                    LAB_MAX_JLENS_LAYER,
                    value=18,
                    step=1,
                    label="Camada da J-lens",
                )
                top_k = gr.Slider(1, 20, value=DEFAULT_TOP_K, step=1, label="Top-k exibido")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperatura")
                top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Top-p")
                seed = gr.Number(value=0, precision=0, label="Seed")
                max_new_tokens = gr.Slider(1, 256, value=64, step=1, label="Máx. novos tokens")
                with gr.Row():
                    generate_button = gr.Button("Gerar", variant="primary")
                    stop_button = gr.Button("Interromper")
                    clear_button = gr.Button("Limpar")
                    export_button = gr.Button("Exportar sessão")
            with gr.Column(scale=5, elem_classes=["lab-response"]):
                response = gr.Textbox(
                    label="Resposta do modelo",
                    lines=14,
                    interactive=False,
                )
                status_message = gr.Markdown("")
            with gr.Column(scale=5):
                prediction_caption = gr.Markdown("Previsões da J-lens")
                predictions = gr.Dataframe(
                    headers=list(_PREDICTION_HEADERS),
                    datatype=["number", "number", "str", "str", "str", "str"],
                    label="Previsões da J-lens",
                    interactive=False,
                )
                export_file = gr.File(label="Exportação JSON", interactive=False)

        components.update(
            {
                "condition_dropdown": condition_dropdown,
                "layer_slider": layer_slider,
                "preset_radio": preset_radio,
                "manual_mode": manual_mode,
                "manual_coefficient": manual_coefficient,
                "top_k": top_k,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "generate_button": generate_button,
            }
        )

        def sync_preset(condition: str) -> list[Any]:
            resolved = controller.resolve_preset(condition)  # type: ignore[arg-type]
            manual = controller.manual_coefficient_update(
                condition=resolved.condition,
                manual_mode=False,
                coefficient=resolved.coefficient,
            )
            return [
                condition_dropdown,
                gr.update(value=resolved.condition),
                manual_coefficient,
                gr.update(value=manual.value, interactive=manual.interactive),
            ]

        preset_radio.change(
            sync_preset,
            inputs=[preset_radio],
            outputs=[condition_dropdown, condition_dropdown, manual_coefficient, manual_coefficient],
        )

        condition_dropdown.change(
            sync_preset,
            inputs=[condition_dropdown],
            outputs=[condition_dropdown, condition_dropdown, manual_coefficient, manual_coefficient],
        )

        def on_manual_mode(condition: str, enabled: bool, coefficient: float) -> Any:
            update = controller.manual_coefficient_update(
                condition=condition,  # type: ignore[arg-type]
                manual_mode=enabled,
                coefficient=coefficient,
            )
            return gr.update(value=update.value, interactive=update.interactive)

        manual_mode.change(
            on_manual_mode,
            inputs=[condition_dropdown, manual_mode, manual_coefficient],
            outputs=[manual_coefficient],
        )

        def run_generation(
            prompt_text: str,
            condition: str,
            manual_enabled: bool,
            coefficient: float,
            layer: float,
            top_k_value: float,
            temp: float,
            nucleus: float,
            seed_value: float,
            max_tokens: float,
        ) -> Iterator[list[Any]]:
            request = controller.build_generation_request(
                prompt=prompt_text,
                condition=condition,  # type: ignore[arg-type]
                manual_mode=manual_enabled,
                coefficient=coefficient,
                j_lens_layer=int(layer),
                top_k=int(top_k_value),
                temperature=float(temp),
                top_p=float(nucleus),
                seed=int(seed_value),
                max_new_tokens=int(max_tokens),
            )
            outputs_map = {
                **components,
                "response": response,
                "status_message": status_message,
                "prediction_caption": prediction_caption,
                "predictions": predictions,
            }
            for tick in controller.stream_generation(
                engine,
                request,
                state=view_state,
                session=session,
            ):
                payload: list[Any] = []
                if tick.lock:
                    payload.extend(_apply_updates(tick.lock, outputs_map))
                payload.extend(
                    [
                        response,
                        view_state.response_text,
                        status_message,
                        view_state.status_message or "",
                        prediction_caption,
                        view_state.prediction_caption,
                        predictions,
                        view_state.prediction_rows,
                    ]
                )
                yield payload
            unlock = controller.generation_unlock_updates(
                condition=condition,  # type: ignore[arg-type]
                manual_mode=manual_enabled,
            )
            yield _apply_updates(unlock, outputs_map)

        generate_button.click(
            run_generation,
            inputs=[
                prompt,
                condition_dropdown,
                manual_mode,
                manual_coefficient,
                layer_slider,
                top_k,
                temperature,
                top_p,
                seed,
                max_new_tokens,
            ],
            outputs=[
                condition_dropdown,
                layer_slider,
                preset_radio,
                manual_mode,
                manual_coefficient,
                top_k,
                temperature,
                top_p,
                seed,
                max_new_tokens,
                generate_button,
                response,
                status_message,
                prediction_caption,
                predictions,
            ],
            concurrency_limit=1,
        )

        stop_button.click(lambda: session.cancel_event.set(), outputs=[])

        def on_clear() -> tuple[str, list[list[Any]], str, str, None]:
            controller.clear_session(view_state)
            return "", [], "", "", None

        clear_button.click(
            on_clear,
            outputs=[response, predictions, prediction_caption, status_message, export_file],
        )

        def on_export() -> Any:
            try:
                path = controller.export_session(
                    view_state,
                    explicit=True,
                    holder=export_holder,
                )
            except ValueError as exc:
                return gr.update(value=None), gr.update(value=str(exc))
            return gr.update(value=str(path)), gr.update(value="Exportação pronta para download.")

        export_button.click(
            on_export,
            outputs=[export_file, status_message],
        )

    demo.engine = engine  # type: ignore[attr-defined]
    demo.auth = auth  # type: ignore[attr-defined]
    return demo