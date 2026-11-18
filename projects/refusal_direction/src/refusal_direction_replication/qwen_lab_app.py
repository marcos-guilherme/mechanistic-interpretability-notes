"""Gradio-facing controller and bootstrap for the Qwen intervention lab."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .directions import load_direction
from .pipeline import load_causal_lm, transformer_module_lists
from .qwen_code_eval_runtime import _default_gpu_probe, require_a100_preflight
from .qwen_dose import METADATA_LEAK_KEYS, sha256_file
from .qwen_jlens import QWEN_MODEL, QWEN_REVISION, assert_jlens_contract, wrap_hf_model_for_jlens
from .qwen_lab import (
    DEFAULT_JLENS_VERSION,
    DEFAULT_TOP_K,
    JLENS_VERSION_LABELS,
    LAB_MAX_JLENS_LAYER,
    LAB_MIN_JLENS_LAYER,
    LAB_MODEL_REVISION,
    LAB_PRESETS,
    MAX_NEW_TOKENS_LIMIT,
    GenerationEvent,
    GenerationRequest,
    JLensVersion,
    LabCondition,
    TokenPrediction,
    generation_event_to_dict,
    generation_request_to_dict,
    normalize_jlens_version,
)
from .qwen_lab_readout import (
    FROZEN_LENS_SHA256,
    FROZEN_LENS_SHA256_BY_VERSION,
    FROZEN_LENS_V2_SHA256,
    LENS_CONTRACT_QWEN,
    LensProvenance,
    LensReadoutRegistry,
    LiveJLensReadout,
    verify_frozen_lens_hash,
)
from .qwen_lab_runtime import (
    LabOOMError,
    build_random_control_direction,
    operational_status_log,
    wrap_lab_model_for_jlens,
    InterventionEngine,
)
from .gates import OPERATIONAL_ERROR_MESSAGES

FROZEN_DIRECTION_SHA256 = (
    "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
)
DEFAULT_DIRECTION_RELATIVE = Path("artifacts/qwen25_coder_7b/selected_direction.pt")
DEFAULT_LENS_RELATIVE = Path(
    "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
)
DEFAULT_LENS_V2_RELATIVE = Path(
    "artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt"
)
EXPORT_SCHEMA_VERSION = 1
# Task 4 Colab bootstrap should install this exact Gradio release.
GRADIO_PIN_VERSION = "5.49.1"
GENERATION_OUTPUT_COUNT = 16
GENERATION_CONTROL_COMPONENTS: tuple[str, ...] = (
    "condition_dropdown",
    "layer_slider",
    "jlens_selector",
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
SESSION_EXPORT_ROOT = Path(tempfile.gettempdir()) / "qwen_lab_sessions"
SESSION_TTL_SECONDS = 24 * 60 * 60
STATE_TTL_SECONDS = SESSION_TTL_SECONDS
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_LOCAL_PATH_RE = re.compile(
    r"(?:^[A-Za-z]:[\\/]|^/content/|^artifacts[\\/]|selected_direction\.pt|jacobian_lens\.pt)"
)
_CREDENTIAL_RE = re.compile(r"(?:\bhf[_-]?token\b|\bauth[_-]?token\b)\s*[:=]", re.I)

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
    "Var. rank",
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


@dataclass
class LabClientSession:
    session_hash: str
    view_state: LabViewState = field(default_factory=LabViewState)
    generation: GenerationSession = field(default_factory=GenerationSession)
    export_dir: Path = field(default_factory=Path)
    created_at: float = field(default_factory=time.time)


class LabSessionRegistry:
    """Thread-safe per-tab session state; the inference engine stays global."""

    def __init__(self, *, export_root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, LabClientSession] = {}
        self._export_root = export_root or SESSION_EXPORT_ROOT
        self._export_root.mkdir(parents=True, exist_ok=True)

    def _purge_stale(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [
            key
            for key, client in self._sessions.items()
            if client.created_at < cutoff
        ]
        for key in stale:
            self._drop(key)

    def _drop(self, session_hash: str) -> None:
        client = self._sessions.pop(session_hash, None)
        if client is None:
            return
        remove_secure_temp_file(client.view_state.export_path)
        if client.export_dir.is_dir():
            shutil.rmtree(client.export_dir, ignore_errors=True)

    def get_or_create(self, session_hash: str) -> LabClientSession:
        cleaned = str(session_hash or "").strip()
        if not cleaned:
            raise ValueError("session_hash is required")
        with self._lock:
            self._purge_stale()
            client = self._sessions.get(cleaned)
            if client is None:
                export_dir = self._export_root / cleaned
                export_dir.mkdir(parents=True, exist_ok=True)
                client = LabClientSession(session_hash=cleaned, export_dir=export_dir)
                self._sessions[cleaned] = client
            return client

    def get_required(self, session_hash: str) -> LabClientSession:
        with self._lock:
            client = self._sessions.get(str(session_hash).strip())
            if client is None:
                raise KeyError("unknown lab session")
            return client

    def clear(self, session_hash: str, *, controller: LabUIController) -> LabClientSession:
        client = self.get_or_create(session_hash)
        controller.clear_session(client.view_state)
        client.generation.reset_cancel()
        return client

    def release(self, session_hash: str) -> None:
        with self._lock:
            self._drop(str(session_hash).strip())


def lab_operational_log(**fields: Any) -> dict[str, Any]:
    return operational_status_log(**fields)


def new_lab_session_hash() -> str:
    """Per-browser session id; use as Gradio State factory (not a precomputed string)."""
    return secrets.token_hex(16)


def make_lab_session_delete_callback(
    registry: LabSessionRegistry,
) -> Callable[[Any], None]:
    """Release registry state and export dir when Gradio expires the State."""

    def _on_delete(session_hash: Any) -> None:
        if session_hash:
            registry.release(str(session_hash))

    return _on_delete


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
    lens_v2_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = root.resolve()
    direction = (
        Path(direction_path)
        if direction_path is not None
        else root / DEFAULT_DIRECTION_RELATIVE
    )
    lens = Path(lens_path) if lens_path is not None else root / DEFAULT_LENS_RELATIVE
    lens_v2 = (
        Path(lens_v2_path)
        if lens_v2_path is not None
        else root / DEFAULT_LENS_V2_RELATIVE
    )
    if not direction.is_file():
        raise FileNotFoundError(f"direction artifact missing: {direction}")
    if not lens.is_file():
        raise FileNotFoundError(f"lens artifact missing: {lens}")
    if not lens_v2.is_file():
        raise FileNotFoundError(f"lens artifact missing: {lens_v2}")
    return direction.resolve(), lens.resolve(), lens_v2.resolve()


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
    if resolved_revision is None or not str(resolved_revision).strip():
        raise ValueError("resolved model revision is missing")
    observed = str(resolved_revision).strip()
    if observed != LAB_MODEL_REVISION:
        raise ValueError(
            f"resolved revision {observed!r} != expected {LAB_MODEL_REVISION!r}"
        )
    return observed


def _assert_tokenizer_revision(tokenizer: Any, expected_revision: str) -> None:
    for attribute in ("revision", "_commit_hash"):
        observed = getattr(tokenizer, attribute, None)
        if observed is None:
            continue
        text = str(observed).strip()
        if not text:
            raise ValueError("tokenizer revision is missing")
        if text != expected_revision:
            raise ValueError(
                f"tokenizer revision {text!r} != expected {expected_revision!r}"
            )
        return


def resolve_auth_credentials(
    *,
    user: str | None = None,
    password: str | None = None,
) -> tuple[str, str] | None:
    resolved_user = (user if user is not None else os.environ.get("QWEN_LAB_AUTH_USER", "")).strip()
    resolved_password = (
        password if password is not None else os.environ.get("QWEN_LAB_AUTH_PASSWORD", "")
    ).strip()
    if resolved_user and resolved_password:
        return resolved_user, resolved_password
    if resolved_user or resolved_password:
        raise ValueError("auth requires both username and password")
    return None


def require_launch_authentication(
    *,
    share: bool,
    server_name: str,
    auth: tuple[str, str] | None,
) -> tuple[str, str]:
    host = str(server_name).strip().lower()
    exposed = bool(share) or host not in _LOOPBACK_HOSTS
    if not exposed:
        if auth is None:
            return ("", "")
        return auth
    if auth is None or not auth[0].strip() or not auth[1].strip():
        raise ValueError("public lab launch requires complete authentication")
    return auth


def sanitize_error_message(exc: BaseException) -> str:
    if isinstance(exc, LabOOMError):
        return OPERATIONAL_ERROR_MESSAGES["memory"]
    if isinstance(exc, ValueError):
        message = str(exc).strip()
        if message and len(message) < 160 and "\n" not in message:
            return message
    return "Falha na geração. Consulte o status operacional do servidor."


def assert_ui_strings_safe(text: str) -> None:
    lowered = text.casefold()
    if any(marker.casefold() in lowered for marker in UI_FORBIDDEN_SUBSTRINGS):
        raise ValueError("ui strings must not include secrets or local artifact paths")
    if re.search(r"[A-Za-z]:\\\\", text):
        raise ValueError("ui strings must not include windows paths")
    if "/content/" in lowered:
        raise ValueError("ui strings must not include colab paths")


def _string_is_local_path(value: str) -> bool:
    return bool(_LOCAL_PATH_RE.search(value.strip()))


def _string_is_credential_leak(value: str) -> bool:
    return bool(_CREDENTIAL_RE.search(value))


def assert_export_document_safe(document: Mapping[str, Any]) -> None:
    def visit(node: Any, *, metadata: bool) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lowered = str(key).casefold()
                if metadata and lowered in EXPORT_FORBIDDEN_KEYS:
                    raise ValueError(f"export must not include forbidden key {key!r}")
                child_meta = metadata or lowered in {
                    "parameters",
                    "provenance",
                    "artifacts",
                    "paths",
                }
                if lowered == "request":
                    visit(value, metadata=False)
                elif lowered == "events":
                    for item in value:
                        visit(item, metadata=False)
                else:
                    visit(value, metadata=child_meta)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item, metadata=metadata)
        elif isinstance(node, str) and metadata:
            if _string_is_local_path(node):
                raise ValueError("export metadata must not include local paths")
            if _string_is_credential_leak(node):
                raise ValueError("export metadata must not include credentials")

    visit(document, metadata=True)


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


def _load_versioned_readout(
    *,
    path: Path,
    bound: Any,
    decode_token: Any,
    version: JLensVersion,
    lens_contract: Literal["qwen", "fixture"],
) -> tuple[LiveJLensReadout, LensProvenance]:
    from jlens.lens import JacobianLens

    digest = FROZEN_LENS_SHA256_BY_VERSION[version]
    lens = JacobianLens.load(str(path))
    readout = LiveJLensReadout(
        lens,
        bound,
        decode_token=decode_token,
        lens_path=str(path),
        expected_lens_sha256=digest,
        lens_contract=lens_contract,
    )
    provenance = LensProvenance(
        version=version,
        sha256=digest,
        label=JLENS_VERSION_LABELS[version],
    )
    return readout, provenance


def build_production_stack(
    *,
    model: Any,
    tokenizer: Any,
    direction_path: Path,
    lens_path: Path,
    lens_v2_path: Path,
    lens_contract: Literal["qwen", "fixture"] = LENS_CONTRACT_QWEN,
) -> tuple[InterventionEngine, LensReadoutRegistry]:
    assert_jlens_contract()
    direction = load_direction(direction_path)
    blocks, attentions, mlps = transformer_module_lists(model)
    bound = wrap_lab_model_for_jlens(model, tokenizer)
    decode_token = lambda token_id: tokenizer.decode(  # noqa: E731
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    readout_v1, provenance_v1 = _load_versioned_readout(
        path=lens_path,
        bound=bound,
        decode_token=decode_token,
        version="v1",
        lens_contract=lens_contract,
    )
    readout_v2, provenance_v2 = _load_versioned_readout(
        path=lens_v2_path,
        bound=bound,
        decode_token=decode_token,
        version="v2",
        lens_contract=lens_contract,
    )
    registry = LensReadoutRegistry(
        {
            "v1": (readout_v1, provenance_v1),
            "v2": (readout_v2, provenance_v2),
        }
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
        readouts=registry,
    )
    return engine, registry


def _lens_preflight_summary(
    *,
    device_name: Any = None,
    model_revision: str | None = None,
    direction_digest: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "jlens_version": DEFAULT_JLENS_VERSION,
        "lens_sha256": _short_hash(FROZEN_LENS_V2_SHA256),
        "lens_v1_sha256": _short_hash(FROZEN_LENS_SHA256),
        "lens_v2_sha256": _short_hash(FROZEN_LENS_V2_SHA256),
    }
    if device_name is not None:
        summary["device_name"] = device_name
    if model_revision is not None:
        summary["model_revision"] = _short_hash(model_revision)
    if direction_digest is not None:
        summary["direction_sha256"] = _short_hash(direction_digest)
    return summary


def bootstrap_intervention_lab(
    root: Path,
    *,
    direction_path: str | Path | None = None,
    lens_path: str | Path | None = None,
    lens_v2_path: str | Path | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] | None = None,
    load_model: Callable[..., tuple[Any, Any, Any]] | None = None,
    load_stack: Callable[..., Any] | None = None,
) -> LabRuntimeBundle:
    probe = gpu_probe or _default_gpu_probe
    preflight = require_a100_preflight(probe)
    root = root.resolve()
    direction_file, lens_file, lens_v2_file = resolve_lab_artifact_paths(
        root,
        direction_path=direction_path or os.environ.get("QWEN_LAB_DIRECTION"),
        lens_path=lens_path or os.environ.get("QWEN_LAB_LENS"),
        lens_v2_path=lens_v2_path or os.environ.get("QWEN_LAB_LENS_V2"),
    )
    direction_digest = verify_direction_artifact(direction_file)
    verify_frozen_lens_hash(lens_file, FROZEN_LENS_SHA256)
    verify_frozen_lens_hash(lens_v2_file, FROZEN_LENS_V2_SHA256)
    if load_stack is not None:
        built = load_stack(
            root=root,
            direction_path=direction_file,
            lens_path=lens_file,
            lens_v2_path=lens_v2_file,
            lens_contract=LENS_CONTRACT_QWEN,
        )
        engine = built.engine if hasattr(built, "engine") else built
        summary = _lens_preflight_summary(
            device_name=preflight.get("device_name"),
            direction_digest=direction_digest,
        )
        summary.update(dict(getattr(built, "preflight_summary", {})))
        summary.setdefault("jlens_version", DEFAULT_JLENS_VERSION)
        summary.setdefault("lens_sha256", _short_hash(FROZEN_LENS_V2_SHA256))
        return LabRuntimeBundle(engine=engine, preflight_summary=summary)

    loader = load_model or _load_bf16_model
    model, tokenizer, _dtype = loader(
        model_name=QWEN_MODEL,
        revision=LAB_MODEL_REVISION,
    )
    resolved = getattr(model.config, "_commit_hash", None)
    revision = _assert_model_revision(resolved)
    _assert_tokenizer_revision(tokenizer, LAB_MODEL_REVISION)
    engine, _registry = build_production_stack(
        model=model,
        tokenizer=tokenizer,
        direction_path=direction_file,
        lens_path=lens_file,
        lens_v2_path=lens_v2_file,
        lens_contract=LENS_CONTRACT_QWEN,
    )
    summary = _lens_preflight_summary(
        device_name=preflight.get("device_name"),
        model_revision=revision,
        direction_digest=direction_digest,
    )
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
        j_lens_version: str = DEFAULT_JLENS_VERSION,
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
            j_lens_version=normalize_jlens_version(j_lens_version),
        )

    @staticmethod
    def format_status_message(
        *,
        status: str | None,
        version: str,
        sha256: str | None,
    ) -> str:
        label = JLENS_VERSION_LABELS[normalize_jlens_version(version)]
        prefix = f"Estado: {status}" if status else "Gerando…"
        if sha256:
            return f"{prefix} | J-lens {label} | {sha256}"
        return f"{prefix} | J-lens {label}"

    def generation_lock_updates(self) -> tuple[ComponentUpdate, ...]:
        targets = (
            "condition_dropdown",
            "layer_slider",
            "jlens_selector",
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
            ComponentUpdate(component="jlens_selector", interactive=True),
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
                "n/d"
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
            f"Camada {event.j_lens_layer} | passo {event.step} | "
            f"token produzido: {event.generated_token_text} (id {event.generated_token_id})"
        )
        state.events.append(event)
        version = getattr(event, "j_lens_version", DEFAULT_JLENS_VERSION)
        digest = getattr(event, "j_lens_sha256", None)
        if event.status:
            state.terminal_status = event.status
        state.status_message = self.format_status_message(
            status=event.status,
            version=version,
            sha256=digest,
        )

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
        state.status_message = self.format_status_message(
            status=None,
            version=request.j_lens_version,
            sha256=FROZEN_LENS_SHA256_BY_VERSION[request.j_lens_version],
        )
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
            terminal = getattr(engine, "terminal_status", None) or state.terminal_status
            state.terminal_status = terminal
            digest = None
            if state.events:
                digest = state.events[-1].j_lens_sha256
            if digest is None:
                digest = FROZEN_LENS_SHA256_BY_VERSION[request.j_lens_version]
            if terminal:
                state.status_message = self.format_status_message(
                    status=terminal,
                    version=request.j_lens_version,
                    sha256=digest,
                )
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
        version = normalize_jlens_version(state.last_request.j_lens_version)
        digest = None
        if state.events:
            digest = state.events[-1].j_lens_sha256
        if not digest:
            digest = FROZEN_LENS_SHA256_BY_VERSION[version]
        document = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "terminal_status": terminal,
            "request": generation_request_to_dict(state.last_request),
            "events": [generation_event_to_dict(event) for event in state.events],
            "parameters": {
                "model_revision": LAB_MODEL_REVISION,
                "direction_sha256": FROZEN_DIRECTION_SHA256,
                "j_lens_version": version,
                "lens_sha256": digest,
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


def sync_preset_updates(
    controller: LabUIController,
    condition: LabCondition,
) -> tuple[Any, Any]:
    """Return exactly two Gradio updates for condition + manual coefficient."""
    import gradio as gr

    resolved = controller.resolve_preset(condition)
    manual = controller.manual_coefficient_update(
        condition=resolved.condition,
        manual_mode=False,
        coefficient=resolved.coefficient,
    )
    return (
        gr.update(value=resolved.condition),
        gr.update(value=manual.value, interactive=manual.interactive),
    )


def pack_generation_outputs(
    state: LabViewState,
    controller: LabUIController,
    *,
    condition: LabCondition,
    manual_mode: bool,
    lock: Sequence[ComponentUpdate] = (),
    unlock: Sequence[ComponentUpdate] = (),
) -> tuple[Any, ...]:
    """Pack one yield row for the generate callback (fixed cardinality)."""
    import gradio as gr

    lock_map = {item.component: item for item in lock}
    unlock_map = {item.component: item for item in unlock}

    def control(name: str) -> Any:
        if name in lock_map:
            item = lock_map[name]
            kwargs: dict[str, Any] = {"interactive": False}
            if item.value is not None:
                kwargs["value"] = item.value
            return gr.update(**kwargs)
        if name in unlock_map:
            item = unlock_map[name]
            kwargs = {}
            if item.value is not None:
                kwargs["value"] = item.value
            if item.interactive is not None:
                kwargs["interactive"] = item.interactive
            return gr.update(**kwargs) if kwargs else gr.update()
        if unlock:
            resolved = (
                controller.manual_coefficient_update(
                    condition=condition,
                    manual_mode=manual_mode,
                    coefficient=0.0,
                )
                if name == "manual_coefficient"
                else None
            )
            if resolved is not None and name == "manual_coefficient":
                return gr.update(value=resolved.value, interactive=resolved.interactive)
        return gr.update()

    return (
        control("condition_dropdown"),
        control("layer_slider"),
        control("jlens_selector"),
        control("preset_radio"),
        control("manual_mode"),
        control("manual_coefficient"),
        control("top_k"),
        control("temperature"),
        control("top_p"),
        control("seed"),
        control("max_new_tokens"),
        control("generate_button"),
        state.response_text,
        state.status_message or "",
        state.prediction_caption,
        state.prediction_rows,
    )


def create_run_generation_handler(
    engine: InterventionEngine,
    registry: LabSessionRegistry,
    controller: LabUIController | None = None,
) -> Callable[..., Iterator[tuple[Any, ...]]]:
    controller = controller or LabUIController()

    def run_generation(
        session_hash: str,
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
        j_lens_version: str = DEFAULT_JLENS_VERSION,
    ) -> Iterator[tuple[Any, ...]]:
        client = registry.get_or_create(session_hash)
        condition_name = condition  # type: ignore[assignment]
        generator_exit = False
        frozen_version = normalize_jlens_version(j_lens_version)
        try:
            request = controller.build_generation_request(
                prompt=prompt_text,
                condition=condition_name,
                manual_mode=manual_enabled,
                coefficient=coefficient,
                j_lens_layer=int(layer),
                top_k=int(top_k_value),
                temperature=float(temp),
                top_p=float(nucleus),
                seed=int(seed_value),
                max_new_tokens=int(max_tokens),
                j_lens_version=frozen_version,
            )
            for tick in controller.stream_generation(
                engine,
                request,
                state=client.view_state,
                session=client.generation,
            ):
                yield pack_generation_outputs(
                    client.view_state,
                    controller,
                    condition=condition_name,
                    manual_mode=manual_enabled,
                    lock=tick.lock,
                )
        except GeneratorExit:
            generator_exit = True
            raise
        except Exception as exc:
            client.view_state.status_message = sanitize_error_message(exc)
        finally:
            if not generator_exit:
                unlock = controller.generation_unlock_updates(
                    condition=condition_name,
                    manual_mode=manual_enabled,
                )
                yield pack_generation_outputs(
                    client.view_state,
                    controller,
                    condition=condition_name,
                    manual_mode=manual_enabled,
                    unlock=unlock,
                )

    return run_generation


def build_gradio_app(
    engine: InterventionEngine,
    *,
    preflight_summary: Mapping[str, Any],
    auth: tuple[str, str] | None = None,
    session_registry: LabSessionRegistry | None = None,
) -> Any:
    import gradio as gr

    controller = LabUIController()
    registry = session_registry or LabSessionRegistry()
    run_generation = create_run_generation_handler(engine, registry, controller)

    preflight_line = (
        f"GPU: {preflight_summary.get('device_name', 'n/d')} | "
        f"revisão {preflight_summary.get('model_revision', 'n/d')} | "
        f"direção {preflight_summary.get('direction_sha256', 'n/d')} | "
        f"J-lens {preflight_summary.get('jlens_version', DEFAULT_JLENS_VERSION)} "
        f"{preflight_summary.get('lens_sha256', 'n/d')} | "
        f"v1 {preflight_summary.get('lens_v1_sha256', 'n/d')} | "
        f"v2 {preflight_summary.get('lens_v2_sha256', 'n/d')}"
    )

    session_delete = make_lab_session_delete_callback(registry)
    with gr.Blocks(css=LAB_CSS, title="Laboratório de intervenção Qwen") as demo:
        session_hash = gr.State(
            new_lab_session_hash,
            time_to_live=STATE_TTL_SECONDS,
            delete_callback=session_delete,
        )
        gr.Markdown(
            '<div class="lab-header"><h1>Laboratório de intervenção</h1></div>',
        )
        gr.Markdown(f'<div class="lab-disclaimer">{LAB_DISCLAIMER}</div>')
        gr.Markdown(f'<div class="lab-preflight">{preflight_line}</div>')

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
                jlens_selector = gr.Radio(
                    choices=[
                        ("v2 (contexto longo)", "v2"),
                        ("v1 (100 amostras)", "v1"),
                    ],
                    value=DEFAULT_JLENS_VERSION,
                    label="J-lens",
                )
                top_k = gr.Slider(1, 20, value=DEFAULT_TOP_K, step=1, label="Top-k exibido")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperatura")
                top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Top-p")
                seed = gr.Number(value=0, precision=0, label="Seed")
                max_new_tokens = gr.Slider(1, MAX_NEW_TOKENS_LIMIT, value=64, step=1, label="Máx. novos tokens")
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

        def sync_preset(condition: str) -> tuple[Any, Any]:
            return sync_preset_updates(controller, condition)  # type: ignore[arg-type]

        preset_radio.change(
            sync_preset,
            inputs=[preset_radio],
            outputs=[condition_dropdown, manual_coefficient],
        )

        condition_dropdown.change(
            sync_preset,
            inputs=[condition_dropdown],
            outputs=[condition_dropdown, manual_coefficient],
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

        generate_button.click(
            run_generation,
            inputs=[
                session_hash,
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
                jlens_selector,
            ],
            outputs=[
                condition_dropdown,
                layer_slider,
                jlens_selector,
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

        def on_stop(hash_value: str) -> None:
            client = registry.get_or_create(hash_value)
            client.generation.cancel_event.set()

        stop_button.click(on_stop, inputs=[session_hash], outputs=[])

        def on_clear(hash_value: str) -> tuple[str, list[list[Any]], str, str, Any]:
            client = registry.clear(hash_value, controller=controller)
            return "", [], "", client.view_state.status_message, gr.update(value=None)

        clear_button.click(
            on_clear,
            inputs=[session_hash],
            outputs=[response, predictions, prediction_caption, status_message, export_file],
        )

        def on_export(hash_value: str) -> tuple[Any, str]:
            client = registry.get_or_create(hash_value)
            try:
                path = controller.export_session(
                    client.view_state,
                    explicit=True,
                    holder=client.export_dir,
                )
            except ValueError as exc:
                return gr.update(value=None), sanitize_error_message(exc)
            return gr.update(value=str(path)), "Exportação pronta para download."

        export_button.click(
            on_export,
            inputs=[session_hash],
            outputs=[export_file, status_message],
        )

    demo.engine = engine  # type: ignore[attr-defined]
    demo.auth = auth  # type: ignore[attr-defined]
    demo.session_registry = registry  # type: ignore[attr-defined]
    demo.gradio_pin_version = GRADIO_PIN_VERSION  # type: ignore[attr-defined]
    demo.lab_session_state = session_hash  # type: ignore[attr-defined]
    demo.lab_session_delete_callback = session_delete  # type: ignore[attr-defined]
    return demo