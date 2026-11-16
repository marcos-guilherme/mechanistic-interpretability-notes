"""Single-session token-by-token generation for the Qwen intervention lab."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import gc
import math
import threading
import time
from typing import Any

import torch

from .gates import OPERATIONAL_ERROR_MESSAGES
from .interventions import (
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
)
from .model import format_chat_instruction
from .pipeline import random_direction_control, temporary_hooks
from .qwen_code_eval import COMPACT_FORBIDDEN_KEYS, compact_contains_leak
from .qwen_jlens import is_cuda_oom, wrap_hf_model_for_jlens
from .qwen_lab import (
    LAB_DIRECTION_LAYER,
    GenerationEvent,
    GenerationRequest,
    escape_token_display,
)
from .qwen_lab_readout import (
    FROZEN_LENS_SHA256,
    LENS_CONTRACT_FIXTURE,
    LENS_CONTRACT_QWEN,
    LiveJLensReadout,
    validate_qwen_lab_lens,
)
from .runtime import seed_everything


RANDOM_CONTROL_SEED = 1042
Cancelled = Callable[[], bool] | Any | None
LAB_LOG_FORBIDDEN = COMPACT_FORBIDDEN_KEYS | {
    "accumulated_text",
    "generated_token_text",
    "generated_token",
}


class ConcurrentGenerationError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("generation already active")


class LabOOMError(RuntimeError):
    error_code = "memory"

    def __init__(self) -> None:
        super().__init__(OPERATIONAL_ERROR_MESSAGES["memory"])
        self.error_code = "memory"


class _TokenizerProxy:
    """Delegates to a tokenizer without writing ``add_bos_token`` on it."""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(
            self,
            "add_bos_token",
            getattr(inner, "add_bos_token", False),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"add_bos_token", "_inner"}:
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_inner")(*args, **kwargs)


def wrap_lab_model_for_jlens(model: Any, tokenizer: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("force_bos", False)
    return wrap_hf_model_for_jlens(model, _TokenizerProxy(tokenizer), **kwargs)


def operational_status_log(**fields: Any) -> dict[str, Any]:
    forbidden = {key.casefold() for key in LAB_LOG_FORBIDDEN}
    if any(str(key).casefold() in forbidden for key in fields):
        raise ValueError("operational logs must not contain raw text")
    payload = {key: value for key, value in fields.items() if value is not None}
    if compact_contains_leak(payload):
        raise ValueError("operational logs must not contain raw text")
    return payload


def condition_hook_specs(
    *,
    condition: str,
    coefficient: float,
    blocks: Sequence[Any],
    attentions: Sequence[Any],
    mlps: Sequence[Any],
    direction: torch.Tensor,
    random_control: torch.Tensor,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    if condition == "baseline":
        return [], []
    if condition == "ablation":
        return directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            direction,
            coefficient=float(coefficient),
        )
    if condition == "random_control":
        return directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            random_control,
            coefficient=float(coefficient),
        )
    if condition == "addition":
        return (
            [
                (
                    blocks[LAB_DIRECTION_LAYER],
                    residual_intervention_pre_hook(
                        direction,
                        mode="add",
                        coefficient=float(coefficient),
                    ),
                )
            ],
            [],
        )
    raise ValueError(f"unknown condition: {condition}")


def build_random_control_direction(direction: torch.Tensor) -> torch.Tensor:
    return random_direction_control(direction, seed=RANDOM_CONTROL_SEED)


def _tensor_direction(direction: Any) -> torch.Tensor:
    vector = getattr(direction, "vector", direction)
    if not isinstance(vector, torch.Tensor):
        raise TypeError("direction must be a tensor or expose .vector")
    return vector


def _is_cancelled(cancelled: Cancelled) -> bool:
    if cancelled is None:
        return False
    is_set = getattr(cancelled, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if callable(cancelled):
        return bool(cancelled())
    return bool(cancelled)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError, AttributeError):
        return torch.device("cpu")


def _gpu_memory_bytes() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    return None


def _encode_prompt(tokenizer: Any, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(input_ids)
    else:
        mask = mask.to(device)
    return input_ids, mask


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if float(top_p) >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    restored = torch.empty_like(logits).scatter_(-1, sorted_indices, filtered)
    return restored


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> int:
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    vector = logits.float().reshape(-1)
    if temperature == 0.0:
        return int(torch.argmax(vector).item())
    scaled = vector / float(temperature)
    filtered = _apply_top_p(scaled, top_p)
    probs = torch.softmax(filtered, dim=-1)
    if generator.device != probs.device:
        raise RuntimeError("sampling generator device must match logits")
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item())


def _clear_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _drop_exception_frames(exc: BaseException) -> None:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _release_after_oom() -> None:
    gc.collect()
    _clear_cuda_cache()


class InterventionEngine:
    """Apply frozen intervention semantics and stream synchronized J-lens events."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        blocks: Sequence[Any],
        attentions: Sequence[Any],
        mlps: Sequence[Any],
        direction: Any,
        random_control: torch.Tensor,
        readout: LiveJLensReadout,
        lens_path: str | None = None,
        expected_lens_sha256: str | None = FROZEN_LENS_SHA256,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._blocks = list(blocks)
        self._attentions = list(attentions)
        self._mlps = list(mlps)
        self._direction = _tensor_direction(direction)
        self._random_control = random_control
        self._readout = readout
        self._lock = threading.Lock()
        self._session_held = False
        self._active = False
        self.terminal_status: str | None = None
        self._bind_lens_contract(
            readout,
            lens_path=lens_path,
            expected_lens_sha256=expected_lens_sha256,
        )

    @property
    def active(self) -> bool:
        return self._active

    def _bind_lens_contract(
        self,
        readout: LiveJLensReadout,
        *,
        lens_path: str | None,
        expected_lens_sha256: str | None,
    ) -> None:
        contract = getattr(readout, "lens_contract", LENS_CONTRACT_QWEN)
        if contract == LENS_CONTRACT_FIXTURE:
            return
        if contract != LENS_CONTRACT_QWEN:
            raise ValueError(f"unknown lens_contract: {contract!r}")
        if getattr(readout, "lens_validated", False):
            return
        path = lens_path if lens_path is not None else getattr(readout, "lens_path", None)
        if path is None:
            raise ValueError("lens_path is required for Qwen lab lens validation")
        if expected_lens_sha256 is None:
            raise ValueError("expected_lens_sha256 is required for Qwen lab lens validation")
        validate_qwen_lab_lens(
            readout._lens,
            lens_path=str(path),
            expected_lens_sha256=expected_lens_sha256,
        )
        readout.lens_path = str(path)
        readout.lens_validated = True

    def _end_session(self) -> None:
        if not self._session_held:
            return
        self._session_held = False
        self._active = False
        self._lock.release()

    def generate(
        self,
        request: GenerationRequest,
        cancelled: Cancelled = None,
    ) -> Iterator[GenerationEvent]:
        if not self._lock.acquire(blocking=False):
            raise ConcurrentGenerationError()
        self._session_held = True
        self._active = True
        self.terminal_status = None
        try:
            yield from self._stream(request, cancelled)
        except Exception as exc:
            if is_cuda_oom(exc):
                self.terminal_status = "error_memory"
                _drop_exception_frames(exc)
                _release_after_oom()
                raise LabOOMError() from None
            self.terminal_status = "error"
            raise
        finally:
            self._end_session()

    def _stream(
        self,
        request: GenerationRequest,
        cancelled: Cancelled,
    ) -> Iterator[GenerationEvent]:
        seed_everything(int(request.seed))
        self._readout._prior_ranks = {}
        device = _model_device(self._model)
        generator = torch.Generator(device=device).manual_seed(int(request.seed))
        pre_hooks, forward_hooks = condition_hook_specs(
            condition=str(request.condition),
            coefficient=float(request.coefficient),
            blocks=self._blocks,
            attentions=self._attentions,
            mlps=self._mlps,
            direction=self._direction,
            random_control=self._random_control,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["residual"] = hidden[:, -1, :].detach()

        selected = self._blocks[int(request.j_lens_layer)]
        forward_hooks = [*forward_hooks, (selected, capture_hook)]
        rendered = format_chat_instruction(self._tokenizer, request.prompt)
        input_ids, attention_mask = _encode_prompt(self._tokenizer, rendered, device)
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        past = None
        next_ids = input_ids
        cache_position = torch.arange(input_ids.shape[1], device=device)
        generated_ids: list[int] = []
        max_new_tokens = int(request.max_new_tokens)

        with temporary_hooks(pre_hooks=pre_hooks, forward_hooks=forward_hooks):
            with torch.inference_mode():
                for step in range(1, max_new_tokens + 1):
                    if _is_cancelled(cancelled):
                        self.terminal_status = "cancelled"
                        operational_status_log(
                            status="cancelled",
                            step=step - 1 if step > 1 else None,
                            condition=request.condition,
                        )
                        return
                    started = time.perf_counter()
                    try:
                        outputs = self._model(
                            input_ids=next_ids,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                            cache_position=cache_position,
                        )
                        past = getattr(outputs, "past_key_values", None)
                        logits = outputs.logits[:, -1, :].float()
                        token_id = _sample_token(
                            logits,
                            temperature=float(request.temperature),
                            top_p=float(request.top_p),
                            generator=generator,
                        )
                        residual = captured.pop("residual")
                        predictions = self._readout.predict(
                            residual.reshape(-1),
                            layer=int(request.j_lens_layer),
                            top_k=int(request.top_k),
                        )
                    except Exception:
                        captured.clear()
                        past = None
                        outputs = None
                        logits = None
                        residual = None
                        raise
                    del outputs
                    del logits
                    del residual
                    generated_ids.append(int(token_id))
                    raw_text = _decode_token(self._tokenizer, token_id)
                    display = escape_token_display(token_id=token_id, text=raw_text)
                    accumulated = self._tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    status = None
                    if eos_id is not None and token_id == int(eos_id):
                        status = "completed"
                    elif step >= max_new_tokens:
                        status = "max_tokens"
                    if _is_cancelled(cancelled):
                        status = "cancelled"
                    event = GenerationEvent(
                        step=step,
                        generated_token_id=token_id,
                        generated_token_text=display,
                        accumulated_text=accumulated,
                        effective_condition=request.condition,
                        effective_coefficient=float(request.coefficient),
                        j_lens_layer=int(request.j_lens_layer),
                        predictions=predictions,
                        step_time_seconds=time.perf_counter() - started,
                        gpu_memory_bytes=_gpu_memory_bytes(),
                        status=status,
                    )
                    if status is not None:
                        self.terminal_status = status
                        operational_status_log(
                            status=status,
                            step=step,
                            condition=request.condition,
                            token_id=token_id,
                        )
                    yield event
                    if status is not None:
                        return
                    if _is_cancelled(cancelled):
                        self.terminal_status = "cancelled"
                        operational_status_log(
                            status="cancelled",
                            step=step,
                            condition=request.condition,
                            token_id=token_id,
                        )
                        return
                    next_ids = torch.tensor(
                        [[token_id]],
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    ones = torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask, ones], dim=1)
                    cache_position = torch.tensor(
                        [input_ids.shape[1] + step - 1],
                        device=device,
                    )
