from __future__ import annotations

import inspect
import math
import threading
from types import SimpleNamespace

import pytest
import torch
from jlens.lens import JacobianLens
from torch import nn

from refusal_direction_replication.directions import CandidateDirection
from refusal_direction_replication.gates import OPERATIONAL_ERROR_MESSAGES
from refusal_direction_replication.interventions import (
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
)
from refusal_direction_replication.pipeline import (
    random_direction_control,
    temporary_hooks,
    transformer_module_lists,
)
from refusal_direction_replication.qwen_code_eval import COEFFICIENTS, CONDITIONS
from refusal_direction_replication.qwen_lab import (
    DEFAULT_TOP_K,
    GenerationRequest,
    escape_token_display,
)
from refusal_direction_replication.qwen_lab_readout import LiveJLensReadout


RANDOM_CONTROL_SEED = 1042
CACHE_PARITY_ATOL = 1e-4


class LabTokenizer:
    def __init__(self, *, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.add_bos_token = False
        self.chat_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        self.chat_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        content = messages[0]["content"]
        text = f"<user>{content}</user>"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = [2 + (ord(character) % 20) for character in text]
        return ids or [2]

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(97 + (int(token) % 26)) for token in ids)

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        if isinstance(text, list):
            text = text[0]
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeBlock(nn.Module):
    def __init__(self, d_model: int, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor, *args, **kwargs):
        del args, kwargs
        changed = hidden + 0.05 * self.proj(hidden)
        if self.tuple_output:
            return (changed, {"aux": True})
        return changed


class FakeCausalLM(nn.Module):
    def __init__(
        self,
        *,
        n_layers: int = 4,
        d_model: int = 8,
        vocab_size: int = 32,
        seed: int = 0,
        tuple_output: bool = False,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                FakeBlock(d_model, tuple_output=tuple_output)
                for _ in range(n_layers)
            ]
        )
        self.config = SimpleNamespace(eos_token_id=0)
        self._fail_next: BaseException | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if self._fail_next is not None:
            error = self._fail_next
            self._fail_next = None
            raise error
        hidden = self.embed(input_ids)
        for block in self.model.layers:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        logits = self.lm_head(hidden)
        cache = past_key_values
        if use_cache:
            previous = 0 if past_key_values is None else int(past_key_values)
            cache = previous + int(input_ids.shape[1])
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _valid_request(**overrides) -> GenerationRequest:
    payload = dict(
        prompt="hello",
        condition="baseline",
        coefficient=0.0,
        j_lens_layer=2,
        top_k=DEFAULT_TOP_K,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=4,
    )
    payload.update(overrides)
    return GenerationRequest(**payload)


def _tiny_readout(model: FakeCausalLM, tokenizer: LabTokenizer) -> LiveJLensReadout:
    d_model = int(model.embed.embedding_dim)
    n_layers = len(model.model.layers)
    lens = JacobianLens(
        {layer: torch.eye(d_model) for layer in range(n_layers)},
        n_prompts=1,
        d_model=d_model,
    )
    return LiveJLensReadout(
        lens,
        SimpleNamespace(
            unembed=lambda residual: model.lm_head(residual.float()),
            n_layers=n_layers,
            d_model=d_model,
        ),
        decode_token=lambda token_id: tokenizer.decode([int(token_id)]),
        lens_contract="fixture",
    )


def _make_engine(
    *,
    n_layers: int = 4,
    tuple_output: bool = False,
    seed: int = 0,
    j_lens_layer: int = 2,
):
    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    model = FakeCausalLM(
        n_layers=n_layers,
        tuple_output=tuple_output,
        seed=seed,
    )
    tokenizer = LabTokenizer()
    blocks, attentions, mlps = transformer_module_lists(model)
    direction = torch.nn.functional.normalize(torch.arange(8, dtype=torch.float32), dim=0)
    control = random_direction_control(direction, seed=RANDOM_CONTROL_SEED)
    engine = InterventionEngine(
        model,
        tokenizer,
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=direction,
        random_control=control,
        readout=_tiny_readout(model, tokenizer),
    )
    return engine, model, tokenizer, direction, control


def _collect(engine, request: GenerationRequest, cancelled=None):
    return list(engine.generate(request, cancelled=cancelled))


def test_runtime_module_avoids_forbidden_production_paths() -> None:
    from refusal_direction_replication import qwen_lab_runtime as module

    source = inspect.getsource(module)
    assert "generate_responses" not in source
    assert "gradio" not in source.casefold()
    assert "_skip_qwen_contract" not in source


def test_negative_temperature_is_rejected_by_request_contract() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _valid_request(temperature=-0.25)


def test_engine_does_not_revalidate_fixture_or_already_checked_lens(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refusal_direction_replication import qwen_lab_readout as readout_module
    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    calls: list[object] = []

    def spy(*args, **kwargs) -> None:
        del args, kwargs
        calls.append(True)

    monkeypatch.setattr(readout_module, "validate_qwen_lab_lens", spy)
    monkeypatch.setattr(
        "refusal_direction_replication.qwen_lab_runtime.validate_qwen_lab_lens",
        spy,
    )
    engine, model, tokenizer, direction, control = _make_engine()
    _collect(engine, _valid_request(max_new_tokens=1))
    assert calls == []

    path = tmp_path / "lens.pt"
    path.write_bytes(b"lens")
    readout = _tiny_readout(model, tokenizer)
    readout.lens_contract = "qwen"
    readout.lens_validated = False
    readout.lens_path = None
    blocks, attentions, mlps = transformer_module_lists(model)
    qwen_engine = InterventionEngine(
        model,
        tokenizer,
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=direction,
        random_control=control,
        readout=readout,
        lens_path=str(path),
    )
    _collect(qwen_engine, _valid_request(max_new_tokens=1, seed=1))
    _collect(qwen_engine, _valid_request(max_new_tokens=1, seed=2))
    assert len(calls) == 1


def test_qwen_engine_requires_lens_path() -> None:
    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    engine, model, tokenizer, direction, control = _make_engine()
    readout = _tiny_readout(model, tokenizer)
    readout.lens_contract = "qwen"
    readout.lens_validated = False
    readout.lens_path = None
    blocks, attentions, mlps = transformer_module_lists(model)
    with pytest.raises(ValueError, match="lens_path"):
        InterventionEngine(
            model,
            tokenizer,
            blocks=blocks,
            attentions=attentions,
            mlps=mlps,
            direction=direction,
            random_control=control,
            readout=readout,
        )


def test_baseline_installs_no_intervention_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    from refusal_direction_replication import qwen_lab_runtime as module

    installed: dict[str, list] = {}
    original = module.temporary_hooks

    def spy(*, pre_hooks=(), forward_hooks=()):
        installed["pre"] = list(pre_hooks)
        installed["forward"] = list(forward_hooks)
        return original(pre_hooks=pre_hooks, forward_hooks=forward_hooks)

    monkeypatch.setattr(module, "temporary_hooks", spy)
    engine, model, _tokenizer, _direction, _control = _make_engine()
    request = _valid_request(condition="baseline", coefficient=0.0, max_new_tokens=2)
    _collect(engine, request)
    blocks, _attentions, _mlps = transformer_module_lists(model)
    assert installed["pre"] == []
    assert [module_ for module_, _hook in installed["forward"]] == [
        blocks[request.j_lens_layer]
    ]


def test_four_preset_and_manual_coefficients_match_code_eval_builders() -> None:
    from refusal_direction_replication.qwen_lab_runtime import condition_hook_specs

    model = FakeCausalLM(n_layers=19, seed=3)
    blocks, attentions, mlps = transformer_module_lists(model)
    vector = torch.nn.functional.normalize(torch.randn(8), dim=0)
    control = random_direction_control(vector, seed=RANDOM_CONTROL_SEED)
    hidden = torch.randn(1, 2, 8)

    def apply_specs(pre_hooks, forward_hooks, tensor: torch.Tensor) -> torch.Tensor:
        current = tensor
        for module, hook in pre_hooks:
            if module is blocks[18] or module in blocks:
                result = hook(module, (current,))
                if result is not None:
                    current = result[0] if isinstance(result, tuple) else result
        for module, hook in forward_hooks:
            if module is blocks[18] or module in list(attentions) + list(mlps):
                result = hook(module, (current,), current)
                if result is not None:
                    current = result[0] if isinstance(result, tuple) else result
        return current

    cases = [
        ("baseline", 0.0),
        ("ablation", COEFFICIENTS["ablation"]),
        ("ablation", 0.4),
        ("addition", COEFFICIENTS["addition"]),
        ("addition", 0.7),
        ("random_control", COEFFICIENTS["random_control"]),
        ("random_control", 0.3),
    ]
    for condition, coefficient in cases:
        pre, forward = condition_hook_specs(
            condition=condition,
            coefficient=coefficient,
            blocks=blocks,
            attentions=attentions,
            mlps=mlps,
            direction=vector,
            random_control=control,
        )
        if condition == "baseline":
            assert pre == []
            assert forward == []
            continue
        if condition == "ablation":
            expected_pre, expected_forward = directional_ablation_hook_specs(
                blocks, attentions, mlps, vector, coefficient=coefficient
            )
        elif condition == "random_control":
            expected_pre, expected_forward = directional_ablation_hook_specs(
                blocks, attentions, mlps, control, coefficient=coefficient
            )
        else:
            expected_pre = [
                (
                    blocks[18],
                    residual_intervention_pre_hook(
                        vector, mode="add", coefficient=coefficient
                    ),
                )
            ]
            expected_forward = []
        assert [module for module, _hook in pre] == [
            module for module, _hook in expected_pre
        ]
        assert [module for module, _hook in forward] == [
            module for module, _hook in expected_forward
        ]
        assert torch.allclose(
            apply_specs(pre, forward, hidden),
            apply_specs(expected_pre, expected_forward, hidden),
            atol=1e-6,
        )


def test_random_control_is_deterministic_and_orthogonal() -> None:
    vector = torch.nn.functional.normalize(torch.arange(8, dtype=torch.float32), dim=0)
    first = random_direction_control(vector, seed=RANDOM_CONTROL_SEED)
    second = random_direction_control(vector, seed=RANDOM_CONTROL_SEED)
    assert torch.equal(first, second)
    assert torch.allclose(
        torch.dot(first.float(), vector.float()),
        torch.tensor(0.0),
        atol=1e-5,
    )
    engine, _model, _tokenizer, direction, control = _make_engine(n_layers=19)
    from refusal_direction_replication.qwen_lab_runtime import condition_hook_specs

    pre, _forward = condition_hook_specs(
        condition="random_control",
        coefficient=0.2,
        blocks=engine._blocks,
        attentions=engine._attentions,
        mlps=engine._mlps,
        direction=direction,
        random_control=control,
    )
    hidden = direction.repeat(1, 2, 1)
    changed = pre[0][1](pre[0][0], (hidden,))[0]
    # Ablating the control direction must leave a vector parallel to `direction`
    # almost unchanged, unlike ablating `direction` itself.
    direction_pre, _ = directional_ablation_hook_specs(
        engine._blocks,
        engine._attentions,
        engine._mlps,
        direction,
        coefficient=1.0,
    )
    ablated_direction = direction_pre[0][1](direction_pre[0][0], (hidden,))[0]
    assert changed.norm() > ablated_direction.norm()


def test_one_event_per_token_uses_matching_residual_readout() -> None:
    engine, model, tokenizer, _direction, _control = _make_engine()
    request = _valid_request(max_new_tokens=3, j_lens_layer=2, top_k=4)
    blocks, _attentions, _mlps = transformer_module_lists(model)
    residuals: list[torch.Tensor] = []

    def spy(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        residuals.append(hidden[:, -1, :].detach().clone())

    handle = blocks[request.j_lens_layer].register_forward_hook(spy)
    try:
        events = _collect(engine, request)
    finally:
        handle.remove()
    assert len(events) == request.max_new_tokens
    assert [event.step for event in events] == list(range(1, request.max_new_tokens + 1))
    assert len(residuals) >= len(events)
    captured = residuals[-len(events) :]
    fresh = _tiny_readout(model, tokenizer)
    generated_ids: list[int] = []
    for event, residual in zip(events, captured, strict=True):
        expected = fresh.predict(
            residual.reshape(-1),
            layer=request.j_lens_layer,
            top_k=request.top_k,
        )
        assert [item.token_id for item in event.predictions] == [
            item.token_id for item in expected
        ]
        generated_ids.append(event.generated_token_id)
        raw_token = tokenizer.decode([event.generated_token_id])
        assert event.generated_token_text == escape_token_display(
            token_id=event.generated_token_id,
            text=raw_token,
        )
        assert event.accumulated_text == tokenizer.decode(generated_ids)
        assert event.effective_condition == request.condition
        assert event.effective_coefficient == request.coefficient
        assert event.j_lens_layer == request.j_lens_layer


def test_accumulated_text_decodes_full_id_sequence_not_escaped_concat() -> None:
    engine, _model, tokenizer, _direction, _control = _make_engine()

    def marked_decode(ids, **_kwargs) -> str:
        return "<" + ",".join(str(int(token)) for token in ids) + ">"

    tokenizer.decode = marked_decode  # type: ignore[method-assign]
    events = _collect(engine, _valid_request(max_new_tokens=2, temperature=0.0))
    ids = [event.generated_token_id for event in events]
    assert events[-1].accumulated_text == "<" + ",".join(str(token) for token in ids) + ">"
    assert events[0].accumulated_text == f"<{ids[0]}>"
    assert events[-1].generated_token_text == escape_token_display(
        token_id=ids[-1],
        text=f"<{ids[-1]}>",
    )
    assert events[-1].accumulated_text != (
        events[0].generated_token_text + events[1].generated_token_text
    )


def test_generate_is_a_generator_function() -> None:
    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    assert inspect.isgeneratorfunction(InterventionEngine.generate)


def test_chat_template_uses_generation_prompt() -> None:
    engine, _model, tokenizer, _direction, _control = _make_engine()
    _collect(engine, _valid_request(max_new_tokens=1))
    assert tokenizer.chat_calls
    assert tokenizer.chat_calls[0]["add_generation_prompt"] is True
    assert tokenizer.chat_calls[0]["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_greedy_and_top_p_sampling_are_reproducible() -> None:
    engine, _model, _tokenizer, _direction, _control = _make_engine()
    greedy = _valid_request(temperature=0.0, seed=11, max_new_tokens=3)
    first = [event.generated_token_id for event in _collect(engine, greedy)]
    second = [event.generated_token_id for event in _collect(engine, greedy)]
    assert first == second
    sampled = _valid_request(temperature=0.8, top_p=0.9, seed=11, max_new_tokens=3)
    sampled_first = [event.generated_token_id for event in _collect(engine, sampled)]
    sampled_second = [event.generated_token_id for event in _collect(engine, sampled)]
    assert sampled_first == sampled_second


def test_eos_max_tokens_and_cancellation_terminal_states() -> None:
    engine, model, tokenizer, _direction, _control = _make_engine()
    eos_id = tokenizer.eos_token_id
    _force_token(model, eos_id)
    eos_events = _collect(engine, _valid_request(max_new_tokens=5, temperature=0.0))
    assert eos_events[-1].status == "completed"
    assert eos_events[-1].generated_token_id == eos_id
    assert len(eos_events) == 1

    engine, model, tokenizer, _direction, _control = _make_engine(seed=4)
    _force_token(model, 3)
    max_events = _collect(engine, _valid_request(max_new_tokens=3, temperature=0.0))
    assert len(max_events) == 3
    assert max_events[-1].status == "max_tokens"
    assert all(event.status is None for event in max_events[:-1])

    engine, _model, _tokenizer, _direction, _control = _make_engine(seed=5)
    seen: list[object] = []

    def cancelled() -> bool:
        return len(seen) >= 1

    for event in engine.generate(_valid_request(max_new_tokens=6), cancelled=cancelled):
        seen.append(event)
    assert len(seen) == 1
    assert engine.terminal_status == "cancelled"
    assert seen[-1].status in {None, "cancelled"}


def test_threading_event_cancellation() -> None:
    engine, _model, _tokenizer, _direction, _control = _make_engine()
    flag = threading.Event()
    events = []
    for event in engine.generate(_valid_request(max_new_tokens=5), cancelled=flag):
        events.append(event)
        flag.set()
    assert len(events) == 1
    assert engine.terminal_status == "cancelled"


def test_concurrent_generation_is_rejected() -> None:
    from refusal_direction_replication.qwen_lab_runtime import ConcurrentGenerationError

    engine, _model, _tokenizer, _direction, _control = _make_engine()
    unused = engine.generate(_valid_request(max_new_tokens=3, seed=9))
    assert engine.active is False
    unused.close()
    assert engine.active is False

    iterator = engine.generate(_valid_request(max_new_tokens=5))
    first = next(iterator)
    assert first.step == 1
    assert engine.active is True
    with pytest.raises(ConcurrentGenerationError, match="already active"):
        next(engine.generate(_valid_request(max_new_tokens=2)))
    remaining = list(iterator)
    assert remaining
    assert engine.active is False
    second = _collect(engine, _valid_request(max_new_tokens=1, seed=2))
    assert second[0].step == 1


def test_close_releases_session_after_first_token() -> None:
    engine, _model, _tokenizer, _direction, _control = _make_engine()
    iterator = engine.generate(_valid_request(max_new_tokens=5))
    next(iterator)
    iterator.close()
    assert engine.active is False
    events = _collect(engine, _valid_request(max_new_tokens=1, seed=3))
    assert events[0].step == 1


def test_concurrent_generation_from_two_threads() -> None:
    from refusal_direction_replication.qwen_lab_runtime import ConcurrentGenerationError

    engine, _model, _tokenizer, _direction, _control = _make_engine()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def worker(seed: int) -> None:
        iterator = engine.generate(_valid_request(max_new_tokens=3, seed=seed))
        barrier.wait()
        try:
            list(iterator)
            outcomes.append("ok")
        except ConcurrentGenerationError:
            outcomes.append("err")

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in (1, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("ok") == 1
    assert outcomes.count("err") == 1
    assert engine.active is False
    recovered = _collect(engine, _valid_request(max_new_tokens=1, seed=8))
    assert recovered[0].step == 1


def test_tuple_block_outputs_are_captured_without_replacing_output() -> None:
    engine, model, _tokenizer, _direction, _control = _make_engine(tuple_output=True)
    replacements: list[object] = []
    original = model.model.layers[2].register_forward_hook

    def tracking_register(hook, *args, **kwargs):
        def wrapped(module, inputs, output):
            result = hook(module, inputs, output)
            replacements.append(result)
            return result

        return original(wrapped, *args, **kwargs)

    model.model.layers[2].register_forward_hook = tracking_register  # type: ignore[method-assign]
    events = _collect(engine, _valid_request(max_new_tokens=1, j_lens_layer=2))
    assert events
    assert replacements
    assert all(item is None for item in replacements)


def test_hooks_and_active_state_cleared_on_all_exits() -> None:
    engine, model, _tokenizer, _direction, _control = _make_engine()
    _collect(engine, _valid_request(max_new_tokens=2))
    _assert_clean(engine, model)

    flag = threading.Event()
    iterator = engine.generate(_valid_request(max_new_tokens=5), cancelled=flag)
    next(iterator)
    flag.set()
    list(iterator)
    _assert_clean(engine, model)

    engine, model, _tokenizer, _direction, _control = _make_engine(seed=8)
    model._fail_next = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        _collect(engine, _valid_request(max_new_tokens=2))
    assert engine.terminal_status == "error"
    _assert_clean(engine, model)

    engine, model, _tokenizer, _direction, _control = _make_engine(seed=9)
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    model._fail_next = oom_type("CUDA out of memory")
    with pytest.raises(Exception, match="memory"):
        _collect(engine, _valid_request(max_new_tokens=2))
    assert engine.terminal_status == "error_memory"
    _assert_clean(engine, model)
    recovered = _collect(engine, _valid_request(max_new_tokens=1, seed=3))
    assert recovered[0].step == 1
    _assert_clean(engine, model)


def test_second_generation_succeeds_after_generic_error() -> None:
    engine, model, _tokenizer, _direction, _control = _make_engine()
    model._fail_next = RuntimeError("transient")
    with pytest.raises(RuntimeError, match="transient"):
        _collect(engine, _valid_request(max_new_tokens=2))
    assert engine.terminal_status == "error"
    events = _collect(engine, _valid_request(max_new_tokens=2, seed=4))
    assert len(events) == 2
    assert engine.active is False


def test_only_selected_block_has_observational_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_lab_runtime as module

    installed: dict[str, list] = {}
    original = module.temporary_hooks

    def spy(*, pre_hooks=(), forward_hooks=()):
        installed["forward"] = list(forward_hooks)
        return original(pre_hooks=pre_hooks, forward_hooks=forward_hooks)

    monkeypatch.setattr(module, "temporary_hooks", spy)
    engine, model, _tokenizer, _direction, _control = _make_engine(n_layers=19)
    request = _valid_request(
        condition="addition",
        coefficient=0.5,
        j_lens_layer=4,
        max_new_tokens=1,
    )
    _collect(engine, request)
    blocks, _attentions, _mlps = transformer_module_lists(model)
    capture_modules = [module_ for module_, _hook in installed["forward"]]
    assert capture_modules.count(blocks[4]) == 1
    assert blocks[18] not in capture_modules
    assert all(module_ is blocks[4] for module_ in capture_modules)


def test_operational_logs_omit_prompt_and_generation_text() -> None:
    from refusal_direction_replication.qwen_code_eval import compact_contains_leak
    from refusal_direction_replication.qwen_lab_runtime import operational_status_log

    engine, _model, _tokenizer, _direction, _control = _make_engine()
    events = _collect(engine, _valid_request(prompt="secret prompt", max_new_tokens=2))
    record = operational_status_log(
        status=events[-1].status or engine.terminal_status,
        step=events[-1].step,
        condition=events[-1].effective_condition,
        token_id=events[-1].generated_token_id,
    )
    serialized = str(record).casefold()
    assert "secret prompt" not in serialized
    assert "accumulated_text" not in record
    assert "generated_token_text" not in record
    assert "prompt" not in record
    assert compact_contains_leak(record) is False
    with pytest.raises(ValueError, match="text"):
        operational_status_log(status="completed", prompt="hello")
    with pytest.raises(ValueError, match="text"):
        operational_status_log(
            status="completed",
            accumulated_text=events[-1].accumulated_text,
        )


def test_wrap_helper_does_not_mutate_generation_tokenizer() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from refusal_direction_replication.qwen_lab_runtime import wrap_lab_model_for_jlens

    tokenizer = LabTokenizer()
    tokenizer.add_bos_token = False
    config = Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    bound = wrap_lab_model_for_jlens(model, tokenizer)
    assert tokenizer.add_bos_token is False
    assert bound.n_layers == 2
    assert bound.d_model == 16


def test_incremental_cache_matches_full_forward() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    config = Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        eos_token_id=0,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).eval()
    tokenizer = LabTokenizer(vocab_size=64)
    blocks, attentions, mlps = transformer_module_lists(model)
    d_model = 16
    lens = JacobianLens(
        {0: torch.eye(d_model), 1: torch.eye(d_model)},
        n_prompts=1,
        d_model=d_model,
    )
    readout = LiveJLensReadout(
        lens,
        SimpleNamespace(unembed=lambda residual: model.lm_head(residual.float())),
        decode_token=lambda token_id: tokenizer.decode([int(token_id)]),
        lens_contract="fixture",
    )
    direction = torch.nn.functional.normalize(torch.ones(d_model), dim=0)
    engine = InterventionEngine(
        model,
        tokenizer,
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=direction,
        random_control=random_direction_control(direction, seed=RANDOM_CONTROL_SEED),
        readout=readout,
    )
    request = _valid_request(
        j_lens_layer=1,
        max_new_tokens=2,
        temperature=0.0,
        top_k=4,
    )
    residuals: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        residuals.append(hidden[:, -1, :].detach().clone())

    incremental_logits: list[torch.Tensor] = []
    original_forward = model.forward

    def tracking_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        incremental_logits.append(outputs.logits[:, -1, :].detach().clone())
        return outputs

    model.forward = tracking_forward  # type: ignore[method-assign]
    handle = blocks[1].register_forward_hook(capture)
    try:
        events = _collect(engine, request)
        decode_residual = residuals[1]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": request.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(rendered)["input_ids"]
        first_token = torch.tensor(
            [[events[0].generated_token_id]], dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, first_token], dim=1)
        with torch.no_grad():
            full = original_forward(input_ids=full_ids, use_cache=False)
        full_residual = residuals[-1]
        full_logits = full.logits[:, -1, :].float()
    finally:
        handle.remove()
        model.forward = original_forward  # type: ignore[method-assign]

    assert events[1].generated_token_id == int(full_logits.argmax(dim=-1).item())
    assert torch.allclose(
        decode_residual.float(),
        full_residual.float(),
        atol=CACHE_PARITY_ATOL,
        rtol=0.0,
    )
    assert torch.allclose(
        incremental_logits[1].float(),
        full_logits,
        atol=CACHE_PARITY_ATOL,
        rtol=0.0,
    )


def test_seed_everything_is_called_with_request_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from refusal_direction_replication import qwen_lab_runtime as module

    seeds: list[int] = []
    original = module.seed_everything

    def spy(seed: int) -> None:
        seeds.append(int(seed))
        original(seed)

    monkeypatch.setattr(module, "seed_everything", spy)
    engine, _model, _tokenizer, _direction, _control = _make_engine()
    _collect(engine, _valid_request(seed=123, max_new_tokens=1))
    assert seeds == [123]


def test_oom_raises_stable_lab_error_and_clears_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_lab_runtime as module
    from refusal_direction_replication.qwen_lab_runtime import LabOOMError

    engine, model, _tokenizer, _direction, _control = _make_engine()
    cleared: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append("cache"))
    monkeypatch.setattr(module.gc, "collect", lambda: cleared.append("gc") or 0)
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    model._fail_next = oom_type("CUDA out of memory")
    with pytest.raises(LabOOMError) as info:
        _collect(engine, _valid_request(max_new_tokens=1))
    assert "cache" in cleared
    assert "gc" in cleared
    assert engine.terminal_status == "error_memory"
    assert info.value.error_code == "memory"
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True
    assert str(info.value) == OPERATIONAL_ERROR_MESSAGES["memory"]
    assert "hello" not in str(info.value)
    recovered = _collect(engine, _valid_request(max_new_tokens=1, seed=6))
    assert recovered[0].step == 1


def test_stream_discards_step_tensors_before_yield() -> None:
    from refusal_direction_replication import qwen_lab_runtime as module

    source = inspect.getsource(module.InterventionEngine._stream)
    assert "del outputs" in source
    assert "del logits" in source


def _force_token(model: FakeCausalLM, token_id: int) -> None:
    def boost(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
        boosted = output.clone()
        boosted[..., int(token_id)] = output.max() + 10.0
        return boosted

    model.lm_head.register_forward_hook(boost)


def _assert_clean(engine, model) -> None:
    assert engine.active is False
    blocks, attentions, mlps = transformer_module_lists(model)
    for module in [*blocks, *attentions, *mlps]:
        assert dict(module._forward_hooks) == {}
        assert dict(module._forward_pre_hooks) == {}


class _TrackingReadout:
    def __init__(self, inner: LiveJLensReadout, name: str) -> None:
        self.inner = inner
        self.name = name
        self.calls: list[int] = []
        self.lens_contract = "fixture"
        self.lens_validated = True
        self.lens_path = None
        self._prior_ranks: dict[int, int] = {}

    def predict(self, residual, *, layer: int, top_k: int):
        self.calls.append(int(layer))
        self.inner._prior_ranks = self._prior_ranks
        predictions = self.inner.predict(residual, layer=layer, top_k=top_k)
        self._prior_ranks = dict(self.inner._prior_ranks)
        return predictions


def _registry_engine(readout_v1: LiveJLensReadout, readout_v2: LiveJLensReadout):
    from refusal_direction_replication.qwen_lab import JLENS_VERSION_LABELS
    from refusal_direction_replication.qwen_lab_readout import (
        FROZEN_LENS_SHA256,
        FROZEN_LENS_V2_SHA256,
        LensProvenance,
        LensReadoutRegistry,
    )
    from refusal_direction_replication.qwen_lab_runtime import InterventionEngine

    engine, model, tokenizer, direction, control = _make_engine()
    tracked_v1 = _TrackingReadout(readout_v1, "v1")
    tracked_v2 = _TrackingReadout(readout_v2, "v2")
    registry = LensReadoutRegistry(
        {
            "v1": (
                tracked_v1,
                LensProvenance(
                    version="v1",
                    sha256=FROZEN_LENS_SHA256,
                    label=JLENS_VERSION_LABELS["v1"],
                ),
            ),
            "v2": (
                tracked_v2,
                LensProvenance(
                    version="v2",
                    sha256=FROZEN_LENS_V2_SHA256,
                    label=JLENS_VERSION_LABELS["v2"],
                ),
            ),
        }
    )
    blocks, attentions, mlps = transformer_module_lists(model)
    selected = InterventionEngine(
        model,
        tokenizer,
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=direction,
        random_control=control,
        readouts=registry,
    )
    return selected, tracked_v1, tracked_v2, registry


def test_engine_defaults_to_v2_readout() -> None:
    engine, _model, tokenizer, _direction, _control = _make_engine()
    readout_v1 = _tiny_readout(engine._model, tokenizer)
    readout_v2 = _tiny_readout(engine._model, tokenizer)
    selected, tracked_v1, tracked_v2, _registry = _registry_engine(readout_v1, readout_v2)
    events = _collect(selected, _valid_request(max_new_tokens=2))
    assert tracked_v2.calls == [2, 2]
    assert tracked_v1.calls == []
    assert {event.j_lens_version for event in events} == {"v2"}
    assert {event.j_lens_sha256 for event in events} == {
        "b80c084a5ce083d2fc8589a62c9a9b2542b6a9dc6b5c006018cc4956eef37a7a"
    }


def test_engine_uses_explicit_v1_readout() -> None:
    engine, _model, tokenizer, _direction, _control = _make_engine()
    readout_v1 = _tiny_readout(engine._model, tokenizer)
    readout_v2 = _tiny_readout(engine._model, tokenizer)
    selected, tracked_v1, tracked_v2, _registry = _registry_engine(readout_v1, readout_v2)
    events = _collect(
        selected, _valid_request(max_new_tokens=2, j_lens_version="v1")
    )
    assert tracked_v1.calls == [2, 2]
    assert tracked_v2.calls == []
    assert {event.j_lens_version for event in events} == {"v1"}
    assert {event.j_lens_sha256 for event in events} == {
        "b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e"
    }


def test_engine_freezes_readout_for_entire_generation() -> None:
    engine, _model, tokenizer, _direction, _control = _make_engine()
    readout_v1 = _tiny_readout(engine._model, tokenizer)
    readout_v2 = _tiny_readout(engine._model, tokenizer)
    selected, tracked_v1, tracked_v2, registry = _registry_engine(readout_v1, readout_v2)
    iterator = selected.generate(_valid_request(max_new_tokens=3, j_lens_version="v1"))
    first = next(iterator)
    registry._entries["v1"] = registry._entries["v2"]
    rest = list(iterator)
    events = [first, *rest]
    assert len(events) == 3
    assert tracked_v1.calls == [2, 2, 2]
    assert tracked_v2.calls == []
    assert [event.j_lens_version for event in events] == ["v1", "v1", "v1"]
