# Task 1 implementation report

## Summary

Generalized the refusal-direction pipeline for an isolated Qwen2.5-Coder-7B
profile while preserving Gemma defaults, paths, and public API compatibility.

## RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_profile.py -q
```

Result (before implementation):

```text
ImportError: cannot import name 'resolve_refusal_token_ids' from
'refusal_direction_replication.model'
ERROR tests/test_qwen_profile.py
```

`configs/qwen25_coder_7b.json` was also missing, so config field tests failed at
collection/import time as expected for strict TDD.

## GREEN evidence

Focused new tests:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_profile.py -q
```

Result: `12 passed`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `49 passed in 6.02s` (baseline 37 + 12 new)

## Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/configs/qwen25_coder_7b.json` | Added Qwen profile |
| `projects/refusal_direction/src/refusal_direction_replication/model.py` | Generic loader, chat template formatting, refusal prefix token resolution, chat boundary offsets |
| `projects/refusal_direction/src/refusal_direction_replication/pipeline.py` | Profile-aware config/paths, parameterized refusal tokens and format style, `transformer_module_lists`, revision-aware loading |
| `projects/refusal_direction/tests/test_qwen_profile.py` | Added focused offline tests |

## Self-review

- **Gemma defaults:** `load_gemma`, `format_instruction`, `tokenize_instructions()` default, `GEMMA_REFUSAL_TOKEN_IDS`, and stage entrypoints (`profile="gemma_2b.json"`) unchanged for existing callers.
- **Namespaced outputs:** Qwen config points at `results/qwen25_coder_7b` and `artifacts/qwen25_coder_7b`; Gemma still uses `results/` and `artifacts/`.
- **28 Qwen candidates:** Config sets `position_offsets: [-1]`; unit test asserts one candidate per layer with that offset.
- **Refusal proxy:** Prefix-based `resolve_refusal_token_ids` is wired for profile configs; Gemma keeps the fixed single-token proxy.
- **No forbidden edits:** Canonical Gemma results, notebooks, Colab transport, and other projects were not touched.

## Concerns

1. **Chat-template offsets vs config:** `end_of_instruction_offsets(chat_template)` can return multiple suffix positions (e.g. `(-2, -1)` in the mock), but Qwen extraction uses config `position_offsets: [-1]` only. Extraction behavior matches the brief; the helper is diagnostic/generic, not the sole selector for Qwen.
2. **`load_causal_lm` on Gemma:** `_load_model_for_config` now passes `model_revision` from config (`"main"` for Gemma). This should be equivalent but was not exercised against live weights in CI.
3. **Shared datasets:** Upstream splits remain under `artifacts/datasets/` for all profiles; Qwen-specific artifact dirs do not duplicate dataset JSON unless a future task adds them.

## Status

`DONE_WITH_CONCERNS`