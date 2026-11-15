# Task 3 implementation report

## Summary

Implemented a resumable Qwen causal pilot with unfiltered splits, 28
offset-`-1` candidates, pure abort gates, a Colab-safe remote entrypoint, and
an offline compact-artifact validator. Gemma defaults, four-split sampling,
behavior filtering, paths, and command builders are unchanged. No Colab session
or weight download was started.

## RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py -q --tb=line
```

Result (before implementation):

```text
ImportError while importing test module '.../tests/test_qwen_pilot.py'.
ModuleNotFoundError: No module named 'refusal_direction_replication.gates'
ERROR tests/test_qwen_pilot.py
Interrupted: 1 error during collection
1 error in 5.50s
```

The suite failed because the gate module, Qwen pilot controls, test-split
fetch, orchestrator, validator, and remote script did not exist yet.

## GREEN evidence

Focused new tests:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py -q
```

Result: `43 passed in 9.90s`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `127 passed in 11.30s` (Task 2 baseline 84 + 43 new)

## Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/configs/qwen25_coder_7b.json` | Test split, no behavior filter, val selection, test finals, gate thresholds |
| `projects/refusal_direction/scripts/fetch_upstream_splits.py` | Fetch `harmful_test.json` and `harmless_test.json` from the pinned commit |
| `projects/refusal_direction/src/refusal_direction_replication/gates.py` | Pure structured baseline/causal gates and compact evaluation payload |
| `projects/refusal_direction/src/refusal_direction_replication/pipeline.py` | Profile-controlled filtering/splits, unfiltered Qwen baseline, test finals, preflight, CUDA release |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot.py` | Resumable stage runner with skip/abort/fail and stable run ID |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot_validate.py` | Compact Qwen artifact validator |
| `projects/refusal_direction/scripts/qwen_run_pilot.py` | Remote-safe Colab entry: no argparse, no `__file__` |
| `projects/refusal_direction/scripts/validate_qwen_pilot.py` | Local validator CLI |
| `tools/colab/run_refusal.py` | Qwen-only `pilot` action; Gemma actions unchanged |
| `projects/refusal_direction/tests/test_qwen_pilot.py` | Focused TDD coverage for Task 3 |

## Self-review

- **TDD:** Focused tests were written and collected first; they failed on the
  missing `gates` module before production code existed.
- **Gemma defaults:** Missing `test_per_class` keeps four splits; missing
  `filter_by_behavior` still filters by first-token sign; selection and final
  conditions still use validation. Existing Gemma tests remain green.
- **Unfiltered Qwen rates:** `maybe_filter_examples` / `maybe_filter_scores`
  keep mixed-sign items when `filter_by_behavior` is false, so baseline rates
  are not 100%/0% tautologies from score filtering.
- **28 candidates:** Qwen still uses `position_offsets: [-1]` over 28 layers.
  Extraction uses the full sampled training sets.
- **Gates:** Decisions use generated refusal rates and KL, not the first-token
  proxy. Thresholds, observed values, and reasons are stored on every check.
- **Remote exec:** `qwen_run_pilot.py` hardcodes `/content/refusal_direction`,
  ignores Colab kernel argv, and has no argparse/`__file__` dependency.
- **Privacy:** Compact JSON stores IDs, hashes, and aggregate rates. Raw
  prompts/generations stay under ignored `artifacts/` and fail validation if
  leaked into results JSON.
- **No forbidden work:** No Colab start, no weight download, no Gemma result
  overwrite, no Task 4.

## Concerns

1. **Float boundaries:** `0.3 - 0.1` is `0.1999...` in IEEE floats, so a true
   `>= 0.20` addition-increase check fails at those literals. Tests use values
   clearly above/below the threshold; live rates will have the same comparison.
2. **Preflight probes** are the first two rows of the raw train JSON, not the
   seeded sampled train split, so probe hashes are not the dataset hashes used
   later. Texts are still unpublished.
3. **Next-stage `initialize()`** overwrites a completed `status.json` snapshot.
   Resume skip still works via `events.jsonl` `stage_completed` records.
4. **Gemma evaluation manifest** still uses `intervene`/`evaluate` names. Qwen
   pilot stages are `interventions`/`evaluation`. Compact Qwen files are
   namespaced, so this is naming drift only.
5. **No live GPU evidence yet.** Preflight/load/generation paths are mocked
   offline; Colab execution is intentionally deferred to post-review.

## Status

`DONE_WITH_CONCERNS`
