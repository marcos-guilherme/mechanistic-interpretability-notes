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

---

## Review fix round (task-3-review.md)

Addressed every Critical and Important finding. Train/validation/test routing
and Gemma defaults were left unchanged.

### RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py -q --tb=line
```

Result (failing tests added before production changes):

```text
ImportError: cannot import name 'normalize_prompt_text' from
'refusal_direction_replication.data'
ERROR tests/test_qwen_pilot.py
Interrupted: 1 error during collection
1 error in 5.45s
```

### GREEN evidence

Focused suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py -q
```

Result: `58 passed in 15.43s`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `142 passed in 15.15s` (previous 127 + 15 review-fix tests)

### Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/src/refusal_direction_replication/data.py` | Normalized prompt content hashes |
| `projects/refusal_direction/src/refusal_direction_replication/metrics.py` | `is_valid_generation` |
| `projects/refusal_direction/src/refusal_direction_replication/gates.py` | Rational `Fraction` comparisons; operational-error payload fields |
| `projects/refusal_direction/src/refusal_direction_replication/pipeline.py` | Valid-generation counting, content hashes, generated-token preflight scores, revision pin, run/commit provenance |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot.py` | Operational-error compact JSON before fail/re-raise; pass `run_id` into stages |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot_validate.py` | Strengthened pass, isolation, provenance, candidate-grid, gate-field, and leakage checks |
| `projects/refusal_direction/tests/test_qwen_pilot.py` | Focused rejection tests for each review finding |

### Review items addressed

| Finding | Fix |
|--------|-----|
| Critical: pass with failing gate | `scientific_pass_requires_both_gates` rejects `status=pass` unless both gates passed |
| Valid generations | Empty/whitespace/`U+FFFD` excluded from refusal denominator; `n_valid`/`n_invalid` and IDs persisted |
| Content-based isolation | Per-split content hashes required; overlap and missing IDs/hashes fail validation |
| Preflight generated scores + revision | `generated_token_scores` must be finite; resolved HF revision must equal the pin |
| Rational gate boundaries | `Fraction.limit_denominator(10000)`; exact `0.30 - 0.10 == 0.20` test |
| Operational error JSON | Compact `pilot_evaluation.json` with status, stage, category, sanitized message |
| Run ID + implementation commit | Added to Qwen compact provenance; mixed/missing commits fail |
| Stronger validator | Model ID, requested+resolved revision, unique 28 layer/offset grid, full gate fields, generic leak strings |
| Focused rejection tests | Added for each requirement above and the operational-error path |

### Self-review (review round)

- Gemma `summarize_generated_condition()` and stage defaults are unchanged; valid-generation counting is opt-in via `filter_by_behavior=false`.
- Gate math no longer uses raw binary-float subtraction for rate gaps.
- Preflight no longer sets `generation_finite` from prompt-forward logits.
- Compact results still omit prompt/generation text.

### Concerns (review round)

1. **Remote commit stamp:** Colab archives exclude `.git`, so `implementation_commit` falls back to `unknown` unless `REFUSAL_IMPLEMENTATION_COMMIT` is set before the remote pilot.
2. **Generic leak heuristic** flags 8+ word strings and long non-hex values; gate `reason`/`message` fields are allowlisted.
3. **No GPU execution** in this round; scientific gates are still unexercised on Qwen weights.

### Status (review round)

`DONE_WITH_CONCERNS`

---

## Second review round (task-3-review.md)

Addressed every item in the second-review section. Gemma `build_archive()`
defaults remain unstamped; only `--profile qwen25-coder` sync writes
`IMPLEMENTATION_COMMIT` into the Colab archive.

### RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py tests/test_colab_orchestrator.py -q --tb=line
```

Result (failing tests added before production changes):

```text
FAILED tests/test_qwen_pilot.py::test_validator_rejects_unknown_or_empty_implementation_commit
FAILED tests/test_qwen_pilot.py::test_validator_rejects_missing_run_id_on_any_artifact
FAILED tests/test_qwen_pilot.py::test_status_invariants_reject_mismatched_gate_outcomes
FAILED tests/test_qwen_pilot.py::test_validator_rejects_generation_bookkeeping_mismatch
FAILED tests/test_qwen_pilot.py::test_validator_rejects_flags_that_do_not_match_valid_ids
FAILED tests/test_qwen_pilot.py::test_preflight_checks_variable_length_generated_scores_independently
FAILED tests/test_qwen_pilot.py::test_operational_error_uses_fixed_catalog_message
FAILED tests/test_qwen_pilot.py::test_validator_inspects_message_fields_instead_of_allowlisting
FAILED tests/test_qwen_pilot.py::test_causal_gate_compares_kl_directly_without_fraction_rounding
FAILED tests/test_qwen_pilot.py::test_implementation_commit_hash_rejects_unknown
FAILED tests/test_qwen_pilot.py::test_implementation_commit_hash_reads_stamped_file
FAILED tests/test_colab_orchestrator.py::test_gemma_archive_does_not_stamp_implementation_commit
FAILED tests/test_colab_orchestrator.py::test_qwen_archive_stamps_exact_implementation_commit
13 failed, 62 passed in 16.96s
```

### GREEN evidence

Focused suites:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py tests/test_colab_orchestrator.py tests/test_qwen_observability.py -q
```

Result: `101 passed in 15.17s`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `155 passed in 14.52s` (previous 142 + 13 second-review tests)

### Files changed

| File | Change |
|------|--------|
| `tools/colab/run_refusal.py` | Qwen-only archive stamp of exact `HEAD`; Gemma `build_archive()` default unchanged |
| `projects/refusal_direction/src/refusal_direction_replication/pipeline.py` | Read stamp/env/git; reject `unknown`; check generated scores per tensor |
| `projects/refusal_direction/src/refusal_direction_replication/gates.py` | Direct float KL compare; catalogued operational error codes/messages |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot.py` | Store bounded `error_code` and fixed message; never persist exception text |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot_validate.py` | Reject unknown/missing provenance; status/generation/message invariants |
| `projects/refusal_direction/tests/test_qwen_pilot.py` | Focused rejection tests for each second-review item |
| `projects/refusal_direction/tests/test_colab_orchestrator.py` | Gemma unstamped vs Qwen stamped archive members |

### Review items addressed

| Finding | Fix |
|--------|-----|
| Stamp exact commit in Colab archive | Qwen sync adds `refusal_direction/IMPLEMENTATION_COMMIT`; pipeline reads stamp/env before git |
| Reject unknown/empty commit and missing run IDs | Validator `code_provenance` / `single_run_id`; `implementation_commit_hash()` raises |
| Status invariants | `baseline_gate_failed`/`causal_gate_failed`/`pass`/`operational_error` checked against gate outcomes and `operational_success` |
| Generated-condition bookkeeping | `n_valid + n_invalid`, disjoint complete IDs, flags aligned, rate from valid refusals |
| Variable-length generated scores | Preflight checks each score tensor; no `torch.cat` |
| Operational-error exception text | Catalog `error_code` + fixed message; validator inspects `message` instead of allowlisting |
| KL vs rate comparison | Rate diffs stay rational; KL compared as `float <= 0.10` |

### Self-review (second review round)

- Gemma `build_archive()` still defaults to `stamp_commit=False`; only Qwen sync stamps.
- Compact operational-error JSON no longer contains exception strings that could include prompts.
- KL of `0.10` still passes; `0.100001` now fails instead of rounding to `1/10`.

### Concerns (second review round)

1. **Remote still needs the stamp.** If a Qwen archive is built without `stamp_commit=True` and git is absent, `implementation_commit_hash()` now raises (operational error) instead of writing `unknown`.
2. **No GPU execution** in this round; scientific gates remain unexercised on Qwen weights.

### Status (second review round)

`DONE_WITH_CONCERNS`

---

## Third review round (task-3-review.md)

Addressed every item in the third-review section. Config and provenance are
checked independently, gate comparators/reasons are canonical, commits must be
40 lowercase hex characters, and missing stamps fail inside the pilot error
boundary.

### RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py::test_validator_rejects_wrong_provenance_when_config_is_correct tests/test_qwen_pilot.py::test_validator_rejects_missing_provenance_revision_when_config_is_correct tests/test_qwen_pilot.py::test_validator_rejects_missing_or_wrong_upstream_commit tests/test_qwen_pilot.py::test_validator_rejects_wrong_gate_comparator tests/test_qwen_pilot.py::test_validator_rejects_noncanonical_gate_reason tests/test_qwen_pilot.py::test_validator_rejects_non_lowercase_sha_implementation_commit tests/test_qwen_pilot.py::test_missing_implementation_commit_produces_operational_error -q --tb=line
```

Result (failing tests added before production changes):

```text
FAILED tests/test_qwen_pilot.py::test_validator_rejects_wrong_provenance_when_config_is_correct
FAILED tests/test_qwen_pilot.py::test_validator_rejects_missing_provenance_revision_when_config_is_correct
FAILED tests/test_qwen_pilot.py::test_validator_rejects_missing_or_wrong_upstream_commit
FAILED tests/test_qwen_pilot.py::test_validator_rejects_wrong_gate_comparator
FAILED tests/test_qwen_pilot.py::test_validator_rejects_noncanonical_gate_reason
FAILED tests/test_qwen_pilot.py::test_validator_rejects_non_lowercase_sha_implementation_commit
FAILED tests/test_qwen_pilot.py::test_missing_implementation_commit_produces_operational_error
7 failed in 8.85s
```

The missing-stamp test failed with `FileNotFoundError` for `pilot_evaluation.json`
because commit resolution ran outside the orchestration `try`.

### GREEN evidence

Focused suites:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py tests/test_colab_orchestrator.py tests/test_qwen_observability.py -q
```

Result: `108 passed in 17.70s`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `162 passed in 17.22s` (previous 155 + 7 third-review tests)

### Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot_validate.py` | Independent config/provenance/upstream checks; exact comparators and canonical reasons; 40 lowercase hex commits |
| `projects/refusal_direction/src/refusal_direction_replication/pipeline.py` | Stamp/env/git commits accepted only if they are 40 lowercase hex |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_pilot.py` | Resolve commit after stage initialize, inside the operational-error boundary |
| `projects/refusal_direction/tests/test_qwen_pilot.py` | Focused rejection tests for each third-review item |

### Review items addressed

| Finding | Fix |
|--------|-----|
| Independent model/revision + upstream | Config and provenance each must match the pinned Qwen model, revision, and Arditi commit |
| Exact comparator and canonical reason | Each named check has one required comparator; stored reason must match recomputed gate text |
| 40 lowercase hex commits | Validator and `implementation_commit_hash()` reject uppercase, empty, `unknown`, and wrong length |
| Commit resolution error boundary | Missing/corrupt stamp writes `operational_error` and failed status before re-raising |

### Self-review (third review round)

- A correct config can no longer hide a wrong or missing provenance model, requested revision, resolved revision, or upstream commit.
- `min_*` checks must be `gte` and `max_*` checks must be `lte`; swapping them fails even if `passed` is still true.
- Missing stamps now fail closed after `initialize()`, so `status.json` can be marked failed.

### Concerns (third review round)

1. **No GPU execution** in this round; scientific gates remain unexercised on Qwen weights.
2. **Git SHAs that are not 40 lowercase hex** (for example uppercase) are treated as corrupt and become operational errors.

### Status (third review round)

`DONE_WITH_CONCERNS`

---

## Live Colab import failure (task-3-live-bug.md)

The first live `pilot` exec failed with
`ModuleNotFoundError: No module named 'refusal_direction_replication'`
because `colab_bootstrap.py` installs the package in a subprocess while
`colab exec` reuses the already-running kernel `sys.path`.

### RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py::test_qwen_remote_scripts_prepare_src_before_package_import tests/test_qwen_observability.py::test_qwen_remote_scripts_prepare_src_before_package_import -q --tb=short
```

Result (failing tests added before production changes):

```text
FAILED tests/test_qwen_pilot.py::test_qwen_remote_scripts_prepare_src_before_package_import
FAILED tests/test_qwen_observability.py::test_qwen_remote_scripts_prepare_src_before_package_import
2 failed in 6.10s
```

### GREEN evidence

Focused suites:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_pilot.py tests/test_colab_orchestrator.py tests/test_qwen_observability.py -q
```

Result: `110 passed in 21.16s`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `164 passed in 19.72s` (previous 162 + 2 import-bootstrap tests)

### Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/scripts/qwen_run_pilot.py` | Insert `/content/refusal_direction/src` on `sys.path` before package import |
| `projects/refusal_direction/scripts/qwen_run_status.py` | Same remote import bootstrap |
| `projects/refusal_direction/scripts/colab_pack_qwen_results.py` | Same remote import bootstrap |
| `projects/refusal_direction/tests/test_qwen_pilot.py` | Source-order test for the three Qwen remote scripts |
| `projects/refusal_direction/tests/test_qwen_observability.py` | Same source-order test; Gemma pack script left unchanged |

### Self-review (live import fix)

- Package modules still import `refusal_direction_replication` normally; only Colab-exec entrypoints bootstrap `sys.path`.
- Gemma `colab_pack_results.py` does not import the package and was not modified.
- Remote scripts still avoid argparse and `__file__`.

### Concerns (live import fix)

1. The live session still needs a resync/re-exec of these scripts; this commit does not start or stop Colab.
2. Scientific GPU gates remain unexercised.

### Status (live import fix)

`DONE_WITH_CONCERNS`
