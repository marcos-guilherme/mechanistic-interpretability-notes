# Task 2 implementation report

## Summary

Added observable/resumable Qwen progress storage, a Qwen status CLI, profile-aware
Colab orchestration, allowlisted Qwen packing, and validated all-or-nothing local
fetch. Default Gemma Colab behavior is unchanged without `--profile`.

## RED evidence

Command:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_observability.py -q
```

Result (before implementation):

```text
ModuleNotFoundError: No module named 'refusal_direction_replication.run_progress'
ERROR tests/test_qwen_observability.py
```

## GREEN evidence

Focused new tests:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_observability.py -q
```

Result: `19 passed`

Full refusal project suite:

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `73 passed in 7.30s` (baseline 58 + 15 new)

## Files changed

| File | Change |
|------|--------|
| `projects/refusal_direction/src/refusal_direction_replication/run_progress.py` | Atomic status/events contract, transition and progress validation, resume skip |
| `projects/refusal_direction/src/refusal_direction_replication/qwen_colab_transport.py` | Qwen pack allowlist, archive validation, safe extract |
| `projects/refusal_direction/scripts/qwen_run_status.py` | Human-readable Qwen status CLI |
| `projects/refusal_direction/scripts/colab_pack_qwen_results.py` | Remote Qwen-only pack script |
| `tools/colab/run_refusal.py` | `--profile`, default sessions, `status` action, Qwen fetch path |
| `projects/refusal_direction/tests/test_qwen_observability.py` | Focused TDD tests |

## Self-review

- **Gemma defaults:** `fetch` without `--profile` still uses `colab_pack_results.py` and `/content/refusal_results.tar.gz` with the prior extract behavior.
- **Qwen isolation:** Pack/fetch allowlists only namespaced `results/qwen25_coder_7b` and `artifacts/qwen25_coder_7b` paths.
- **Atomic status:** `status.json` uses tempfile + `os.replace`; partial JSON is not published.
- **Resume:** `should_skip_stage` reads completed state and `stage_completed` events.
- **No forbidden work:** No model loads, dataset downloads, GPU runs, or canonical Gemma result edits.

## Concerns

1. **`events.jsonl` append** is not crash-atomic across OS boundaries; only `status.json` is replace-atomic as specified.
2. **Traversal test** uses absolute member paths because GNU tar normalizes `..` when archives are written via the stdlib.
3. **`run_refusal.py`** prepends the project `src` path for local Qwen fetch helpers; remote Colab exec still depends on the synced source tree.

## Status

`DONE_WITH_CONCERNS`

---

## Review fix round (task-2-review.md)

### RED evidence

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_observability.py -q
```

```text
FAILED test_events_append_json_lines_with_run_id - KeyError: 'run_id'
FAILED test_resume_does_not_inherit_other_run_id - assert not True
FAILED test_terminal_state_rejects_progress_mutation - ...
FAILED test_normalize_rejects_raw_dotdot_without_lstrip_sanitization - ...
FAILED test_qwen_fetch_rejects_directory_member - ...
FAILED test_qwen_fetch_rejects_duplicate_members - ...
FAILED test_qwen_fetch_rollback_after_install_failure - ...
FAILED test_qwen_status_cli_prints_human_readable_success - ...
10 failed, 16 passed
```

### GREEN evidence

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests/test_qwen_observability.py -q
```

Result: `26 passed`

```text
cd projects/refusal_direction
.venv\Scripts\python -m pytest tests -q
```

Result: `84 passed in 7.38s` (baseline 58 + 26 observability tests)

### Fix commit

`19a51beb4c05d96ba30f321bc846359155d0ae56`

### Review items addressed

| Finding | Fix |
|--------|-----|
| Remote `status` Colab exec | `remote_main()` uses `/content/refusal_direction`; no `__file__` or argparse/ipykernel parsing |
| Import-light status CLI | Reads config JSON directly; imports only `run_progress` constants |
| Tar member validation | Validates every member; rejects links, directories, and non-regular files |
| Raw `..` paths | Rejects `..` in raw path components; removed `lstrip` traversal sanitization |
| Fetch rollback | Pre-records backups; restores all originals on install failure |
| Run-scoped resume | Events carry `run_id`; `should_skip_stage` matches `(run_id, stage)` |
| Terminal mutations | Completed/aborted/failed reject progress/message unless resuming via state change |
| Lazy Qwen transport import | `run_refusal` loads Qwen extract only on Qwen `fetch` |
| Atomic status durability | Temp status file flushed and `fsync`d before `os.replace` |
| Test quality | Added behavioral tests for links, dirs, dupes, size caps, rollback, status output, lazy import |

### Self-review (review round)

- Remote status entrypoint ignores noisy `sys.argv` and always resolves the Qwen artifact namespace from `REMOTE_ROOT`.
- Fetch validation fails closed before publishing partial installs.
- Focused observability suite now has 26 behavioral tests (not 19).

### Status (review round)

`DONE`