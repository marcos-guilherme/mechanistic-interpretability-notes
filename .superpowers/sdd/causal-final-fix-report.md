# Causal J-lens final fix wave

## Status

DONE

One integrated pre-A100 fix wave. No model download, Colab start, or local GPU
execution was performed.

## RED evidence

All production changes were preceded by focused failing tests.

### Guaranteed causal stop

Command:

```text
uv run pytest tests/test_colab_orchestrator.py -k causal_stop -q
```

Output:

```text
FF
2 failed, 26 deselected
```

Both tests failed because causal stop called `run(stop_command(...))` once,
propagating transient and persistent errors instead of retrying and reporting an
uncertain A100 stop.

### Runtime, hook, pilot, and lens-cache gates

Command:

```text
uv run pytest tests/test_qwen_causal_jlens_runtime.py -k "dual_readout_prepares or real_hook_registry or runtime_production_config or panel_requires" -q
```

Output:

```text
FFFF
4 failed, 29 deselected
```

Failures showed that `DualLensReadout` had no preparation API, hook-registry
drift was not detected, the production config contract was partial, and panel
generation loaded the model without independently checking `01_pilot.json`.

### Validator and terminal-reuse gates

Command:

```text
uv run pytest tests/test_qwen_causal_jlens_validate.py -k "strict_observation_schema or event_sequence or runtime_reuses_terminal" -q
```

Output:

```text
FF..F.FFFFFF.
9 failed, 4 passed, 31 deselected
```

Failures covered stale terminal reuse, unknown observation fields, invalid
rank/layer values, and unvalidated event order/transitions/counter monotonicity.

### Transport observation gate

Command:

```text
uv run pytest tests/test_qwen_causal_jlens_transport.py -k invalid_observation_before -q
```

Output:

```text
FFFF
4 failed, 26 deselected
```

Archive creation accepted raw/unknown observation fields, nonpositive or
fractional ranks, and Boolean steps.

## Implemented fixes

- Causal stop now uses the same retrying `try_stop_session()` path as J-lens v2.
  Persistent uncertainty prints `A100 MAY STILL BE RUNNING`, the exact retry
  command, and returns nonzero.
- Runtime production configuration is compared exactly with the packaged
  frozen profile before any model load. Fixture differences require the
  explicit `allow_fixture_contract=True` path and remain allowlisted.
- A pre-existing passing terminal result is returned only after the full
  offline validator accepts current config hash, model/EOS identity, run
  identity, source hashes, events, checkpoint, and outputs. Invalid or
  non-passing terminal files abort.
- Every generation snapshots real `_forward_pre_hooks` and `_forward_hooks`
  keys on all involved blocks, attention modules, and MLPs, then rejects any
  post-generation difference.
- Panel generation independently loads same-identity `01_pilot.json` and
  requires status `pass`, exact pilot IDs/conditions, and the exact 16-pair
  matrix before model loading or row reuse.
- Both lenses are pruned to layers 19, 22, 25, and 26 and their eight float32
  Jacobians are moved to the model device once. `rank_target` uses the prepared
  cache directly and rejects residual/Jacobian device mismatch, eliminating
  per-readout CPU-to-CUDA transfer. Production storage is 411,041,792 bytes,
  approximately 411 MB decimal.
- Observation validation now requires the exact eight-key schema, configured
  datasets/conditions/layers, exact lens enums, strict nonnegative integer
  step/layer/target IDs, and strict positive integer ranks. Runtime, offline
  validation, and transport packing all enforce it.
- Event validation now requires exact
  prepare→pilot→panel→analysis→finalize order, pending→running→completed
  transitions, one run identity, fixed totals, and monotonic completed counters
  within every stage.

## Files changed

- `tools/colab/run_refusal.py`
- `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_runtime.py`
- `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_validate.py`
- `projects/refusal_direction/src/refusal_direction_replication/qwen_colab_transport.py`
- `projects/refusal_direction/tests/test_colab_orchestrator.py`
- `projects/refusal_direction/tests/test_qwen_causal_jlens_runtime.py`
- `projects/refusal_direction/tests/test_qwen_causal_jlens_validate.py`
- `projects/refusal_direction/tests/test_qwen_causal_jlens_transport.py`
- `.superpowers/sdd/causal-final-fix-report.md`

## GREEN evidence

Command:

```text
uv run pytest tests/test_qwen_causal_jlens.py tests/test_qwen_causal_jlens_runtime.py tests/test_qwen_causal_jlens_validate.py tests/test_qwen_causal_jlens_transport.py tests/test_colab_orchestrator.py tests/test_qwen_code_eval_transport.py -q
```

Output:

```text
........................................................................ [ 46%]
........................................................................ [ 93%]
..........                                                               [100%]
154 passed in 80.64s (0:01:20)
```

`git diff --check` also completed successfully.

## Self-review

- Rechecked each of the six review requirements against implementation and
  focused tests.
- Confirmed the causal stop change leaves non-causal profiles unchanged.
- Confirmed production strictness is the default and fixture relaxation is
  explicit and field-limited.
- Confirmed pilot and terminal gates execute before model loading.
- Confirmed cache preparation preserves canonical parity tests and the readout
  path no longer calls `JacobianLens.transport`.
- Confirmed archive creation fails before writing an archive when observation
  rows are malformed.
- Confirmed only focused causal and nearby shared-file regressions were run.
- Preserved unrelated pre-existing worktree changes and excluded them from this
  commit.

## Concerns

None. The actual A100/model path was intentionally not exercised in this fix
wave; the CPU fixtures cover preparation, pruning, device agreement, parity,
and transfer-path behavior.
