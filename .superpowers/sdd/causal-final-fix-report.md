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

## Whole-implementation rereview fix wave

### Status

DONE

Commit: the separate commit containing this appended section and the rereview
fixes.

### RED evidence

#### Independent profile pin and raw pilot gate

Command:

```text
uv run pytest tests/test_qwen_causal_jlens_runtime.py -k "semantically_unchanged_profile or raw_pilot_gate or panel_requires" -q
```

Output:

```text
FF                                                                       [100%]
2 failed, 32 deselected
```

The failures showed that `_require_config_contract` had no independently pinned
profile-byte argument and that panel execution reached model loading when the
compact pilot summary existed but raw pilot rows were absent.

#### Resume, finalize reconciliation, and validator profile pin

Command:

```text
uv run pytest tests/test_qwen_causal_jlens_validate.py -k "caught_failure_terminal or abrupt_interruption or passing_terminal_reconciles or semantically_unchanged_profile" -q
```

Output:

```text
FFFF                                                                     [100%]
4 failed, 44 deselected
```

The caught-failure case was rejected as permanently non-reusable, the passing
terminal was strictly validated before status/event reconciliation, and a
semantically unchanged profile-byte mutation reached source validation instead
of failing the independent digest. The first abrupt-interruption run also
exposed a missing test import; that fixture error was corrected before the
production implementation, and the completed GREEN test exercises the repeated
initialized attempt through final offline validation.

### Implemented rereview fixes

- Added immutable `FROZEN_CAUSAL_PROFILE_SHA256` for the exact committed
  `qwen25_coder_7b_causal_jlens.json` bytes. Runtime and validator verify the
  packaged canonical file and the deployed production profile independently;
  fixture relaxation remains explicit.
- Same-identity `operational_error` and `aborted` terminals are identity-checked
  and atomically retired with their manifest into non-transported artifacts,
  after which stage artifacts/checkpoint state are resumed.
- A passing terminal now reconciles finalize status/events before strict full
  offline validation, closing the terminal-write/stage-completion crash window.
- Event validation models repeated attempts within one stage. Pending, running,
  failed, aborted, and hard-interrupted attempts may restart the same stage;
  completed counters remain monotonic across attempts, and only a completed
  attempt may advance the frozen stage order.
- Panel now validates the raw pilot matrix before panel reconciliation or model
  work: exact 16 response keys, unique complete observations, strict terminal
  semantics, target-token alignment, same-identity checkpoint completion, and
  exact compact summary counts are all required. Missing or inconsistent pilot
  rows block instead of being regenerated.
- Added end-to-end fixture tests for caught failure, interruption after
  initialization, and passing-terminal reconciliation. Every case ends with a
  full validator pass and unique raw response/observation keys.

### GREEN evidence

Focused event-attempt command:

```text
uv run pytest tests/test_qwen_causal_jlens_validate.py -k "caught_failure_terminal or abrupt_interruption or passing_terminal_reconciles or event_sequence" -q
```

Output:

```text
.......                                                                  [100%]
7 passed, 41 deselected
```

Final causal/shared suite:

```text
uv run pytest tests/test_qwen_causal_jlens.py tests/test_qwen_causal_jlens_runtime.py tests/test_qwen_causal_jlens_validate.py tests/test_qwen_causal_jlens_transport.py tests/test_colab_orchestrator.py tests/test_qwen_code_eval_transport.py -q
```

Output:

```text
........................................................................ [ 45%]
........................................................................ [ 90%]
...............                                                          [100%]
159 passed in 99.41s (0:01:39)
```

### Rereview files changed

- `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens.py`
- `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_runtime.py`
- `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_validate.py`
- `projects/refusal_direction/tests/test_qwen_causal_jlens_runtime.py`
- `projects/refusal_direction/tests/test_qwen_causal_jlens_validate.py`
- `.superpowers/sdd/causal-final-fix-report.md`

### Rereview self-review and concerns

Rechecked terminal identity before retirement, finalize reconciliation ordering,
attempt transitions/counter monotonicity, raw-pilot validation ordering, and the
independence of the profile digest. Existing stop, Jacobian cache, observation
schema, clean-run event, validator, transport, and shared orchestrator tests
remain green.

Concerns: none. No model download, Colab start, or local GPU execution occurred.
