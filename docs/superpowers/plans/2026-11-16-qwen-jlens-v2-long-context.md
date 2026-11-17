# Qwen J-lens v2 Long-Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajustar e avaliar uma J-lens v2 isolada para o Qwen2.5-Coder-7B-Instruct usando contextos de 512 tokens, 200 funções de holdout e uma quantidade de ajuste escolhida por probe sob o teto de 12 horas.

**Architecture:** O perfil v2 reutiliza o cálculo Jacobiano e os contratos da v1, mas mantém configuração, sessão Colab, staging, checkpoints, artefatos e resultados próprios. O probe escolhe a maior quantidade segura em uma escada fixa; o runtime avalia v1, v2 e logit lens no mesmo holdout da v2. Checkpoints atômicos ficam na VM e cópias validadas periódicas são baixadas para sobreviver à perda da sessão.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, `anthropics/jacobian-lens` no commit fixado, Hugging Face Datasets, pytest, uv, Colab CLI e A100.

## Global Constraints

- Não alterar `configs/qwen25_coder_7b_jlens.json` nem os resultados e artefatos congelados da v1.
- Não alterar a lente usada pelo laboratório ou pela avaliação de código.
- Fixar `max_seq_len=512`, `min_tokens_before_truncation=512` e `holdout_count=200`.
- Escolher `fit_count` somente entre `100, 150, 200, 250, 300, 400, 600, 800`.
- Recusar o ajuste se nem 100 funções couberem na projeção de 12 horas.
- Manter os textos de código e prompts fora dos JSONs versionáveis e dos arquivos de log.
- Gravar checkpoint atômico após cada função e validar identidade completa antes de retomar.
- Preservar a cópia local anterior quando uma nova cópia de checkpoint falhar na validação.
- Executar cada mudança de produção somente depois de observar o teste correspondente falhar.

---

### Task 1: Perfil v2 isolado

**Files:**
- Create: `projects/refusal_direction/configs/qwen25_coder_7b_jlens_v2.json`
- Create: `projects/refusal_direction/tests/test_qwen_jlens_v2.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/pipeline.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py`

**Interfaces:**
- Produces: `JLENS_V2_PROFILE`, `V2_MAX_SEQ_LEN`, `V2_HOLDOUT_COUNT`, `V2_FIT_COUNT_CANDIDATES`, `V2_MAX_FIT_COUNT` e `V2_FINAL_LENS_FILENAME`.
- Preserves: `JLENS_PROFILE`, `FIT_COUNT`, `HOLDOUT_COUNT` e todos os caminhos da v1.

- [ ] **Step 1: Write the failing isolation test**

```python
def test_v2_profile_is_isolated_from_v1_lab_and_code_eval() -> None:
    v1 = load_config(ROOT, profile="qwen25_coder_7b_jlens.json")
    v2 = load_config(ROOT, profile="qwen25_coder_7b_jlens_v2.json")
    lab = load_config(ROOT, profile="qwen25_coder_7b_lab.json")

    assert v2["max_seq_len"] == 512
    assert v2["min_tokens_before_truncation"] == 512
    assert v2["holdout_count"] == 200
    assert v2["fit_count_candidates"] == [100, 150, 200, 250, 300, 400, 600, 800]
    assert v2["results_dir"] == "results/qwen25_coder_7b_jlens_v2"
    assert v2["artifacts_dir"] == "artifacts/qwen25_coder_7b_jlens_v2"
    assert v2["source_v1_lens"].endswith(
        "qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert v1["artifacts_dir"] != v2["artifacts_dir"]
    assert lab["source_lens"] == v2["source_v1_lens"]
```

- [ ] **Step 2: Verify the test fails because the profile is absent**

Run:

```powershell
uv run pytest tests/test_qwen_jlens_v2.py::test_v2_profile_is_isolated_from_v1_lab_and_code_eval -v
```

Expected: FAIL while loading `qwen25_coder_7b_jlens_v2.json`.

- [ ] **Step 3: Add the profile and constants**

The new JSON copies the model, dataset, revision, layer and upstream pins from
the v1 profile and sets:

```json
{
  "max_seq_len": 512,
  "min_tokens_before_truncation": 512,
  "holdout_count": 200,
  "fit_count_candidates": [100, 150, 200, 250, 300, 400, 600, 800],
  "max_fit_count": 800,
  "results_dir": "results/qwen25_coder_7b_jlens_v2",
  "artifacts_dir": "artifacts/qwen25_coder_7b_jlens_v2",
  "exclusion_hashes_path": "artifacts/qwen25_coder_7b_jlens_v2/exclusion_hashes.txt",
  "source_v1_lens": "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
}
```

Add the v2 profile to the J-lens exception in `validate_profile_config` without
weakening validation for other profiles.

- [ ] **Step 4: Run v2 and v1 profile tests**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py::test_v2_profile_is_isolated_from_v1_lab_and_code_eval tests/test_qwen_jlens.py::test_jlens_profile_is_isolated_from_pilot_and_dose -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/configs/qwen25_coder_7b_jlens_v2.json projects/refusal_direction/tests/test_qwen_jlens_v2.py projects/refusal_direction/src/refusal_direction_replication/pipeline.py projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py
git commit -m "feat(jlens): add isolated long-context profile"
```

### Task 2: Seleção do tamanho pelo probe e corpus fixo

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py`
- Modify: `projects/refusal_direction/tests/test_qwen_jlens_v2.py`

**Interfaces:**
- Produces: `select_fit_count(probe_seconds, candidates, limit_seconds) -> DurationProjection`.
- Produces: `slice_v2_corpus(corpus, fit_count, max_fit_count) -> CorpusSplit`.
- Guarantees: o holdout corresponde sempre às mesmas 200 funções, qualquer que seja o `fit_count` escolhido.

- [ ] **Step 1: Write failing selection and corpus tests**

```python
def test_select_fit_count_picks_largest_safe_candidate() -> None:
    selected = select_fit_count(
        probe_seconds=80.0,
        candidates=[100, 150, 200, 250, 300, 400, 600, 800],
        limit_seconds=43_200,
    )
    assert selected.n_prompts == 400
    assert selected.projected_seconds == 32_000.0


def test_select_fit_count_rejects_when_minimum_exceeds_limit() -> None:
    with pytest.raises(ValueError, match="minimum candidate"):
        select_fit_count(
            probe_seconds=500.0,
            candidates=[100, 150],
            limit_seconds=43_200,
        )


def test_slice_v2_corpus_keeps_holdout_fixed() -> None:
    pool = _corpus(fit=800, holdout=200)
    small = slice_v2_corpus(pool, fit_count=100, max_fit_count=800)
    large = slice_v2_corpus(pool, fit_count=600, max_fit_count=800)
    assert small.fit == pool.fit[:100]
    assert large.fit == pool.fit[:600]
    assert small.holdout == large.holdout == pool.holdout
    assert set(item.content_hash for item in large.fit).isdisjoint(
        item.content_hash for item in large.holdout
    )
```

- [ ] **Step 2: Verify all three tests fail for missing functions**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py -k "select_fit_count or slice_v2_corpus" -v
```

Expected: FAIL with import errors for the two new functions.

- [ ] **Step 3: Implement deterministic selection and slicing**

```python
def select_fit_count(
    *,
    probe_seconds: float,
    candidates: Sequence[int],
    limit_seconds: int = MAX_PROJECTED_SECONDS,
) -> DurationProjection:
    ordered = sorted({int(value) for value in candidates})
    if not ordered or ordered[0] <= 0:
        raise ValueError("fit count candidates must be positive")
    safe = [
        project_fit_duration(
            probe_seconds=probe_seconds,
            n_prompts=value,
            limit_seconds=limit_seconds,
        )
        for value in ordered
        if probe_seconds * value <= limit_seconds
    ]
    if not safe:
        raise ValueError("minimum candidate exceeds projected runtime limit")
    return safe[-1]
```

Build the 1,000-item pool with `fit_count=800`, `holdout_count=200` and
`min_tokens=512`. Slicing must never move holdout items into fit.

- [ ] **Step 4: Run corpus and v1 regression tests**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py tests/test_qwen_jlens.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py projects/refusal_direction/tests/test_qwen_jlens_v2.py
git commit -m "feat(jlens): select safe long-context fit size"
```

### Task 3: Checkpoint parametrizado e retomada segura

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py`
- Modify: `projects/refusal_direction/tests/test_qwen_jlens_v2.py`

**Interfaces:**
- Changes: `write_fit_checkpoint(..., fit_count: int, **identity)`.
- Changes: `validate_checkpoint_structure(payload, fit_count)` must use the selected count supplied by the caller.
- Preserves: callers v1 may omit `fit_count` and retain the default 100.

- [ ] **Step 1: Write failing tests for counts above 100 and mismatched identity**

```python
def test_v2_checkpoint_accepts_progress_above_v1_limit(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    write_fit_checkpoint(
        path,
        jacobian_sum={0: torch.eye(4)},
        n_done=150,
        next_idx=150,
        dim_batch=8,
        fit_count=200,
        **_identity(),
    )
    payload = load_fit_checkpoint(path)
    validate_checkpoint_structure(payload, fit_count=200)
    assert payload["fit_count"] == 200


def test_v2_checkpoint_rejects_progress_past_selected_fit(tmp_path: Path) -> None:
    payload = _checkpoint_payload(n_done=201, next_idx=201, fit_count=200)
    with pytest.raises(ValueError, match="outside 0..200"):
        validate_checkpoint_structure(payload, fit_count=200)
```

- [ ] **Step 2: Verify the tests fail on the hardcoded v1 limit**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py -k "checkpoint" -v
```

Expected: FAIL because the existing writer or validator assumes 100.

- [ ] **Step 3: Persist and validate selected fit count**

Store `fit_count` in the checkpoint payload. Keep `config_hash`,
`corpus_hash`, `model_revision`, source layers, target layer, skip count and
`run_id` in the identity check. Reject a payload whose stored `fit_count`
differs from the selected count.

- [ ] **Step 4: Run v2 and v1 checkpoint tests**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py -k "checkpoint" -v
uv run pytest tests/test_qwen_jlens.py -k "checkpoint" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py projects/refusal_direction/tests/test_qwen_jlens_v2.py
git commit -m "feat(jlens): parameterize resumable checkpoints"
```

### Task 4: Runtime probe, ajuste e progresso dinâmico

**Files:**
- Create: `projects/refusal_direction/tests/test_qwen_jlens_v2_runtime.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_runtime.py`

**Interfaces:**
- Consumes: `select_fit_count`, `slice_v2_corpus` and the v2 profile.
- Produces: selected `n_fit` in probe, fit, evaluation, status and event artifacts.
- Preserves: v1 still runs with 100 fit and 20 holdout.

- [ ] **Step 1: Write a tiny CPU runtime test**

```python
def test_runtime_selects_fit_count_and_checkpoints_each_item(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    result = run_qwen_jlens(
        tmp_path,
        profile=config.name,
        load_rows=lambda _: _rows(6),
        load_model=_tiny_model,
        wrap_model=_tiny_wrapper,
        jacobian_for_prompt_fn=_constant_jacobian,
        probe_clock=_clock_for_seconds(1.0),
    )
    assert result["status"] == "pass"
    assert result["n_fit"] == 4
    checkpoint = load_fit_checkpoint(
        tmp_path / "artifacts/v2/jlens_fit_checkpoint.pt"
    )
    assert checkpoint["n_done"] == checkpoint["fit_count"] == 4
```

Add separate tests for abort when the minimum candidate exceeds the cap and
resume from `n_done=2`.

- [ ] **Step 2: Verify runtime tests fail**

```powershell
uv run pytest tests/test_qwen_jlens_v2_runtime.py -v
```

Expected: FAIL because runtime progress and evaluation still use module
constants 100 and 20.

- [ ] **Step 3: Refactor runtime totals to corpus lengths**

After the v2 probe, slice the fit pool and derive:

```python
fit_total = len(corpus.fit)
holdout_total = len(corpus.holdout)
```

Use these values in stage initialization, progress updates, events,
checkpoint validation and `build_jlens_evaluation`. Write the selected count
to `01_jlens_probe.json` before fitting. Preserve the v1 branch exactly.

- [ ] **Step 4: Run v2 runtime and v1 runtime regressions**

```powershell
uv run pytest tests/test_qwen_jlens_v2_runtime.py tests/test_qwen_jlens_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/tests/test_qwen_jlens_v2_runtime.py projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_runtime.py
git commit -m "feat(jlens): run probe-selected v2 fits"
```

### Task 5: Comparação v1, v2 e logit lens no mesmo holdout

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_runtime.py`
- Modify: `projects/refusal_direction/tests/test_qwen_jlens_v2.py`
- Modify: `projects/refusal_direction/tests/test_qwen_jlens_v2_runtime.py`

**Interfaces:**
- Produces: `summarize_holdout_subset_stability(records, source_layers, n_parts=2)`.
- Produces: `04_jlens_compare.json` with methods `jlens_v2`, `jlens_v1` and `logit_lens`.
- Requires: all three methods use identical holdout ids, token positions and target token ids.

- [ ] **Step 1: Write failing shared-holdout tests**

```python
def test_dual_holdout_comparison_requires_shared_observations() -> None:
    comparison = summarize_dual_holdout_metrics(
        v2_records=_records("v2", ids=["a", "b"]),
        v1_records=_records("v1", ids=["a", "b"]),
        source_layers=[0, 1],
    )
    assert set(comparison["methods"]) == {"jlens_v2", "jlens_v1", "logit_lens"}
    assert comparison["shared_holdout_ids"] == ["a", "b"]
    assert len(comparison["stability"]["halves"]) == 2


def test_dual_holdout_rejects_different_ids() -> None:
    with pytest.raises(ValueError, match="same holdout"):
        summarize_dual_holdout_metrics(
            v2_records=_records("v2", ids=["a"]),
            v1_records=_records("v1", ids=["b"]),
            source_layers=[0],
        )
```

- [ ] **Step 2: Verify comparison tests fail**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py -k "dual_holdout" -v
```

Expected: FAIL because the comparison helper is missing.

- [ ] **Step 3: Implement compact comparison**

Load the frozen v1 lens from `source_v1_lens`. Run both trained lenses over the
same v2 holdout and call the ordinary logit lens only once. Store aggregate and
per-item counters, never raw code, decoded tokens or prompt text. Split the 200
ids deterministically into the first and second 100 for the stability block.

- [ ] **Step 4: Verify runtime does not replace v1**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py tests/test_qwen_jlens_v2_runtime.py -q
uv run pytest tests/test_qwen_jlens.py tests/test_qwen_lab_transport.py tests/test_qwen_code_eval_runtime.py -q
```

Expected: PASS and the frozen v1 hash/path assertions remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/src/refusal_direction_replication/qwen_jlens.py projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_runtime.py projects/refusal_direction/tests/test_qwen_jlens_v2.py projects/refusal_direction/tests/test_qwen_jlens_v2_runtime.py
git commit -m "feat(jlens): compare v1 and v2 on shared holdout"
```

### Task 6: Validador específico da v2

**Files:**
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_v2_validate.py`
- Create: `projects/refusal_direction/scripts/validate_qwen_jlens_v2.py`
- Create: `projects/refusal_direction/tests/test_qwen_jlens_v2_validate.py`

**Interfaces:**
- Produces: `validate_qwen_jlens_v2(root: Path) -> dict[str, bool]`.
- Validates: selected count, 200 holdouts, hashes, checkpoint events,
  comparison artifact and absence of text leakage.

- [ ] **Step 1: Write failing validator tests**

```python
def test_v2_validator_accepts_selected_count_and_comparison(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    checks = validate_qwen_jlens_v2(tmp_path)
    assert all(checks.values())


def test_v2_validator_rejects_non_candidate_fit_count(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=175, n_holdout=200)
    with pytest.raises(ValueError, match="fit count candidate"):
        validate_qwen_jlens_v2(tmp_path)
```

- [ ] **Step 2: Verify tests fail because validator is absent**

```powershell
uv run pytest tests/test_qwen_jlens_v2_validate.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Implement validator without changing the v1 validator**

Check compact schema fields, model and dataset revisions, candidate count,
holdout size, stage status, checkpoint sequence, lens shape/hash and matching
shared holdout ids. Reject keys from `JLENS_LEAK_KEYS`.

- [ ] **Step 4: Run both validators**

```powershell
uv run pytest tests/test_qwen_jlens_v2_validate.py tests/test_qwen_jlens_validate.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/src/refusal_direction_replication/qwen_jlens_v2_validate.py projects/refusal_direction/scripts/validate_qwen_jlens_v2.py projects/refusal_direction/tests/test_qwen_jlens_v2_validate.py
git commit -m "feat(jlens): validate long-context results"
```

### Task 7: Transporte e backup validado

**Files:**
- Create: `projects/refusal_direction/tests/test_qwen_jlens_v2_transport.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_colab_transport.py`

**Interfaces:**
- Produces: namespace v2 para archive, staging, checkpoint, status and events.
- Produces: `atomic_replace_validated_jlens_v2_backup(staged, destination)`.
- Guarantees: a cópia anterior não é removida antes da validação completa.

- [ ] **Step 1: Write failing archive and backup tests**

```python
def test_v2_backup_keeps_previous_file_when_staging_is_corrupt(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"previous-valid-copy")
    staged = tmp_path / "download.pt"
    staged.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        atomic_replace_validated_jlens_v2_backup(
            staged,
            destination,
            max_fit_count=800,
        )
    assert destination.read_bytes() == b"previous-valid-copy"


def test_v2_archive_rejects_v1_member_paths() -> None:
    members = [_tar_info("artifacts/qwen25_coder_7b_jlens/status.json")]
    with pytest.raises(ValueError, match="unsafe"):
        validate_qwen_jlens_v2_archive(members)
```

- [ ] **Step 2: Verify transport tests fail**

```powershell
uv run pytest tests/test_qwen_jlens_v2_transport.py -v
```

Expected: FAIL because the v2 transport API is missing.

- [ ] **Step 3: Add a separate v2 allowlist and atomic replacement**

Use:

```python
JLENS_V2_RESULTS_DIR = "results/qwen25_coder_7b_jlens_v2"
JLENS_V2_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_jlens_v2"
JLENS_V2_REMOTE_ARCHIVE = Path("/content/qwen_jlens_v2_results.tar.gz")
REMOTE_JLENS_V2_CHECKPOINT_STAGING = "/content/jlens_v2_fit_checkpoint.pt"
```

Validate staged checkpoint structure with upper bound 800 and identity against
the local v2 profile before `os.replace`. Exclude `corpus.jsonl` from every
archive.

- [ ] **Step 4: Run v2 and v1 transport suites**

```powershell
uv run pytest tests/test_qwen_jlens_v2_transport.py tests/test_qwen_jlens_transport.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/tests/test_qwen_jlens_v2_transport.py projects/refusal_direction/src/refusal_direction_replication/qwen_colab_transport.py
git commit -m "feat(jlens): add validated v2 checkpoint transport"
```

### Task 8: Perfil Colab, scripts remotos e ação de backup

**Files:**
- Create: `projects/refusal_direction/scripts/qwen_run_jlens_v2.py`
- Create: `projects/refusal_direction/scripts/qwen_run_jlens_v2_status.py`
- Create: `projects/refusal_direction/scripts/colab_pack_qwen_jlens_v2_results.py`
- Create: `projects/refusal_direction/scripts/colab_install_jlens_v2_checkpoint.py`
- Modify: `tools/colab/run_refusal.py`
- Modify: `projects/refusal_direction/tests/test_qwen_jlens_v2_transport.py`
- Modify: `projects/refusal_direction/tests/test_colab_orchestrator.py`

**Interfaces:**
- Adds profile: `qwen25-coder-jlens-v2` with session
  `refusal-qwen-jlens-v2`.
- Adds action: `checkpoint-backup`.
- Requires: v2 always starts on A100 and `stop` attempts the final backup before
  terminating the session.

- [ ] **Step 1: Write failing orchestrator tests**

```python
def test_v2_start_forces_a100(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "qwen25-coder-jlens-v2"],
    )
    MODULE.main()
    assert commands[-1][-1] == "A100"


def test_v2_stop_backs_up_before_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "stop", "--profile", "qwen25-coder-jlens-v2"],
    )
    MODULE.main()
    assert command_kinds(commands)[-2:] == ["checkpoint-backup", "stop"]
```

Also assert v2 sync uploads the frozen v1 lens to a v2-owned staging path and
never writes into the remote v1 artifact directory.

- [ ] **Step 2: Verify orchestrator tests fail**

```powershell
uv run pytest tests/test_qwen_jlens_v2_transport.py tests/test_colab_orchestrator.py -k "v2" -v
```

Expected: FAIL because the profile and action are absent.

- [ ] **Step 3: Implement profile and scripts**

Wire start, sync, bootstrap, auth, jlens, status, checkpoint-backup, fetch and
stop. Give `jlens` a 46,800-second execution timeout. Backup downloads to a
temporary file, validates it, then replaces the previous local snapshot.

- [ ] **Step 4: Run all orchestrator and transport tests**

```powershell
uv run pytest tests/test_qwen_jlens_v2_transport.py tests/test_qwen_jlens_transport.py tests/test_colab_orchestrator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/scripts/qwen_run_jlens_v2.py projects/refusal_direction/scripts/qwen_run_jlens_v2_status.py projects/refusal_direction/scripts/colab_pack_qwen_jlens_v2_results.py projects/refusal_direction/scripts/colab_install_jlens_v2_checkpoint.py tools/colab/run_refusal.py projects/refusal_direction/tests/test_qwen_jlens_v2_transport.py projects/refusal_direction/tests/test_colab_orchestrator.py
git commit -m "feat(jlens): operate v2 safely on Colab"
```

### Task 9: Documentação e regressão local completa

**Files:**
- Modify: `projects/refusal_direction/PROCESSO.md`
- Create: `projects/refusal_direction/JLENS_V2.md`

**Interfaces:**
- Documents: corpus, probe, fit selection, checkpoint recovery, holdout,
  comparison and explicit limits.

- [ ] **Step 1: Add a documentation assertion**

```python
def test_v2_documentation_names_safety_boundaries() -> None:
    text = (ROOT / "JLENS_V2.md").read_text(encoding="utf-8")
    for required in (
        "512 tokens",
        "200 amostras",
        "checkpoint",
        "12 horas",
        "não substitui",
    ):
        assert required in text
```

Place this test in `tests/test_qwen_jlens_v2.py`.

- [ ] **Step 2: Verify the documentation test fails**

```powershell
uv run pytest tests/test_qwen_jlens_v2.py::test_v2_documentation_names_safety_boundaries -v
```

Expected: FAIL because `JLENS_V2.md` is absent.

- [ ] **Step 3: Write the operator documentation**

Document the exact command sequence:

```powershell
python tools/colab/run_refusal.py start --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py sync --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py bootstrap --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py auth-hf --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py jlens --profile qwen25-coder-jlens-v2 --timeout 46800
python tools/colab/run_refusal.py checkpoint-backup --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py fetch --profile qwen25-coder-jlens-v2
python tools/colab/run_refusal.py stop --profile qwen25-coder-jlens-v2
```

Explain that `checkpoint-backup` runs in a second terminal at progress
milestones and that v2 does not replace the lens used by the Gradio lab.

- [ ] **Step 4: Run the complete focused regression suite**

```powershell
uv run pytest tests/test_qwen_jlens.py tests/test_qwen_jlens_runtime.py tests/test_qwen_jlens_validate.py tests/test_qwen_jlens_transport.py tests/test_qwen_jlens_v2.py tests/test_qwen_jlens_v2_runtime.py tests/test_qwen_jlens_v2_validate.py tests/test_qwen_jlens_v2_transport.py tests/test_qwen_lab_transport.py tests/test_qwen_code_eval.py tests/test_qwen_code_eval_runtime.py tests/test_colab_orchestrator.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add projects/refusal_direction/JLENS_V2.md projects/refusal_direction/PROCESSO.md projects/refusal_direction/tests/test_qwen_jlens_v2.py
git commit -m "docs(jlens): document long-context execution"
```

### Task 10: Execução A100, monitoramento e comparação

**Files:**
- Generated locally after validated fetch:
  `projects/refusal_direction/results/qwen25_coder_7b_jlens_v2/*.json`
- Generated outside Git:
  `projects/refusal_direction/artifacts/qwen25_coder_7b_jlens_v2/*`
- Modify after results:
  `projects/refusal_direction/JLENS_V2.md`
  and `projects/refusal_direction/PROCESSO.md`

**Interfaces:**
- Consumes: tested v2 profile and orchestrator.
- Produces: final lens, checkpoint, holdout metrics and v1/v2 comparison.

- [ ] **Step 1: Start, sync, bootstrap and authenticate**

Run the first four commands documented in Task 9. Verify `status` reports A100,
the pinned Qwen revision and the v2 profile before starting the fit.

- [ ] **Step 2: Run the probe and fit**

```powershell
python tools/colab/run_refusal.py jlens --profile qwen25-coder-jlens-v2 --timeout 46800
```

Expected: probe selects one candidate under 43,200 seconds, then progress
advances one unit per completed function.

- [ ] **Step 3: Preserve checkpoints at progress milestones**

In a second terminal, run `checkpoint-backup` after the first completed item
and then near 25%, 50%, 75% and 100%. Verify each downloaded payload before
replacing the preceding local copy.

- [ ] **Step 4: Fetch, validate and stop**

Fetch results, run `scripts/validate_qwen_jlens_v2.py`, and stop the A100 in a
`finally` path. Confirm `colab sessions` shows no remaining v2 session.

- [ ] **Step 5: Compare and document**

Record selected fit count, real wall time, lens hash, v1/v2/logit metrics,
subset stability and limitations. Do not change the lab's frozen lens.

- [ ] **Step 6: Commit versionable results and documentation**

```powershell
git add projects/refusal_direction/results/qwen25_coder_7b_jlens_v2 projects/refusal_direction/JLENS_V2.md projects/refusal_direction/PROCESSO.md
git commit -m "exp(jlens): record long-context comparison"
```
