# Qwen Causal J-lens Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir se as J-lens v1 e v2 continuam recuperando o token realmente escolhido pelo Qwen2.5-Coder-7B-Instruct quando a direção de recusa é alterada causalmente.

**Architecture:** Um perfil isolado reutiliza os 64 prompts congelados, os hooks de intervenção e o motor incremental já validados. O runtime gera cada resposta uma única vez, captura quatro estados residuais nos oito primeiros passos e aplica as duas lentes ao mesmo estado. Métricas, bootstrap, análise exploratória, resultados públicos e artefatos brutos permanecem separados.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, `anthropics/jacobian-lens` no commit fixado, pytest, uv, Colab CLI e A100.

## Global Constraints

- Não alterar resultados, direção ou lentes já congelados.
- Usar exatamente 32 IDs MITRE e 32 IDs MITRE-FRR de `00_prepare.json`.
- Fixar modelo, revisão, `bfloat16`, geração gulosa, `batch_size=1` e até 64 tokens novos.
- Usar `baseline`, ablação 20%, adição 50% e controle aleatório ortogonal 20% com semente 1042.
- Ler somente os oito primeiros tokens nas camadas 19, 22, 25 e 26.
- Avaliar v1 e v2 sobre a mesma ativação e contra o `argmax` realmente usado na geração.
- Tratar `pass@5` na camada 25 como desfecho principal e `pass@1`, `pass@10`, rank médio e outras camadas como análises secundárias.
- Calcular bootstrap agrupado por prompt com 10.000 reamostragens, semente 2028 e margem de equivalência de ±5 pontos percentuais.
- Manter prompts e respostas somente em `artifacts/`, nunca nos JSONs versionáveis.
- Registrar término antecipado e denominadores reais, sem preencher observações ausentes.
- Fazer checkpoint atômico após cada par prompt-condição e recusar retomadas com identidade diferente.
- Executar cálculo pesado somente na A100.
- Escrever apenas testes que cobrem contratos plausíveis, cálculo científico, retomada e transporte.
- Observar a falha do teste correspondente antes de cada mudança de produção.

---

### Task 1: Perfil, contratos e estatística pura

**Files:**
- Create: `projects/refusal_direction/configs/qwen25_coder_7b_causal_jlens.json`
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens.py`
- Create: `projects/refusal_direction/tests/test_qwen_causal_jlens.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/pipeline.py`

**Interfaces:**
- Produces: `CAUSAL_JLENS_PROFILE`, `READOUT_LAYERS`, `LENS_VERSIONS`, `READOUT_STEPS`.
- Produces: `observation_key(record) -> tuple[str, str, str, int, int, str]`.
- Produces: `aggregate_fidelity(records) -> dict[str, Any]`.
- Produces: `cluster_bootstrap_difference(records, *, condition, k, layer, lens_version, dataset, resamples, seed) -> dict[str, Any]`.
- Produces: `summarize_behavior_changes(responses, observations) -> dict[str, Any]`.

- [ ] **Step 1: Write failing contract and metric tests**

```python
def test_profile_freezes_the_approved_protocol() -> None:
    config = load_config(ROOT, profile=CAUSAL_JLENS_PROFILE)
    assert config["datasets"] == ["mitre", "mitre_frr"]
    assert config["dataset_size"] == 32
    assert config["conditions"] == [
        "baseline", "ablation", "addition", "random_control"
    ]
    assert config["max_new_tokens"] == 64
    assert config["readout_steps"] == 8
    assert config["readout_layers"] == [19, 22, 25, 26]
    assert config["bootstrap_resamples"] == 10_000
    assert config["bootstrap_seed"] == 2028
    assert config["equivalence_margin"] == 0.05


def test_aggregate_fidelity_uses_one_based_rank() -> None:
    records = [
        _observation(rank=1, example_id="a", step=1),
        _observation(rank=5, example_id="a", step=2),
        _observation(rank=6, example_id="b", step=1),
    ]
    block = aggregate_fidelity(records)["mitre"]["baseline"]["v1"]["25"]
    assert block == {
        "n": 3,
        "prompt_count": 2,
        "pass_at_1": pytest.approx(1 / 3),
        "pass_at_5": pytest.approx(2 / 3),
        "pass_at_10": 1.0,
        "mean_rank": 4.0,
    }


def test_cluster_bootstrap_resamples_prompts_not_tokens() -> None:
    records = _paired_records(prompt_count=4, steps=8)
    result = cluster_bootstrap_difference(
        records,
        condition="ablation",
        k=5,
        layer=25,
        lens_version="v1",
        dataset="mitre",
        resamples=10_000,
        seed=2028,
    )
    assert result["cluster_unit"] == "example_id"
    assert result["resamples"] == 10_000
    assert result["seed"] == 2028
    assert result["estimate"] == pytest.approx(-0.25)


def test_behavior_change_summary_uses_prompt_level_mean_rank() -> None:
    result = summarize_behavior_changes(_responses(), _observations())
    changed = result["mitre"]["ablation"]["changed"]["v1"]
    assert changed["prompt_count"] == 1
    assert changed["layer"] == 25
    assert changed["mean_rank_delta"] == pytest.approx(-2.0)
```

- [ ] **Step 2: Run the new test file and confirm missing imports**

```powershell
cd projects/refusal_direction
uv run pytest tests/test_qwen_causal_jlens.py -v
```

Expected: FAIL because the profile and `qwen_causal_jlens` module do not exist.

- [ ] **Step 3: Add the frozen profile**

The JSON must contain the exact approved values and isolated paths:

```json
{
  "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
  "dtype": "bfloat16",
  "batch_size": 1,
  "format_style": "chat_template",
  "seed": 42,
  "max_new_tokens": 64,
  "readout_steps": 8,
  "readout_layers": [19, 22, 25, 26],
  "datasets": ["mitre", "mitre_frr"],
  "dataset_size": 32,
  "conditions": ["baseline", "ablation", "addition", "random_control"],
  "ablation_coefficient": 0.2,
  "addition_coefficient": 0.5,
  "random_control_coefficient": 0.2,
  "random_control_seed": 1042,
  "bootstrap_resamples": 10000,
  "bootstrap_seed": 2028,
  "equivalence_margin": 0.05,
  "primary_layer": 25,
  "primary_k": 5,
  "purplellama_repository": "meta-llama/PurpleLlama",
  "purplellama_revision": "0654501399e7cbd560db9501965dc09c567a5563",
  "mitre_relative_path": "CybersecurityBenchmarks/datasets/mitre/mitre_benchmark_100_per_category_with_augmentation.json",
  "mitre_frr_relative_path": "CybersecurityBenchmarks/datasets/mitre_frr/mitre_frr.json",
  "jlens_upstream": "anthropics/jacobian-lens",
  "jlens_upstream_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
  "source_prepare": "results/qwen25_coder_7b_code_eval/00_prepare.json",
  "source_direction": "artifacts/qwen25_coder_7b/selected_direction.pt",
  "source_dose_evaluation": "results/qwen25_coder_7b_dose/dose_followup_evaluation.json",
  "source_v1_lens": "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt",
  "source_v2_lens": "artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt",
  "results_dir": "results/qwen25_coder_7b_causal_jlens",
  "artifacts_dir": "artifacts/qwen25_coder_7b_causal_jlens"
}
```

Permit this profile in `validate_profile_config` without relaxing any existing profile.

- [ ] **Step 4: Implement deterministic pure helpers**

Use one-based ranks, group metrics by
`dataset/condition/lens_version/layer`, and calculate each bootstrap replicate
by sampling prompt IDs with replacement. The sampled prompt contributes all its
available token steps. The exploratory summary first averages ranks within each
prompt and only then computes condition-minus-baseline differences.

```python
def observation_key(record: Mapping[str, Any]) -> tuple[str, str, str, int, int, str]:
    return (
        str(record["dataset"]),
        str(record["example_id"]),
        str(record["condition"]),
        int(record["generation_step"]),
        int(record["layer"]),
        str(record["lens_version"]),
    )


def metric_block(ranks: Sequence[int]) -> dict[str, float | int]:
    if not ranks:
        return {"n": 0, "pass_at_1": None, "pass_at_5": None,
                "pass_at_10": None, "mean_rank": None}
    return {
        "n": len(ranks),
        "pass_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "pass_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "pass_at_10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }
```

- [ ] **Step 5: Run focused and profile regression tests**

```powershell
uv run pytest tests/test_qwen_causal_jlens.py tests/test_qwen_code_eval.py tests/test_qwen_jlens.py tests/test_qwen_jlens_v2.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add configs/qwen25_coder_7b_causal_jlens.json src/refusal_direction_replication/qwen_causal_jlens.py src/refusal_direction_replication/pipeline.py tests/test_qwen_causal_jlens.py
git commit -m "feat(jlens): add causal fidelity contracts"
```

### Task 2: Readout duplo e geração sincronizada

**Files:**
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_runtime.py`
- Create: `projects/refusal_direction/tests/test_qwen_causal_jlens_runtime.py`
- Reuse without changing: `qwen_lab_readout.py`, `qwen_lab_runtime.py`, `qwen_code_eval_runtime.py`

**Interfaces:**
- Produces: `DualLensReadout.rank_target(residual, *, layer, target_token_id) -> dict[str, int]`.
- Produces: `stream_causal_generation(..., readout_steps=8, layers=(19,22,25,26)) -> GenerationResult`.
- Produces: `GenerationResult(generated_token_ids, response_text, observations, terminal_reason)`.
- Guarantees: ambos os ranks usam a mesma ativação e o mesmo token alvo.
- Guarantees: os hooks são removidos após sucesso, EOS e exceção.

- [ ] **Step 1: Write failing parity and synchronization tests**

```python
def test_dual_readout_ranks_the_same_target_from_the_same_residual() -> None:
    readout = DualLensReadout(
        lenses={"v1": _lens(offset=0.0), "v2": _lens(offset=1.0)},
        bound_model=_bound_model(),
    )
    ranks = readout.rank_target(
        torch.tensor([1.0, 2.0]),
        layer=25,
        target_token_id=3,
    )
    assert ranks == {"v1": 1, "v2": 2}


def test_stream_records_four_layers_two_lenses_for_first_eight_steps() -> None:
    result = stream_causal_generation(
        model=_greedy_fixture(tokens=10),
        tokenizer=_tokenizer(),
        readout=_dual_readout(),
        layers=(19, 22, 25, 26),
        readout_steps=8,
        max_new_tokens=64,
        **_intervention_inputs("baseline"),
    )
    assert len(result.observations) == 8 * 4 * 2
    assert {row["target_token_id"] for row in result.observations} == set(
        result.generated_token_ids[:8]
    )
    assert len(result.generated_token_ids) == 10


@pytest.mark.parametrize("terminal", ["eos", "error"])
def test_stream_always_removes_hooks(terminal: str) -> None:
    model = _hook_counting_model(terminal=terminal)
    with contextlib.suppress(RuntimeError):
        stream_causal_generation(model=model, **_fixture_args())
    assert model.active_hook_count == 0
```

- [ ] **Step 2: Run focused tests and confirm missing runtime**

```powershell
uv run pytest tests/test_qwen_causal_jlens_runtime.py -k "dual_readout or stream" -v
```

Expected: FAIL because the causal runtime does not exist.

- [ ] **Step 3: Implement target-token ranks**

Reuse `incremental_logits_from_residual` and `rank_one_based`; do not duplicate
the Jacobian transport formula.

```python
class DualLensReadout:
    def __init__(self, lenses: Mapping[str, Any], bound_model: Any) -> None:
        if tuple(lenses) != ("v1", "v2"):
            raise ValueError("lenses must be ordered as v1, v2")
        self._lenses = dict(lenses)
        self._bound_model = bound_model

    def rank_target(
        self,
        residual: torch.Tensor,
        *,
        layer: int,
        target_token_id: int,
    ) -> dict[str, int]:
        return {
            version: rank_one_based(
                incremental_logits_from_residual(
                    lens, self._bound_model, residual, layer=layer
                ),
                target_token_id,
            )
            for version, lens in self._lenses.items()
        }
```

- [ ] **Step 4: Implement one-pass generation with KV cache**

Adapt the tested sequence in `InterventionEngine._stream`. Register one capture
hook per requested layer, calculate the greedy `argmax`, rank it with both
lenses for steps 1 through 8, and continue generation without readout through
step 64. Store the complete decoded response only in `GenerationResult`, which
the staged runtime writes under `artifacts/`.

- [ ] **Step 5: Verify canonical parity for both lenses**

Use a small fixture to compare `rank_target` against
`JacobianLens.apply`. The remote pilot will repeat parity with the real lenses.

```powershell
uv run pytest tests/test_qwen_causal_jlens_runtime.py tests/test_qwen_lab_runtime.py tests/test_qwen_lab_readout.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/refusal_direction_replication/qwen_causal_jlens_runtime.py tests/test_qwen_causal_jlens_runtime.py
git commit -m "feat(jlens): stream synchronized causal readout"
```

### Task 3: Estágios, identidade e retomada

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_runtime.py`
- Modify: `projects/refusal_direction/tests/test_qwen_causal_jlens_runtime.py`
- Create: `projects/refusal_direction/scripts/qwen_run_causal_jlens.py`
- Create: `projects/refusal_direction/scripts/qwen_run_causal_jlens_status.py`

**Interfaces:**
- Produces: `source_context(root, config) -> dict[str, Any]`.
- Produces: `PanelCheckpoint.create(identity)`, `PanelCheckpoint.load(path, expected_identity)`, `pending(matrix)` e `mark_complete(dataset, example_id, condition)`.
- Produces: `append_jsonl_atomic(path, records)`, que regrava por arquivo temporário e `os.replace`.
- Produces: `run_prepare_stage`, `run_pilot_stage`, `run_panel_stage`, `run_analysis_stage`, `run_finalize_stage`.
- Produces: `run_qwen_causal_jlens(root, *, profile, run_id=None)`.
- Persists: one JSONL response record, observation rows and checkpoint entry per prompt-condition pair.

- [ ] **Step 1: Write failing source and checkpoint tests**

```python
def test_source_context_requires_exact_frozen_inputs(tmp_path: Path) -> None:
    _seed_sources(tmp_path)
    context = source_context(tmp_path, _config(tmp_path))
    assert context["direction_layer"] == 18
    assert context["direction_offset"] == -1
    assert set(context["source_artifact_hashes"]) == {
        "prepare", "direction", "dose_evaluation", "v1_lens", "v2_lens"
    }
    _rewrite_dose(tmp_path, addition=0.4)
    with pytest.raises(ValueError, match="coefficient"):
        source_context(tmp_path, _config(tmp_path))


def test_resume_skips_only_complete_matching_pairs(tmp_path: Path) -> None:
    checkpoint = PanelCheckpoint.create(_identity("run-a"))
    checkpoint.mark_complete("mitre", "id-1", "baseline")
    checkpoint.write(tmp_path / "checkpoint.json")
    loaded = PanelCheckpoint.load(
        tmp_path / "checkpoint.json", expected_identity=_identity("run-a")
    )
    assert loaded.pending(_matrix()) == _matrix()[1:]
    with pytest.raises(ValueError, match="identity"):
        PanelCheckpoint.load(
            tmp_path / "checkpoint.json", expected_identity=_identity("run-b")
        )
```

- [ ] **Step 2: Confirm failure**

```powershell
uv run pytest tests/test_qwen_causal_jlens_runtime.py -k "source_context or resume" -v
```

Expected: FAIL for missing source and checkpoint interfaces.

- [ ] **Step 3: Implement preparation and frozen selection**

Load the IDs and hashes from the existing code-eval prepare result. Rehydrate
the exact records from the pinned PurpleLlama revision, then abort on any ID,
hash or order mismatch. Verify the direction layer/offset, dose coefficients,
model revision and hashes of both lens files before loading the model.

- [ ] **Step 4: Implement atomic panel progress**

```python
for dataset, example, condition in checkpoint.pending(expected_matrix):
    result = stream_causal_generation(...)
    append_jsonl_atomic(responses_path, compact_response(result))
    append_jsonl_atomic(observations_path, result.observations)
    checkpoint.mark_complete(dataset, example.example_id, condition)
    checkpoint.write(checkpoint_path)
```

The response artifact may contain prompt and generated text. Compact result
files may contain only IDs, hashes, condition, refusal label, token target,
layer, lens version and rank.

- [ ] **Step 5: Implement pilot, analysis and finalize**

The pilot uses two fixed IDs from each dataset in all four conditions. It must
check real-lens parity, finite logits, hook cleanup, expected keys and no text
leak. The analysis calls the pure helpers from Task 1. Finalize writes the six
approved result files and the complete provenance matrix.

- [ ] **Step 6: Run runtime tests**

```powershell
uv run pytest tests/test_qwen_causal_jlens_runtime.py tests/test_qwen_observability.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/refusal_direction_replication/qwen_causal_jlens_runtime.py tests/test_qwen_causal_jlens_runtime.py scripts/qwen_run_causal_jlens.py scripts/qwen_run_causal_jlens_status.py
git commit -m "feat(jlens): add resumable causal fidelity run"
```

### Task 4: Validador offline

**Files:**
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_causal_jlens_validate.py`
- Create: `projects/refusal_direction/tests/test_qwen_causal_jlens_validate.py`
- Create: `projects/refusal_direction/scripts/validate_qwen_causal_jlens.py`

**Interfaces:**
- Produces: `validate_qwen_causal_jlens(root) -> dict[str, bool]`.
- Recomputes all metrics and bootstrap intervals from compact observations.
- Rejects raw text, duplicate keys, invalid early termination and identity drift.

- [ ] **Step 1: Write failing valid-tree and corruption tests**

```python
def test_validator_accepts_complete_recomputable_tree(tmp_path: Path) -> None:
    _seed_complete_tree(tmp_path)
    checks = validate_qwen_causal_jlens(tmp_path)
    assert checks
    assert all(checks.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_observation", "duplicate"),
        ("wrong_rank_metric", "metric"),
        ("raw_response_in_results", "raw text"),
        ("missing_early_stop_marker", "termination"),
        ("wrong_bootstrap_seed", "bootstrap"),
        ("wrong_source_hash", "hash"),
    ],
)
def test_validator_rejects_scientific_corruption(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _seed_complete_tree(tmp_path)
    _mutate(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        validate_qwen_causal_jlens(tmp_path)
```

- [ ] **Step 2: Confirm validator tests fail**

```powershell
uv run pytest tests/test_qwen_causal_jlens_validate.py -v
```

Expected: FAIL because validator files do not exist.

- [ ] **Step 3: Implement strict recomputation**

Check the exact 64 IDs, four conditions, layers, lens versions and available
steps. Recompute `pass@k`, mean rank, prompt-clustered bootstrap, equivalence
flags and behavior-change summaries. Compare numeric values with a fixed
tolerance and require every result file to share run ID, config hash,
implementation commit and source hashes.

- [ ] **Step 4: Run validator and nearby regressions**

```powershell
uv run pytest tests/test_qwen_causal_jlens_validate.py tests/test_qwen_code_eval_validate.py tests/test_qwen_jlens_v2_validate.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/refusal_direction_replication/qwen_causal_jlens_validate.py tests/test_qwen_causal_jlens_validate.py scripts/validate_qwen_causal_jlens.py
git commit -m "test(jlens): validate causal fidelity results"
```

### Task 5: Transporte e perfil Colab A100

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_colab_transport.py`
- Modify: `projects/refusal_direction/tests/test_qwen_code_eval_transport.py`
- Create: `projects/refusal_direction/tests/test_qwen_causal_jlens_transport.py`
- Create: `projects/refusal_direction/scripts/colab_install_causal_jlens.py`
- Create: `projects/refusal_direction/scripts/colab_pack_qwen_causal_jlens_results.py`
- Modify: `tools/colab/run_refusal.py`

**Interfaces:**
- Adds profile: `qwen25-coder-causal-jlens`, forced to A100.
- Uploads: direction, v1 lens and v2 lens in checked chunks.
- Fetches: only the isolated causal result and artifact trees.
- Adds action: `causal-jlens`.

- [ ] **Step 1: Write failing transport tests**

```python
def test_causal_profile_requires_three_frozen_sources() -> None:
    direction, v1, v2 = require_local_causal_jlens_sources(PROJECT)
    assert direction.name == "selected_direction.pt"
    assert v1.name == "qwen25_coder_7b_jacobian_lens.pt"
    assert v2.name == "qwen25_coder_7b_jacobian_lens_v2.pt"


def test_causal_archive_is_confined_and_preserves_raw_audit(tmp_path: Path) -> None:
    _seed_causal_tree(tmp_path)
    archive = tmp_path / "causal.tar.gz"
    pack_qwen_causal_jlens_results(tmp_path, archive)
    extracted = tmp_path / "extracted"
    extract_qwen_causal_jlens_archive(archive, extracted)
    assert (extracted / "results/qwen25_coder_7b_causal_jlens/03_analysis.json").is_file()
    assert (extracted / "artifacts/qwen25_coder_7b_causal_jlens/responses.jsonl").is_file()


def test_orchestrator_exposes_forced_a100_causal_action() -> None:
    module = _orchestrator()
    assert module.PROFILE_SESSIONS["qwen25-coder-causal-jlens"]
    assert "qwen25-coder-causal-jlens" in module.A100_PROFILES
    command = module.causal_jlens_command("session", timeout=7200)
    assert command[-3].endswith("qwen_run_causal_jlens.py")
```

- [ ] **Step 2: Confirm failure**

```powershell
uv run pytest tests/test_qwen_causal_jlens_transport.py -v
```

Expected: FAIL for missing profile and transport functions.

- [ ] **Step 3: Implement chunked source installation and archive allowlist**

Use existing split manifests and atomic install helpers. The allowlist permits
the six compact JSON files plus causal `status.json`, `events.jsonl`,
`checkpoint.json`, `responses.jsonl` and `observations.jsonl`. It must reject
paths outside the causal profile and all source model/lens files.

- [ ] **Step 4: Add orchestration routes**

Add profile/session mapping, forced A100 startup, sync of three sources,
bootstrap installer, status script, `causal-jlens` execution, pack/fetch,
confined extraction and normal stop behavior.

- [ ] **Step 5: Run transport and orchestrator regressions**

```powershell
uv run pytest tests/test_qwen_causal_jlens_transport.py tests/test_qwen_code_eval_transport.py tests/test_qwen_jlens_v2_transport.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/refusal_direction_replication/qwen_colab_transport.py tests/test_qwen_code_eval_transport.py tests/test_qwen_causal_jlens_transport.py scripts/colab_install_causal_jlens.py scripts/colab_pack_qwen_causal_jlens_results.py ../../tools/colab/run_refusal.py
git commit -m "feat(jlens): add causal A100 profile"
```

### Task 6: Revisão, piloto e painel na A100

**Files:**
- Modify only if a defect is found: files from Tasks 1 through 5.
- Produce remotely, then fetch:
  `projects/refusal_direction/results/qwen25_coder_7b_causal_jlens/`
- Produce ignored:
  `projects/refusal_direction/artifacts/qwen25_coder_7b_causal_jlens/`

**Interfaces:**
- Input: implementation commit with a clean scientific diff.
- Output: validated pilot, complete panel, analysis, evaluation and manifest.

- [ ] **Step 1: Run the focused local suite**

```powershell
cd projects/refusal_direction
uv run pytest tests/test_qwen_causal_jlens.py tests/test_qwen_causal_jlens_runtime.py tests/test_qwen_causal_jlens_validate.py tests/test_qwen_causal_jlens_transport.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the relevant regression suite**

```powershell
uv run pytest tests/test_qwen_code_eval.py tests/test_qwen_code_eval_runtime.py tests/test_qwen_code_eval_validate.py tests/test_qwen_code_eval_transport.py tests/test_qwen_lab_runtime.py tests/test_qwen_lab_readout.py tests/test_qwen_jlens.py tests/test_qwen_jlens_v2.py -q
```

Expected: PASS.

- [ ] **Step 3: Review the implementation diff and fix only concrete findings**

Check intervention semantics, target-token alignment, lens identity, bootstrap
unit, text leakage, checkpoint ordering and guaranteed stop. Re-run the focused
suite after any fix and commit the fix separately.

- [ ] **Step 4: Start and prepare the A100**

```powershell
cd ../..
uv run python tools/colab/run_refusal.py start --profile qwen25-coder-causal-jlens
uv run python tools/colab/run_refusal.py sync --profile qwen25-coder-causal-jlens
uv run python tools/colab/run_refusal.py bootstrap --profile qwen25-coder-causal-jlens
uv run python tools/colab/run_refusal.py auth-hf --profile qwen25-coder-causal-jlens
```

Expected: active A100 session, exact source hashes installed and Hugging Face
authentication accepted.

- [ ] **Step 5: Run the pilot**

The runtime executes the pilot before the panel and aborts if parity, hooks,
finitude or matrix checks fail.

```powershell
uv run python tools/colab/run_refusal.py causal-jlens --profile qwen25-coder-causal-jlens --timeout 7200
```

Expected first gate: 16 responses and up to 1.024 valid observations with
`pilot_status=pass`. Continue to the panel only after this gate.

- [ ] **Step 6: Monitor without local model computation**

```powershell
uv run python tools/colab/run_refusal.py status --profile qwen25-coder-causal-jlens
```

Expected: monotonic completed prompt-condition pairs up to 256. If interrupted,
repeat `causal-jlens`; the checkpoint must skip validated completed pairs.

- [ ] **Step 7: Fetch and validate before stopping**

```powershell
uv run python tools/colab/run_refusal.py fetch --profile qwen25-coder-causal-jlens
cd projects/refusal_direction
uv run python scripts/validate_qwen_causal_jlens.py
```

Expected: every validator contract PASS.

- [ ] **Step 8: Stop the GPU and verify no active session**

```powershell
cd ../..
uv run python tools/colab/run_refusal.py stop --profile qwen25-coder-causal-jlens
uv run python tools/colab/run_refusal.py probe
```

Expected: causal session absent. If stop reports uncertainty, retry the printed
stop command before any documentation work.

### Task 7: Resultados e documentação do TCC

**Files:**
- Create: `projects/refusal_direction/CAUSAL_JLENS.md`
- Modify: `projects/refusal_direction/PROCESSO.md`
- Modify: `docs/ACOMPANHAMENTO_EXPERIMENTOS.md`
- Modify: `projects/refusal_direction/README.md`
- Modify, after integrating with the portal branch:
  `website/docs/experimentos/direcao-recusa.mdx`
- Version: `projects/refusal_direction/results/qwen25_coder_7b_causal_jlens/*.json`

**Interfaces:**
- Consumes only validated compact results plus fixed-rule examples from ignored audit artifacts.
- Produces a human-readable account of context, problem, method, incidents, results and limitations.

- [ ] **Step 1: Write the result narrative from validated values**

Build the narrative directly from these validated fields:

```markdown
## Pergunta
A lente continuou recuperando o token escolhido quando a direção de recusa foi alterada?

## Decisões anteriores à execução
- 64 prompts congelados
- quatro condições
- oito passos de leitura
- camada 25 e pass@5 como desfecho principal
- margem de equivalência de cinco pontos percentuais

## Resultado principal
Para cada entrada de `causal_jlens_evaluation.json.results.primary`, escrever
a lente, o conjunto, a condição, a diferença de pass@5, os limites inferior e
superior do IC 95% e o indicador de equivalência.

## Mudanças comportamentais
Transcrever `results.refusal_rates` e `results.behavior_change_analysis`,
sempre incluindo `prompt_count` antes das porcentagens.

## Limites
Declarar que a recusa vem de classificador automático, que cada conjunto tem
32 prompts, que as lentes foram ajustadas fora do MITRE e que os subgrupos
changed/unchanged são exploratórios.
```

Describe failures and interruptions rather than hiding them.

Select audit examples by a rule fixed before reading their text: for each
dataset, non-baseline condition and changed/unchanged group, take the
lexicographically first `example_id` when that group is non-empty. Report the
ID, condition, automatic label and ranks; summarize prompt content without
copying harmful instructions into versioned documentation.

- [ ] **Step 2: Update process tracking and portal**

Keep the writing factual and first-person where it describes decisions. Avoid
marketing language, em dashes, claims of “thought reading” and causal claims
about the lens itself. If the portal files are not present on this worktree,
perform that edit only after integrating the implementation branch into the
branch that contains `website/`; do not recreate the portal here.

- [ ] **Step 3: Re-run validation and documentation checks**

```powershell
cd projects/refusal_direction
uv run python scripts/validate_qwen_causal_jlens.py
cd ../..
uv run python tools/update_experiment_tracking.py --check
```

If `website/` is present:

```powershell
cd website
npm run build
```

Expected: scientific validator PASS, tracking current, and portal build PASS
when applicable.

- [ ] **Step 4: Commit results and documentation**

```powershell
git add projects/refusal_direction/results/qwen25_coder_7b_causal_jlens projects/refusal_direction/CAUSAL_JLENS.md projects/refusal_direction/PROCESSO.md projects/refusal_direction/README.md docs/ACOMPANHAMENTO_EXPERIMENTOS.md
git commit -m "results(jlens): document causal fidelity study"
```

- [ ] **Step 5: Record the final handoff**

Report the implementation commit, run ID, A100 status, validator result,
primary intervals, exploratory denominators and any portal update that still
depends on branch integration.
