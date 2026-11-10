# IOI Replication

Replication of the GPT-2 Small indirect-object-identification circuit from
Wang et al., [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).

## Two tracks

The notebooks are intentionally split into two tracks:

1. **01–04, didactic:** compact prompts and progressively introduced direct
   logit attribution, activation patching, path patching, and validation. These
   notebooks teach the workflow; they are not paper-scale estimates.
2. **05–10, paper-faithful:** deterministic paired IOI/pABC datasets, the
   published 26-node circuit, semantic path patching, Backup Name Movers,
   faithfulness/completeness/minimality, and Figure 8 adversarial protocols.
   Deviations from historical code are recorded in each artifact.

The paper-faithful track uses TransformerLens **4.3.0** in GPT-2 compatibility
mode (`fold_ln=False`, `center_writing_weights=False`,
`center_unembed=False`, `refactor_factored_attn_matrices=False`). CUDA is used
when available; ordinary notebook loading falls back to CPU. The full runner
requires CUDA by default so an accidental slow CPU refresh fails early; pass
`--allow-cpu` only for an intentional fallback.

## Environment

```text
uv sync
uv run python -m ipykernel install --user --name ioi-replication --display-name "Python (IOI Replication)"
uv run jupyter lab
```

Select `Python (IOI Replication)`. The recorded environment is Python 3.11,
PyTorch CUDA, TransformerLens 4.3.0, and an RTX 3050 with 6144 MiB VRAM.

## Reproduction commands

Run all tests:

```text
uv run pytest -v
```

Execute any one notebook by replacing `<name>`:

```text
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
```

Individual paper-track commands:

```text
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_paper_dataset_and_baseline.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_circuit_discovery.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_path_patching.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/08_backup_name_movers.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_faithfulness_completeness_minimality.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/10_adversarial_examples.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
```

The safe full sequence installs the project kernel, executes 05→10, stops on
the first error, records compact timings/status in
`results/paper/execution_manifest.json`, and rebuilds
`results/paper/summary.json` from canonical artifacts:

```text
uv run python scripts/run_paper_notebooks.py
uv run python scripts/build_paper_summary.py --check
```

## Recorded configurations and cost

Times below are the previous in-place CUDA execution timings and are planning
figures; the current authoritative durations are in the execution manifest.
Peak VRAM was not instrumented, so the table states the tested capacity rather
than inventing a peak measurement.

| Notebook | Seeds | Target / reference N | Batches | Prior time | Tested VRAM |
|---|---|---:|---:|---:|---:|
| 05 baseline | 42 | 256 / — | 32 | 0:32 | 6 GiB |
| 06 circuit discovery | 106 | 4 / matched ABC | full batch | 2:14 | 6 GiB |
| 07 path patching | 107 | 2 / matched ABC | full batch | 1:17 | 6 GiB |
| 08 backups | 308 / 309 | 16 / 300 (10×30 groups) | target full / ref 16 | 0:24 | 6 GiB |
| 09 validation | 409 / 410 / 411 | 100 / 300 (10×30 groups) | target 50 / ref 16 | 13:11 | 6 GiB |
| 10 adversarial | base 510; IO 511; double IO 512; S 513 | 500 / — | 16 | 0:26 | 6 GiB |

Notebook 09 keeps all **10 random subsets, 10 greedy runs, 10 greedy steps, and
10 candidates per step**. The reference in 08–09 is independently sampled,
balanced pABC grouped by pattern/template, not the historical same-batch
reference.

## Observed versus paper

These are comparisons, not equality claims. Sampling, independent reference
construction, TransformerLens compatibility behavior, prompt filtering,
semantic alignment, and finite-sample uncertainty can all shift values.

| Experiment | Observed artifact | Paper target / report | Interpretation |
|---|---:|---:|---|
| Baseline IO−S | 3.644 (N=256) | 3.56 (N=100,000) | Close point estimates; observed SE 0.109 does not establish identity. |
| Baseline IO accuracy / P(IO) | 98.83% / 52.02% | 99.3% / 49% | Accuracy is similar; probability differs by about +3 points. |
| Faithfulness | F(M)=3.732, F(C)=3.120, gap=0.613, retained=83.6% | gap=0.46, retained=87% | Same definitions, but an independent pABC reference and N=100 widen uncertainty (gap CI95 0.348–0.878). |
| Completeness greedy | maximum recorded incompleteness 4.122 across 10 full greedy runs | paper provides the greedy procedure/figure, not a single canonical scalar target | Search is seed- and reference-sensitive; compare the saved curves and removed sets, not only the maximum. |
| Minimality | 26/26 point estimates >1%; 24/26 CI95 lower bounds >1% | 26 circuit nodes assessed against the 1% threshold | Pointwise replication is stronger than interval-supported replication; two nodes remain uncertain. |
| Adversarial baseline | IO−S 3.271, P(IO) 48.59%, S>IO 0.6% | 3.55, 49%, 0.7% | Probability/error align more closely than absolute IO−S. |
| Adversarial IO single | IO−S 0.946, P(IO) 32.24%, S>IO 31.0% | 1.23, 36%, 23.4% | Absolute outcomes diverge, while the observed IO−S delta −2.324 matches the paper target delta −2.32 closely. |
| Adversarial S control | IO−S 3.409, P(IO) 56.06%, S>IO 0.4% | 3.64, 59%, 0.4% | Direction and error rate agree; levels remain lower. |

Notebook 06 has only four prompts and notebook 07 has two. Their unexpected
signs or rankings are diagnostics, not evidence that overturns the published
circuit. Notebook 08 uses 16 target prompts; its corrected compensation
denominator is the primary baseline DLA minus the pABC substitution DLA, both
at baseline scale. Notebook 10 retokenizes after insertion without BOS to avoid
the historical `advex.py` coordinate inconsistency and keeps single- and
double-clause protocols separate.

Machine-readable provenance, configs, and consolidated key results live in
`results/paper/summary.json`; tests derive and compare those values against the
source artifacts to prevent copied-number drift.
