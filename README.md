# IOI Replication

Incremental replication of the GPT-2 Small indirect object identification
(IOI) circuit analysis, using TransformerLens 4 compatibility mode.

The project is organized as four learning notebooks:

1. `01_ioi_baseline.ipynb`: IOI prompts and logit difference
2. `02_direct_logit_attribution.ipynb`: residual decomposition and direct attribution
3. `03_activation_patching.ipynb`: causal activation patching
4. `04_path_patching_and_validation.ipynb`: path patching and circuit validation

Each notebook starts from the outputs established in the previous phase, while
remaining independently executable.

## Start Jupyter

```powershell
uv sync
uv run jupyter lab
```

Select the `Python (IOI Replication)` kernel.

## Reproduce

Run the tests:

```powershell
uv run pytest -v
```

Execute an individual notebook:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace `
  notebooks/01_ioi_baseline.ipynb `
  --ExecutePreprocessor.kernel_name=ioi-replication `
  --ExecutePreprocessor.timeout=1200
```

## Scientific scope

This project reproduces the qualitative IOI analysis workflow:

- positive clean IOI logit difference;
- exact reconstruction of the logit difference from the scaled residual stream;
- direct attribution and attention inspection for candidate Name Mover Heads;
- residual-position and head-level activation patching;
- selected sender-to-receiver path interventions;
- circuit faithfulness, completeness, and minimality checks using mean ablation.

Assertions validated in this environment include tensor shapes, normalized
recovery endpoints, equality between manual and built-in residual patching,
and numerical reconstruction of the baseline logit difference.

Exact values and head rankings are version-, prompt-, and intervention-sensitive.
They should not be treated as universal constants or as a full reproduction of
every experiment in the original IOI paper. In particular, the path-patching
notebook scans selected candidate edges rather than every possible edge, and
its validation uses the eight local IOI prompts plus matched ABC controls.
