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
