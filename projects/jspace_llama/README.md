# J-space on Llama 3.1 8B Instruct

This project will test a minimal causal J-space intervention on
`meta-llama/Llama-3.1-8B-Instruct` using the prefitted lens at
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`.

The intended workflow runs on a Colab NVIDIA L4 with gated Hugging Face model
access, downloads model and lens assets only into ignored caches, and retains
only compact aggregate JSON results. The planned causal check swaps the
intermediate concept `spider` for `ant` and compares the resulting answer with
baseline and control interventions.

The reusable core now provides pinned configuration, CUDA/runtime validation,
memory-bounded readout and interventions, controls, provenance capture, and a
notebook-callable artifact pipeline. `notebooks/01_readout_and_causal_swap.ipynb`
is the thin entry point for the real run. Results are not yet available; the 8B
model is intentionally not executed in the repository test suite.

## Colab L4 workflow

Install the verified CLI version:

```powershell
uv tool install google-colab-cli==0.7.5
```

First obtain access to the gated `meta-llama/Llama-3.1-8B-Instruct`
repository with your Hugging Face account, then save its token at the standard
local cache path:

```powershell
uv run hf auth login
```

From the repository root, run:

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb --timeout 7200
python tools/colab/run_jspace.py fetch
python tools/colab/run_jspace.py stop
```

The runner uses ADC authentication and the `jspace-llama` session by default.
The archive contains only the project inputs needed remotely, and bootstrap
installs `uv==0.12.1` before exporting the locked environment. `fetch` accepts
only `results/execution_manifest.json` and `results/final_evaluation.json`;
both must exist. Notebook execution validates the fresh ignored
`_output.ipynb` written by colab-cli and retains it for diagnosis. Execution
failures leave the Colab session available for diagnosis, so stop it
explicitly. If L4 allocation is unavailable, stop the partial session and
explicitly retry with `--gpu T4` or `--gpu A100`.

## Result artifacts and validation

The real pipeline writes raw continuations, rendered prompts, workspace slots,
and other diagnostics only to the ignored
`artifacts/readout_and_causal_swap.json`. The two fetched files,
`results/execution_manifest.json` and `results/final_evaluation.json`, contain
compact aggregate evidence and reproducibility metadata only.

Validate fetched or fixture artifacts locally without loading the 8B model:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_results.py
uv run pytest -q
```

`success` means that provenance is complete and at least one tested alpha
passes every primary `main`-prompt gate: baseline and no-op preserve eight,
the random control neither degrades nor comparably reroutes to six, the target
cleanly reroutes to six without eight or `ant`, and the selected-layer readout
contains `spider`. Any expected runtime failure, incomplete provenance, or
failed gate is `inconclusive` with a machine-readable reason; it is not negative
evidence. Paraphrase counts are secondary transparency data and are not a
general claim. Unexpected programming errors are recorded best-effort in the
manifest and then re-raised.
