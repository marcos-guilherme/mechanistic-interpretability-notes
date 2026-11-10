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
is the thin entry point for the real run. The observed execution is documented
below; the 8B model is intentionally not executed in the repository test suite.

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
installs `uv==0.12.1` before exporting the locked environment. Bootstrap
atomically replaces `/content/jspace_llama_project`; this non-package
basename prevents the `/content` checkout from shadowing the editable
`jspace_llama` installation as a namespace package. `fetch` accepts only
`results/execution_manifest.json` and `results/final_evaluation.json`; both
must exist. Notebook execution validates the fresh ignored
`_output.ipynb` written by colab-cli and retains it for diagnosis. Execution
failures leave the Colab session available for diagnosis, so stop it
explicitly. If L4 allocation is unavailable, stop the partial session and
explicitly retry with `--gpu T4` or `--gpu A100`.

## Observed run

Run `a116c3da-bb8f-4a31-a39a-4fd09dd6fcad` exercised the software, Colab L4,
and compact-artifact path through Hugging Face asset resolution. It resolved
model revision `0e9e39f249a16976918f6564b8830bc894c89659` and lens revision
`d2527c3cdbaa60dd4b65cbca1604921f044e989a`, then ended as `inconclusive` with
reason code `asset_load_failed` because the Hugging Face account was not
authorized for the gated model repository.

The model weights were not downloaded, so no J-space readout or causal
intervention ran. This outcome is neither positive nor negative evidence for
J-space in Llama. Consequently, lens integrity and readout spider presence are
recorded as `evaluated: false` with `null` measurements, rather than as false
scientific observations. The source revision was
`bda81311ed758749598d9cd41fb7bbf0d5d3897a`; its dirty flag is `true` because
unrelated user work existed outside this project at sync time.

The two compact result artifacts are intentionally versioned as an
inconclusive execution record. To rerun, first obtain official gated-model
access for the Hugging Face account, then repeat `auth-hf`, `execute`, `fetch`,
and the local validation commands below. A future real rerun will replace this
inconclusive record with newly measured values; the existing run is not
retrofitted with operational stage metadata that it did not originally emit.

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

`success` means that scoped source provenance is clean and digest-verified,
lens integrity and readout are explicitly evaluated, and at least one tested
alpha passes every primary `main`-prompt gate. The constrained answer parser
must uniquely read eight for baseline and no-op, a coherent non-six answer for
the random control, and uniquely six for the target without `ant` leakage or
degradation. Negated, conflicting, missing, repetitive, or excessively
incoherent answers do not pass. Any expected runtime failure, incomplete
provenance, unmeasured state, or failed gate is `inconclusive` with a
machine-readable reason; it is not negative evidence. Paraphrase counts are
secondary transparency data and are not a general claim. Unexpected
programming errors are recorded best-effort in the manifest and then re-raised.
