# Colab CLI

Install:

```powershell
uv tool install google-colab-cli
```

Authenticate once, manually:

```powershell
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Run Gemma 2B:

```powershell
python tools/colab/run_refusal.py probe
python tools/colab/run_refusal.py start --gpu L4
python tools/colab/run_refusal.py sync
python tools/colab/run_refusal.py bootstrap
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/01_dataset_and_baseline.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/02_extract_candidate_directions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/03_causal_interventions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/04_evaluation_and_validation.ipynb
python tools/colab/run_refusal.py fetch
python tools/colab/run_refusal.py stop
```

If L4 allocation is denied, stop any partial session and explicitly repeat
`start --gpu T4`. Colab Pro does not guarantee a particular accelerator.

The archive excludes local environments, caches, and `artifacts/`. Bootstrap
downloads the pinned upstream splits inside the remote runtime. `fetch`
downloads only aggregate files from `results/`; raw prompts and generated
completions stay in the remote ignored directory.

## Run J-space on Llama 3.1 8B Instruct

Before starting, request and receive access to the gated
`meta-llama/Llama-3.1-8B-Instruct` repository with the same Hugging Face
account used locally. Store that account's token at the standard cache path:

```powershell
uv run hf auth login
```

Run the workflow explicitly from the repository root:

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

`auth-hf` temporarily uploads the cached token and moves it to the standard
remote Hugging Face cache without displaying it. `execute` requires both the
notebook and timeout. `fetch` fails unless both compact aggregate artifacts,
`results/execution_manifest.json` and `results/final_evaluation.json`, exist.
It never downloads raw generations, tensors, model or lens files, caches, or
credentials. A failed execution intentionally leaves the session running for
diagnosis; run `stop` explicitly when finished. If L4 allocation is denied,
explicitly stop the partial session and retry `start --gpu T4` (or `A100`).
