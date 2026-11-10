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
