# Colab CLI

Install:

```powershell
uv tool install google-colab-cli==0.7.5
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

## Executar J-space no Llama 3.1 8B Instruct

Antes de iniciar, é preciso solicitar e receber acesso ao repositório *gated*
`meta-llama/Llama-3.1-8B-Instruct` com a mesma conta do Hugging Face usada
localmente. O token dessa conta deve estar no cache padrão:

```powershell
uv run hf auth login
```

O projeto mantém dois fluxos independentes. O primeiro reproduz a tentativa
causal já executada, cujo resultado permanece inconclusivo e cujos JSONs
canônicos não foram alterados:

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

O segundo executa o atlas observacional. Ele usa os 93 prompts multihop
vendorizados da Anthropic na revisão
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, sob licença Apache 2.0. O notebook
fornece o prompt bruto, sem *chat template*, e não fornece o campo `target` ao
modelo. Ele compara J-lens direta, logit lens e, no painel fixo de 12 prompts,
`gradient_pursuit`; não gera texto e não modifica ativações.

Na raiz do repositório, o fluxo exato é:

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb --timeout 7200
python tools/colab/run_jspace.py fetch --result-set observation
python tools/colab/run_jspace.py stop
```

O perfil causal padrão aceita somente `results/execution_manifest.json` e
`results/final_evaluation.json`. O perfil `observation` aceita somente os dois
JSONs compactos do atlas e suas seis figuras PNG/SVG. Essa separação impede que
um download observacional substitua os resultados causais.

`auth-hf` envia temporariamente o token em cache e o move para o cache remoto
padrão do Hugging Face sem exibi-lo. `execute` exige notebook e timeout,
valida o `_output.ipynb` novo produzido pela colab-cli e preserva esse arquivo
ignorado para diagnóstico. O bootstrap remoto instala `uv==0.12.1` antes de
exportar o ambiente fixado e substitui
`/content/jspace_llama_project`; o nome diferente do pacote evita que o
checkout sombreie a instalação editável de `jspace_llama` quando o runtime
está em `/content`.

A validação local de um conjunto observacional completo é:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_observation.py
```

O validador exige coerência entre schemas, hashes, contagens, métricas
recalculadas, proveniência e o artefato bruto ignorado. Antes da primeira
execução remota, ele deve terminar com status 1 e informar a ausência de
`jspace_observation_manifest.json`, em vez de validar os JSONs causais como
atlas. Um `success` observacional significa que o conjunto está completo e
válido; não significa `pass@1`, `pass@5` ou `pass@10` alto.

Nenhum perfil baixa gerações brutas, tensores, pesos, lentes, caches ou
credenciais. Uma falha deixa a sessão aberta para diagnóstico, portanto
`stop` deve ser executado explicitamente. Se a L4 não estiver disponível,
encerre a sessão parcial antes de tentar `start --gpu T4` ou `--gpu A100`.
