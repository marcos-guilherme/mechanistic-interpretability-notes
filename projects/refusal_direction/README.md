# Refusal Direction Replication

Replicação parcial de Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, usando `google/gemma-2b-it`.

O experimento extrai direções pela diferença entre as médias das ativações de pedidos nocivos e inofensivos, testa cada candidata por ablação e adição e compara a direção escolhida com um controle aleatório ortogonal. O Git preserva métricas agregadas e valores numéricos por exemplo, além de configurações e proveniência, sem os textos dos prompts ou das gerações.

A narrativa de estudo e reprodução, incluindo decisões, evidências e
limitações, está em [`PROCESSO.md`](PROCESSO.md).

## Resultado observado

Execução de 10 de novembro de 2026 em uma NVIDIA L4:

- direção selecionada: camada 12, último token do template/sufixo, o *newline*
  após o marcador `model`;
- recusa em pedidos nocivos pós-filtro (N=28): 100% na linha de base e 0% após
  ablação;
- recusa indevida em pedidos inofensivos (N=32): 0% na linha de base e 93,75%
  após adição (30/32);
- controle aleatório: 100% de recusa nos pedidos nocivos pós-filtro (N=28);
- KL em pedidos inofensivos (N=32) após ablação: 0,0306, abaixo do limite de
  0,1;
- variação média da entropia cruzada após ablação: -0,4664, contra -0,4652 no
  controle; a proximidade torna essa métrica evidência fraca de especificidade,
  como detalhado no [PROCESSO](PROCESSO.md).

Esses números reproduzem o padrão causal central: remover a direção elimina a
recusa, adicioná-la induz recusa e a ablação com uma orientação aleatória
ortogonal não produz o mesmo efeito. A ablação normaliza a direção antes da
projeção, portanto a comparação de controle não é justificada por igualdade de
norma. A conclusão é restrita a esse padrão causal neste modelo, seed e
recorte; a mesma validação efetiva foi usada para selecionar a direção e
avaliar o efeito final.

## Escopo e desvios

Esta não é uma reprodução integral do artigo. Usamos um único modelo, 128
exemplos de treino por classe e amostramos 32 exemplos de validação por classe.
Após o filtro comportamental, o treino efetivo ficou com 121 pedidos nocivos e
127 inofensivos; na validação, foram efetivamente avaliados 28 pedidos nocivos
e 32 inofensivos. A preservação foi medida nos pedidos inofensivos dessa
validação. Não executamos todos os modelos, benchmarks de jailbreak ou
conjuntos externos de perda do trabalho original.

O Colab atual usa Python 3.13; por isso a execução remota usa Transformers 4.48.3. Em Python anterior a 3.13, o ambiente mantém Transformers 4.44.2. A lógica de extração, ablação, adição e seleção segue o commit upstream registrado nos artefatos.

## Reproduzir e validar

Os notebooks devem ser executados na ordem:

1. `01_dataset_and_baseline.ipynb`
2. `02_extract_candidate_directions.ipynb`
3. `03_causal_interventions.ipynb`
4. `04_evaluation_and_validation.ipynb`

O Gemma é um modelo com acesso restrito: aceite previamente sua licença na
[página do modelo](https://huggingface.co/google/gemma-2b-it). Prepare o
ambiente e faça login local no Hugging Face dentro do diretório do projeto;
depois, volte à raiz:

```powershell
Set-Location projects/refusal_direction
uv sync
uv run hf auth login
Set-Location ../..
```

Não cole nem exponha o token em notebook, comando, log ou documentação.

Ainda na raiz do repositório, instale a versão verificada da CLI do Colab e
faça a autenticação ADC manual:

```powershell
uv tool install google-colab-cli==0.7.5
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Na raiz do repositório, a sequência operacional do Colab é:

```powershell
python tools/colab/run_refusal.py start --gpu L4
python tools/colab/run_refusal.py sync
python tools/colab/run_refusal.py bootstrap
python tools/colab/run_refusal.py auth-hf
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/01_dataset_and_baseline.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/02_extract_candidate_directions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/03_causal_interventions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/04_evaluation_and_validation.ipynb
python tools/colab/run_refusal.py fetch
python tools/colab/run_refusal.py stop
```

Execute `auth-hf` antes do primeiro `execute` e use `fetch` para baixar os
resultados agregados. O comando `auth-hf` transfere para a VM o token já
armazenado pelo cliente Hugging Face local sem imprimi-lo; ao terminar ou se
houver erro, encerre a sessão com `stop`.

Para conferir os resultados baixados:

```powershell
Set-Location projects/refusal_direction
uv run python scripts/validate_results.py
uv run pytest -q
```

As métricas completas estão em `results/final_evaluation.json`.
