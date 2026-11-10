# Refusal Direction Replication

Replicação parcial de Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, usando `google/gemma-2b-it`.

O experimento extrai direções pela diferença entre as médias das ativações de pedidos nocivos e inofensivos, testa cada candidata por ablação e adição e compara a direção escolhida com um controle aleatório ortogonal. Os prompts e as gerações completas permanecem em `artifacts/`, fora do Git; apenas métricas agregadas e proveniência são versionadas.

## Resultado observado

Execução de 10 de novembro de 2026 em uma NVIDIA L4:

- direção selecionada: camada 12, último token da instrução;
- recusa em pedidos nocivos: 100% na linha de base e 0% após ablação;
- recusa indevida em pedidos inofensivos: 0% na linha de base e 93,75% após adição;
- controle aleatório: 100% de recusa nos pedidos nocivos;
- KL em pedidos inofensivos após ablação: 0,0306, abaixo do limite de 0,1;
- variação média da entropia cruzada após ablação: -0,4664, isto é, não houve aumento nessa amostra.

Esses números reproduzem o padrão causal central: remover a direção elimina a recusa, adicioná-la induz recusa e uma direção aleatória de mesma norma não produz o mesmo efeito.

## Escopo e desvios

Esta não é uma reprodução integral do artigo. Usamos um único modelo, 128 exemplos de treino e 32 de validação por classe, além de medir preservação nos pedidos inofensivos da própria validação. Não executamos todos os modelos, benchmarks de jailbreak ou conjuntos externos de perda do trabalho original.

O Colab atual usa Python 3.13; por isso a execução remota usa Transformers 4.48.3. Em Python anterior a 3.13, o ambiente mantém Transformers 4.44.2. A lógica de extração, ablação, adição e seleção segue o commit upstream registrado nos artefatos.

## Reproduzir e validar

Os notebooks devem ser executados na ordem:

1. `01_dataset_and_baseline.ipynb`
2. `02_extract_candidate_directions.ipynb`
3. `03_causal_interventions.ipynb`
4. `04_evaluation_and_validation.ipynb`

Para conferir os resultados baixados:

```powershell
uv run python scripts/validate_results.py
uv run pytest -q
```

As métricas completas estão em `results/final_evaluation.json`.
