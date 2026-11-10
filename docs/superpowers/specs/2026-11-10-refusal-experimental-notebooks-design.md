# Notebooks experimentais para Refusal Direction

## Objetivo

Adicionar uma camada didática e interativa à replicação de *Refusal Direction*.
Os novos notebooks devem permitir acompanhar o mecanismo em execução: observar
tokens, logits, formas dos tensores, ativações residuais, direções candidatas,
intervenções e métricas por exemplo.

Os quatro notebooks existentes continuam sendo a execução canônica e enxuta do
pipeline. Os novos arquivos serão companheiros laboratoriais em
`projects/refusal_direction/notebooks/experimental/`, sem substituir nem
duplicar a responsabilidade dos notebooks canônicos.

## Princípios

- O código científico permanece em `src/refusal_direction_replication/`.
- O notebook abre as operações importantes em células pequenas, em vez de
  esconder o experimento inteiro atrás de uma chamada `run_*_stage`.
- Cada valor exibido deve ter uma pergunta científica associada.
- Prompts nocivos, tokens de autenticação e gerações completas não aparecem em
  células, saídas ou arquivos versionados.
- A execução rápida usa uma amostra pequena e real; resultados completos podem
  ser carregados dos artefatos ou recalculados no Colab.
- Os notebooks devem continuar úteis quando não houver GPU, desde que os
  artefatos versionados estejam disponíveis.

## Estrutura

Serão criados quatro notebooks:

1. `01_baseline_lab.ipynb`
2. `02_candidate_directions_lab.ipynb`
3. `03_causal_interventions_lab.ipynb`
4. `04_evaluation_lab.ipynb`

Cada notebook seguirá o mesmo ritmo:

1. pergunta experimental;
2. configuração e escolha do modo de execução;
3. amostra concreta e segura;
4. código real dividido em etapas;
5. inspeção dos valores intermediários;
6. visualização;
7. checagens explícitas;
8. interpretação e limites;
9. perguntas para o leitor.

## Modos de execução

Uma célula inicial define `EXECUTION_MODE` com três valores:

- `"quick"`: executa o modelo em uma amostra pequena;
- `"artifacts"`: carrega resultados já produzidos e refaz apenas a análise;
- `"full"`: executa a fase completa, destinada ao Colab com GPU.

O padrão será `"artifacts"` quando os arquivos necessários existirem e
`"quick"` nos demais casos. O modo escolhido e a origem dos dados devem ficar
visíveis no notebook.

Se um artefato exigido estiver ausente, o notebook deve explicar qual fase o
produz e como executá-la. Falhas de CUDA, licença do Gemma ou autenticação do
Hugging Face devem apresentar uma mensagem operacional curta, sem imprimir
credenciais.

## Notebook 01 — Baseline

O primeiro laboratório acompanha a passagem de uma instrução segura pelo
modelo até o score local de recusa:

- aplicação do chat template;
- tokenização e identificação da posição final;
- forma do tensor de logits;
- seleção dos tokens marcadores de recusa;
- cálculo do log-odds por exemplo;
- comparação entre classes nociva e inofensiva por distribuição, não apenas
  pela média;
- inspeção de flags e casos próximos da fronteira zero.

Os textos nocivos reais serão representados apenas por identificadores, hashes
e rótulos. Um exemplo inofensivo poderá ser mostrado integralmente.

## Notebook 02 — Direções candidatas

O segundo laboratório mostra como a direção é construída:

- registro de um hook em uma camada;
- captura do residual em uma amostra pequena;
- leitura dos eixos
  `exemplo × camada × posição × dimensão residual`;
- cálculo explícito de média nociva menos média inofensiva;
- norma e similaridade entre algumas candidatas;
- mapa camada–posição das normas;
- comparação entre magnitude estatística e evidência causal.

O notebook produzirá ou carregará as 90 candidatas da configuração verificada,
mas deixará claro que norma alta não seleciona sozinha uma direção.

## Notebook 03 — Intervenções causais

O terceiro laboratório abre a mecânica causal:

- projeção de um vetor residual sobre a direção;
- verificação numérica da ortogonalidade após ablação;
- adição e ablação por hooks;
- comparação do score antes e depois da intervenção;
- critérios de elegibilidade por ablação, adição, KL e camada;
- ranking das candidatas;
- controle aleatório ortogonal com a mesma norma;
- visualizações que relacionem efeito causal e custo de preservação.

As fórmulas serão acompanhadas por código curto que reproduz a operação em um
tensor pequeno antes de aplicá-la ao Gemma.

## Notebook 04 — Avaliação

O quarto laboratório reúne:

- baseline;
- ablação em pedidos nocivos;
- ablação em pedidos inofensivos;
- adição em pedidos inofensivos;
- controle aleatório.

Serão visualizadas taxa de recusa, falsa recusa, log-odds, divergência KL,
entropia cruzada e deltas por exemplo. As três condições causais e a condição
de preservação aparecerão como checagens booleanas acompanhadas pelos valores
que sustentam cada conclusão.

O fechamento distinguirá resultado observado, interpretação causal local e
afirmações que o experimento não permite fazer.

## Fluxo de dados

Os notebooks usam as funções menores já existentes em `model.py`,
`directions.py`, `interventions.py`, `metrics.py` e `pipeline.py`. Chamadas
`run_*_stage` serão reservadas ao modo `"full"` ou à produção dos artefatos
canônicos.

Os resultados versionáveis continuam em `results/`. Pesos, datasets
normalizados, tensores de direção, caches e gerações permanecem em
`artifacts/`, fora do Git. Os laboratórios não criam um segundo formato de
resultado científico.

## Visualizações

As visualizações usarão Matplotlib e Pandas, já presentes nas dependências de
desenvolvimento. O conjunto mínimo inclui:

- distribuição de log-odds por classe;
- mapa camada–posição das candidatas;
- ranking causal com KL e efeito de ablação;
- comparação das condições experimentais;
- distribuição dos deltas individuais de entropia cruzada.

Cada gráfico terá título, eixos, unidade ou definição do score e uma célula
curta explicando como interpretá-lo.

## Testes

O desenvolvimento será guiado por testes estruturais escritos antes dos
notebooks. Eles verificarão:

- existência e ordem dos quatro arquivos;
- JSON válido e células executáveis;
- presença dos três modos de execução;
- uso das funções científicas reais;
- presença das inspeções intermediárias e visualizações esperadas;
- ausência de tokens, prompts nocivos e gerações completas;
- saídas limpas antes de versionar;
- manutenção dos notebooks canônicos sem alterações comportamentais.

A suíte não executará o Gemma nem exigirá GPU. Fórmulas e funções reutilizadas
continuam cobertas pelos testes unitários do pacote. A validação final inclui a
suíte `pytest`, o validador de resultados e abertura dos notebooks pelo parser
do Jupyter.

## Critérios de conclusão

O trabalho estará concluído quando:

- os quatro laboratórios puderem ser abertos em ordem e funcionarem como uma
  narrativa única;
- o leitor conseguir localizar de onde vem cada score e cada tensor;
- o modo por artefatos permitir estudar os resultados sem GPU;
- os modos rápido e completo apontarem para o mesmo código científico;
- nenhum dado sensível ou conteúdo nocivo for versionado;
- documentação, testes e validações passarem.
