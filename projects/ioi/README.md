# Replicação do circuito IOI

Escolhi o circuito de identificação do objeto indireto como meu primeiro
estudo porque ele oferece um exemplo concreto de interpretabilidade
mecanística: um comportamento simples, um circuito publicado e várias formas
de testar causalmente os componentes propostos. Minha meta não foi apenas
executar notebooks, mas entender como se passa de uma previsão do modelo para
uma explicação em termos de cabeças de atenção e caminhos internos.

A referência é o circuito de GPT-2 Small apresentado por Wang et al. em
[*Interpretability in the
Wild*](https://arxiv.org/abs/2211.00593). Em
[`PROCESSO.md`](PROCESSO.md), conto com mais detalhes como conectei as
perguntas, intervenções, evidências e decisões de reprodução.

## Duas etapas

Dividi os notebooks em duas etapas porque precisei aprender as ferramentas
antes de tentar uma reprodução mais próxima do artigo:

1. **01–04, didática:** uso prompts compactos e introduzo progressivamente
   atribuição direta aos logits, *activation patching*, *path patching* e
   validação. Esses notebooks ensinam o fluxo; não produzem estimativas em
   escala de artigo.
2. **05–10, fiel ao artigo:** uso datasets pareados IOI/pABC determinísticos,
   o circuito publicado de 26 nós, *path patching* semântico, *Backup Name
   Movers*, *faithfulness/completeness/minimality* e os protocolos adversariais
   da Figura 8. Registrei em cada artefato os desvios em relação ao código
   histórico.

A etapa fiel usa TransformerLens **4.3.0** no modo de compatibilidade com
GPT-2 e o contrato explícito de TL4.3 `fold_ln=True`,
`center_writing_weights=True`, `center_unembed=True`,
`fold_value_biases=True`, `refactor_factored_attn_matrices=False`,
`no_processing=False` e `disable_warnings=False`. CUDA é usada quando
disponível; o carregamento comum dos notebooks usa CPU como fallback. O runner
completo exige CUDA por padrão para impedir que uma reexecução lenta em CPU
passe despercebida; `--allow-cpu` deve ser usado apenas de forma intencional.
Cada artefato da etapa registra essa configuração diretamente do runtime, em
vez de repetir um rótulo copiado.

## Ambiente

```text
uv sync
uv run python -m ipykernel install --user --name ioi-replication --display-name "Python (IOI Replication)"
uv run jupyter lab
```

Selecione `Python (IOI Replication)`. A execução registrada usou Python 3.11,
PyTorch CUDA, TransformerLens 4.3.0 e uma RTX 3050 com 6144 MiB de VRAM.

## Comandos de reprodução

Para executar todos os testes:

```text
uv run pytest -v
```

Para executar um notebook isolado, substitua `<name>`:

```text
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
```

Comandos individuais da etapa fiel:

```text
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_paper_dataset_and_baseline.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_circuit_discovery.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_path_patching.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/08_backup_name_movers.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/09_faithfulness_completeness_minimality.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/10_adversarial_examples.ipynb --ExecutePreprocessor.kernel_name=ioi-replication --ExecutePreprocessor.timeout=7200
```

A sequência completa instala o kernel do projeto, executa 05→10, interrompe no
primeiro erro, registra tempos e estados compactos em
`results/paper/execution_manifest.json` e reconstrói
`results/paper/summary.json` a partir dos artefatos canônicos:

```text
uv run python scripts/run_paper_notebooks.py
uv run python scripts/build_paper_summary.py --check
```

## Configurações e custo registrados

Os tempos abaixo vêm da execução CUDA registrada no manifesto. Eu não
instrumentei o pico de VRAM; por isso, a tabela informa a capacidade testada em
vez de inventar uma medição de pico.

| Notebook | Seeds | N alvo / referência | Lotes | Tempo da execução | VRAM testada |
|---|---|---:|---:|---:|---:|
| 05 baseline | 42 | 256 / sem ref. | 32 | 0:26 | 6 GiB |
| 06 validação diagnóstica | 106 | 64 / ABC pareado | lote completo | 13:47 | 6 GiB |
| 07 grade de caminhos publicada | 107 | 64 / ABC pareado | lote completo | 9:22 | 6 GiB |
| 08 backups | 308 / 309 | 16 / 300 (10×30 grupos) | alvo completo / ref. 16 | 0:35 | 6 GiB |
| 09 validação | 409 / 410 / 411 | 100 / 300 (10×30 grupos) | alvo 50 / ref. 16 | 13:20 | 6 GiB |
| 10 adversarial | base 510; IO 511; double IO 512; S 513 | 500 / sem ref. | 16 | 0:37 | 6 GiB |

No notebook 09, mantive todos os **10 subconjuntos aleatórios, 10 execuções
gulosas, 10 passos gulosos e 10 candidatos por passo**. A referência de 08–09
é pABC balanceada, amostrada de forma independente e agrupada por
padrão/template, não a referência histórica do mesmo lote.

## Observado versus artigo

Leio os valores abaixo como comparações, não como afirmações de igualdade.
Amostragem, construção da referência independente, compatibilidade do
TransformerLens, filtragem de prompts, alinhamento semântico e incerteza de
amostras finitas podem deslocar os resultados.

| Experimento | Artefato observado | Alvo ou resultado do artigo | Interpretação |
|---|---:|---:|---|
| Baseline IO−S | 3.644 (N=256) | 3.56 (N=100,000) | As estimativas pontuais são próximas; o SE observado de 0.109 não estabelece identidade. |
| Acurácia IO / P(IO) no baseline | 98.83% / 52.02% | 99.3% / 49% | A acurácia é semelhante; a probabilidade difere em cerca de +3 pontos. |
| Faithfulness | F(M)=3.732, F(C)=3.120, gap=0.613, retido=83.6% | gap=0.46, retido=87% | As definições são as mesmas, mas uma referência pABC independente e N=100 ampliam a incerteza (IC95 do gap 0.348–0.878). |
| Completeness gulosa | incompletude máxima registrada de 4.122 em 10 execuções gulosas completas | o artigo fornece o procedimento/figura, não um único escalar canônico | A busca é sensível à seed e à referência; é preciso comparar curvas e conjuntos removidos, não apenas o máximo. |
| Minimality | 26/26 estimativas pontuais >1%; 24/26 limites inferiores do IC95 >1% | 26 nós avaliados contra o limiar de 1% | A reprodução pontual é mais forte que a sustentada por intervalo; dois nós permanecem incertos. |
| Baseline adversarial | IO−S 3.271, P(IO) 48.59%, S>IO 0.6% | 3.55, 49%, 0.7% | Probabilidade e erro se alinham melhor que o IO−S absoluto. |
| IO single adversarial | IO−S 0.946, P(IO) 32.24%, S>IO 31.0% | 1.23, 36%, 23.4% | Os valores absolutos divergem, mas o delta IO−S observado de −2.324 fica próximo do alvo −2.32. |
| Controle S adversarial | IO−S 3.411, P(IO) 56.17%, S>IO 0.2% | 3.64, 59%, 0.4% | A direção concorda; os níveis absolutos e a pequena estimativa de erro diferem. |

O notebook 06 valida diagnosticamente a lista publicada de 26 nós em N=64;
ele não faz descoberta cega do circuito. O notebook 07 avalia os 88 pares
concretos sender/receiver induzidos pelas seis arestas de classe publicadas em
N=64; os exemplos selecionados também preservam os dois modos históricos de
congelamento de MLP e a comparação moderna em todas as posições. Sinais ou
rankings inesperados nesses diagnósticos de amostra finita não invalidam, por
si sós, o circuito publicado.

O notebook 08 usa 16 prompts-alvo; seu denominador de compensação corrigido é
a DLA basal das primárias menos a DLA da substituição pABC, ambas na escala
basal. O notebook 10 retokeniza depois da inserção sem BOS para evitar a
inconsistência histórica de coordenadas de `advex.py` e mantém separados os
protocolos de uma cláusula (`single`) e duas cláusulas (`double`), com seeds
independentes para base e ataques.

A proveniência, as configurações e os principais resultados consolidados
ficam em `results/paper/summary.json`. Os testes derivam e comparam esses
valores com os artefatos-fonte para evitar que números copiados manualmente se
afastem da evidência.
