# Acompanhamento dos experimentos

Este é meu ponto de entrada para acompanhar, em ordem cronológica, os estudos
de **J-space** e **Direção de Recusa**. Eu combino aqui o relato humano das
perguntas, decisões e incidentes com fatos derivados dos artefatos
versionados. Para auditar um número, sigo o link até o JSON; para entender por
que uma decisão foi tomada, consulto o `PROCESSO.md` correspondente.

## Como usar este painel

Leio cada linha do tempo como um diário científico compacto: a pergunta vem
antes da ação, e o resultado vem antes da interpretação. Os JSONs canônicos
são as fontes dos números e estados; os arquivos `PROCESSO.md` preservam a
narrativa detalhada, inclusive resultados negativos, correções e limites. Os
READMEs resumem a reprodução, e os notebooks são os pontos de entrada
executáveis.

<!-- BEGIN GENERATED STATUS -->
### Retrato automático dos resultados

**J-space causal.** Modelo `meta-llama/Llama-3.1-8B-Instruct`; execução `20eabeef-17f5-49dc-a4a1-fcafc20a9533`. Estado operacional: `completed`. Conclusão científica: **inconclusiva** (`primary_gates_failed`). [Manifesto](../projects/jspace_llama/results/execution_manifest.json) · [avaliação](../projects/jspace_llama/results/final_evaluation.json).

**J-space atlas.** Estado operacional: `success` na execução `2fcb8a49-136f-4868-91cc-f98f9dc2e57d`, com 93 itens. Conclusão científica: retrato **observacional**, não evidência causal. Na faixa de camadas, pass@1/5/10: J-lens `0,344086 / 0,548387 / 0,596774`; logit lens `0,284946 / 0,392473 / 0,435484`. [Manifesto](../projects/jspace_llama/results/jspace_observation_manifest.json) · [resultados](../projects/jspace_llama/results/jspace_observation.json).

**Direção de Recusa.** Modelo `google/gemma-2b-it`; 4 estágios completos (baseline, extração, intervenção, avaliação). Conclusão científica: evidência **causal local**. Direção selecionada na camada 12, posição -1. Recusa nociva: baseline 100%, ablação 0%, controle aleatório 100%; adição em pedidos inofensivos 93,75%; KL inofensiva após ablação `0,030571`. [Manifesto](../projects/refusal_direction/results/execution_manifest.json) · [intervenções](../projects/refusal_direction/results/03_interventions.json) · [avaliação](../projects/refusal_direction/results/final_evaluation.json).
<!-- END GENERATED STATUS -->

## Duas noções de sucesso

Eu separo explicitamente **sucesso operacional** de **conclusão científica**.
Uma execução pode terminar, produzir artefatos íntegros e passar pela
validação sem confirmar a hipótese. Inversamente, um manifesto isolado não
prova que toda a cadeia experimental foi concluída: o manifesto precisa ser lido
com os demais artefatos. Neste painel, `success` ou “concluído” descreve primeiro o
pipeline; conclusões causais ou observacionais aparecem separadamente e com o
escopo que os dados permitem.

## Estudo 1: J-space no Llama 3.1 8B Instruct

### Estado atual

Mantenho duas evidências separadas. A tentativa causal canônica terminou
operacionalmente, mas seu estado científico é `inconclusive`, com motivo
`primary_gates_failed`. O atlas observacional, por sua vez, terminou com
`status=success` no schema v2 e cobre 93 itens; esse sucesso é operacional e
não transforma a observação em evidência causal. A narrativa completa está no
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md), e a visão operacional
resumida está no [`README.md`](../projects/jspace_llama/README.md).

### Cronologia retrospectiva

**1. Formulação causal.** Perguntei se uma representação intermediária de
`spider` poderia ser lida e trocada por `ant`, fazendo a resposta mudar de
oito para seis pernas sem degradar a geração. Planejei baseline, no-op,
controle aleatório ortogonal e intervenção-alvo, com critérios que exigiam
todas as precondições em conjunto. Essa formulação está descrita no
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md) e implementada no
[`notebook causal`](../projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb).

**2. Primeira tentativa bloqueada por autorização.** A execução
`a116c3da-bb8f-4a31-a39a-4fd09dd6fcad` parou antes da inferência porque minha
conta ainda não tinha acesso ao modelo *gated*. Tratei isso como falha de
infraestrutura, não como resultado contra a hipótese; após a aprovação,
repeti o protocolo. O incidente e a exigência de autenticação estão
registrados no
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md) e no
[`README.md`](../projects/jspace_llama/README.md).

**3. Execução causal canônica: completa, porém inconclusiva.** A execução
`20eabeef-17f5-49dc-a4a1-fcafc20a9533` carregou modelo e lente, verificou a
integridade e executou as quatro condições em três prompts. Operacionalmente,
o terminal ficou `completed`; cientificamente, o status foi `inconclusive`,
causa `primary_gates_failed`. O baseline e o no-op não produziram a resposta
única esperada, `spider` não apareceu no readout, e a intervenção não gerou
seis de modo válido, além de vazar `ant` e apresentar repetição. Decidi não
chamar isso de refutação do J-space, porque as precondições do teste não foram
estabelecidas. Evidências:
[`execution_manifest.json`](../projects/jspace_llama/results/execution_manifest.json)
e
[`final_evaluation.json`](../projects/jspace_llama/results/final_evaluation.json).

**4. Separação entre observação e causalidade.** Diante do resultado
inconclusivo, separei “o que a lente expõe?” de “o que causa a resposta?”. A
primeira pergunta passou a ser um atlas sem geração, edição, *clamp* ou troca
de conceito; a segunda ficou reservada para um novo experimento
pré-registrado. Essa decisão metodológica, que impede selecionar
retrospectivamente uma intervenção favorável, está no
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md).

**5. Construção do atlas de 93 prompts.** Fixei os 93 itens de
`lens-eval-multihop` e comparei J-lens direta, logit lens e um painel
qualitativo de `gradient_pursuit`. O `target` permaneceu apenas como metadado:
o modelo recebeu o prompt bruto, e as leituras observaram ativações já
produzidas. O desenho e a entrada executável estão no
[`README.md`](../projects/jspace_llama/README.md), no
[`notebook observacional`](../projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb)
e na
[`documentação do conjunto`](../projects/jspace_llama/data/anthropic-jlens/README.md).

**6. Incidente de memória e validação; correção.** A primeira validação
numérica de `gradient_pursuit` criou em GPU uma máscara `isfinite` do tamanho
inteiro do dicionário; a L4 encerrou a tentativa com
`resource_exhausted`, sem publicar artefatos parciais. Substituí a checagem por
blocos de até um milhão de elementos e repeti em sessão limpa. Também corrigi
uma validação que confundia o token legítimo `/x` com um caminho, tornando-a
sensível ao campo e ao schema. Preservei ambos como incidentes operacionais,
não como achados científicos, no
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md).

**7. Execução observacional canônica, schema v2.** A execução
`2fcb8a49-136f-4868-91cc-f98f9dc2e57d` processou os 93 itens e terminou com
`status=success`. Na *band* 12–28, a J-lens obteve
`pass@1/5/10 = 0.344086 / 0.548387 / 0.596774`; a logit lens obteve
`0.284946 / 0.392473 / 0.435484`. O schema v2 também registra IDs e hashes da
tokenização dos prompts. Esses números vêm de
[`jspace_observation.json`](../projects/jspace_llama/results/jspace_observation.json);
run, schema, contagem e proveniência vêm de
[`jspace_observation_manifest.json`](../projects/jspace_llama/results/jspace_observation_manifest.json).

**8. Interpretação observacional cuidadosa.** Neste recorte, a J-lens
recuperou mais intermediários que a logit lens, e o painel mostrou acertos e
falhas (por exemplo, `spider` chegou ao rank 2 na camada 19). Interpreto esse
recorte como um mapa útil da leitura interna, condicionado ao tokenizer, à lente, às
camadas e ao conjunto. Não houve intervenção nem geração; portanto, o atlas
**não é causal** e não reclassifica a execução causal inconclusiva. Para
valores exatos, uso o
[`JSON observacional`](../projects/jspace_llama/results/jspace_observation.json);
para contexto e cautelas, uso o
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md).

### Próximo passo

Meu próximo passo é pré-registrar uma nova execução causal: hipótese, prompts
ainda não usados no atlas, regras de parsing, controles, camada e intervenção
devem ser definidos antes de observar o resultado. Não vou escolher
retrospectivamente conceito ou `alpha` a partir do painel. Esse compromisso
está detalhado em
[`PROCESSO.md`](../projects/jspace_llama/PROCESSO.md#próximo-passo-científico).

## Estudo 2: Direção de Recusa no Gemma 2B IT

### Estado atual

Os quatro estágios (baseline, extração, intervenção e avaliação) estão
declarados como completos, e os quatro JSONs esperados estão presentes. No
`google/gemma-2b-it`, os resultados reproduzem o padrão causal de necessidade
e suficiência **neste recorte**. Não generalizo para outros modelos, amostras
ou definições de
recusa, porque houve filtro comportamental, uma única seed e reuso da mesma
validação para seleção e avaliação. A narrativa detalhada está no
[`PROCESSO.md`](../projects/refusal_direction/PROCESSO.md), e o resumo
reprodutível está no
[`README.md`](../projects/refusal_direction/README.md).

### Cronologia retrospectiva

**1. Pergunta local.** Perguntei se uma direção extraída das ativações do
Gemma 2B IT seria necessária e suficiente para controlar a recusa em um
recorte delimitado. Defini necessidade por ablação, suficiência por adição e
especificidade por comparação com uma direção aleatória ortogonal. Não
pretendi identificar “todo o alinhamento” ou toda a nocividade. A formulação
está no
[`PROCESSO.md`](../projects/refusal_direction/PROCESSO.md).

**2. Baseline e filtro comportamental.** No
[`notebook 01`](../projects/refusal_direction/notebooks/01_dataset_and_baseline.ipynb),
amostrei 128 pedidos por classe no treino e 32 por classe na validação, depois
mantive apenas exemplos cujo sinal basal concordava com a classe. Restaram
121/127 exemplos de treino e 28/32 de validação, nocivos/inofensivos. O
baseline nocivo foi 100% e o inofensivo 0%, mas esses valores são
condicionais ao filtro, não taxas da amostra original. Fonte numérica:
[`01_baseline.json`](../projects/refusal_direction/results/01_baseline.json).

**3. Extração de 90 candidatas.** No
[`notebook 02`](../projects/refusal_direction/notebooks/02_extract_candidate_directions.ipynb),
calculei a diferença entre médias de ativações nocivas e inofensivas nas 18
camadas e cinco posições finais do template, produzindo 90 direções. As cinco
candidatas da camada 0 eram nulas e passaram a ser tratadas como no-op, sem
interpretação causal. Contagens, formas e normas estão em
[`02_candidates.json`](../projects/refusal_direction/results/02_candidates.json).

**4. Seleção da camada 12, posição `-1`.** No
[`notebook 03`](../projects/refusal_direction/notebooks/03_causal_interventions.ipynb),
selecionei entre candidatas que induziam recusa por adição, mantinham KL
inofensiva no máximo 0,1 e não pertenciam às camadas finais excluídas. A
escolhida foi a camada 12, posição `-1` (o último token do sufixo do template,
não o último token da instrução), com norma `17.1552`. A decisão e os escores
estão em
[`03_interventions.json`](../projects/refusal_direction/results/03_interventions.json).

**5. Ablação, adição e controle.** A ablação projetou a direção para fora ao
longo das camadas; a adição inseriu a direção na camada selecionada; o
controle usou uma orientação aleatória ortogonal. Nos 28 pedidos nocivos
pós-filtro, a recusa caiu de 100% no baseline para 0% após ablação. Nos 32
inofensivos, a adição produziu 93.75% de recusa. O controle aleatório manteve
100% de recusa nociva e não reproduziu o efeito. Evidências:
[`03_interventions.json`](../projects/refusal_direction/results/03_interventions.json)
e
[`notebook 03`](../projects/refusal_direction/notebooks/03_causal_interventions.ipynb).

**6. Avaliação e cautelas.** No
[`notebook 04`](../projects/refusal_direction/notebooks/04_evaluation_and_validation.ipynb),
consolidei as taxas e a preservação. A KL inofensiva após ablação foi
`0.03057`, abaixo de 0,1. Já as variações de entropia cruzada da ablação
(`-0.4664`) e do controle (`-0.4652`) ficaram muito próximas, por isso essa
métrica isolada oferece pouca evidência de especificidade; além disso, não há
KL do controle. Minha conclusão é causal, mas local ao modelo, ao filtro, à
amostra e à seed. Fonte:
[`final_evaluation.json`](../projects/refusal_direction/results/final_evaluation.json).

**7. Incidentes operacionais.** Precisei compatibilizar Python 3.13 com
Transformers, separar Jupyter nas dependências de desenvolvimento, autenticar
o repositório *gated*, tratar vetores nulos sem gerar valores inválidos,
recarregar módulos em um kernel persistente e retomar conexões longas sem
confundir queda do cliente com perda da sessão. Registro esses fatos porque
eles afetam a reprodutibilidade, mas não os apresento como resultados
científicos. A descrição completa está no
[`PROCESSO.md`](../projects/refusal_direction/PROCESSO.md#incidentes-reais-e-o-que-ensinaram).

**8. Fechamento operacional.** O manifesto lista os quatro estágios e os
quatro arquivos de resultado. Eu o leio junto dos JSONs, pois a lista é
emitida pelo estágio final e, sozinha, não comprova o sucesso de cada
notebook; tampouco registra o encerramento da GPU. Evidência:
[`execution_manifest.json`](../projects/refusal_direction/results/execution_manifest.json).

### Próximo passo

Eu repetiria o experimento com várias seeds, reservaria um conjunto de teste
final não usado na seleção e mediria a sensibilidade ao filtro, à posição, ao
coeficiente de adição e à definição de recusa. Só depois ampliaria a conclusão
para outros modelos ou distribuições. Esses limites e extensões estão no
[`PROCESSO.md`](../projects/refusal_direction/PROCESSO.md#limites-e-próximos-passos).

## Modelo leve para futuras entradas

### AAAA-MM-DD: título curto

- **Pergunta:** o que eu queria descobrir?
- **O que fiz:** qual execução, análise ou correção realizei?
- **Resultado:** o que os artefatos mostram, inclusive se foi negativo ou
  inconclusivo?
- **Decisão ou limitação:** o que muda (ou não muda) a partir disso?
- **Evidências:** links relativos para JSONs, `PROCESSO.md`, notebook ou figura.
