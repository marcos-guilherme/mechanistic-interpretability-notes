# Termo de entrega — circuito IOI no GPT-2 Small

> **Data de referência:** 11 de novembro de 2026
> **Entrega principal:** reprodução parcial do circuito de identificação do
> objeto indireto no GPT-2 Small
> **Atividades iniciadas:** direção de recusa no Gemma 2B IT e representações
> conceituais em J-space no Llama 3.1 8B Instruct

## Objetivo deste documento

Este documento registra o caminho que percorri, as fontes que orientaram as
escolhas, as técnicas que aprendi, os resultados que obtive e os limites que
encontrei. Ele também documenta por que iniciei duas investigações
complementares depois do estudo de circuitos no GPT-2 Small.

O foco desta entrega é a reprodução de
[*Interpretability in the Wild: a Circuit for Indirect Object Identification
in GPT-2 Small*](https://arxiv.org/abs/2211.00593), de Wang et al. (2022). Os
estudos sobre direção de recusa e J-space aparecem como desdobramentos em
andamento, e não como resultados principais desta semana.

## Contexto e percurso até a entrega

Nas semanas anteriores, direcionei minha pesquisa para interpretabilidade
mecanicista e, mais especificamente, para o estudo de circuitos internos de
modelos de linguagem. Eu buscava um caso em que fosse possível acompanhar o
processo completo: partir de um comportamento observável, localizar
componentes internos, intervir nesses componentes e avaliar se eles realmente
causam o comportamento analisado.

Algumas leituras tiveram papel importante nesse direcionamento. Em
[*The Case for Analysing Toy Language Models*](https://www.alignmentforum.org/posts/GWCgZrzWCZCuzGktv/200-cop-in-mi-the-case-for-analysing-toy-language-models),
Nanda (2022) argumenta que modelos menores oferecem ambientes mais tratáveis
para investigar mecanismos internos. A proposta não é assumir que tudo o que
é observado nesses modelos será transferido para modelos maiores, mas usar
casos controlados para desenvolver perguntas, técnicas e critérios de
validação.

O trabalho
[*A Mathematical Framework for Transformer
Circuits*](https://transformer-circuits.pub/2021/framework/index.html), de
Elhage et al. (2021), forneceu parte do vocabulário para pensar transformers
como composições de caminhos no fluxo residual. O
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
transformou essas ideias em operações executáveis, permitindo capturar
ativações, decompor contribuições e aplicar intervenções por meio de hooks.

Também estudei
[*Attribution Patching Outperforms Automated Circuit
Discovery*](https://arxiv.org/abs/2310.10348), de Syed, Rager e Conmy (2023).
O artigo apresenta o *Edge Attribution Patching* como uma forma
computacionalmente mais barata de priorizar arestas importantes. Essa leitura
reforçou uma distinção metodológica que mantive durante a reprodução: métodos
de atribuição podem ajudar a localizar candidatos, mas a confirmação de uma
explicação exige intervenções causais.

Esse percurso me levou à tarefa de identificação do objeto indireto, ou IOI,
no GPT-2 Small. O caso oferece uma tarefa linguística delimitada, um circuito
publicado de 26 cabeças de atenção e diferentes procedimentos para testar
componentes e caminhos internos.

## Entrega principal: reprodução do circuito IOI

### Pergunta da reprodução

A tarefa IOI avalia se o modelo consegue completar frases como “Maria e João
foram à loja; João deu um presente para...”, escolhendo o nome correspondente
ao objeto indireto. O artigo de Wang et al. propõe que esse comportamento é
produzido por um circuito de 26 cabeças de atenção, organizadas em sete
classes funcionais.

Minha pergunta foi se esse circuito manteria poder explicativo e causal quando
reconstruído localmente, com dados, posições semânticas, intervenções e
métricas explicitamente controlados. O objetivo não foi descobrir um novo
circuito do zero, mas aprender como uma explicação mecanística publicada pode
ser examinada, reproduzida e limitada pelas evidências.

### Organização da reprodução

Dividi o trabalho em duas etapas. A primeira foi didática e utilizou um
conjunto pequeno de prompts para introduzir gradualmente as métricas e as
intervenções. A segunda buscou maior proximidade com o protocolo do artigo,
usando pares de prompts IOI e prompts de referência, posições semânticas e o
circuito publicado.

Essa separação foi importante porque os primeiros notebooks serviram para
aprender a formular e executar as perguntas, enquanto os posteriores
produziram os resultados usados nesta entrega. Os detalhes de cada etapa estão
no [README do projeto](../../projects/ioi/README.md) e no
[registro do processo](../../projects/ioi/PROCESSO.md).

### Técnicas aprendidas

**Atribuição direta aos logits.** Usei a contribuição direta de componentes
para o logit final a fim de observar quais cabeças favoreciam o objeto
indireto ou o sujeito. Essa técnica é útil para formar hipóteses, mas uma
contribuição elevada não demonstra, sozinha, que o componente seja
causalmente necessário.

**Activation patching.** Substituí ativações de uma execução corrompida por
ativações correspondentes de uma execução limpa. Quando essa substituição
recupera parte do comportamento, ela fornece evidência de que o componente e
a posição analisados carregam informação relevante para a tarefa.

**Path patching.** Em vez de testar apenas componentes isolados, investiguei
como a informação percorre caminhos entre componentes emissores e receptores.
Essa etapa aproxima a análise de uma explicação em forma de circuito, pois
testa relações causais internas, e não somente a importância individual de
uma cabeça.

**Ablação e validação do circuito.** Avaliei o comportamento do modelo quando
componentes eram substituídos por valores de referência. Também examinei
*faithfulness*, *completeness* e *minimality*: quanto do comportamento o
circuito preserva, como modelo e circuito respondem à remoção de componentes
e se cada nó possui contribuição necessária.

**Controles e exemplos adversariais.** Testei alterações nos prompts que
produzem previsões específicas sobre o comportamento do circuito. Essa etapa
ajuda a verificar se a explicação continua útil fora dos exemplos usados para
apresentá-la.

## Resultados obtidos

Na linha de base, sem intervenções, o GPT-2 Small escolheu o objeto indireto
correto em **98,83%** dos 256 exemplos avaliados. A margem média entre a
pontuação do objeto indireto e a do sujeito foi **3,644**. Essa margem, chamada
de diferença de logits IO−S, indica quanto o modelo favoreceu a resposta
correta em relação à principal alternativa considerada na tarefa.

Quando a análise foi restringida ao circuito publicado, suas 26 cabeças
preservaram **83,6%** da margem observada no modelo. Esse percentual não é uma
nova medida de acurácia: ele representa quanto do efeito IO−S foi mantido pelo
circuito após as ablações usadas no teste de *faithfulness*.

Na avaliação de *minimality*, as 26 cabeças superaram pontualmente o limiar de
1% de contribuição, mas somente 24 mantiveram o limite inferior do intervalo
de confiança acima desse valor. Esse resultado mostra por que uma estimativa
pontual e uma conclusão sustentada por incerteza não devem ser tratadas como
equivalentes.

No protocolo adversarial que repete o objeto indireto, a variação observada da
diferença IO−S foi **−2,324**, próxima de **−2,32** reportado no artigo. Os
valores absolutos não foram idênticos, mas a direção e o tamanho da mudança
principal foram recuperados.

Por esses motivos, classifico o estudo como uma **reprodução parcial dos
resultados centrais**, e não como reprodução exata. O
[resumo canônico](../../projects/ioi/results/paper/summary.json) reúne os
valores consolidados, e o
[manifesto de execução](../../projects/ioi/results/paper/execution_manifest.json)
registra ambiente, sequência dos notebooks e estado da execução.

## O que os resultados permitem concluir

A reprodução oferece evidência de que o circuito publicado preserva parte
substancial do comportamento de IOI no recorte local. As técnicas de patching,
ablação e validação também recuperaram aspectos importantes da organização
causal proposta pelo trabalho original.

Entretanto, os resultados não demonstram que o circuito seja uma explicação
completa para linguagem em geral, nem que o mesmo conjunto de componentes
apareça em outros modelos. O notebook de diagnóstico valida a lista publicada
de cabeças; ele não realiza descoberta cega. Além disso, os experimentos usam
um modelo pequeno, templates estruturados e uma tarefa sintética com
alternativas bem definidas.

O principal valor desta etapa foi metodológico. O ambiente controlado tornou
visíveis distinções que seriam mais difíceis de acompanhar em um modelo maior:

1. observar uma ativação não é o mesmo que explicar sua função;
2. atribuir uma saída a um componente não demonstra necessidade causal;
3. localizar componentes não basta para estabelecer as conexões entre eles;
4. executar uma intervenção não substitui controles e critérios de validação;
5. obter números próximos aos do artigo não elimina diferenças de amostra,
   implementação e incerteza.

## Investigações iniciadas a partir dessas limitações

As limitações do caso IOI motivaram a busca por métodos complementares. Isso
não significa que os estudos seguintes dependam do mesmo circuito. Cada um
formula uma hipótese diferente sobre como comportamentos ou conceitos podem
estar organizados internamente.

### Direção de recusa

Iniciei a leitura e a reprodução de
[*Refusal in Language Models Is Mediated by a Single
Direction*](https://arxiv.org/abs/2406.11717), de Arditi et al. (2024). O
artigo investiga se o comportamento de recusa é mediado por uma direção
aproximadamente unidimensional no espaço residual: removê-la deve reduzir a
recusa, enquanto adicioná-la pode induzir recusa em pedidos inofensivos.

A adaptação utiliza o `google/gemma-2b-it` e considera extração de direções por
diferença de médias, ablação, adição, controle aleatório e medidas de
preservação. Nesta entrega, essa linha representa a passagem para uma hipótese
sobre um comportamento menos restrito do que a escolha entre dois nomes em
prompts IOI.

O projeto e suas fontes específicas estão documentados no
[README](../../projects/refusal_direction/README.md) e no
[registro do processo](../../projects/refusal_direction/PROCESSO.md).

### Representações conceituais em J-space

Também iniciei o estudo de
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://arxiv.org/abs/2607.15495), de Gurnee et al. (2026), e da
[apresentação da Anthropic](https://www.anthropic.com/research/global-workspace).
O trabalho propõe a Jacobian lens como uma forma de relacionar estados internos
a conceitos verbalizáveis e investiga se alguns desses conceitos são usados
causalmente durante o raciocínio.

A adaptação usa o `meta-llama/Llama-3.1-8B-Instruct` e uma J-lens
pré-ajustada. O objetivo inicial é observar quais conceitos a lente recupera
e, somente depois de estabelecer uma linha de base adequada, avaliar
intervenções sobre representações intermediárias.

O projeto está documentado no
[README](../../projects/jspace_llama/README.md) e no
[registro do processo](../../projects/jspace_llama/PROCESSO.md).

## Síntese metodológica

A partir desta entrega, organizei meu entendimento do processo de reprodução
em um fluxo:

1. definir um comportamento e uma pergunta científica;
2. estabelecer uma linha de base válida;
3. medir e localizar componentes ou representações candidatas;
4. realizar intervenções causais;
5. comparar a intervenção com controles informativos;
6. avaliar preservação, especificidade e incerteza;
7. registrar ambiente, amostras, desvios e artefatos;
8. limitar a conclusão ao que foi efetivamente medido.

Esse fluxo conecta as três linhas sem tratá-las como equivalentes. IOI estuda
um circuito de componentes em uma tarefa controlada; Refusal Direction testa
uma direção associada a um comportamento; J-space investiga representações
conceituais intermediárias. A conexão entre elas é metodológica: em todos os
casos, uma hipótese sobre o interior do modelo precisa ser ligada a uma
mudança observável por meio de intervenções e controles.

## Planejamento

Para a próxima semana, pretendo continuar as investigações iniciadas a partir
das limitações observadas em IOI. No Gemma 2B IT, avançarei no estudo da
direção de recusa por meio de extração, ablação, adição e controles. No Llama
3.1 8B Instruct, continuarei o estudo de J-space, priorizando a observação dos
conceitos recuperados pela Jacobian lens antes de novas intervenções causais.

Também pretendo manter a mesma disciplina de documentação aplicada à
reprodução IOI: separar os resultados locais das afirmações dos artigos,
registrar amostras efetivas e proveniência, explicitar adaptações e classificar
execuções malsucedidas ou incompletas como inconclusivas, em vez de
transformá-las em evidência negativa.

## Artefatos relacionados

- [Metodologia comum de replicação](../replicacoes/METODOLOGIA.md)
- [Índice das três replicações](../replicacoes/README.md)
- [Fontes e referências comentadas](../REFERENCIAS.md)
- [Processo da reprodução IOI](../../projects/ioi/PROCESSO.md)
- [Resumo canônico dos resultados de IOI](../../projects/ioi/results/paper/summary.json)
- [Manifesto da execução de IOI](../../projects/ioi/results/paper/execution_manifest.json)
- [Processo da direção de recusa](../../projects/refusal_direction/PROCESSO.md)
- [Processo do estudo de J-space](../../projects/jspace_llama/PROCESSO.md)

## Referências

1. NANDA, Neel. [*The Case for Analysing Toy Language
   Models*](https://www.alignmentforum.org/posts/GWCgZrzWCZCuzGktv/200-cop-in-mi-the-case-for-analysing-toy-language-models).
   2022.
2. ELHAGE, Nelson et al. [*A Mathematical Framework for Transformer
   Circuits*](https://transformer-circuits.pub/2021/framework/index.html).
   2021.
3. WANG, Kevin; VARIENGIEN, Alexandre; CONMY, Arthur; SHLEGERIS, Buck;
   STEINHARDT, Jacob. [*Interpretability in the Wild: a Circuit for Indirect
   Object Identification in GPT-2
   Small*](https://arxiv.org/abs/2211.00593). 2022.
4. SYED, Aaquib; RAGER, Can; CONMY, Arthur. [*Attribution Patching
   Outperforms Automated Circuit Discovery*](https://arxiv.org/abs/2310.10348).
   2023.
5. ARDITI, Andy et al. [*Refusal in Language Models Is Mediated by a Single
   Direction*](https://arxiv.org/abs/2406.11717). 2024.
6. GURNEE, Wes et al. [*Verbalizable Representations Form a Global Workspace
   in Language Models*](https://arxiv.org/abs/2607.15495). 2026.
7. ANTHROPIC. [*A global workspace in language
   models*](https://www.anthropic.com/research/global-workspace). 2026.
8. TRANSFORMERLENS. [Repositório oficial no
   GitHub](https://github.com/TransformerLensOrg/TransformerLens).
9. REDWOOD RESEARCH. [*Easy-Transformer*, revisão histórica usada como
   referência](https://github.com/redwoodresearch/Easy-Transformer/tree/ea15315dd24481e9e2ac5c3ef335d82907a1dc34).
10. ARDITI, Andy et al. [Implementação de
    `refusal_direction`](https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3).
11. ANTHROPIC. [Implementação de referência da Jacobian
    lens](https://github.com/anthropics/jacobian-lens).
12. HAOHWW. [Implementação de J-space usada no
    projeto](https://github.com/haohww/jspace/tree/04c822a255fe0a10d84e08ec499e5e09f4b7d37a).
