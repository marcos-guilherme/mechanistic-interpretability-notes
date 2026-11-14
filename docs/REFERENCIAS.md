# Fontes e referências

Este documento reúne as fontes que mais influenciaram minha formação e os
três estudos deste repositório. Não é uma revisão sistemática da literatura.
É um mapa comentado: separo os textos que me ajudaram a aprender, os artigos
que originaram os experimentos, as implementações que usei como referência e
as páginas dos modelos executados.

## Como ler este documento

Uso cinco rótulos:

- **fonte formativa:** material que orientou meu percurso de estudo e minha
  maneira de conduzir pesquisa;
- **fundamento:** texto ou ferramenta que forneceu conceitos e vocabulário
  comuns aos experimentos;
- **artigo original:** publicação que propôs o resultado ou método reproduzido;
- **implementação:** código usado para compreender ou reconstruir o protocolo;
- **modelo:** página oficial dos pesos empregados na execução.

Esses papéis não são intercambiáveis. Um guia de leitura pode ter sido
decisivo para eu encontrar um trabalho, mas continua diferente do artigo que
apresentou o experimento. Da mesma forma, um repositório ajuda a reconstruir o
procedimento, mas não substitui a descrição científica nem os resultados da
publicação.

## Formação em interpretabilidade mecanística

### [How To Become A Mechanistic Interpretability Researcher](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)

**Papel:** fonte formativa (Neel Nanda, 2025).

Este texto foi importante para transformar leitura em prática. A orientação
de aprender uma base suficiente e partir cedo para projetos concretos ajudou
a dar forma à sequência deste repositório: comecei com um circuito conhecido,
avancei para uma intervenção mais compacta e depois tentei trabalhar com uma
representação conceitual. Também aproveitei a ênfase em ciclos curtos de
feedback, ceticismo diante de resultados interessantes e escrita como parte da
pesquisa, não como uma tarefa posterior.

Preparei também uma [síntese aprofundada em
português](GUIA_PESQUISA_EM_INTERPRETABILIDADE.md), escrita com palavras
próprias e voltada às ideias que considero transferíveis para pesquisa.

### [An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1)

**Papel:** fonte formativa (Neel Nanda, 2024).

Usei esta lista comentada como mapa de entrada para uma área com muitas linhas
de pesquisa e vocabulários diferentes. Ela me ajudou a priorizar fundamentos,
identificar trabalhos que valiam uma leitura mais profunda e entender como
circuitos, superposição, representações lineares e intervenções causais se
relacionam. As escolhas de Nanda são assumidamente pessoais; registro aqui a
influência que tiveram no meu percurso, não como se representassem um consenso
ou uma revisão exaustiva da área.

## Fundamentos

### [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

**Papel:** fundamento (Nelson Elhage et al., 2021).

Este trabalho forneceu parte do vocabulário usado para pensar o transformer
como uma composição de caminhos no fluxo residual. As distinções entre
circuitos QK e OV, escrita aditiva no residual e composição entre cabeças
ajudam a entender por que os experimentos de IOI procuram não apenas
ativações importantes, mas rotas causais entre componentes.

### [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

**Papel:** fundamento e implementação.

TransformerLens foi a principal ferramenta para inspecionar e intervir no
GPT-2 Small. Hooks, caches de ativação, decomposição do residual e acesso aos
resultados de cabeças tornam executáveis várias perguntas introduzidas pela
literatura de circuitos. No projeto de IOI, fixei a versão e registrei
explicitamente as transformações de carregamento porque mudanças de contrato
da biblioteca podem alterar uma reprodução.

## Circuito IOI

### [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)

**Papel:** artigo original (Kevin Wang, Alexandre Variengien, Arthur Conmy,
Buck Shlegeris e Jacob Steinhardt, 2022).

Este é o trabalho reproduzido em `projects/ioi`. O artigo identifica em GPT-2
Small um circuito de 26 cabeças distribuídas em sete classes funcionais,
introduz o *path patching* no estudo e avalia a explicação por
*faithfulness*, *completeness* e *minimality*. Minha reprodução separa uma
etapa didática de uma etapa mais fiel ao protocolo e compara os resultados
locais sem afirmar igualdade numérica com o artigo.

### [Easy-Transformer no commit histórico usado pelo projeto](https://github.com/redwoodresearch/Easy-Transformer/tree/ea15315dd24481e9e2ac5c3ef335d82907a1dc34)

**Papel:** implementação.

O código histórico complementa detalhes que não cabem integralmente no artigo,
sobretudo para completude, minimalidade e exemplos adversariais. Usei esse
commit como referência de proveniência, mas adaptei pontos incompatíveis com o
TransformerLens atual e documentei separadamente correções de tokenização,
posições semânticas e referências de ablação.

## Direção de recusa

### [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

**Papel:** artigo original (Andy Arditi et al., 2024).

Este é o trabalho reproduzido em `projects/refusal_direction`. O artigo mostra
que uma direção aproximadamente unidimensional do fluxo residual medeia a
recusa em vários modelos: removê-la reduz a recusa e adicioná-la pode induzir
recusa em pedidos inofensivos. Meu estudo testa esse padrão causal em um
recorte menor, com Gemma 2B IT, e mantém explícitos os limites decorrentes do
modelo, da seed, da seleção e dos conjuntos avaliados.

### [Implementação `refusal_direction` no commit usado pelo projeto](https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3)

**Papel:** implementação.

Esse commit forneceu os splits e o protocolo de referência para extração,
ablação, adição e seleção da direção. Fixar a revisão foi importante para não
misturar alterações futuras do código upstream com a execução registrada
localmente.

### [`google/gemma-2b-it`](https://huggingface.co/google/gemma-2b-it)

**Papel:** modelo.

Esta é a página oficial do modelo usado na reprodução. Além dos pesos e da
licença, ela documenta o acesso restrito que precisa ser aceito antes da
execução. No repositório, o hash efetivamente resolvido é registrado nos
artefatos porque solicitar apenas a revisão móvel `main` não garante os mesmos
pesos no futuro.

## J-space

### [Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495)

**Papel:** artigo original (Wes Gurnee et al., 2026).

Este é o trabalho científico que apresenta a Jacobian lens e investiga o
J-space como um conjunto privilegiado de representações verbalizáveis,
reportáveis e causalmente usado em certas formas de raciocínio. O estudo em
`projects/jspace_llama` não tenta reproduzir toda essa análise: ele executou um
protocolo causal mínimo em um modelo aberto menor, com uma lente previamente
ajustada. A execução foi operacionalmente completa, mas cientificamente
inconclusiva (`primary_gates_failed`).

### [A global workspace in language models](https://www.anthropic.com/research/global-workspace)

**Papel:** visão geral do artigo (Anthropic, 2026).

Esta apresentação oferece uma entrada mais acessível para a hipótese, os
experimentos e seus limites. Ela foi útil para formar a intuição de que o
J-space não é simplesmente o texto produzido pelo modelo nem todo o seu
processamento interno. Para atribuição acadêmica e detalhes metodológicos,
considero o artigo de Gurnee et al. a fonte primária.

### [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)

**Papel:** implementação de referência.

Este repositório contém a implementação aberta dos métodos centrais da
Jacobian lens. Ele é a referência mais próxima do trabalho original; minha
execução no Llama usa um pipeline separado e uma lente publicada, portanto
essa distinção precisa permanecer visível.

### [`haohww/jspace` no commit fixado](https://github.com/haohww/jspace/tree/04c822a255fe0a10d84e08ec499e5e09f4b7d37a)

**Papel:** implementação usada localmente.

O projeto de J-space utiliza esse pacote para carregar a lente, produzir
readouts e aplicar intervenções. Fixei o commit no ambiente para que uma
execução futura não receba silenciosamente uma versão diferente do pipeline.

### [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)

**Papel:** lente publicada.

Desse repositório vem a J-lens pré-ajustada para
`Llama-3.1-8B-Instruct`. A revisão e o caminho exatos do arquivo são
registrados na proveniência. Usar uma lente pronta reduz o escopo do
experimento: testa-se readout e intervenção, não o treinamento da lente.

### [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

**Papel:** modelo.

Esta é a página oficial do modelo escolhido para o marco causal
`spider → ant`. O repositório é restrito; autenticar o cliente do Hugging Face
não substitui a aprovação da licença. Uma primeira tentativa parou nesse
limite, antes da inferência. Depois que o acesso foi aprovado, a execução
causal canônica completou o protocolo, mas permaneceu cientificamente
inconclusiva (`primary_gates_failed`) porque os critérios primários falharam.
O atlas executado em seguida foi validado como resultado observacional; ele
não contém intervenção e não reclassifica a tentativa causal.
