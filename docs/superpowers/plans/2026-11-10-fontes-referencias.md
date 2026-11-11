# Fontes e referências Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Criar um índice comentado das fontes do repositório e uma síntese aprofundada em português do guia de pesquisa de Neel Nanda.

**Architecture:** `docs/REFERENCIAS.md` será o índice central, organizado pelo papel de cada fonte e pelos três experimentos. `docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md` será um texto autoral em português que acompanha os temas do artigo de Nanda sem reproduzir uma tradução integral. Os índices existentes apontarão para o documento central, que por sua vez ligará a síntese ao artigo original.

**Tech Stack:** Markdown, links HTTPS para fontes primárias, busca textual com `rg`, validação de formatação com Git e diagnóstico Markdown do editor.

## Global Constraints

- Tratar os textos de Neel Nanda como fontes formativas, não como fontes primárias dos três experimentos.
- Identificar separadamente artigo original, implementação de referência e página do modelo.
- Usar primeira pessoa apenas para explicar como uma fonte orientou o trabalho.
- Manter a seleção curta e diretamente relacionada à formação, ao desenho experimental ou à implementação.
- Escrever a síntese de Nanda com palavras próprias; não fazer tradução frase a frase nem reproduzir trechos extensos.
- Apresentar a síntese como material independente e não oficial, sempre com link e atribuição ao original.
- Preservar commits e revisões já registrados na proveniência do repositório.
- Não modificar artefatos científicos, resultados, notebooks ou código experimental.

---

### Task 1: Criar o índice comentado de fontes

**Files:**
- Create: `docs/REFERENCIAS.md`
- Reference: `projects/ioi/PROCESSO.md`
- Reference: `projects/refusal_direction/PROCESSO.md`
- Reference: `projects/jspace_llama/PROCESSO.md`

**Interfaces:**
- Consumes: títulos, URLs, commits e papéis das fontes já registrados nos três `PROCESSO.md`.
- Produces: um índice central com âncoras estáveis para formação, fundamentos, IOI, direção de recusa e J-space.

- [ ] **Step 1: Confirmar os metadados das fontes primárias**

Consultar as páginas originais e confirmar estes registros:

```text
Neel Nanda (2025)
How To Become A Mechanistic Interpretability Researcher
https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher

Neel Nanda (2024)
An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2
https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1

Elhage et al. (2021)
A Mathematical Framework for Transformer Circuits
https://transformer-circuits.pub/2021/framework/index.html

Wang, Variengien, Conmy, Shlegeris e Steinhardt (2022)
Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
https://arxiv.org/abs/2211.00593

Arditi et al. (2024)
Refusal in Language Models Is Mediated by a Single Direction
https://arxiv.org/abs/2406.11717

Gurnee et al. (2026)
Verbalizable Representations Form a Global Workspace in Language Models
https://arxiv.org/abs/2607.15495
```

Expected: cada página confirma título, autoria e ano; a página da Anthropic
`https://www.anthropic.com/research/global-workspace` é tratada como visão
geral do trabalho de Gurnee et al., não como substituta do artigo.

- [ ] **Step 2: Escrever a estrutura e as entradas comentadas**

Criar `docs/REFERENCIAS.md` com estes títulos:

```markdown
# Fontes e referências

## Como ler este documento
## Formação em interpretabilidade mecanística
## Fundamentos
## Circuito IOI
## Direção de recusa
## J-space
```

Usar em cada entrada o formato:

```markdown
### [Título](URL)

**Papel:** fonte formativa | fundamento | artigo original | implementação | modelo

Comentário em primeira pessoa explicando, em um parágrafo curto, o que foi
aproveitado e qual parte do repositório se relaciona à fonte.
```

Cobrir, além dos seis trabalhos confirmados no passo anterior:

```text
TransformerLens
https://github.com/TransformerLensOrg/TransformerLens

Easy-Transformer usado como referência histórica de IOI
https://github.com/redwoodresearch/Easy-Transformer/tree/ea15315dd24481e9e2ac5c3ef335d82907a1dc34

Implementação de refusal_direction fixada pelo projeto
https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3

Gemma 2B IT
https://huggingface.co/google/gemma-2b-it

Visão geral da Anthropic sobre J-space
https://www.anthropic.com/research/global-workspace

Implementação da Jacobian lens
https://github.com/anthropics/jacobian-lens

Pipeline jspace fixado pelo projeto
https://github.com/haohww/jspace/tree/04c822a255fe0a10d84e08ec499e5e09f4b7d37a

Lente publicada no Hugging Face
https://huggingface.co/neuronpedia/jacobian-lens

Llama 3.1 8B Instruct
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Na seção de formação, explicar que os dois textos de Nanda ajudaram a escolher
uma estratégia de aprendizado por projetos, leitura seletiva e ciclos curtos
de feedback. Em cada seção experimental, nomear explicitamente o artigo
original antes de implementações e modelos.

- [ ] **Step 3: Verificar cobertura e linguagem**

Run:

```powershell
rg -n "How To Become|Favourite Mechanistic|Mathematical Framework|Interpretability in the Wild|Single Direction|Verbalizable Representations|Easy-Transformer|refusal_direction|jacobian-lens|Llama-3.1-8B-Instruct" docs/REFERENCIAS.md
rg -n "tradução oficial|consenso da área|artigo original de Neel" docs/REFERENCIAS.md
git diff --check -- docs/REFERENCIAS.md
```

Expected: a primeira busca encontra todas as fontes; a segunda não encontra
afirmações incorretas; `git diff --check` não produz saída.

- [ ] **Step 4: Commit**

```powershell
git add docs/REFERENCIAS.md
git commit -m "docs: add annotated source index"
```

---

### Task 2: Criar a síntese aprofundada do guia de Neel Nanda

**Files:**
- Create: `docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md`
- Modify: `docs/REFERENCIAS.md`

**Interfaces:**
- Consumes: o artigo *How To Become A Mechanistic Interpretability Researcher* e a entrada formativa criada na Task 1.
- Produces: uma síntese autoral em português, ligada ao original e ao índice de fontes.

- [ ] **Step 1: Delimitar autoria e escopo no início do guia**

Abrir o documento com:

```markdown
# Guia de pesquisa em interpretabilidade mecanística

Este texto é uma síntese comentada, em português e escrita com palavras
próprias, de [*How To Become A Mechanistic Interpretability
Researcher*](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher),
de Neel Nanda. Não é uma tradução integral nem oficial. Organizei aqui as
ideias que considero mais úteis para aprender interpretabilidade mecanística e
para conduzir pesquisa empírica de modo mais geral.
```

- [ ] **Step 2: Escrever a síntese temática completa**

Usar estes títulos e cobrir os pontos indicados:

```markdown
## A ideia central: aprender fazendo
```

Explicar o princípio de aprender o mínimo viável, começar experimentos cedo e
usar a prática para revelar lacunas reais.

```markdown
## Etapa 1: construir uma base suficiente
```

Cobrir fundamentos de transformers, álgebra linear, ferramentas de
interpretabilidade e leitura panorâmica; enfatizar amplitude antes de
perfeição.

```markdown
## Etapa 2: aprender com miniprojetos
```

Explicar projetos de um a cinco dias, ganho de informação por unidade de
tempo, exploração e testes que possam refutar resultados promissores.

```markdown
## Etapa 3: avançar para projetos completos
```

Cobrir ciclos de uma a duas semanas, retrospectivas, critérios para continuar
ou mudar de direção e evolução gradual das habilidades lentas.

```markdown
## Quatro movimentos da pesquisa
```

Apresentar ideação, exploração, entendimento e comunicação como movimentos
distintos, com objetivos e tipos de evidência diferentes.

```markdown
## Ciclos de feedback e tomada de decisão
```

Distinguir habilidades rápidas, médias e lentas; explicar por que não se tenta
otimizar todas ao mesmo tempo e como registrar decisões e resultados.

```markdown
## Ceticismo e boa ciência
```

Cobrir baselines reais, controles, busca de contraexemplos, prevenção de
seleção conveniente, leitura atenta dos dados e honestidade sobre limitações.

```markdown
## Escrita, mentoria e desenvolvimento profissional
```

Explicar escrita como parte da pesquisa, valor de trabalho público, procura de
feedback e mentores e cuidado para não confundir popularidade com relevância.

```markdown
## O que é específico da interpretabilidade e o que é geral
```

Separar ferramentas próprias da área — hooks, ativações, circuitos e
intervenções — de princípios transferíveis: ciclos curtos, hipóteses
refutáveis, controles, priorização e comunicação.

```markdown
## Como isso aparece neste repositório
```

Relacionar IOI à aprendizagem das ferramentas, direção de recusa a um projeto
causal mais compacto e J-space à tentativa de uma intervenção conceitual mais
ambiciosa. Registrar que o resultado inconclusivo de J-space exemplifica a
separação entre falha operacional e evidência científica.

Encerrar com um link para `REFERENCIAS.md` e para o artigo original. Não usar
tradução frase a frase e não incluir citações diretas longas.

- [ ] **Step 3: Ligar o guia ao índice de fontes**

Na entrada de *How To Become A Mechanistic Interpretability Researcher* em
`docs/REFERENCIAS.md`, acrescentar ao final:

```markdown
Preparei também uma [síntese aprofundada em
português](GUIA_PESQUISA_EM_INTERPRETABILIDADE.md), escrita com palavras
próprias e voltada às ideias que considero transferíveis para pesquisa.
```

- [ ] **Step 4: Verificar estrutura, atribuição e originalidade**

Run:

```powershell
rg -n "^## " docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md
rg -n "Não é uma tradução integral nem oficial|Neel Nanda|artigo original|IOI|direção de recusa|J-space" docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md
git diff --check -- docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md docs/REFERENCIAS.md
```

Expected: dez seções temáticas, atribuição explícita, vínculo com os três
estudos e nenhuma falha de formatação.

- [ ] **Step 5: Commit**

```powershell
git add docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md docs/REFERENCIAS.md
git commit -m "docs: add Portuguese research guide"
```

---

### Task 3: Integrar os documentos aos índices e validar

**Files:**
- Modify: `README.md:20-28`
- Modify: `docs/replicacoes/README.md:7-13`
- Verify: `docs/REFERENCIAS.md`
- Verify: `docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md`

**Interfaces:**
- Consumes: os dois documentos criados nas Tasks 1 e 2.
- Produces: caminhos de descoberta a partir da raiz e do índice das replicações.

- [ ] **Step 1: Adicionar o índice de fontes ao README principal**

Na lista em “Como este repositório está organizado”, adicionar:

```markdown
- as [fontes e referências](docs/REFERENCIAS.md) distinguem os artigos
  originais, os materiais de formação e as implementações que orientaram os
  estudos.
```

- [ ] **Step 2: Adicionar o índice de fontes à documentação comum**

Depois do parágrafo que aponta para `METODOLOGIA.md` e `MODELO.md`, adicionar:

```markdown
As [fontes e referências](../REFERENCIAS.md) registram os artigos originais,
os materiais de formação e as implementações que orientaram os três estudos.
```

- [ ] **Step 3: Validar arquivos, caminhos e formatação**

Run:

```powershell
python -c "from pathlib import Path; paths=['docs/REFERENCIAS.md','docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md','README.md','docs/replicacoes/README.md']; missing=[p for p in paths if not Path(p).is_file()]; assert not missing, missing; print('PASS: documentation files exist')"
rg -n "docs/REFERENCIAS.md|../REFERENCIAS.md|GUIA_PESQUISA_EM_INTERPRETABILIDADE.md" README.md docs/replicacoes/README.md docs/REFERENCIAS.md
rg -n "\b(TBD|TODO)\b|a definir" docs/REFERENCIAS.md docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md
git diff --check -- README.md docs/replicacoes/README.md docs/REFERENCIAS.md docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md
```

Expected: o script imprime `PASS`; os três links locais aparecem; a busca por
itens incompletos não encontra resultados; `git diff --check` não produz
saída. Conferir também que o editor não relata erros de lint nos quatro
arquivos.

- [ ] **Step 4: Revisar o diff sem absorver mudanças externas**

Run:

```powershell
git diff -- README.md docs/replicacoes/README.md docs/REFERENCIAS.md docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md
git status --short
```

Expected: o diff contém apenas a integração dos novos documentos e alterações
já autorizadas nesses dois índices; arquivos de J-space que já estavam
modificados permanecem fora deste trabalho.

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/replicacoes/README.md
git commit -m "docs: link research references"
```

