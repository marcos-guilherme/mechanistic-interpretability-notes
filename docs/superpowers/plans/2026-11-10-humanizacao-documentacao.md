# Humanização seletiva da documentação — Plano de implementação

> **Para agentes executores:** SUB-SKILL OBRIGATÓRIA: usar
> `superpowers:subagent-driven-development` (recomendado) ou
> `superpowers:executing-plans` para executar este plano tarefa por tarefa. Os
> passos usam caixas de seleção (`- [ ]`) para acompanhamento.

**Objetivo:** Tornar a documentação narrativa mais pessoal e humana, em
primeira pessoa singular, apresentando o repositório como registro do processo
prático de entrada do autor em interpretabilidade mecanística.

**Arquitetura:** A revisão será feita em camadas. O README geral apresentará a
trajetória; os READMEs de projeto explicarão a motivação de cada estudo; os
`PROCESSO.md` conectarão decisões, dificuldades e aprendizados às evidências
técnicas. Comandos, tabelas, métricas, hashes, critérios científicos e
artefatos canônicos não serão alterados.

**Tecnologias:** Markdown, Git, validadores Python e pytest já existentes.

## Restrições globais

- Usar primeira pessoa singular apenas em motivação, decisões, incidentes,
  interpretação e aprendizados.
- Manter um tom pessoal, direto e sóbrio, sem linguagem promocional.
- Não alterar números, resultados, hashes, versões ou critérios científicos.
- Não converter comandos, tabelas, schemas ou artefatos em narrativa pessoal.
- Manter IOI e Refusal Direction como estudos com resultados observados.
- Manter J-space como execução inconclusiva por bloqueio de acesso, sem
  resultado científico positivo ou negativo.
- Preservar todas as alterações já existentes na árvore de trabalho.

---

### Tarefa 1: Apresentar a trajetória no README geral e na documentação compartilhada

**Arquivos:**
- Modificar: `README.md`
- Modificar: `docs/replicacoes/README.md`
- Modificar: `docs/replicacoes/METODOLOGIA.md`

**Interfaces:**
- Consome: estados e links dos três projetos já registrados no índice.
- Produz: uma porta de entrada autoral e uma explicação compartilhada da
  metodologia.

- [ ] **Passo 1: Reescrever integralmente o README geral em português**

O novo texto deve começar com esta ideia, preservando-a de forma natural:

```markdown
# Replicações de interpretabilidade mecanística

Criei este repositório para armazenar e documentar meu processo prático de
entrada na área de interpretabilidade mecanística. Mais do que reunir código,
quero registrar o caminho que percorri para transformar artigos em
experimentos executáveis: o que tentei reproduzir, as decisões que tomei, os
problemas que encontrei e até onde cada resultado realmente permite concluir.
```

Em seguida, explicar a progressão:

```markdown
Organizei o trabalho em três estudos. Comecei pelo circuito de identificação
do objeto indireto em GPT-2 Small, avancei para uma direção causal de recusa no
Gemma 2B IT e, por fim, preparei uma intervenção conceitual em J-space no
Llama 3.1 8B Instruct. Nem todas as etapas terminaram da mesma forma: duas
produziram resultados científicos e a tentativa com J-space ficou inconclusiva
antes da inferência, por falta de autorização para acessar os pesos do modelo.
Mantive esse resultado porque a falha operacional também faz parte de um
processo experimental reproduzível.
```

O restante do README deve conter:

- uma seção `## Como este repositório está organizado`;
- links para `docs/replicacoes/README.md`,
  `docs/replicacoes/METODOLOGIA.md` e `docs/replicacoes/MODELO.md`;
- uma seção `## Estudos` com estado e função formativa de cada projeto;
- uma seção `## Como ler os resultados` distinguindo narrativa de artefatos
  canônicos;
- uma nota de que cada projeto mantém ambiente, testes e resultados próprios.

- [ ] **Passo 2: Humanizar a abertura do índice das replicações**

Substituir a abertura impessoal por dois parágrafos em primeira pessoa:

```markdown
Usei esta área para reunir a documentação comum das três replicações e evitar
que decisões importantes ficassem espalhadas apenas entre notebooks, código e
logs. Aqui explico como organizei o trabalho e onde encontrar a evidência de
cada estudo.

Ao longo dos experimentos, procurei manter duas leituras em paralelo: uma
conceitual, voltada a entender a hipótese e o papel de cada intervenção, e
outra operacional, voltada a permitir que a execução seja repetida e
auditada.
```

Manter o índice e os links existentes, alterando apenas transições narrativas.

- [ ] **Passo 3: Contextualizar a origem da metodologia comum**

Adicionar ao início de `METODOLOGIA.md`, depois do título:

```markdown
Construí esta metodologia enquanto avançava pelas três replicações. Ela reúne
os cuidados que se tornaram necessários na prática: separar resultado do
artigo e resultado local, registrar amostras efetivas, controlar intervenções,
preservar proveniência e tratar falhas de ambiente como inconclusivas.
```

Preservar as doze etapas metodológicas em tom objetivo.

- [ ] **Passo 4: Verificar a apresentação e o estado dos projetos**

Executar:

```powershell
rg -n "Criei este repositório|processo prático|inconclusiv" README.md docs/replicacoes/README.md docs/replicacoes/METODOLOGIA.md
```

Esperado: o propósito pessoal aparece no README; o J-space é descrito como
inconclusivo; a metodologia explica sua origem prática.

- [ ] **Passo 5: Registrar a tarefa**

```powershell
git add README.md docs/replicacoes/README.md docs/replicacoes/METODOLOGIA.md
git commit -m "docs: present interpretability learning journey"
```

---

### Tarefa 2: Humanizar e unificar os READMEs dos três projetos

**Arquivos:**
- Modificar: `projects/ioi/README.md`
- Modificar: `projects/refusal_direction/README.md`
- Modificar: `projects/jspace_llama/README.md`

**Interfaces:**
- Consome: resultados e comandos já presentes em cada README.
- Produz: uma motivação autoral para cada experimento sem alterar seu contrato
  de reprodução.

- [ ] **Passo 1: Reescrever a abertura do README de IOI**

A abertura deve explicar em português:

```markdown
Escolhi o circuito de identificação do objeto indireto como meu primeiro
estudo porque ele oferece um exemplo concreto de interpretabilidade
mecanística: um comportamento simples, um circuito publicado e várias formas
de testar causalmente os componentes propostos. Minha meta não foi apenas
executar notebooks, mas entender como se passa de uma previsão do modelo para
uma explicação em termos de cabeças de atenção e caminhos internos.
```

Traduzir o restante da prosa inglesa para português, sem modificar comandos,
nomes de arquivos, configurações, tabelas ou números.

- [ ] **Passo 2: Reescrever a abertura do README de Refusal Direction**

Usar a seguinte direção narrativa:

```markdown
Depois de trabalhar com um circuito distribuído no GPT-2 Small, quis testar
uma hipótese mecanística mais compacta: a possibilidade de um comportamento
ser controlado por uma única direção no espaço residual. Nesta replicação,
investiguei essa hipótese no Gemma 2B IT por meio de extração, ablação, adição
e um controle aleatório ortogonal.
```

Manter as métricas e ressalvas existentes sem alterações numéricas.

- [ ] **Passo 3: Reescrever a abertura e a execução observada do README de J-space**

A abertura deve dizer:

```markdown
O J-space foi o passo seguinte da minha exploração: em vez de analisar um
circuito ou uma direção comportamental, quis testar se seria possível ler e
alterar um conceito intermediário usado pelo modelo. Preparei um experimento
causal mínimo no Llama 3.1 8B Instruct, usando uma J-lens já ajustada e a troca
`spider → ant`.
```

Traduzir a prosa operacional para português. Na seção de execução observada,
usar primeira pessoa para explicar a tentativa, mas manter explícito que os
pesos não foram baixados e nenhuma inferência ou intervenção foi realizada.

- [ ] **Passo 4: Conferir que a personalização não alterou evidências**

Executar:

```powershell
rg -n "3.644|98.83|93,75|0,0306|asset_load_failed|a116c3da" projects --glob README.md
```

Esperado: os números e identificadores anteriores continuam presentes nos
respectivos arquivos.

- [ ] **Passo 5: Registrar a tarefa**

```powershell
git add projects/ioi/README.md projects/refusal_direction/README.md projects/jspace_llama/README.md
git commit -m "docs: humanize project introductions"
```

---

### Tarefa 3: Levar a voz autoral aos relatos de processo

**Arquivos:**
- Modificar: `projects/ioi/PROCESSO.md`
- Modificar: `projects/refusal_direction/PROCESSO.md`
- Modificar: `projects/jspace_llama/PROCESSO.md`

**Interfaces:**
- Consome: as sínteses acadêmicas e evidências canônicas já documentadas.
- Produz: relatos que conectam escolhas e aprendizados aos métodos e
  resultados.

- [ ] **Passo 1: Humanizar contexto, transições e contribuição do IOI**

Na seção `Contexto`, acrescentar por que o estudo foi escolhido. Transformar
“Este projeto pode ser lido...” em “Organizei este projeto...”. Na contribuição
para o TCC, explicar em primeira pessoa o que o estudo ensinou sobre a diferença
entre atribuição, causalidade e validação de circuitos.

Não alterar a descrição dos notebooks, as convenções IOI/pABC, as métricas ou
os resultados.

- [ ] **Passo 2: Humanizar decisões e incidentes de Refusal Direction**

Na síntese, ligar o estudo à progressão iniciada em IOI. Em `Incidentes reais e
o que ensinaram`, usar construções como “Durante a execução, encontrei...” e
“Esse problema me mostrou...”, mantendo as explicações técnicas de Python,
Colab, autenticação, vetores nulos e kernel persistente.

Na conclusão, registrar em primeira pessoa por que a afirmação permanece
restrita ao modelo, seed e recorte.

- [ ] **Passo 3: Humanizar o estado inconclusivo do J-space**

Na síntese e nos resultados, explicar em primeira pessoa a escolha do marco
causal e a tentativa em L4. Manter a formulação explícita:

```markdown
Não interpreto essa execução como evidência contra o J-space. O bloqueio
aconteceu antes de qualquer inferência, readout ou intervenção; portanto, o
resultado científico continua não medido.
```

Na contribuição para o TCC, explicar que registrar corretamente uma falha de
infraestrutura também é parte da prática científica.

- [ ] **Passo 4: Verificar que os estados científicos continuam distintos**

Executar:

```powershell
rg -n "reprodução parcial|reprodução causal parcial|inconclusivo|não medido|não houve inferência" projects --glob PROCESSO.md
```

Esperado: IOI e Refusal Direction preservam resultados observados; J-space
permanece inequivocamente inconclusivo.

- [ ] **Passo 5: Registrar a tarefa**

```powershell
git add projects/ioi/PROCESSO.md projects/refusal_direction/PROCESSO.md projects/jspace_llama/PROCESSO.md
git commit -m "docs: add author voice to experiment narratives"
```

---

### Tarefa 4: Validar coerência, artefatos e reprodução

**Arquivos:**
- Verificar: `README.md`
- Verificar: `docs/replicacoes/*.md`
- Verificar: `projects/*/README.md`
- Verificar: `projects/*/PROCESSO.md`

**Interfaces:**
- Consome: todos os textos revisados.
- Produz: documentação coerente e validada contra os artefatos existentes.

- [ ] **Passo 1: Procurar linguagem desatualizada ou placeholders**

Executar:

```powershell
rg -n "not yet executed|no results are available|TBD|TODO|<[^>]+>" README.md docs/replicacoes projects --glob "*.md"
```

Esperado: nenhuma descrição desatualizada do J-space e nenhum placeholder novo.
O placeholder `<nome>` já existente em um comando de reprodução do IOI é
permitido.

- [ ] **Passo 2: Verificar o diff textual**

Executar:

```powershell
git diff --check
```

Esperado: saída vazia, desconsiderando avisos de conversão LF/CRLF.

- [ ] **Passo 3: Validar IOI**

Executar em `projects/ioi`:

```powershell
cmd /c "uv run python scripts/build_paper_summary.py --check && uv run pytest -q"
```

Esperado: resumo consistente e 130 testes aprovados.

- [ ] **Passo 4: Validar Refusal Direction**

Executar em `projects/refusal_direction`:

```powershell
cmd /c "uv run python scripts/validate_results.py && uv run pytest -q"
```

Esperado: critérios científicos aprovados e 37 testes aprovados.

- [ ] **Passo 5: Validar J-space**

Executar em `projects/jspace_llama`:

```powershell
cmd /c "uv run python scripts/validate_results.py && uv run pytest -q"
```

Esperado: schemas, hashes, proveniência e afirmações consistentes; 300 testes
aprovados.

- [ ] **Passo 6: Revisar o estado final**

Executar:

```powershell
git status --short
git log -5 --oneline
```

Esperado: apenas arquivos deliberadamente modificados aparecem; nenhum
artefato temporário, cache, peso ou resultado bruto foi adicionado.
