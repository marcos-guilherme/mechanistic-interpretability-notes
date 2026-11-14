# Modelo de replicação

Copie este roteiro para `projects/<estudo>/PROCESSO.md` e substitua os campos
entre `<...>`. Ao copiar, ajuste todos os links relativos para o novo local. No
destino esperado, o link para a metodologia deve apontar para
`../../docs/replicacoes/METODOLOGIA.md`. Remova seções que não se aplicam, mas
não apague uma limitação apenas porque ela enfraquece a conclusão. Neste
arquivo-fonte, a [metodologia comum](METODOLOGIA.md) explica o propósito de cada
etapa.

> **Estado:** `<planejado | em execução | parcial | concluído>`
>
> **Última atualização:** `<data verificável>`

## Visão geral

**Pergunta científica.** `<qual relação observável ou causal será testada?>`

**Hipótese.** `<qual resultado apoiaria a hipótese?>`

**Condições negativas ou inconclusivas.** `<o que distinguiria um resultado
negativo de uma falha de ambiente, dados ou instrumento?>`

**Limite da conclusão.** `<o que este estudo não poderá afirmar, mesmo se for
bem-sucedido?>`

## Trabalho de referência

- Artigo: `<título e link>`
- Código upstream: `<repositório e link>`
- Commit ou versão fixada: `<identificador verificável>`
- Modelo e revisão: `<identificador exato>`
- Licença e restrições relevantes: `<registro ou não se aplica>`

### Escopo e fidelidade

**Mantido do trabalho de referência:** `<definições, dados, métricas,
procedimentos e controles mantidos>`

**Adaptado:** `<mudanças necessárias e justificativas>`

**Fora do escopo:** `<experimentos e afirmações não reproduzidos>`

**Extensões locais:** `<análises novas, explicitamente não atribuídas ao
artigo>`

## Desenho experimental

### Linha de base

Descreva a tarefa, população, pré-processamento, configuração de geração e
métricas usadas antes de qualquer intervenção.

**Critério de validade da linha de base:** `<condição verificável>`

### Captura e instrumentação

- Estado capturado: `<componente, camada, posição e instante>`
- Alinhamento: `<tokenização e regra de posição>`
- Hook ou mecanismo: `<entrada/saída e ciclo de vida>`
- Controle de neutralidade: `<comparação no-op ou equivalente>`

### Intervenção causal

**Operação:** `<definição matemática e referência à implementação>`

**Seleção:** `<dados e métrica usados para escolher componente ou
hiperparâmetro>`

**Avaliação final:** `<dados separados e critérios pré-declarados>`

**Critérios de validação pré-declarados:** `<condições verificáveis de sucesso,
resultado negativo e resultado inconclusivo>`

### Controles

- Linha de base: `<descrição>`
- No-op: `<descrição ou justificativa para não se aplicar>`
- Aleatório pareado: `<como escala, norma ou distribuição são pareadas>`
- Outros controles: `<posição, camada, sinal, referência ou tarefa>`
- Preservação: `<coerência, perda, desempenho geral ou outra medida>`

## Dados e amostras

| Etapa | Amostra planejada | Amostra inicial | Exclusões e motivo | Amostra efetiva | Identificador ou hash |
|---|---|---|---|---|---|
| Seleção/treino | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` |
| Validação | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` |
| Avaliação final | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` | `<registrar>` |

Não preencha células ausentes com estimativas. Use `não medido`, `não se
aplica` ou deixe o estudo como planejado até que o valor venha de um artefato.

## Ambiente e proveniência

- Python e dependências: `<lockfile, manifesto ou artefato>`
- Hardware e GPU: `<modelo observado na execução>`
- Dispositivo e dtype: `<valores observados>`
- Sementes: `<valores configurados>`
- Configuração de geração: `<arquivo canônico>`
- Commit local e estado do Git: `<valores registrados>`
- Ativos locais não versionados: `<hashes e função, sem conteúdo restrito>`

## Execução

Pré-requisitos:

```text
<comandos mínimos, sem segredos>
```

Linha de base e experimento:

```text
<comandos reproduzíveis>
```

Validação:

```text
<testes e verificadores de resultados>
```

Encerramento de GPU:

```text
<comando explícito de encerramento e comando de recuperação>
```

Registre no manifesto como foi confirmado que a sessão remota terminou.

## Resultados

### Observado nesta replicação

Esta seção contém somente resultados produzidos por execuções registradas neste
repositório. Para cada afirmação, aponte para o artefato canônico, informe a
amostra efetiva e registre incerteza quando disponível.

- `<resultado observado>` — fonte: `<link relativo para artefato>`
- `<controle observado>` — fonte: `<link relativo para artefato>`
- `<medida de preservação>` — fonte: `<link relativo para artefato>`

Se o estudo ainda não foi executado, escreva apenas: `Resultados ainda não
disponíveis; estudo planejado.`

### Reportado no artigo

Resuma apenas valores e afirmações atribuíveis à fonte citada. Não apresente
esses números como linha de base local nem os copie para um artefato observado.

- `<resultado do artigo>` — fonte: `<seção, figura ou tabela e link>`

### Comparação

Compare definições, direção do efeito, amostras e incerteza antes de comparar
números. Use linguagem como `compatível`, `divergente` ou `não comparável` com
justificativa; evite declarar reprodução exata apenas pela proximidade de uma
estimativa.

## Desvios e limitações

| Desvio ou limitação | Motivo | Momento | Impacto possível |
|---|---|---|---|
| `<registrar>` | `<registrar>` | `<antes/durante/depois>` | `<registrar>` |

Inclua falhas parciais, alterações de dependência, revisão de modelo, hardware,
filtros posteriores e diferenças de amostra. Um estado inconclusivo deve ser
mantido como tal.

## Segurança de dados

Não inclua neste documento prompts nocivos, gerações brutas, credenciais,
tokens, caminhos com segredos ou dados sujeitos a restrições. Registre onde os
arquivos locais ignorados ficam, quem pode acessá-los e quais hashes permitem
relacioná-los aos resultados agregados.

Antes de versionar, revise saídas de notebooks, logs e metadados. Confirme que
os artefatos públicos contêm apenas configurações permitidas, proveniência e
métricas agregadas.

## Fontes narrativas e artefatos canônicos

A narrativa ajuda a estudar o experimento; os artefatos canônicos sustentam as
afirmações verificáveis.

| Papel | Fonte | Regra |
|---|---|---|
| Explicação e contexto | `<README ou notebook>` | Pode interpretar, mas deve ligar à evidência. |
| Configuração executada | `<JSON, YAML ou manifesto>` | É a fonte da configuração observada. |
| Métricas e amostras | `<resultado agregado>` | É a fonte dos números reportados. |
| Proveniência da execução | `<manifesto>` | É a fonte de versões, hardware e estado. |
| Afirmações do paper | `<artigo citado>` | Nunca devem aparecer como observação local. |

Quando a narrativa divergir de um artefato canônico, não escolha o valor mais
conveniente. Investigue a origem, corrija a derivação ou registre a divergência.

## Checklist de encerramento

- [ ] A pergunta, a hipótese e as condições inconclusivas estão explícitas.
- [ ] Escopo fiel, adaptações e extensões locais estão separados.
- [ ] Linha de base, intervenção e controles são comparáveis.
- [ ] Amostras planejada, inicial e efetiva, exclusões e hashes foram registrados.
- [ ] Resultados observados e resultados do artigo estão separados.
- [ ] A narrativa aponta para artefatos canônicos por links relativos válidos.
- [ ] Desvios e limitações foram registrados sem números inventados.
- [ ] Prompts nocivos, gerações brutas, segredos e ativos pesados estão fora do Git.
- [ ] Testes e validadores de resultados foram executados.
- [ ] A sessão de GPU foi encerrada e essa condição foi verificada.
