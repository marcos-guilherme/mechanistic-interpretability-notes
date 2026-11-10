# Processo de estudo e reprodução causal do J-space

[← Voltar à metodologia comum](../../docs/replicacoes/METODOLOGIA.md)

> **Estado:** inconclusivo por falha de acesso ao ativo
>
> **Última atualização:** 10 de novembro de 2026

## Síntese acadêmica

### Contexto

O artigo da Anthropic
[*A global workspace in language models*](https://www.anthropic.com/research/global-workspace)
descreve o J-space como um espaço interno no qual conceitos relevantes para o
raciocínio corrente podem ser lidos e manipulados. A técnica usa uma
*Jacobian lens* (J-lens) para relacionar estados internos do modelo a conceitos
interpretáveis. O resultado mais forte não é apenas correlacional: em
experimentos do trabalho original, alterar um conceito intermediário modifica
a resposta posterior do modelo, sugerindo uso causal da representação.

Escolhi investigar se um marco causal mínimo desse tipo poderia ser
transferido para um modelo aberto menor, o
`meta-llama/Llama-3.1-8B-Instruct`, usando uma J-lens já ajustada e publicada
para esse modelo. Depois de trabalhar com circuitos e direções residuais, quis
dar um passo em direção a intervenções sobre conceitos intermediários. Minha
proposta não é reproduzir toda a análise da Anthropic, mas testar a cadeia
mínima “ler um conceito → intervir nele → observar uma mudança específica na
resposta”.

### Problema de pesquisa

A pergunta central é: **uma representação intermediária associada a `spider`
pode ser identificada no J-space do Llama 3.1 8B Instruct e causalmente
redirecionada para `ant`, alterando a resposta de oito para seis pernas sem
degradar a geração?**

A hipótese seria apoiada se quatro condições ocorressem conjuntamente:

1. o modelo sem intervenção respondesse oito;
2. o readout da J-lens identificasse `spider`;
3. a troca `spider → ant` produzisse uma resposta única equivalente a seis,
   sem simplesmente revelar a palavra `ant`;
4. o no-op e a perturbação aleatória pareada não produzissem o mesmo
   redirecionamento.

Falha de download, ausência de autorização, incompatibilidade da lente,
baseline inválida ou proveniência incompleta tornam a execução inconclusiva.
Essas condições não constituem evidência negativa sobre J-space.

### Métodos

O experimento foi projetado para o prompt principal “The animal that spins
webs has this many legs:” e duas paráfrases. A geração é determinística
(`do_sample=false`), limitada a 12 novos tokens e executada com seed 42. A
J-lens pré-ajustada
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`
é carregada do repositório `neuronpedia/jacobian-lens`; uma lente nova não é
treinada.

Quatro condições usam exatamente os mesmos prompts e parâmetros:

- **baseline:** modelo sem hooks;
- **no-op:** troca `spider → spider`, para detectar efeitos da
  instrumentação;
- **controle aleatório:** direção ortogonal aleatória pareada pela norma;
- **intervenção-alvo:** clamp causal `spider → ant` durante o *prefill*.

O readout e as intervenções são limitados em memória para permitir execução em
uma NVIDIA L4. O parser aceita apenas respostas inequívocas equivalentes a
oito ou seis e rejeita negação, conflito, qualificação, repetição, incoerência
e vazamento explícito do conceito-alvo. Os artefatos públicos armazenam apenas
métricas compactas e proveniência; prompts renderizados, continuações e
diagnósticos detalhados permanecem fora do Git.

### Resultados

A tentativa que registrei, identificada por
`a116c3da-bb8f-4a31-a39a-4fd09dd6fcad`, iniciou em uma NVIDIA L4 e resolveu as
revisões do modelo e da lente, mas terminou com estado `inconclusive` e código
`asset_load_failed`. A conta do Hugging Face estava autenticada localmente,
porém não autorizada a acessar o repositório restrito do Llama. Os pesos não
foram baixados.

Consequentemente, não houve inferência do modelo, readout de `spider`, teste de
integridade da lente nem intervenção `spider → ant`. Os campos científicos
correspondentes estão registrados como não avaliados e `null`, e não como
falhas da hipótese. O resultado disponível é operacional: o fluxo chegou até a
resolução dos ativos, registrou ambiente e motivo terminal e falhou de forma
segura. Não interpreto essa execução como evidência contra o J-space. O
bloqueio aconteceu antes de qualquer inferência, readout ou intervenção;
portanto, o resultado científico continua não medido. **Ainda não há resultado
científico positivo ou negativo sobre J-space no Llama nesta replicação.**

### Contribuição para o TCC

Mesmo inconclusivo, este estudo tem uma função metodológica importante no meu
TCC: ele me obrigou a separar falha de infraestrutura de resultado
experimental. Em conjunto com IOI e Refusal Direction, representa o terceiro
nível de análise causal que explorei — a intervenção sobre conceitos em um
espaço de trabalho aprendido por uma lente. A tentativa também me mostrou, de
forma concreta, que autenticação técnica não equivale a licença de acesso e
que valores não medidos devem permanecer ausentes, em vez de serem
interpretados como zeros ou resultados negativos.

## Trabalho de referência e implementação

- Artigo: Anthropic, [*A global workspace in language
  models*](https://www.anthropic.com/research/global-workspace).
- Código de referência da lente: [Anthropic
  `jacobian-lens`](https://github.com/anthropics/jacobian-lens).
- Pipeline usado localmente: [`haohww/jspace`](https://github.com/haohww/jspace),
  commit `04c822a255fe0a10d84e08ec499e5e09f4b7d37a`.
- Modelo: `meta-llama/Llama-3.1-8B-Instruct`, revisão resolvida
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- Lente: `neuronpedia/jacobian-lens`, revisão resolvida
  `d2527c3cdbaa60dd4b65cbca1604921f044e989a`.

### Escopo e fidelidade

**Mantido:** uso de uma Jacobian lens, leitura de conceitos intermediários,
intervenção causal entre conceitos e exigência de mudança comportamental
específica.

**Adaptado:** modelo aberto de 8 bilhões de parâmetros, lente pré-ajustada,
apenas uma relação factual e uma GPU L4. A implementação usa o pacote
`jspace`, e não uma reprodução independente de todo o código do artigo.

**Fora do escopo:** ajuste da lente, múltiplos modelos e tarefas, geometria
global do workspace, reportabilidade ampla, ablações em escala, comparações
estatísticas e generalização do mecanismo.

## Desenho experimental

### Linha de base e readout

A linha de base é válida somente se o prompt principal produzir uma resposta
única equivalente a oito sem degradação. Em seguida, o readout deve avaliar
explicitamente se `spider` aparece nas camadas examinadas. A ausência dessa
condição impede interpretar uma troca causal posterior.

### Intervenção e controles

Para cada prompt e valor de `alpha`, as quatro condições são executadas sobre a
mesma tokenização. O no-op testa neutralidade dos hooks. O controle aleatório
é ortogonal e pareado pela norma com a alteração-alvo, para verificar se uma
perturbação genérica produz o mesmo efeito. A intervenção-alvo faz o clamp da
representação-fonte para a representação-alvo apenas no *prefill*, evitando
alterações sucessivas durante a geração autoregressiva.

O critério principal exige baseline e no-op com oito, intervenção-alvo com
seis, controle aleatório com uma resposta coerente diferente de seis, ausência
de vazamento de `ant` e ausência de degradação. As paráfrases são dados
secundários de transparência, não uma afirmação de generalização.

## Ambiente e proveniência observados

- GPU: NVIDIA L4.
- CUDA: 13.2.
- Dtype solicitado: `bfloat16`.
- Python: 3.13.16.
- PyTorch: 2.15.0.
- Transformers: 5.22.2.
- `huggingface_hub`: 2.4.0.
- Seed: 42.
- Commit local: `bda81311ed758749598d9cd41fb7bbf0d5d3897a`.
- Estado do Git no envio: sujo por alterações externas ao projeto.

O manifesto registra `provenance_complete=false`, pois o bloqueio ocorreu
antes de medir camadas, hash e tamanho da lente. Esses campos não devem ser
completados retrospectivamente.

## Execução e reprodução

O modelo é *gated*. Antes da execução, é necessário entrar no site do Hugging
Face com a mesma conta usada no CLI e aceitar ou solicitar acesso na
[página oficial do modelo](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
O comando `hf auth login` confirma identidade, mas não concede a licença.

Na raiz do repositório:

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb --timeout 7200
python tools/colab/run_jspace.py fetch
python tools/colab/run_jspace.py stop
```

Depois de baixar os artefatos:

```powershell
Set-Location projects/jspace_llama
uv run python scripts/validate_results.py
uv run pytest -q
```

A sessão de GPU deve ser encerrada mesmo após erro ou queda da conexão local.

## Evidência canônica

- [`results/execution_manifest.json`](results/execution_manifest.json):
  ambiente, revisões resolvidas, configuração e motivo terminal.
- [`results/final_evaluation.json`](results/final_evaluation.json): estado
  científico agregado e campos explicitamente não avaliados.
- [`README.md`](README.md): instruções operacionais resumidas.
- [`notebooks/01_readout_and_causal_swap.ipynb`](notebooks/01_readout_and_causal_swap.ipynb):
  ponto de entrada da execução real.

O artefato final registra zero prompts e zero alphas avaliados. Esses zeros são
contagens operacionais de avaliações realizadas, não medidas do efeito causal.

## Desvios e limitações

| Desvio ou limitação | Motivo | Impacto |
|---|---|---|
| Modelo e lente não carregados | Falta de autorização para o repositório gated | Nenhuma hipótese científica foi testada |
| Apenas um marco causal planejado | Restrição deliberada de escopo | Não replica a abrangência do artigo |
| Uma seed e um valor inicial de `alpha` | Experimento mínimo | Não permite estimar variabilidade ou sensibilidade |
| Estado Git sujo no envio | Alterações do usuário fora do projeto | Proveniência registrada, mas não inteiramente limpa |
| Lente pré-ajustada | Custo de ajuste e foco causal | Não valida o procedimento de treinamento da lente |

## Próximo passo científico

Depois de obter a aprovação oficial de acesso ao Llama, repetirei `auth-hf`,
`execute`, `fetch` e a validação local. Somente um novo artefato com readout,
intervenções e controles efetivamente avaliados poderá substituir a conclusão
inconclusiva. Se o baseline ou o readout falhar, manterei o protocolo como
inconclusivo e o ajustarei antes de qualquer interpretação causal.
