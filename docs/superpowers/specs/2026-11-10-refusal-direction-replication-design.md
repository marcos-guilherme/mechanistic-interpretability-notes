# Monorepo e replicação de Refusal Direction

## Objetivo

Transformar o repositório atual em um monorepo de replicações de
interpretabilidade mecanística, mantendo a replicação IOI reproduzível e
isolando uma nova replicação principal de *Refusal in Language Models Is
Mediated by a Single Direction*.

A primeira entrega deve reproduzir, em `google/gemma-2b-it`, três resultados
centrais:

1. extração de direções candidatas por diferença de médias entre instruções
   nocivas e benignas;
2. redução de recusa por ablação direcional;
3. indução de recusa em instruções benignas por adição da direção.

`meta-llama/Meta-Llama-3-8B-Instruct` será uma extensão condicionada à
disponibilidade de L4 ou A100 no Colab.

## Não objetivos

- Reproduzir os 13 modelos do artigo.
- Executar modelos de 72B parâmetros.
- Reproduzir inicialmente a ortogonalização dos pesos, o ajuste LoRA ou a
  análise mecanística de sufixos adversariais.
- Automatizar geração de jailbreaks.
- Tratar recusa como sinônimo de alinhamento completo.
- Compartilhar tokens do Google, Hugging Face ou APIs de avaliação.

## Estrutura do monorepo

Cada replicação será um projeto Python independente, com lockfile e ambiente
próprios. A raiz servirá para documentação e ferramentas de orquestração, sem
um ambiente Python compartilhado.

```text
mechanistic-interpretability-replications/
├── projects/
│   ├── ioi/
│   │   ├── src/ioi_replication/
│   │   ├── notebooks/
│   │   ├── tests/
│   │   ├── scripts/
│   │   ├── results/
│   │   ├── pyproject.toml
│   │   └── uv.lock
│   └── refusal_direction/
│       ├── src/refusal_direction_replication/
│       ├── notebooks/
│       ├── tests/
│       ├── scripts/
│       ├── configs/
│       ├── results/
│       ├── pyproject.toml
│       └── uv.lock
├── tools/colab/
├── docs/
├── .gitignore
└── README.md
```

O projeto IOI será movido sem alterar seus nomes de pacote, notebooks,
artefatos ou definições experimentais. Todos os comandos e testes serão
atualizados para executar a partir de `projects/ioi`.

O uso de projetos independentes, em vez de um único workspace `uv`, evita que
o contrato de TransformerLens 4.3.0 e o índice CUDA do IOI imponham versões ao
ambiente Hugging Face da replicação de recusa.

## Projeto `refusal_direction`

### Pacote

O pacote `refusal_direction_replication` terá módulos pequenos e separados:

- `runtime.py`: dispositivo, dtype, seed e metadados de hardware;
- `data.py`: carregamento, filtragem e splits nocivo/benigno;
- `model.py`: carregamento do modelo e aplicação do chat template;
- `directions.py`: médias residuais e candidatos por camada/posição;
- `interventions.py`: adição e ablação direcional via hooks;
- `metrics.py`: recusa, falsa recusa, segurança opcional e perda de linguagem;
- `artifacts.py`: schemas, hashes, configuração e proveniência.

O código oficial será tratado como fonte metodológica, com URL e commit fixos
registrados nos artefatos. Somente o código necessário será portado, após
verificação de licença e atribuição; o repositório upstream inteiro não será
vendorizado.

### Dados

Para manter a definição do artigo:

- treino: 128 instruções nocivas e 128 benignas;
- validação: 32 instruções nocivas e 32 benignas;
- nocivas: combinação documentada de AdvBench, MaliciousInstruct, TDC2023 e
  HarmBench;
- benignas: Alpaca;
- avaliação causal: amostras separadas das usadas na extração e seleção.

Os splits e filtros serão determinísticos, com seed, hashes e IDs persistidos.
Dados protegidos por termos ou com conteúdo sensível não serão copiados para
o Git. Apenas manifestos e métricas agregadas serão versionados. Outputs brutos
potencialmente nocivos ficarão sob `artifacts/`, ignorado pelo Git.

### Extração e seleção

Para cada camada `l` e posição pós-instrução `i`, o candidato será:

```text
r[l,i] = média_residual(nocivo, l, i)
       - média_residual(benigno, l, i)
```

Os candidatos serão avaliados na validação por três critérios:

1. ablação reduz recusa em prompts nocivos;
2. adição aumenta recusa em prompts benignos;
3. a intervenção preserva coerência e mantém pequena a alteração de
   cross-entropy em texto benigno.

A direção escolhida, sua norma, camada, posição, seed e métricas de seleção
serão salvas em um artefato JSON acompanhado do tensor `.pt`.

### Intervenções

A adição será aplicada na camada de origem, em todas as posições:

```text
x' = x + r
```

A ablação direcional será aplicada em todas as camadas e posições usando a
direção unitária:

```text
x' = x - (x · r_hat) r_hat
```

As intervenções serão testadas contra baseline e controles de direção
aleatória de mesma norma. O resultado principal deve distinguir:

- recusa em prompts nocivos;
- falsa recusa em prompts benignos;
- perda de linguagem/coerência;
- alteração de cross-entropy.

## Notebooks

Os notebooks serão finos: apresentam o raciocínio e chamam funções testadas do
pacote.

1. `01_dataset_and_baseline.ipynb`: dados, chat template e baseline de recusa;
2. `02_extract_candidate_directions.ipynb`: diferença de médias por
   camada/posição;
3. `03_causal_interventions.ipynb`: adição, ablação e controle aleatório;
4. `04_evaluation_and_validation.ipynb`: recusa, falsa recusa, CE loss,
   incerteza e comparação com o artigo.

Cada notebook grava um artefato compacto em `results/` e pode ser executado
localmente ou pelo Colab CLI.

## Colab CLI

Será usada a CLI oficial `google-colab-cli`, instalada localmente com:

```text
uv tool install google-colab-cli
```

A autenticação será um passo manual único via Application Default
Credentials, incluindo os escopos `openid`, `cloud-platform`,
`userinfo.email` e `colaboratory`. Tokens e arquivos de credenciais nunca
serão lidos, copiados ou versionados pela automação.

`tools/colab/run_refusal.py`, implementado apenas com a biblioteca padrão,
orquestrará:

1. `probe`: verificar identidade, cota e hardware disponível;
2. `start`: criar sessão nomeada, com GPU explicitamente solicitada;
3. `sync`: compactar e enviar somente `projects/refusal_direction`;
4. `bootstrap`: instalar o projeto no runtime;
5. `execute`: rodar um notebook e preservar seu output local;
6. `fetch`: baixar resultados agregados e logs;
7. `stop`: encerrar explicitamente a sessão.

Arquivos múltiplos serão enviados como um arquivo compactado único. Isso evita
depender de semântica não documentada de upload recursivo. Scripts Python
remotos farão extração, instalação e compactação dos resultados; comandos
interativos como `colab drivemount`, `colab auth`, `colab repl` e `colab
console` não farão parte do fluxo automatizado.

A sessão padrão será `refusal-gemma`. A GPU recomendada será L4, com T4 como
fallback explícito para Gemma 2B. A100 poderá ser solicitada manualmente. Não
haverá fallback silencioso nem criação de múltiplas sessões. Um bloco
`try/finally` e um comando de recuperação separado garantirão que `colab stop`
continue disponível após falhas.

## Reprodutibilidade

Cada artefato registrará:

- modelo e revisão Hugging Face;
- commit da metodologia upstream;
- versões de Python, PyTorch e Transformers;
- dtype, dispositivo e GPU;
- seeds e hashes dos splits;
- camada e posição da direção;
- fórmula e escopo de cada intervenção;
- tamanho de amostra e intervalos de confiança;
- commit local e estado dirty do Git.

Os resultados locais e do Colab usarão o mesmo schema. Diferenças de hardware
serão documentadas, não mascaradas.

## Testes

Os testes rápidos não exigirão GPU nem download de modelos:

- construção e separação dos splits;
- cálculo de diferença de médias;
- projeção e ablação vetorial;
- instalação e remoção correta dos hooks;
- métricas de recusa e falsa recusa;
- serialização e validação dos artefatos;
- geração dos comandos Colab sem segredos;
- falha segura quando sessão, GPU ou output estiver ausente.

Um teste de integração opcional usará um modelo minúsculo. Os notebooks terão
testes estruturais e uma execução GPU separada.

## Critérios de conclusão

A primeira fase estará concluída quando:

1. a suíte IOI continuar passando em `projects/ioi`;
2. a nova suíte rápida passar sem GPU;
3. Gemma 2B produzir baseline, direção selecionada e intervenções causais;
4. a ablação reduzir a recusa nociva e a adição aumentar a falsa recusa em
   comparação ao baseline e ao controle aleatório;
5. CE loss e coerência mostrarem que o efeito não é apenas degradação global;
6. artefatos e notebooks puderem ser reproduzidos em uma sessão Colab nomeada
   e encerrada explicitamente;
7. README registrar comandos, limitações e diferenças em relação ao artigo.

## Segurança e limites interpretativos

A replicação é defensiva e estuda a fragilidade de mecanismos de recusa. Não
serão automatizadas otimizações de jailbreak nem publicadas respostas nocivas
brutas. Uma direção que controla linguagem de recusa não será descrita como
“componente do alinhamento” sem evidência adicional sobre detecção de risco,
representação de política e comportamento fora da distribuição.
