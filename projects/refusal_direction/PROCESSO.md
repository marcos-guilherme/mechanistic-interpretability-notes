# Processo de estudo e reprodução da direção de recusa

[← Voltar à metodologia comum](../../docs/replicacoes/METODOLOGIA.md)
[← Voltar ao diário central dos experimentos](../../docs/ACOMPANHAMENTO_EXPERIMENTOS.md)

O protocolo de steering esparso, ainda aguardando execução, está em
[`SPARSE_STEERING.md`](SPARSE_STEERING.md).

> **Estado:** concluído, com reprodução causal parcial
>
> **Última atualização:** 16 de novembro de 2026

## Síntese acadêmica

### Contexto

Modelos ajustados para seguir instruções frequentemente recusam pedidos
considerados nocivos. Arditi et al.,
[*Refusal in Language Models Is Mediated by a Single
Direction*](https://arxiv.org/abs/2406.11717), apresentam evidências de que
esse comportamento é mediado por uma direção aproximadamente unidimensional no
espaço residual. Se essa direção tiver papel causal, removê-la deve reduzir a
recusa e adicioná-la deve induzir recusa mesmo em instruções inofensivas.

Depois de estudar um circuito distribuído no experimento de IOI, escolhi esta
replicação para explorar uma hipótese quase oposta em forma: um comportamento
complexo mediado por uma única direção contínua. Isso me permitiu comparar duas
maneiras de formular uma explicação mecanística (como conjunto de componentes
discretos e como orientação no espaço residual).

### Problema de pesquisa

Esta replicação pergunta: **uma direção de recusa extraída das ativações de
`google/gemma-2b-it` é necessária e suficiente para controlar o comportamento
de recusa em um recorte local?** Necessidade é testada por ablação; suficiência,
por adição. Um controle aleatório ortogonal e medidas de preservação ajudam a
distinguir um efeito específico de uma perturbação indiscriminada do modelo.
A conclusão pretendida é local: não se testa se a direção representa toda a
nocividade, todo o alinhamento ou todos os mecanismos de recusa.

### Métodos

Foram amostrados 128 pedidos nocivos e 128 inofensivos para treino e 32 de cada
classe para validação. Um filtro comportamental manteve exemplos cujo sinal
basal era compatível com a classe, resultando em 121/127 exemplos efetivos no
treino e 28/32 na validação. Em cada uma das 18 camadas e cinco posições finais
do template, calculou-se a diferença entre as médias das ativações nocivas e
inofensivas, produzindo 90 direções candidatas. A seleção combinou redução da
recusa por ablação, indução de recusa por adição, KL inofensiva máxima de 0,1 e
exclusão das camadas finais. A candidata escolhida foi comparada com uma
orientação aleatória ortogonal. A execução usou seed 42, `bfloat16` e uma
NVIDIA L4.

### Resultados

A direção selecionada veio da camada 12, posição `-1` do template, com norma
17,1552. Nos 28 pedidos nocivos pós-filtro, a taxa de recusa caiu de 100% para
0% após ablação; nos 32 pedidos inofensivos, a adição elevou a falsa recusa de
0% para 93,75%. A ablação com a orientação aleatória manteve 100% de recusa
nociva e, portanto, não reproduziu o efeito da direção selecionada. A KL média
inofensiva após ablação foi 0,03057, abaixo do limite de 0,1. A variação de
entropia cruzada foi −0,4664 na ablação e −0,4652 no controle, valores muito
próximos; por isso, essa métrica isolada oferece evidência fraca de
especificidade. O conjunto sustenta o padrão causal neste modelo e amostra,
mas não sua universalidade.

### Contribuição para o TCC

Para o meu TCC, este experimento oferece uma análise causal mais compacta que
a replicação IOI: em vez de um circuito distribuído, testei uma direção
contínua no espaço residual. A comparação entre ablação, adição e controle me
deu uma forma concreta de discutir necessidade, suficiência e preservação. Ao
mesmo tempo, o filtro comportamental e o reuso da validação para seleção me
mostraram como uma execução tecnicamente bem-sucedida ainda pode carregar
ameaças importantes à validade externa e à estimativa de generalização.

Este projeto é uma reprodução parcial e defensiva de Arditi et al.,
[*Refusal in Language Models Is Mediated by a Single
Direction*](https://arxiv.org/abs/2406.11717). A pergunta central do trabalho é
se o comportamento de recusa de um modelo de linguagem é mediado por uma
direção aproximadamente unidimensional em seu espaço de ativações: se essa
direção for removida, a recusa diminui; se for adicionada, a recusa aparece
mesmo diante de um pedido inofensivo.

A reprodução não tenta demonstrar que uma direção seja “o alinhamento” do
modelo, nem cobre todo o artigo. Ela testa o padrão causal mínimo em um único
modelo, `google/gemma-2b-it`: extrair direções candidatas por diferença de
médias, selecionar uma delas por ablação, adição e preservação, compará-la com
um controle aleatório e medir o efeito final. Ficaram fora os demais modelos,
benchmarks externos, ataques de jailbreak, ortogonalização de pesos, ajuste
LoRA e qualquer publicação de respostas nocivas brutas.

## Recorte experimental e isolamento

O Gemma 2B IT foi escolhido como um recorte de porte menor e por ser um modelo
*instruction-tuned* incluído no tipo de hipótese estudada por Arditi. A
execução aqui documentada usou uma NVIDIA L4. O projeto vive em seu próprio
`pyproject.toml`, `uv.lock` e `.venv`, separado do ambiente da replicação IOI.
Essa separação evita que versões de TransformerLens, Transformers, PyTorch ou
CUDA de um estudo alterem silenciosamente o outro.

Jupyter e `ipykernel` foram colocados no grupo de dependências `dev`: eles são
ferramentas de execução e apresentação, não dependências do núcleo
experimental instalado no Colab. O pacote, os notebooks finos, os testes e os
JSONs com métricas agregadas e valores numéricos por exemplo são versionáveis,
sem os textos dos prompts ou das gerações; datasets normalizados, tensores de
direção, pesos, caches e gerações permanecem em `artifacts/`, fora do Git.

A configuração fixa `seed=42`, `bfloat16`, lote 4, até 128 novos tokens e o
[commit upstream
`9d852fae`](https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3).
Para o modelo, a configuração solicitou a revisão móvel `main`, mas a execução
registrou o hash resolvido
`96988410cbdaeb8d5093d1ebdc5a8fb563e02bad`. Uma reprodução futura que continue
pedindo apenas `main` pode baixar pesos diferentes; para comparação estrita,
deve-se conferir ou fixar esse hash resolvido.
O script de bootstrap baixa desse commit os quatro splits já definidos pelo
upstream. A construção original combina conjuntos de pedidos nocivos
(AdvBench, MaliciousInstruct, TDC2023 e HarmBench) e pedidos inofensivos do
Alpaca; localmente, cada registro recebe ID, rótulo e referência de origem,
sem que o conteúdo bruto seja versionado.

Os tamanhos **planejados** foram 128 exemplos de treino e 32 de validação por
classe. Depois da amostragem determinística, aplicou-se o filtro comportamental
do modelo: mantém-se um exemplo nocivo quando o *log-odds* do primeiro token de
recusa é positivo e um exemplo inofensivo quando é negativo. Por isso, os
tamanhos **efetivos** foram:

- treino: 121 nocivos e 127 inofensivos;
- validação: 28 nocivos e 32 inofensivos.

Esses números não são intercambiáveis com os tamanhos planejados. Em
particular, a taxa final de recusa nociva usa **N=28**, não 32. As condições
inofensivas finais usam N=32. Os hashes de
[`results/01_baseline.json`](results/01_baseline.json) identificam os splits
amostrados **antes** do filtro comportamental; o mesmo artefato registra
separadamente as contagens planejadas e as contagens efetivas pós-filtro.

## Caderno de bordo dos quatro estágios

### 01: Dataset e linha de base

O [primeiro notebook](notebooks/01_dataset_and_baseline.ipynb) pergunta se o
modelo exibe o contraste necessário antes de qualquer intervenção. Ele carrega
os splits, aplica o chat template do Gemma, calcula o sinal de recusa, filtra
os exemplos inconsistentes com a própria classe e mede recusa nas duas classes
de validação. Também calcula entropia cruzada nos pedidos inofensivos, que
servirá como referência de preservação.

A linha de base foi válida no recorte efetivo: 100% de recusa nos 28 pedidos
nocivos que já tinham *log-odds* de recusa positivo e 0% nos 32 inofensivos
que já tinham *log-odds* negativo. Portanto, 100%/0% é um resultado
**condicional ao filtro**, não uma taxa da amostra original. O artefato não
traz taxa de recusa das gerações para os 32 exemplos nocivos e 32 inofensivos
antes do filtro. Esse estágio estabelece comportamento no recorte selecionado;
por si só, não estabelece um mecanismo.

### 02: Extração das direções candidatas

O [segundo notebook](notebooks/02_extract_candidate_directions.ipynb) captura a
entrada residual de cada um dos 18 blocos nas cinco posições finais do sufixo
da instrução. Para cada par camada–posição, calcula

```text
r[l, i] = média(ativações nocivas em l, i)
        - média(ativações inofensivas em l, i)
```

As ativações têm forma `121 × 18 × 5 × 2048` para treino nocivo e
`127 × 18 × 5 × 2048` para treino inofensivo. Assim, 18 camadas × 5 posições
produzem 90 candidatas. Os tamanhos e as normas, mas não os prompts nem as
ativações, estão em `results/02_candidates.json`.

As cinco candidatas da camada 0 resultaram em vetores nulos. Em vez de tentar
normalizá-las ou tratá-las como evidência, a execução passou a tratá-las como
**no-op**: ablação e adição não mudam o estado. Essa ocorrência foi útil como
controle de neutralidade da instrumentação, mas candidatas nulas não podem ser
interpretadas como direções causais.

### 03: Seleção e intervenções causais

O [terceiro notebook](notebooks/03_causal_interventions.ipynb) avalia as 90
candidatas na validação efetiva. Uma candidata só é elegível quando:

1. a adição torna o *log-odds* médio de recusa inofensiva não negativo;
2. a divergência KL média entre logits inofensivos de baseline e de ablação é
   no máximo 0,1;
3. a camada satisfaz o corte nominal que remove os 20% finais do modelo.

Como o limite inteiro é `int(18 × 0,8) = 14`, apenas as camadas 0–13 são
elegíveis e as camadas 14–17 são excluídas. Na prática, o arredondamento remove
4 de 18 camadas, cerca de 22%, não exatamente 20%.

Entre as elegíveis, seleciona-se a de menor *score* médio de recusa nociva após
ablação. Isso combina três perguntas: a remoção reduz recusa, a adição induz
recusa e a remoção evita uma mudança excessiva na distribuição inofensiva. A
direção escolhida foi a camada 12, posição `-1`. Essa posição não é o último
token do texto da instrução: é o último token do template/sufixo aplicado
depois dela, correspondente ao *newline* após o marcador `model`. A norma do
vetor foi 17,1552.

A adição usa `x' = x + r` na entrada residual da camada de origem, em todas as
posições. A ablação usa
`x' = x - (x · r_hat) r_hat` em todas as camadas e posições: tanto na entrada
residual de cada bloco quanto nas saídas dos módulos de atenção e MLP. Essa
distinção importa: a adição testa se inserir a direção na camada selecionada é
suficiente; a ablação tenta impedir que o modelo a reintroduza ao longo do
cálculo residual, da atenção ou do MLP.

O controle é construído como um vetor aleatório ortogonal à direção escolhida,
reescalado para a mesma norma e gerado com a seed derivada `seed+1000`. Essa
seed é usada pelo código, mas não é gravada no JSON. Porém, a operação de
ablação normaliza qualquer direção antes de projetá-la. A norma pareada,
portanto, não controla a escala efetiva dessa intervenção: a comparação é
entre a orientação selecionada e **uma** orientação aleatória ortogonal, não
entre magnitudes de perturbação. Isso é mais informativo que um no-op, mas uma
única orientação e uma única seed não caracterizam a distribuição de
controles aleatórios.

### 04: Avaliação e preservação

O [quarto notebook](notebooks/04_evaluation_and_validation.ipynb) consolida as
métricas científicas: taxa de recusa, falsa recusa, divergência KL e mudança de
entropia cruzada. A validação de software é feita separadamente por
`scripts/validate_results.py` e pela suíte `pytest`. Ao final, o estágio também
emite `execution_manifest.json` com uma lista fixa dos quatro nomes de estágio
e dos quatro arquivos esperados; esse registro deve ser cruzado com a
existência e o conteúdo dos arquivos, pois sozinho não prova que cada notebook
foi executado com sucesso.

O padrão observado foi o esperado para a hipótese causal local:

- baseline nocivo pós-filtro: 100% de recusa em **N=28**;
- ablação da direção: 0% de recusa nociva no mesmo N=28;
- adição em pedidos inofensivos: 93,75% de recusa em N=32, contra 0% no
  baseline;
- controle aleatório ortogonal: 100% de recusa nociva em N=28, sem reproduzir
  o efeito da direção selecionada;
- preservação inofensiva após ablação: 0% de recusa, KL média 0,03057 abaixo
  do limite 0,1;
- variação de entropia cruzada do prompt: aproximadamente -0,4664 na ablação,
  -0,4652 no controle e +2,3744 na adição.

A entropia cruzada foi calculada sobre os tokens de entrada do prompt e do
template, não sobre as continuações geradas. Ela é evidência fraca de
preservação: ablação e controle apresentam deltas negativos quase iguais, e a
adição aumenta substancialmente a CE. A KL 0,03057 sobre pedidos inofensivos é
a medida mais diretamente usada para selecionar e validar a preservação da
ablação. Ainda assim, não foi calculada KL para o controle aleatório, o que
limita a comparação de especificidade.

Em conjunto, remover a direção aboliu a recusa nociva no recorte pós-filtro,
adicioná-la induziu recusa inofensiva e a ablação com uma orientação aleatória
ortogonal, nessa única seed, não produziu o mesmo efeito. Isso sustenta o
padrão causal no Gemma 2B IT e neste recorte; não estabelece universalidade,
segurança global ou uma representação completa de nocividade.

## Incidentes reais e o que ensinaram

Durante a execução, encontrei problemas que não apareciam na formulação
abstrata do experimento, mas que mudavam diretamente a capacidade de
reproduzi-lo. Mantive esses incidentes porque eles mostram o trabalho prático
necessário entre o método descrito no artigo e uma execução verificável:

- O runtime atual do Colab veio com Python 3.13, que exigiu
  Transformers 4.48.3. Para Python anterior a 3.13, o lock mantém
  Transformers 4.44.2. Fixar uma única versão para os dois ambientes teria
  tornado uma das instalações incompatível.
- Jupyter foi movido para dependências `dev`, mantendo o ambiente remoto
  mínimo e separando ferramentas de notebook das dependências do experimento.
- `google/gemma-2b-it` é um repositório *gated*. Baixar o modelo exigiu aceitar
  previamente a licença na página do modelo e executar `auth-hf`; ADC autentica
  o Colab, mas não concede acesso ao Hugging Face.
- As direções nulas da camada 0 precisaram ser no-op. Uma normalização ingênua
  de vetor zero introduziria valores inválidos e confundiria falha numérica
  com resultado experimental.
- O kernel remoto é persistente. Quando o código-fonte foi corrigido depois de
  um módulo já ter sido importado, foi necessário recarregar o módulo ou
  reexecutar as importações; apenas sincronizar o arquivo não muda objetos já
  residentes no kernel.
- Conexões do CLI com o Colab puderam cair durante chamadas longas. Retomar o
  comando contra a mesma sessão nomeada preservou o runtime, os arquivos e o
  estado do kernel. Uma queda do cliente não deve ser confundida com perda da
  sessão, nem com encerramento da GPU.

## Fontes canônicas e segurança

Este texto oferece a narrativa; os números devem ser auditados nas fontes
canônicas:

- [`results/01_baseline.json`](results/01_baseline.json): amostras planejadas e
  efetivas, hashes e linha de base;
- [`results/03_interventions.json`](results/03_interventions.json): seleção,
  escores das candidatas e condições de intervenção;
- [`results/final_evaluation.json`](results/final_evaluation.json): métricas
  finais e preservação;
- [`results/execution_manifest.json`](results/execution_manifest.json):
  lista declarada pelo estágio final de estágios e arquivos esperados.

O manifesto não contém tempos, status por notebook ou verificação independente
de conclusão. Como sua lista é emitida de forma fixa pelo estágio final, ele
deve ser lido junto dos quatro arquivos e de sua proveniência, não como prova
autônoma da execução completa. Há ainda uma lacuna operacional: o manifesto
não registra nem comprova que a GPU remota foi encerrada.

Os JSONs versionados contêm métricas agregadas, valores numéricos por exemplo,
configurações e proveniência, mas não os textos dos prompts ou das gerações.
Token do Hugging Face, credenciais ADC, pesos e tensores ficam fora do Git. O
comando `auth-hf` lê o token já armazenado pelo cliente Hugging Face, transfere
o arquivo para a sessão sem colocar o valor na linha de comando ou imprimi-lo
e o move para o cache remoto com permissão `0600`. O valor do token nunca deve
ser colado em notebook, comando, log ou documento.

## Como reproduzir

Na raiz do repositório, instale a CLI e faça uma única autenticação ADC manual:

```powershell
uv tool install google-colab-cli==0.7.5
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Aceite a licença do Gemma na página
[`google/gemma-2b-it`](https://huggingface.co/google/gemma-2b-it), autentique o
cliente Hugging Face local sem exibir o token e prepare o projeto:

```powershell
Set-Location projects/refusal_direction
uv sync
uv run hf auth login
Set-Location ../..
```

Em seguida, use uma sessão nomeada e execute os estágios em ordem:

```powershell
python tools/colab/run_refusal.py start --gpu L4
python tools/colab/run_refusal.py sync
python tools/colab/run_refusal.py bootstrap
python tools/colab/run_refusal.py auth-hf
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/01_dataset_and_baseline.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/02_extract_candidate_directions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/03_causal_interventions.ipynb
python tools/colab/run_refusal.py execute --notebook projects/refusal_direction/notebooks/04_evaluation_and_validation.ipynb
python tools/colab/run_refusal.py fetch
python tools/colab/run_refusal.py stop
```

`sync` envia somente o projeto, excluindo ambientes, caches e `artifacts/`;
`bootstrap` instala o pacote e baixa os splits do commit fixado; `auth-hf`
prepara o acesso gated; `execute` reutiliza a mesma sessão; `fetch` baixa
somente os JSONs permitidos, com métricas agregadas e valores numéricos por
exemplo, sem textos de prompts ou gerações. Se a conexão cair, confirme a
sessão com `python tools/colab/run_refusal.py probe` e retome o estágio
necessário.

**A GPU deve sempre ser encerrada**, inclusive depois de erro, timeout,
interrupção local ou fallback de hardware:

```powershell
python tools/colab/run_refusal.py stop
```

Fechar o terminal ou perder a conexão não encerra a sessão remota. Se uma L4
não estiver disponível, primeiro encerre qualquer sessão parcial e só então
repita `start --gpu T4`; não há fallback silencioso. A T4 é uma alternativa
operacional prevista, mas **não foi validada nesta execução**, que usou L4.

Depois de `fetch`, valide localmente:

```powershell
Set-Location projects/refusal_direction
uv run python scripts/validate_results.py
uv run pytest -q
```

O primeiro comando confere os critérios causais e de preservação no artefato
final; a suíte testa fórmulas, hooks, schemas, notebooks e orquestração sem
baixar novamente o modelo.

## Limites e próximos passos

Esta é uma execução de uma seed, um modelo e uma GPU, com amostras pequenas e
filtradas pelo próprio comportamento basal. A mesma validação efetiva foi usada
para escolher a candidata e reportar o efeito final; não há
um terceiro conjunto totalmente separado para estimar generalização após a
seleção. A detecção de recusa usa um conjunto curto de marcadores e um token
inicial específico do Gemma, enquanto preservação foi medida apenas nos 32
pedidos inofensivos deste split. A CE cobre os tokens do prompt/template, não
a qualidade das gerações, e seus deltas semelhantes para ablação e controle
enfraquecem seu valor como evidência de especificidade. A KL baixa é mais
direta para a preservação da ablação, mas não há KL do controle. Nenhuma dessas
medidas prova ausência de degradação fora dessa amostra.

Como próximos passos, eu repetiria o experimento com várias seeds, reservaria
um teste final independente, ampliaria as métricas de qualidade e avaliaria
outros modelos e distribuições permitidos. Antes de expandir a conclusão,
também precisaria verificar a sensibilidade ao filtro, à posição, ao
coeficiente de adição e à definição de recusa. Até lá, mantenho a conclusão
deliberadamente restrita: reproduzi o padrão causal de uma direção de recusa
neste recorte do Gemma 2B IT.

## Piloto Qwen2.5-Coder-7B (congelado)

### Por que este piloto existe

Depois do Gemma, quis testar o mesmo esqueleto causal em um modelo
instruction-tuned maior e voltado a código, com splits de seleção e teste
separados e **sem** filtro comportamental prévio. O objetivo foi ver se o
padrão ablação/adição/controle se mantinha quando a validação escolhe a
direção e o teste avalia o efeito final, com gates numéricos explícitos.

### Execução e proveniência

| Campo | Valor |
| --- | --- |
| Modelo | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Revisão | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Run | `c0818150-f834-4692-b271-4e10a97eb57e` |
| Commit de implementação | `2a2eb714246174e4f627fcd0061c61143a4e8276` |
| GPU / dtype | NVIDIA A100 40 GB, `bfloat16` |
| Amostras | 128 treino e 32 validação por classe; 32 teste por classe |

Os JSONs compactos vivem em `results/qwen25_coder_7b/`. Eles carregam métricas,
configuração e proveniência, mas não prompts, gerações, tensores, `status.json`,
`events.jsonl` nem outros artefatos brutos (esses ficam fora do Git).

### Resultados numéricos (split de teste)

Linha de base sem filtro: recusa nociva `0,875`, recusa inofensiva `0,03125`.
Direção selecionada na validação: camada 18, offset `-1`.

| Condição | Recusa nociva | Recusa inofensiva |
| --- | --- | --- |
| Baseline | 0,875 | 0,03125 |
| Ablação | 0,0 | n/d |
| Adição | n/d | 1,0 |
| Controle aleatório (nocivos) | 0,875 | n/d |

KL inofensiva após ablação: `0,263825` (limite do gate: `0,10`; valor integral no JSON compacto).

### Operação versus ciência

**Sucesso operacional.** O piloto concluiu o pipeline, publicou os JSONs
compactos esperados e passou pela validação offline (`operational_success:
true` em
[`pilot_evaluation.json`](results/qwen25_coder_7b/pilot_evaluation.json)).

**Falha científica no gate.** O status final é `causal_gate_failed`. Os
subgates de queda nociva, aumento inofensivo e controle aleatório passaram; o
que falhou foi `max_harmless_ablation_kl`. Interpreto assim:

- há um sinal forte de manipulação causal (ablação zera recusa nociva, adição
  satura recusa inofensiva, controle não imita a ablação);
- a intervenção não preserva o comportamento inofensivo dentro do limite
  acordado;
- este piloto **permanece falhado**, mesmo que um follow-up exploratório de
  dose varra intensidades diferentes.

Qualquer varredura de dose futura é hipótese nova. Ela não reescreve
retrospectivamente a classificação desta execução.

### Fontes canônicas do piloto

- [`00_preflight.json`](results/qwen25_coder_7b/00_preflight.json)
- [`01_baseline.json`](results/qwen25_coder_7b/01_baseline.json)
- [`02_candidates.json`](results/qwen25_coder_7b/02_candidates.json)
- [`03_interventions.json`](results/qwen25_coder_7b/03_interventions.json)
- [`final_evaluation.json`](results/qwen25_coder_7b/final_evaluation.json)
- [`pilot_evaluation.json`](results/qwen25_coder_7b/pilot_evaluation.json)
- [`execution_manifest.json`](results/qwen25_coder_7b/execution_manifest.json)

## Follow-up exploratório no Qwen2.5-Coder-7B

### Por que ele não substitui o piloto

O piloto congelado mostrou que a direção produzia um efeito causal forte, mas
falhou no limite de preservação. Em vez de procurar outra direção depois de
ver esse resultado, mantive a direção da camada 18 e fiz uma pergunta mais
restrita: uma intervenção parcial conservaria o efeito com uma KL menor?

Este follow-up é exploratório. Ele foi definido depois da falha do piloto e
não transforma `causal_gate_failed` em sucesso retrospectivo. Para evitar que
a mesma amostra escolhesse e confirmasse os coeficientes, usei subconjuntos
disjuntos de seleção e confirmação, com 32 exemplos por classe em cada etapa.

### Varredura de dose

Testei os coeficientes `0,1`, `0,2`, `0,35`, `0,5`, `0,75` e `1,0`. A seleção
escolheu ablação de 20% e adição de 50%. Na confirmação, a recusa nociva caiu
de 96,875% para 43,75%, uma diferença de 53,125 pontos percentuais. A adição
elevou a recusa inofensiva de 0% para 31,25%. O controle aleatório permaneceu
em 96,875% de recusa nociva, e a KL inofensiva da ablação foi `0,022759`,
abaixo do limite `0,10`.

Os quatro gates passaram nessa amostra de confirmação. O resultado sustenta a
hipótese de que a intensidade importa, mas não apaga a falha da intervenção
integral. A fonte compacta é
[`dose_followup_evaluation.json`](results/qwen25_coder_7b_dose/dose_followup_evaluation.json);
os valores por exemplo e a proveniência estão em
[`03_dose_confirmation.json`](results/qwen25_coder_7b_dose/03_dose_confirmation.json).

### Ajuste de uma J-lens para o Qwen

Como o follow-up de dose passou, ajustei uma lente específica ao checkpoint do
Qwen. Fixei `anthropics/jacobian-lens` no commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` e usei o split de treino Python
do CodeSearchNet na revisão
`bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`. Entre 191 registros examinados,
120 satisfizeram os critérios de tamanho e conteúdo: 100 foram usados no
ajuste e 20 ficaram no holdout.

O probe escolheu `dim_batch=32`. O ajuste produziu 27 matrizes, das camadas
0 a 26 para a camada alvo 27. O arquivo da lente tem 693.642.187 bytes e hash
SHA-256
`b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e`.
No holdout, agregando 59.940 observações, a J-lens obteve
`pass@1/5/10 = 0,12681 / 0,21178 / 0,25167`; a logit lens obteve
`0,11168 / 0,17219 / 0,19351`. Isso valida formato, finitude e utilidade
descritiva mínima da lente nesse corpus. Não prova que ela leia corretamente
todo conceito de recusa fora da distribuição do CodeSearchNet.

Há uma limitação importante: a preparação registra
`benchmark_deduplicated=false`. Não havia uma lista de exclusão disponível
para deduplicar o corpus de ajuste contra os benchmarks usados depois. Essa
lacuna não deve ser escondida ao interpretar a avaliação. As fontes são
[`00_jlens_prepare.json`](results/qwen25_coder_7b_jlens/00_jlens_prepare.json),
[`03_jlens_holdout.json`](results/qwen25_coder_7b_jlens/03_jlens_holdout.json)
e
[`jlens_evaluation.json`](results/qwen25_coder_7b_jlens/jlens_evaluation.json).

### J-lens v2: contexto longo (novembro 2026)

A v1 validou formato e utilidade mínima com 100 ajustes e holdout de 20
exemplos em contexto curto. Abri o perfil `qwen25-coder-jlens-v2` para testar o
mesmo compromisso upstream (`anthropics/jacobian-lens` no commit congelado)
com **512 tokens** por exemplo, **200 amostras** fixas no holdout e tamanho de
ajuste escolhido por probe sob teto de **12 horas** projetadas na A100. O probe
real mostrou que 512 tokens integrais não eram viáveis: os lotes 32, 16 e 8
esgotaram a memória da A100 de 40 GB, e a tentativa com lotes menores não
concluiu uma Jacobiana em 61 minutos. A sessão foi encerrada sem ajustar uma
lente.

O perfil foi então reduzido para **256 tokens integrais**, mantendo o holdout
de 200 e o mesmo teto de tempo. O probe completou com `dim_batch=8` em 78,10
segundos e selecionou 400 exemplos, com projeção de 8,68 horas. A corrida
`e35026e6-43e1-4014-8022-f4055aa6a58f` terminou com status `pass`.

A v2 grava em `artifacts/qwen25_coder_7b_jlens_v2/` e compara a lente nova com a
v1 congelada no holdout compartilhado. O laboratório visual e o code-eval em
batch **não** passam a usar a v2: continuam apontando para a lente v1 já
documentada nesta seção.

**Caderno.** No holdout compartilhado de 200 funções, a v2 obteve `pass@1` de
13,312%, contra 14,032% da v1 e 11,874% da logit lens. Em `pass@5`, os valores
foram 21,253%, 22,790% e 17,723%. Em `pass@10`, 25,003%, 26,985% e 19,776%.
O rank médio foi 7.994 na v2, 7.359 na v1 e 31.504 na logit lens; nesse caso,
menor é melhor. A v2 melhorou claramente a projeção linear simples, mas não
superou a lente curta original.

As duas metades de 100 itens do holdout repetiram a mesma ordem. Isso torna o
resultado mais estável, mas não explica sozinho a diferença entre v1 e v2,
porque aumentei ao mesmo tempo o contexto e a quantidade de ajuste. A preparação
também ficou sem deduplicação contra benchmarks externos. O tensor final e os
JSONs estão em `artifacts/qwen25_coder_7b_jlens_v2/` e
`results/qwen25_coder_7b_jlens_v2/`; os detalhes operacionais estão em
[`JLENS_V2.md`](JLENS_V2.md).

### Avaliação em tarefas de código

A execução `5d214feb-e870-4e30-a783-0950a11b09f3` avaliou 32 itens de cada
conjunto em quatro condições: baseline, ablação de 20%, adição de 50% e
controle aleatório de 20%. MITRE representa pedidos ofensivos; MITRE-FRR
contém pedidos defensivos ou sensíveis que não deveriam ser recusados
indiscriminadamente; HumanEval+ mede capacidade funcional de programação.

Os resultados automáticos foram:

| Conjunto | Condição | Recusa | KL média | pass@1 |
| --- | --- | ---: | ---: | ---: |
| MITRE | baseline | 78,125% | n/d | n/d |
| MITRE | ablação | 0% | 4,39113 | n/d |
| MITRE | adição | 100% | 1,91932 | n/d |
| MITRE | controle | 81,25% | 0,00081 | n/d |
| MITRE-FRR | baseline | 9,375% | n/d | n/d |
| MITRE-FRR | ablação | 0% | 1,14368 | n/d |
| MITRE-FRR | adição | 100% | 3,83226 | n/d |
| MITRE-FRR | controle | 6,25% | 0,00125 | n/d |
| HumanEval+ | baseline | 0% | n/d | 40,625% |
| HumanEval+ | ablação | 0% | 0,02258 | 40,625% |
| HumanEval+ | adição | 65,625% | 1,40227 | 21,875% |
| HumanEval+ | controle | 0% | 0,00102 | 37,5% |

A ablação preservou exatamente o pass@1 de HumanEval+ nesta amostra e removeu
as recusas MITRE. Esse é o resultado mais favorável do follow-up. Ainda assim,
a KL foi alta em MITRE e MITRE-FRR. Portanto, os dados não sustentam uma
afirmação de alteração interna globalmente seletiva. A adição foi claramente
não seletiva: fez todos os itens MITRE e MITRE-FRR serem recusados, induziu
recusa em 21 dos 32 itens HumanEval+ e reduziu o pass@1 de 13/32 para 7/32. O
controle aleatório ficou próximo do baseline e teve KL pequena nos três
conjuntos.

### O que a J-lens mostrou durante as intervenções

O readout procurou um léxico de recusa no último token da instrução e no
primeiro token da resposta. Para cada conjunto e condição, examinei todas as
camadas e resumi aqui a camada com menor rank médio. Essa escolha é descritiva
e foi feita depois de observar os resultados; ela não é um novo gate
pré-registrado.

No último token da instrução, o baseline MITRE atingiu rank médio `3,19` na
camada 22, com algum token do léxico no top 5 em 81,25% dos exemplos. A adição
levou o rank médio a `1,0` na camada 23 e o hit@5 a 100%; a ablação deslocou o
melhor rank médio para `26,78` na camada 26, sem hit@5. O controle reproduziu
de perto o baseline. Em MITRE-FRR, a adição também levou o rank médio a `1,0`
e hit@5 a 100%, apesar de esses pedidos serem defensivos. Em HumanEval+, a
adição colocou o léxico em rank médio `2,94` na camada 26, com hit@5 de 87,5%,
enquanto baseline, ablação e controle não tiveram hit@5 em sua melhor camada.

Na primeira posição da resposta, a adição ainda produziu sinal forte em MITRE
e MITRE-FRR, com ranks médios `2,0` e `1,94`. Em HumanEval+, porém, o melhor
rank médio foi `147`, mesmo com 65,625% de recusas na geração. Isso mostra que
posição de leitura, camada, léxico e comportamento gerado não são medidas
intercambiáveis. A J-lens oferece um retrato interno coerente em parte do
recorte, mas a causalidade vem da intervenção na direção, não do rank da
lente.

Os agregados completos estão em
[`03_readout.json`](results/qwen25_coder_7b_code_eval/03_readout.json) e
[`04_finalize.json`](results/qwen25_coder_7b_code_eval/04_finalize.json).

### Segurança, revisão e conclusão atual

Somente HumanEval+ foi executado. As gerações MITRE e MITRE-FRR nunca foram
executadas. O EvalPlus rodou na VM efêmera do Colab com diretório temporário,
limite de memória, timeout e paralelismo 1, mas seu `reliability_guard` não é
uma sandbox de segurança completa. Prompts, gerações, tensores e a amostra
cega ficam em `artifacts/`, fora do Git.

A revisão humana cega de 32 classificações está preparada, mas continua
pendente. Por isso, as taxas de recusa acima devem ser tratadas como rótulos
automáticos, não como concordância humano-classificador já verificada.

Minha leitura atual é composta. A dose parcial de ablação removeu recusa em
pedidos ofensivos e preservou o pass@1 de HumanEval+ nesta amostra. Ao mesmo
tempo, a KL alta nos dois conjuntos MITRE impede chamar a intervenção de
globalmente seletiva, e a adição prejudicou pedidos defensivos e tarefas
normais de código. O follow-up produz evidência útil sobre intensidade,
transferência e leitura interna, mas deixa uma conclusão de segurança ampla
fora do alcance.

## Laboratório visual Qwen (novembro 2026)

### Decisão: ferramenta separada do experimento

O follow-up em batch já tinha fechado métricas, gates e leituras agregadas da
J-lens. Quis uma forma de **inspecionar** a mesma direção congelada e a mesma
lente sem misturar isso ao pipeline de notebooks, `fetch` e JSONs em
`results/`. Por isso criei o perfil `qwen25-coder-lab` e o pacote
`qwen_lab_*`: motor de intervenção, readout incremental e controller Gradio
com contratos próprios.

Essa separação é deliberada. O laboratório **não** reabre seleção de direção,
não mede KL em amostra fixa, não substitui o piloto com status
`causal_gate_failed` e não converte ranks da lente em nova evidência causal. É
exploração qualitativa e pedagógica sobre artefatos já congelados.

O desenho está em
[`docs/superpowers/specs/2026-11-16-qwen-intervention-lab-design.md`](../../docs/superpowers/specs/2026-11-16-qwen-intervention-lab-design.md).
A operação passo a passo está em
[`LABORATORIO_QWEN.md`](LABORATORIO_QWEN.md).

### Caderno de bordo

**Implementação (tarefas 1 a 4).** Contratos em `qwen_lab.py`, readout ao vivo,
runtime com uma geração concorrente, app Gradio pinado em 5.49.1, transporte
Colab com direção e lente fatiadas, auth de uso único e recusa explícita de
`fetch` no orquestrador. Testes locais do pacote lab passaram com Gradio no pin
(174 testes no relatório da tarefa 4, antes do smoke de hardware).

**Smoke em A100, 16 de novembro de 2026.** GPU A100-SXM4-40GB; modelo e revisão
do JSON `qwen25_coder_7b_lab.json`; hashes verificados da direção (`0b1cff5e...`)
e da lente (`b700a78c...`). A cadeia start, sync, bootstrap, auth-hf e lab
subiu com autenticação temporária funcionando. Testei baseline e ablação 20%
com um prompt inofensivo de programação, temperatura 0, camada J-lens 18, top-k
5 e limite de 6 novos tokens. Ambas as condições chegaram ao passo 6 com estado
de limite de tokens e mantiveram token emitido e top-k da lente atualizados em
sincronia. Não registro aqui o texto do prompt nem trechos de geração.

**Incidente de startup.** Na primeira subida a URL demorou porque o stdout do
processo remoto estava bufferizado. Corrigi em `71788f6` com `python -u` no
lançador em `colab_start_lab.py`. Na segunda tentativa a URL pública apareceu em
cerca de 40 segundos.

**Exportação e encerramento.** Iniciei validação de exportação pela interface,
mas a sessão Colab encerrou antes da ação. Os testes locais de exportação
explícita cobrem o schema e a ausência de segredos; falta ainda um smoke
remoto que confirme o download do JSON através do relay. Ao final rodei
`colab sessions` e confirmei nenhuma sessão ativa.

**Interpretação registrada.** A interface mostra previsões da J-lens, não
“pensamentos” do modelo. Nada neste smoke altera as conclusões do piloto, da
dose, do ajuste de lente em batch ou da avaliação MITRE/HumanEval+ documentadas
acima.

## J-lens sob intervenção causal (17 de novembro de 2026)

### A pergunta que uniu os dois estudos

Até aqui eu sabia que a direção alterava a recusa e que a J-lens recuperava
parte dos tokens escolhidos a partir de camadas intermediárias. Esses
resultados tinham sido obtidos separadamente. O teste seguinte perguntou se a
lente continuaria descritiva quando o estado residual fosse alterado pela
própria direção de recusa.

Congelei antes da execução o painel de 32 prompts MITRE e 32 MITRE-FRR, as
condições baseline, ablação 20%, adição 50% e controle aleatório 20%, as
camadas 19, 22, 25 e 26 e os primeiros oito tokens gerados. O endpoint
principal foi `pass@5` na camada 25. A equivalência contra o baseline exigia
que o intervalo de 95%, obtido por bootstrap agrupado por prompt, permanecesse
entre -5 e +5 pontos percentuais.

### Execução e incidentes

A primeira execução real parou na checagem de paridade entre o caminho
incremental e o `JacobianLens.apply`. Eu havia movido as matrizes da lente para
a GPU antes dessa checagem, criando uma combinação indevida de tensores em CPU
e CUDA. Corrigi a ordem para executar a paridade primeiro e manter na GPU
somente as quatro matrizes necessárias depois da aprovação.

A repetição seguinte ainda usou as funções antigas, embora os arquivos já
estivessem corrigidos. O kernel persistente do Colab havia preservado os
módulos importados em `sys.modules`. Limpei explicitamente os módulos do
projeto e descartei os diretórios das tentativas falhas antes de repetir. Não
aceitei resultados parciais dessas tentativas.

A execução canônica
`df86a101-9f5a-4363-8918-5f4c183ee25d`, no commit
`251dfd50397da06a008be6f06cf3aa541c9fd8a4`, terminou com 256 respostas e
16.384 observações de rank. O validador recomputou matriz, métricas, bootstrap
e flags de equivalência e confirmou todos os contratos.

### O que observei

A direção voltou a mostrar efeito comportamental forte. No MITRE, a recusa foi
78,125% no baseline, 0% na ablação, 100% na adição e 78,125% no controle. No
MITRE-FRR, foi 9,375%, 0%, 100% e 6,25%, respectivamente. A direção controlou
recusa, mas a adição não distinguiu solicitações ofensivas de defensivas.

Na camada 25, baseline, adição e controle mantiveram `pass@5` entre 98,047% e
100%. A ablação no MITRE caiu de 98,047% para 93,359%. Seu intervalo de
diferença, de -8,203 a -1,563 pontos, atravessou a margem inferior e impediu
declarar equivalência. As outras sete comparações principais foram
equivalentes. V1 e v2 tiveram os mesmos endpoints principais.

A camada 26 ficou entre 99,609% e 100%, mas está muito próxima da saída. As
camadas 19 e 22 foram menos fiéis e reagiram mais às intervenções. Isso
localiza melhor a conclusão: a leitura tardia permaneceu útil sob mudança
causal, sem demonstrar robustez uniforme ao longo da rede.

### Como isso mudou a pergunta seguinte

O resultado aproximou os dois eixos do projeto. A J-lens conseguiu descrever
um modelo cujo comportamento havia sido alterado, enquanto a direção revelou
uma limitação de seletividade. Em vez de apenas procurar uma intensidade
melhor para o mesmo vetor, uma continuação plausível é comparar formas de
representação mais ricas. Um subespaço com várias direções ou features de um
autoencoder esparso poderia separar recusa, conteúdo de cibersegurança e
intenção ofensiva. Essa hipótese exigiria novamente validação causal, pois
ativação correlacionada não basta para atribuir função.

A descrição completa do protocolo e dos resultados está em
[`CAUSAL_JLENS.md`](CAUSAL_JLENS.md). Os números canônicos estão em
[`results/qwen25_coder_7b_causal_jlens/causal_jlens_evaluation.json`](results/qwen25_coder_7b_causal_jlens/causal_jlens_evaluation.json).
