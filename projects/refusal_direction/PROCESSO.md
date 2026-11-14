# Processo de estudo e reprodução da direção de recusa

[← Voltar à metodologia comum](../../docs/replicacoes/METODOLOGIA.md)
[← Voltar ao diário central dos experimentos](../../docs/ACOMPANHAMENTO_EXPERIMENTOS.md)

> **Estado:** concluído, com reprodução causal parcial
>
> **Última atualização:** 10 de novembro de 2026

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
