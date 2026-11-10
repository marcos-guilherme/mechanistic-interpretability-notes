# J-space no Llama 3.1 8B Instruct

O J-space foi o passo seguinte da minha exploração: em vez de analisar um
circuito ou uma direção comportamental, quis testar se seria possível ler e
alterar um conceito intermediário usado pelo modelo. Preparei um experimento
causal mínimo no `meta-llama/Llama-3.1-8B-Instruct`, usando uma J-lens já
ajustada e a troca `spider → ant`.

Planejei a execução para uma NVIDIA L4 no Colab, com acesso autenticado ao
modelo *gated* do Hugging Face. Pesos e lente são baixados apenas para caches
ignorados, enquanto o Git recebe somente JSONs agregados e compactos. A lente
pré-ajustada usada pelo projeto é
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`.

Implementei configuração fixada, validação de CUDA e runtime, readout e
intervenções com uso limitado de memória, controles, captura de proveniência e
um pipeline de artefatos chamável pelo notebook.
`notebooks/01_readout_and_causal_swap.ipynb` é a entrada fina para a execução
real. A suíte do repositório testa o software sem carregar o modelo de 8B.

Em [`PROCESSO.md`](PROCESSO.md), explico o contexto científico, o problema, os
métodos, o resultado inconclusivo e as limitações em formato voltado ao TCC.

## Fluxo no Colab L4

Instale a versão verificada da CLI:

```powershell
uv tool install google-colab-cli==0.7.5
```

Primeiro, obtenha acesso ao repositório *gated*
`meta-llama/Llama-3.1-8B-Instruct` com a mesma conta do Hugging Face usada no
CLI. Depois, salve o token no cache local padrão:

```powershell
uv run hf auth login
```

Na raiz do repositório, execute:

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

O runner usa autenticação ADC e a sessão `jspace-llama` por padrão. O arquivo
enviado contém apenas as entradas necessárias no runtime remoto, e o bootstrap
instala `uv==0.12.1` antes de exportar o ambiente fixado. Ele substitui
atomicamente `/content/jspace_llama_project`; esse nome, que não é o nome do
pacote, impede que o checkout em `/content` sombreie a instalação editável de
`jspace_llama` como *namespace package*.

O comando `fetch` aceita apenas `results/execution_manifest.json` e
`results/final_evaluation.json`, e exige que ambos existam. A execução valida o
`_output.ipynb` novo e ignorado que a colab-cli produz e o preserva para
diagnóstico. Uma falha deixa a sessão disponível para investigação, por isso é
necessário encerrá-la explicitamente. Se não houver L4, encerre a sessão
parcial antes de tentar `--gpu T4` ou `--gpu A100`.

## A execução que consegui realizar

Na execução `a116c3da-bb8f-4a31-a39a-4fd09dd6fcad`, consegui exercitar o
software, a L4 no Colab e o caminho de artefatos compactos até a resolução dos
ativos no Hugging Face. O fluxo resolveu a revisão do modelo
`0e9e39f249a16976918f6564b8830bc894c89659` e a revisão da lente
`d2527c3cdbaa60dd4b65cbca1604921f044e989a`, mas terminou como `inconclusive`,
com o código `asset_load_failed`, porque minha conta ainda não estava
autorizada para o repositório *gated* do modelo.

Os pesos não foram baixados. Portanto, não houve inferência, readout de
J-space ou intervenção causal. Não interpreto essa execução como evidência
positiva nem negativa sobre J-space no Llama. Registrei a integridade da lente
e a presença de `spider` no readout como `evaluated: false`, com medições
`null`, em vez de transformar ausência de medição em observação científica
falsa. A revisão do código-fonte foi
`bda81311ed758749598d9cd41fb7bbf0d5d3897a`; o campo `dirty` ficou `true`
porque já havia alterações fora deste projeto no momento do envio.

Mantive os dois artefatos compactos como registro dessa execução inconclusiva.
Para repetir o experimento, preciso primeiro obter acesso oficial ao modelo e
então refazer `auth-hf`, `execute`, `fetch` e a validação local. Uma execução
futura substituirá esse registro com valores efetivamente medidos; não
preencho retroativamente metadados que a tentativa original não produziu.

## Artefatos e validação

O pipeline real grava continuações brutas, prompts renderizados, slots do
workspace e outros diagnósticos apenas em
`artifacts/readout_and_causal_swap.json`, que é ignorado pelo Git. Os arquivos
baixados `results/execution_manifest.json` e
`results/final_evaluation.json` contêm somente evidência agregada compacta e
metadados de reprodutibilidade.

É possível validar localmente os artefatos baixados ou as fixtures sem carregar
o modelo de 8B:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_results.py
uv run pytest -q
```

O estado `success` exige proveniência limpa e verificada por digest, integridade
da lente e readout explicitamente avaliados e pelo menos um `alpha` que passe
por todos os critérios do prompt principal `main`. O parser deve identificar
unicamente oito no baseline e no no-op, uma resposta coerente diferente de seis
no controle aleatório e unicamente seis na intervenção-alvo, sem vazamento de
`ant` nem degradação.

Respostas negadas, conflitantes, ausentes, repetitivas ou excessivamente
incoerentes não passam. Falhas esperadas de runtime, proveniência incompleta,
estados não medidos ou critérios não satisfeitos produzem `inconclusive` com
motivo legível por máquina; não são tratados como evidência negativa. As
contagens das paráfrases são dados secundários de transparência, não uma
afirmação geral. Erros inesperados de programação são registrados da melhor
forma possível no manifesto e depois relançados.
