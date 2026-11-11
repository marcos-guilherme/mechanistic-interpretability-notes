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

A primeira tentativa registrada, `a116c3da-bb8f-4a31-a39a-4fd09dd6fcad`,
parou antes da inferência porque minha conta ainda não tinha autorização para
o modelo. Depois que o acesso foi aprovado, encontrei uma segunda dificuldade:
o TorchAudio pré-instalado no Colab usava CUDA 13.0, enquanto o PyTorch do
ambiente usava CUDA 13.2. Como o experimento não usa áudio, removi essa
dependência incompatível no bootstrap e registrei a correção com um teste.

A execução canônica atual,
`20eabeef-17f5-49dc-a4a1-fcafc20a9533`, completou o fluxo em uma NVIDIA L4. O
modelo de 32 camadas e a lente de aproximadamente 1,04 GB foram carregados; o
hash SHA-256 da lente foi verificado como
`d5300a1575c0b5ad6e01b098c75adb072f3a6c246f184ce2fb35a3550fc8c68b`.
O experimento avaliou as camadas 12–28, os três prompts, as quatro condições e
`alpha=4.0`, com proveniência completa e código-fonte limpo.

O resultado continuou `inconclusive`, agora por razões científicas do próprio
protocolo (`primary_gates_failed`), e não por falha de acesso. No prompt
principal:

- o parser não encontrou uma resposta única equivalente a oito no baseline ou
  no no-op;
- o readout foi executado, mas não encontrou `spider`;
- o controle aleatório também não produziu uma resposta válida para o critério;
- a troca `spider → ant` não produziu seis de forma válida, vazou a palavra
  `ant` e apresentou repetição.

As duas paráfrases também não passaram. Assim, esta execução não demonstrou o
redirecionamento causal planejado, mas também não é uma refutação geral do
J-space: o baseline necessário para interpretar a intervenção não foi
estabelecido e o conceito-fonte não apareceu no readout. Os artefatos preservam
essa distinção em vez de converter critérios não satisfeitos em evidência
negativa mais ampla.

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
