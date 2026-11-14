# J-space no Llama 3.1 8B Instruct

O J-space foi o passo seguinte da minha exploração: em vez de analisar um
circuito ou uma direção comportamental, quis entender se seria possível ler e,
em uma etapa separada, alterar um conceito intermediário usado pelo modelo.
Este projeto agora reúne duas investigações que não devem ser confundidas: uma
tentativa causal mínima, já executada e inconclusiva, e um atlas observacional
executado e validado sobre 93 prompts. O primeiro resultado continua sendo
causalmente inconclusivo; o segundo descreve o que as lentes recuperam, sem
transformar essa leitura em evidência de mecanismo.

Na etapa causal, usei uma J-lens já ajustada no
`meta-llama/Llama-3.1-8B-Instruct` e planejei a troca `spider → ant`. Essa
execução e seus dois artefatos canônicos foram preservados sem alteração. O
atlas novo não repete a troca: ele amplia primeiro a observação do que a lente
expõe em um conjunto fixo de prompts, sem gerar texto e sem editar ativações.

Planejei a execução para uma NVIDIA L4 no Colab, com acesso autenticado ao
modelo *gated* do Hugging Face. Pesos e lente são baixados apenas para caches
ignorados. Na etapa observacional, o perfil de transporte traz os JSONs
compactos e as figuras canônicas, além de um sidecar bruto usado só na
validação local; esse sidecar continua ignorado pelo Git. A lente pré-ajustada
usada pelo projeto é
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`.

Implementei configuração fixada, validação de CUDA e runtime, readout e
intervenções com uso limitado de memória, controles, captura de proveniência e
um pipeline de artefatos chamável pelo notebook.
`notebooks/01_readout_and_causal_swap.ipynb` é a entrada fina para a execução
causal. A suíte do repositório testa o software sem carregar o modelo de 8B.

Em [`PROCESSO.md`](PROCESSO.md), explico o contexto científico, os métodos, o
resultado causal inconclusivo, o desenho do atlas e as limitações em formato
voltado ao TCC.

## O que o atlas observa

O conjunto de avaliação veio do repositório da Anthropic
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), na
revisão fixada `581d398613e5602a5af361e1c34d3a92ea82ba8e`. Copiei somente os
93 itens de `lens-eval-multihop`, sob a
[licença Apache 2.0](data/anthropic-jlens/LICENSE), e fixei também o SHA-256 do
arquivo local. Cada item contém `prompt`, `target` e uma lista de conceitos
intermediários esperados.

Nesta avaliação, envio o `prompt` bruto diretamente à lente e ao modelo, sem
aplicar o *chat template*. Isso preserva a formulação e a posição de leitura
definidas pela avaliação original: o último token do prompt é a posição
imediatamente anterior a `target`. O campo `target` serve para descrever e
auditar essa avaliação, mas não é concatenado ao prompt nem fornecido ao
modelo.

O atlas mantém três leituras separadas:

- a **J-lens direta** usa a projeção ajustada pelo Jacobiano para ranquear os
  tokens intermediários em cada camada ajustada;
- a **logit lens** usa a leitura direta sem o ajuste do Jacobiano, como
  comparação sobre os mesmos prompts, posições e camadas;
- `gradient_pursuit` faz uma decomposição esparsa da ativação residual já
  observada, apenas no painel qualitativo fixo de 12 prompts. Ele não altera a
  ativação e seus coeficientes não são tratados como os ranks das duas lentes.

Para as leituras diretas, o resultado reporta `pass@1`, `pass@5` e
`pass@10`, tanto na *band* de workspace resolvida em runtime quanto no conjunto
de camadas ajustadas da lente. O painel de 12 prompts, os ranks por camada e a
própria *band* também são diagnósticos observacionais. Eles podem orientar uma
hipótese futura, mas esta documentação não seleciona conceito, direção ou
`alpha` para uma nova intervenção.

Aqui, `success` tem sentido operacional: os 93 itens, a auditoria de
tokenização, o painel, os escopos de camadas, a proveniência e os hashes
precisam formar um conjunto completo e validado. Isso não exige `pass@k` alto;
uma execução completa pode ser operacionalmente bem-sucedida mesmo com
recuperação baixa. A execução canônica
`2fcb8a49-136f-4868-91cc-f98f9dc2e57d` terminou com esse `success` operacional
em uma NVIDIA L4; não houve geração de texto nem intervenção.

No conjunto completo, as métricas extraídas do JSON validado foram:

- J-lens na *band* 12–28: `pass@1=0,344086`,
  `pass@5=0,548387` e `pass@10=0,596774`;
- logit lens na mesma *band*: `0,284946`, `0,392473` e `0,435484`;
- J-lens em todas as camadas ajustadas, 0–30: `0,344086`, `0,564516` e
  `0,629032`;
- logit lens em todas as camadas: `0,284946`, `0,392473` e `0,446237`.

A auditoria encontrou 103 intermediários: 82 eram representáveis por um único
token e 21, por múltiplos tokens. O subconjunto single-token cobre 81 dos 93
itens e teve recuperação maior: na *band*, J-lens `0,395062 / 0,641975 /
0,691358` e logit lens `0,333333 / 0,456790 / 0,506173`; em todas as camadas,
J-lens `0,395062 / 0,654321 / 0,716049` e logit lens `0,333333 / 0,456790 /
0,518519`, sempre na ordem `pass@1/5/10`. Isso descreve este conjunto e esta
tokenização; não demonstra superioridade causal da J-lens. Na versão canônica
do schema, a auditoria também registra os IDs e hashes dos tokens dos 93
prompts, e o painel reutiliza esses IDs em vez de tokenizar o texto novamente.

O painel ajuda a tornar a escala dos ranks concreta. `Brazil` em
`amazon-language`, `Mars` em `mars-color`, `gold` em `atomic-79-symbol`,
`March` em `etym-wargod-month`, `oxygen` em `chem-photosynthesis-Z` e `east`
em `inv-sunrise-opposite` chegaram ao rank 1 da J-lens. Para `spider` em
`spider-legs`, o melhor resultado foi rank 2 na camada 19. Em contraste, os
três intermediários de `nhop-primary-planet` ficaram mal posicionados dentro
da *band*: `3` em rank 1999 na camada 12, `three` em 1090 na camada 12 e
`third` em 514 na camada 18. Considerando todas as camadas, esses ranks
melhoram para 24, 286 e 349, respectivamente, o que também mostra por que
*band* e busca global não devem ser tratadas como a mesma medida.

O heatmap é compatível com esse quadro: muitos itens melhoram a partir das
camadas intermediárias e se concentram em ranks menores aproximadamente entre
17 e 28, enquanto algumas linhas continuam ruins mesmo nessa região. Ele
mostra o menor rank por item e camada em escala logarítmica, não uma fronteira
causal nem uma camada universal. O `gradient_pursuit` por vezes reencontra o
mesmo texto (`spider`, por exemplo, aparece entre seus componentes em várias
camadas), mas decompõe a ativação por coeficientes; não é outro cálculo do
rank direto e não foi fundido com `pass@k`.

As figuras reproduzem as métricas e a estrutura do JSON. O gráfico de barras e
o heatmap são legíveis; já o painel preserva tokens brutos, inclusive quebras
de linha, scripts variados e símbolos de controle, o que torna algumas células
visualmente ruidosas. Por isso, o painel serve como visão qualitativa, enquanto o
JSON validado permanece a fonte para valores exatos. A interpretação ainda é
limitada pelo tokenizer, pela lente treinada no corpus Salesforce-Wikitext,
pelo recorte de camadas e pelo painel exploratório de 12 itens.

## Fluxo causal no Colab L4

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

## Fluxo observacional no Colab L4

Com a mesma instalação, autorização do modelo e autenticação descritas acima,
o atlas usa um notebook e um perfil de resultados próprios. Na raiz do
repositório:

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb --timeout 7200
python tools/colab/run_jspace.py fetch --result-set observation
python tools/colab/run_jspace.py stop
```

O perfil `observation` é separado do perfil causal padrão e tem exatamente
nove membros: dois JSONs compactos, seis figuras canônicas PNG/SVG e
`artifacts/jspace_observation_raw.json`. Esse sidecar bruto contém os registros
observacionais no nível dos prompts (não saídas geradas nem continuações) e
serve somente para o validador local recalcular hashes e reconciliar o conteúdo
compacto. O empacotamento e o download validam o JSON, o caminho e o limite de
tamanho; a publicação local dos nove membros é transacional. O sidecar
permanece ignorado pelo Git e não integra o commit de evidência canônica. O
perfil também não substitui `execution_manifest.json` nem
`final_evaluation.json`.

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

Na execução causal, o pipeline grava continuações brutas, prompts renderizados,
slots do workspace e outros diagnósticos apenas em
`artifacts/readout_and_causal_swap.json`, que é ignorado pelo Git. Os arquivos
baixados `results/execution_manifest.json` e
`results/final_evaluation.json` contêm somente evidência agregada compacta e
metadados de reprodutibilidade.

É possível validar localmente os artefatos causais ou as fixtures sem carregar
o modelo de 8B:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_results.py
uv run pytest -q
```

O atlas tem um validador independente, que confere schemas, hashes, contagens,
métricas recalculadas e coerência entre o artefato bruto ignorado e os JSONs
compactos:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_observation.py
```

Depois de `fetch --result-set observation`, os nove membros chegaram juntos,
inclusive o sidecar necessário para a validação independente. O comando agora
termina com `PASS`: reconcilia o sidecar com os dois JSONs compactos, seus
schemas, hashes, contagens e métricas, e valida as seis figuras. Somente os
dois JSONs compactos e as seis figuras formam a evidência canônica versionada;
o sidecar bruto e o notebook de saída continuam ignorados.

No protocolo causal, o estado `success` exige proveniência limpa e verificada
por digest, integridade da lente e readout explicitamente avaliados e pelo
menos um `alpha` que passe por todos os critérios do prompt principal `main`.
O parser deve identificar
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

O atlas não muda essa conclusão causal. Uma confirmação causal futura exigirá
uma hipótese nova, prompts pré-declarados que estabeleçam as precondições
comportamentais e uma nova execução registrada; não deve ser construída
escolhendo retrospectivamente conceito, direção ou `alpha` a partir do painel.
