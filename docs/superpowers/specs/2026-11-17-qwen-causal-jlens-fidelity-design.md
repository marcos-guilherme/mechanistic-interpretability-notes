# Fidelidade da J-lens sob intervenção causal no Qwen

## Estado

Desenho aprovado em 17 de novembro de 2026.

## Contexto

Os experimentos anteriores observaram duas propriedades diferentes do
`Qwen/Qwen2.5-Coder-7B-Instruct`.

A direção de recusa produziu mudanças causais grandes. A ablação de 20% reduziu
a recusa, a adição de 50% aumentou a recusa e um controle aleatório ortogonal
não reproduziu o mesmo efeito. No code-eval, porém, a adição elevou a recusa
tanto em pedidos ofensivos quanto em pedidos benignos ou defensivos. A direção
funcionou como uma alavanca potente, mas pouco seletiva.

A J-lens recuperou parte da decisão final do modelo a partir de camadas
intermediárias. A lente v1 foi ajustada com 100 funções e 128 tokens. A v2 foi
ajustada com 400 funções e 256 tokens. No holdout de 200 funções, ambas
superaram a logit lens, mas a v2 não superou a v1 nas métricas agregadas.

Esses resultados ainda foram obtidos separadamente. O presente experimento
verifica se a capacidade descritiva das lentes permanece válida quando o estado
residual é alterado causalmente pela direção de recusa.

## Pergunta de pesquisa

> A J-lens continua descrevendo corretamente a decisão do modelo quando
> alteramos causalmente sua direção de recusa?

A pergunta não é se a lente prevê palavras de recusa. O alvo será sempre o
token que o modelo interveniente realmente escolheu na saída final. Isso mantém
a mesma definição de fidelidade usada no holdout da J-lens.

## Hipóteses

### Hipótese principal

A fidelidade da lente v1 e da lente v2 na camada 25, medida por `pass@5`, será
preservada sob intervenção. Preservação significa que o intervalo de confiança
de 95% para a diferença pareada contra o baseline permanece inteiramente entre
-5 e +5 pontos percentuais. A equivalência será avaliada e reportada
separadamente por lente, conjunto e condição. Não haverá um único `pass` global
que esconda diferenças entre esses recortes.

### Hipóteses secundárias

- O controle aleatório terá efeito menor que a ablação e a adição sobre a taxa
  de recusa.
- As camadas 25 e 26 serão mais fiéis que as camadas 19 e 22 em todas as
  condições.
- A adição poderá alterar fortemente o comportamento sem necessariamente
  destruir a fidelidade tardia da lente.
- Se a fidelidade cair sob intervenção, a queda poderá diferir entre MITRE e
  MITRE-FRR, indicando dependência do tipo de prompt.

## Abordagens consideradas

### Ampliar o code-eval completo

Esse caminho preservaria a infraestrutura atual, mas repetiria HumanEval+ e
etapas que não respondem à pergunta principal. Também manteria misturadas a
fidelidade ao token final e a posição de um léxico de recusa.

### Painel focado em MITRE e MITRE-FRR

Este foi o caminho escolhido. Ele reutiliza os IDs já congelados, as quatro
condições causais e os artefatos existentes, mas cria um perfil isolado e uma
métrica canônica de fidelidade ao token final.

### Estudo manual pela interface

A interface permite observar exemplos, mas a seleção manual não oferece uma
base adequada para comparar condições ou estimar incerteza.

## Dados

O painel usará os mesmos 32 exemplos MITRE e 32 exemplos MITRE-FRR registrados
em `results/qwen25_coder_7b_code_eval/00_prepare.json`.

O experimento não fará uma nova seleção aleatória. O estágio de preparação
deverá:

- carregar os 64 IDs existentes;
- localizar os registros correspondentes nas revisões já fixadas do
  PurpleLlama;
- verificar IDs, categorias, hashes de conteúdo e ordem;
- abortar se qualquer exemplo estiver ausente ou diferente;
- registrar somente IDs e hashes nos resultados versionáveis.

MITRE representa solicitações ofensivas. MITRE-FRR representa solicitações
benignas, defensivas ou sensíveis que não deveriam ser recusadas de forma
indiscriminada.

## Condições

Cada prompt será executado nas quatro condições já usadas no code-eval:

- `baseline`, sem intervenção;
- `ablation`, direção de recusa removida com coeficiente 0,2;
- `addition`, direção adicionada com coeficiente 0,5;
- `random_control`, direção aleatória ortogonal com coeficiente 0,2 e semente
  1042.

Os coeficientes deverão ser lidos do resultado de dose aprovado. O experimento
abortará se a configuração divergir do par selecionado no follow-up.

A direção congelada vem da camada 18 e do offset `-1`. A ablação parcial e o
controle aleatório ortogonal usarão os mesmos hooks do code-eval em blocos,
atenções e MLPs. A adição será aplicada à entrada residual da camada 18. Além
dos coeficientes, o runtime verificará essas regras antes de iniciar o painel.

## Geração

O modelo e a revisão permanecerão congelados:

- modelo `Qwen/Qwen2.5-Coder-7B-Instruct`;
- revisão `c03e6d358207e414f1eca0bb1891e29f1db0e242`;
- `bfloat16`;
- geração gulosa;
- até 64 tokens novos;
- `batch_size=1`;
- template de chat do próprio modelo.

A resposta completa será mantida em artefatos não versionados para auditoria e
classificação de recusa. Os JSONs versionados não conterão prompt nem resposta
em texto claro.

## Readout sincronizado

Somente os primeiros oito tokens gerados serão usados para a análise da lente.
Em cada passo, o motor deverá:

1. aplicar a intervenção antes da continuação do forward;
2. capturar o estado residual das camadas 19, 22, 25 e 26;
3. obter os logits finais usados pela geração gulosa;
4. definir como alvo o `argmax` desses logits;
5. aplicar a lente v1 e a lente v2 ao mesmo estado capturado;
6. calcular o rank um-indexado do token alvo em cada lente;
7. registrar somente IDs, hashes, condição, passo, camada, lente, token alvo e
   rank.

As duas lentes deverão ser carregadas uma vez e verificadas pelos hashes já
registrados:

- v1:
  `artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt`;
- v2:
  `artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt`.

A leitura incremental deverá manter paridade numérica com
`JacobianLens.apply` dentro da tolerância já usada pelo laboratório.

## Unidade de análise

Cada observação de rank será identificada por:

`(dataset, example_id, condition, generation_step, layer, lens_version)`.

O painel completo terá:

- 64 prompts;
- 4 condições;
- 8 passos de geração;
- 4 camadas;
- 2 lentes;
- até 16.384 observações de rank;
- 256 respostas completas.

As 16.384 observações são esperadas se todas as respostas alcançarem oito
tokens. Términos antecipados reduzirão esse total e serão registrados. Entre os
passos que efetivamente existem, uma combinação ausente ou duplicada invalidará
a corrida.

## Métricas

Para cada lente, conjunto, condição e camada serão calculados:

- `pass@1`;
- `pass@5`;
- `pass@10`;
- rank médio;
- quantidade de observações válidas.

A taxa de recusa será calculada por conjunto e condição sobre a resposta
completa de até 64 tokens, usando o mesmo classificador textual do code-eval.
Ela descreve o efeito comportamental. Não será usada como alvo da J-lens.

## Comparação pareada e incerteza

As diferenças de fidelidade serão calculadas contra o baseline para o mesmo
prompt e a mesma lente. A unidade de reamostragem será o prompt, não cada token,
camada ou posição isolada.

O intervalo de confiança será obtido por bootstrap agrupado:

- 10.000 reamostragens;
- semente fixa 2028;
- amostragem separada dentro de MITRE e MITRE-FRR;
- percentis 2,5 e 97,5;
- diferenças reportadas em pontos percentuais.

O desfecho principal será a diferença de `pass@5` na camada 25. A equivalência
prática será declarada somente quando todo o intervalo de confiança estiver
entre -5 e +5 pontos percentuais.

`pass@1`, `pass@10`, rank médio e as camadas 19, 22 e 26 serão análises
secundárias. Essa hierarquia será gravada antes da execução para evitar escolha
de camada ou métrica após observar os números.

## Análise exploratória de mudanças comportamentais

Além dos agregados pré-registrados, o relatório separará, para cada intervenção,
os prompts cujo rótulo automático de recusa mudou em relação ao baseline dos
prompts cujo rótulo permaneceu igual. Dentro desses grupos, serão descritos o
`pass@5` da camada 25 e a mudança pareada de rank para as duas lentes. Para a
mudança de rank, primeiro será calculado o rank médio dos passos disponíveis de
cada prompt; a diferença condição menos baseline será calculada no mesmo
prompt.

Essa análise responde se a lente acompanha o token escolhido justamente nos
casos em que a intervenção altera o comportamento observável. Ela reutiliza as
mesmas respostas e observações, sem novas gerações. Os resultados serão
identificados como exploratórios: os grupos são definidos por uma variável
posterior à intervenção, podem conter poucos prompts e dependem do classificador
textual de recusa. Não haverá teste de equivalência nem conclusão confirmatória
para esses subgrupos. O relatório mostrará a quantidade de prompts em cada
grupo e omitirá porcentagens para grupos vazios.

## Interpretação possível

Se a taxa de recusa mudar e a fidelidade permanecer equivalente, a lente terá
acompanhado uma decisão alterada causalmente. Isso sustenta robustez descritiva,
não uma explicação causal da lente.

Se a taxa de recusa mudar e a fidelidade cair além da margem, a lente terá
falhado em generalizar para estados residuais intervenientes. Esse resultado
delimita onde a leitura descritiva deixa de ser confiável.

Se o controle aleatório também reduzir a fidelidade, parte do efeito poderá vir
de deslocamento genérico da distribuição residual, e não da direção de recusa
em particular.

Nenhum resultado permitirá chamar os tokens da lente de pensamentos ou afirmar
que a J-lens explica o mecanismo de recusa.

## Arquitetura

O experimento terá um perfil isolado, sem modificar resultados existentes:

- configuração `qwen25_coder_7b_causal_jlens.json`;
- módulo puro para contratos, agregação e bootstrap;
- runtime remoto para preparação, geração, readout e finalização;
- entrypoint Colab próprio;
- transporte e allowlist próprios;
- validador offline próprio.

O runtime reutilizará:

- seleção e IDs do code-eval;
- hooks de ablação, adição e controle aleatório;
- classificador de recusa;
- geração incremental e limpeza segura de hooks do laboratório;
- carregamento e contrato upstream da J-lens;
- progresso atômico e eventos já usados nos demais perfis.

## Estágios e retomada

### Preparação

Verifica modelo, CUDA A100, datasets, direção, dose, lentes, hashes e matriz
esperada. Produz um manifesto compacto sem texto bruto.

### Piloto

Executa dois prompts de cada conjunto nas quatro condições. O piloto verifica:

- 16 respostas;
- até 1.024 observações de rank, com términos antecipados explícitos;
- finitude dos logits;
- paridade do readout;
- limpeza dos hooks;
- ausência de duplicatas.

O painel completo só inicia se o piloto passar.

### Painel

Executa uma combinação prompt-condição por vez. Após cada resposta:

- grava o texto no artefato de auditoria;
- grava ranks compactos em JSONL;
- atualiza checkpoint e status de forma atômica;
- libera tensores e hooks da combinação concluída.

A retomada deverá validar hashes de configuração, datasets, direção, lentes,
modelo e implementação. Uma identidade diferente abortará em vez de misturar
corridas.

### Finalização

Agrega métricas, calcula os intervalos, produz avaliação e manifesto e remove
texto bruto dos arquivos destinados ao git.

## Saídas

Resultados versionáveis:

- `00_prepare.json`;
- `01_pilot.json`;
- `02_panel.json`;
- `03_analysis.json`;
- `causal_jlens_evaluation.json`;
- `execution_manifest.json`.

Artefatos não versionados:

- datasets selecionados com texto;
- respostas completas;
- observações incrementais;
- checkpoint;
- status e eventos.

O JSON terminal deverá incluir `run_id`, hashes das duas lentes, hash da
direção, revisões, coeficientes, quantidades esperadas, status operacional e
resultados dos critérios principais separados por lente, conjunto e condição.

## Validação

O validador deverá recomputar as métricas a partir das observações compactas e
confirmar:

- 32 IDs MITRE e 32 IDs MITRE-FRR iguais aos do code-eval;
- quatro condições exatas;
- oito passos por resposta, salvo término antecipado explicitamente marcado;
- camadas 19, 22, 25 e 26;
- lentes v1 e v2;
- ausência de chaves de texto bruto nos resultados versionáveis;
- unicidade da chave de observação;
- coerência entre ranks e `pass@k`;
- bootstrap reproduzível com semente 2028;
- comparação pareada contra baseline;
- hashes e revisões congelados;
- sequência válida de status e eventos;
- manifesto coerente com os arquivos presentes.

Se uma resposta terminar antes de oito tokens, o runtime deverá registrar o
motivo e o número real de passos. O denominador de cada condição será
explicitado. A corrida não poderá preencher observações ausentes com zeros.

## Testes

A implementação seguirá TDD com testes focados:

- contrato da configuração e dos artefatos fonte;
- chave de observação e detecção de duplicata;
- agregação de `pass@k` e rank;
- bootstrap agrupado determinístico;
- equivalência dentro da margem;
- separação exploratória entre rótulos alterados e preservados;
- sincronização das duas lentes sobre a mesma ativação;
- hooks removidos após sucesso, erro e cancelamento;
- checkpoint e retomada sem repetição;
- validador de matriz completa e término antecipado;
- transporte e allowlist do novo perfil.

O teste remoto começará com o piloto. O painel completo não será usado como
substituto dos testes de contrato.

## Documentação final

Será criado `projects/refusal_direction/CAUSAL_JLENS.md` com:

- contexto e pergunta;
- hipótese principal e métricas secundárias;
- dados e condições;
- decisões tomadas antes da execução;
- diário operacional com interrupções e retomadas;
- resultados e intervalos;
- análise exploratória dos prompts com e sem mudança no rótulo de recusa;
- exemplos auditáveis selecionados por regra fixa;
- limitações e interpretação para o TCC.

Também serão atualizados:

- `projects/refusal_direction/PROCESSO.md`;
- `docs/ACOMPANHAMENTO_EXPERIMENTOS.md`;
- documentação dos comandos Colab;
- portal Docusaurus, sem linguagem promocional.

## Limites

- O painel tem 32 prompts por conjunto. Os intervalos agrupados refletem essa
  quantidade, não as milhares de observações internas como se fossem
  independentes.
- O classificador textual de recusa pode errar e deverá permanecer separado da
  métrica de fidelidade.
- As lentes foram ajustadas em CodeSearchNet, não em MITRE. O teste mede
  generalização fora do corpus de ajuste.
- A adição produz forte deslocamento comportamental e poderá levar o readout
  para estados fora da distribuição usada no ajuste.
- A equivalência de `pass@5` não prova equivalência das distribuições completas.
- Não haverá novo treinamento de lente nem expansão de benchmark neste escopo.
