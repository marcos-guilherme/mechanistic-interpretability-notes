# Atlas observacional do J-space no Llama 3.1 8B

## Objetivo

Antes de escolher outra intervenção causal, esta etapa vai observar quais
direções rotuladas pela J-lens aparecem no estado interno do
`meta-llama/Llama-3.1-8B-Instruct`. A exploração não parte de `spider`, `ant` ou
qualquer outro conceito-alvo e não executa geração nem altera ativações.

O resultado anterior permanece como evidência canônica da primeira tentativa
causal. Ele foi inconclusivo porque o baseline comportamental não foi
estabelecido, `spider` não apareceu no readout e a intervenção degradou a
continuação. O novo atlas será um experimento separado; ele não reclassificará
nem substituirá essa execução.

Os itens recuperados pela J-lens serão descritos como **direções rotuladas por
tokens** ou **candidatos interpretáveis**. Um token com coeficiente alto não
demonstra, por si só, que o modelo possui um conceito discreto com aquele
significado.

## Pergunta exploratória

Quais direções rotuladas por tokens aparecem com maior saliência e persistência
nas camadas cobertas pela J-lens, como esses padrões variam entre contextos e
como evoluem ao longo das posições de conteúdo de alguns prompts?

Esta pergunta é descritiva. O atlas poderá sugerir hipóteses para uma fase
causal posterior, mas não será usado para afirmar que uma direção é necessária
ou suficiente para o comportamento do modelo.

## Painel pré-definido de prompts

O painel terá oito prompts curtos em inglês, dois por categoria:

| Categoria | Identificador | Texto |
|---|---|---|
| Fato concreto | `fact_capital` | `The capital of France is` |
| Fato concreto | `fact_freezing` | `Water freezes at` |
| Senso comum e narrativa | `common_glass` | `When Maya dropped the glass, it` |
| Senso comum e narrativa | `common_key` | `To unlock the door, he used a` |
| Raciocínio abstrato | `abstract_syllogism` | `If all daxes are blue and this is a dax, then it is` |
| Raciocínio abstrato | `abstract_arithmetic` | `Two plus three equals` |
| Controle lexical | `control_repeat` | `Repeat exactly: meadow` |
| Controle lexical | `control_ready` | `Reply with one word: ready` |

Os prompts serão fixados na configuração e renderizados com o chat template do
modelo e `add_generation_prompt=True`. O texto, os IDs tokenizados e seus
digests serão registrados na proveniência. Nenhum prompt será alterado depois
de observar o atlas canônico.

## Desenho em duas etapas

### 1. Varredura ampla

Para cada um dos oito prompts, o readout será calculado no último slot do
prefill, imediatamente antes da geração, em todas as camadas ajustadas e
validadas da J-lens. Cada slot armazenará:

- camada, posição e token observado;
- os 25 token IDs recuperados, seus textos decodificados, ranks e coeficientes;
- variância explicada pela decomposição;
- identificador do prompt e categoria.

Essa etapa compara contextos com custo controlado e mostra o conteúdo disponível
no ponto em que o modelo começaria a responder.

### 2. Leitura profunda

Três prompts serão escolhidos automaticamente a partir da varredura ampla. Para
cada prompt, a assinatura será o conjunto não vazio de token IDs que aparece em
pelo menos duas camadas. Quando houver pelo menos três prompts elegíveis, serão
avaliadas todas as combinações de três e escolhida a que maximiza a distância
média de Jaccard, definida como `1 - |A ∩ B| / |A ∪ B|`; empates serão
resolvidos pela ordem fixa da configuração. Quando houver menos de três prompts
elegíveis, o fallback fixo será `fact_capital`, `common_glass` e
`abstract_syllogism`.

Nos três prompts escolhidos, o readout será repetido para todas as posições do
texto do usuário e para o último slot do prefill. O span do usuário será
identificado por offsets do tokenizer sobre o chat template renderizado. A
execução falhará explicitamente se os offsets não permitirem um mapeamento
inequívoco; posições do template não serão silenciosamente tratadas como
conteúdo.

## Métricas e interpretação

A unidade de agregação será o token ID, preservando separadamente o texto
decodificado. Isso evita unir tokens distintos apenas porque sua apresentação é
parecida. Para cada token ID, o resumo registrará:

- número e proporção de prompts em que aparece;
- número de slots e camadas em que aparece;
- primeira e última camada;
- maior coeficiente e o slot correspondente;
- rank mediano e melhor rank;
- persistência, medida pelo número de pares de camadas adjacentes em que o token
  aparece.

Coeficientes brutos e variância explicada serão preservados para diagnóstico,
mas a comparação visual entre camadas usará uma saliência baseada em rank,
`(k - rank + 1) / k`. Essa decisão evita assumir que a escala absoluta dos
coeficientes é diretamente comparável entre todas as camadas.

O atlas não aplicará testes confirmatórios nem escolherá um “conceito vencedor”.
Direções candidatas para uma intervenção futura deverão combinar presença em
mais de um prompt, persistência entre camadas, leitura posicional coerente e
uma hipótese comportamental que possa ser testada em prompts separados.

## Arquitetura

O experimento reutilizará o carregamento validado do modelo, da J-lens, da
proveniência e do runner Colab. A lógica causal existente continuará isolada.

Novos componentes terão responsabilidades separadas:

1. uma configuração observacional fixa conterá prompts, `k`, política de
   posições e regra de seleção profunda;
2. um coletor registrará ativações de prefill sem geração e as manterá fora do
   Git;
3. um leitor processará uma camada por vez, materializará o dicionário uma única
   vez por camada, calculará todos os slots necessários e o removerá da GPU;
4. um agregador produzirá assinaturas, seleção determinística e resumos;
5. um pipeline observacional gravará artefatos atômicos e proveniência;
6. um gerador de figuras produzirá as visualizações estáticas;
7. um notebook fino apenas chamará o pipeline.

As ativações serão transferidas para CPU após a captura. Durante o readout,
somente a camada corrente do dicionário e os vetores necessários retornarão à
GPU. Essa ordem preserva a estratégia limitada em memória usada na primeira
execução e evita rematerializar o dicionário para cada prompt.

## Artefatos e visualizações

O experimento terá artefatos próprios, sem modificar
`execution_manifest.json` ou `final_evaluation.json`:

- `artifacts/jspace_observation_raw.json`: slots completos e diagnósticos,
  ignorado pelo Git;
- `results/jspace_observation_manifest.json`: proveniência e estado da
  execução, incluindo SHA-256 do artefato bruto local;
- `results/jspace_observation.json`: resumo compacto validado por schema;
- `results/jspace_observation_layers.{png,svg}`: heatmap de token por camada na
  varredura ampla;
- `results/jspace_observation_variance.{png,svg}`: variância explicada por
  camada e prompt;
- `results/jspace_observation_positions.{png,svg}`: leitura posição por camada
  dos três prompts aprofundados.

O heatmap amplo mostrará somente os token IDs mais persistentes, com texto
decodificado como rótulo e saliência baseada em rank. O JSON compacto manterá
todos os resumos necessários para reconstruir as figuras. Tokens especiais,
whitespace e fragmentos subword não serão removidos do dado bruto; qualquer
filtro de apresentação será explícito e registrado.

## Falhas e estados

Falhas de acesso, download, CUDA, integridade da lente, dimensões, camadas,
tokenização, mapeamento de offsets, memória ou decomposição numérica terão
códigos distintos. Escritas serão atômicas para não misturar artefatos de
execuções diferentes.

Uma execução só terá estado `success` quando:

- a proveniência e a integridade da lente estiverem completas;
- a varredura ampla contiver exatamente um slot por combinação de oito prompts
  e camadas esperadas;
- as camadas esperadas tiverem sido processadas;
- exatamente três prompts tiverem sido selecionados pela regra registrada;
- todas as posições de conteúdo desses prompts tiverem sido processadas;
- o resumo compacto e as três famílias de figuras forem validados.

`success` significa apenas que a observação foi executada como especificado. Não
significa que uma interpretação semântica ou hipótese causal foi confirmada.

## Testes

Os testes locais usarão modelos, tokenizers, ativações e dicionários falsos para
verificar:

- captura e liberação de memória camada por camada;
- ranks, agregações, persistência e regra de desempate;
- seleção por distância de Jaccard e fallback;
- mapeamento exato das posições de conteúdo;
- separação entre artefatos brutos e compactos;
- schemas, hashes, proveniência e escrita atômica;
- geração determinística das matrizes usadas pelas figuras;
- ausência de chamadas de geração ou intervenção no pipeline observacional;
- preservação dos artefatos canônicos do experimento causal.

Um teste de integração do notebook confirmará que ele apenas chama o pipeline,
não contém lógica duplicada e permanece sem outputs versionados.

## Separação entre exploração e confirmação

O painel de oito prompts será o conjunto de seleção exploratória. Se o atlas
sugerir uma direção candidata, a fase seguinte terá uma especificação própria:
conceito, camadas, intensidade, comportamento esperado, controles e prompts
confirmatórios serão fixados antes da execução. Os prompts deste atlas não
servirão como evidência confirmatória da intervenção escolhida a partir deles.

Essa separação permite usar a exploração para formular uma hipótese sem
apresentar a mesma observação que motivou a hipótese como sua confirmação.
