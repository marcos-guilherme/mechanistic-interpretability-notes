# Atlas observacional do J-space no Llama 3.1 8B

## Objetivo

Antes de escolher outra intervenção causal, esta etapa vai observar quais
direções rotuladas pela J-lens aparecem no estado interno do
`meta-llama/Llama-3.1-8B-Instruct`. A exploração não trata `spider → ant` como
hipótese privilegiada, não executa geração e não altera ativações.

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

Nos prompts multihop publicados com o estudo, a J-lens do Llama 3.1 8B recupera
os conceitos intermediários não escritos no texto? Quais outros tokens aparecem
no workspace, e como o ranking direto da J-lens difere da logit lens e da
decomposição esparsa por *gradient pursuit*?

Esta pergunta é descritiva. O atlas poderá sugerir hipóteses para uma fase
causal posterior, mas não será usado para afirmar que uma direção é necessária
ou suficiente para o comportamento do modelo.

## Fonte e conjunto de prompts

Será usada a cópia vendorizada de `lens-eval-multihop.json` incluída no
`jspace` fixado no commit
`04c822a255fe0a10d84e08ec499e5e09f4b7d37a`. Ela reproduz o conjunto sintético
publicado pela Anthropic sob Apache 2.0 no repositório
`anthropics/jacobian-lens`. A cópia foi verificada contra o commit upstream
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; ambas têm SHA-256
`2cd5a47aa3d7627708843bc61f03ab1d10eda4184f357c5cbf93c97184e32c9c`. O
arquivo contém 93 itens com:

- `name`: identificador estável;
- `prompt`: completação textual que termina imediatamente antes da resposta;
- `target`: resposta esperada, usada para definir a posição de leitura;
- `intermediates`: conceitos intermediários que o prompt exige sem
  necessariamente mencioná-los.

Os prompts serão usados como texto bruto, exatamente como publicados, sem chat
template, instrução adicional ou geração. O texto, a ordem dos itens, o hash do
arquivo, os IDs tokenizados e seus digests serão registrados na proveniência.

O conjunto contém o caso `spider-legs`:

```text
Fact: The number of legs on the animal that spins webs is
```

O prompt original termina com um espaço. Seu intermediário é `spider` e o alvo
final é `8`. Isso permite reexaminar a hipótese anterior sem depender da
paráfrase e do chat template usados na primeira execução causal.

## Desenho observacional

### 1. Auditoria de tokenização

Antes do readout, cada intermediário será tokenizado nas variantes com e sem
espaço inicial. A execução registrará a sequência completa, o token ID usado
pela regra canônica de primeiro token e se o conceito é single-token no
tokenizer do Llama.

O score canônico seguirá a regra publicada, baseada no primeiro token. Um score
secundário será calculado somente sobre intermediários single-token. Assim, a
replicação permanece comparável sem esconder diferenças entre o tokenizer do
Llama e os tokenizers dos modelos avaliados no estudo.

### 2. Avaliação completa dos 93 prompts

Para cada item, o readout será calculado no último token do prompt, a posição
que antecede `target`. Em todas as camadas do workspace band, serão obtidos:

- ranking direto do intermediário pela J-lens;
- ranking do mesmo token pela logit lens;
- melhor rank e camada para cada método;
- `pass@1`, `pass@5` e `pass@10` agregados.

O workspace band será a análise principal. A repetição em todas as camadas
ajustadas pela lente será uma análise secundária explicitamente identificada,
para mostrar se um acerto depende de camadas externas ao band.

### 3. Painel qualitativo fixo

Doze itens serão aprofundados, escolhidos antes da execução para cobrir relações
e domínios diferentes:

`carnival-ocean`, `amazon-language`, `mars-color`, `spider-legs`,
`basketball-players`, `atomic-79-symbol`, `planet-3-moons`,
`rhyme-rain-neighbor`, `etym-wargod-month`, `chem-photosynthesis-Z`,
`nhop-primary-planet` e `inv-sunrise-opposite`.

Para cada camada do workspace band, o painel armazenará separadamente:

- os dez primeiros tokens do ranking direto da J-lens;
- os dez primeiros tokens da logit lens;
- até 25 direções selecionadas por *gradient pursuit*, com coeficientes e
  variância explicada.

Nenhum resultado da avaliação completa alterará essa seleção.

## Métricas e interpretação

Para cada intermediário, o melhor rank será o mínimo entre as camadas do escopo
avaliado. `pass@k` será a média, entre os 93 itens, da fração de intermediários
cujo melhor rank é menor ou igual a `k`, com ranks apresentados em base 1. Essa
é a definição canônica do conjunto.

Serão publicados separadamente:

- J-lens no workspace band, resultado principal;
- logit lens no mesmo band, baseline de método;
- J-lens e logit lens em todas as camadas ajustadas, análises secundárias;
- scores canônicos por primeiro token e scores do subconjunto single-token.

A unidade registrada será sempre o token ID, preservando o texto decodificado.
O *gradient pursuit* não será incluído em `pass@k`: ele representa uma
decomposição esparsa do residual, não o mesmo ranking produzido pelo readout
direto da lente. Seus coeficientes, ranks e variância explicada serão usados
somente na análise qualitativa dos 12 itens.

O atlas não aplicará testes confirmatórios nem escolherá um “conceito vencedor”.
Direções candidatas para uma intervenção futura deverão combinar presença em
mais de um prompt, persistência entre camadas, coerência com os intermediários
esperados e uma hipótese comportamental que possa ser testada em prompts
separados.

## Arquitetura

O experimento reutilizará o carregamento validado do modelo, da J-lens, da
proveniência e do runner Colab. A lógica causal existente continuará isolada.

Novos componentes terão responsabilidades separadas:

1. um carregador validará licença, revisão, hash, schema, ordem e contagem dos
   93 itens vendorizados;
2. um auditor de tokenização produzirá as variantes canônicas e o subconjunto
   single-token;
3. um avaliador obterá os rankings diretos da J-lens e da logit lens no último
   token de cada prompt;
4. um leitor qualitativo processará o painel fixo por camada e manterá separado
   o resultado de *gradient pursuit*;
5. um pipeline observacional gravará artefatos atômicos e proveniência;
6. um gerador de figuras produzirá as visualizações estáticas;
7. um notebook fino apenas chamará o pipeline.

As ativações do painel serão transferidas para CPU após a captura. Durante a
decomposição esparsa, somente a camada corrente do dicionário e os vetores
necessários retornarão à GPU. Essa ordem preserva a estratégia limitada em
memória usada na primeira execução e evita rematerializar o dicionário para
cada prompt.

O pipeline não chamará geração, hooks de intervenção, `LensSwap`, clamp ou
perturbações aleatórias. A única alteração sobre o modelo será a instrumentação
necessária para registrar ativações de um forward pass.

## Artefatos e visualizações

O experimento terá artefatos próprios, sem modificar
`execution_manifest.json` ou `final_evaluation.json`:

- `artifacts/jspace_observation_raw.json`: slots completos e diagnósticos,
  ignorado pelo Git;
- `results/jspace_observation_manifest.json`: proveniência e estado da
  execução, incluindo SHA-256 do artefato bruto local;
- `results/jspace_observation.json`: resumo compacto validado por schema;
- `results/jspace_observation_pass_at_k.{png,svg}`: comparação entre J-lens e
  logit lens no band e em todas as camadas;
- `results/jspace_observation_rank_heatmap.{png,svg}`: rank do intermediário
  esperado por item e camada;
- `results/jspace_observation_panel.{png,svg}`: trajetórias dos tokens nos 12
  casos qualitativos.

O JSON compacto manterá os agregados e os dados necessários para reconstruir as
figuras. Tokens especiais, whitespace e fragmentos subword não serão removidos
do dado bruto; qualquer filtro de apresentação será explícito e registrado.

## Falhas e estados

Falhas de acesso, download, hash ou schema do conjunto, CUDA, integridade da
lente, dimensões, camadas, tokenização, memória ou decomposição numérica terão
códigos distintos. Escritas serão atômicas para não misturar artefatos de
execuções diferentes.

Uma execução só terá estado `success` quando:

- a proveniência e a integridade da lente estiverem completas;
- o arquivo de origem corresponder à revisão, ao hash e aos 93 itens esperados;
- os 93 prompts tiverem auditoria de tokenização e rankings dos dois métodos;
- as camadas do workspace band tiverem sido processadas como análise principal;
- todas as camadas ajustadas tiverem sido processadas como análise secundária;
- os 12 itens fixos tiverem o readout qualitativo completo;
- o resumo compacto e as três famílias de figuras forem validados.

`success` significa apenas que a observação foi executada como especificado. Não
significa que uma interpretação semântica ou hipótese causal foi confirmada.

## Testes

Os testes locais usarão modelos, tokenizers, ativações e dicionários falsos para
verificar:

- captura e liberação de memória camada por camada;
- validação do arquivo vendorizado, de seu hash e de seus 93 itens;
- variantes de tokenização, primeiro token e filtro single-token;
- ranks em base 1, mínimo entre camadas e `pass@1/5/10`;
- separação entre workspace band e análise de todas as camadas;
- separação entre ranking direto e decomposição por *gradient pursuit*;
- separação entre artefatos brutos e compactos;
- schemas, hashes, proveniência e escrita atômica;
- geração determinística das matrizes usadas pelas figuras;
- ausência de chamadas de geração ou intervenção no pipeline observacional;
- preservação dos artefatos canônicos do experimento causal.

Um teste de integração do notebook confirmará que ele apenas chama o pipeline,
não contém lógica duplicada e permanece sem outputs versionados.

## Separação entre exploração e confirmação

Os 93 prompts oficiais serão o conjunto de seleção exploratória. Se o atlas
sugerir uma direção candidata, a fase seguinte terá uma especificação própria:
conceito, camadas, intensidade, comportamento esperado, controles e prompts
confirmatórios serão fixados antes da execução. Os prompts deste atlas não
servirão como evidência confirmatória da intervenção escolhida a partir deles.

Essa separação permite usar a exploração para formular uma hipótese sem
apresentar a mesma observação que motivou a hipótese como sua confirmação.
