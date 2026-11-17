# J-lens v2 do Qwen com contexto longo

## Objetivo

Construir uma segunda J-lens para o `Qwen/Qwen2.5-Coder-7B-Instruct` sem
substituir o artefato atual. A nova execução deve testar se contextos de 512
tokens produzem uma lente mais fiel do que a primeira execução, ajustada com
100 funções truncadas em 128 tokens.

O experimento continua sendo uma medida de fidelidade ao processamento interno
do modelo. Ele não mede sozinho qualidade de código, recusa ou segurança.

## Corpus

O corpus permanece no split de treino Python do CodeSearchNet, com a revisão já
fixada no repositório. Cada item usa a função completa disponível no campo
`whole_func_string`, limitada a 512 tokens. Só entram funções com pelo menos
512 tokens, código e documentação não vazios, conteúdo único e hash ausente da
lista de exclusão.

Serão reservadas 200 funções para holdout. Elas não participam do cálculo das
matrizes Jacobianas. A quantidade de funções de ajuste será decidida por um
probe real na A100. O orquestrador deve escolher a maior opção segura entre
100, 150, 200, 250, 300, 400, 600 e 800, respeitando uma projeção máxima de 12
horas para o ajuste.

## Artefatos isolados

A execução v2 terá configuração, diretório de resultados, diretório de
artefatos, identidade de corpus e `run_id` próprios. A lente de 100 amostras e
seus resultados não serão alterados. A interface continuará usando a lente
congelada atual até que a v2 seja avaliada e aprovada explicitamente.

## Checkpoints e retomada

Após cada função processada, a execução grava atomicamente:

* as somas das Jacobianas em `float32`;
* o número de funções concluídas;
* o próximo índice;
* o `dim_batch`;
* os hashes da configuração, do corpus e da revisão do modelo.

Uma retomada só é aceita quando todas essas identidades coincidem. Checkpoints
incompatíveis devem falhar de forma explícita.

O checkpoint dentro da VM protege contra a interrupção do processo. Para
proteger contra a perda da sessão Colab, o operador fará cópias periódicas para
a máquina local em marcos de progresso. Cada cópia deve ser validada antes de
substituir a anterior. A última cópia válida pode ser reenviada para uma nova
sessão.

## Avaliação

As lentes v1 e v2 serão avaliadas nas mesmas 200 funções de holdout da v2. Para
cada posição válida e cada camada fonte, será medido o rank do token escolhido
pela camada final do Qwen.

O relatório comparará:

* `pass@1`, `pass@5` e `pass@10`;
* rank médio;
* resultados por camada e por função;
* a logit lens como referência;
* estabilidade das métricas entre subconjuntos do holdout.

A v2 só será candidata a substituir a lente da interface se melhorar a
fidelidade no holdout sem apresentar regressões amplas entre camadas. A troca
do artefato não faz parte desta execução.

## Operação

A sequência será: testes locais, início da A100, sincronização, instalação,
autenticação, probe, seleção da quantidade de ajuste, execução retomável,
cópias periódicas, holdout, download dos resultados e encerramento da A100.
Qualquer falha terminal deve encerrar a GPU depois de preservar o último
checkpoint acessível.

## Limites

O probe de 128 tokens levou 42,53 segundos por função. Isso não permite prever
com precisão o custo de 512 tokens, pois atenção e cálculo Jacobiano não têm
escala estritamente linear. O primeiro probe de 512 tokens é a fonte de verdade
para a decisão de tamanho.

Janelas independentes de 128 tokens não serão usadas nesta versão. Elas
aumentariam a cobertura de texto, mas removeriam o contexto causal anterior das
janelas intermediárias. Uma implementação com KV cache exigiria demonstrar
paridade com o método original e fica fora deste experimento.
