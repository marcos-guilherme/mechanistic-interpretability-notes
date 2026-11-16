# Laboratório visual de intervenção no Qwen

## Estado

Desenho aprovado em 16 de novembro de 2026.

## Objetivo

Criar uma interface experimental simples, executada em uma A100, para conversar
com `Qwen/Qwen2.5-Coder-7B-Instruct` e observar como a direção de recusa e a
J-lens alteram a geração.

A primeira versão deve permitir que uma pessoa escreva um prompt, escolha uma
condição de intervenção, selecione uma camada da J-lens e acompanhe, token por
token, a resposta do modelo e os tokens favorecidos pela leitura daquela
camada.

Esta interface começa como uma ferramenta de exploração do experimento já
realizado. O motor de inferência será separado da camada visual para que, no
futuro, outras lentes, modelos e visualizações possam ser adicionados sem
reescrever a instrumentação.

## Interpretação correta da visualização

A interface não apresentará os tokens da J-lens como pensamentos literais do
modelo. O rótulo usado será **previsões da J-lens**.

Para uma camada e uma posição, a lente transforma o estado residual observado
em uma distribuição sobre o vocabulário. Os tokens exibidos são os maiores
scores dessa distribuição. Eles indicam o que a lente torna legível naquela
representação, mas não constituem uma transcrição do raciocínio do modelo nem
uma explicação causal por si só.

A relação causal continua vindo da intervenção na direção de recusa. A J-lens
serve como leitura sincronizada do estado interno durante essa intervenção.

## Abordagens consideradas

### Notebook com widgets

Seria o caminho mais curto, mas deixaria estado, callbacks e visualizações
presos ao notebook. É adequado para uma demonstração pontual, não para o início
de uma ferramenta.

### Gradio acoplado diretamente ao script experimental

Produziria uma interface rapidamente, mas misturaria carregamento do modelo,
intervenções, streaming e componentes visuais no mesmo módulo. A manutenção
ficaria difícil assim que uma segunda visualização fosse adicionada.

### Motor isolado com frontend Gradio

Esta foi a abordagem escolhida. O backend terá contratos próprios para sessão,
geração, intervenção e readout. O Gradio consumirá esses contratos sem conhecer
os detalhes dos hooks do modelo. A primeira entrega permanece pequena, mas o
núcleo poderá ser reaproveitado por um frontend mais elaborado.

## Escopo da primeira versão

### Modelo e artefatos fixos

- modelo `Qwen/Qwen2.5-Coder-7B-Instruct`;
- revisão `c03e6d358207e414f1eca0bb1891e29f1db0e242`;
- direção selecionada na camada 18, posição `-1`;
- J-lens ajustada para as camadas de origem 0 a 26, com alvo na camada 27;
- controle aleatório ortogonal reproduzível com a seed já usada no estudo.

A interface não terá seletor de modelo nesta etapa. Antes de iniciar o servidor,
um preflight verificará revisão, hashes da direção e da lente, dimensões,
finitude, CUDA e presença de uma A100 com memória suficiente.

### Controles principais

- campo de prompt;
- condição: baseline, ablação, adição ou controle aleatório;
- presets:
  - baseline, coeficiente zero;
  - ablação de 20%;
  - adição de 50%;
  - controle aleatório de 20%;
- modo avançado com coeficiente manual entre `0,0` e `1,0`;
- camada da J-lens entre 0 e 26;
- quantidade de previsões exibidas, com padrão top 10;
- temperatura, `top_p`, seed e limite de novos tokens;
- botões para gerar, interromper, limpar e exportar a sessão.

O coeficiente manual será interpretado conforme a condição escolhida. Ablação e
adição continuarão sendo operações distintas, não extremos de um único slider
assinado.

### Saída principal

A tela seguirá a direção visual **Laboratório de intervenção**:

1. controles e prompt à esquerda;
2. resposta do modelo no centro;
3. microscópio da J-lens à direita.

Durante a geração, o painel central receberá os tokens emitidos pelo modelo. No
mesmo passo, o painel da direita receberá:

- camada observada;
- índice do passo de geração;
- token efetivamente produzido;
- top tokens da J-lens;
- score e barra relativa de cada token;
- mudança de posição desde o passo anterior, quando o token já estava no top.

A camada será escolhida antes da geração e ficará bloqueada enquanto a sessão
estiver ativa. Para mudar a camada, a pessoa interrompe ou conclui a geração e
executa novamente o prompt. Isso evita capturar todas as camadas em cada passo
apenas para permitir uma troca visual.

## Arquitetura

### `InterventionEngine`

Responsável por:

- carregar modelo, tokenizer e artefatos uma única vez;
- aplicar chat template;
- construir hooks de baseline, ablação, adição e controle;
- executar geração incremental com cache de atenção;
- capturar a ativação da camada selecionada no token corrente;
- respeitar cancelamento entre passos;
- limpar hooks mesmo quando há erro ou interrupção.

O motor terá uma sessão de geração por vez. Essa restrição evita concorrência
sobre hooks globais e mantém o uso de VRAM previsível.

### `LiveJLensReadout`

Responsável por:

- carregar e validar somente as matrizes já ajustadas;
- receber a ativação da camada selecionada;
- produzir logits e top tokens para o passo corrente;
- devolver estruturas numéricas sem HTML;
- manter o histórico mínimo necessário para calcular variações de rank.

O caminho incremental será validado contra `JacobianLens.apply` em prompts
curtos. Para a mesma camada e posição, os logits e o top-k deverão coincidir
dentro de tolerância numérica. Essa comparação impede que a otimização para
streaming crie uma leitura diferente daquela usada no experimento.

### `GenerationEvent`

Cada passo produzirá um evento com:

- `step`;
- `generated_token_id`;
- `generated_token_text`;
- `accumulated_text`;
- condição e coeficiente efetivos;
- camada observada;
- top-k da J-lens;
- tempo do passo;
- uso de memória da GPU;
- estado terminal ou erro seguro, quando aplicável.

O frontend receberá eventos, mas não terá acesso direto ao modelo ou aos hooks.

### `GradioApp`

Responsável por:

- validar campos da interface;
- iniciar e cancelar uma geração;
- atualizar resposta e readout por streaming;
- bloquear controles incompatíveis durante uma execução;
- mostrar preflight, hardware, revisão e hashes resumidos;
- preparar uma exportação solicitada pela pessoa usuária.

O Gradio não deverá conter lógica científica além de apresentação e validação
de entrada.

## Fluxo de uma interação

1. O runtime inicia a A100 e autentica o acesso ao modelo gated.
2. Modelo, direção e lente são carregados e passam pelo preflight.
3. O Gradio abre um link temporário protegido por usuário e senha gerados para
   a sessão.
4. A pessoa envia o prompt e os controles escolhidos.
5. O backend fixa a seed e instala os hooks da condição.
6. A geração incremental executa um passo.
7. O token final é amostrado e a ativação da camada escolhida é enviada ao
   readout.
8. Um `GenerationEvent` atualiza a resposta e os tokens da J-lens.
9. O ciclo se repete até fim, cancelamento ou limite de tokens.
10. Os hooks são removidos em um bloco `finally`.

## Acesso remoto e privacidade

O servidor será iniciado dentro da sessão Colab A100 por um novo perfil do
orquestrador existente. A primeira versão usará o link temporário do Gradio com
autenticação. A senha será gerada a cada execução, exibida somente no terminal
local e nunca gravada no Git.

O link do Gradio passa por um relay externo. A documentação deverá deixar claro
que prompts e respostas transitam por esse serviço durante a sessão.

Por padrão:

- prompts e gerações permanecem somente em memória;
- não há histórico persistente no servidor;
- logs registram estados operacionais, não conteúdo textual;
- exportação só ocorre ao clicar em **Exportar sessão**;
- o JSON exportado é baixado pelo usuário e não entra no pipeline de resultados;
- direção, lente, token Hugging Face e senha não são expostos pela interface.

Encerrar a sessão A100 invalida a interface e o link temporário.

## Tratamento de erros

### Falha de preflight

A interface não abre se GPU, revisão, direção ou lente estiverem incorretas. O
terminal informa qual contrato falhou sem imprimir credenciais.

### Falta de memória

O backend interrompe a geração, remove hooks, limpa caches temporários e mantém
o processo disponível para uma nova tentativa. Não faz fallback silencioso de
hardware, dtype ou modelo.

### Queda do navegador

O backend cancela a geração quando a desconexão for detectada ou quando a
sessão expirar. Uma reconexão começa sem recuperar prompts anteriores.

### Cancelamento

O botão de interrupção define um sinal consultado entre tokens. A resposta
parcial permanece visível e pode ser exportada, mas recebe estado
`cancelled`.

### Tokenização visual

Tokens com espaços, quebras de linha ou bytes especiais serão exibidos em forma
escapada e também terão seu ID disponível. Assim, a interface não esconde
diferenças importantes do tokenizer.

## Testes

### Motor

- preservação exata do baseline sem hooks;
- paridade dos presets com os coeficientes do estudo;
- coeficiente manual respeitado em ablação, adição e controle;
- controle ortogonal determinístico;
- hooks sempre removidos após sucesso, erro e cancelamento;
- uma única geração concorrente;
- seed reproduzível;
- streaming termina corretamente em EOS e limite.

### J-lens

- camada fora de 0 a 26 rejeitada;
- matriz, shape, dtype, revisão e hash validados;
- top-k incremental igual ao caminho canônico dentro da tolerância;
- tokens especiais renderizados sem ambiguidade;
- somente a camada selecionada participa do readout ao vivo.

### Interface

- campos inválidos bloqueados antes da inferência;
- controles travados e liberados no momento correto;
- cancelamento interrompe novos passos;
- nenhuma entrada textual aparece em logs;
- exportação exige ação explícita;
- senha, token e caminhos locais não aparecem na página.

### Runtime A100

Um smoke test remoto executará um prompt inofensivo curto em baseline e
ablação, verificará pelo menos três eventos sincronizados e encerrará a sessão.
O teste não executará código gerado nem usará prompts MITRE.

## Critérios de aceitação

- a interface abre por link temporário autenticado;
- o Qwen é carregado uma única vez;
- um prompt gera texto token por token;
- cada token produzido atualiza o top-k da J-lens para a camada escolhida;
- os quatro presets e o coeficiente manual funcionam;
- a geração pode ser cancelada sem deixar hooks instalados;
- nenhum prompt é persistido por padrão;
- a sessão exportada contém parâmetros, tokens, scores e tempos suficientes para
  reproduzir a visualização, sem incluir pesos, lente ou credenciais;
- o preflight e todos os testes locais passam;
- a A100 é encerrada após o smoke test.

## Fora do escopo inicial

- troca de modelo;
- ajuste de novas J-lenses pela interface;
- comparação simultânea das quatro condições;
- heatmap de todas as camadas por todos os passos;
- execução de código produzido;
- contas de usuário, banco de dados ou histórico compartilhado;
- publicação permanente da aplicação.

Esses itens podem ser adicionados depois sobre os contratos do motor. O próximo
passo visual mais natural é o comparador causal. O microscópio completo de
camadas por passos vem depois, quando houver medição suficiente do custo do
readout ao vivo.
