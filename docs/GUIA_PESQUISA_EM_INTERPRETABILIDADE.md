# Guia de pesquisa em interpretabilidade mecanística

Este texto é uma síntese comentada, em português e escrita com palavras
próprias, de [*How To Become A Mechanistic Interpretability
Researcher*](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher),
de Neel Nanda. Não é uma tradução integral nem oficial. Organizei aqui as
ideias que considero mais úteis para aprender interpretabilidade mecanística e
para conduzir pesquisa empírica de modo mais geral.

O artigo original é amplo: discute formação técnica, escolha de projetos,
execução de experimentos, desenvolvimento de julgamento científico, escrita,
mentoria e carreira. Esta síntese preserva essa progressão, mas dá mais espaço
ao que dialoga com os estudos deste repositório. Recomendo consultar o texto de
Nanda para exemplos, links e recomendações específicas que não reproduzo aqui.

## A ideia central: aprender fazendo

A tese prática do guia é que não é necessário dominar toda a área antes de
começar a pesquisar. É preciso construir uma base que permita formular uma
pergunta, executar um experimento e perceber quando algo está errado. A partir
daí, projetos reais revelam com mais precisão quais lacunas técnicas precisam
ser preenchidas.

Essa inversão é importante. Um plano puramente preparatório tende a crescer
sem limite: sempre existe mais álgebra linear, mais arquitetura de modelos,
mais literatura e mais ferramentas para estudar. Sem um problema concreto,
fica difícil decidir o que merece profundidade agora e o que pode esperar. O
experimento transforma conhecimento abstrato em uma necessidade observável.

Aprender fazendo não significa improvisar sem rigor. Significa alternar estudo
e prática em ciclos curtos:

1. aprender o suficiente para tentar;
2. construir uma versão pequena do experimento;
3. observar onde a compreensão ou a implementação falha;
4. estudar essa lacuna específica;
5. repetir com controles e critérios melhores.

Esse ciclo produz duas formas de progresso ao mesmo tempo. A primeira é
técnica: aprende-se a carregar modelos, capturar ativações, construir hooks,
medir efeitos e depurar tensores. A segunda é científica: aprende-se a
distinguir uma observação interessante de uma conclusão sustentada.

## Etapa 1: construir uma base suficiente

A primeira etapa deve oferecer autonomia mínima, não uma sensação ilusória de
completude. Para interpretabilidade mecanística, essa base combina quatro
camadas.

A primeira é entender um transformer em detalhe suficiente para acompanhar o
fluxo da informação. Embeddings, atenção, MLPs, normalização, fluxo residual e
unembedding não podem ser apenas nomes. É necessário saber que tensores cada
componente recebe e produz, como as posições interagem e onde uma intervenção
pode alterar o cálculo.

A segunda é desenvolver intuição linear. Grande parte da área formula
perguntas em termos de direções, projeções, decomposições e subespaços.
Conseguir raciocinar sobre produto interno, norma, mudança de base e projeção é
mais útil inicialmente do que acumular formalismo sem conexão com os
experimentos.

A terceira é aprender as famílias de métodos. Atribuição direta aos logits,
*activation patching*, *path patching*, ablação, adição de direções, probes e
lentes respondem a perguntas diferentes. Saber executar uma técnica não basta;
é preciso entender qual afirmação ela autoriza. Uma correlação com o logit, uma
ativação localizada e uma intervenção causal não são evidências equivalentes.

A quarta é dominar o fluxo de trabalho: notebooks, ambientes isolados,
controle de versões, testes pequenos, inspeção de shapes, seeds e registro de
artefatos. Em pesquisa computacional, uma hipótese pode parecer falsa quando o
problema real é uma posição de token incorreta, um hook aplicado no lugar
errado ou uma versão diferente da biblioteca.

Nessa etapa, amplitude vale mais que perfeição. O objetivo é reconhecer as
principais ideias, saber onde buscar detalhes e conseguir implementar um
experimento simples. Quando a preparação deixa de desbloquear ações novas,
provavelmente é hora de avançar para um projeto.

Assistentes de IA podem acelerar explicações, exemplos de código e depuração,
mas não devem funcionar como fonte final de verdade. Saídas precisam ser
conferidas contra shapes reais, documentação, código upstream e artigos. Uma
resposta plausível é especialmente perigosa quando elimina o erro sem
preservar o significado científico da operação.

## Etapa 2: aprender com miniprojetos

Miniprojetos transformam conhecimento básico em habilidade de pesquisa. O
escopo deve permitir uma resposta em poucos dias, não uma reprodução completa
de um artigo. Bons exemplos incluem testar uma cabeça conhecida em prompts
novos, comparar duas métricas de patching, verificar a sensibilidade a uma
posição ou construir um controle que tente explicar um efeito por uma causa
mais simples.

O objetivo principal não é produzir um resultado publicável. É praticar
rápido:

- formular uma pergunta operacional;
- escolher uma métrica que realmente responda a ela;
- construir a menor execução informativa;
- visualizar resultados intermediários;
- detectar falhas de implementação;
- tentar refutar a interpretação inicial;
- registrar o que mudou depois do teste.

Na fase exploratória, vale escolher o próximo passo pela quantidade de
incerteza que ele pode eliminar, não por parecer sofisticado. Se não sabemos
se o problema está no dataset, na tokenização ou no hook, um experimento
simples que separa essas hipóteses vale mais do que ampliar o sweep.

Quando aparece um resultado forte, a postura deve mudar. Em vez de
imediatamente explicá-lo, deve-se tentar quebrá-lo. O efeito continua sob outra
seed? O baseline está correto? Um controle aleatório produz algo parecido? A
métrica muda porque o comportamento mudou ou porque o denominador ficou
pequeno? A seleção do exemplo favoreceu o resultado? Quanto mais atraente a
descoberta, maior a prioridade de testar explicações alternativas.

O aprendizado central dos miniprojetos é o ciclo de feedback. Executar,
inspecionar e corrigir em horas cria intuições que seriam difíceis de obter
apenas com leitura. Também ensina quando abandonar uma direção: um projeto que
não responde mais a perguntas novas pode ter cumprido sua função formativa,
mesmo sem uma conclusão positiva.

## Etapa 3: avançar para projetos completos

Um projeto completo exige mais do que aumentar o número de amostras. Ele
precisa de uma pergunta delimitada, critérios de sucesso e fracasso,
proveniência, controles, validação e uma narrativa capaz de conectar operação
e evidência.

Trabalhar em ciclos de uma ou duas semanas ajuda a manter essas exigências
visíveis. Ao final de cada ciclo, convém fazer uma retrospectiva:

- o que aprendi que não sabia no início?
- qual hipótese ficou mais ou menos provável?
- quais resultados dependem de decisões frágeis?
- qual é o experimento mais informativo agora?
- há motivo concreto para continuar, ou apenas custo já investido?

Continuar deve ser uma decisão baseada em evidência de progresso. Mudar de
direção não é desperdiçar trabalho quando o ciclo produziu uma conclusão
clara, um método reutilizável ou uma hipótese descartada. Persistência sem
aprendizado pode apenas prolongar um projeto mal definido.

Projetos maiores também exigem habilidades que amadurecem mais devagar:
escolher problemas relevantes, avaliar literatura, reconhecer quando um
resultado é novo, priorizar entre caminhos plausíveis e decidir o nível de
evidência necessário para sustentar uma afirmação. Essas capacidades não são
pré-requisitos que precisam estar prontas antes do projeto; desenvolvem-se com
comparação, retrospectiva e feedback de outras pessoas.

## Quatro movimentos da pesquisa

O processo de pesquisa fica mais claro quando dividido em quatro movimentos
com objetivos diferentes.

**Ideação** escolhe o problema. Aqui importam relevância, viabilidade, conexão
com trabalhos anteriores e possibilidade de obter feedback em tempo razoável.
Uma ideia ampla precisa virar uma pergunta para a qual exista uma operação e
uma observação possíveis.

**Exploração** constrói intuição. É a fase de abrir os dados, fazer gráficos,
testar casos pequenos, variar detalhes e descobrir quais fenômenos realmente
existem. O código pode ser provisório, mas as observações precisam ser
registradas para que a exploração não vire uma sequência sem memória.

**Entendimento** transforma um padrão em uma afirmação resistente. Entram
controles, ablações, hipóteses alternativas, novas amostras e testes
adversariais. O objetivo já não é encontrar algo interessante, mas descobrir
sob quais condições ele permanece verdadeiro e o que o explica.

**Comunicação** organiza evidência para outra pessoa. Isso inclui selecionar a
pergunta central, separar resultado de interpretação, expor limitações e
permitir que o leitor encontre os artefatos que sustentam cada número. Escrever
pode revelar buracos no raciocínio; por isso, não deve ser adiado até o fim.

Esses movimentos não formam uma linha rígida. Uma tentativa de explicar o
resultado pode revelar a necessidade de nova exploração; um controle pode
invalidar a ideia inicial; a escrita pode mostrar que duas métricas estavam
respondendo a perguntas diferentes. O valor da divisão está em reconhecer qual
tipo de trabalho está sendo feito e qual critério deve orientá-lo.

## Ciclos de feedback e tomada de decisão

Habilidades de pesquisa operam em ritmos diferentes. Algumas melhoram em
minutos ou horas: escrever um hook, executar um teste, encontrar um erro de
shape e comparar duas curvas. Outras exigem vários ciclos: escolher
experimentos informativos, perceber quando um resultado não sustenta a
interpretação e decidir se é hora de mudar de direção. Julgamento sobre ideias
e importância pode levar meses ou anos.

Tentar desenvolver tudo ao mesmo tempo cria uma exigência impossível. No
início, é mais produtivo investir nas habilidades com feedback rápido e usar
projetos bem delimitados para praticar as intermediárias. As habilidades
lentas crescem pela exposição repetida a decisões, resultados e críticas.

Um bom registro reduz o custo desses ciclos. Para cada experimento, vale
anotar:

- a pergunta;
- a operação realizada;
- o resultado esperado sob cada hipótese;
- o que foi observado;
- as limitações descobertas;
- a decisão seguinte e seu motivo.

Esse registro impede que a memória reorganize retrospectivamente o projeto
como se tudo tivesse sido planejado. Também facilita distinguir um resultado
negativo de uma execução inválida. Se o modelo não carregou ou o dataset não
foi construído, nenhuma hipótese científica foi testada.

Na priorização, o próximo experimento deve reduzir uma incerteza importante.
Mais escala nem sempre produz mais entendimento. Às vezes, uma asserção
manual, um exemplo inspecionado em detalhe ou um controle simples decide entre
duas explicações melhor que uma grade extensa de parâmetros.

## Ceticismo e boa ciência

Pesquisa empírica precisa ser construída para sobreviver ao entusiasmo do
próprio pesquisador. Interpretabilidade é especialmente vulnerável a histórias
plausíveis: mapas de atenção, direções lineares e visualizações podem parecer
explicativos antes de demonstrarem causalidade ou especificidade.

Um baseline válido vem primeiro. Antes de explicar por que o modelo executa
uma tarefa, é necessário confirmar que ele realmente a executa no conjunto
avaliado e que a métrica seleciona a posição e os tokens corretos. Intervenções
devem ser comparadas com no-ops, controles aleatórios ou referências
semânticas adequadas. Quando uma direção é escolhida e avaliada no mesmo
conjunto, essa dependência precisa limitar a conclusão.

Resultados devem ser lidos em sua escala real. Médias escondem variação;
estimativas pequenas podem ser dominadas por ruído; uma taxa perfeita depois
de filtrar exemplos pelo próprio comportamento do modelo não descreve a
população original. Intervalos, contagens efetivas e exemplos individuais
ajudam a impedir conclusões maiores que os dados.

É igualmente importante fazer as coisas simples que funcionam: conferir
tokenização, imprimir posições, validar extremos conhecidos, comparar duas
implementações da mesma fórmula, fixar revisões e reconstruir resumos a partir
dos artefatos. Rigor não depende de um método sofisticado; muitas vezes depende
de tornar impossível que um erro silencioso pareça evidência.

Uma boa conclusão também sabe parar. Se dois nós passam um limiar apenas na
estimativa pontual, a incerteza precisa aparecer. Se o controle não mede a
mesma propriedade, ele não demonstra especificidade completa. Se a execução
parou antes da inferência, o resultado é inconclusivo, não negativo.

## Escrita, mentoria e desenvolvimento profissional

Escrever faz parte do método porque obriga a ligar pergunta, operação e
evidência. Um documento útil não apresenta apenas o número final: explica de
qual artefato ele veio, qual população usa, quais escolhas o produziram e o
que ainda não permite concluir.

O leitor conhece menos contexto do que o autor imagina. Termos retomados
precisam de uma lembrança curta; métricas parecidas precisam de definições;
mudanças de convenção devem ser sinalizadas. Figuras e notebooks ajudam, mas
não substituem uma narrativa que diga por que cada experimento existe.

Trabalho público e auditável também é uma forma de demonstrar capacidade.
Código executável, resultados com proveniência e uma análise honesta de
limitações mostram mais sobre prática científica do que uma lista de assuntos
estudados. Um resultado inconclusivo bem documentado pode ser valioso quando
ensina onde a cadeia experimental parou e como repetir a tentativa.

Feedback externo acelera habilidades lentas. Mentores e colegas podem perceber
que uma pergunta está ampla demais, que um controle não isola a variável ou
que a contribuição já apareceu na literatura. O objetivo não é buscar apenas
pessoas conhecidas, mas encontrar interlocutores capazes de dedicar atenção ao
trabalho e oferecer crítica concreta.

Também convém separar relevância de popularidade. Uma técnica em evidência pode
não ser a melhor ferramenta para a pergunta. A escolha deve partir do tipo de
evidência necessário e da capacidade de executar um teste informativo.

## O que é específico da interpretabilidade e o que é geral

Algumas competências deste guia pertencem diretamente à interpretabilidade
mecanística: entender o fluxo residual, localizar hooks, manipular caches,
projetar ativações, construir ablações e acompanhar caminhos entre
componentes. Elas dependem da arquitetura do modelo e das convenções da
biblioteca.

Outros princípios se transferem para quase toda pesquisa empírica:

- começar com uma pergunta pequena o bastante para receber feedback;
- tornar hipóteses distinguíveis por observações;
- preferir experimentos informativos a experimentos apenas grandes;
- tratar resultados surpreendentes como provisórios;
- procurar controles e explicações alternativas;
- registrar decisões antes de conhecer o resultado;
- separar falha operacional de evidência científica;
- escrever cedo para testar a coerência do raciocínio;
- declarar o limite exato da conclusão.

Essa distinção permite usar o guia sem transformar interpretabilidade em um
modelo universal de ciência. Hooks e circuitos são ferramentas locais; ciclos
de feedback, ceticismo e comunicação são hábitos mais gerais.

## Como isso aparece neste repositório

O projeto de **IOI** funcionou como etapa de aprendizagem das ferramentas.
Parti de uma tarefa e de um circuito publicados, reconstruí primeiro as
operações em exemplos pequenos e depois aproximei dados, métricas e
intervenções do artigo. Essa progressão tornou visível a diferença entre
atribuir um logit, localizar uma ativação e demonstrar necessidade causal.

A **direção de recusa** foi um problema causal mais compacto. A hipótese de uma
direção no fluxo residual permitiu comparar ablação, adição, controle
ortogonal e preservação. Ao mesmo tempo, o projeto mostrou que um efeito forte
continua limitado pela seleção, pela seed, pelo modelo e pelo conjunto em que
foi escolhido.

O estudo de **J-space** elevou a ambição para uma intervenção em um conceito
intermediário. O software e o fluxo de execução chegaram ao ambiente remoto,
mas o acesso aos pesos falhou antes da inferência. Registrar esse estado como
inconclusivo aplica um princípio geral deste guia: a cadeia de evidência
precisa existir antes da interpretação. Ausência de medição não é resultado
negativo.

Os três estudos também ilustram a passagem de prática guiada para perguntas
mais abertas. Essa passagem não elimina a necessidade de retornar aos
fundamentos. Cada novo experimento revelou detalhes de tokenização, ambiente,
proveniência ou desenho causal que exigiram estudo adicional. O aprendizado
não ocorreu antes dos projetos; ocorreu por meio deles.

As fontes dos experimentos e outros materiais formativos estão reunidos em
[`REFERENCIAS.md`](REFERENCIAS.md). Para os exemplos, links e recomendações
completas de Neel Nanda, consulte o [artigo
original](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher).
