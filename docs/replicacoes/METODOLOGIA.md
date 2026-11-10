# Metodologia comum de replicação

Construí esta metodologia enquanto avançava pelas três replicações. Ela reúne
os cuidados que se tornaram necessários na prática: separar resultado do
artigo e resultado local, registrar amostras efetivas, controlar intervenções,
preservar proveniência e tratar falhas de ambiente como inconclusivas.

Uma replicação útil precisa permitir duas leituras ao mesmo tempo. A primeira é
conceitual: qual afirmação científica está sendo testada, por que cada
experimento responde a ela e até onde a conclusão alcança. A segunda é
operacional: quais dados, versões, configurações, intervenções e artefatos
permitem repetir a execução e auditar o resultado. Os notebooks podem conectar
essas leituras, mas não substituem código testado nem artefatos canônicos.

## 1. Formular a pergunta científica

Comece com uma pergunta que possa ser respondida por observações e controles
definidos antes da execução. Expresse a hipótese principal, o resultado que a
apoiaria e as condições que tornariam o estudo negativo ou inconclusivo. Uma
mudança comportamental, isoladamente, não demonstra um mecanismo: a pergunta
deve indicar que relação entre representação, intervenção e efeito será
testada.

Registre também os limites da interpretação. Uma direção que controla um
comportamento em uma amostra, por exemplo, não representa automaticamente todo
o conceito associado a esse comportamento nem garante generalização para
outros modelos, tarefas ou distribuições.

## 2. Delimitar escopo e fidelidade

Declare o que será reproduzido do trabalho de referência e o que ficará fora.
Fixe, quando aplicável, artigo, repositório upstream, commit, modelo e revisão.
Separe três categorias:

- elementos fiéis, cuja definição e procedimento são mantidos;
- adaptações necessárias, como hardware, biblioteca ou tamanho de amostra;
- extensões locais, que respondem a perguntas novas e não devem ser atribuídas
  ao artigo.

Fidelidade não significa copiar toda decisão histórica sem avaliação. Quando
uma incompatibilidade ou correção exigir mudança, preserve a intenção
experimental, documente o desvio e avalie seu efeito provável.

## 3. Estabelecer a linha de base

A linha de base confirma que o modelo, a tarefa e a métrica funcionam antes da
intervenção. Ela deve usar o mesmo pré-processamento, prompts, configurações de
geração e população de avaliação das comparações posteriores, salvo quando uma
diferença for parte explícita do experimento.

Registre resultados por exemplo quando isso for seguro e permitido, antes de
agregá-los. Verifique casos de sucesso e falha, porque uma média plausível pode
ocultar uma tarefa mal formada, vazamento entre conjuntos ou uma métrica que
mede outra propriedade.

## 4. Capturar e instrumentar

Defina exatamente onde e quando o estado interno é observado: componente,
camada, posição, lado do hook, etapa de tokenização e forma dos tensores. Testes
pequenos devem conferir alinhamento de tokens, remoção dos hooks, dtype,
dispositivo e ausência de mutação acidental.

A instrumentação precisa ser semanticamente neutra. Compare uma execução
instrumentada sem intervenção com a linha de base ou use um controle no-op. Se
o simples ato de capturar ativações altera a saída, o experimento causal ainda
não está isolado.

## 5. Realizar a intervenção causal

Especifique a operação matematicamente e no código: estado de entrada, direção
ou componente alterado, coeficiente, camadas e posições afetadas. A seleção da
intervenção deve usar dados distintos da avaliação final sempre que a própria
seleção otimizar a métrica que será reportada.

Uma evidência causal combina efeito direcionado, especificidade e preservação.
Além de verificar se a variável-alvo muda, examine se a saída continua coerente
e se o resultado não decorre apenas de degradação global, vazamento do alvo ou
mudança não controlada na geração.

## 6. Incluir controles informativos

Os controles dependem da hipótese, mas normalmente incluem linha de base, no-op
e perturbação aleatória pareada por escala. Também podem ser necessários
controle de posição, camada, sinal, componente alternativo, prompt ou conjunto
de referência.

Um controle só é comparável se compartilhar a mesma amostra e o mesmo
procedimento, alterando apenas o fator que pretende testar. Resultados
semelhantes entre intervenção-alvo e controle enfraquecem a interpretação
específica, mesmo quando ambos mudam a métrica principal.

## 7. Validar resultado e execução

A validação deve combinar testes de software e critérios científicos. Testes
rápidos conferem fórmulas, contratos de artefatos, determinismo e falhas
seguras. A execução completa verifica os critérios pré-declarados, os controles
e medidas de preservação. Quando couber, reporte incerteza e resultados por
semente, não apenas uma estimativa pontual.

Não transforme ausência de evidência em evidência de ausência. Falha de
download, ativo incompatível, baseline inválida, amostra insuficiente ou
controle contaminado tornam a execução inconclusiva; não constituem resultado
negativo sobre a hipótese.

## 8. Registrar proveniência

Cada artefato canônico deve identificar o suficiente para reconstruir sua
origem: modelo e revisão, fontes e hashes dos dados permitidos, versões do
ambiente, hardware, dtype, sementes, configuração de geração, código upstream
fixado, commit local e estado de alterações locais quando disponível.

Prefira gerar resumos a partir dos artefatos-fonte. Valores transcritos
manualmente para READMEs tendem a divergir; se forem úteis para leitura, devem
apontar para o arquivo canônico do qual foram derivados.

## 9. Distinguir amostra inicial de amostra efetiva

Registre separadamente quantos exemplos foram reunidos, quantos foram
planejados e quantos realmente contribuíram para cada métrica. Documente
duplicatas, filtros, erros de tokenização, saídas ausentes, exclusões e
interseções entre conjuntos. A amostra efetiva é o denominador da análise e não
pode ser inferida apenas pelo tamanho inicial do arquivo.

Filtros definidos depois de observar resultados podem introduzir viés. Quando
forem inevitáveis, preserve a contagem antes e depois, explique o motivo e, se
possível, apresente uma análise de sensibilidade.

## 10. Documentar desvios

Mantenha um registro explícito de desvios do artigo e do plano local:
dependências substituídas, revisão de modelo, hardware, redução de amostra,
mudança de métrica, correção de bug ou interrupção de execução. Para cada
desvio, descreva motivo, momento e possível impacto.

Desvios não invalidam automaticamente uma replicação. O problema é ocultá-los
ou comparar valores como se o protocolo fosse idêntico. A conclusão deve ser
proporcional à fidelidade realmente alcançada.

## 11. Proteger dados e resultados

Não versione segredos, credenciais, tokens, dados restritos, prompts nocivos,
gerações brutas potencialmente nocivas, pesos, caches ou tensores pesados.
Mantenha esse material em locais ignorados e com acesso mínimo. Artefatos
versionados devem conter apenas métricas agregadas, configurações, hashes,
identificadores permitidos e proveniência necessária.

Antes de compartilhar um notebook ou log, revise células de saída e metadados:
eles podem reter conteúdo bruto ou credenciais mesmo quando o código-fonte está
limpo. Um hash referencia um artefato local sem publicá-lo, mas não remove a
obrigação de controlar esse arquivo.

## 12. Encerrar recursos de GPU

Toda execução remota ou sob demanda deve ter encerramento explícito, inclusive
após erro. Use uma etapa de limpeza garantida e mantenha um comando de
recuperação independente para interromper a sessão caso o fluxo principal seja
interrompido.

No manifesto, registre se a sessão foi encerrada e como essa condição foi
verificada. Terminar o processo local ou fechar um notebook não prova que a GPU
remota deixou de consumir recursos.

## Critério de encerramento do estudo

Uma replicação está pronta para ser apresentada quando a pergunta e o escopo
estão claros, a linha de base é válida, intervenção e controles são
comparáveis, as amostras efetivas e os desvios estão registrados, os artefatos
canônicos carregam proveniência suficiente e a execução pode ser repetida sem
expor dados sensíveis. Se algum desses pontos faltar, descreva o estado como
parcial ou planejado, em vez de antecipar uma conclusão.

Para aplicar este processo a um novo projeto, use o
[roteiro copiável](MODELO.md). O [índice das replicações](README.md) mostra
como os estudos atuais ligam narrativa, código e resultados.
