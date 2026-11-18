# Steering esparso da direção de recusa

Estado: aguardando execução

## Contexto

A direção densa congelada do Qwen controla recusa, mas os experimentos
anteriores não demonstraram seletividade suficiente entre pedidos ofensivos e
defensivos. Este estudo testa uma decomposição esparsa dessa mesma direção.

## Pergunta

Remover da reconstrução esparsa os componentes associados ao lado defensivo
preserva o efeito sobre pedidos ofensivos e reduz o efeito colateral defensivo?

## Dados

O protocolo prevê um corpus não rotulado de cinco milhões de tokens para o SAE,
256 pares revisados divididos em descoberta (128), seleção (64) e teste (64),
um painel externo MITRE/MITRE-FRR e HumanEval+. Textos, gerações, ativações e
pesos permanecem fora do Git.

## Método

Será ajustado um SAE TopK na entrada residual da camada 18 da revisão congelada
do `Qwen2.5-Coder-7B-Instruct`. A direção densa será reconstruída com
coeficientes assinados; componentes associados ao lado defensivo no split de
descoberta serão removidos. Cardinalidade e dose serão congeladas no split de
seleção antes da abertura do teste. Direção densa, reconstrução sem filtro,
vetor filtrado e controle aleatório serão comparados por adição e ablação.

## Critérios de sucesso

O SAE e a reconstrução precisam passar seus gates de qualidade. No teste, os
gates de Pareto de adição e ablação serão avaliados separadamente por bootstrap
pareado de 10.000 reamostragens, seed 2029. Um ganho pontual não basta.

## Execução

A execução pesada está pendente e será feita somente em A100. Antes dela, o
validador offline e os contratos locais usam apenas fixtures JSON compactas.

## Resultados

Ainda não há resultados. Nenhuma hipótese foi classificada como aprovada ou
reprovada.

## Incidentes

Nenhum incidente de execução científica foi registrado, pois a execução ainda
não começou.

## Limitações

O protocolo cobre um modelo, uma revisão, uma camada, um SAE e painéis
congelados. Associações de features serão interpretações locais ao recorte, não
conceitos universais.

## Evidências

Após a execução, esta seção apontará somente para resultados compactos que
passarem pelo validador offline independente. A especificação e o plano
aprovados permanecem as fontes do método enquanto o estado for aguardando
execução.
