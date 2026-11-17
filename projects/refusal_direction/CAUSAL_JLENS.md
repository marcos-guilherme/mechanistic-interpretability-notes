# Fidelidade da J-lens sob intervenção na direção de recusa

## Por que fiz este experimento

Os estudos anteriores deixaram duas evidências separadas no
`Qwen/Qwen2.5-Coder-7B-Instruct`. A direção da camada 18 alterou causalmente a
recusa, enquanto a J-lens recuperou parte da decisão final a partir de estados
intermediários. Faltava verificar se a lente continuava descritiva quando o
próprio estado interno do modelo era alterado.

A pergunta deste experimento foi:

> A J-lens continua recuperando o token escolhido pelo modelo quando
> aumentamos ou reduzimos causalmente sua direção de recusa?

O alvo não era uma palavra de recusa escolhida previamente. Em cada passo, a
lente precisava recuperar o token que o modelo interveniente realmente gerou.
Assim, a métrica avalia fidelidade ao modelo alterado, e não concordância com
uma resposta considerada desejável.

## Como o teste foi organizado

Usei 32 prompts MITRE, que representam solicitações ofensivas, e 32 prompts
MITRE-FRR, que representam solicitações defensivas ou sensíveis. São os mesmos
IDs congelados na avaliação de código anterior. Cada prompt foi executado em
quatro condições:

- baseline, sem intervenção;
- ablação de 20% da direção de recusa;
- adição de 50% da direção;
- controle aleatório ortogonal de 20%.

A geração foi gulosa, com até 64 tokens. Nos primeiros oito tokens gerados,
registrei o rank do token realmente escolhido pelo modelo nas camadas 19, 22,
25 e 26. As lentes v1 e v2 leram exatamente o mesmo estado residual em cada
passo. O endpoint principal pré-definido foi `pass@5` na camada 25.

Para comparar cada intervenção ao baseline, usei bootstrap agrupado por prompt
com 10.000 reamostragens, seed 2028 e margem de equivalência de cinco pontos
percentuais. MITRE e MITRE-FRR foram analisados separadamente. O protocolo
completo está em
[`docs/superpowers/specs/2026-11-17-qwen-causal-jlens-fidelity-design.md`](../../docs/superpowers/specs/2026-11-17-qwen-causal-jlens-fidelity-design.md).

## O que foi executado

A execução canônica foi `df86a101-9f5a-4363-8918-5f4c183ee25d`, com o commit
de implementação `251dfd50397da06a008be6f06cf3aa541c9fd8a4`. Foram produzidas:

- 64 prompts;
- quatro condições por prompt;
- 256 respostas;
- 16.384 observações de rank;
- duas versões da lente;
- quatro camadas e oito passos por resposta.

O pipeline terminou com `status=pass`. O piloto real, a matriz esperada, os
IDs, os hashes, as métricas, o bootstrap e a ausência de texto bruto passaram
pela validação offline.

## Resultados comportamentais

A intervenção alterou de fato o comportamento. No MITRE, a taxa de recusa foi
78,125% no baseline, 0% após ablação, 100% após adição e 78,125% no controle
aleatório. No MITRE-FRR, as taxas foram 9,375%, 0%, 100% e 6,25%,
respectivamente.

Portanto, a adição não foi seletiva: induziu recusa em todos os prompts
ofensivos e defensivos. A ablação também removeu todas as recusas nas duas
classes. O controle aleatório ficou próximo do baseline. Esse padrão reforça
a interpretação da direção como uma alavanca causal potente, mas pouco
específica.

## Resultado principal da J-lens

Na camada 25, as lentes v1 e v2 produziram os mesmos resultados de `pass@5`.

No MITRE, o baseline foi 98,047%. A ablação caiu para 93,359%, diferença de
-4,688 pontos percentuais, com intervalo de 95% entre -8,203 e -1,563. Como o
intervalo ultrapassou a margem inferior de -5 pontos, a equivalência não foi
demonstrada. A adição subiu para 100%, diferença de +1,953 pontos, e foi
equivalente ao baseline. O controle aleatório permaneceu em 98,047%, com
diferença zero.

No MITRE-FRR, o baseline foi 98,438%. Ablação e adição atingiram 100%, com
diferença de +1,563 pontos e equivalência demonstrada. O controle aleatório
ficou em 98,047%, diferença de -0,391 ponto, também equivalente.

Assim, sete das oito comparações principais foram equivalentes. A exceção foi
a ablação no conjunto ofensivo, para as duas lentes. Isso não significa que a
lente tenha deixado de funcionar: mesmo nessa condição ela manteve `pass@5`
acima de 93%. Significa que, com a margem pré-definida, não foi possível
afirmar preservação equivalente ao baseline.

## O que apareceu nas demais camadas

A camada 26 foi a mais fiel, com `pass@5` entre 99,609% e 100% em todas as
condições. Como ela está muito próxima da saída, esse resultado é esperado e
não deve ser tratado como leitura antecipada profunda.

As camadas 19 e 22 foram muito menos confiáveis e mais sensíveis à
intervenção. No MITRE baseline, por exemplo, a v1 obteve 14,063% na camada 19,
42,188% na camada 22, 98,047% na camada 25 e 99,609% na camada 26. A leitura
se tornou útil principalmente nas camadas tardias.

A v2, ajustada com mais dados e contexto, não apresentou vantagem consistente
sobre a v1. Nas camadas 25 e 26, os resultados foram quase idênticos. Nas
camadas anteriores, as diferenças variaram conforme conjunto e condição, sem
um ganho geral da v2.

## Interpretação

O resultado sustenta uma conclusão delimitada: a J-lens tardia continuou
descrevendo a decisão do Qwen mesmo quando a direção de recusa produziu
mudanças comportamentais grandes. A lente não apenas repetiu o baseline, pois
o modelo passou de recusar quase sempre para nunca recusar, ou para recusar
sempre, dependendo da intervenção.

Ao mesmo tempo, a queda sob ablação no MITRE indica que essa robustez não é
uniforme. A leitura na camada 25 pode perder alguma fidelidade quando a
intervenção remove um comportamento já presente em grande parte dos prompts.
Esse achado é mais informativo que um `pass` global: mostra onde a descrição
permaneceu estável e onde a equivalência falhou.

O experimento também reforça uma limitação da direção atual. Ela controla
recusa, mas não separa adequadamente intenção ofensiva de uso defensivo. Essa
limitação motiva testar representações mais seletivas, como subespaços com
várias direções ou features obtidas por autoencoders esparsos.

## Limitações

O estudo cobre um modelo, uma direção, uma seed e 32 prompts por conjunto. A
recusa foi detectada por um critério lexical, que não substitui avaliação
semântica humana. A fidelidade foi medida nos oito primeiros tokens e não
avalia se a resposta ou o código estão corretos. As lentes foram ajustadas em
CodeSearchNet, enquanto o painel usa prompts de cibersegurança. A camada 26 é
próxima demais da saída para representar uma previsão muito antecipada.

Equivalência estatística também não significa identidade. Ela afirma apenas
que o intervalo estimado permaneceu dentro da margem escolhida. Na comparação
MITRE com ablação, essa condição não foi satisfeita.

## Incidentes de execução

As primeiras tentativas foram descartadas antes de aceitar resultados. A
checagem de paridade encontrou uma incompatibilidade CPU/GPU porque as
matrizes da lente tinham sido movidas para a GPU antes da comparação com o
caminho canônico. A ordem foi corrigida: primeiro a paridade é executada com
as matrizes em CPU; depois as quatro camadas usadas no painel são mantidas na
GPU.

Uma repetição ainda executou módulos antigos preservados pelo kernel do Colab.
Os módulos do projeto foram removidos do cache do interpretador antes da
tentativa final. Nenhuma das tentativas falhas foi incorporada aos resultados.

## Onde estão as evidências

O resumo canônico está em
[`results/qwen25_coder_7b_causal_jlens/causal_jlens_evaluation.json`](results/qwen25_coder_7b_causal_jlens/causal_jlens_evaluation.json).
Os estágios intermediários e o manifesto ficam no mesmo diretório. Os
artefatos versionáveis contêm métricas, IDs, hashes e proveniência, mas não
contêm prompts nem gerações brutas.

Para repetir a validação:

```powershell
Set-Location projects/refusal_direction
uv run python scripts/validate_qwen_causal_jlens.py
```
