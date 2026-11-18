# Replicações

Usei esta área para reunir a documentação comum das três replicações e evitar
que decisões importantes ficassem espalhadas apenas entre notebooks, código e
logs. Aqui explico como organizei o trabalho e onde encontrar a evidência de
cada estudo.

Como porta de entrada cronológica para J-space e Direção de Recusa, o
[diário e painel dos experimentos](../ACOMPANHAMENTO_EXPERIMENTOS.md) reúne
decisões, incidentes, resultados e evidências dos dois estudos. Ele complementa
este índice e a metodologia, sem substituir nenhum dos dois.

Ao longo dos experimentos, procurei manter duas leituras em paralelo: uma
conceitual, voltada a entender a hipótese e o papel de cada intervenção, e
outra operacional, voltada a permitir que a execução seja repetida e
auditada. A [metodologia comum](METODOLOGIA.md) reúne os critérios que adotei;
o [modelo de replicação](MODELO.md) ajuda a aplicá-los em um novo estudo.

As [fontes e referências](../REFERENCIAS.md) registram os artigos originais,
os materiais de formação e as implementações que orientaram os três estudos.

## Índice dos estudos

### IOI

Replicação do circuito de identificação do objeto indireto em GPT-2 Small, com
uma etapa didática e outra voltada à fidelidade ao artigo. O projeto contém
ambiente próprio, notebooks, testes e artefatos canônicos de resultados.

- [Visão geral e instruções de reprodução](../../projects/ioi/README.md)
- [Processo de estudo e reprodução](../../projects/ioi/PROCESSO.md)
- [Resumo canônico dos resultados](../../projects/ioi/results/paper/summary.json)
- [Manifesto da execução](../../projects/ioi/results/paper/execution_manifest.json)

### Refusal Direction

Replicação parcial do resultado causal de direção de recusa em
`google/gemma-2b-it`. O estudo compara linha de base, ablação, adição e controle
aleatório. O Git preserva métricas agregadas e valores numéricos por exemplo,
além de configurações e proveniência, sem os textos dos prompts ou das
gerações.

- [Visão geral, escopo e instruções de reprodução](../../projects/refusal_direction/README.md)
- [Processo de estudo e reprodução](../../projects/refusal_direction/PROCESSO.md)
- [Avaliação final canônica (Gemma)](../../projects/refusal_direction/results/final_evaluation.json)
- [Manifesto da execução (Gemma)](../../projects/refusal_direction/results/execution_manifest.json)
- Piloto causal **congelado** em `Qwen/Qwen2.5-Coder-7B-Instruct`
  (`causal_gate_failed`; operação concluída, gate de selectividade falhou):
  [piloto](../../projects/refusal_direction/results/qwen25_coder_7b/pilot_evaluation.json)
  · [avaliação](../../projects/refusal_direction/results/qwen25_coder_7b/final_evaluation.json)
- Follow-up exploratório no Qwen, sem reclassificar o piloto:
  [dose](../../projects/refusal_direction/results/qwen25_coder_7b_dose/dose_followup_evaluation.json)
  · [J-lens](../../projects/refusal_direction/results/qwen25_coder_7b_jlens/jlens_evaluation.json)
  · [avaliação de código](../../projects/refusal_direction/results/qwen25_coder_7b_code_eval/04_finalize.json)

### J-space Llama: tentativa causal e avaliação da J-lens em 93 prompts

Organizei este estudo em três etapas distintas. A primeira é a tentativa causal
já executada: depois que o acesso ao modelo foi aprovado, uma execução em
NVIDIA L4 carregou e verificou a lente, executou readout, baseline, no-op,
controle aleatório e a troca `spider → ant`. O baseline não forneceu uma
resposta numérica válida, `spider` não apareceu no readout e a intervenção
vazou `ant` e se degradou por repetição. O resultado permanece inconclusivo,
foi preservado sem alteração e não constitui evidência contra o mecanismo mais
amplo de J-space.

A segunda etapa é uma análise sem intervenção, agora executada e validada. Ela
lê os 93 prompts multihop da Anthropic, em uma revisão fixada e sob licença
Apache 2.0, compara J-lens direta e logit lens e mantém um painel qualitativo
fixo de 12 prompts com `gradient_pursuit`. O protocolo usa prompt bruto, não
fornece `target` ao modelo, não gera texto e não edita ativações. O `success`
obtido significa que a execução e as métricas foram validadas, não que uma
relação causal tenha sido demonstrada.

Na *band* 12–28, os `pass@1/5/10` foram `0,344086 / 0,548387 / 0,596774` para
a J-lens e `0,284946 / 0,392473 / 0,435484` para a logit lens. Nas camadas
ajustadas 0–30, foram `0,344086 / 0,564516 / 0,629032` e
`0,284946 / 0,392473 / 0,446237`, respectivamente. Dos 103 intermediários, 82
eram single-token; nesse subconjunto, a recuperação foi maior, mas continua
restrita a este tokenizer, esta lente e este conjunto. O heatmap concentra
muitos ranks menores nas camadas intermediárias e tardias, sem uma fronteira
uniforme, e o painel combina exemplos claros com falhas. `spider`, por exemplo,
chegou ao rank 2 da J-lens na camada 19. O `gradient_pursuit` é mantido
separado do ranking direto porque decompõe ativações por coeficientes, em vez de
medir o mesmo rank.

Uma terceira etapa, se for realizada, será uma nova confirmação causal. Ela
precisará de hipótese, prompts ainda não usados nesta avaliação e controles
pré-declarados próprios, sem selecionar retrospectivamente conceito, direção,
camada, `alpha` ou itens *held-out* a partir destes resultados.

- [Visão geral e fluxo implementado](../../projects/jspace_llama/README.md)
- [Processo de estudo e reprodução](../../projects/jspace_llama/PROCESSO.md)
- [Avaliação causal final](../../projects/jspace_llama/results/final_evaluation.json)
- [Manifesto da execução causal](../../projects/jspace_llama/results/execution_manifest.json)
- [Resultado da avaliação da J-lens](../../projects/jspace_llama/results/jspace_observation.json)
- [Manifesto da execução observacional](../../projects/jspace_llama/results/jspace_observation_manifest.json)

## Como ler este repositório

Nos textos e notebooks, apresento a pergunta, o raciocínio e as limitações de
cada estudo. Os artefatos versionados continuam sendo a fonte dos valores
observados. Se a narrativa divergir deles, trato isso como um problema a
investigar e corrigir, não como uma oportunidade de escolher o número mais
conveniente. Também mantenho as afirmações dos artigos identificadas como
referências externas, separadas do que medi neste repositório.
