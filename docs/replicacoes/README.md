# Replicações

Usei esta área para reunir a documentação comum das três replicações e evitar
que decisões importantes ficassem espalhadas apenas entre notebooks, código e
logs. Aqui explico como organizei o trabalho e onde encontrar a evidência de
cada estudo.

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
- [Avaliação final canônica](../../projects/refusal_direction/results/final_evaluation.json)
- [Manifesto da execução](../../projects/refusal_direction/results/execution_manifest.json)

### J-space Llama — execução inconclusiva

O núcleo reutilizável e a orquestração de uma intervenção causal mínima em
J-space sobre Llama estão implementados, incluindo validação de runtime,
intervenções, controles e o notebook de entrada. Depois que o acesso ao modelo
foi aprovado, uma execução em NVIDIA L4 carregou e verificou a lente, executou
readout, baseline, no-op, controle aleatório e troca-alvo. Nenhum alpha passou
pelos critérios do prompt principal: o baseline não forneceu uma resposta
numérica válida, `spider` não apareceu no readout e a intervenção-alvo vazou
`ant` e se degradou por repetição. O estado permanece inconclusivo e não
constitui evidência contra o mecanismo mais amplo de J-space.

- [Visão geral e fluxo implementado](../../projects/jspace_llama/README.md)
- [Processo de estudo e reprodução](../../projects/jspace_llama/PROCESSO.md)
- [Avaliação final canônica](../../projects/jspace_llama/results/final_evaluation.json)
- [Manifesto da execução](../../projects/jspace_llama/results/execution_manifest.json)

## Como ler este repositório

Nos READMEs e notebooks, apresento a pergunta, o raciocínio e as limitações de
cada estudo. Os artefatos canônicos versionados continuam sendo a fonte dos
valores observados. Se a narrativa divergir deles, trato isso como um problema
a investigar e corrigir, não como uma oportunidade de escolher o número mais
conveniente. Também mantenho as afirmações dos artigos identificadas como
referências externas, separadas do que medi neste repositório.
