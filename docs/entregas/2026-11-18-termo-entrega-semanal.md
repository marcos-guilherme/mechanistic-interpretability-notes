# Termo de entrega semanal

**Participantes da Entrega [matriculados em Residência em IA]:**  
Marcos Guilherme Dias de Oliveira Alves

**Entrega:** Gate de fechamento dos estudos práticos de Interpretabilidade
Mecanicista, com integração entre Refusal Direction e J-lens.

## Tema e foco da última semana

Na décima e última semana, meu foco foi integrar dois métodos de
Interpretabilidade Mecanicista: Refusal Direction e J-lens. Apliquei
intervenções de refusal no Qwen2.5-Coder e usei a lente para acompanhar como
essas mudanças apareciam nas representações internas e na geração do modelo.

Também deixei de trabalhar apenas com modelos de propósito geral e escolhi um
modelo voltado para código. Essa escolha permitiu aproximar os experimentos de
problemas que me interessam diretamente, como segurança, comportamento de
refusal e preservação da capacidade de programação após uma intervenção.

Esse trabalho deu continuidade às replicações parciais que eu havia concluído
na semana anterior:

- **Circuit Study:** *Interpretability in the Wild: a Circuit for Indirect
  Object Identification in GPT-2 Small*, de Wang et al. (2022). O artigo
  investiga como partes do GPT-2 Small participam de uma tarefa linguística.
- **Refusal Direction:** *Refusal in Language Models Is Mediated by a Single
  Direction*, de Arditi et al. (2024). O trabalho investiga uma direção interna
  associada ao comportamento de refusal.
- **J-space e J-lens:** *Verbalizable Representations Form a Global Workspace
  in Language Models*, de Gurnee et al. (2026). O artigo apresenta o J-space e
  uma forma de observar representações internas que podem ser verbalizadas.

## Conclusões das replicações

- **Circuito IOI:** foi a reprodução mais próxima do artigo. Repeti o
  comportamento principal e as intervenções propostas no circuito.
- **Refusal Direction:** reproduzi o efeito central em um recorte menor e
  transferi o método para um modelo de código. A direção altera o comportamento
  de refusal, mas não é totalmente seletiva e também pode afetar outros
  comportamentos.
- **J-space:** a reprodução foi parcial. A lente ajudou a observar mudanças
  internas do modelo, mas a tentativa causal no Llama ficou inconclusiva.

Os três artigos foram reproduzidos em escalas diferentes, com adaptações aos
modelos e recursos disponíveis. Não considero nenhuma das três uma reprodução
integral dos trabalhos originais.

Na integração realizada nesta semana, concluí que os dois métodos podem ser
usados em conjunto. A Refusal Direction permite alterar o comportamento do
modelo, enquanto a J-lens permite acompanhar parte do que muda durante a
geração. A lente continuou oferecendo uma leitura útil mesmo após as
intervenções, embora tenham aparecido diferenças em alguns cenários.

## Finalização

No fechamento do percurso:

- reuni as três reproduções e o experimento de integração em um repositório
  documentado;
- desenvolvi uma interface em Gradio para testar as intervenções e acompanhar
  a J-lens durante a geração;
- preparei um experimento com Autoencoder Esparso para investigar a
  seletividade da Refusal Direction;
- registrei também os resultados negativos e inconclusivos, sem apresentá-los
  como confirmações;
- consolidei a ferramenta visual e a documentação produzida durante o estudo.

**Ambientes utilizados:**

- RTX 3050 local para o circuito IOI;
- Google Colab com GPUs L4 e A100 para Refusal Direction, J-space e J-lens;
- modelos GPT-2 Small, Gemma, Llama 3.1 e Qwen2.5-Coder.

**Artefato final:** repositório experimental com as reproduções, os resultados,
a documentação e a interface de inspeção, organizado como base para a escrita
do TCC.
