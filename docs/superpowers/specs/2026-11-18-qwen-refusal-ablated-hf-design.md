# Qwen Coder com Refusal Direction ablada

## Objetivo

Produzir uma versão baixável do `Qwen/Qwen2.5-Coder-7B-Instruct` que aplique
internamente a ablação de 20% da Refusal Direction selecionada no experimento.
O modelo deverá poder ser carregado pelo Hugging Face sem depender do
laboratório Gradio ou dos hooks do repositório.

## Formato

O artefato será um checkpoint completo e versionado, com:

- pesos originais do Qwen na revisão congelada;
- configuração própria que registra a dose de 20%, a camada e o hash da
  direção;
- código de modelagem que aplica a mesma projeção usada no runtime atual;
- carregamento por `AutoModelForCausalLM.from_pretrained` com
  `trust_remote_code=True`;
- tokenizer e template de chat da revisão original;
- model card com método, proveniência, limitações e riscos.

Não será usada uma ortogonalização aproximada dos pesos, pois ela não
reproduziria exatamente a sequência de projeções da intervenção parcial de
20%.

## Equivalência

Antes do upload, a implementação deverá ser comparada com o runtime de hooks
já validado. O teste usará prompts congelados e verificará:

- mesmos tokens gerados em modo guloso;
- logits numericamente equivalentes dentro de tolerância congelada;
- mesma taxa de refusal nos painéis usados no experimento;
- mesma divergência KL;
- preservação do resultado de código na amostra congelada.

Uma divergência fora da tolerância bloqueia a publicação.

## Publicação

A primeira versão será enviada para um repositório privado ou gated no Hugging
Face. Depois da validação remota, o repositório poderá ser tornado público por
decisão explícita.

O model card deverá informar que:

- o modelo é uma variante experimental;
- a intervenção reduz parcialmente o comportamento de refusal;
- os resultados não demonstram remoção universal ou seletiva da recusa;
- a avaliação foi feita em amostras limitadas;
- o uso exige `trust_remote_code=True`;
- a revisão AI-assisted dos pares pertence ao experimento exploratório
  posterior e não valida este checkpoint.

## Segurança e proveniência

O repositório não incluirá prompts privados, gerações de avaliação,
credenciais, ativações ou dados de revisão. Serão publicados apenas os pesos,
o tokenizer, o código necessário para carregamento, a configuração, o model
card e resultados compactos de equivalência.

Os hashes do modelo base, da direção, do código de exportação e do artefato
final serão registrados.
