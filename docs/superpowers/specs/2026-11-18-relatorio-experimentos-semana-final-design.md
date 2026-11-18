# Relatório dos experimentos da semana final

## Objetivo

Produzir um relatório local de cinco a sete páginas que registre os
experimentos realizados na décima semana do estudo sobre interpretabilidade
aplicada ao alinhamento de LLMs.

## Forma

O texto será escrito em primeira pessoa e seguirá o processo da semana, sem
adotar formato de artigo nem tom promocional. O relatório terá:

1. contexto e mudança para um modelo de código;
2. transferência e ajuste da Refusal Direction no Qwen2.5-Coder;
3. J-lens v2 e comparação com v1 e logit lens;
4. teste da J-lens durante intervenções;
5. avaliação de refusal e capacidade de código;
6. laboratório visual em Gradio;
7. experimento esparso preparado, sem afirmar resultado ainda inexistente;
8. limitações e fechamento.

Serão usados quadros compactos para os números que ajudam a interpretar as
conclusões. Nomes de arquivos, hashes e detalhes internos de implementação só
aparecerão quando forem necessários para compreender a proveniência.

## Limites científicos

O relatório deve distinguir:

- execução concluída de hipótese confirmada;
- fidelidade descritiva da J-lens de qualidade ou correção da geração;
- efeito causal da direção de seletividade entre pedidos ofensivos e
  defensivos;
- resultados concluídos de protocolos ainda aguardando execução.

A revisão dos 256 pares do experimento esparso será descrita como
`AI-assisted`, e a execução pesada continuará marcada como pendente.

## Artefatos

Serão gerados localmente:

- `docs/entregas/2026-11-18-relatorio-experimentos-semana-final.md`;
- `docs/entregas/2026-11-18-relatorio-experimentos-semana-final.pdf`.

Os dois arquivos serão ignorados pelo Git. Um arquivo LaTeX temporário poderá
ser usado para renderização e removido após a compilação.

## Verificação

O PDF deve abrir corretamente, ter entre cinco e sete páginas, conter o nome
do autor e a data, e não apresentar texto cortado. O Markdown e o PDF devem
permanecer no disco após a renderização e não aparecer em `git status`.
