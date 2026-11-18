# Limite de saída do Qwen Intervention Lab

## Objetivo

Permitir gerações de até 512 novos tokens na interface Gradio sem alterar o
comportamento inicial da ferramenta.

## Mudança

O controle `Máx. novos tokens` continuará iniciando em 64, mas seu limite
superior passará de 256 para `MAX_NEW_TOKENS_LIMIT`, atualmente definido como
512 no contrato do backend.

Usar a constante existente evita que interface e validação aceitem limites
diferentes.

## Validação

Um teste focado verificará que a interface usa o limite compartilhado. O
runtime continuará interrompendo a geração ao atingir EOS, quando o usuário
cancelar ou ao alcançar o máximo selecionado.

## Fora do escopo

Esta mudança não aumenta a janela de contexto do modelo, não altera o valor
inicial de 64 tokens e não modifica amostragem, streaming ou intervenções.
