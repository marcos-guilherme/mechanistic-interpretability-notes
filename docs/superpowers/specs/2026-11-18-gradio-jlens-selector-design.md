# Seleção de J-lens no laboratório Gradio

## Objetivo

Permitir que a interface do Qwen use a J-lens v1 ou a J-lens v2 durante a
geração, com a v2 selecionada por padrão.

## Comportamento

A interface terá um seletor `J-lens` com duas opções:

- `v2 (contexto longo)`, selecionada por padrão;
- `v1 (100 amostras)`, mantida para comparação.

A lente escolhida será usada para todos os tokens daquela geração. Uma
execução já iniciada não muda de lente no meio do texto. O nome e o hash da
lente usada devem aparecer no status e no arquivo exportado.

## Artefatos

O laboratório verificará e transportará:

- v1: `artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt`,
  SHA-256 `b700a78c90e82f18e057db01cf3dd520beda1f51f1663e2dbf0c1c481efc9e4e`;
- v2: `artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt`,
  SHA-256 `b80c084a5ce083d2fc8589a62c9a9b2542b6a9dc6b5c006018cc4956eef37a7a`.

Uma lente ausente ou com hash incorreto bloqueia a inicialização. As duas
lentes podem ficar carregadas na memória da sessão, mas somente a selecionada
participa do readout de cada geração.

## Verificação

Os testes devem cobrir seleção padrão da v2, seleção explícita da v1,
isolamento por geração, proveniência exportada e transporte dos dois
artefatos. O smoke remoto deve abrir a interface na A100 e executar pelo menos
um prompt curto com cada lente antes de liberar o link para teste manual.
