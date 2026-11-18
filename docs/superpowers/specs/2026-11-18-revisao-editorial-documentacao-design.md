# Revisão editorial da documentação pública

## Objetivo

Renomear a documentação para **Interpretabilidade Aplicada ao Alinhamento de
LLMs** e aproximar o texto público da escrita usada pelo autor no termo de
entrega e nos relatos do projeto.

## Critério editorial

O texto deve:

- usar primeira pessoa quando descrever decisões e atividades do autor;
- preferir frases concretas, dizendo o que foi testado, observado ou concluído;
- manter termos técnicos necessários, com explicação curta quando aparecem;
- remover tom promocional, institucional ou genérico;
- substituir rótulos criados para a documentação, como “atlas observacional”,
  por descrições diretas do experimento;
- preservar números, estados experimentais, limitações e distinções entre
  observação e causalidade.

## Escopo

A revisão cobre a página inicial, configuração e metadados do Docusaurus, os
documentos públicos listados no manifesto e os testes que verificam esses
textos. Arquivos históricos de especificação e planejamento não serão
reescritos.

O termo “atlas observacional” será substituído por expressões contextuais como
“avaliação da J-lens em 93 prompts” ou “análise sem intervenção”, conforme a
frase. “Portal” será evitado quando “documentação” ou “site” for mais natural.

## Verificação

Depois da revisão, a sincronização de conteúdo, os testes editoriais, o
type-check e o build público devem passar. Uma busca final deve confirmar que
o título antigo e “atlas observacional” não aparecem no conteúdo publicado.
