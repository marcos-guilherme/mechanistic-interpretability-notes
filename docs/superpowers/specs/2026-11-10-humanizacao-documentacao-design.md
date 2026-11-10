# Humanização seletiva da documentação

## Objetivo

Revisar a documentação narrativa do repositório para que ela comunique, em
primeira pessoa singular, o processo prático de entrada do autor na área de
interpretabilidade mecanística. O texto deve mostrar motivações, decisões,
dificuldades, resultados e aprendizados sem perder a precisão necessária para
reproduzir e auditar os experimentos.

O README geral deve deixar explícito que o repositório não é apenas uma coleção
de implementações. Ele é um registro organizado do percurso de estudo e
experimentação do autor, incluindo tentativas bem-sucedidas, resultados
parciais e execuções inconclusivas.

## Voz editorial

A primeira pessoa singular será usada quando o texto tratar de:

- motivação para criar o repositório ou escolher um experimento;
- decisões tomadas durante a implementação;
- dificuldades e incidentes encontrados;
- interpretação cuidadosa do que foi observado;
- aprendizados e relação do estudo com a formação do autor.

O tom será pessoal, direto e sóbrio. A revisão não deve introduzir linguagem
promocional, dramatização, falsa certeza ou frases genéricas sobre paixão e
inovação. A voz humana deve aparecer por meio de decisões concretas e limites
assumidos.

## Escopo

### README geral

O `README.md` da raiz será reescrito em português. A abertura explicará:

1. que o autor criou o repositório para armazenar e documentar sua entrada
   prática em interpretabilidade mecanística;
2. que os estudos foram organizados como uma progressão entre circuitos,
   direções no espaço residual e intervenções conceituais;
3. que resultados negativos, parciais ou inconclusivos são mantidos como parte
   do processo;
4. como navegar entre documentação, projetos e artefatos canônicos.

A descrição do J-space será atualizada para registrar que houve uma execução
em L4, mas ela foi inconclusiva antes do experimento científico por falta de
autorização para acessar o modelo gated.

### READMEs dos projetos

As aberturas dos três READMEs serão humanizadas para explicar por que cada
replicação foi escolhida e o que o autor buscou aprender. Seções operacionais,
comandos, versões, tabelas e contratos de validação continuarão objetivas.

### Documentos `PROCESSO.md`

As sínteses acadêmicas, transições, incidentes, contribuições para o TCC,
limitações e próximos passos poderão usar primeira pessoa. Métricas,
procedimentos, fórmulas e descrições dos artefatos permanecerão factuais.

### Documentação compartilhada

O índice `docs/replicacoes/README.md` ganhará uma abertura autoral que explique
o papel dos três estudos no percurso. A introdução de
`docs/replicacoes/METODOLOGIA.md` poderá registrar que a metodologia foi
construída a partir das necessidades encontradas durante as replicações.
O arquivo `MODELO.md` permanecerá neutro, pois sua função é servir como roteiro
reutilizável.

## Limites da revisão

- Não alterar números, resultados, hashes, versões ou critérios científicos.
- Não converter comandos, tabelas ou schemas em narrativa pessoal.
- Não esconder falhas operacionais nem suavizar limitações.
- Não afirmar que o J-space foi testado cientificamente nesta execução.
- Não reescrever código, notebooks ou artefatos JSON.
- Não substituir explicações técnicas por relatos autobiográficos.

## Critérios de aceitação

- O primeiro parágrafo do README geral identifica claramente o propósito
  pessoal e formativo do repositório.
- A primeira pessoa aparece de forma natural nos trechos narrativos, sem ser
  repetida em cada parágrafo.
- Cada projeto continua distinguindo intenção, execução, evidência e
  conclusão.
- O estado dos três estudos permanece correto: IOI e Refusal Direction com
  resultados observados; J-space inconclusivo por bloqueio de acesso.
- Todos os links e comandos existentes continuam válidos.
- Testes e validadores dos três projetos continuam passando.
