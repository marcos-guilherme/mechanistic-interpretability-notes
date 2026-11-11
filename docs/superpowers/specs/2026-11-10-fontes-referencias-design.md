# Design: documento de fontes e referências

## Objetivo

Criar um ponto de entrada único para as fontes que orientaram as três
replicações do repositório. O documento deve distinguir claramente materiais
de formação, artigos originais, implementações de referência e páginas dos
modelos. Ele será uma seleção comentada, não uma revisão sistemática da
literatura nem uma bibliografia exaustiva de dependências.

Como material complementar, criar uma síntese aprofundada em português de
*How To Become A Mechanistic Interpretability Researcher*. A síntese deve
preservar as ideias e a progressão argumentativa úteis para pesquisa sem
reproduzir ou traduzir o texto integralmente.

## Local e integração

O documento será criado em `docs/REFERENCIAS.md`, com o título **Fontes e
referências**. O `README.md` da raiz e `docs/replicacoes/README.md` receberão
links curtos para ele. Os documentos de cada experimento continuarão contendo
as referências locais necessárias; o novo arquivo funcionará como índice
explicativo comum, sem substituir essas citações.

A síntese será criada em
`docs/GUIA_PESQUISA_EM_INTERPRETABILIDADE.md`. A entrada dedicada ao texto de
Neel Nanda em `docs/REFERENCIAS.md` apontará tanto para o artigo original
quanto para essa versão comentada em português.

## Estrutura

1. **Como ler este documento:** explica os quatro papéis atribuídos às fontes:
   formação, artigo original, implementação e modelo.
2. **Formação em interpretabilidade mecanística:** apresenta os dois textos de
   Neel Nanda indicados pelo autor do repositório:
   - *How To Become A Mechanistic Interpretability Researcher*;
   - *An Extremely Opinionated Annotated List of My Favourite Mechanistic
     Interpretability Papers v2*.
3. **Fundamentos:** reúne apenas fontes conceituais diretamente úteis para
   compreender circuitos de transformers e as ferramentas usadas nas
   replicações, incluindo o *Transformer Circuits* framework e
   TransformerLens.
4. **Circuito IOI:** identifica *Interpretability in the Wild* como artigo
   original e explica sua relação com o estudo em `projects/ioi`.
5. **Direção de recusa:** identifica *Refusal in Language Models Is Mediated
   by a Single Direction* como artigo original, além do código upstream e da
   página do Gemma.
6. **J-space:** identifica *A global workspace in language models* como
   trabalho original, separando o código `jacobian-lens`, o pipeline `jspace`,
   a lente publicada e a página do Llama.

## Estrutura da síntese em português

A síntese será explicitamente apresentada como um texto independente, baseado
no artigo e escrito com palavras próprias. Ela acompanhará os principais
blocos do original:

1. a proposta de aprender o mínimo necessário e começar a pesquisar;
2. fundamentos técnicos e prática inicial;
3. miniprojetos e ciclos rápidos de feedback;
4. passagem a projetos de pesquisa completos;
5. ideação, exploração, entendimento e apresentação dos resultados;
6. ceticismo, controles, priorização e critérios de boa ciência;
7. escrita, comunicação, mentoria e desenvolvimento profissional;
8. aplicação dessas ideias ao percurso IOI → direção de recusa → J-space.

O texto destacará quais recomendações são específicas de interpretabilidade
mecanística e quais são transferíveis para pesquisa empírica em geral. Não
haverá tradução frase a frase, reprodução de trechos extensos ou aparência de
tradução oficial. A autoria do original e o link direto aparecerão no início.

## Formato de cada entrada

Cada entrada terá:

- título e link direto;
- autor ou organização e ano, quando confirmados pela fonte;
- classificação do papel da fonte;
- comentário curto, em primeira pessoa, explicando como ela orientou o estudo;
- vínculo com o projeto correspondente quando houver.

Os comentários não atribuirão a textos didáticos a autoria dos experimentos.
Também não transformarão recomendações pessoais de leitura em consenso da
área. Commits ou revisões fixadas serão preservados quando já fazem parte da
proveniência do repositório.

## Critérios de seleção e verificação

Entram apenas fontes com influência clara sobre a formação, o desenho
experimental ou a implementação. Bibliotecas auxiliares, ferramentas
operacionais e datasets secundários ficam fora, salvo quando forem necessários
para explicar a origem direta de um protocolo.

Antes da escrita final:

- confirmar título, autoria, ano e URL nas páginas originais;
- conferir os links e revisões já registrados nos arquivos `PROCESSO.md`;
- garantir que cada experimento tenha seu artigo original identificado;
- verificar que os dois textos de Neel Nanda apareçam como fontes formativas;
- conferir que a síntese remeta claramente ao artigo de Neel Nanda e não seja
  apresentada como tradução integral ou oficial;
- evitar duplicação extensa com os documentos dos projetos.

## Validação

A conclusão exige:

- busca por links quebrados ou placeholders no novo documento;
- conferência manual dos links externos principais;
- comparação da síntese com o artigo para verificar cobertura temática sem
  reprodução textual extensa;
- `git diff --check`;
- ausência de erros de lint nos arquivos Markdown alterados.

