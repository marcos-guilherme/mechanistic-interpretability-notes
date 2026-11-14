# Portal público das replicações — desenho aprovado

**Data:** 14 de novembro de 2026
**Estado:** aprovado para implementação incremental

## Objetivo

O portal será a camada pública, em português, para leitura das três
replicações de interpretabilidade mecanística. Ele deve facilitar a navegação
por perguntas, métodos, resultados e limitações sem substituir a documentação
científica existente nem transformar uma execução operacional em uma
conclusão mais forte.

O público principal inclui estudantes e pesquisadores que desejam entender o
percurso experimental e auditar as evidências canônicas. A experiência deve
ser legível para uma primeira visita e, ao mesmo tempo, permitir que uma
pessoa tecnicamente experiente alcance os JSONs e a proveniência associados a
cada afirmação.

## Decisões de produto

- O portal é público, escrito em português e cientificamente curado.
- Os Markdown existentes continuam sendo a fonte canônica. O portal não cria
  uma segunda versão editorial desses textos.
- O projeto Docusaurus fica isolado em `website/`.
- A sincronização lê somente a allowlist de `website/content-manifest.json` e
  materializa derivados descartáveis em `website/.generated/`.
- A configuração e o workflow `.github/workflows/deploy-docs.yml` ficam
  preparados nesta etapa para um futuro GitHub Pages, sem criar ou alterar
  remote, executar deploy ou publicar o site.
- Notebooks, código-fonte, pesos, caches, segredos, artefatos brutos e áreas de
  planejamento interno não fazem parte do conteúdo público.

## Arquitetura de conteúdo

### Camada canônica

Os documentos na raiz, em `docs/` e nos diretórios dos três projetos
permanecem editáveis em seus locais atuais. Links, revisão científica e
histórico pertencem a esses arquivos. JSONs de resultado, manifestos de
execução, figuras aprovadas e a licença do conjunto de avaliação
`lens-eval-multihop` também permanecem em seus caminhos de origem. Os JSONs
completos de J-space continuam canônicos no repositório, mesmo quando não
integram o recorte público.

Nenhum arquivo sob `website/.generated/` pode ser editado manualmente ou
tratado como fonte. A pasta deve poder ser apagada e reconstruída sem perda de
informação.

### Navegação pública

A barra lateral terá três grupos, na ordem definida pelo manifesto:

1. **Início:** visão geral do repositório e índice das replicações;
2. **Método e percurso:** metodologia comum, guia de pesquisa, referências e
   acompanhamento dos experimentos;
3. **Estudos:** pares de visão geral e processo para IOI, Direção de Recusa e
   J-space.

Cada página possui `id`, `slug`, título, grupo de sidebar e ordem estáveis no
manifesto. O `slug` é a URL pública; o caminho de origem não deve vazar para a
estrutura de URL. IDs e slugs não podem se repetir.

Os JSONs e as figuras não recebem páginas narrativas independentes nesta
etapa. Eles são evidências e assets endereçáveis a partir dos documentos que
os citam. A licença Apache do conjunto de avaliação `lens-eval-multihop` deve
acompanhar o material observacional correspondente.

### Hierarquia editorial

A página inicial apresenta o objetivo do repositório, os três estudos e uma
explicação curta da diferença entre resultado operacional, evidência
observacional e conclusão causal. Cada estudo deve conduzir primeiro à visão
geral e depois ao relato de processo e às evidências.

O portal não uniformiza conclusões diferentes. Em particular, J-space deve
preservar a cronologia: tentativa inicial bloqueada por acesso, execução
causal canônica operacionalmente completa e cientificamente inconclusiva
(`primary_gates_failed`), e atlas posterior validado, mas não causal.

## Manifesto e regras de sincronização

`website/content-manifest.json` é o contrato único de publicação. Ele contém:

- `schema_version`, para evolução validada do formato;
- `generated_root`, fixado em `website/.generated`;
- `public_pages`, com origem, identidade, rota e posição editorial;
- `allowed_assets`, com as figuras PNG explicitamente revisadas;
- `allowed_evidence`, com o recorte público de JSONs canônicos, manifestos e
  licença permitidos;
- `deny_prefixes` e `deny_globs`, como defesa adicional contra publicação
  acidental.

A allowlist é exata: descobrir um arquivo novo no repositório não o torna
publicável. Para publicar uma nova página ou evidência, é necessário revisar e
adicionar seu caminho ao manifesto. As regras de negação têm precedência; uma
entrada simultaneamente permitida e negada deve fazer a sincronização falhar,
nunca contornar a negação. Elas são defesa secundária aplicada às fontes de
conteúdo: a existência do workflow em `.github/`, prefixo negado para
publicação, não o transforma em conteúdo do portal nem cria conflito com a
allowlist.

Todos os caminhos são relativos à raiz do repositório, usam `/`, não podem ser
absolutos, conter `..`, links simbólicos que escapem da raiz ou diferenças de
caixa que colidam no destino. O sincronizador deve rejeitar origens ausentes,
destinos duplicados e extensões incompatíveis com o tipo declarado.

## Pipeline

O pipeline de construção seguirá esta ordem:

1. carregar o manifesto como JSON estrito e validar a versão conhecida;
2. validar unicidade, caminhos, extensões, existência das origens e regras de
   negação;
3. criar uma árvore temporária no mesmo volume de
   `website/.generated/`;
4. copiar somente páginas, assets e evidências explicitamente permitidos;
5. acrescentar aos Markdown derivados apenas o front matter necessário ao
   Docusaurus e reescrever links entre páginas, figuras e evidências
   permitidas para suas rotas ou cópias públicas;
6. converter links para notebooks, scripts, `MODELO.md`, ferramentas e demais
   destinos negados ou não publicados em texto não clicável; quando uma URL
   real do repositório estiver explicitamente configurada, o sincronizador
   pode apontar para ela sem copiar o destino;
7. verificar que os arquivos gerados correspondem exatamente à allowlist;
8. substituir `website/.generated/` pela árvore validada;
9. executar verificação de links e o build de produção do Docusaurus.

A sincronização não modifica os Markdown de origem. Uma falha antes da troca
final não pode deixar uma árvore pública parcialmente atualizada. O build deve
ser determinístico para a mesma revisão do repositório, lockfile e manifesto;
datas de execução não devem ser injetadas nas páginas.

Links entre documentos permitidos devem apontar para seus slugs. Links para
evidências e figuras permitidas devem apontar para as cópias publicadas. A
presença de um link em uma página permitida não publica nem torna permitido o
seu destino. Um destino negado ou ausente da allowlist não torna o conjunto
impossível: ele deve ser convertido em texto não clicável ou, somente com URL
real configurada, em link para o repositório. A validação deve falhar se o
derivado mantiver um link local não resolvido ou tentar copiar o destino.

## Segurança e publicação

O portal publica apenas arquivos versionados e explicitamente listados. A
sincronização deve operar sem credenciais e nunca ler `.env`, caches de
modelos, pesos, outputs de notebooks, `artifacts/`, `tools/colab`,
`docs/superpowers`, `docs/entregas`, código, scripts, configurações internas,
`.github/`, `.cursor/`, credenciais ou chaves. O workflow pode existir em
`.github/` porque esse diretório é negado como conteúdo publicável, não como
infraestrutura versionada.

JSON é servido como evidência estática, sem execução. SVGs não integram a
allowlist, inclusive pelo risco de `foreignObject`, HTML incorporado e outros
conteúdos ativos; as figuras públicas usam somente PNG. Markdown e MDX
derivados não podem introduzir imports ou expressões executáveis a partir de
metadados não validados.

A evidência pública de J-space inclui `final_evaluation.json`,
`jspace_observation_manifest.json`, as figuras PNG e a licença do conjunto de
avaliação. `jspace_observation.json` e o `execution_manifest.json` causal
contêm prompts brutos e, por isso, permanecem canônicos no repositório, mas
não são copiados para `static` nem incluídos na allowlist pública.

A preparação para GitHub Pages inclui build estático, URLs relativas corretas,
página 404 e o workflow `.github/workflows/deploy-docs.yml`. O workflow deve
derivar a URL pública e o `baseUrl` da identidade pública efetivamente
configurada — organização ou usuário, nome do repositório e, se houver,
domínio — sem inventar valores ou depender de um remote local.

O workflow deve falhar de modo fechado antes de qualquer deploy enquanto a
identidade pública e a licença aplicável ao portal não estiverem
explicitamente configuradas e validadas. Nesta etapa ele pode verificar e
construir, mas não recebe credencial de publicação, não cria remote, não
publica branch, não chama APIs de deploy e não publica o site.

Antes de uma publicação futura, uma revisão humana deve conferir os textos,
licenças, figuras, links externos, dados potencialmente sensíveis e a
correspondência entre afirmações e artefatos.

## Layout editorial acadêmico

A apresentação deve privilegiar leitura longa e rastreabilidade:

- largura de texto confortável, hierarquia tipográfica sóbria e espaçamento
  consistente;
- cabeçalho compacto com nome do projeto, navegação e busca;
- sidebar persistente em telas largas e recolhível em telas estreitas;
- sumário local para documentos longos;
- blocos semanticamente distintos para estado, método, resultado, evidência e
  limitação;
- figuras com legenda, fonte e link para o arquivo correspondente;
- tabelas com cabeçalhos claros e rolagem horizontal apenas quando necessária;
- links de evidência visíveis junto à afirmação que sustentam.

A identidade visual deve ser acadêmica, não promocional: fundo neutro,
contraste alto, uma cor de destaque contida e ausência de carrosséis, contagem
animada, parallax ou cartões decorativos sem função informativa. O tema escuro
pode existir, desde que mantenha a mesma hierarquia e contraste.

## Acessibilidade

O objetivo mínimo é WCAG 2.2 nível AA nas páginas centrais. A implementação
deve oferecer:

- navegação completa por teclado, foco visível e link para pular ao conteúdo;
- landmarks e ordem de títulos sem saltos artificiais;
- contraste AA para texto, controles, links e estados;
- zoom de 200% sem perda de conteúdo ou sobreposição;
- texto alternativo ou descrição adjacente para cada figura publicada;
- legendas e explicações que não dependam apenas de cor;
- tabelas com associação entre cabeçalhos e células;
- respeito a `prefers-reduced-motion`;
- idioma da página definido como `pt-BR` e rótulos de interface em português.

Resultados automatizados não substituem verificação por teclado, leitor de
tela e inspeção das descrições científicas.

## Testes e critérios de sucesso

A primeira versão é aceita somente quando:

1. o manifesto é JSON válido, tem schema reconhecido e passa na validação de
   contrato;
2. todas as origens permitidas existem e nenhum caminho negado chega a
   `website/.generated/` ou ao build;
3. IDs, slugs, destinos e posições de sidebar são únicos e determinísticos;
4. a geração feita duas vezes a partir do mesmo estado não produz diff;
5. links internos, âncoras, figuras e downloads de evidência permitidos
   resolvem, e destinos não publicados aparecem como texto ou como link para
   uma URL real do repositório explicitamente configurada;
6. o build de produção termina sem warning de link quebrado;
7. testes de fumaça cobrem `/`, as páginas centrais, os seis documentos dos
   estudos e a página 404;
8. uma varredura do artefato final não encontra notebooks, `.drawio`, `.env`,
   pesos, caches, `artifacts`, `docs/superpowers`, `docs/entregas` ou
   `tools/colab`;
9. auditoria automatizada de acessibilidade não encontra violações críticas ou
   sérias, seguida de revisão manual de teclado, foco, headings e figuras;
10. a configuração e o workflow derivam URL e `baseUrl` da identidade pública
    configurada e bloqueiam deploy sem identidade e licença, sem criar remote,
    publicar branch ou realizar deploy nesta etapa.

O conteúdo também deve passar por revisão científica: títulos e estados
correspondem aos Markdown canônicos; links de evidência apontam para os JSONs
corretos; o atlas J-space é descrito como observacional; limitações não são
ocultadas pelo layout.

## Exclusões e limites

Esta etapa não inclui:

- criação ou publicação de notebooks, código de experimentos ou executores;
- inclusão de outputs de notebook, logs, prompts ou gerações brutas;
- publicação de pesos, ativações, caches, sidecars em `artifacts/` ou dados
  fora da licença explicitamente permitida;
- edição automática dos documentos canônicos durante o build;
- tradução para outros idiomas, comentários, autenticação, analytics,
  newsletter ou backend;
- busca semântica, visualização interativa dos JSONs ou reexecução de
  experimentos no navegador;
- afirmação de que o portal reproduz, valida ou amplia os resultados
  científicos;
- criação de remote ou domínio, habilitação de credenciais/permissões de
  deploy, execução do workflow de deploy ou publicação no GitHub Pages.

O portal é uma camada de leitura e auditoria de um recorte aprovado. Ampliar
esse recorte exige revisão editorial, atualização explícita do manifesto e
novos testes de segurança e proveniência.
