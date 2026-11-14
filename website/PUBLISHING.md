# Checklist do primeiro push e da primeira publicação

Este documento é uma verificação humana. Ele não autoriza a publicação por si
só: o workflow também aplica um gate técnico antes de enviar qualquer artefato
ao GitHub Pages.

## Estado atual

**Não está pronto para publicar.** Neste momento, não há remote Git configurado
e não há um arquivo `LICENSE` na raiz do projeto. A licença third-party da lente
em `projects/jspace_llama/data/anthropic-jlens/LICENSE` cobre aquele material de
terceiros; ela não é a licença raiz deste projeto e não satisfaz o gate de
publicação.

## 1. Identidade do repositório

- [ ] Definir o owner (proprietário, usuário ou organização).
- [ ] Definir o nome do repositório.
- [ ] Confirmar a URL do repositório que será pública.
- [ ] Escolher e proteger a branch principal.
- [ ] Confirmar a autoria e a identidade pública que devem aparecer no site.
- [ ] Revisar o texto do footer (rodapé) e remover o aviso provisório somente
      quando autoria e licença estiverem decididas.
- [ ] Para a primeira publicação, confirmar a URL e o `baseUrl` derivados
      estritamente de `GITHUB_REPOSITORY`.

Não crie remote nem faça o primeiro push antes de esses itens terem uma resposta
explícita e revisada.

## 2. Licença do projeto

- [ ] Escolher uma licença aplicável ao projeto e registrar seu identificador
      SPDX.
- [ ] Criar um arquivo `LICENSE` real, com conteúdo além de whitespace e sem
      placeholders, na raiz do projeto.
- [ ] Conferir separadamente as licenças de conteúdo e dependências de
      terceiros.
- [ ] Confirmar que a licença third-party em
      `projects/jspace_llama/data/anthropic-jlens/LICENSE` não está sendo usada
      como substituta da licença raiz.
- [ ] Conferir se o texto do rodapé representa corretamente a licença escolhida.

## 3. GitHub Pages e GitHub Actions

- [ ] Habilitar GitHub Pages com a origem **GitHub Actions**.
- [ ] Revisar `.github/workflows/deploy-docs.yml` na branch principal.
- [ ] Criar as seguintes **Repository variables** (não usar secrets):
  - `PUBLICATION_APPROVED`: somente `true` libera o fluxo;
  - `PUBLIC_AUTHOR_NAME`: nome público explícito, sem placeholder;
  - `PUBLIC_LICENSE_SPDX`: expressão SPDX válida para a licença raiz, com
    identificadores e exceções existentes.
- [ ] Confirmar que o environment `github-pages` tem as regras de proteção
      desejadas.
- [ ] Manter o acionamento manual (`workflow_dispatch`) até uma decisão
      explícita de automatizar publicações.

O workflow falha antes de qualquer upload quando a aprovação, a identidade
pública, a branch principal, a licença declarada ou o `LICENSE` raiz não passam
pelo gate.

### Domínio próprio: etapa futura

O workflow atual não aceita variável de domínio. Se um domínio próprio for
adotado no futuro, primeiro configure-o em **Settings → Pages**, ajuste DNS e
HTTPS e aguarde a confirmação do GitHub Pages. Só então adapte e valide no
workflow a URL e o `baseUrl`; não antecipe essa divergência na primeira
publicação.

## 4. Auditoria final

- [ ] Revisar textos, autoria, licença, footer, links e figuras.
- [ ] Verificar que nenhum dado sensível ou artefato não público entrou no
      build.
- [ ] Confirmar a correspondência entre afirmações e evidências publicadas.
- [ ] Executar a validação completa com Node 20 em uma instalação limpa:

```powershell
nvm use 20
Set-Location website
npm ci
npx playwright install --with-deps chromium
npm run validate:project-page
```

- [ ] Entender o limite dessa validação: `validate:project-page` usa uma URL
      local fixa e o `baseUrl` fixo `/ioi-replication/`; ela não representa a
      identidade final do repositório.
- [ ] Confirmar separadamente a geração final feita pelo workflow, depois de
      derivar URL e `baseUrl` de `GITHUB_REPOSITORY`.
- [ ] Conferir a auditoria final do build destinado à URL pública:
      `npm run audit:build`.
- [ ] Executar manualmente o workflow e revisar a URL retornada pelo deploy
      antes de divulgar o site.
