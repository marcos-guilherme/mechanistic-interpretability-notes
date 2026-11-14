# Portal das replicações

Scaffold Docusaurus 3 em TypeScript para a camada pública do projeto. Os
documentos canônicos não vivem aqui: uma etapa posterior materializará o
recorte permitido pelo `content-manifest.json` em `.generated/`.

## Ambiente

- Node.js 20
- npm 10 ou posterior

```bash
npm install
npm test
npm run sync
npm run check
npm run typecheck
npm run build
```

`npm run sync` valida `content-manifest.json`, materializa somente o recorte
permitido em `.generated/docs` e `.generated/static` e troca a árvore publicada
de forma segura. `npm run check` reconstrói esse recorte temporariamente e
falha se caminhos ou bytes divergirem, sem alterar `.generated/`.

`npm start` e `npm run build` executam a sincronização automaticamente. A pasta
`.generated/` é derivada dos documentos canônicos e não deve ser editada nem
versionada.

Em publicação, defina `DOCUSAURUS_URL` e, quando necessário,
`DOCUSAURUS_BASE_URL`. Na ausência dessas variáveis, a configuração deriva a
identidade de `GITHUB_REPOSITORY`; sem nenhuma identidade pública, usa
`http://localhost:3000` e `/`.
