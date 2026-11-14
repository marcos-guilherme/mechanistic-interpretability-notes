import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, readdir, rm, stat, symlink, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  ManifestError,
  publishStaging,
  syncContent,
} from '../scripts/sync-content.mjs';

const encoder = new TextEncoder();

function baseManifest() {
  return {
    schema_version: 1,
    generated_root: 'website/.generated',
    public_pages: [
      {
        source: 'docs/README.md',
        id: 'inicio',
        slug: '/inicio',
        title: 'Início',
        sidebar_label: 'Começar',
        sidebar: 'principal',
        order: 10,
      },
      {
        source: 'docs/other.md',
        id: 'outra',
        slug: '/outra',
        title: 'Outra página',
        sidebar: 'principal',
        order: 20,
      },
    ],
    allowed_assets: [
      {
        source: 'media/chart.png',
        destination: 'assets/demo/chart.png',
        kind: 'figure',
      },
    ],
    allowed_evidence: [
      {
        source: 'results/result.json',
        destination: 'evidence/demo/result.json',
        kind: 'result',
      },
    ],
    deny_prefixes: ['private/', 'website/.generated/'],
    deny_globs: ['**/notebooks/**', '**/scripts/**', '**/*.ipynb', '**/*.svg'],
  };
}

async function makeFixture(t, {manifest = baseManifest()} = {}) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sync-content-'));
  t.after(() => rm(root, {recursive: true, force: true}));

  await mkdir(path.join(root, 'website'), {recursive: true});
  await mkdir(path.join(root, 'docs', 'guide'), {recursive: true});
  await mkdir(path.join(root, 'media'), {recursive: true});
  await mkdir(path.join(root, 'results'), {recursive: true});
  await mkdir(path.join(root, 'private'), {recursive: true});
  await mkdir(path.join(root, 'notebooks'), {recursive: true});

  await writeFile(
    path.join(root, 'docs', 'README.md'),
    [
      '# Documento canônico',
      '',
      '[Página](other.md#secao)',
      '[`Página em código`](other.md)',
      '[Página',
      'multilinha](other.md)',
      '![Figura](../media/chart.png)',
      '[Evidência](../results/result.json)',
      '[Externo](https://example.com/x)',
      '[Email](mailto:test@example.com)',
      '[Âncora](#local)',
      '[Privado](../private/secret.md)',
      '[Notebook](../notebooks/demo.ipynb)',
      '[Diretório](guide)',
      '',
      '`[inline](other.md)`',
      '',
      '```md',
      '[fenced](other.md)',
      '```',
      '',
    ].join('\n'),
  );
  await writeFile(path.join(root, 'docs', 'other.md'), '# Outra\n\n## Secao\n');
  await writeFile(path.join(root, 'docs', 'guide', 'README.md'), '# Guia\n');
  await writeFile(path.join(root, 'private', 'secret.md'), '# Segredo\n');
  await writeFile(path.join(root, 'notebooks', 'demo.ipynb'), '{}');
  await writeFile(path.join(root, 'media', 'chart.png'), encoder.encode('fake-png'));
  await writeFile(path.join(root, 'results', 'result.json'), '{"ok":true}\n');
  await writeFile(
    path.join(root, 'website', 'content-manifest.json'),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  return root;
}

async function listFiles(root) {
  const found = [];

  async function visit(directory) {
    for (const entry of await readdir(directory, {withFileTypes: true})) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else {
        found.push(path.relative(root, absolute).replaceAll(path.sep, '/'));
      }
    }
  }

  await visit(root);
  return found.sort();
}

async function snapshotTree(root) {
  const result = new Map();
  for (const relative of await listFiles(root)) {
    result.set(relative, await readFile(path.join(root, relative)));
  }
  return result;
}

test('gera páginas com frontmatter e copia somente artefatos permitidos', async (t) => {
  const root = await makeFixture(t);

  const result = await syncContent({root});
  const generated = path.join(root, 'website', '.generated');
  const page = await readFile(path.join(generated, 'docs', 'inicio.md'), 'utf8');
  const otherPage = await readFile(
    path.join(generated, 'docs', 'outra.md'),
    'utf8',
  );

  assert.deepEqual(result.counts, {pages: 2, assets: 1, evidence: 1});
  assert.match(
    page,
    /^---\nid: inicio\ntitle: "Início"\nslug: \/inicio\nsidebar_position: 10\nsidebar_label: "Começar"\nhide_title: true\n---\n\n# Documento canônico/,
  );
  assert.match(otherPage, /sidebar_label: "Outra página"/);
  assert.deepEqual(await listFiles(generated), [
    'docs/inicio.md',
    'docs/outra.md',
    'static/assets/demo/chart.png',
    'static/evidence/demo/result.json',
  ]);
  assert.deepEqual(
    await readFile(path.join(generated, 'static', 'assets', 'demo', 'chart.png')),
    Buffer.from(encoder.encode('fake-png')),
  );
});

test('reescreve links permitidos e neutraliza links negados sem tocar em código', async (t) => {
  const manifest = baseManifest();
  manifest.public_pages.push({
    source: 'docs/guide/README.md',
    id: 'guia',
    slug: '/guia',
    title: 'Guia',
    sidebar: 'principal',
    order: 30,
  });
  const root = await makeFixture(t, {manifest});

  await syncContent({root});
  const page = await readFile(
    path.join(root, 'website', '.generated', 'docs', 'inicio.md'),
    'utf8',
  );

  assert.match(page, /\[Página\]\(\/outra#secao\)/);
  assert.match(page, /\[`Página em código`\]\(\/outra\)/);
  assert.match(page, /\[Página\nmultilinha\]\(\/outra\)/);
  assert.match(page, /!\[Figura\]\(\/assets\/demo\/chart\.png\)/);
  assert.match(page, /\[Evidência\]\(\/evidence\/demo\/result\.json\)/);
  assert.match(page, /\[Externo\]\(https:\/\/example\.com\/x\)/);
  assert.match(page, /\[Email\]\(mailto:test@example\.com\)/);
  assert.match(page, /\[Âncora\]\(#local\)/);
  assert.match(page, /Privado \(disponível no repositório-fonte\)/);
  assert.match(page, /Notebook \(disponível no repositório-fonte\)/);
  assert.match(page, /\[Diretório\]\(\/guia\)/);
  assert.match(page, /`\[inline\]\(other\.md\)`/);
  assert.match(page, /```md\n\[fenced\]\(other\.md\)\n```/);
});

test('falha quando uma imagem local não está explicitamente permitida', async (t) => {
  const root = await makeFixture(t);
  await writeFile(path.join(root, 'docs', 'other.md'), '![Negada](../media/nope.png)\n');
  await writeFile(path.join(root, 'media', 'nope.png'), encoder.encode('nope'));

  await assert.rejects(
    syncContent({root}),
    /Imagem local não permitida.*media\/nope\.png/i,
  );
});

test('rejeita MDX executável em páginas públicas', async (t) => {
  const cases = [
    ['import', 'import secret from "../private/customer-data.json"\n\n# Título\n'],
    ['export', 'export const leaked = true\n\n# Título\n'],
    ['expressão', '# Título\n\n{process.env.SECRET}\n'],
    ['componente JSX', '# Título\n\n<PrivateData source="../private/data.json" />\n'],
    ['atributo executável', '# Título\n\n<figure onClick={() => alert("x")} />\n'],
  ];

  for (const [name, markdown] of cases) {
    await t.test(name, async (t) => {
      const root = await makeFixture(t);
      await writeFile(path.join(root, 'docs', 'other.md'), markdown);
      await assert.rejects(syncContent({root}), /MDX executável.*docs\/other\.md/i);
    });
  }
});

test('preserva matemática e HTML semântico estático permitido', async (t) => {
  const root = await makeFixture(t);
  await writeFile(
    path.join(root, 'docs', 'other.md'),
    [
      '# Documento seguro',
      '',
      'A métrica é $x_{i} = \\{1, 2\\}$.',
      '',
      '$$',
      '\\sum_{i=1}^{n} x_i',
      '$$',
      '',
      '<figure class="resultado">',
      '  <img src="../media/chart.png" alt="Resultado do experimento">',
      '  <figcaption>Resultado agregado.</figcaption>',
      '</figure>',
      '',
    ].join('\n'),
  );

  await syncContent({root});
  const page = await readFile(
    path.join(root, 'website', '.generated', 'docs', 'outra.md'),
    'utf8',
  );
  assert.match(page, /\$x_\{i\} = \\\{1, 2\\\}\$/);
  assert.match(page, /\\sum_\{i=1\}\^\{n\} x_i/);
  assert.match(page, /<figure class="resultado">/);
  assert.match(page, /src="\/assets\/demo\/chart\.png"/);
  assert.match(page, /<figcaption>Resultado agregado\.<\/figcaption>/);
});

test('deny tem precedência sobre allowlist', async (t) => {
  const manifest = baseManifest();
  manifest.allowed_evidence[0] = {
    source: 'private/result.json',
    destination: 'evidence/demo/result.json',
    kind: 'result',
  };
  const root = await makeFixture(t, {manifest});
  await writeFile(path.join(root, 'private', 'result.json'), '{}');

  await assert.rejects(syncContent({root}), /negado.*private\/result\.json/i);
});

test('deny ignora caixa e também valida o caminho real de symlink interno', async (t) => {
  await t.test('caminho com caixa diferente', async (t) => {
    const manifest = baseManifest();
    manifest.allowed_evidence[0] = {
      source: 'PRIVATE/result.json',
      destination: 'evidence/demo/result.json',
      kind: 'result',
    };
    const root = await makeFixture(t, {manifest});

    await assert.rejects(syncContent({root}), /negado.*PRIVATE\/result\.json/i);
  });

  await t.test('alias interno para diretório negado', async (t) => {
    const manifest = baseManifest();
    manifest.allowed_evidence[0] = {
      source: 'public-alias/result.json',
      destination: 'evidence/demo/result.json',
      kind: 'result',
    };
    const root = await makeFixture(t, {manifest});
    await writeFile(path.join(root, 'private', 'result.json'), '{}');
    await symlink(
      path.join(root, 'private'),
      path.join(root, 'public-alias'),
      process.platform === 'win32' ? 'junction' : 'dir',
    );

    await assert.rejects(
      syncContent({root}),
      /negado.*private\/result\.json|caminho real.*negado/i,
    );
  });
});

test('rejeita travessia, caminhos absolutos e duplicatas', async (t) => {
  const cases = [
    ['travessia', (m) => (m.public_pages[0].source = '../README.md'), /caminho.*\.\./i],
    ['absoluto', (m) => (m.public_pages[0].source = path.resolve('README.md')), /relativo/i],
    ['id duplicado', (m) => (m.public_pages[1].id = m.public_pages[0].id), /id.*duplicado/i],
    ['slug duplicado', (m) => (m.public_pages[1].slug = m.public_pages[0].slug), /slug.*duplicado/i],
    ['ordem duplicada', (m) => (m.public_pages[1].order = m.public_pages[0].order), /order.*duplicado/i],
    ['origem duplicada', (m) => (m.public_pages[1].source = m.public_pages[0].source), /source.*duplicado/i],
    [
      'destino duplicado',
      (m) => {
        m.allowed_evidence.push({
          source: 'results/other.json',
          destination: m.allowed_evidence[0].destination,
          kind: 'result',
        });
      },
      /destination.*duplicado/i,
    ],
  ];

  for (const [name, mutate, expected] of cases) {
    await t.test(name, async (t) => {
      const manifest = baseManifest();
      mutate(manifest);
      const root = await makeFixture(t, {manifest});
      await writeFile(path.join(root, 'results', 'other.json'), '{}');
      await assert.rejects(syncContent({root}), expected);
    });
  }
});

test('rejeita colisões de source, destination, id e slug que diferem só por caixa', async (t) => {
  const cases = [
    ['source', (m) => (m.public_pages[1].source = 'DOCS/readme.MD')],
    ['id', (m) => (m.public_pages[1].id = 'INICIO')],
    ['slug', (m) => (m.public_pages[1].slug = '/INICIO')],
    [
      'destination',
      (m) => {
        m.allowed_evidence.push({
          source: 'results/other.json',
          destination: 'evidence/demo/RESULT.JSON',
          kind: 'result',
        });
      },
    ],
  ];

  for (const [field, mutate] of cases) {
    await t.test(field, async (t) => {
      const manifest = baseManifest();
      mutate(manifest);
      const root = await makeFixture(t, {manifest});
      await writeFile(path.join(root, 'results', 'other.json'), '{}');
      await assert.rejects(syncContent({root}), new RegExp(`${field}.*duplicado`, 'i'));
    });
  }
});

test('reescreve Markdown e HTML robustamente sem alterar código ou externos', async (t) => {
  const manifest = baseManifest();
  manifest.public_pages.push({
    source: 'docs/paren(test).md',
    id: 'parenteses',
    slug: '/parenteses',
    title: 'Parênteses',
    sidebar: 'principal',
    order: 30,
  });
  const root = await makeFixture(t, {manifest});
  await writeFile(path.join(root, 'docs', 'paren(test).md'), '# Parênteses\n');
  await writeFile(
    path.join(root, 'docs', 'README.md'),
    [
      '# Documento canônico',
      '',
      '[Com título](other.md "Título do link")',
      '[Parênteses](paren\\(test\\).md \'Título\')',
      '[Rótulo \\[escapado\\]](other.md)',
      '[Root](/docs/other.md?x=1#secao)',
      '![Figura inline](../media/chart.png "Gráfico")',
      '[Completa][outra-pagina] e [Colapsada][].',
      '[Rótulo com colon][x:y].',
      '![Figura referência][figura].',
      '[Privada referência][privada].',
      '<a class="local" href="../results/result.json">Evidência HTML</a>',
      '<a href="/docs/other.md">Página HTML root</a>',
      '<a href="../private/secret.md">Privada HTML</a>',
      '<img alt="Figura HTML" src="../media/chart.png">',
      '<img src="/media/chart.png" alt="Figura root">',
      '<img alt="Responsiva" srcset="../media/chart.png, ../media/chart.png 2x, https://example.com/chart.png 3x, data:image/png;base64,AAAA 4x">',
      '<a href="https://example.com/x">Externo HTML</a>',
      '<a data-href="../private/secret.md">Data href não é link</a>',
      '<!-- <a href="../private/secret.md">Comentário</a> -->',
      '<pre>',
      '<img src="../media/nope.png"><a href="other.md">Pré</a>',
      '</pre>',
      '<code>',
      '<img src="../media/nope.png"><a href="other.md">Código</a>',
      '</code>',
      '',
      '[outra-pagina]: other.md "Outra página"',
      '[Colapsada]: other.md',
      '[x:y]: other.md "Com colon"',
      '[figura]: ../media/chart.png \'Figura\'',
      '[privada]: ../private/secret.md',
      '',
      '    [indentado](other.md)',
      '',
      '`[inline](other.md)`',
      '',
      '```md',
      '[fenced](other.md)',
      '```',
      '',
    ].join('\n'),
  );

  await syncContent({root});
  const page = await readFile(
    path.join(root, 'website', '.generated', 'docs', 'inicio.md'),
    'utf8',
  );

  assert.match(page, /\[Com título\]\(\/outra "Título do link"\)/);
  assert.match(page, /\[Parênteses\]\(\/parenteses 'Título'\)/);
  assert.match(page, /\[Rótulo \\\[escapado\\\]\]\(\/outra\)/);
  assert.match(page, /\[Root\]\(\/outra\?x=1#secao\)/);
  assert.match(page, /!\[Figura inline\]\(\/assets\/demo\/chart\.png "Gráfico"\)/);
  assert.match(page, /\[outra-pagina\]: \/outra "Outra página"/i);
  assert.match(page, /\[Colapsada\]: \/outra/);
  assert.match(page, /\[x:y\]: \/outra "Com colon"/);
  assert.match(page, /\[figura\]: \/assets\/demo\/chart\.png 'Figura'/);
  assert.match(
    page,
    /Privada referência \(disponível no repositório-fonte\)\./,
  );
  assert.doesNotMatch(page, /\[Privada referência\]\[privada\]/);
  assert.doesNotMatch(page, /\[privada\]:\s*\.\.\/private\/secret\.md/);
  assert.match(page, /<a class="local" href="\/evidence\/demo\/result\.json">/);
  assert.match(page, /<a href="\/outra">Página HTML root<\/a>/);
  assert.match(page, /<a>Privada HTML<\/a>/);
  assert.match(page, /<img alt="Figura HTML" src="\/assets\/demo\/chart\.png">/);
  assert.match(page, /<img src="\/assets\/demo\/chart\.png" alt="Figura root">/);
  assert.match(
    page,
    /srcset="\/assets\/demo\/chart\.png, \/assets\/demo\/chart\.png 2x, https:\/\/example\.com\/chart\.png 3x, data:image\/png;base64,AAAA 4x"/,
  );
  assert.match(page, /<a href="https:\/\/example\.com\/x">Externo HTML<\/a>/);
  assert.match(page, /<a data-href="\.\.\/private\/secret\.md">Data href não é link<\/a>/);
  assert.match(page, /<!-- <a href="\.\.\/private\/secret\.md">Comentário<\/a> -->/);
  assert.match(
    page,
    /<pre>\n<img src="\.\.\/media\/nope\.png"><a href="other\.md">Pré<\/a>\n<\/pre>/,
  );
  assert.match(
    page,
    /<code>\n<img src="\.\.\/media\/nope\.png"><a href="other\.md">Código<\/a>\n<\/code>/,
  );
  assert.match(page, /^    \[indentado\]\(other\.md\)$/m);
  assert.match(page, /`\[inline\]\(other\.md\)`/);
  assert.match(page, /```md\n\[fenced\]\(other\.md\)\n```/);
});

test('reescreve somente a URL quando o título inline contém ](', async (t) => {
  const root = await makeFixture(t);
  await writeFile(
    path.join(root, 'docs', 'README.md'),
    '[Página](other.md "Título ]( preservado")\n',
  );

  await syncContent({root});
  const page = await readFile(
    path.join(root, 'website', '.generated', 'docs', 'inicio.md'),
    'utf8',
  );

  assert.match(page, /\[Página\]\(\/outra "Título \]\( preservado"\)/);
});

test('usa a primeira definição duplicada e sanitiza destinos locais posteriores', async (t) => {
  const root = await makeFixture(t);
  await writeFile(
    path.join(root, 'docs', 'README.md'),
    [
      '[Página][duplicada]',
      '',
      '[duplicada]: other.md "Pública"',
      '[duplicada]: ../private/secret.md "Privada"',
      '',
    ].join('\n'),
  );

  await syncContent({root});
  const page = await readFile(
    path.join(root, 'website', '.generated', 'docs', 'inicio.md'),
    'utf8',
  );

  assert.match(page, /^\[Página\]\[duplicada\]$/m);
  assert.match(page, /^\[duplicada\]: \/outra "Pública"$/m);
  assert.doesNotMatch(page, /\.\.\/private\/secret\.md/);
  assert.match(page, /<!-- referência local não publicada: duplicada -->/);
});

test('falha para imagens locais não permitidas em referência e HTML', async (t) => {
  for (const [kind, markdown] of [
    ['referência', '![Negada][img]\n\n[img]: ../media/nope.png\n'],
    [
      'HTML',
      '<img src="../media/chart.png" srcset="../media/chart.png 1x, ../media/nope.png 2x" alt="Negada">\n',
    ],
  ]) {
    await t.test(kind, async (t) => {
      const root = await makeFixture(t);
      await writeFile(path.join(root, 'docs', 'other.md'), markdown);
      await writeFile(path.join(root, 'media', 'nope.png'), encoder.encode('nope'));

      await assert.rejects(
        syncContent({root}),
        /Imagem local não permitida.*media\/nope\.png/i,
      );
    });
  }

  const root = await makeFixture(t);
  await writeFile(
    path.join(root, 'docs', 'other.md'),
    '<img srcset="../media/chart.png 1x ../media/nope.png 2x" alt="Ambígua">\n',
  );
  await assert.rejects(syncContent({root}), /srcset.*ambígu/i);
});

test('valida schema, arrays, campos, tipos, extensões e kinds', async (t) => {
  const cases = [
    ['schema', (m) => (m.schema_version = 2), /schema_version/i],
    ['generated root', (m) => (m.generated_root = '.generated'), /generated_root/i],
    ['array', (m) => (m.public_pages = {}), /public_pages.*array/i],
    ['campo', (m) => delete m.public_pages[0].title, /title/i],
    ['tipo', (m) => (m.public_pages[0].order = '10'), /order/i],
    [
      'sidebar_label opcional',
      (m) => (m.public_pages[0].sidebar_label = ''),
      /sidebar_label.*vazi/i,
    ],
    ['página', (m) => (m.public_pages[0].source = 'docs/page.txt'), /\.md/i],
    ['figura', (m) => (m.allowed_assets[0].source = 'media/chart.svg'), /png/i],
    ['kind figura', (m) => (m.allowed_assets[0].kind = 'image'), /kind/i],
    ['evidência ativa', (m) => (m.allowed_evidence[0].source = 'results/x.html'), /extensão/i],
    ['kind evidência', (m) => (m.allowed_evidence[0].kind = 'other'), /kind/i],
  ];

  for (const [name, mutate, expected] of cases) {
    await t.test(name, async (t) => {
      const manifest = baseManifest();
      mutate(manifest);
      const root = await makeFixture(t, {manifest});
      await assert.rejects(syncContent({root}), expected);
    });
  }
});

test('rejeita origem ausente, JSON inválido e symlink que escapa da raiz', async (t) => {
  await t.test('origem ausente', async (t) => {
    const root = await makeFixture(t);
    await rm(path.join(root, 'docs', 'other.md'));
    await assert.rejects(syncContent({root}), /Origem inexistente.*docs\/other\.md/i);
  });

  await t.test('JSON inválido', async (t) => {
    const root = await makeFixture(t);
    await writeFile(path.join(root, 'results', 'result.json'), '{invalid');
    await assert.rejects(syncContent({root}), /JSON inválido.*result\.json/i);
  });

  await t.test('symlink externo', async (t) => {
    const manifest = baseManifest();
    manifest.public_pages[1].source = 'docs/external/outside.md';
    const root = await makeFixture(t, {manifest});
    const outside = await mkdtemp(path.join(os.tmpdir(), 'sync-outside-'));
    t.after(() => rm(outside, {recursive: true, force: true}));
    await writeFile(path.join(outside, 'outside.md'), '# Fora\n');
    await symlink(
      outside,
      path.join(root, 'docs', 'external'),
      process.platform === 'win32' ? 'junction' : 'dir',
    );
    await assert.rejects(syncContent({root}), /symlink|fora da raiz/i);
  });
});

test('é determinístico, idempotente e --check detecta stale sem escrever', async (t) => {
  const root = await makeFixture(t);
  const generated = path.join(root, 'website', '.generated');

  await syncContent({root});
  const first = await snapshotTree(generated);
  await syncContent({root});
  const second = await snapshotTree(generated);
  assert.deepEqual(second, first);

  const pagePath = path.join(generated, 'docs', 'inicio.md');
  await writeFile(pagePath, 'stale\n');
  const before = await readFile(pagePath);
  const beforeMtime = (await stat(pagePath)).mtimeMs;
  await assert.rejects(syncContent({root, check: true}), /desatualizado/i);
  assert.deepEqual(await readFile(pagePath), before);
  assert.equal((await stat(pagePath)).mtimeMs, beforeMtime);
  assert.deepEqual(
    (await readdir(path.join(root, 'website'))).filter((name) => name.includes('.tmp-')),
    [],
  );

  await syncContent({root});
  const checked = await syncContent({root, check: true});
  assert.equal(checked.upToDate, true);
});

test('limpa staging quando a construção falha', async (t) => {
  const root = await makeFixture(t);
  await writeFile(path.join(root, 'results', 'result.json'), '{invalid');

  await assert.rejects(syncContent({root}), ManifestError);
  assert.deepEqual(
    (await readdir(path.join(root, 'website'))).filter(
      (name) => name.startsWith('.generated.tmp-') || name.startsWith('.generated.bak-'),
    ),
    [],
  );
});

test('restaura publicação anterior e limpa staging se a troca falhar', async (t) => {
  const root = await makeFixture(t);
  const website = path.join(root, 'website');
  const generatedPath = path.join(website, '.generated');
  const stagingPath = path.join(website, '.generated.tmp-test');
  await mkdir(generatedPath, {recursive: true});
  await mkdir(stagingPath, {recursive: true});
  await writeFile(path.join(generatedPath, 'old.txt'), 'antigo');
  await writeFile(path.join(stagingPath, 'new.txt'), 'novo');

  let calls = 0;
  const {rename} = await import('node:fs/promises');
  const failingRename = async (source, destination) => {
    calls += 1;
    if (calls === 2) {
      throw new Error('falha injetada');
    }
    return rename(source, destination);
  };

  await assert.rejects(
    publishStaging({generatedPath, stagingPath, rename: failingRename}),
    /falha injetada/,
  );
  assert.equal(await readFile(path.join(generatedPath, 'old.txt'), 'utf8'), 'antigo');
  await assert.rejects(stat(stagingPath), {code: 'ENOENT'});
  assert.deepEqual(
    (await readdir(website)).filter((name) => name.startsWith('.generated.bak-')),
    [],
  );
});

test('preserva a nova publicação quando a limpeza do backup falha após o commit', async (t) => {
  const root = await makeFixture(t);
  const website = path.join(root, 'website');
  const generatedPath = path.join(website, '.generated');
  const stagingPath = path.join(website, '.generated.tmp-test');
  await mkdir(generatedPath, {recursive: true});
  await mkdir(stagingPath, {recursive: true});
  await writeFile(path.join(generatedPath, 'old.txt'), 'antigo');
  await writeFile(path.join(stagingPath, 'new.txt'), 'novo');

  const removeCalls = [];
  const failingRemove = async (target, options) => {
    removeCalls.push(target);
    if (path.basename(target).startsWith('.generated.bak-')) {
      throw new Error('falha na limpeza pós-commit');
    }
    return rm(target, options);
  };

  await assert.rejects(
    publishStaging({
      generatedPath,
      stagingPath,
      remove: failingRemove,
    }),
    /falha na limpeza pós-commit/,
  );

  assert.equal(await readFile(path.join(generatedPath, 'new.txt'), 'utf8'), 'novo');
  await assert.rejects(stat(path.join(generatedPath, 'old.txt')), {code: 'ENOENT'});
  const backups = (await readdir(website)).filter((name) =>
    name.startsWith('.generated.bak-'),
  );
  assert.equal(backups.length, 1);
  assert.equal(
    await readFile(path.join(website, backups[0], 'old.txt'), 'utf8'),
    'antigo',
  );
  assert.equal(
    removeCalls.filter((target) => path.basename(target).startsWith('.generated.bak-'))
      .length,
    1,
  );
});
