import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {mkdtemp, mkdir, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {promisify} from 'node:util';
import test from 'node:test';

const execFileAsync = promisify(execFile);
const auditScript = path.resolve('scripts/audit-build.mjs');

async function write(root, relative, contents = '') {
  const target = path.join(root, ...relative.split('/'));
  await mkdir(path.dirname(target), {recursive: true});
  await writeFile(target, contents);
}

async function makeFixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'portal-build-audit-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const build = path.join(root, 'build');
  const manifest = path.join(root, 'content-manifest.json');
  const data = {
    public_pages: [{slug: '/sobre-o-projeto'}],
    allowed_assets: [
      {source: 'media/figure.png', destination: 'assets/ioi/figure.png'},
    ],
    allowed_evidence: [
      {source: 'results/result.json', destination: 'evidence/ioi/result.json'},
    ],
  };
  await writeFile(manifest, `${JSON.stringify(data)}\n`);
  for (const html of [
    'index.html',
    '404.html',
    'search.html',
    'sobre-o-projeto.html',
  ]) {
    await write(build, html, '<!doctype html><html lang="pt-BR"></html>');
  }
  await write(root, 'media/figure.png', 'canonical-png');
  await write(root, 'results/result.json', '{"ok":true}');
  await write(build, 'assets/ioi/figure.png', 'canonical-png');
  await write(build, 'evidence/ioi/result.json', '{"ok":true}');
  await write(
    build,
    'assets/images/figure-0123456789abcdef0123456789abcdef.png',
    'canonical-png',
  );
  await write(
    build,
    'assets/files/result-0123456789abcdef0123456789abcdef.json',
    '{"ok":true}',
  );
  await write(build, 'assets/js/runtime.0123abcd.js', 'console.log("chunk")');
  await write(
    build,
    'assets/js/runtime.0123abcd.js.LICENSE.txt',
    '/*! legitimate dependency license */',
  );
  await write(build, 'assets/css/styles.0123abcd.css', 'body{}');
  await write(
    build,
    'assets/fonts/KaTeX_Main-Regular-0123456789abcdef0123456789abcdef.woff2',
    'font',
  );
  await write(build, 'search-index.json', '[]');
  await write(build, 'sitemap.xml', '<urlset/>');
  await write(build, 'img/favicon.svg', '<svg/>');
  return {build, manifest, root};
}

async function audit({build, manifest, root, baseUrl = '/'}) {
  try {
    const result = await execFileAsync(
      process.execPath,
      [
        auditScript,
        '--build-dir',
        build,
        '--manifest',
        manifest,
        '--root',
        root,
        '--base-url',
        baseUrl,
      ],
      {cwd: path.resolve('.')},
    );
    return {code: 0, stdout: result.stdout, stderr: result.stderr};
  } catch (error) {
    return {
      code: error.code,
      stdout: error.stdout ?? '',
      stderr: error.stderr ?? '',
    };
  }
}

test('aceita somente rotas e conteúdo sincronizado declarados no manifesto', async (t) => {
  const fixture = await makeFixture(t);
  const result = await audit(fixture);
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stdout, /auditoria do build aprovada/i);
});

test('rejeita conteúdo sincronizado não declarado', async (t) => {
  const fixture = await makeFixture(t);
  await write(fixture.build, 'assets/ioi/not-in-manifest.png', 'png');
  const result = await audit(fixture);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /não declarado.*assets\/ioi\/not-in-manifest\.png/i);
});

test('rejeita qualquer arquivo fora das classes legítimas do build', async (t) => {
  const fixture = await makeFixture(t);
  await write(fixture.build, 'private/customer-data.csv', 'customer,secret');
  const result = await audit(fixture);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /arquivo não permitido.*private\/customer-data\.csv/i);
});

test('rejeita cópia hash não declarada gerada pelo pipeline MDX', async (t) => {
  const fixture = await makeFixture(t);
  await write(
    fixture.build,
    'assets/files/private-result-0123456789abcdef0123456789abcdef.json',
    '{}',
  );
  const result = await audit(fixture);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /não declarado.*assets\/files\/private-result/i);
});

test('rejeita cópias hash adulteradas de assets e evidências permitidos', async (t) => {
  for (const relative of [
    'assets/images/figure-0123456789abcdef0123456789abcdef.png',
    'assets/files/result-0123456789abcdef0123456789abcdef.json',
  ]) {
    await t.test(relative, async (t) => {
      const fixture = await makeFixture(t);
      await write(fixture.build, relative, 'tampered');
      const result = await audit(fixture);
      assert.notEqual(result.code, 0);
      assert.match(result.stderr, /conteúdo divergente.*assets\/(?:files|images)\//i);
    });
  }
});

test('rejeita rotas de conteúdo não declaradas', async (t) => {
  const fixture = await makeFixture(t);
  await write(fixture.build, 'plano-secreto.html', '<!doctype html>');
  const result = await audit(fixture);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /rota html não esperada.*plano-secreto/i);
});

test('aceita assets CSS absolutos no baseUrl raiz', async (t) => {
  const fixture = await makeFixture(t);
  await write(
    fixture.build,
    'assets/css/styles.0123abcd.css',
    '@font-face{src:url(/assets/fonts/font.woff2)}',
  );
  const result = await audit({...fixture, baseUrl: '/'});
  assert.equal(result.code, 0, result.stderr);
});

test('aceita assets CSS prefixados no baseUrl do projeto', async (t) => {
  const fixture = await makeFixture(t);
  await write(
    fixture.build,
    'assets/css/styles.0123abcd.css',
    '@font-face{src:url(/ioi-replication/assets/fonts/font.woff2)}',
  );
  const result = await audit({...fixture, baseUrl: '/ioi-replication/'});
  assert.equal(result.code, 0, result.stderr);
});

test('rejeita assets CSS que escapam do baseUrl do projeto', async (t) => {
  const fixture = await makeFixture(t);
  await write(
    fixture.build,
    'assets/css/styles.0123abcd.css',
    '@font-face{src:url(/assets/fonts/font.woff2)}',
  );
  const result = await audit({...fixture, baseUrl: '/ioi-replication/'});
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /baseurl.*assets\/css\/styles\.0123abcd\.css/i);
});

test('rejeita assets CSS com prefixo de outro projeto', async (t) => {
  const fixture = await makeFixture(t);
  await write(
    fixture.build,
    'assets/css/styles.0123abcd.css',
    '@font-face{src:url(/outro-projeto/assets/fonts/font.woff2)}',
  );
  const result = await audit({...fixture, baseUrl: '/ioi-replication/'});
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /baseurl.*assets\/css\/styles\.0123abcd\.css/i);
});

test('rejeita classes proibidas sem bloquear chunks legítimos', async (t) => {
  const forbidden = [
    'notebooks/demo.ipynb',
    'diagramas/fluxo.drawio',
    '.env.production',
    'models/model.safetensors',
    'docs/superpowers/plans/internal.html',
    'docs/entregas/report.pdf',
    'tools/colab/run.py',
    'projects/ioi/src/model.py',
    'credentials.json',
  ];

  for (const relative of forbidden) {
    await t.test(relative, async (t) => {
      const fixture = await makeFixture(t);
      await write(fixture.build, relative, 'forbidden');
      const result = await audit(fixture);
      assert.notEqual(result.code, 0);
      assert.match(result.stderr, /conteúdo proibido/i);
    });
  }
});
