import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const searchTranslationIds = [
  'theme.SearchBar.label',
  'theme.SearchBar.seeAll',
  'theme.SearchBar.noResultsText',
  'theme.SearchBar.searchInContext',
  'theme.SearchBar.seeAllOutsideContext',
  'theme.SearchPage.existingResultsTitle',
  'theme.SearchPage.emptyResultsTitle',
  'theme.SearchPage.documentsFound.plurals',
  'theme.SearchPage.noResultsText',
  'theme.SearchPage.searchContext.everywhere',
];

test('expõe scripts explícitos para validar GitHub Project Pages', async () => {
  const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
  for (const script of [
    'build:project-page',
    'serve:project-page',
    'audit:build',
    'audit:project-page',
    'test:e2e:project-page',
    'validate:project-page',
  ]) {
    assert.equal(typeof packageJson.scripts[script], 'string', `script ausente: ${script}`);
  }
  assert.match(
    packageJson.scripts['build:project-page'],
    /DOCUSAURUS_BASE_URL=\/ioi-replication\//,
  );
  assert.match(
    packageJson.scripts['build:project-page'],
    /DOCUSAURUS_URL=http:\/\/127\.0\.0\.1:4173/,
  );
  assert.match(
    packageJson.scripts['audit:project-page'],
    /--base-url\s+\/ioi-replication\//,
  );
  assert.match(
    packageJson.scripts['test:e2e:project-page'],
    /audit:project-page/,
  );
  assert.match(
    packageJson.scripts['validate:project-page'],
    /npm run build && npm run audit:build && npm run test:e2e:project-page/,
  );
});

test('fornece todas as traduções reais usadas pelo plugin de busca', async () => {
  const translations = JSON.parse(
    await readFile('i18n/pt-BR/code.json', 'utf8'),
  );
  assert.deepEqual(Object.keys(translations).sort(), searchTranslationIds.sort());
  assert.equal(translations['theme.SearchBar.label'].message, 'Buscar');
  assert.equal(
    translations['theme.SearchPage.emptyResultsTitle'].message,
    'Buscar na documentação',
  );
  assert.match(
    translations['theme.SearchPage.documentsFound.plurals'].message,
    /\|/,
  );
});

test('usa axe compatível com Node 20 e cobre WCAG 2.2', async () => {
  const packageJson = JSON.parse(await readFile('package.json', 'utf8'));
  const packageLock = JSON.parse(await readFile('package-lock.json', 'utf8'));
  assert.equal(packageJson.devDependencies['@axe-core/playwright'], '4.13.0');
  assert.equal(
    packageLock.packages['node_modules/@axe-core/playwright'].version,
    '4.13.0',
  );
  assert.doesNotMatch(
    packageLock.packages['node_modules/@axe-core/playwright'].engines?.node ?? '',
    />=22/,
  );

  const e2e = await readFile('tests/e2e/portal.spec.ts', 'utf8');
  assert.match(e2e, /wcag22aa/);
});

test('usa título completo, marca curta na navbar e rodapé MIT', async () => {
  const config = await readFile('docusaurus.config.ts', 'utf8');
  const readme = await readFile('../README.md', 'utf8');
  const manifest = JSON.parse(await readFile('content-manifest.json', 'utf8'));
  const sobre = manifest.public_pages.find((page) => page.id === 'inicio');

  assert.match(
    config,
    /title: 'Interpretabilidade Aplicada ao Alinhamento de LLMs'/,
  );
  assert.match(config, /navbar:\s*\{[\s\S]*?title: 'Interpretabilidade'/);
  assert.doesNotMatch(
    config,
    /navbar:\s*\{[\s\S]*?title: 'Interpretabilidade Aplicada/,
  );
  assert.match(config, /copyright: 'Licença MIT\.'/);
  assert.match(
    readme,
    /^# Interpretabilidade Aplicada ao Alinhamento de LLMs$/m,
  );
  assert.equal(
    sobre?.title,
    'Interpretabilidade Aplicada ao Alinhamento de LLMs',
  );
});

test('não duplica entradas no .gitignore do portal', async () => {
  const entries = (await readFile('.gitignore', 'utf8'))
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  assert.equal(new Set(entries).size, entries.length);
});
