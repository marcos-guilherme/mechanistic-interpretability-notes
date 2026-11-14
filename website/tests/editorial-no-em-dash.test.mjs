import assert from 'node:assert/strict';
import {readdir, readFile, stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const EM_DASH = '\u2014';
const websiteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const repoRoot = path.dirname(websiteRoot);
const manifestPath = path.join(websiteRoot, 'content-manifest.json');

async function collectUiCopyPaths() {
  const paths = [
    path.join(websiteRoot, 'docusaurus.config.ts'),
    path.join(websiteRoot, 'src', 'pages'),
    path.join(websiteRoot, 'src', 'components'),
  ];

  const files = [];

  for (const entry of paths) {
    const info = await stat(entry);
    if (info.isFile()) {
      files.push(entry);
      continue;
    }
      if (info.isDirectory()) {
      await walkDirectory(entry, files);
    }
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  files.push(manifestPath);
  for (const page of manifest.public_pages) {
    if (typeof page.sidebar_label === 'string') {
      files.push({
        virtual: `content-manifest.json#${page.id}.sidebar_label`,
        content: page.sidebar_label,
      });
    }
    if (typeof page.title === 'string') {
      files.push({
        virtual: `content-manifest.json#${page.id}.title`,
        content: page.title,
      });
    }
  }

  for (const page of manifest.public_pages) {
    files.push(path.join(repoRoot, page.source));
  }

  return files;
}

async function walkDirectory(directory, files) {
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await walkDirectory(fullPath, files);
      continue;
    }
    if (/\.(tsx|ts|jsx|js)$/.test(entry.name)) {
      files.push(fullPath);
    }
  }
}

function findEmDashOccurrences(label, content) {
  const hits = [];
  const lines = content.split(/\r?\n/u);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
      if (line.includes(EM_DASH)) {
      hits.push({label, line: index + 1, text: line.trim()});
    }
  }
  return hits;
}

test('conteúdo público e copy do portal não usam travessão (U+2014)', async () => {
  const files = await collectUiCopyPaths();
  const violations = [];

  for (const file of files) {
    if (typeof file === 'string') {
      const content = await readFile(file, 'utf8');
      violations.push(
        ...findEmDashOccurrences(path.relative(repoRoot, file), content),
      );
      continue;
    }

    violations.push(...findEmDashOccurrences(file.virtual, file.content));
  }

  assert.deepEqual(
    violations,
    [],
    violations.length
      ? `Encontrado travessão em:\n${violations
          .map(
            (hit) =>
              `- ${hit.label}:${hit.line} ${hit.text}`,
          )
          .join('\n')}`
      : undefined,
  );
});