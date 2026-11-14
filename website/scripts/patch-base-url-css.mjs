import {readFile, readdir, writeFile} from 'node:fs/promises';
import path from 'node:path';

function normalizeBaseUrl(value) {
  const normalized = value.replace(/^\/+|\/+$/g, '');
  return normalized ? `/${normalized}/` : '/';
}

async function listCssFiles(root) {
  const files = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, {withFileTypes: true})) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.name.endsWith('.css')) {
        files.push(absolute);
      }
    }
  }
  await visit(root);
  return files;
}

const baseUrl = normalizeBaseUrl(process.env.DOCUSAURUS_BASE_URL ?? '/');
if (baseUrl !== '/') {
  const cssRoot = path.resolve('build', 'assets', 'css');
  let replacements = 0;
  for (const file of await listCssFiles(cssRoot)) {
    const original = await readFile(file, 'utf8');
    const patched = original.replace(
      /url\(\s*(['"]?)\/assets\//gi,
      (_match, quote) => `url(${quote}${baseUrl}assets/`,
    );
    if (patched !== original) {
      await writeFile(file, patched);
      replacements += 1;
    }
  }
  console.log(`Base URL aplicado aos assets CSS em ${replacements} arquivo(s).`);
}
