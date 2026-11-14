import {readFile, readdir, stat} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {fileURLToPath} from 'node:url';

function toPosix(value) {
  return value.replaceAll(path.sep, '/');
}

function normalizeBaseUrl(value) {
  const normalized = value.trim().replace(/^\/+|\/+$/g, '');
  return normalized ? `/${normalized}/` : '/';
}

function absoluteAssetUrls(css) {
  return [...css.matchAll(/url\(\s*(['"]?)(\/[^'")\s]+)\1\s*\)/gi)]
    .map((match) => match[2])
    .filter((url) => /(?:^|\/)assets\//i.test(url));
}

async function listFiles(root) {
  const files = [];

  async function visit(directory) {
    for (const entry of await readdir(directory, {withFileTypes: true})) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else {
        files.push(toPosix(path.relative(root, absolute)));
      }
    }
  }

  await visit(root);
  return files.sort();
}

function routeFromHtml(relative) {
  if (!relative.toLowerCase().endsWith('.html')) {
    return null;
  }
  const withoutExtension = relative.slice(0, -'.html'.length);
  if (withoutExtension === 'index') {
    return '/';
  }
  if (withoutExtension.endsWith('/index')) {
    return `/${withoutExtension.slice(0, -'/index'.length)}`;
  }
  return `/${withoutExtension}`;
}

function isForbidden(relative) {
  const normalized = relative.normalize('NFC').toLowerCase();
  const segments = normalized.split('/');
  const basename = segments.at(-1) ?? '';
  const extension = path.posix.extname(basename);
  const forbiddenDirectorySequences = [
    'docs/superpowers',
    'docs/entregas',
    'tools/colab',
  ];
  const forbiddenDirectories = new Set([
    'notebooks',
    'plans',
    'src',
    'tests',
    'test',
    'scripts',
    'models',
    'weights',
  ]);
  const forbiddenExtensions = new Set([
    '.ipynb',
    '.drawio',
    '.py',
    '.pyi',
    '.ts',
    '.tsx',
    '.jsx',
    '.mjs',
    '.cjs',
    '.sh',
    '.ps1',
    '.bat',
    '.cmd',
    '.toml',
    '.safetensors',
    '.bin',
    '.pt',
    '.pth',
    '.ckpt',
    '.onnx',
    '.pem',
    '.key',
    '.p12',
    '.pfx',
    '.map',
  ]);

  return (
    forbiddenDirectorySequences.some(
      (sequence) =>
        normalized === sequence ||
        normalized.startsWith(`${sequence}/`) ||
        normalized.includes(`/${sequence}/`),
    ) ||
    segments.some((segment) => forbiddenDirectories.has(segment)) ||
    forbiddenExtensions.has(extension) ||
    basename === '.env' ||
    basename.startsWith('.env.') ||
    basename === '.npmrc' ||
    basename === '.pypirc' ||
    basename === '.netrc' ||
    basename === 'package.json' ||
    basename.endsWith('-lock.json') ||
    /(?:credential|secret|token)/i.test(basename) ||
    basename === 'id_rsa' ||
    basename === 'id_ed25519'
  );
}

function generatedAssetPattern(destination, directory) {
  const basename = path.posix.basename(destination);
  const extension = path.posix.extname(basename);
  const stem = basename.slice(0, -extension.length);
  const escape = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(
    `^assets/${directory}/${escape(stem)}-[0-9a-f]{32}${escape(extension)}$`,
    'i',
  );
}

function isDocusaurusAsset(relative, files) {
  if (
    /^assets\/js\/[a-z0-9~_-]+\.[0-9a-f]{8}\.js$/i.test(relative) ||
    /^assets\/css\/[a-z0-9~_-]+\.[0-9a-f]{8}\.css$/i.test(relative) ||
    /^assets\/fonts\/KaTeX_[A-Za-z0-9-]+-[0-9a-f]{32}\.(?:woff2?|ttf)$/i.test(
      relative,
    )
  ) {
    return true;
  }
  if (/^assets\/js\/.+\.js\.LICENSE\.txt$/i.test(relative)) {
    return files.includes(relative.slice(0, -'.LICENSE.txt'.length));
  }
  return ['search-index.json', 'sitemap.xml', 'img/favicon.svg'].includes(relative);
}

export async function auditBuild({
  buildDir,
  manifestPath,
  repoRoot,
  expectedBaseUrl = '/',
}) {
  if (!(await stat(buildDir).catch(() => null))?.isDirectory()) {
    throw new Error(`Diretório de build inexistente: ${buildDir}`);
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const files = await listFiles(buildDir);
  const violations = [];
  const normalizedBaseUrl = normalizeBaseUrl(expectedBaseUrl);
  const expectedAssetPrefix = `${normalizedBaseUrl}assets/`;

  const expectedRoutes = new Set([
    '/',
    '/404',
    '/search',
    ...manifest.public_pages.map(({slug}) => slug.replace(/\/+$/, '') || '/'),
  ]);
  const actualRoutes = new Set(
    files.map(routeFromHtml).filter((route) => route !== null),
  );
  for (const route of actualRoutes) {
    if (!expectedRoutes.has(route)) {
      violations.push(`rota HTML não esperada: ${route}`);
    }
  }
  for (const route of expectedRoutes) {
    if (!actualRoutes.has(route)) {
      violations.push(`rota HTML ausente: ${route}`);
    }
  }

  const contentEntries = [
    ...manifest.allowed_assets,
    ...manifest.allowed_evidence,
  ];
  const contentByDestination = new Map(
    contentEntries.map((entry) => [entry.destination, entry]),
  );
  const sourceContents = new Map();
  for (const entry of contentEntries) {
    const sourcePath = path.resolve(repoRoot, ...entry.source.split('/'));
    const contents = await readFile(sourcePath).catch(() => null);
    if (!contents) {
      violations.push(`fonte canônica ausente: ${entry.source}`);
    } else {
      sourceContents.set(entry.source, contents);
    }
  }
  const generatedCopies = [
    ...contentEntries.map((entry) => ({
      entry,
      pattern: generatedAssetPattern(entry.destination, 'files'),
    })),
    ...manifest.allowed_assets.map((entry) => ({
      entry,
      pattern: generatedAssetPattern(entry.destination, 'images'),
    })),
  ];

  for (const relative of files) {
    if (isForbidden(relative)) {
      violations.push(`conteúdo proibido no build: ${relative}`);
      continue;
    }
    const route = routeFromHtml(relative);
    if (route !== null) {
      if (!expectedRoutes.has(route)) {
        violations.push(`rota HTML não esperada: ${route}`);
      }
      continue;
    }

    const directEntry = contentByDestination.get(relative);
    if (directEntry) {
      const expected = sourceContents.get(directEntry.source);
      const actual = await readFile(path.join(buildDir, relative));
      if (!expected?.equals(actual)) {
        violations.push(`conteúdo divergente da fonte canônica: ${relative}`);
      }
      continue;
    }

    if (relative.startsWith('assets/files/') || relative.startsWith('assets/images/')) {
      const candidates = generatedCopies.filter(({pattern}) =>
        pattern.test(relative),
      );
      if (candidates.length === 0) {
        violations.push(`conteúdo sincronizado não declarado: ${relative}`);
        continue;
      }
      const actual = await readFile(path.join(buildDir, relative));
      if (
        !candidates.some(({entry}) =>
          sourceContents.get(entry.source)?.equals(actual),
        )
      ) {
        violations.push(`conteúdo divergente da fonte canônica: ${relative}`);
      }
      continue;
    }

    if (isDocusaurusAsset(relative, files)) {
      if (relative.endsWith('.css')) {
        const css = await readFile(path.join(buildDir, relative), 'utf8');
        const invalidUrl = absoluteAssetUrls(css).find(
          (url) => !url.startsWith(expectedAssetPrefix),
        );
        if (invalidUrl) {
          violations.push(
            `asset CSS fora do baseUrl ${normalizedBaseUrl} em ${relative}: ${invalidUrl}`,
          );
        }
      }
      continue;
    }

    violations.push(
      relative.startsWith('assets/') || relative.startsWith('evidence/')
        ? `conteúdo sincronizado não declarado: ${relative}`
        : `arquivo não permitido no build: ${relative}`,
    );
  }
  for (const destination of contentByDestination.keys()) {
    if (!files.includes(destination)) {
      violations.push(`conteúdo sincronizado ausente: ${destination}`);
    }
  }

  if (violations.length > 0) {
    throw new Error(violations.join('\n'));
  }

  return {
    files: files.length,
    routes: actualRoutes.size,
    synchronizedFiles: contentByDestination.size,
  };
}

function readArgument(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1] ?? '')) {
  try {
    const buildDir = path.resolve(readArgument('--build-dir') ?? 'build');
    const manifestPath = path.resolve(
      readArgument('--manifest') ?? 'content-manifest.json',
    );
    const repoRoot = path.resolve(readArgument('--root') ?? '..');
    const expectedBaseUrl =
      readArgument('--base-url') ?? process.env.DOCUSAURUS_BASE_URL ?? '/';
    const result = await auditBuild({
      buildDir,
      manifestPath,
      repoRoot,
      expectedBaseUrl,
    });
    console.log(
      `Auditoria do build aprovada: ${result.routes} rotas, ` +
        `${result.synchronizedFiles} arquivos sincronizados, ` +
        `${result.files} arquivos totais.`,
    );
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
