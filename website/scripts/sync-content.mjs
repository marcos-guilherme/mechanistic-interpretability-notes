import {
  access,
  copyFile,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename as fsRename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {fileURLToPath, pathToFileURL} from 'node:url';
import remarkMdx from 'remark-mdx';
import remarkMath from 'remark-math';
import remarkParse from 'remark-parse';
import {unified} from 'unified';

const EXPECTED_SCHEMA_VERSION = 1;
const EXPECTED_GENERATED_ROOT = 'website/.generated';
const PAGE_FIELDS = {
  source: 'string',
  id: 'string',
  slug: 'string',
  title: 'string',
  sidebar: 'string',
  order: 'number',
};
const FILE_FIELDS = {
  source: 'string',
  destination: 'string',
  kind: 'string',
};

export class ManifestError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ManifestError';
  }
}

function toPosix(value) {
  return value.replaceAll('\\', '/');
}

function isWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

function assertRelativePath(value, field) {
  if (
    typeof value !== 'string' ||
    value.length === 0 ||
    path.isAbsolute(value) ||
    /^[a-zA-Z]:/.test(value)
  ) {
    throw new ManifestError(`${field} deve ser um caminho relativo não vazio.`);
  }

  const normalized = toPosix(value);
  const segments = normalized.split('/');
  if (segments.includes('..') || segments.includes('') || segments.includes('.')) {
    throw new ManifestError(`${field} contém caminho inválido ou '..': ${value}`);
  }
  return normalized;
}

function requireRecord(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ManifestError(`${label} deve ser um objeto.`);
  }
}

function requireFields(record, fields, label) {
  requireRecord(record, label);
  for (const [field, type] of Object.entries(fields)) {
    if (typeof record[field] !== type) {
      throw new ManifestError(`${label}.${field} deve ser ${type}.`);
    }
    if (type === 'string' && record[field].length === 0) {
      throw new ManifestError(`${label}.${field} não pode ser vazio.`);
    }
    if (type === 'number' && !Number.isFinite(record[field])) {
      throw new ManifestError(`${label}.${field} deve ser um número finito.`);
    }
  }
}

function assertUnique(items, field, label) {
  const seen = new Set();
  for (const item of items) {
    const value = item[field];
    const comparisonValue =
      typeof value === 'string' ? value.normalize('NFC').toLowerCase() : value;
    if (seen.has(comparisonValue)) {
      throw new ManifestError(`${label}.${field} duplicado: ${value}`);
    }
    seen.add(comparisonValue);
  }
}

function globToRegExp(glob) {
  let expression = '^';
  for (let index = 0; index < glob.length; index += 1) {
    const character = glob[index];
    if (character === '*') {
      if (glob[index + 1] === '*') {
        index += 1;
        if (glob[index + 1] === '/') {
          index += 1;
          expression += '(?:.*/)?';
        } else {
          expression += '.*';
        }
      } else {
        expression += '[^/]*';
      }
    } else if (character === '?') {
      expression += '[^/]';
    } else {
      expression += character.replace(/[\\^$.[\]{}()+|]/g, '\\$&');
    }
  }
  return new RegExp(`${expression}$`);
}

function createDenyMatcher(manifest) {
  const prefixes = manifest.deny_prefixes.map((prefix) => {
    const normalized = assertRelativePath(prefix.replace(/\/$/, ''), 'deny_prefixes');
    return `${normalized.toLowerCase()}/`;
  });
  const patterns = manifest.deny_globs.map((glob, index) => {
    if (typeof glob !== 'string' || glob.length === 0 || path.isAbsolute(glob)) {
      throw new ManifestError(`deny_globs[${index}] deve ser um glob relativo.`);
    }
    if (toPosix(glob).split('/').includes('..')) {
      throw new ManifestError(`deny_globs[${index}] contém '..'.`);
    }
    return globToRegExp(toPosix(glob).toLowerCase());
  });

  return (relativePath) => {
    const normalized = toPosix(relativePath).replace(/^\/+/, '').toLowerCase();
    return (
      prefixes.some(
        (prefix) => normalized === prefix.slice(0, -1) || normalized.startsWith(prefix),
      ) || patterns.some((pattern) => pattern.test(normalized))
    );
  };
}

function validatePage(page, index) {
  const label = `public_pages[${index}]`;
  requireFields(page, PAGE_FIELDS, label);
  if (page.sidebar_label !== undefined) {
    if (typeof page.sidebar_label !== 'string') {
      throw new ManifestError(`${label}.sidebar_label deve ser string.`);
    }
    if (page.sidebar_label.length === 0) {
      throw new ManifestError(`${label}.sidebar_label não pode ser vazio.`);
    }
  }
  page.source = assertRelativePath(page.source, `${label}.source`);
  if (path.posix.extname(page.source).toLowerCase() !== '.md') {
    throw new ManifestError(`${label}.source deve usar extensão .md.`);
  }
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(page.id)) {
    throw new ManifestError(`${label}.id possui formato inválido.`);
  }
  if (!page.slug.startsWith('/') || page.slug.includes('..') || /\s/.test(page.slug)) {
    throw new ManifestError(`${label}.slug deve ser uma rota absoluta válida.`);
  }
}

function validateAsset(asset, index) {
  const label = `allowed_assets[${index}]`;
  requireFields(asset, FILE_FIELDS, label);
  asset.source = assertRelativePath(asset.source, `${label}.source`);
  asset.destination = assertRelativePath(asset.destination, `${label}.destination`);
  if (
    path.posix.extname(asset.source).toLowerCase() !== '.png' ||
    path.posix.extname(asset.destination).toLowerCase() !== '.png'
  ) {
    throw new ManifestError(`${label} aceita somente arquivos PNG.`);
  }
  if (asset.kind !== 'figure') {
    throw new ManifestError(`${label}.kind deve ser "figure".`);
  }
  if (!asset.destination.startsWith('assets/')) {
    throw new ManifestError(`${label}.destination deve começar com assets/.`);
  }
}

function validateEvidence(evidence, index) {
  const label = `allowed_evidence[${index}]`;
  requireFields(evidence, FILE_FIELDS, label);
  evidence.source = assertRelativePath(evidence.source, `${label}.source`);
  evidence.destination = assertRelativePath(evidence.destination, `${label}.destination`);
  if (!['result', 'manifest', 'license'].includes(evidence.kind)) {
    throw new ManifestError(`${label}.kind não é permitido.`);
  }
  if (!evidence.destination.startsWith('evidence/')) {
    throw new ManifestError(`${label}.destination deve começar com evidence/.`);
  }

  const sourceExtension = path.posix.extname(evidence.source).toLowerCase();
  const destinationExtension = path.posix.extname(evidence.destination).toLowerCase();
  if (['.svg', '.html', '.htm'].includes(sourceExtension)) {
    throw new ManifestError(`${label}.source usa extensão ativa não permitida.`);
  }
  if (evidence.kind === 'license') {
    if (
      !['', '.txt'].includes(sourceExtension) ||
      destinationExtension !== '.txt'
    ) {
      throw new ManifestError(`${label} de licença deve publicar texto .txt.`);
    }
  } else if (sourceExtension !== '.json' || destinationExtension !== '.json') {
    throw new ManifestError(`${label} deve usar extensão .json.`);
  }
}

export function validateManifest(rawManifest) {
  requireRecord(rawManifest, 'manifesto');
  if (rawManifest.schema_version !== EXPECTED_SCHEMA_VERSION) {
    throw new ManifestError(
      `schema_version deve ser ${EXPECTED_SCHEMA_VERSION}.`,
    );
  }
  if (rawManifest.generated_root !== EXPECTED_GENERATED_ROOT) {
    throw new ManifestError(
      `generated_root deve ser "${EXPECTED_GENERATED_ROOT}".`,
    );
  }

  for (const field of [
    'public_pages',
    'allowed_assets',
    'allowed_evidence',
    'deny_prefixes',
    'deny_globs',
  ]) {
    if (!Array.isArray(rawManifest[field])) {
      throw new ManifestError(`${field} deve ser um array.`);
    }
  }

  const manifest = structuredClone(rawManifest);
  manifest.public_pages.forEach(validatePage);
  manifest.allowed_assets.forEach(validateAsset);
  manifest.allowed_evidence.forEach(validateEvidence);

  for (const [index, prefix] of manifest.deny_prefixes.entries()) {
    if (typeof prefix !== 'string' || prefix.length === 0) {
      throw new ManifestError(`deny_prefixes[${index}] deve ser string.`);
    }
  }

  assertUnique(manifest.public_pages, 'id', 'public_pages');
  assertUnique(manifest.public_pages, 'slug', 'public_pages');
  assertUnique(manifest.public_pages, 'order', 'public_pages');
  assertUnique(
    [
      ...manifest.public_pages,
      ...manifest.allowed_assets,
      ...manifest.allowed_evidence,
    ],
    'source',
    'manifesto',
  );
  assertUnique(
    [...manifest.allowed_assets, ...manifest.allowed_evidence],
    'destination',
    'manifesto',
  );

  const isDenied = createDenyMatcher(manifest);
  for (const entry of [
    ...manifest.public_pages,
    ...manifest.allowed_assets,
    ...manifest.allowed_evidence,
  ]) {
    if (isDenied(entry.source)) {
      throw new ManifestError(`Caminho negado tem precedência: ${entry.source}`);
    }
  }

  return {...manifest, isDenied};
}

async function readManifest(root) {
  const manifestPath = path.join(root, 'website', 'content-manifest.json');
  let text;
  try {
    text = await readFile(manifestPath, 'utf8');
  } catch (error) {
    throw new ManifestError(`Não foi possível ler ${EXPECTED_GENERATED_ROOT.replace('/.generated', '/content-manifest.json')}: ${error.code ?? error.message}`);
  }

  try {
    return validateManifest(JSON.parse(text));
  } catch (error) {
    if (error instanceof ManifestError) {
      throw error;
    }
    throw new ManifestError(`JSON inválido em website/content-manifest.json.`);
  }
}

async function validateSource(rootRealPath, root, source, isDenied) {
  const absolute = path.resolve(root, ...source.split('/'));
  if (!isWithin(root, absolute)) {
    throw new ManifestError(`Origem fora da raiz: ${source}`);
  }

  let sourceRealPath;
  try {
    const sourceStat = await stat(absolute);
    if (!sourceStat.isFile()) {
      throw new ManifestError(`Origem não é arquivo: ${source}`);
    }
    sourceRealPath = await realpath(absolute);
  } catch (error) {
    if (error instanceof ManifestError) {
      throw error;
    }
    throw new ManifestError(`Origem inexistente: ${source}`);
  }

  if (!isWithin(rootRealPath, sourceRealPath)) {
    throw new ManifestError(`Symlink aponta para fora da raiz: ${source}`);
  }
  const realRelativePath = toPosix(path.relative(rootRealPath, sourceRealPath));
  if (isDenied(realRelativePath)) {
    throw new ManifestError(
      `Caminho real negado tem precedência: ${realRelativePath} (via ${source})`,
    );
  }
  return absolute;
}

function splitTarget(target) {
  const hashIndex = target.indexOf('#');
  const queryIndex = target.indexOf('?');
  const indexes = [hashIndex, queryIndex].filter((index) => index >= 0);
  const suffixIndex = indexes.length > 0 ? Math.min(...indexes) : target.length;
  return {
    pathname: target.slice(0, suffixIndex),
    suffix: target.slice(suffixIndex),
  };
}

function isExternalTarget(target) {
  return (
    target.startsWith('#') ||
    target.startsWith('//') ||
    /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(target)
  );
}

function normalizeLinkSource(currentSource, linkPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(linkPath);
  } catch {
    decoded = linkPath;
  }
  const normalizedPath = toPosix(decoded);
  const joined = normalizedPath.startsWith('/')
    ? path.posix.normalize(normalizedPath.slice(1))
    : path.posix.normalize(
        path.posix.join(path.posix.dirname(currentSource), normalizedPath),
      );
  return joined.replace(/^\.\//, '').replace(/\/$/, '');
}

function comparisonKey(value) {
  return value.normalize('NFC').toLowerCase();
}

function resolveLocalTarget(target, context) {
  if (target === '' || isExternalTarget(target)) {
    return {kind: 'external'};
  }

  const {pathname, suffix} = splitTarget(target);
  const source = normalizeLinkSource(context.currentSource, pathname);
  const sourceKey = comparisonKey(source);
  const page =
    context.pagesBySource.get(sourceKey) ??
    context.pagesBySource.get(comparisonKey(`${source}/README.md`));
  const asset = context.assetsBySource.get(sourceKey);
  const evidence = context.evidenceBySource.get(sourceKey);
  const published =
    context.publishedTargets.get(comparisonKey(pathname.replace(/^\/+/, '')));

  if (page) {
    return {kind: 'page', destination: `${page.slug}${suffix}`, source};
  }
  if (asset) {
    return {
      kind: 'asset',
      destination: `/${asset.destination}${suffix}`,
      source,
    };
  }
  if (evidence) {
    return {
      kind: 'evidence',
      destination: `/${evidence.destination}${suffix}`,
      source,
    };
  }
  if (published) {
    return {
      kind: published.kind,
      destination: `/${published.destination}${suffix}`,
      source,
    };
  }
  return {kind: 'unpublished', source};
}

function visitTree(node, visitor) {
  visitor(node);
  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      visitTree(child, visitor);
    }
  }
}

function nodeOffsets(node) {
  const start = node.position?.start?.offset;
  const end = node.position?.end?.offset;
  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    throw new ManifestError('Parser Markdown não forneceu offsets confiáveis.');
  }
  return {start, end};
}

function inlineDestinationCursor(raw, node, nodeStart) {
  let cursor = raw.startsWith('![') ? 2 : 1;
  if (Array.isArray(node.children) && node.children.length > 0) {
    cursor = Math.max(
      cursor,
      nodeOffsets(node.children.at(-1)).end - nodeStart,
    );
  }

  while (cursor < raw.length) {
    if (raw[cursor] === '\\') {
      cursor += 2;
      continue;
    }
    if (raw[cursor] === ']' && raw[cursor + 1] === '(') {
      return cursor + 2;
    }
    cursor += 1;
  }
  throw new ManifestError('Link Markdown inline inválido.');
}

function destinationRange(source, node, definition = false) {
  const {start, end} = nodeOffsets(node);
  const raw = source.slice(start, end);
  let cursor;

  if (definition) {
    cursor = 1;
    let marker = -1;
    while (cursor < raw.length) {
      if (raw[cursor] === '\\') {
        cursor += 2;
        continue;
      }
      if (raw[cursor] === ']') {
        let candidate = cursor + 1;
        while (/[ \t]/.test(raw[candidate] ?? '')) {
          candidate += 1;
        }
        if (raw[candidate] === ':') {
          marker = candidate;
          break;
        }
      }
      cursor += 1;
    }
    if (marker < 0) {
      throw new ManifestError('Definição Markdown inválida.');
    }
    cursor = marker + 1;
  } else {
    cursor = inlineDestinationCursor(raw, node, start);
  }

  while (/\s/.test(raw[cursor] ?? '')) {
    cursor += 1;
  }

  if (raw[cursor] === '<') {
    const destinationStart = cursor + 1;
    cursor = destinationStart;
    while (cursor < raw.length && raw[cursor] !== '>') {
      if (raw[cursor] === '\\') {
        cursor += 1;
      }
      cursor += 1;
    }
    return {
      start: start + destinationStart,
      end: start + cursor,
    };
  }

  const destinationStart = cursor;
  let parenthesisDepth = 0;
  while (cursor < raw.length) {
    const character = raw[cursor];
    if (character === '\\') {
      cursor += 2;
      continue;
    }
    if (character === '(') {
      parenthesisDepth += 1;
    } else if (character === ')') {
      if (!definition && parenthesisDepth === 0) {
        break;
      }
      parenthesisDepth -= 1;
    } else if (/\s/.test(character) && parenthesisDepth === 0) {
      break;
    }
    cursor += 1;
  }
  return {
    start: start + destinationStart,
    end: start + cursor,
  };
}

function labelSource(markdown, node) {
  if (!Array.isArray(node.children) || node.children.length === 0) {
    return node.alt ?? '';
  }
  const first = nodeOffsets(node.children[0]).start;
  const last = nodeOffsets(node.children.at(-1)).end;
  return markdown.slice(first, last);
}

function htmlTokens(markdown, node) {
  const {start, end} = nodeOffsets(node);
  const raw = markdown.slice(start, end);
  const tokens = [];
  let cursor = 0;

  while (cursor < raw.length) {
    const opener = raw.indexOf('<', cursor);
    if (opener < 0) {
      break;
    }
    if (raw.startsWith('<!--', opener)) {
      const commentEnd = raw.indexOf('-->', opener + 4);
      const tokenEnd = commentEnd < 0 ? raw.length : commentEnd + 3;
      tokens.push({kind: 'comment', start: opener, end: tokenEnd});
      cursor = tokenEnd;
      continue;
    }

    let nameStart = opener + 1;
    let closing = false;
    if (raw[nameStart] === '/') {
      closing = true;
      nameStart += 1;
    }
    if (!/[a-zA-Z]/.test(raw[nameStart] ?? '')) {
      cursor = opener + 1;
      continue;
    }

    let nameEnd = nameStart + 1;
    while (/[a-zA-Z0-9:-]/.test(raw[nameEnd] ?? '')) {
      nameEnd += 1;
    }
    let quote = '';
    let tokenEnd = nameEnd;
    for (; tokenEnd < raw.length; tokenEnd += 1) {
      const character = raw[tokenEnd];
      if (quote) {
        if (character === quote) {
          quote = '';
        }
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === '>') {
        tokenEnd += 1;
        break;
      }
    }
    if (tokenEnd > raw.length || raw[tokenEnd - 1] !== '>') {
      break;
    }

    let beforeClose = tokenEnd - 2;
    while (beforeClose >= nameEnd && /\s/.test(raw[beforeClose])) {
      beforeClose -= 1;
    }
    tokens.push({
      kind: 'tag',
      start: opener,
      end: tokenEnd,
      name: raw.slice(nameStart, nameEnd).toLowerCase(),
      nameEnd,
      closing,
      selfClosing: raw[beforeClose] === '/',
    });
    cursor = tokenEnd;
  }
  return tokens;
}

function htmlProtectedRanges(markdown, htmlNodes) {
  const ranges = [];
  const stack = [];

  for (const node of htmlNodes) {
    const nodeStart = nodeOffsets(node).start;
    for (const token of htmlTokens(markdown, node)) {
      const absoluteStart = nodeStart + token.start;
      const absoluteEnd = nodeStart + token.end;
      if (token.kind === 'comment') {
        ranges.push({start: absoluteStart, end: absoluteEnd});
        continue;
      }
      if (!['pre', 'code'].includes(token.name)) {
        continue;
      }
      if (!token.closing && !token.selfClosing) {
        stack.push({name: token.name, start: absoluteStart});
        continue;
      }
      if (token.closing) {
        const openingIndex = stack.findLastIndex(
          (opening) => opening.name === token.name,
        );
        if (openingIndex >= 0) {
          const [opening] = stack.splice(openingIndex, 1);
          ranges.push({start: opening.start, end: absoluteEnd});
        }
      }
    }
  }

  for (const opening of stack) {
    ranges.push({start: opening.start, end: markdown.length});
  }
  return ranges;
}

function overlapsProtectedRange(start, end, protectedRanges) {
  return protectedRanges.some(
    (range) => start < range.end && end > range.start,
  );
}

function srcsetUrlRanges(value) {
  const ranges = [];
  let cursor = 0;

  while (cursor < value.length) {
    while (/\s/.test(value[cursor] ?? '')) {
      cursor += 1;
    }
    if (cursor >= value.length) {
      break;
    }
    if (value[cursor] === ',') {
      throw new ManifestError('srcset HTML ambíguo: candidato vazio.');
    }

    const start = cursor;
    while (cursor < value.length && !/\s/.test(value[cursor])) {
      cursor += 1;
    }
    let end = cursor;
    let separatedInUrl = false;
    while (end > start && value[end - 1] === ',') {
      separatedInUrl = true;
      end -= 1;
    }
    if (end === start) {
      throw new ManifestError('srcset HTML ambíguo: URL vazia.');
    }
    ranges.push({start, end});
    if (separatedInUrl) {
      continue;
    }

    while (/\s/.test(value[cursor] ?? '')) {
      cursor += 1;
    }
    const descriptorStart = cursor;
    while (cursor < value.length && value[cursor] !== ',') {
      cursor += 1;
    }
    const descriptor = value.slice(descriptorStart, cursor).trim();
    if (descriptor !== '') {
      const descriptors = descriptor.split(/\s+/);
      if (
        descriptors.length !== 1 ||
        !/^(?:[1-9]\d*w|(?:\d+(?:\.\d+)?|\.\d+)x)$/.test(descriptors[0])
      ) {
        throw new ManifestError('srcset HTML ambíguo: descriptor inválido.');
      }
    }

    if (cursor < value.length) {
      cursor += 1;
    }
  }
  return ranges;
}

function htmlAttributeEdits(markdown, node, context, protectedRanges) {
  const {start} = nodeOffsets(node);
  const raw = markdown.slice(start, nodeOffsets(node).end);
  const edits = [];

  for (const token of htmlTokens(markdown, node)) {
    const tokenStart = start + token.start;
    const tokenEnd = start + token.end;
    if (
      token.kind !== 'tag' ||
      token.closing ||
      overlapsProtectedRange(tokenStart, tokenEnd, protectedRanges)
    ) {
      continue;
    }

    let cursor = token.nameEnd;
    while (cursor < token.end - 1) {
      while (/\s/.test(raw[cursor] ?? '')) {
        cursor += 1;
      }
      if (cursor >= token.end - 1 || raw[cursor] === '/') {
        break;
      }

      const attributeStart = cursor;
      while (
        cursor < token.end - 1 &&
        !/[\s"'<>/=]/.test(raw[cursor])
      ) {
        cursor += 1;
      }
      const attribute = raw.slice(attributeStart, cursor).toLowerCase();
      while (/\s/.test(raw[cursor] ?? '')) {
        cursor += 1;
      }
      if (raw[cursor] !== '=') {
        continue;
      }
      cursor += 1;
      while (/\s/.test(raw[cursor] ?? '')) {
        cursor += 1;
      }

      const quote = raw[cursor] === '"' || raw[cursor] === "'" ? raw[cursor] : '';
      if (quote) {
        cursor += 1;
      }
      const valueStart = cursor;
      if (quote) {
        while (cursor < token.end - 1 && raw[cursor] !== quote) {
          cursor += 1;
        }
      } else {
        while (
          cursor < token.end - 1 &&
          !/[\s"'=<>`]/.test(raw[cursor])
        ) {
          cursor += 1;
        }
      }
      const valueEnd = cursor;
      if (quote && raw[cursor] === quote) {
        cursor += 1;
      }
      if (!['href', 'src', 'srcset'].includes(attribute)) {
        continue;
      }

      const target = raw.slice(valueStart, valueEnd);
      if (attribute === 'srcset') {
        for (const range of srcsetUrlRanges(target)) {
          const candidate = target.slice(range.start, range.end);
          const resolved = resolveLocalTarget(candidate, context);
          if (resolved.kind === 'external') {
            continue;
          }
          if (resolved.kind !== 'asset') {
            throw new ManifestError(
              `Imagem local não permitida em srcset: ${resolved.source}`,
            );
          }
          edits.push({
            start: start + valueStart + range.start,
            end: start + valueStart + range.end,
            replacement: resolved.destination,
          });
        }
        continue;
      }

      const resolved = resolveLocalTarget(target, context);
      if (resolved.kind === 'external') {
        continue;
      }
      if (attribute === 'src') {
        if (resolved.kind !== 'asset') {
          throw new ManifestError(`Imagem local não permitida: ${resolved.source}`);
        }
        edits.push({
          start: start + valueStart,
          end: start + valueEnd,
          replacement: resolved.destination,
        });
        continue;
      }
      if (resolved.kind === 'unpublished') {
        let removalStart = attributeStart;
        while (removalStart > token.nameEnd && /[ \t]/.test(raw[removalStart - 1])) {
          removalStart -= 1;
        }
        edits.push({
          start: start + removalStart,
          end: start + cursor,
          replacement: '',
        });
      } else {
        edits.push({
          start: start + valueStart,
          end: start + valueEnd,
          replacement: resolved.destination,
        });
      }
    }
  }
  return edits;
}

function applyEdits(markdown, edits) {
  const ordered = edits.sort((left, right) => right.start - left.start);
  let previousStart = markdown.length + 1;
  let result = markdown;
  for (const edit of ordered) {
    if (edit.end > previousStart) {
      throw new ManifestError('Patches Markdown sobrepostos.');
    }
    result =
      result.slice(0, edit.start) + edit.replacement + result.slice(edit.end);
    previousStart = edit.start;
  }
  return result;
}

const STATIC_HTML_ELEMENTS = new Set([
  'a',
  'abbr',
  'b',
  'blockquote',
  'br',
  'caption',
  'cite',
  'code',
  'col',
  'colgroup',
  'dd',
  'del',
  'details',
  'div',
  'dl',
  'dt',
  'em',
  'figcaption',
  'figure',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'hr',
  'i',
  'img',
  'kbd',
  'li',
  'mark',
  'ol',
  'p',
  'pre',
  'q',
  's',
  'small',
  'span',
  'strong',
  'sub',
  'summary',
  'sup',
  'table',
  'tbody',
  'td',
  'tfoot',
  'th',
  'thead',
  'tr',
  'u',
  'ul',
]);

function validatePublicMarkdown(markdown, source) {
  const markdownTree = unified().use(remarkParse).parse(markdown);
  const htmlNodes = [];
  visitTree(markdownTree, (node) => {
    if (node.type === 'html') {
      htmlNodes.push(node);
    }
  });

  const masked = [...markdown];
  for (const node of htmlNodes) {
    const {start, end} = nodeOffsets(node);
    const raw = markdown.slice(start, end);
    for (const token of htmlTokens(markdown, node)) {
      if (token.kind === 'comment') {
        continue;
      }
      if (!STATIC_HTML_ELEMENTS.has(token.name)) {
        throw new ManifestError(`MDX executável não permitido em ${source}.`);
      }
      const rawTag = raw.slice(token.start, token.end);
      if (
        /[{}]/.test(rawTag) ||
        /\s(?:on[a-z]+|dangerouslySetInnerHTML|srcDoc)\s*=/i.test(rawTag)
      ) {
        throw new ManifestError(`MDX executável não permitido em ${source}.`);
      }
    }
    for (let index = start; index < end; index += 1) {
      if (masked[index] !== '\n' && masked[index] !== '\r') {
        masked[index] = ' ';
      }
    }
  }

  let tree;
  try {
    tree = unified()
      .use(remarkParse)
      .use(remarkMath)
      .use(remarkMdx)
      .parse(masked.join(''));
  } catch (error) {
    throw new ManifestError(`Markdown/MDX inválido em ${source}: ${error.message}`);
  }

  visitTree(tree, (node) => {
    if (
      node.type === 'mdxjsEsm' ||
      node.type === 'mdxFlowExpression' ||
      node.type === 'mdxTextExpression'
    ) {
      throw new ManifestError(`MDX executável não permitido em ${source}.`);
    }
    if (
      node.type !== 'mdxJsxFlowElement' &&
      node.type !== 'mdxJsxTextElement'
    ) {
      return;
    }
    throw new ManifestError(`MDX executável não permitido em ${source}.`);
  });
}

export function rewriteMarkdownLinks(markdown, context) {
  const tree = unified().use(remarkParse).parse(markdown);
  const definitions = new Map();
  const allDefinitions = [];
  const references = [];
  const htmlNodes = [];
  const edits = [];

  visitTree(tree, (node) => {
    if (node.type === 'definition') {
      if (!definitions.has(node.identifier)) {
        definitions.set(node.identifier, node);
      }
      allDefinitions.push(node);
    } else if (node.type === 'linkReference' || node.type === 'imageReference') {
      references.push(node);
    } else if (node.type === 'html') {
      htmlNodes.push(node);
    }
  });

  const protectedRanges = htmlProtectedRanges(markdown, htmlNodes);

  visitTree(tree, (node) => {
    if (node.type === 'link' || node.type === 'image') {
      const {start, end} = nodeOffsets(node);
      if (overlapsProtectedRange(start, end, protectedRanges)) {
        return;
      }
      const resolved = resolveLocalTarget(node.url, context);
      if (resolved.kind === 'external') {
        return;
      }
      if (node.type === 'image' && resolved.kind !== 'asset') {
        throw new ManifestError(`Imagem local não permitida: ${resolved.source}`);
      }
      if (resolved.kind === 'unpublished') {
        const {start, end} = nodeOffsets(node);
        edits.push({
          start,
          end,
          replacement: `${labelSource(markdown, node)} (disponível no repositório-fonte)`,
        });
      } else {
        edits.push({
          ...destinationRange(markdown, node),
          replacement: resolved.destination,
        });
      }
    } else if (node.type === 'html') {
      edits.push(
        ...htmlAttributeEdits(markdown, node, context, protectedRanges),
      );
    }
  });

  for (const reference of references) {
    const referenceOffsets = nodeOffsets(reference);
    if (
      overlapsProtectedRange(
        referenceOffsets.start,
        referenceOffsets.end,
        protectedRanges,
      )
    ) {
      continue;
    }
    const definition = definitions.get(reference.identifier);
    if (!definition) {
      continue;
    }
    const resolved = resolveLocalTarget(definition.url, context);
    if (resolved.kind === 'external') {
      continue;
    }
    if (reference.type === 'imageReference' && resolved.kind !== 'asset') {
      throw new ManifestError(`Imagem local não permitida: ${resolved.source}`);
    }
    if (resolved.kind === 'unpublished') {
      const {start, end} = nodeOffsets(reference);
      edits.push({
        start,
        end,
        replacement: `${labelSource(markdown, reference)} (disponível no repositório-fonte)`,
      });
    }
  }

  for (const definition of allDefinitions) {
    const definitionOffsets = nodeOffsets(definition);
    if (
      overlapsProtectedRange(
        definitionOffsets.start,
        definitionOffsets.end,
        protectedRanges,
      )
    ) {
      continue;
    }
    const resolved = resolveLocalTarget(definition.url, context);
    if (resolved.kind === 'external') {
      continue;
    }
    if (resolved.kind === 'unpublished') {
      const {start, end} = nodeOffsets(definition);
      edits.push({
        start,
        end,
        replacement: `<!-- referência local não publicada: ${definition.identifier} -->`,
      });
    } else {
      edits.push({
        ...destinationRange(markdown, definition, true),
        replacement: resolved.destination,
      });
    }
  }

  return applyEdits(markdown, edits);
}

function frontmatter(page) {
  return [
    '---',
    `id: ${page.id}`,
    `title: ${JSON.stringify(page.title)}`,
    `slug: ${page.slug}`,
    `sidebar_position: ${page.order}`,
    `sidebar_label: ${JSON.stringify(page.sidebar_label ?? page.title)}`,
    'hide_title: true',
    '---',
    '',
    '',
  ].join('\n');
}

async function buildStaging({root, rootRealPath, manifest, stagingPath}) {
  const entries = [
    ...manifest.public_pages,
    ...manifest.allowed_assets,
    ...manifest.allowed_evidence,
  ];
  const sourcePaths = new Map();
  for (const entry of entries) {
    sourcePaths.set(
      entry.source,
      await validateSource(rootRealPath, root, entry.source, manifest.isDenied),
    );
  }

  const contextBase = {
    pagesBySource: new Map(
      manifest.public_pages.map((page) => [comparisonKey(page.source), page]),
    ),
    assetsBySource: new Map(
      manifest.allowed_assets.map((asset) => [comparisonKey(asset.source), asset]),
    ),
    evidenceBySource: new Map(
      manifest.allowed_evidence.map((evidence) => [
        comparisonKey(evidence.source),
        evidence,
      ]),
    ),
    publishedTargets: new Map(
      [
        ...manifest.allowed_assets.map((entry) => ({...entry, kind: 'asset'})),
        ...manifest.allowed_evidence.map((entry) => ({...entry, kind: 'evidence'})),
      ].map((entry) => [comparisonKey(entry.destination), entry]),
    ),
    isDenied: manifest.isDenied,
  };

  for (const evidence of manifest.allowed_evidence) {
    if (evidence.kind !== 'license') {
      const contents = await readFile(sourcePaths.get(evidence.source), 'utf8');
      try {
        JSON.parse(contents);
      } catch {
        throw new ManifestError(`JSON inválido: ${evidence.source}`);
      }
    }
  }

  await mkdir(path.join(stagingPath, 'docs'), {recursive: true});
  for (const page of manifest.public_pages) {
    const canonical = await readFile(sourcePaths.get(page.source), 'utf8');
    validatePublicMarkdown(canonical, page.source);
    const transformed = rewriteMarkdownLinks(canonical, {
      ...contextBase,
      currentSource: page.source,
    });
    const destination = path.join(stagingPath, 'docs', `${page.id}.md`);
    await writeFile(destination, `${frontmatter(page)}${transformed}`);
  }

  for (const entry of [...manifest.allowed_assets, ...manifest.allowed_evidence]) {
    const destination = path.join(
      stagingPath,
      'static',
      ...entry.destination.split('/'),
    );
    await mkdir(path.dirname(destination), {recursive: true});
    await copyFile(sourcePaths.get(entry.source), destination);
  }
}

async function fileMap(root) {
  const files = new Map();

  async function visit(directory) {
    let entries;
    try {
      entries = await readdir(directory, {withFileTypes: true});
    } catch (error) {
      if (error.code === 'ENOENT') {
        return;
      }
      throw error;
    }
    entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
    for (const entry of entries) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.isFile()) {
        files.set(toPosix(path.relative(root, absolute)), await readFile(absolute));
      } else {
        throw new ManifestError(`Tipo de arquivo inesperado na árvore gerada.`);
      }
    }
  }

  await visit(root);
  return files;
}

async function treesEqual(leftRoot, rightRoot) {
  const [left, right] = await Promise.all([fileMap(leftRoot), fileMap(rightRoot)]);
  if (left.size !== right.size) {
    return false;
  }
  for (const [relativePath, contents] of left) {
    const other = right.get(relativePath);
    if (!other || !contents.equals(other)) {
      return false;
    }
  }
  return true;
}

async function exists(target) {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

export async function publishStaging({
  generatedPath,
  stagingPath,
  rename = fsRename,
  remove = rm,
}) {
  const parent = path.dirname(generatedPath);
  const backupPath = path.join(
    parent,
    `${path.basename(generatedPath)}.bak-${process.pid}-${Date.now()}`,
  );
  const hadGenerated = await exists(generatedPath);
  let backupExists = false;
  let committed = false;

  try {
    if (hadGenerated) {
      await rename(generatedPath, backupPath);
      backupExists = true;
    }
    await rename(stagingPath, generatedPath);
    committed = true;
    if (backupExists) {
      await remove(backupPath, {recursive: true, force: true});
      backupExists = false;
    }
  } catch (error) {
    if (!committed && backupExists) {
      await remove(generatedPath, {recursive: true, force: true});
      await rename(backupPath, generatedPath);
      backupExists = false;
    }
    throw error;
  } finally {
    await remove(stagingPath, {recursive: true, force: true});
  }
}

export async function syncContent({root, check = false} = {}) {
  if (!root) {
    throw new ManifestError('A raiz do repositório é obrigatória.');
  }
  const absoluteRoot = path.resolve(root);
  const rootRealPath = await realpath(absoluteRoot);
  const manifest = await readManifest(absoluteRoot);
  const generatedPath = path.join(
    absoluteRoot,
    ...EXPECTED_GENERATED_ROOT.split('/'),
  );
  const parent = path.dirname(generatedPath);
  const stagingPath = path.join(
    parent,
    `${path.basename(generatedPath)}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  );

  await mkdir(parent, {recursive: true});
  try {
    await buildStaging({
      root: absoluteRoot,
      rootRealPath,
      manifest,
      stagingPath,
    });

    if (check) {
      const upToDate =
        (await exists(generatedPath)) &&
        (await treesEqual(stagingPath, generatedPath));
      if (!upToDate) {
        throw new ManifestError('Conteúdo gerado desatualizado.');
      }
      return {
        upToDate: true,
        counts: {
          pages: manifest.public_pages.length,
          assets: manifest.allowed_assets.length,
          evidence: manifest.allowed_evidence.length,
        },
      };
    }

    await publishStaging({generatedPath, stagingPath});
    return {
      upToDate: true,
      counts: {
        pages: manifest.public_pages.length,
        assets: manifest.allowed_assets.length,
        evidence: manifest.allowed_evidence.length,
      },
    };
  } finally {
    await rm(stagingPath, {recursive: true, force: true});
  }
}

async function runCli() {
  const args = process.argv.slice(2);
  if (args.some((argument) => argument !== '--check')) {
    throw new ManifestError('Uso: node scripts/sync-content.mjs [--check]');
  }
  const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(scriptDirectory, '..', '..');
  const check = args.includes('--check');
  const result = await syncContent({root, check});
  const {pages, assets, evidence} = result.counts;
  process.stdout.write(
    check
      ? `OK: conteúdo atualizado (${pages} páginas, ${assets} figuras, ${evidence} evidências).\n`
      : `OK: gerado (${pages} páginas, ${assets} figuras, ${evidence} evidências).\n`,
  );
}

const invokedPath = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : '';
if (import.meta.url === invokedPath) {
  runCli().catch((error) => {
    process.stderr.write(`Erro: ${error.message}\n`);
    process.exitCode = 1;
  });
}
