import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {createServer} from 'node:http';
import path from 'node:path';

const host = process.env.HOST ?? '127.0.0.1';
const port = Number(process.env.PORT ?? 4173);
const baseUrl = '/ioi-replication/';
const buildDir = path.resolve('build');

const contentTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.txt', 'text/plain; charset=utf-8'],
  ['.woff2', 'font/woff2'],
  ['.xml', 'application/xml; charset=utf-8'],
]);

async function existingFile(candidates) {
  for (const candidate of candidates) {
    const info = await stat(candidate).catch(() => null);
    if (info?.isFile()) {
      return candidate;
    }
  }
  return null;
}

function sendFile(response, file, statusCode = 200, method = 'GET') {
  response.writeHead(statusCode, {
    'Content-Type':
      contentTypes.get(path.extname(file).toLowerCase()) ??
      'application/octet-stream',
  });
  if (method === 'HEAD') {
    response.end();
  } else {
    createReadStream(file).pipe(response);
  }
}

const server = createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url ?? '/', `http://${host}:${port}`);
    if (!requestUrl.pathname.startsWith(baseUrl)) {
      sendFile(response, path.join(buildDir, '404.html'), 404, request.method);
      return;
    }

    const relativeUrl = decodeURIComponent(
      requestUrl.pathname.slice(baseUrl.length),
    ).replace(/^\/+|\/+$/g, '');
    const relativePath = relativeUrl || 'index';
    const resolved = path.resolve(buildDir, relativePath);
    if (
      resolved !== buildDir &&
      !resolved.startsWith(`${buildDir}${path.sep}`)
    ) {
      sendFile(response, path.join(buildDir, '404.html'), 404, request.method);
      return;
    }

    const file = await existingFile([
      resolved,
      `${resolved}.html`,
      path.join(resolved, 'index.html'),
    ]);
    if (file) {
      sendFile(response, file, 200, request.method);
      return;
    }
    sendFile(response, path.join(buildDir, '404.html'), 404, request.method);
  } catch {
    response.writeHead(400, {'Content-Type': 'text/plain; charset=utf-8'});
    response.end('Requisição inválida.');
  }
});

server.listen(port, host, () => {
  console.log(`Project Pages disponível em http://${host}:${port}${baseUrl}`);
});
