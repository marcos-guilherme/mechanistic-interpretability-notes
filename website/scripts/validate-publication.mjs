import {lstat, readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';
import process from 'node:process';
import {fileURLToPath} from 'node:url';

const require = createRequire(import.meta.url);
const spdxExceptions = require('spdx-exceptions');
const parseSpdxExpression = require('spdx-expression-parse');
const spdxLicenseIds = require('spdx-license-ids');

const knownExceptions = new Set(spdxExceptions);
const knownLicenseIds = new Set(spdxLicenseIds);
const operators = new Set(['AND', 'OR', 'WITH']);
const placeholderPattern =
  /(^|[^a-z0-9_])(tbd|todo|placeholder|changeme|change me)([^a-z0-9_]|$)/i;

function validateKnownIdentifiers(expression) {
  const tokens = expression.match(/[A-Za-z0-9][A-Za-z0-9.-]*/g) ?? [];
  let expectsException = false;

  for (const token of tokens) {
    const operator = token.toUpperCase();
    if (operators.has(operator)) {
      expectsException = operator === 'WITH';
      continue;
    }

    if (expectsException) {
      if (!knownExceptions.has(token)) {
        throw new Error(`Exceção SPDX inexistente: ${token}.`);
      }
      expectsException = false;
    } else if (!knownLicenseIds.has(token)) {
      throw new Error(`Identificador SPDX inexistente: ${token}.`);
    }
  }
}

export async function validatePublicationLicense({expression, licensePath}) {
  const normalizedExpression = expression?.trim() ?? '';
  if (!normalizedExpression) {
    throw new Error('PUBLIC_LICENSE_SPDX deve declarar uma expressão SPDX.');
  }

  validateKnownIdentifiers(normalizedExpression);
  try {
    parseSpdxExpression(normalizedExpression);
  } catch (error) {
    throw new Error(`Expressão SPDX inválida: ${error.message}`);
  }

  const metadata = await lstat(licensePath).catch(() => null);
  if (!metadata?.isFile() || metadata.isSymbolicLink()) {
    throw new Error('LICENSE deve ser um arquivo regular na raiz do projeto.');
  }

  const contents = await readFile(licensePath, 'utf8');
  if (!contents.trim()) {
    throw new Error('LICENSE contém somente whitespace; inclua o texto da licença.');
  }
  if (placeholderPattern.test(contents)) {
    throw new Error('LICENSE contém placeholder e não libera a publicação.');
  }
}

function argumentValue(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1] ?? '')) {
  try {
    const licensePath = path.resolve(argumentValue('--license') ?? '../LICENSE');
    await validatePublicationLicense({
      expression: process.env.PUBLIC_LICENSE_SPDX,
      licensePath,
    });
    console.log('Licença de publicação validada.');
  } catch (error) {
    console.error(`::error::${error.message}`);
    process.exitCode = 1;
  }
}
