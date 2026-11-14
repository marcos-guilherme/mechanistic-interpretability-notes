import assert from 'node:assert/strict';
import {mkdtemp, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {validatePublicationLicense} from '../scripts/validate-publication.mjs';

async function licenseFixture(t, contents) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'publication-license-'));
  t.after(() => rm(directory, {recursive: true, force: true}));
  const licensePath = path.join(directory, 'LICENSE');
  await writeFile(licensePath, contents);
  return licensePath;
}

test('aceita identificadores e exceções SPDX existentes', async (t) => {
  const licensePath = await licenseFixture(t, 'Project license text\n');

  await assert.doesNotReject(() =>
    validatePublicationLicense({
      expression:
        '(MIT OR Apache-2.0) AND GPL-2.0-only WITH Classpath-exception-2.0',
      licensePath,
    }),
  );
});

test('rejeita identificador SPDX inexistente', async (t) => {
  const licensePath = await licenseFixture(t, 'Project license text\n');

  await assert.rejects(
    () =>
      validatePublicationLicense({
        expression: 'Definitely-Not-A-License-9.9',
        licensePath,
      }),
    /identificador SPDX inexistente.*Definitely-Not-A-License-9\.9/i,
  );
});

test('rejeita exceção SPDX inexistente', async (t) => {
  const licensePath = await licenseFixture(t, 'Project license text\n');

  await assert.rejects(
    () =>
      validatePublicationLicense({
        expression: 'GPL-2.0-only WITH Not-A-Real-Exception',
        licensePath,
      }),
    /exceção SPDX inexistente.*Not-A-Real-Exception/i,
  );
});

test('rejeita sintaxe SPDX malformada mesmo com identificadores existentes', async (t) => {
  const licensePath = await licenseFixture(t, 'Project license text\n');

  await assert.rejects(
    () =>
      validatePublicationLicense({
        expression: 'MIT AND (Apache-2.0 OR)',
        licensePath,
      }),
    /expressão SPDX inválida/i,
  );
});

test('rejeita LICENSE contendo somente whitespace', async (t) => {
  const licensePath = await licenseFixture(t, ' \t\r\n  \n');

  await assert.rejects(
    () => validatePublicationLicense({expression: 'MIT', licensePath}),
    /LICENSE.*somente whitespace|LICENSE.*conteúdo/i,
  );
});
