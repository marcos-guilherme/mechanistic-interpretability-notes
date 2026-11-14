import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {load} from 'js-yaml';

const repositoryRoot = path.resolve('..');
const workflowPath = path.join(
  repositoryRoot,
  '.github',
  'workflows',
  'deploy-docs.yml',
);
const publishingPath = path.resolve('PUBLISHING.md');

async function readWorkflow() {
  const source = await readFile(workflowPath, 'utf8');
  return {source, workflow: load(source)};
}

function stepCommand(step) {
  return typeof step?.run === 'string' ? step.run : '';
}

function findStepIndex(steps, predicate) {
  return steps.findIndex((step) => predicate(step, stepCommand(step)));
}

function requiredJobs(workflow, jobName, visited = new Set()) {
  const needs = workflow.jobs?.[jobName]?.needs;
  for (const dependency of Array.isArray(needs) ? needs : needs ? [needs] : []) {
    if (!visited.has(dependency)) {
      visited.add(dependency);
      requiredJobs(workflow, dependency, visited);
    }
  }
  return visited;
}

function isPublishingAction(step) {
  return /actions\/(?:upload-pages-artifact|deploy-pages)@/.test(
    String(step.uses ?? ''),
  );
}

test('mantém a publicação restrita a acionamento manual', async () => {
  const {workflow} = await readWorkflow();
  assert.deepEqual(Object.keys(workflow.on ?? {}), ['workflow_dispatch']);
});

test('restringe o acionamento à branch principal configurada no repositório', async () => {
  const {workflow} = await readWorkflow();
  const steps = workflow.jobs?.verify?.steps ?? [];
  const gate = steps.find((step) =>
    /DISPATCH_REF_TYPE/.test(stepCommand(step)),
  );
  const command = stepCommand(gate);

  assert.equal(gate?.env?.DISPATCH_REF_TYPE, '${{ github.ref_type }}');
  assert.equal(gate?.env?.DISPATCH_REF_NAME, '${{ github.ref_name }}');
  assert.equal(
    gate?.env?.DEFAULT_BRANCH,
    '${{ github.event.repository.default_branch }}',
  );
  assert.match(command, /DISPATCH_REF_TYPE[^]*branch/);
  assert.match(command, /DISPATCH_REF_NAME[^]*DEFAULT_BRANCH/);
  assert.match(command, /::error::/);
});

test('usa Node 20, instalação reproduzível e Chromium com dependências', async () => {
  const {workflow} = await readWorkflow();
  const steps = workflow.jobs?.verify?.steps ?? [];
  const setupNode = steps.find((step) =>
    String(step.uses ?? '').startsWith('actions/setup-node@'),
  );

  assert.equal(String(setupNode?.with?.['node-version']), '20');
  assert.equal(setupNode?.with?.cache, 'npm');
  assert.equal(setupNode?.with?.['cache-dependency-path'], 'website/package-lock.json');
  assert.ok(
    steps.some(
      (step) =>
        stepCommand(step).trim() === 'npm ci' &&
        step['working-directory'] === 'website',
    ),
    'o workflow deve executar npm ci em website',
  );
  assert.ok(
    steps.some((step) =>
      /playwright\s+install\s+--with-deps\s+chromium/.test(stepCommand(step)),
    ),
    'o workflow deve instalar Chromium e suas dependências',
  );
});

test('falha de modo fechado antes de qualquer upload', async () => {
  const {source, workflow} = await readWorkflow();
  const steps = workflow.jobs?.verify?.steps ?? [];
  const gateIndex = findStepIndex(
    steps,
    (step) => /PUBLICATION_APPROVED/.test(stepCommand(step)),
  );
  const uploadIndexes = steps
    .map((step, index) => ({index, uses: String(step.uses ?? '')}))
    .filter(({uses}) => /upload-pages-artifact@/.test(uses))
    .map(({index}) => index);

  assert.notEqual(gateIndex, -1, 'gate de publicação ausente');
  assert.ok(uploadIndexes.length > 0, 'upload do artefato Pages ausente');
  assert.ok(uploadIndexes.every((index) => gateIndex < index));

  const gate = steps[gateIndex];
  const command = stepCommand(gate);
  assert.equal(gate.env?.PUBLICATION_APPROVED, '${{ vars.PUBLICATION_APPROVED }}');
  assert.equal(gate.env?.PUBLIC_AUTHOR_NAME, '${{ vars.PUBLIC_AUTHOR_NAME }}');
  assert.equal(gate.env?.PUBLIC_LICENSE_SPDX, '${{ vars.PUBLIC_LICENSE_SPDX }}');
  assert.match(command, /PUBLICATION_APPROVED/);
  assert.match(command, /PUBLIC_AUTHOR_NAME/);
  assert.match(command, /PUBLIC_LICENSE_SPDX/);
  assert.match(command, /GITHUB_WORKSPACE.*LICENSE/);
  assert.match(command, /-L\s+"\$license_path"/);
  assert.match(command, /grep\s+-q[^]*\[\^\[:space:\]\]/);
  assert.match(command, /::error::/);
  assert.match(command, /placeholder|changeme|tbd|todo/i);
  assert.doesNotMatch(source, /secrets\./i);
  assert.doesNotMatch(command, /projects[/\\]jspace/i);
});

test('valida a licença com Node depois do npm ci e antes de publicar', async () => {
  const {workflow} = await readWorkflow();
  const steps = workflow.jobs?.verify?.steps ?? [];
  const installIndex = findStepIndex(
    steps,
    (_step, command) => command.trim() === 'npm ci',
  );
  const licenseValidationIndex = findStepIndex(
    steps,
    (_step, command) => /node scripts\/validate-publication\.mjs/.test(command),
  );
  const uploadIndex = steps.findIndex(isPublishingAction);
  const validationStep = steps[licenseValidationIndex];

  assert.ok(installIndex >= 0 && installIndex < licenseValidationIndex);
  assert.ok(licenseValidationIndex < uploadIndex);
  assert.equal(validationStep?.['working-directory'], 'website');
  assert.equal(
    validationStep?.env?.PUBLIC_LICENSE_SPDX,
    '${{ vars.PUBLIC_LICENSE_SPDX }}',
  );
});

test('valida, deriva a URL pública e audita o build final antes do upload', async () => {
  const {source, workflow} = await readWorkflow();
  const steps = workflow.jobs?.verify?.steps ?? [];
  const uploadIndex = findStepIndex(steps, (step) =>
    /upload-pages-artifact@/.test(String(step.uses ?? '')),
  );
  const deriveIndex = findStepIndex(
    steps,
    (_step, command) =>
      /GITHUB_REPOSITORY/.test(command) &&
      /DOCUSAURUS_URL/.test(command) &&
      /DOCUSAURUS_BASE_URL/.test(command),
  );
  const finalBuildIndex = steps.findIndex(
    (step, index) =>
      index > deriveIndex && /npm run build/.test(stepCommand(step)),
  );
  const finalAuditIndex = steps.findIndex(
    (step, index) =>
      index > finalBuildIndex && /npm run audit:build/.test(stepCommand(step)),
  );
  const fixedProjectValidationIndex = findStepIndex(
    steps,
    (_step, command) =>
      /npm run build:project-page/.test(command) &&
      /npm run audit:project-page/.test(command),
  );

  for (const required of [
    'npm test',
    'npm run check',
    'npm run typecheck',
    'npm run build',
    'npm run audit:build',
  ]) {
    const index = findStepIndex(steps, (_step, command) =>
      command.includes(required),
    );
    assert.ok(index >= 0 && index < uploadIndex, `${required} deve preceder o upload`);
  }

  assert.ok(fixedProjectValidationIndex >= 0);
  assert.ok(fixedProjectValidationIndex < deriveIndex);
  assert.match(stepCommand(steps[deriveIndex]), /GITHUB_ENV/);
  assert.ok(deriveIndex >= 0 && deriveIndex < finalBuildIndex);
  assert.ok(finalBuildIndex < finalAuditIndex);
  assert.ok(finalAuditIndex < uploadIndex);
  assert.doesNotMatch(source, /PUBLIC_DOMAIN/);
});

test('usa somente as actions oficiais de GitHub Pages com permissões mínimas', async () => {
  const {workflow} = await readWorkflow();
  const verify = workflow.jobs?.verify;
  const deploy = workflow.jobs?.deploy;
  const allUses = Object.values(workflow.jobs ?? {})
    .flatMap((job) => job.steps ?? [])
    .map((step) => String(step.uses ?? ''));

  assert.equal(workflow.permissions?.contents, 'read');
  assert.deepEqual(deploy?.permissions, {
    pages: 'write',
    'id-token': 'write',
  });
  assert.equal(deploy?.needs, 'verify');
  assert.equal(deploy?.environment?.name, 'github-pages');
  assert.match(String(deploy?.environment?.url ?? ''), /page_url/);
  assert.ok(workflow.concurrency?.group);
  assert.equal(workflow.concurrency?.['cancel-in-progress'], false);
  assert.ok(allUses.some((value) => /actions\/configure-pages@/.test(value)));
  assert.ok(allUses.some((value) => /actions\/upload-pages-artifact@/.test(value)));
  assert.ok(allUses.some((value) => /actions\/deploy-pages@/.test(value)));
  assert.ok((verify?.steps ?? []).length > 0);
});

test('não permite ignorar falhas nem publicar em job paralelo ao gate', async () => {
  const {workflow} = await readWorkflow();
  const verifySteps = workflow.jobs?.verify?.steps ?? [];
  const gateIndex = findStepIndex(
    verifySteps,
    (_step, command) => /PUBLICATION_APPROVED/.test(command),
  );

  for (const [jobName, job] of Object.entries(workflow.jobs ?? {})) {
    assert.ok(
      job['continue-on-error'] === undefined ||
        job['continue-on-error'] === false,
      `${jobName} ignora falhas`,
    );

    for (const [index, step] of (job.steps ?? []).entries()) {
      assert.ok(
        step['continue-on-error'] === undefined ||
          step['continue-on-error'] === false,
        `${jobName}[${index}] ignora falhas`,
      );
      assert.doesNotMatch(
        stepCommand(step),
        /\|\|\s*true\b|;\s*true\b|\bexit\s+0\b|\bset\s+\+e\b/,
        `${jobName}[${index}] neutraliza falhas`,
      );

      if (!isPublishingAction(step)) {
        continue;
      }

      assert.equal(
        job.if,
        undefined,
        `${jobName} não deve alterar a condição padrão de sucesso`,
      );
      assert.equal(
        step.if,
        undefined,
        `${jobName}[${index}] não deve alterar a condição padrão de sucesso`,
      );
      if (jobName === 'verify') {
        assert.ok(index > gateIndex, 'upload deve ocorrer depois do gate');
      } else {
        assert.ok(
          requiredJobs(workflow, jobName).has('verify'),
          `${jobName} deve depender transitivamente de verify`,
        );
      }
    }
  }
});

test('não chama provedores de modelo, Python, notebooks ou experimentos', async () => {
  const {source} = await readWorkflow();
  assert.doesNotMatch(
    source,
    /hugging[\s-]?face|\bhf[_-]|\bgcp\b|google[\s-]?cloud|python|\.py\b|ipynb|jupyter|notebook|experimento|experiment/i,
  );
});

test('documenta a checklist humana e o bloqueio atual de publicação', async () => {
  const publishing = await readFile(publishingPath, 'utf8');

  for (const expected of [
    /owner|propriet[aá]rio/i,
    /nome do reposit[oó]rio/i,
    /URL do reposit[oó]rio/i,
    /branch principal/i,
    /LICENSE.*raiz/is,
    /third[- ]party|terceiros/i,
    /projects\/jspace_llama\/data\/anthropic-jlens\/LICENSE/i,
    /autoria|identidade p[uú]blica/i,
    /footer|rodap[eé]/i,
    /dom[ií]nio/i,
    /GitHub Pages/i,
    /GitHub Actions/i,
    /PUBLICATION_APPROVED/,
    /PUBLIC_AUTHOR_NAME/,
    /PUBLIC_LICENSE_SPDX/,
    /auditoria final/i,
    /Node 20/i,
    /validate:project-page/,
    /configur\w*(?:-o)?[^]*Settings[^]*Pages/i,
    /DNS.*HTTPS/is,
    /geraç[aã]o final.*GITHUB_REPOSITORY/is,
  ]) {
    assert.match(publishing, expected);
  }

  assert.doesNotMatch(publishing, /`PUBLIC_DOMAIN`/);
  assert.match(
    publishing,
    /validate:project-page[^]*local[^]*(fix|fixa|fixo)/i,
  );
  assert.match(publishing, /n[aã]o h[aá] remote/i);
  assert.match(publishing, /n[aã]o h[aá].*LICENSE.*raiz/is);
  assert.match(publishing, /n[aã]o est[aá] pronto para publicar/i);
});
