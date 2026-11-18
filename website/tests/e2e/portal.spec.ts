import AxeBuilder from '@axe-core/playwright';
import {expect, test, type Page} from '@playwright/test';

const mainLandmark = (page: Page) => page.locator('main, [role="main"]');
const projectPath = (route: string) =>
  `/ioi-replication${route === '/' ? '/' : route}`;

const routes = [
  ['/', /^Interpretabilidade Aplicada ao Alinhamento de LLMs$/],
  ['/sobre-o-projeto', /^Interpretabilidade Aplicada ao Alinhamento de LLMs$/],
  ['/replicacoes', /^Replicações$/i],
  ['/metodologia', /Metodologia comum de replicação/i],
  [
    '/guia-pesquisa-interpretabilidade',
    /Guia de pesquisa em interpretabilidade mecanística/i,
  ],
  ['/referencias', /Fontes e referências/i],
  ['/acompanhamento-experimentos', /Acompanhamento dos experimentos/i],
  ['/estudos/ioi', /Replicação do circuito IOI/i],
  [
    '/estudos/ioi/processo',
    /Processo de estudo e reprodução do circuito IOI/i,
  ],
  ['/estudos/direcao-de-recusa', /Replicação da direção de recusa/i],
  [
    '/estudos/direcao-de-recusa/processo',
    /Processo de estudo e reprodução da direção de recusa/i,
  ],
  ['/estudos/jspace-llama', /J-space no Llama 3\.1 8B Instruct/i],
  [
    '/estudos/jspace-llama/processo',
    /Processo de estudo causal e observacional do J-space/i,
  ],
] as const;

async function expectNoBrokenAssets(page: Page, route: string) {
  const brokenResponses: string[] = [];
  page.on('response', (response) => {
    if (response.status() >= 400) {
      brokenResponses.push(`${response.status()} ${response.url()}`);
    }
  });

  const response = await page.goto(projectPath(route), {waitUntil: 'networkidle'});
  expect(response?.status(), `status de ${route}`).toBe(200);

  const assetUrls = await page
    .locator('img[src], script[src], link[rel="stylesheet"][href]')
    .evaluateAll((elements) =>
      elements.map((element) => {
        if (element instanceof HTMLLinkElement) {
          return element.href;
        }
        return (element as HTMLImageElement | HTMLScriptElement).src;
      }),
    );

  for (const assetUrl of new Set(assetUrls)) {
    const assetResponse = await page.context().request.get(assetUrl);
    expect(assetResponse.status(), `asset ${assetUrl} em ${route}`).toBeLessThan(400);
  }
  expect(brokenResponses, `respostas quebradas em ${route}`).toEqual([]);
}

test.describe('smoke do artefato Project Pages', () => {
  for (const [route, heading] of routes) {
    test(`${route} responde e apresenta o conteúdo principal`, async ({page}) => {
      await expectNoBrokenAssets(page, route);
      await expect(mainLandmark(page)).toHaveCount(1);
      await expect(page.getByRole('heading', {level: 1, name: heading})).toBeVisible();
    });
  }

  test('404 mantém status e conteúdo localizados', async ({page}) => {
    const response = await page.goto(projectPath('/rota-que-nao-existe'), {
      waitUntil: 'networkidle',
    });
    expect(response?.status()).toBe(404);
    await expect(
      page.getByRole('heading', {level: 1, name: 'Página não encontrada'}),
    ).toBeVisible();
    await expect(mainLandmark(page)).toHaveCount(1);
  });
});

test('busca está em pt-BR e encontra conteúdo publicado', async ({page}) => {
  await page.goto(projectPath('/'), {waitUntil: 'networkidle'});
  await expect(page.getByPlaceholder('Buscar')).toBeVisible();

  await page.goto(projectPath('/search'), {waitUntil: 'networkidle'});
  await expect(
    page.getByRole('heading', {level: 1, name: 'Buscar na documentação'}),
  ).toBeVisible();
  const search = page.getByRole('searchbox');
  await expect(search).toHaveAccessibleName('Buscar');
  await search.fill('circuito IOI');
  await expect(
    page.getByRole('heading', {
      level: 1,
      name: /Resultados da busca por "circuito IOI"/i,
    }),
  ).toBeVisible();
  await expect(mainLandmark(page)).toContainText(/documentos? encontrados?/i);
  await expect(mainLandmark(page).locator('article').first()).toBeVisible();
});

test('navegação funciona no desktop e no menu móvel', async ({page}, testInfo) => {
  await page.goto(projectPath('/'), {waitUntil: 'networkidle'});

  if (testInfo.project.name === 'mobile-chromium') {
    await page
      .getByRole('button', {name: 'Alternar painel de navegação'})
      .click();
    const mobileNavigation = page.locator('.navbar-sidebar');
    await expect(mobileNavigation).toBeVisible();
    await mobileNavigation.getByRole('link', {name: 'Sobre'}).click();
  } else {
    await page.getByRole('link', {name: 'Sobre', exact: true}).click();
  }

  await expect(page).toHaveURL(/\/ioi-replication\/sobre-o-projeto$/);
  await expect(
    page.getByRole('heading', {
      level: 1,
      name: /^Interpretabilidade Aplicada ao Alinhamento de LLMs$/,
    }),
  ).toBeVisible();
});

test('exibe marca curta na navbar e licença MIT no rodapé', async ({page}) => {
  await page.goto(projectPath('/'), {waitUntil: 'networkidle'});
  await expect(page.locator('.navbar__title:visible')).toHaveText(
    'Interpretabilidade',
  );
  await expect(page.locator('footer')).toContainText('Licença MIT.');
});

test.describe('acessibilidade representativa', () => {
  for (const route of ['/', '/estudos/ioi/processo', '/search']) {
    test(`${route} não tem violações serious/critical`, async ({page}) => {
      await page.goto(projectPath(route), {waitUntil: 'networkidle'});
      const results = await new AxeBuilder({page})
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
        .analyze();
      const severe = results.violations.filter(({impact}) =>
        ['serious', 'critical'].includes(impact ?? ''),
      );
      expect(
        severe,
        severe
          .map(({id, impact, help, nodes}) => `${impact} ${id}: ${help} (${nodes.length})`)
          .join('\n'),
      ).toEqual([]);

      const main = mainLandmark(page);
      await expect(main).toHaveCount(1);
      const headingLevels = await main
        .locator('h1, h2, h3, h4')
        .evaluateAll((headings) =>
          headings.map((heading) => Number(heading.tagName.slice(1))),
        );
      expect(headingLevels[0]).toBe(1);
      for (let index = 1; index < headingLevels.length; index += 1) {
        expect(
          headingLevels[index] - headingLevels[index - 1],
          `salto de heading em ${route}: ${headingLevels.join(' → ')}`,
        ).toBeLessThanOrEqual(1);
      }

      for (const image of await main.locator('img').all()) {
        await expect(image).toHaveAttribute('alt', /\S+/);
      }
    });
  }

  test('teclado alcança um controle com foco visível', async ({page}) => {
    await page.goto(projectPath('/'), {waitUntil: 'networkidle'});
    await page.keyboard.press('Tab');
    const focus = page.locator(':focus');
    await expect(focus).toBeVisible();
    const style = await focus.evaluate((element) => {
      const computed = getComputedStyle(element);
      return {
        outlineStyle: computed.outlineStyle,
        outlineWidth: Number.parseFloat(computed.outlineWidth),
      };
    });
    expect(style.outlineStyle).not.toBe('none');
    expect(style.outlineWidth).toBeGreaterThanOrEqual(2);
  });
});

test('respeita preferência de cor escura', async ({browser}) => {
  const context = await browser.newContext({colorScheme: 'dark'});
  const page = await context.newPage();
  await page.goto('/ioi-replication/', {waitUntil: 'networkidle'});
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await context.close();
});
