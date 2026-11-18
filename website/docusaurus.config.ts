import type {Config} from '@docusaurus/types';
import type {Options, ThemeConfig} from '@docusaurus/preset-classic';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

const localUrl = 'http://localhost:3000';

function normalizeBaseUrl(value: string): string {
  const normalized = value.replace(/^\/+|\/+$/g, '');
  return normalized ? `/${normalized}/` : '/';
}

function resolveDeployment(): {url: string; baseUrl: string} {
  const configuredUrl = process.env.DOCUSAURUS_URL?.replace(/\/+$/, '');
  const configuredBaseUrl = process.env.DOCUSAURUS_BASE_URL;

  if (configuredUrl) {
    return {
      url: configuredUrl,
      baseUrl: configuredBaseUrl ? normalizeBaseUrl(configuredBaseUrl) : '/',
    };
  }

  const [owner, repository, ...extra] = (
    process.env.GITHUB_REPOSITORY ?? ''
  ).split('/');

  if (owner && repository && extra.length === 0) {
    const isUserOrOrganizationSite =
      repository.toLowerCase() === `${owner.toLowerCase()}.github.io`;

    return {
      url: `https://${owner}.github.io`,
      baseUrl: configuredBaseUrl
        ? normalizeBaseUrl(configuredBaseUrl)
        : isUserOrOrganizationSite
          ? '/'
          : `/${repository}/`,
    };
  }

  return {
    url: localUrl,
    baseUrl: configuredBaseUrl ? normalizeBaseUrl(configuredBaseUrl) : '/',
  };
}

const deployment = resolveDeployment();

const config: Config = {
  title: 'Interpretabilidade Aplicada ao Alinhamento de LLMs',
  tagline: 'Reproduções parciais de IOI, direção de recusa e J-space',
  favicon: 'img/favicon.svg',
  url: deployment.url,
  baseUrl: deployment.baseUrl,
  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  onDuplicateRoutes: 'throw',

  i18n: {
    defaultLocale: 'pt-BR',
    locales: ['pt-BR'],
    localeConfigs: {
      'pt-BR': {
        label: 'Português (Brasil)',
        htmlLang: 'pt-BR',
      },
    },
  },

  staticDirectories: ['static', '.generated/static'],

  markdown: {
    mermaid: true,
  },

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['pt', 'en'],
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsDir: '.generated/docs',
        docsRouteBasePath: '/',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          path: '.generated/docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'monthly',
          priority: 0.5,
        },
      } satisfies Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    metadata: [
      {
        name: 'description',
        content:
          'Documentei três reproduções de interpretabilidade mecanística: circuito IOI, direção de recusa e J-space, com o que testei, os limites e os resultados inconclusivos.',
      },
      {
        name: 'keywords',
        content:
          'interpretabilidade mecanística, alinhamento, LLMs, replicação, evidência causal',
      },
    ],
    navbar: {
      title: 'Interpretabilidade Aplicada ao Alinhamento de LLMs',
      hideOnScroll: false,
      items: [
        {to: '/sobre-o-projeto', label: 'Sobre', position: 'left'},
        {to: '/replicacoes', label: 'Estudos', position: 'left'},
        {to: '/metodologia', label: 'Método', position: 'left'},
        {
          to: '/acompanhamento-experimentos',
          label: 'Diário',
          position: 'left',
        },
        {to: '/referencias', label: 'Referências', position: 'left'},
        {type: 'search', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'Licença MIT.',
    },
    prism: {
      additionalLanguages: ['python', 'json'],
    },
  } satisfies ThemeConfig,
};

export default config;
