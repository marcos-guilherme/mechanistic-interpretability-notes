import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';
import contentManifest from './content-manifest.json';

const sidebarGroups = [
  {id: 'inicio', label: 'Início'},
  {id: 'metodo-e-percurso', label: 'Método e percurso'},
  {id: 'estudos', label: 'Estudos'},
] as const;

const knownGroups = new Set(sidebarGroups.map(({id}) => id));
for (const page of contentManifest.public_pages) {
  if (!knownGroups.has(page.sidebar as (typeof sidebarGroups)[number]['id'])) {
    throw new Error(`Grupo de sidebar desconhecido no manifesto: ${page.sidebar}`);
  }
}

const sidebars: SidebarsConfig = {
  portalSidebar: sidebarGroups.map(({id, label}) => ({
    type: 'category' as const,
    label,
    collapsed: false,
    items: contentManifest.public_pages
      .filter((page) => page.sidebar === id)
      .sort((left, right) => left.order - right.order)
      .map((page) => page.id),
  })),
};

export default sidebars;
