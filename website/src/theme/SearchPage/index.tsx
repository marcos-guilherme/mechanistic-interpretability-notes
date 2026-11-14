import OriginalSearchPage from '@theme-original/SearchPage';
import {useEffect} from 'react';

export default function SearchPage(): React.JSX.Element {
  useEffect(() => {
    const searchInput =
      document.querySelector<HTMLInputElement>('input[name="q"]');
    const searchMain = searchInput?.closest('.container.margin-vert--lg');
    searchMain?.setAttribute('role', 'main');
    if (searchInput?.getAttribute('aria-label') === 'Search') {
      searchInput.setAttribute('aria-label', 'Buscar');
    }
  }, []);

  return <OriginalSearchPage />;
}
