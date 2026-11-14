import OriginalSearchBar from '@theme-original/SearchBar';
import {useEffect, type ComponentProps} from 'react';

type SearchBarProps = ComponentProps<typeof OriginalSearchBar>;

export default function SearchBar(props: SearchBarProps): React.JSX.Element {
  useEffect(() => {
    document
      .querySelectorAll<HTMLInputElement>(
        '.navbar__search-input[aria-label="Search"]',
      )
      .forEach((input) => input.setAttribute('aria-label', 'Buscar'));
  }, []);

  return <OriginalSearchBar {...props} />;
}
