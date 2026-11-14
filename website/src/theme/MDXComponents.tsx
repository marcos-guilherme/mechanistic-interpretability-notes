import OriginalMDXComponents from '@theme-original/MDXComponents';
import type {ComponentProps} from 'react';

function AccessibleTable(props: ComponentProps<'table'>): React.JSX.Element {
  return <table tabIndex={0} {...props} />;
}

export default {
  ...OriginalMDXComponents,
  table: AccessibleTable,
};
