import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './404.module.css';

export default function NotFound(): React.JSX.Element {
  return (
    <Layout title="Página não encontrada">
      <main className={styles.main}>
        <div className={styles.content}>
          <p className={styles.code}>Erro 404</p>
          <Heading as="h1">Página não encontrada</Heading>
          <p>
            O endereço pode ter mudado ou ainda não fazer parte do recorte
            público do site.
          </p>
          <div className={styles.actions}>
            <Link className={styles.primaryAction} to="/">
              Voltar ao início
            </Link>
            <Link to="/replicacoes">Ver os estudos</Link>
          </div>
        </div>
      </main>
    </Layout>
  );
}
