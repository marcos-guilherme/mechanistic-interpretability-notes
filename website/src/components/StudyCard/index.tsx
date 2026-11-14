import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import type {ReactNode} from 'react';

import styles from './styles.module.css';

type EvidenceKind = 'causal' | 'partial' | 'inconclusive';

type StudyCardProps = {
  title: string;
  model: string;
  evidence: string;
  evidenceKind: EvidenceKind;
  href: string;
  children: ReactNode;
};

export default function StudyCard({
  title,
  model,
  evidence,
  evidenceKind,
  href,
  children,
}: StudyCardProps): ReactNode {
  return (
    <article className={styles.card}>
      <div className={styles.meta}>
        <span className={styles.model}>{model}</span>
        <span className={styles[evidenceKind]}>{evidence}</span>
      </div>
      <Heading as="h3" className={styles.title}>
        {title}
      </Heading>
      <p className={styles.description}>{children}</p>
      <Link
        aria-label={`Ler o registro do estudo ${title}`}
        className={styles.link}
        to={href}>
        Ler o registro <span aria-hidden="true">→</span>
      </Link>
    </article>
  );
}
