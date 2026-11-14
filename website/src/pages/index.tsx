import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import StudyCard from '../components/StudyCard';
import styles from './index.module.css';

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Entrada prática em interpretabilidade mecanística"
      description="Registro documental do processo prático de experimentação e entrada na interpretabilidade mecanística: três estudos, métodos, evidências e limites explícitos.">
      <main>
        <header className={styles.hero}>
          <div className={styles.container}>
            <p className={styles.eyebrow}>Registro documental</p>
            <Heading as="h1" className={styles.heroTitle}>
              Uma entrada prática em interpretabilidade mecanística
            </Heading>
            <p className={styles.heroLead}>
              Este repositório documenta meu processo prático de experimentação e
              entrada na área. Registro três estudos, do enunciado inicial às
              intervenções, incluindo tentativas, decisões, falhas, bloqueios e
              resultados inconclusivos no mesmo plano das conclusões que os
              dados sustentam.
            </p>
            <div className={styles.heroActions}>
              <Link className={styles.primaryAction} to="/replicacoes">
                Ver os estudos
              </Link>
            </div>
          </div>
        </header>

        <section
          className={styles.navigationSection}
          aria-labelledby="navigation-heading">
          <div className={styles.container}>
            <div className={styles.sectionIntro}>
              <p className={styles.eyebrow}>Como navegar</p>
              <Heading as="h2" id="navigation-heading">
                Entradas para acompanhar o trabalho
              </Heading>
              <p>
                Pelo contexto do projeto, pelos critérios comuns ou pelo registro
                cronológico das decisões experimentais.
              </p>
            </div>
            <nav className={styles.navigationGrid} aria-label="Entradas do portal">
              <Link className={styles.navigationCard} to="/sobre-o-projeto">
                <span className={styles.navigationTitle}>Sobre o projeto</span>
                <span>Escopo, organização e lugar de cada estudo.</span>
              </Link>
              <Link className={styles.navigationCard} to="/metodologia">
                <span className={styles.navigationTitle}>Metodologia</span>
                <span>Critérios comuns de reprodução e interpretação.</span>
              </Link>
              <Link
                className={styles.navigationCard}
                to="/acompanhamento-experimentos">
                <span className={styles.navigationTitle}>Diário</span>
                <span>Decisões, incidentes e resultados em sequência.</span>
              </Link>
            </nav>
          </div>
        </section>

        <section
          className={styles.evidenceSection}
          aria-labelledby="evidence-heading">
          <div className={styles.container}>
            <div className={styles.sectionIntro}>
              <p className={styles.eyebrow}>Como ler as evidências</p>
              <Heading as="h2" id="evidence-heading">
                Estudos com conclusões de forças diferentes
              </Heading>
              <p>
                O portal distingue o que uma intervenção sustenta, o que uma
                observação descreve e o que permanece sem conclusão causal.
              </p>
            </div>
            <dl className={styles.evidenceList}>
              <div>
                <dt>Evidência causal</dt>
                <dd>
                  Uma intervenção altera o comportamento medido sob controles
                  explícitos.
                </dd>
              </div>
              <div>
                <dt>Evidência observacional</dt>
                <dd>
                  Um padrão é localizado ou descrito, sem demonstrar que ele
                  causa a resposta.
                </dd>
              </div>
              <div>
                <dt>Resultado inconclusivo</dt>
                <dd>
                  O protocolo não satisfaz os critérios necessários para
                  sustentar a hipótese causal.
                </dd>
              </div>
            </dl>
          </div>
        </section>

        <section className={styles.studiesSection} aria-labelledby="studies-heading">
          <div className={styles.container}>
            <div className={styles.sectionIntro}>
              <p className={styles.eyebrow}>Estudos</p>
              <Heading as="h2" id="studies-heading">
                Estudos registrados, com escopo e limites explícitos
              </Heading>
            </div>
            <div className={styles.studyGrid}>
              <StudyCard
                title="Circuito IOI"
                model="GPT-2 Small"
                evidence="Evidência causal"
                evidenceKind="causal"
                href="/estudos/ioi">
                Reconstrução do circuito de objeto indireto com atribuição,
                patching e testes de necessidade e suficiência.
              </StudyCard>
              <StudyCard
                title="Direção de recusa"
                model="Gemma 2B IT"
                evidence="Reprodução causal parcial"
                evidenceKind="partial"
                href="/estudos/direcao-de-recusa">
                Teste local de uma direção residual associada à recusa, com
                intervenções e controles de especificidade.
              </StudyCard>
              <StudyCard
                title="J-space"
                model="Llama 3.1 8B Instruct"
                evidence="Causal inconclusivo · atlas observacional"
                evidenceKind="inconclusive"
                href="/estudos/jspace-llama">
                Uma tentativa causal preservada como inconclusiva e um atlas
                posterior que descreve recuperações sem convertê-las em causa.
              </StudyCard>
            </div>
          </div>
        </section>

        <section className={styles.statement} aria-labelledby="statement-heading">
          <div className={styles.narrowContainer}>
            <Heading as="h2" id="statement-heading">
              Execuções inconclusivas e bloqueios aparecem aqui com a mesma
              visibilidade dos resultados que passaram nos critérios.
            </Heading>
            <p>
              Em cada percurso documento as escolhas, a proveniência dos números
              e o que ainda não permite concluir, inclusive quando o desfecho
              não foi favorável.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
