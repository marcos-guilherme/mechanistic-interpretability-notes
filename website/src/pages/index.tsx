import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import StudyCard from '../components/StudyCard';
import styles from './index.module.css';

export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Interpretabilidade Aplicada ao Alinhamento de LLMs"
      description="Documentei três reproduções de interpretabilidade mecanística: circuito IOI, direção de recusa e J-space, com o que testei, os limites e os resultados inconclusivos.">
      <main>
        <header className={styles.hero}>
          <div className={styles.container}>
            <p className={styles.eyebrow}>Reproduções parciais</p>
            <Heading as="h1" className={styles.heroTitle}>
              Interpretabilidade Aplicada ao Alinhamento de LLMs
            </Heading>
            <p className={styles.heroLead}>
              Reuni três estudos que executei: o circuito IOI, a direção de
              recusa e o J-space. Registro o que testei, o que observei e até
              onde cada resultado permite concluir. Nenhuma das três é uma
              reprodução integral dos artigos originais.
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
              <p className={styles.eyebrow}>Documentação</p>
              <Heading as="h2" id="navigation-heading">
                Onde está cada parte do trabalho
              </Heading>
              <p>
                Começo pelo contexto dos estudos, pelos critérios que usei para
                reproduzir e pelo registro das decisões experimentais.
              </p>
            </div>
            <nav className={styles.navigationGrid} aria-label="Entradas da documentação">
              <Link className={styles.navigationCard} to="/sobre-o-projeto">
                <span className={styles.navigationTitle}>Sobre o projeto</span>
                <span>Como organizei os três estudos e o que cada um cobre.</span>
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
                O que é causal, o que é observação e o que ficou inconclusivo
              </Heading>
              <p>
                Distingo o que uma intervenção altera, o que uma análise sem
                intervenção apenas descreve e o que não passou nos critérios da
                hipótese.
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
                O que executei e até onde cada estudo chegou
              </Heading>
            </div>
            <div className={styles.studyGrid}>
              <StudyCard
                title="Circuito IOI"
                model="GPT-2 Small"
                evidence="Evidência causal"
                evidenceKind="causal"
                href="/estudos/ioi">
                Reconstruí o circuito de objeto indireto com atribuição,
                patching e testes de necessidade e suficiência.
              </StudyCard>
              <StudyCard
                title="Direção de recusa"
                model="Gemma 2B IT"
                evidence="Reprodução causal parcial"
                evidenceKind="partial"
                href="/estudos/direcao-de-recusa">
                Testei localmente uma direção residual associada à recusa, com
                intervenções e controles de especificidade.
              </StudyCard>
              <StudyCard
                title="J-space"
                model="Llama 3.1 8B Instruct"
                evidence="Causal inconclusivo · J-lens, 93 prompts"
                evidenceKind="inconclusive"
                href="/estudos/jspace-llama">
                A tentativa causal no Llama ficou inconclusiva. Depois avaliei
                a J-lens em 93 prompts, sem intervenção, e não tratei essas
                observações como causa.
              </StudyCard>
            </div>
          </div>
        </section>

        <section className={styles.statement} aria-labelledby="statement-heading">
          <div className={styles.narrowContainer}>
            <Heading as="h2" id="statement-heading">
              Também registrei falhas e resultados inconclusivos, sem
              apresentá-los como confirmações.
            </Heading>
            <p>
              Em cada estudo documento as escolhas, de onde saem os números e o
              que ainda não permite concluir.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
