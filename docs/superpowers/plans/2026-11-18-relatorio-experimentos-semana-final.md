# Relatório dos Experimentos da Semana Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar um relatório local de cinco a sete páginas, em Markdown e PDF, sobre os experimentos da semana final.

**Architecture:** O LaTeX será a fonte narrativa e de diagramação. O PDF será compilado diretamente com XeLaTeX; fonte e PDF permanecerão na pasta local.

**Tech Stack:** Markdown, XeLaTeX e MiKTeX.

## Global Constraints

- Escrever em primeira pessoa e sem tom promocional.
- Preservar números e força das conclusões.
- Identificar o steering esparso como pendente e a revisão como AI-assisted.
- Manter Markdown e PDF fora do Git.

---

### Task 1: Redação e diagramação do relatório

**Files:**
- Create local-only: `docs/entregas/2026-11-18-relatorio-experimentos-semana-final.tex`

- [ ] Reunir apenas métricas sustentadas pelos artefatos validados.
- [ ] Escrever uma narrativa contínua, com poucas seções e duas tabelas no máximo.
- [ ] Revisar repetições, termos artificiais e referências internas desnecessárias.

### Task 2: Compilação do PDF

**Files:**
- Create local-only: `docs/entregas/2026-11-18-relatorio-experimentos-semana-final.pdf`
- Modify: `.gitignore`

- [ ] Adicionar LaTeX, PDF e temporários à exclusão do Git.
- [ ] Compilar duas vezes com XeLaTeX.
- [ ] Remover somente os arquivos auxiliares, preservando o LaTeX.

### Task 3: Verificação

- [ ] Confirmar que o PDF abre e tem entre cinco e sete páginas.
- [ ] Verificar presença do título, autor, data e limitações.
- [ ] Confirmar que o LaTeX e o PDF locais não aparecem em `git status`.
