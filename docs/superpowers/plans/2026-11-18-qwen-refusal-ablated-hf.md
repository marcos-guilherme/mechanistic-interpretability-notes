# Qwen Refusal-Ablated Hugging Face Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exportar uma versão completa e baixável do Qwen2.5-Coder-7B-Instruct que aplique internamente a ablação de 20% da Refusal Direction usada no experimento.

**Architecture:** Uma subclasse Hugging Face do Qwen registra a direção como buffer persistente e instala, durante a inicialização, as mesmas projeções usadas pelo runtime de hooks. Um exportador monta o checkpoint completo a partir da revisão congelada, e um validador compara o modelo exportado com o runtime atual antes de permitir upload privado.

**Tech Stack:** Python 3.11+, PyTorch, Transformers, Safetensors, Hugging Face Hub, pytest e executor Colab A100 existente.

## Global Constraints

- Modelo base: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Revisão base: `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Coeficiente de ablação: `0.2`.
- O checkpoint deve reproduzir `directional_ablation_hook_specs`, não uma ortogonalização aproximada dos pesos.
- O carregamento deve usar `trust_remote_code=True`.
- A publicação inicial deve ser privada.
- Uma divergência acima das tolerâncias congeladas bloqueia o upload.
- Não publicar prompts privados, gerações de avaliação, ativações ou credenciais.

---

### Task 1: Modelo Hugging Face com intervenção persistente

**Files:**
- Create: `projects/refusal_direction/hf_qwen_ablated/configuration_qwen2_refusal_ablated.py`
- Create: `projects/refusal_direction/hf_qwen_ablated/modeling_qwen2_refusal_ablated.py`
- Create: `projects/refusal_direction/hf_qwen_ablated/__init__.py`
- Create: `projects/refusal_direction/tests/test_qwen_hf_ablated_model.py`

- [ ] **Step 1: Escrever testes unitários**

Cobrir configuração, buffer persistente, coeficiente fixo em `[0, 1]`,
instalação única dos hooks e equivalência da projeção com
`ablate_direction`.

- [ ] **Step 2: Confirmar falha inicial**

Run: `uv run pytest tests/test_qwen_hf_ablated_model.py -q`
Expected: FAIL porque o pacote ainda não existe.

- [ ] **Step 3: Implementar configuração e modelo**

Registrar `auto_map`, proveniência, hash da direção, camada de origem e
coeficiente. Instalar pre-hooks nos blocos e forward hooks nas saídas de
atenção e MLP, usando a direção armazenada como buffer.

- [ ] **Step 4: Executar testes**

Run: `uv run pytest tests/test_qwen_hf_ablated_model.py -q`
Expected: PASS.

### Task 2: Exportador determinístico

**Files:**
- Create: `projects/refusal_direction/scripts/export_qwen_hf_ablated.py`
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_hf_export.py`
- Create: `projects/refusal_direction/tests/test_qwen_hf_export.py`
- Create: `projects/refusal_direction/configs/qwen25_coder_7b_hf_ablated.json`

- [ ] **Step 1: Escrever testes do manifesto e das barreiras**

Cobrir revisão base, coeficiente `0.2`, hash do vetor, recusa a sobrescrever
diretório incompleto e allowlist dos arquivos exportados.

- [ ] **Step 2: Implementar exportação**

Carregar o checkpoint base na revisão congelada, construir a subclasse,
inserir a direção validada, salvar pesos em Safetensors, tokenizer, template,
configuração, código remoto e manifesto de proveniência.

- [ ] **Step 3: Executar testes focados**

Run: `uv run pytest tests/test_qwen_hf_export.py -q`
Expected: PASS sem carregar o modelo de 7B localmente.

### Task 3: Validação de equivalência

**Files:**
- Create: `projects/refusal_direction/src/refusal_direction_replication/qwen_hf_validate.py`
- Create: `projects/refusal_direction/scripts/validate_qwen_hf_ablated.py`
- Create: `projects/refusal_direction/tests/test_qwen_hf_validate.py`

- [ ] **Step 1: Escrever testes dos gates**

Cobrir igualdade de tokens gulosa, tolerância de logits, KL, refusal, código,
hashes e bloqueio de upload quando qualquer gate falha.

- [ ] **Step 2: Implementar comparação**

Comparar checkpoint exportado e runtime de hooks nos prompts congelados, sem
salvar textos. Produzir apenas métricas agregadas e hashes.

- [ ] **Step 3: Executar testes**

Run: `uv run pytest tests/test_qwen_hf_validate.py -q`
Expected: PASS.

### Task 4: Exportação e validação na A100

**Files:**
- Modify: `tools/colab/run_refusal.py`
- Create: `projects/refusal_direction/scripts/colab_export_qwen_hf_ablated.py`
- Modify: `tools/colab/tests/test_run_refusal.py`

- [ ] **Step 1: Adicionar perfil isolado**

Criar ações para sincronizar, exportar, validar, buscar o recibo e sempre
encerrar a sessão. O perfil deve exigir A100.

- [ ] **Step 2: Validar o orquestrador localmente**

Run: `uv run pytest tools/colab/tests/test_run_refusal.py -q`
Expected: PASS sem iniciar GPU.

- [ ] **Step 3: Executar remotamente**

Iniciar a A100, autenticar o Hugging Face, exportar e executar a equivalência.
Buscar o manifesto e o recibo compacto mesmo em caso de falha.

### Task 5: Model card e upload privado

**Files:**
- Create: `projects/refusal_direction/hf_qwen_ablated/MODEL_CARD.md`
- Create: `projects/refusal_direction/scripts/publish_qwen_hf_ablated.py`
- Create: `projects/refusal_direction/tests/test_qwen_hf_publish.py`
- Modify: `projects/refusal_direction/README.md`
- Modify: `projects/refusal_direction/PROCESSO.md`

- [ ] **Step 1: Escrever testes de publicação**

Verificar repositório privado, recibo de equivalência obrigatório, allowlist,
ausência de arquivos sensíveis e conteúdo mínimo do model card.

- [ ] **Step 2: Implementar upload bloqueado por validação**

Criar ou atualizar
`marcos-guilherme/Qwen2.5-Coder-7B-Instruct-refusal-ablated-20pct` como
privado somente após todos os gates passarem.

- [ ] **Step 3: Documentar limites**

Explicar que a variante é experimental, reduz parcialmente refusal no recorte
avaliado e não demonstra remoção universal ou seletiva.

- [ ] **Step 4: Publicar e testar recarga**

Recarregar do repositório privado com `trust_remote_code=True`, repetir o
painel guloso mínimo e registrar o hash final.
