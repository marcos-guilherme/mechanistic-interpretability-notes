# Gradio J-lens Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar seleção entre J-lens v1 e v2 ao laboratório Gradio, usando v2 por padrão.

**Architecture:** O bootstrap valida e carrega as duas lentes em readouts separados. Cada requisição escolhe um readout por uma chave estável, sem trocar a lente durante uma geração. O transporte remoto envia e verifica os dois artefatos.

**Tech Stack:** Python, PyTorch, Gradio, pytest e orquestrador Colab A100.

## Global Constraints

- v2 é o padrão.
- v1 permanece disponível para comparação.
- A lente não muda durante uma geração.
- Status e exportação registram versão e hash.
- Hashes devem corresponder à especificação.
- O modelo Qwen continua fixo.

---

### Task 1: Contrato de seleção e readout

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_lab.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_lab_readout.py`
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_lab_runtime.py`
- Modify: testes `test_qwen_lab*.py` correspondentes

- [ ] Escrever testes para v2 padrão, v1 explícita e isolamento por geração.
- [ ] Confirmar falha antes da implementação.
- [ ] Implementar registro de readouts por versão e proveniência.
- [ ] Executar os testes focados.

### Task 2: Interface e bootstrap

**Files:**
- Modify: `projects/refusal_direction/src/refusal_direction_replication/qwen_lab_app.py`
- Modify: `projects/refusal_direction/configs/qwen25_coder_7b_lab.json`
- Modify: `projects/refusal_direction/tests/test_qwen_lab_app.py`

- [ ] Adicionar os dois caminhos e hashes à configuração.
- [ ] Carregar e validar as duas lentes.
- [ ] Adicionar o seletor Gradio com v2 como padrão.
- [ ] Mostrar versão e hash usados no status e na exportação.
- [ ] Executar os testes focados.

### Task 3: Transporte e teste remoto

**Files:**
- Modify: transporte e instalador do perfil `qwen25-coder-lab`
- Modify: `projects/refusal_direction/tests/test_qwen_lab_transport.py`
- Modify: `projects/refusal_direction/LABORATORIO_QWEN.md`

- [ ] Transportar e verificar os dois arquivos de lente.
- [ ] Executar testes locais do perfil sem carregar o modelo.
- [ ] Iniciar A100, executar smoke curto com v1 e v2 e abrir a interface.
- [ ] Encerrar a A100 quando o teste manual terminar.
