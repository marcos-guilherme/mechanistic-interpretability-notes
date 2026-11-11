# Atlas observacional do J-space — Plano de implementação

> **Para agentes executores:** SUB-SKILL OBRIGATÓRIA: usar
> `superpowers:subagent-driven-development` (recomendado) ou
> `superpowers:executing-plans` para executar este plano tarefa por tarefa. Os
> passos usam caixas de seleção (`- [ ]`) para acompanhamento.

**Objetivo:** Medir, sem geração ou intervenção causal, o que a J-lens do
Llama 3.1 8B recupera nos 93 prompts multihop publicados pela Anthropic e
produzir um painel detalhado de 12 casos.

**Arquitetura:** Um carregador local valida a cópia fixada do conjunto oficial e
audita sua tokenização no Llama. Um avaliador calcula rankings diretos da
J-lens e da logit lens no último token dos 93 prompts; um leitor separado aplica
`gradient_pursuit` apenas ao painel pré-registrado. Um pipeline observacional
isolado gera JSONs validados e figuras, enquanto o fluxo causal anterior e seus
artefatos permanecem inalterados.

**Tecnologias:** Python 3.11–3.13, PyTorch, `jspace` fixado, Transformers,
Hugging Face Hub, JSON Schema, Matplotlib, pytest, nbformat e Colab L4.

## Restrições globais

- Modelo: `meta-llama/Llama-3.1-8B-Instruct`, carregado em CUDA com
  `torch.bfloat16`.
- Lente:
  `llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`
  do repositório `neuronpedia/jacobian-lens`.
- Código `jspace`:
  `04c822a255fe0a10d84e08ec499e5e09f4b7d37a`.
- Dataset upstream:
  `anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Dataset local: exatamente 93 itens e SHA-256
  `2cd5a47aa3d7627708843bc61f03ab1d10eda4184f357c5cbf93c97184e32c9c`.
- Usar os prompts como completações brutas, preservando inclusive espaços
  finais; não aplicar chat template.
- Ler o último token do prompt nas camadas do workspace band; analisar todas as
  camadas ajustadas somente como resultado secundário.
- Não chamar geração, `LensSwap`, clamps, controles aleatórios ou qualquer hook
  que modifique ativações.
- Manter ranking direto da J-lens, logit lens e decomposição por
  `gradient_pursuit` como objetos distintos.
- Calcular `pass@1`, `pass@5` e `pass@10` em ranks base 1, seguindo a fração de
  intermediários por item.
- Preservar `results/execution_manifest.json` e
  `results/final_evaluation.json` da execução causal anterior.
- Manter artefatos brutos fora do Git e não registrar credenciais, caches,
  caminhos locais ou tensores.
- Preservar todas as alterações preexistentes e não relacionadas na árvore de
  trabalho.

---

### Task 1: Fixar o conjunto oficial e suas dependências

**Arquivos:**
- Criar: `projects/jspace_llama/data/anthropic-jlens/LICENSE`
- Criar: `projects/jspace_llama/data/anthropic-jlens/README.md`
- Criar:
  `projects/jspace_llama/data/anthropic-jlens/evaluations/lens-eval-multihop.json`
- Modificar: `projects/jspace_llama/pyproject.toml`
- Modificar: `projects/jspace_llama/uv.lock`
- Modificar: `tools/colab/run_jspace.py`
- Modificar: `projects/jspace_llama/src/jspace_llama/provenance.py`
- Modificar: `projects/jspace_llama/scripts/validate_results.py`
- Testar: `projects/jspace_llama/tests/test_project_config.py`
- Testar: `projects/jspace_llama/tests/test_colab_orchestrator.py`
- Testar: `projects/jspace_llama/tests/test_artifact_pipeline.py`

**Interfaces:**
- Consome: os arquivos Apache-2.0 no commit upstream fixado.
- Produz: uma cópia local verificável e incluída no arquivo enviado ao Colab;
  `runtime_source_sha256()` passa a cobrir os dados observacionais.

- [ ] **Passo 1: Escrever testes que exijam o dataset e Matplotlib**

Adicionar a `test_project_config.py`:

```python
import hashlib
import json


DATASET = (
    ROOT
    / "data"
    / "anthropic-jlens"
    / "evaluations"
    / "lens-eval-multihop.json"
)
DATASET_SHA256 = (
    "2cd5a47aa3d7627708843bc61f03ab1d"
    "10eda4184f357c5cbf93c97184e32c9c"
)


def test_observation_dataset_is_pinned_and_complete() -> None:
    payload = DATASET.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == DATASET_SHA256
    document = json.loads(payload)
    assert len(document["items"]) == 93
    assert {item["name"] for item in document["items"]} >= {
        "spider-legs",
        "amazon-language",
        "nhop-primary-planet",
    }


def test_static_figure_dependency_is_runtime_locked() -> None:
    dependencies = set(load_config()["project"]["dependencies"])
    assert any(item.startswith("matplotlib") for item in dependencies)
```

Em `test_colab_orchestrator.py`, ampliar o teste do arquivo remoto para exigir
`jspace_llama/data/anthropic-jlens/evaluations/lens-eval-multihop.json`. Em
`test_artifact_pipeline.py`, ampliar o teste de hash da fonte para demonstrar
que alterar o JSON muda `runtime_source_sha256()`.

- [ ] **Passo 2: Executar os testes e confirmar a falha**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_project_config.py tests/test_colab_orchestrator.py tests/test_artifact_pipeline.py -q
```

Esperado: falhas por dataset ausente, dependência ausente e diretório `data`
fora do arquivo e do hash de proveniência.

- [ ] **Passo 3: Copiar os arquivos oficiais na revisão fixada**

Executar a partir da raiz:

```powershell
New-Item -ItemType Directory -Force projects/jspace_llama/data/anthropic-jlens/evaluations
python -c "import pathlib,subprocess; pathlib.Path('projects/jspace_llama/data/anthropic-jlens/LICENSE').write_bytes(subprocess.check_output(['gh','api','-H','Accept: application/vnd.github.raw+json','repos/anthropics/jacobian-lens/contents/LICENSE?ref=581d398613e5602a5af361e1c34d3a92ea82ba8e']))"
python -c "import pathlib,subprocess; pathlib.Path('projects/jspace_llama/data/anthropic-jlens/README.md').write_bytes(subprocess.check_output(['gh','api','-H','Accept: application/vnd.github.raw+json','repos/anthropics/jacobian-lens/contents/data/evaluations/README.md?ref=581d398613e5602a5af361e1c34d3a92ea82ba8e']))"
python -c "import pathlib,subprocess; pathlib.Path('projects/jspace_llama/data/anthropic-jlens/evaluations/lens-eval-multihop.json').write_bytes(subprocess.check_output(['gh','api','-H','Accept: application/vnd.github.raw+json','repos/anthropics/jacobian-lens/contents/data/evaluations/lens-eval-multihop.json?ref=581d398613e5602a5af361e1c34d3a92ea82ba8e']))"
```

Os comandos escrevem os bytes devolvidos pela API sem conversão de encoding.
Acrescentar ao README local a revisão upstream, o SHA-256 e a observação de que
apenas o conjunto multihop foi copiado.

- [ ] **Passo 4: Incluir dados, figuras e Matplotlib no contrato do projeto**

Adicionar `"matplotlib>=3.9"` às dependências de runtime e executar:

```powershell
cd projects/jspace_llama
uv lock
```

Em `tools/colab/run_jspace.py`, acrescentar:

```python
ARCHIVE_SUFFIXES = {
    "data": frozenset({"", ".json", ".md"}),
    "notebooks": frozenset({".ipynb"}),
    "schemas": frozenset({".json"}),
    "scripts": frozenset({".py"}),
    "src": frozenset({".py"}),
    "tests": frozenset({".py"}),
}
```

Quando `suffix == ""`, aceitar somente o caminho relativo
`data/anthropic-jlens/LICENSE`; qualquer outro arquivo sem extensão dentro de
`data` deve falhar. Incluir `data` nos conjuntos de fontes críticas em
`tools/colab/run_jspace.py`, `provenance.py` e `validate_results.py`. Os três
devem rotular o mesmo arquivo como
`projects/jspace_llama/data/anthropic-jlens/evaluations/lens-eval-multihop.json`
e também cobrir `LICENSE` e `README.md`.

Como o artefato causal é histórico, ampliar `validate_results.py` para
recalcular o digest a partir dos blobs do commit registrado quando
`provenance.source.commit` for diferente do `HEAD` atual e
`scoped_dirty == false`. Usar `git ls-tree` e `git show`, sem checkout. Quando o
commit registrado for o atual, manter a comparação contra os arquivos do
worktree. Acrescentar um teste que valide o manifest causal atual depois da
adição dos novos arquivos e outro que rejeite um commit ou digest adulterado.

- [ ] **Passo 5: Rodar os testes e registrar a tarefa**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_project_config.py tests/test_colab_orchestrator.py tests/test_artifact_pipeline.py -q
cd ../..
git add projects/jspace_llama/data projects/jspace_llama/pyproject.toml projects/jspace_llama/uv.lock projects/jspace_llama/src/jspace_llama/provenance.py projects/jspace_llama/scripts/validate_results.py projects/jspace_llama/tests/test_project_config.py projects/jspace_llama/tests/test_colab_orchestrator.py projects/jspace_llama/tests/test_artifact_pipeline.py tools/colab/run_jspace.py
git commit -m "feat(jspace): pin Anthropic multihop prompts"
```

Esperado: testes selecionados aprovados e somente os arquivos listados no
commit.

---

### Task 2: Validar itens e auditar a tokenização

**Arquivos:**
- Criar: `projects/jspace_llama/src/jspace_llama/observation_data.py`
- Criar: `projects/jspace_llama/tests/test_observation_data.py`

**Interfaces:**
- Consome:
  `load_multihop_items(path: Path) -> tuple[MultihopItem, ...]`.
- Produz:
  `audit_tokenization(items, tokenizer) -> tuple[dict[str, object], ...]` e
  `canonical_first_token(tokenizer, text) -> tuple[int, tuple[int, ...], str]`.

- [ ] **Passo 1: Escrever testes de schema, hash e variantes**

Criar `test_observation_data.py` com:

```python
from pathlib import Path

import pytest

from jspace_llama.observation_data import (
    DatasetValidationError,
    audit_tokenization,
    canonical_first_token,
    load_multihop_items,
)


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {
            " Brazil": [11],
            "Brazil": [12, 13],
            " multiword": [20, 21],
            "multiword": [22, 23],
        }[text]


def test_canonical_token_prefers_leading_space_and_records_full_sequence() -> None:
    assert canonical_first_token(FakeTokenizer(), "Brazil") == (
        11,
        (11,),
        "leading_space",
    )


def test_audit_marks_only_the_selected_variant_as_single_token() -> None:
    items = (
        type("Item", (), {
            "name": "one",
            "intermediates": ("Brazil", "multiword"),
        })(),
    )
    audit = audit_tokenization(items, FakeTokenizer())
    assert audit[0]["intermediates"][0]["single_token"] is True
    assert audit[0]["intermediates"][1]["single_token"] is False


def test_loader_rejects_wrong_digest(tmp_path: Path) -> None:
    path = tmp_path / "lens-eval-multihop.json"
    path.write_text('{"items":[]}', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_multihop_items(path)
```

Acrescentar casos para nomes duplicados, campos extras, lista vazia,
intermediário vazio, prompt sem espaço final e contagem diferente de 93.

- [ ] **Passo 2: Confirmar que os testes falham**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_data.py -q
```

Esperado: falha de importação porque `observation_data.py` ainda não existe.

- [ ] **Passo 3: Implementar o carregador estrito**

Criar:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from jspace_llama.jsonio import strict_json_loads


DATASET_SHA256 = (
    "2cd5a47aa3d7627708843bc61f03ab1d"
    "10eda4184f357c5cbf93c97184e32c9c"
)
EXPECTED_ITEM_COUNT = 93


class DatasetValidationError(ValueError):
    pass


PANEL_NAMES = (
    "carnival-ocean",
    "amazon-language",
    "mars-color",
    "spider-legs",
    "basketball-players",
    "atomic-79-symbol",
    "planet-3-moons",
    "rhyme-rain-neighbor",
    "etym-wargod-month",
    "chem-photosynthesis-Z",
    "nhop-primary-planet",
    "inv-sunrise-opposite",
)


@dataclass(frozen=True)
class MultihopItem:
    name: str
    prompt: str
    target: str
    intermediates: tuple[str, ...]


def load_multihop_items(path: Path) -> tuple[MultihopItem, ...]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise DatasetValidationError("multihop dataset SHA-256 mismatch")
    document = strict_json_loads(payload)
    if not isinstance(document, dict) or set(document) != {"items"}:
        raise DatasetValidationError("multihop dataset root must contain only items")
    raw_items = document["items"]
    if not isinstance(raw_items, list) or len(raw_items) != EXPECTED_ITEM_COUNT:
        raise DatasetValidationError("multihop dataset must contain 93 items")
    items: list[MultihopItem] = []
    names: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict) or set(raw) != {
            "name", "prompt", "target", "intermediates"
        }:
            raise DatasetValidationError("multihop item has unexpected fields")
        if raw["name"] in names:
            raise DatasetValidationError("multihop item name is duplicated")
        if not raw["prompt"].endswith(" "):
            raise DatasetValidationError("multihop prompt must retain final space")
        if (
            not isinstance(raw["intermediates"], list)
            or not raw["intermediates"]
            or not all(
                isinstance(value, str) and value
                for value in raw["intermediates"]
            )
        ):
            raise DatasetValidationError("intermediates must be non-empty strings")
        names.add(raw["name"])
        items.append(MultihopItem(
            name=raw["name"],
            prompt=raw["prompt"],
            target=raw["target"],
            intermediates=tuple(raw["intermediates"]),
        ))
    return tuple(items)
```

Implementar `canonical_first_token()` tentando `" " + text.lstrip()` antes de
`text`, rejeitando ambas apenas quando não produzirem IDs. A auditoria deve
registrar `token_id`, `token_ids`, `variant`, `single_token` e os hashes do
prompt e da sequência tokenizada, sem armazenar caminhos.

- [ ] **Passo 4: Rodar e registrar**

```powershell
uv run pytest tests/test_observation_data.py -q
cd ../..
git add projects/jspace_llama/src/jspace_llama/observation_data.py projects/jspace_llama/tests/test_observation_data.py
git commit -m "feat(jspace): audit multihop tokenization"
```

---

### Task 3: Implementar o ranking direto e `pass@k`

**Arquivos:**
- Criar: `projects/jspace_llama/src/jspace_llama/observation_readout.py`
- Criar: `projects/jspace_llama/tests/test_observation_readout.py`

**Interfaces:**
- Consome: `MultihopItem`, auditoria de tokenização, `model`, `dictionary` e
  duas sequências de camadas.
- Produz:
  `evaluate_direct_readout(...) -> dict[str, object]`,
  `rank_one_based(logits, token_id) -> int` e
  `aggregate_pass_at_k(items, ks=(1, 5, 10)) -> dict[str, float | int]`.

- [ ] **Passo 1: Escrever testes de rank e fração por item**

Criar:

```python
import torch

from jspace_llama.observation_readout import (
    aggregate_pass_at_k,
    rank_one_based,
)


def test_rank_is_one_based_and_uses_competition_ranking() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 2.0])
    assert rank_one_based(logits, 1) == 1
    assert rank_one_based(logits, 3) == 3


def test_pass_at_k_averages_each_items_intermediate_fraction() -> None:
    records = [
        {"intermediates": [{"best_rank": 1}]},
        {"intermediates": [{"best_rank": 2}, {"best_rank": 9}]},
    ]
    assert aggregate_pass_at_k(records) == {
        "n_items": 2,
        "n_intermediates": 3,
        "pass@1": 0.5,
        "pass@5": 0.75,
        "pass@10": 1.0,
    }
```

Adicionar um fake de `lens.apply()` que devolva logits diferentes quando
`use_jacobian=True` e `False`. Verificar que a função usa `positions=[-1]`,
texto bruto, `max_seq_len=512`, registra rank por camada e não chama
`model.generate`.

- [ ] **Passo 2: Confirmar a falha**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_readout.py -q
```

- [ ] **Passo 3: Implementar o algoritmo**

Usar as assinaturas:

```python
def rank_one_based(logits: torch.Tensor, token_id: int) -> int:
    score = logits[int(token_id)]
    return 1 + int((logits > score).sum().item())


def aggregate_pass_at_k(
    records: list[dict[str, object]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float | int]:
    totals = {k: 0.0 for k in ks}
    n_intermediates = 0
    for record in records:
        intermediates = record["intermediates"]
        n_intermediates += len(intermediates)
        for k in ks:
            totals[k] += (
                sum(item["best_rank"] <= k for item in intermediates)
                / len(intermediates)
            )
    n_items = len(records)
    return {
        "n_items": n_items,
        "n_intermediates": n_intermediates,
        **{
            f"pass@{k}": totals[k] / n_items if n_items else 0.0
            for k in ks
        },
    }
```

`evaluate_direct_readout()` deve executar `dictionary.lens.apply()` uma vez por
método e prompt, calcular ranks de todos os intermediários em cada camada e
produzir quatro agregados: J-lens/band, logit-lens/band,
J-lens/todas-as-camadas e logit-lens/todas-as-camadas. A assinatura aceitará
`detail_names: frozenset[str]` e `top: int`; somente os nomes em
`frozenset(PANEL_NAMES)` guardarão os dez tokens líderes por camada. O
subconjunto single-token deve filtrar intermediários pela auditoria, sem alterar
o conjunto canônico.

- [ ] **Passo 4: Rodar testes e registrar**

```powershell
uv run pytest tests/test_observation_readout.py tests/test_observation_data.py -q
cd ../..
git add projects/jspace_llama/src/jspace_llama/observation_readout.py projects/jspace_llama/tests/test_observation_readout.py
git commit -m "feat(jspace): score canonical J-lens readout"
```

---

### Task 4: Ler o painel qualitativo por camada

**Arquivos:**
- Criar: `projects/jspace_llama/src/jspace_llama/observation_panel.py`
- Criar: `projects/jspace_llama/tests/test_observation_panel.py`

**Interfaces:**
- Consome: `PANEL_NAMES` de `observation_data.py`, os 12 registros já
  avaliados, seus prompts, ativações finais, `dictionary`, tokenizer, band e
  `k=25`.
- Produz:
  `read_qualitative_panel(...) -> dict[str, object]` com top 10 da J-lens,
  top 10 da logit lens e resultados de pursuit separados.

- [ ] **Passo 1: Escrever testes de seleção e ciclo de memória**

Importar e conferir no teste:

```python
from jspace_llama.observation_data import PANEL_NAMES
```

Os testes devem comprovar:

```python
def test_panel_is_fixed_and_complete():
    selected = select_panel(fake_93_items())
    assert tuple(item.name for item in selected) == PANEL_NAMES


def test_dictionary_is_materialized_once_and_evicted_per_layer():
    result = read_qualitative_panel(
        model=fake_model,
        dictionary=fake_dictionary,
        items=fake_panel,
        direct_records=fake_direct_records,
        layers=(12, 13),
        k=25,
        top=10,
        activation_reader=fake_activation_reader,
        pursuit_fn=fake_pursuit,
    )
    assert fake_dictionary.unit_calls == [12, 13]
    assert fake_dictionary.evicted == [12, 13]
    assert len(result["items"]) == 12
```

Também verificar que ativações são mantidas em CPU, resultados são JSON-ready e
um erro em `gradient_pursuit` ainda remove a camada do dicionário.

- [ ] **Passo 2: Confirmar a falha**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_panel.py -q
```

- [ ] **Passo 3: Implementar o leitor limitado em memória**

O fluxo deve ser:

```python
@torch.no_grad()
def read_qualitative_panel(
    model,
    dictionary,
    items,
    direct_records,
    *,
    layers,
    k=25,
    top=10,
    activation_reader=None,
    pursuit_fn=None,
):
    selected = select_panel(items)
    activations = (activation_reader or read_final_activations)(
        model, selected, layers
    )
    output = index_direct_top_tokens(direct_records, selected, layers, top)
    for layer in layers:
        try:
            vectors = dictionary.unit(layer)
            for item in selected:
                hidden = activations[item.name].pop(layer).to(vectors.device)
                pursuit = (pursuit_fn or gradient_pursuit)(
                    hidden, vectors, k=k
                )
                append_pursuit_slot(output, item, layer, pursuit, dictionary)
        finally:
            dictionary.evict(layer)
    return output
```

Não reutilizar `_concept_summary()` nem o campo `spider` do readout causal. Os
tokens diretos vêm dos logits já calculados; o pursuit deve ordenar
coeficientes e registrar `token_id`, texto decodificado, rank, coeficiente e
`variance_explained`.

- [ ] **Passo 4: Rodar testes e registrar**

```powershell
uv run pytest tests/test_observation_panel.py tests/test_observation_readout.py -q
cd ../..
git add projects/jspace_llama/src/jspace_llama/observation_panel.py projects/jspace_llama/tests/test_observation_panel.py
git commit -m "feat(jspace): add qualitative workspace panel"
```

---

### Task 5: Construir pipeline, artefatos, schemas e figuras

**Arquivos:**
- Criar: `projects/jspace_llama/src/jspace_llama/observation_artifacts.py`
- Criar: `projects/jspace_llama/src/jspace_llama/observation_figures.py`
- Criar: `projects/jspace_llama/src/jspace_llama/observation_pipeline.py`
- Criar:
  `projects/jspace_llama/schemas/jspace_observation_manifest.schema.json`
- Criar: `projects/jspace_llama/schemas/jspace_observation.schema.json`
- Criar: `projects/jspace_llama/scripts/validate_observation.py`
- Criar: `projects/jspace_llama/tests/test_observation_pipeline.py`
- Criar: `projects/jspace_llama/tests/test_observation_figures.py`
- Modificar: `projects/jspace_llama/src/jspace_llama/pipeline.py`

**Interfaces:**
- Consome: dataset validado, carregamento existente de modelo/lente, readout
  direto e painel qualitativo.
- Produz:
  `run_observation_pipeline(project_root: Path) -> dict[str, object]`, dois JSONs
  compactos, um JSON bruto local e seis figuras estáticas.
- Produz em `pipeline.py`:
  `load_pinned_runtime(config, provenance) -> LoadedRuntime`, usado pelos dois
  pipelines sem alterar o contrato público causal.

- [ ] **Passo 1: Escrever testes do contrato do pipeline**

O fake de execução deve produzir 93 registros e 12 painéis. Testar:

```python
def test_observation_success_depends_on_completeness_not_hits(tmp_path):
    outcome = run_observation_pipeline(
        project_root=tmp_path,
        execute=lambda: complete_payload(all_pass_at_k_zero=True),
    )
    assert outcome["status"] == "success"
    assert outcome["metrics"]["jlens"]["band"]["pass@1"] == 0.0


def test_observation_pipeline_never_replaces_causal_results(tmp_path):
    causal = tmp_path / "results" / "final_evaluation.json"
    causal.parent.mkdir(parents=True)
    causal.write_text('{"preserve":true}', encoding="utf-8")
    run_observation_pipeline(
        project_root=tmp_path,
        execute=lambda: complete_payload(),
    )
    assert causal.read_text(encoding="utf-8") == '{"preserve":true}'
```

Testar também escrita atômica, hash do bruto no manifest, execução incompleta,
erro esperado, erro de programação relançado e validação dos dois schemas.

- [ ] **Passo 2: Escrever testes das matrizes e figuras**

Com backend `Agg`, exigir:

```python
def test_figure_matrices_are_deterministic():
    matrices = build_figure_matrices(compact_result())
    assert matrices["pass_at_k"]["jlens_band"] == [0.2, 0.4, 0.6]
    assert matrices["rank_heatmap"]["shape"] == [93, 17]


def test_all_figure_families_write_png_and_svg(tmp_path):
    paths = write_observation_figures(compact_result(), tmp_path)
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert len(paths) == 6
    assert all(path.stat().st_size > 0 for path in paths)
```

- [ ] **Passo 3: Confirmar as falhas**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_pipeline.py tests/test_observation_figures.py -q
```

- [ ] **Passo 4: Implementar artefatos e schemas**

O manifest deve conter:

```python
{
    "schema_version": 1,
    "artifact": "jspace_observation_manifest",
    "run_id": run_id,
    "status": "success",
    "started_at": started_at,
    "ended_at": ended_at,
    "provenance": provenance,
    "dataset": {
        "upstream_repository": "anthropics/jacobian-lens",
        "upstream_revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
        "sha256": DATASET_SHA256,
        "item_count": 93,
    },
    "raw_artifact_sha256": raw_sha256,
    "reason": None,
}
```

O resultado compacto deve conter `metrics`, `tokenization_summary`,
`per_item_best_ranks`, `panel`, `layers`, `panel_names` e
`manifest_sha256`. Não incluir ativações, logits completos, caminhos ou
continuações. Os schemas devem fechar todos os objetos com
`additionalProperties: false`.

Criar `validate_observation.py` separadamente do validador causal. Ele deve
recalcular hashes, contagens, `pass@k`, seleção dos 12 nomes, estado e
proveniência. Tokens e token IDs são permitidos neste artefato; credenciais,
caches, caminhos absolutos, tensores e valores não finitos continuam proibidos.

- [ ] **Passo 5: Implementar figuras determinísticas**

Usar:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Fixar tamanho, DPI, ordem de métodos, ordem dos itens e mapa de cores. Gerar:

- barras de `pass@1/5/10` para J-lens/logit lens e band/todas as camadas;
- heatmap de rank por item e camada, com escala logarítmica e ranks ausentes
  limitados ao tamanho do vocabulário;
- painel de 12 itens com intermediário esperado, melhor camada/rank e principais
  tokens diretos e de pursuit.

Após salvar cada SVG temporário, remover a declaração `DOCTYPE` emitida pelo
Matplotlib, fazer parse XML e confirmar que a raiz é `svg`. Salvar primeiro em
arquivos temporários e substituir os destinos apenas após os seis arquivos
serem gerados.

- [ ] **Passo 6: Implementar o pipeline real**

Extrair do começo de `execute_real_experiment()` o carregamento compartilhado
de modelo, lente, revisões e integridade para
`load_pinned_runtime(config, provenance) -> LoadedRuntime` em `pipeline.py`.
`LoadedRuntime` será uma dataclass com `bundle`, `model_revision`,
`lens_revision`, `lens_sha256`, `lens_size` e `hardware`. Manter o retorno e os
testes do pipeline causal. O pipeline observacional deve:

1. exigir CUDA;
2. carregar o dataset local validado;
3. baixar e carregar os mesmos assets da execução causal;
4. auditar tokenização;
5. calcular rankings diretos;
6. calcular o painel de pursuit;
7. gravar bruto, compacto, manifest e figuras;
8. retornar o resultado compacto.

Nos testes, substituir `model.generate` por uma função que lança imediatamente
e verificar que ela nunca é chamada. O módulo observacional não deve importar
`jspace_llama.interventions` nem construir solicitações de geração.

- [ ] **Passo 7: Rodar testes e registrar**

```powershell
uv run pytest tests/test_observation_pipeline.py tests/test_observation_figures.py tests/test_artifact_pipeline.py tests/test_final_review_fixes.py -q
cd ../..
git add projects/jspace_llama/src/jspace_llama/observation_artifacts.py projects/jspace_llama/src/jspace_llama/observation_figures.py projects/jspace_llama/src/jspace_llama/observation_pipeline.py projects/jspace_llama/src/jspace_llama/pipeline.py projects/jspace_llama/schemas/jspace_observation_manifest.schema.json projects/jspace_llama/schemas/jspace_observation.schema.json projects/jspace_llama/scripts/validate_observation.py projects/jspace_llama/tests/test_observation_pipeline.py projects/jspace_llama/tests/test_observation_figures.py
git commit -m "feat(jspace): build observational atlas pipeline"
```

---

### Task 6: Integrar notebook e transporte seguro do Colab

**Arquivos:**
- Criar:
  `projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb`
- Modificar: `projects/jspace_llama/scripts/colab_pack_results.py`
- Modificar: `tools/colab/run_jspace.py`
- Modificar: `projects/jspace_llama/tests/test_colab_orchestrator.py`
- Criar: `projects/jspace_llama/tests/test_observation_notebook.py`
- Modificar: `.gitignore`

**Interfaces:**
- Consome: `run_observation_pipeline(project_root=...)`.
- Produz: execução fina no Colab e
  `fetch --result-set observation` com allowlist independente da causal.

- [ ] **Passo 1: Escrever testes de notebook e perfis de resultados**

O notebook deve ser validado com:

```python
def test_observation_notebook_is_thin_and_output_free():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "run_observation_pipeline" in code
    assert "gradient_pursuit" not in code
    assert "lens.apply" not in code
    assert "generate(" not in code
    assert "LensSwap" not in code
    assert "HF_TOKEN" not in code
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
```

Em `test_colab_orchestrator.py`, exigir dois perfis:

```python
assert runner.result_allowlist("causal") == {
    "results/execution_manifest.json",
    "results/final_evaluation.json",
}
assert runner.result_allowlist("observation") == {
    "results/jspace_observation_manifest.json",
    "results/jspace_observation.json",
    "results/jspace_observation_pass_at_k.png",
    "results/jspace_observation_pass_at_k.svg",
    "results/jspace_observation_rank_heatmap.png",
    "results/jspace_observation_rank_heatmap.svg",
    "results/jspace_observation_panel.png",
    "results/jspace_observation_panel.svg",
}
```

Testar rejeição de membros extras, ausência de membros, PNG sem assinatura,
SVG com `script`/`DOCTYPE`, arquivo grande e perfil desconhecido.

- [ ] **Passo 2: Confirmar as falhas**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_notebook.py tests/test_colab_orchestrator.py -q
```

- [ ] **Passo 3: Criar o notebook fino**

O notebook terá células para:

```python
from pathlib import Path
from jspace_llama.observation_pipeline import run_observation_pipeline
```

```python
project_root = Path.cwd()
while not (project_root / "pyproject.toml").is_file():
    if project_root.parent == project_root:
        raise RuntimeError("jspace_llama project root not found")
    project_root = project_root.parent
```

```python
outcome = run_observation_pipeline(project_root=project_root)
outcome
```

As células Markdown devem explicar que o notebook não gera texto nem modifica
ativações.

- [ ] **Passo 4: Adicionar perfis ao pack/fetch**

Em ambos os scripts, definir `RESULT_SETS` com os dois conjuntos exatos do
teste. `colab_pack_results.py` lerá `JSPACE_RESULT_SET`, validará somente o
perfil selecionado e permitirá no máximo 1 MiB para manifests e artefatos
causais, 5 MiB para `jspace_observation.json` e 5 MiB por figura. O arquivo
compactado completo terá limite de 40 MiB.

`run_jspace.py fetch --result-set observation` deve enviar o perfil ao pack,
baixar o arquivo, validar:

- JSON com parser estrito;
- PNG pela assinatura, chunk `IHDR` e dimensões positivas limitadas a 10.000;
- SVG como UTF-8/XML, raiz `svg`, sem `DOCTYPE`, `ENTITY`, `script`,
  `foreignObject` ou referências externas.

Somente após validar todos os membros os arquivos podem substituir destinos
locais.

- [ ] **Passo 5: Permitir figuras canônicas no Git**

Acrescentar à raiz `.gitignore`:

```gitignore
!projects/jspace_llama/results/*.png
!projects/jspace_llama/results/*.svg
```

- [ ] **Passo 6: Rodar testes e registrar**

```powershell
cd projects/jspace_llama
uv run pytest tests/test_observation_notebook.py tests/test_colab_orchestrator.py -q
cd ../..
git add .gitignore projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb projects/jspace_llama/scripts/colab_pack_results.py projects/jspace_llama/tests/test_colab_orchestrator.py projects/jspace_llama/tests/test_observation_notebook.py tools/colab/run_jspace.py
git commit -m "feat(jspace): run observational atlas on Colab"
```

---

### Task 7: Documentar e verificar a implementação local

**Arquivos:**
- Modificar: `projects/jspace_llama/README.md`
- Modificar: `projects/jspace_llama/PROCESSO.md`
- Modificar: `tools/colab/README.md`
- Modificar: `docs/replicacoes/README.md`

**Interfaces:**
- Consome: pipeline e comandos concluídos.
- Produz: documentação que distingue tentativa causal anterior, atlas
  observacional e futura confirmação.

- [ ] **Passo 1: Documentar o novo objetivo**

Explicar no README e no PROCESSO:

- origem Anthropic e licença dos 93 prompts;
- por que o prompt bruto substitui o chat template nesta avaliação;
- diferença entre J-lens direta, logit lens e `gradient_pursuit`;
- `target` não é fornecido ao modelo;
- ausência de geração e edição causal;
- distinção entre sucesso operacional e `pass@k`.

- [ ] **Passo 2: Documentar os comandos exatos**

Adicionar:

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb --timeout 7200
python tools/colab/run_jspace.py fetch --result-set observation
python tools/colab/run_jspace.py stop
```

Adicionar o validador:

```powershell
cd projects/jspace_llama
uv run python scripts/validate_observation.py
```

- [ ] **Passo 3: Rodar a suíte completa**

```powershell
cd projects/jspace_llama
uv run pytest -q
uv run python scripts/validate_results.py
```

Antes da execução remota, o validador observacional deve informar ausência
explícita de artefatos, não validar arquivos causais como se fossem o atlas.

- [ ] **Passo 4: Verificar qualidade e escopo do diff**

```powershell
cd ../..
git diff --check
git status --short
```

Esperado: nenhum erro de whitespace; alterações não relacionadas de IOI e de
outros documentos continuam fora do stage.

- [ ] **Passo 5: Registrar a documentação**

```powershell
git add projects/jspace_llama/README.md projects/jspace_llama/PROCESSO.md tools/colab/README.md docs/replicacoes/README.md
git commit -m "docs(jspace): explain observational atlas"
```

---

### Task 8: Executar na L4, coletar e interpretar o atlas

**Arquivos:**
- Criar pela execução:
  `projects/jspace_llama/results/jspace_observation_manifest.json`
- Criar pela execução:
  `projects/jspace_llama/results/jspace_observation.json`
- Criar pela execução:
  `projects/jspace_llama/results/jspace_observation_pass_at_k.{png,svg}`
- Criar pela execução:
  `projects/jspace_llama/results/jspace_observation_rank_heatmap.{png,svg}`
- Criar pela execução:
  `projects/jspace_llama/results/jspace_observation_panel.{png,svg}`
- Modificar após validação: `projects/jspace_llama/README.md`
- Modificar após validação: `projects/jspace_llama/PROCESSO.md`
- Modificar após validação: `docs/replicacoes/README.md`

**Interfaces:**
- Consome: implementação local limpa e autenticação Hugging Face já
  configurada.
- Produz: execução canônica observacional, figuras e interpretação limitada
  pelas métricas.

- [ ] **Passo 1: Confirmar a sessão e enviar o projeto**

```powershell
python tools/colab/run_jspace.py probe
python tools/colab/run_jspace.py start --gpu L4
python tools/colab/run_jspace.py sync
python tools/colab/run_jspace.py bootstrap
python tools/colab/run_jspace.py auth-hf
```

Esperado: runtime L4, bootstrap concluído e token HF transferido sem aparecer
nos logs.

- [ ] **Passo 2: Executar o notebook observacional**

```powershell
python tools/colab/run_jspace.py execute --notebook projects/jspace_llama/notebooks/02_observe_multihop_jspace.ipynb --timeout 7200
```

Esperado: 93 prompts avaliados, 12 painéis concluídos, nenhuma chamada de
geração/intervenção e término dentro do limite.

- [ ] **Passo 3: Buscar e validar**

```powershell
python tools/colab/run_jspace.py fetch --result-set observation
cd projects/jspace_llama
uv run python scripts/validate_observation.py
```

Esperado: `PASS`, hashes e contagens coerentes, seis figuras válidas e arquivos
causais anteriores byte a byte inalterados.

- [ ] **Passo 4: Interpretar sem selecionar causalmente**

Registrar nos documentos:

- `pass@1/5/10` da J-lens e logit lens no band;
- resultados secundários em todas as camadas;
- tamanho e scores do subconjunto single-token;
- melhor rank/camada de `spider` no caso `spider-legs`;
- exemplos do painel em que o intermediário apareceu ou não apareceu;
- diferenças observadas entre ranking direto e pursuit;
- limitações de tokenizer, modelo, lente e seleção exploratória.

Não escolher ainda conceito, direção, alpha ou prompts para intervenção. Uma
hipótese causal exigirá nova especificação e prompts não usados neste atlas.

- [ ] **Passo 5: Encerrar a GPU mesmo em caso de falha**

```powershell
cd ../..
python tools/colab/run_jspace.py stop
python tools/colab/run_jspace.py probe
```

Esperado: ausência confirmada da sessão GPU.

- [ ] **Passo 6: Rodar verificação final e registrar evidências**

```powershell
cd projects/jspace_llama
uv run pytest -q
uv run python scripts/validate_results.py
uv run python scripts/validate_observation.py
cd ../..
git diff --check
git add projects/jspace_llama/results/jspace_observation_manifest.json projects/jspace_llama/results/jspace_observation.json projects/jspace_llama/results/jspace_observation_pass_at_k.png projects/jspace_llama/results/jspace_observation_pass_at_k.svg projects/jspace_llama/results/jspace_observation_rank_heatmap.png projects/jspace_llama/results/jspace_observation_rank_heatmap.svg projects/jspace_llama/results/jspace_observation_panel.png projects/jspace_llama/results/jspace_observation_panel.svg projects/jspace_llama/README.md projects/jspace_llama/PROCESSO.md docs/replicacoes/README.md
git commit -m "docs(jspace): record observational atlas results"
```

Esperado: testes e os dois validadores aprovados; o commit contém somente
artefatos compactos, figuras e documentação observacional.
