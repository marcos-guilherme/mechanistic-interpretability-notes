"""Atualiza o retrato automático do painel de experimentos."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote


START_MARKER = "<!-- BEGIN GENERATED STATUS -->"
END_MARKER = "<!-- END GENERATED STATUS -->"


class TrackingError(RuntimeError):
    """Erro esperado ao ler ou sincronizar o acompanhamento."""


class TrackingOutOfDate(TrackingError):
    """O bloco gerado não corresponde aos resultados atuais."""


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path, root: Path) -> Mapping[str, Any]:
    label = _display_path(path, root)
    if not path.is_file():
        raise TrackingError(f"arquivo canônico ausente: {label}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrackingError(
            f"JSON inválido em {label} (linha {exc.lineno}, coluna {exc.colno})"
        ) from exc
    except OSError as exc:
        raise TrackingError(f"não foi possível ler {label}: {exc}") from exc

    if not isinstance(value, dict):
        raise TrackingError(f"conteúdo inválido em {label}: esperado objeto JSON")
    return value


def _required(
    value: Mapping[str, Any],
    keys: Sequence[str],
    source: Path,
    root: Path,
) -> Any:
    current: Any = value
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            field = ".".join(traversed)
            label = _display_path(source, root)
            raise TrackingError(f"campo obrigatório ausente em {label}: {field}")
        current = current[key]
    return current


def _invalid_type(
    source: Path,
    root: Path,
    keys: Sequence[str],
    expected: str,
) -> TrackingError:
    field = ".".join(keys)
    label = _display_path(source, root)
    return TrackingError(f"campo inválido em {label}: {field}; esperado {expected}")


def _text(
    value: Any,
    keys: Sequence[str],
    source: Path,
    root: Path,
) -> str:
    if not isinstance(value, str):
        raise _invalid_type(source, root, keys, "texto")
    return value


def _finite_number(
    value: Any,
    keys: Sequence[str],
    source: Path,
    root: Path,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_type(source, root, keys, "número finito")
    number = float(value)
    if not math.isfinite(number):
        raise _invalid_type(source, root, keys, "número finito")
    return number


def _integer(
    value: Any,
    keys: Sequence[str],
    source: Path,
    root: Path,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_type(source, root, keys, "inteiro")
    return value


def _collect_qwen_refusal_pilot(
    root: Path,
    refusal_results: Path,
) -> dict[str, Any] | None:
  """Lê o piloto Qwen congelado, se os JSONs compactos estiverem presentes."""

  qwen_dir = refusal_results / "qwen25_coder_7b"
  pilot_path = qwen_dir / "pilot_evaluation.json"
  evaluation_path = qwen_dir / "final_evaluation.json"
  if not pilot_path.is_file() or not evaluation_path.is_file():
    return None

  pilot = _load_json(pilot_path, root)
  qwen_evaluation = _load_json(evaluation_path, root)

  selected = _required(
    qwen_evaluation,
    ("results", "selected"),
    evaluation_path,
    root,
  )
  if not isinstance(selected, dict):
    raise _invalid_type(
      evaluation_path,
      root,
      ("results", "selected"),
      "objeto",
    )

  return {
    "model": _text(
      _required(
        qwen_evaluation,
        ("config", "model_name"),
        evaluation_path,
        root,
      ),
      ("config", "model_name"),
      evaluation_path,
      root,
    ),
    "model_revision": _text(
      _required(
        qwen_evaluation,
        ("config", "model_revision"),
        evaluation_path,
        root,
      ),
      ("config", "model_revision"),
      evaluation_path,
      root,
    ),
    "run_id": _text(
      _required(pilot, ("run_id",), pilot_path, root),
      ("run_id",),
      pilot_path,
      root,
    ),
    "implementation_commit": _text(
      _required(pilot, ("implementation_commit",), pilot_path, root),
      ("implementation_commit",),
      pilot_path,
      root,
    ),
    "operational_success": bool(
      _required(pilot, ("operational_success",), pilot_path, root)
    ),
    "scientific_status": _text(
      _required(pilot, ("status",), pilot_path, root),
      ("status",),
      pilot_path,
      root,
    ),
    "selected": {
      "layer": _integer(
        _required(selected, ("layer",), evaluation_path, root),
        ("results", "selected", "layer"),
        evaluation_path,
        root,
      ),
      "position_offset": _integer(
        _required(selected, ("position_offset",), evaluation_path, root),
        ("results", "selected", "position_offset"),
        evaluation_path,
        root,
      ),
    },
    "harmful_refusal_rates": {
      "baseline": _finite_number(
        _required(
          qwen_evaluation,
          ("results", "baseline", "harmful_refusal_rate"),
          evaluation_path,
          root,
        ),
        ("results", "baseline", "harmful_refusal_rate"),
        evaluation_path,
        root,
      ),
      "ablation": _finite_number(
        _required(
          qwen_evaluation,
          ("results", "ablation", "harmful_refusal_rate"),
          evaluation_path,
          root,
        ),
        ("results", "ablation", "harmful_refusal_rate"),
        evaluation_path,
        root,
      ),
      "random_control": _finite_number(
        _required(
          qwen_evaluation,
          ("results", "random_control", "harmful_refusal_rate"),
          evaluation_path,
          root,
        ),
        ("results", "random_control", "harmful_refusal_rate"),
        evaluation_path,
        root,
      ),
    },
    "harmless_refusal_rates": {
      "baseline": _finite_number(
        _required(
          qwen_evaluation,
          ("results", "baseline", "harmless_refusal_rate"),
          evaluation_path,
          root,
        ),
        ("results", "baseline", "harmless_refusal_rate"),
        evaluation_path,
        root,
      ),
      "addition": _finite_number(
        _required(
          qwen_evaluation,
          ("results", "addition", "harmless_refusal_rate"),
          evaluation_path,
          root,
        ),
        ("results", "addition", "harmless_refusal_rate"),
        evaluation_path,
        root,
      ),
    },
    "ablation_harmless_kl": _finite_number(
      _required(
        qwen_evaluation,
        ("results", "ablation", "harmless_kl"),
        evaluation_path,
        root,
      ),
      ("results", "ablation", "harmless_kl"),
      evaluation_path,
      root,
    ),
  }


def _string_list(
    value: Any,
    keys: Sequence[str],
    source: Path,
    root: Path,
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise _invalid_type(source, root, keys, "lista de textos")
    return value


def collect_snapshot(repo_root: Path) -> dict[str, Any]:
    """Lê os artefatos canônicos e devolve o retrato usado pelo painel."""

    root = Path(repo_root)
    jspace_results = root / "projects" / "jspace_llama" / "results"
    refusal_results = root / "projects" / "refusal_direction" / "results"

    jspace_manifest_path = jspace_results / "execution_manifest.json"
    jspace_evaluation_path = jspace_results / "final_evaluation.json"
    observation_manifest_path = jspace_results / "jspace_observation_manifest.json"
    observation_path = jspace_results / "jspace_observation.json"
    refusal_manifest_path = refusal_results / "execution_manifest.json"
    refusal_evaluation_path = refusal_results / "final_evaluation.json"

    jspace_manifest = _load_json(jspace_manifest_path, root)
    jspace_evaluation = _load_json(jspace_evaluation_path, root)
    observation_manifest = _load_json(observation_manifest_path, root)
    observation = _load_json(observation_path, root)
    refusal_manifest = _load_json(refusal_manifest_path, root)
    refusal_evaluation = _load_json(refusal_evaluation_path, root)

    jspace_causal = {
        "model": _text(
            _required(
                jspace_manifest,
                ("provenance", "model", "id"),
                jspace_manifest_path,
                root,
            ),
            ("provenance", "model", "id"),
            jspace_manifest_path,
            root,
        ),
        "run_id": _text(
            _required(jspace_manifest, ("run_id",), jspace_manifest_path, root),
            ("run_id",),
            jspace_manifest_path,
            root,
        ),
        "operational_status": _text(
            _required(
                jspace_manifest, ("terminal_status",), jspace_manifest_path, root
            ),
            ("terminal_status",),
            jspace_manifest_path,
            root,
        ),
        "scientific_status": _text(
            _required(
                jspace_evaluation, ("status",), jspace_evaluation_path, root
            ),
            ("status",),
            jspace_evaluation_path,
            root,
        ),
        "reason": _text(
            _required(
                jspace_evaluation,
                ("reason", "code"),
                jspace_evaluation_path,
                root,
            ),
            ("reason", "code"),
            jspace_evaluation_path,
            root,
        ),
    }

    def observation_band(lens: str) -> list[float]:
        return [
            _finite_number(
                _required(
                    observation,
                    ("metrics", lens, "band", metric),
                    observation_path,
                    root,
                ),
                ("metrics", lens, "band", metric),
                observation_path,
                root,
            )
            for metric in ("pass@1", "pass@5", "pass@10")
        ]

    jspace_observation = {
        "run_id": _text(
            _required(
                observation_manifest,
                ("run_id",),
                observation_manifest_path,
                root,
            ),
            ("run_id",),
            observation_manifest_path,
            root,
        ),
        "operational_status": _text(
            _required(
                observation_manifest,
                ("status",),
                observation_manifest_path,
                root,
            ),
            ("status",),
            observation_manifest_path,
            root,
        ),
        "item_count": _integer(
            _required(
                observation_manifest,
                ("dataset", "item_count"),
                observation_manifest_path,
                root,
            ),
            ("dataset", "item_count"),
            observation_manifest_path,
            root,
        ),
        "jlens_band": observation_band("jlens"),
        "logit_lens_band": observation_band("logit_lens"),
    }

    selected = _required(
        refusal_evaluation,
        ("results", "selected"),
        refusal_evaluation_path,
        root,
    )
    if not isinstance(selected, dict):
        raise _invalid_type(
            refusal_evaluation_path,
            root,
            ("results", "selected"),
            "objeto",
        )

    refusal_direction = {
        "model": _text(
            _required(
                refusal_manifest,
                ("config", "model_name"),
                refusal_manifest_path,
                root,
            ),
            ("config", "model_name"),
            refusal_manifest_path,
            root,
        ),
        "completed_stages": _string_list(
            _required(
                refusal_manifest,
                ("results", "completed_stages"),
                refusal_manifest_path,
                root,
            ),
            ("results", "completed_stages"),
            refusal_manifest_path,
            root,
        ),
        "selected": {
            "layer": _integer(
                _required(selected, ("layer",), refusal_evaluation_path, root),
                ("results", "selected", "layer"),
                refusal_evaluation_path,
                root,
            ),
            "position_offset": _integer(
                _required(
                    selected, ("position_offset",), refusal_evaluation_path, root
                ),
                ("results", "selected", "position_offset"),
                refusal_evaluation_path,
                root,
            ),
        },
        "harmful_refusal_rates": {
            "baseline": _finite_number(
                _required(
                    refusal_evaluation,
                    ("results", "baseline", "harmful_refusal_rate"),
                    refusal_evaluation_path,
                    root,
                ),
                ("results", "baseline", "harmful_refusal_rate"),
                refusal_evaluation_path,
                root,
            ),
            "ablation": _finite_number(
                _required(
                    refusal_evaluation,
                    ("results", "ablation", "harmful_refusal_rate"),
                    refusal_evaluation_path,
                    root,
                ),
                ("results", "ablation", "harmful_refusal_rate"),
                refusal_evaluation_path,
                root,
            ),
            "random_control": _finite_number(
                _required(
                    refusal_evaluation,
                    ("results", "random_control", "harmful_refusal_rate"),
                    refusal_evaluation_path,
                    root,
                ),
                ("results", "random_control", "harmful_refusal_rate"),
                refusal_evaluation_path,
                root,
            ),
        },
        "addition_harmless_refusal_rate": _finite_number(
            _required(
                refusal_evaluation,
                ("results", "addition", "harmless_refusal_rate"),
                refusal_evaluation_path,
                root,
            ),
            ("results", "addition", "harmless_refusal_rate"),
            refusal_evaluation_path,
            root,
        ),
        "ablation_harmless_kl": _finite_number(
            _required(
                refusal_evaluation,
                ("results", "ablation", "harmless_kl"),
                refusal_evaluation_path,
                root,
            ),
            ("results", "ablation", "harmless_kl"),
            refusal_evaluation_path,
            root,
        ),
    }

    return {
        "jspace_causal": jspace_causal,
        "jspace_observation": jspace_observation,
        "refusal_direction": refusal_direction,
        "qwen_refusal_pilot": _collect_qwen_refusal_pilot(root, refusal_results),
    }


def _decimal(value: float, places: int = 6) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def _percentage(value: float) -> str:
    percentage = value * 100
    places = 0 if percentage.is_integer() else 2
    return _decimal(percentage, places) + "%"


def _pass_band(values: Sequence[float]) -> str:
    return " / ".join(_decimal(value, 6) for value in values)


def render_status(snapshot: Mapping[str, Any]) -> str:
    """Produz o bloco Markdown compacto em português."""

    causal = snapshot["jspace_causal"]
    observation = snapshot["jspace_observation"]
    refusal = snapshot["refusal_direction"]

    scientific = {
        "inconclusive": "inconclusiva",
    }.get(causal["scientific_status"], causal["scientific_status"])
    stages = {
        "baseline": "baseline",
        "extract": "extração",
        "intervene": "intervenção",
        "evaluate": "avaliação",
    }
    completed_stages = ", ".join(
        stages.get(stage, stage) for stage in refusal["completed_stages"]
    )
    rates = refusal["harmful_refusal_rates"]
    selected = refusal["selected"]
    stage_count = len(refusal["completed_stages"])
    stage_label = "estágio completo" if stage_count == 1 else "estágios completos"

    def canonical_rate(actual: float, expected: float) -> bool:
        return math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    supports_local_causality = (
        all(
            stage in refusal["completed_stages"]
            for stage in ("baseline", "extract", "intervene", "evaluate")
        )
        and canonical_rate(rates["baseline"], 1.0)
        and canonical_rate(rates["ablation"], 0.0)
        and canonical_rate(rates["random_control"], 1.0)
        and canonical_rate(refusal["addition_harmless_refusal_rate"], 0.9375)
        and refusal["ablation_harmless_kl"] <= 0.1
    )
    refusal_conclusion = (
        "Conclusão científica: evidência **causal local**."
        if supports_local_causality
        else "Conclusão científica: o resultado requer **reavaliação**."
    )

    qwen_block = ""
    qwen = snapshot.get("qwen_refusal_pilot")
    if qwen is not None:
        qwen_rates = qwen["harmful_refusal_rates"]
        harmless_rates = qwen["harmless_refusal_rates"]
        qwen_selected = qwen["selected"]
        operational = "concluída" if qwen["operational_success"] else "incompleta"
        qwen_block = (
            "**Direção de Recusa (piloto Qwen).** "
            f"Modelo `{qwen['model']}` na revisão "
            f"`{qwen['model_revision']}`; execução "
            f"`{qwen['run_id']}`. Operação {operational}; "
            f"Conclusão científica: **falha no gate causal** "
            f"(`{qwen['scientific_status']}`): manipulação forte, "
            "mas KL inofensiva após ablação acima do limite 0,10. "
            f"Direção na camada {qwen_selected['layer']}, posição "
            f"{qwen_selected['position_offset']}. Recusa nociva (teste, "
            f"splits sem filtro): baseline {_percentage(qwen_rates['baseline'])}, "
            f"ablação {_percentage(qwen_rates['ablation'])}, controle "
            f"{_percentage(qwen_rates['random_control'])}; recusa inofensiva: "
            f"baseline {_percentage(harmless_rates['baseline'])}, adição "
            f"{_percentage(harmless_rates['addition'])}; KL inofensiva após "
            f"ablação `{_decimal(qwen['ablation_harmless_kl'], 6)}`. "
            "[Piloto](../projects/refusal_direction/results/qwen25_coder_7b/"
            "pilot_evaluation.json) · [avaliação](../projects/refusal_direction/"
            "results/qwen25_coder_7b/final_evaluation.json).\n"
        )

    return (
        "### Retrato automático dos resultados\n\n"
        "**J-space causal.** "
        f"Modelo `{causal['model']}`; execução `{causal['run_id']}`. "
        f"Estado operacional: `{causal['operational_status']}`. "
        f"Conclusão científica: **{scientific}** "
        f"(`{causal['reason']}`). "
        "[Manifesto](../projects/jspace_llama/results/execution_manifest.json) · "
        "[avaliação](../projects/jspace_llama/results/final_evaluation.json).\n\n"
        "**J-space atlas.** "
        f"Estado operacional: `{observation['operational_status']}` na execução "
        f"`{observation['run_id']}`, com {observation['item_count']} itens. "
        "Conclusão científica: retrato **observacional**, não evidência causal. "
        "Na faixa de camadas, pass@1/5/10: "
        f"J-lens `{_pass_band(observation['jlens_band'])}`; "
        f"logit lens `{_pass_band(observation['logit_lens_band'])}`. "
        "[Manifesto](../projects/jspace_llama/results/"
        "jspace_observation_manifest.json) · "
        "[resultados](../projects/jspace_llama/results/"
        "jspace_observation.json).\n\n"
        "**Direção de Recusa.** "
        f"Modelo `{refusal['model']}`; {stage_count} {stage_label} "
        f"({completed_stages}). {refusal_conclusion} "
        f"Direção selecionada na camada {selected['layer']}, posição "
        f"{selected['position_offset']}. Recusa nociva: baseline "
        f"{_percentage(rates['baseline'])}, ablação "
        f"{_percentage(rates['ablation'])}, controle aleatório "
        f"{_percentage(rates['random_control'])}; adição em pedidos inofensivos "
        f"{_percentage(refusal['addition_harmless_refusal_rate'])}; "
        f"KL inofensiva após ablação "
        f"`{_decimal(refusal['ablation_harmless_kl'], 6)}`. "
        "[Manifesto](../projects/refusal_direction/results/"
        "execution_manifest.json) · "
        "[intervenções](../projects/refusal_direction/results/"
        "03_interventions.json) · "
        "[avaliação](../projects/refusal_direction/results/"
        "final_evaluation.json).\n"
        f"{qwen_block}"
    )


def replace_generated_block(document: str, generated: str) -> str:
    """Substitui somente o interior do bloco delimitado pelos marcadores."""

    if document.count(START_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise TrackingError("o documento deve conter exatamente um marcador de cada tipo")

    start = document.index(START_MARKER)
    interior_start = start + len(START_MARKER)
    end = document.index(END_MARKER)
    if interior_start >= end:
        raise TrackingError("marcadores em ordem incorreta")

    newline = "\r\n" if "\r\n" in document else "\n"
    normalized = generated.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    normalized = normalized.replace("\n", newline)
    interior = newline + normalized + newline if normalized else newline
    return document[:interior_start] + interior + document[end:]


def synchronize(document_path: Path, generated: str, check: bool) -> bool:
    """Confere ou atualiza o bloco gerado; retorna se havia mudança."""

    path = Path(document_path)
    try:
        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")
    except FileNotFoundError as exc:
        raise TrackingError(f"documento ausente: {path}") from exc
    except UnicodeDecodeError as exc:
        raise TrackingError(f"documento não está em UTF-8: {path}") from exc
    except OSError as exc:
        raise TrackingError(f"não foi possível ler {path}: {exc}") from exc

    updated = replace_generated_block(original, generated)
    changed = updated != original
    if changed and check:
        raise TrackingOutOfDate(f"bloco gerado desatualizado em {path}")
    if changed:
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(updated.encode("utf-8"))
                temporary.flush()
                os.fsync(temporary.fileno())

            try:
                os.chmod(
                    temporary_path,
                    stat.S_IMODE(path.stat().st_mode),
                )
            except OSError:
                pass

            os.replace(temporary_path, path)
            temporary_path = None
        except OSError as exc:
            raise TrackingError(f"não foi possível escrever {path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except (FileNotFoundError, OSError):
                    pass
    return changed


_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _mask_markdown_code(text: str) -> str:
    """Oculta código cercado e inline, preservando posições e quebras."""

    masked = list(text)
    fence_char: str | None = None
    fence_size = 0
    offset = 0

    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if fence_char is None:
            match = re.match(r" {0,3}(`{3,}|~{3,})", content)
            if match is not None:
                fence_char = match.group(1)[0]
                fence_size = len(match.group(1))
                for index in range(offset, offset + len(content)):
                    masked[index] = " "
        else:
            closing = re.fullmatch(
                rf" {{0,3}}{re.escape(fence_char)}{{{fence_size},}}[ \t]*",
                content,
            )
            for index in range(offset, offset + len(content)):
                masked[index] = " "
            if closing is not None:
                fence_char = None
                fence_size = 0
        offset += len(line)

    position = 0
    masked_text = "".join(masked)
    while position < len(masked_text):
        if masked_text[position] != "`" or _is_escaped(masked_text, position):
            position += 1
            continue

        end_of_run = position
        while end_of_run < len(masked_text) and masked_text[end_of_run] == "`":
            end_of_run += 1
        run_size = end_of_run - position
        search = end_of_run
        closing_end: int | None = None
        while search < len(masked_text):
            candidate = masked_text.find("`" * run_size, search)
            if candidate < 0:
                break
            before_matches = candidate > 0 and masked_text[candidate - 1] == "`"
            after = candidate + run_size
            after_matches = after < len(masked_text) and masked_text[after] == "`"
            if not before_matches and not after_matches:
                closing_end = after
                break
            search = candidate + 1

        if closing_end is None:
            position = end_of_run
            continue
        for index in range(position, closing_end):
            if masked[index] not in "\r\n":
                masked[index] = " "
        masked_text = "".join(masked)
        position = closing_end

    return "".join(masked)


def _finish_inline_link(text: str, position: int) -> int | None:
    while position < len(text) and text[position].isspace():
        position += 1
    if position < len(text) and text[position] == ")":
        return position + 1
    if position >= len(text) or text[position] not in "\"'(":
        return None

    opener = text[position]
    closer = ")" if opener == "(" else opener
    depth = 0
    position += 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if opener == "(" and char == "(":
            depth += 1
        elif char == closer:
            if depth:
                depth -= 1
            else:
                position += 1
                break
        position += 1
    else:
        return None

    while position < len(text) and text[position].isspace():
        position += 1
    return position + 1 if position < len(text) and text[position] == ")" else None


def _parse_inline_destination(
    text: str, position: int
) -> tuple[str, int] | None:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text):
        return None

    if text[position] == "<":
        position += 1
        start = position
        while position < len(text):
            if text[position] == "\\":
                position += 2
                continue
            if text[position] == ">":
                end = _finish_inline_link(text, position + 1)
                return (text[start:position], end) if end is not None else None
            if text[position] in "\r\n":
                return None
            position += 1
        return None

    start = position
    depth = 0
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return text[start:position], position + 1
            depth -= 1
        elif char.isspace() and depth == 0:
            end = _finish_inline_link(text, position)
            return (text[start:position], end) if end is not None else None
        position += 1
    return None


def _inline_link_targets(text: str) -> list[tuple[int, str]]:
    targets: list[tuple[int, str]] = []
    position = 0
    while True:
        opener = text.find("](", position)
        if opener < 0:
            return targets
        label_opener = opener - 1
        while label_opener >= 0 and text[label_opener] not in "\r\n[":
            label_opener -= 1
        if (
            label_opener < 0
            or text[label_opener] != "["
            or _is_escaped(text, label_opener)
            or _is_escaped(text, opener)
        ):
            position = opener + 2
            continue
        parsed = _parse_inline_destination(text, opener + 2)
        if parsed is None:
            position = opener + 2
            continue
        target, end = parsed
        targets.append((opener, target))
        position = end


def _reference_destination(line: str) -> tuple[str | None, bool]:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3 or not stripped.startswith("["):
        return None, False
    delimiter = stripped.find("]:")
    if delimiter <= 1:
        return None, False
    position = delimiter + 2
    while position < len(stripped) and stripped[position] in " \t":
        position += 1
    if position >= len(stripped):
        return None, True
    if stripped[position] == "<":
        end = stripped.find(">", position + 1)
        return (stripped[position + 1 : end] if end >= 0 else None), True

    start = position
    depth = 0
    while position < len(stripped):
        char = stripped[position]
        if char == "\\":
            position += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace() and depth == 0:
            break
        position += 1
    return (stripped[start:position] if depth == 0 else None), True


def _markdown_targets(text: str) -> list[str]:
    visible_text = _mask_markdown_code(text)
    positioned = _inline_link_targets(visible_text)
    lines = visible_text.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        target, is_definition = _reference_destination(line.rstrip("\r\n"))
        if target is None and is_definition and index + 1 < len(lines):
            continuation = lines[index + 1].rstrip("\r\n")
            indentation = len(continuation) - len(continuation.lstrip(" "))
            if indentation <= 3:
                target, _ = _reference_destination(
                    f"[continuation]: {continuation.lstrip(' ')}"
                )
        if target is not None:
            positioned.append((offset, target))
        offset += len(line)
    positioned.sort(key=lambda item: item[0])
    return [target for _, target in positioned]


def _visible_markdown_lines(text: str) -> list[str | None]:
    """Devolve linhas fora de blocos cercados de código."""

    visible: list[str | None] = []
    fence_char: str | None = None
    fence_size = 0
    for line in text.splitlines():
        if fence_char is None:
            opening = re.match(r" {0,3}(`{3,}|~{3,})", line)
            if opening is None:
                visible.append(line)
                continue
            fence_char = opening.group(1)[0]
            fence_size = len(opening.group(1))
            visible.append(None)
            continue

        closing = re.fullmatch(
            rf" {{0,3}}{re.escape(fence_char)}{{{fence_size},}}[ \t]*",
            line,
        )
        visible.append(None)
        if closing is not None:
            fence_char = None
            fence_size = 0
    return visible


def _heading_slug(heading: str) -> str:
    """Converte um título Markdown no fragmento usado pelo documento."""

    heading = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", heading)
    heading = re.sub(r"<[^>]*>", "", heading)
    characters: list[str] = []
    for character in heading.casefold():
        category = unicodedata.category(character)
        if category.startswith(("L", "N")):
            characters.append(character)
        elif character.isspace():
            characters.append(" ")
    return re.sub(r"\s+", "-", "".join(characters).strip())


def _markdown_heading_slugs(text: str) -> set[str]:
    slugs: set[str] = set()
    occurrences: dict[str, int] = {}

    def add_heading(heading: str) -> None:
        base_slug = _heading_slug(heading)
        if not base_slug:
            return
        occurrence = occurrences.get(base_slug, 0)
        slug = base_slug if occurrence == 0 else f"{base_slug}-{occurrence}"
        occurrences[base_slug] = occurrence + 1
        slugs.add(slug)

    lines = _visible_markdown_lines(text)
    for index, line in enumerate(lines):
        if line is None:
            continue
        atx = re.match(r" {0,3}#{1,6}(?:[ \t]+|$)(.*)$", line)
        if atx is not None:
            heading = re.sub(r"[ \t]+#+[ \t]*$", "", atx.group(1))
            add_heading(heading)
            continue

        if (
            index > 0
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", line)
            and lines[index - 1] is not None
            and lines[index - 1].strip()
            and not lines[index - 1].startswith(("    ", "\t"))
        ):
            add_heading(lines[index - 1])
    return slugs


def _read_markdown_heading_slugs(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrackingError(f"não foi possível ler {path}: {exc}") from exc
    return _markdown_heading_slugs(text)


def find_broken_local_links(markdown_path: Path, text: str) -> list[str]:
    """Lista destinos Markdown locais ou fragmentos que não existem."""

    base = Path(markdown_path).parent
    broken: list[str] = []
    document_headings: set[str] | None = None
    for target in _markdown_targets(text):
        if (
            not target
            or target.startswith(("//", "\\\\"))
            or _URI_SCHEME.match(target)
        ):
            continue

        path_part, separator, fragment = target.partition("#")
        path_part = path_part.split("?", 1)[0]
        decoded = unquote(path_part)
        decoded = re.sub(r"\\([\\`*{}\[\]()#+.!_<>'-])", r"\1", decoded)
        local_path = Path(decoded)
        if local_path.is_absolute():
            continue

        target_path = base / local_path if decoded else Path(markdown_path)
        if decoded and not target_path.exists():
            if target not in broken:
                broken.append(target)
            continue

        if not separator or not fragment:
            continue

        expected_slug = unquote(fragment).casefold()
        if decoded:
            if target_path.suffix.casefold() not in {".md", ".markdown"}:
                continue
            headings = _read_markdown_heading_slugs(target_path)
        else:
            if document_headings is None:
                document_headings = _markdown_heading_slugs(text)
            headings = document_headings
            if not headings:
                continue
        if expected_slug not in headings and target not in broken:
            broken.append(target)
    return broken


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sincroniza o retrato automático dos experimentos."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="confere o painel sem modificá-lo",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    document = repo_root / "docs" / "ACOMPANHAMENTO_EXPERIMENTOS.md"

    try:
        generated = render_status(collect_snapshot(repo_root))
        changed = synchronize(document, generated, check=args.check)
        current_text = document.read_bytes().decode("utf-8")
        broken = find_broken_local_links(document, current_text)
        if broken:
            joined = ", ".join(broken)
            raise TrackingError(f"links locais quebrados: {joined}")
    except (TrackingError, OSError, UnicodeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print("painel atualizado; nenhum link local quebrado")
    elif changed:
        print("painel atualizado")
    else:
        print("painel já estava atualizado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
