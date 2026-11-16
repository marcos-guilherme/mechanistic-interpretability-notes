from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import update_experiment_tracking as tracking  # noqa: E402


def test_collect_snapshot_reads_canonical_results() -> None:
    snapshot = tracking.collect_snapshot(REPO_ROOT)

    assert snapshot["jspace_causal"] == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "run_id": "20eabeef-17f5-49dc-a4a1-fcafc20a9533",
        "operational_status": "completed",
        "scientific_status": "inconclusive",
        "reason": "primary_gates_failed",
    }
    assert snapshot["jspace_observation"]["run_id"] == (
        "2fcb8a49-136f-4868-91cc-f98f9dc2e57d"
    )
    assert snapshot["jspace_observation"]["operational_status"] == "success"
    assert snapshot["jspace_observation"]["item_count"] == 93
    assert snapshot["jspace_observation"]["jlens_band"] == pytest.approx(
        [0.3440860215, 0.5483870968, 0.5967741935]
    )
    assert snapshot["jspace_observation"]["logit_lens_band"] == pytest.approx(
        [0.2849462366, 0.3924731183, 0.4354838710]
    )

    refusal = snapshot["refusal_direction"]
    assert refusal["model"] == "google/gemma-2b-it"
    assert refusal["completed_stages"] == [
        "baseline",
        "extract",
        "intervene",
        "evaluate",
    ]
    assert refusal["selected"] == {"layer": 12, "position_offset": -1}
    assert refusal["harmful_refusal_rates"] == {
        "baseline": 1.0,
        "ablation": 0.0,
        "random_control": 1.0,
    }
    assert refusal["addition_harmless_refusal_rate"] == 0.9375
    assert refusal["ablation_harmless_kl"] == pytest.approx(0.0305709913)

    qwen = snapshot["qwen_refusal_pilot"]
    assert qwen is not None
    assert qwen["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert qwen["model_revision"] == "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    assert qwen["run_id"] == "c0818150-f834-4692-b271-4e10a97eb57e"
    assert qwen["implementation_commit"] == "2a2eb714246174e4f627fcd0061c61143a4e8276"
    assert qwen["operational_success"] is True
    assert qwen["scientific_status"] == "causal_gate_failed"
    assert qwen["selected"] == {"layer": 18, "position_offset": -1}
    assert qwen["harmful_refusal_rates"] == {
        "baseline": 0.875,
        "ablation": 0.0,
        "random_control": 0.875,
    }
    assert qwen["harmless_refusal_rates"] == {
        "baseline": 0.03125,
        "addition": 1.0,
    }
    assert qwen["ablation_harmless_kl"] == pytest.approx(0.2638247609138489)

    followup = snapshot["qwen_refusal_followup"]
    assert followup is not None
    assert followup["dose"]["status"] == "pass"
    assert followup["dose"]["selected_pair"] == {
        "ablation": 0.2,
        "addition": 0.5,
    }
    assert followup["jlens"]["status"] == "pass"
    assert followup["jlens"]["n_fit"] == 100
    assert followup["jlens"]["n_holdout"] == 20
    assert followup["jlens"]["benchmark_deduplicated"] is False
    assert followup["code_eval"]["status"] == "pass"
    assert followup["code_eval"]["human_review_status"] == "pending"
    assert followup["code_eval"]["metrics"]["mitre"]["ablation"] == {
        "refusal_rate": 0.0,
        "mean_kl": pytest.approx(4.391125907190144),
    }
    assert followup["code_eval"]["metrics"]["humaneval_plus"]["baseline"] == {
        "refusal_rate": 0.0,
        "mean_kl": None,
        "pass_at_1": 0.40625,
    }
    assert followup["code_eval"]["metrics"]["humaneval_plus"]["ablation"] == {
        "refusal_rate": 0.0,
        "mean_kl": pytest.approx(0.022580120799830183),
        "pass_at_1": 0.40625,
    }


def test_collect_snapshot_omits_qwen_pilot_when_artifacts_absent(
    tmp_path: Path,
) -> None:
    artifact_paths = [
        "projects/jspace_llama/results/execution_manifest.json",
        "projects/jspace_llama/results/final_evaluation.json",
        "projects/jspace_llama/results/jspace_observation_manifest.json",
        "projects/jspace_llama/results/jspace_observation.json",
        "projects/refusal_direction/results/execution_manifest.json",
        "projects/refusal_direction/results/final_evaluation.json",
    ]
    for relative in artifact_paths:
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    snapshot = tracking.collect_snapshot(tmp_path)

    assert snapshot["qwen_refusal_pilot"] is None


def test_replace_generated_block_preserves_human_text_and_is_idempotent() -> None:
    original = (
        "# Diário\n\nTexto humano antes.\n\n"
        f"{tracking.START_MARKER}\nconteúdo antigo\n{tracking.END_MARKER}\n\n"
        "Texto humano depois.\n"
    )

    updated = tracking.replace_generated_block(original, "novo retrato\n")

    assert updated == (
        "# Diário\n\nTexto humano antes.\n\n"
        f"{tracking.START_MARKER}\nnovo retrato\n{tracking.END_MARKER}\n\n"
        "Texto humano depois.\n"
    )
    assert tracking.replace_generated_block(updated, "novo retrato\n") == updated


def test_replace_generated_block_preserves_crlf_line_endings() -> None:
    original = (
        f"# Diário\r\n\r\n{tracking.START_MARKER}\r\nantigo\r\n"
        f"{tracking.END_MARKER}\r\n\r\nNarrativa.\r\n"
    )

    updated = tracking.replace_generated_block(original, "linha 1\nlinha 2\n")

    assert "\n" not in updated.replace("\r\n", "")
    assert (
        f"{tracking.START_MARKER}\r\nlinha 1\r\nlinha 2\r\n"
        f"{tracking.END_MARKER}"
    ) in updated


@pytest.mark.parametrize(
    "document",
    [
        "sem marcadores",
        "<!-- BEGIN GENERATED STATUS -->\nsem final",
        "<!-- END GENERATED STATUS -->\nsem início",
        (
            "<!-- BEGIN GENERATED STATUS -->\na\n<!-- END GENERATED STATUS -->\n"
            "<!-- BEGIN GENERATED STATUS -->\nb\n<!-- END GENERATED STATUS -->"
        ),
    ],
)
def test_replace_generated_block_rejects_invalid_markers(document: str) -> None:
    with pytest.raises(tracking.TrackingError, match="marcador"):
        tracking.replace_generated_block(document, "novo\n")


def test_synchronize_check_detects_stale_content_without_writing(
    tmp_path: Path,
) -> None:
    document = tmp_path / "tracking.md"
    original = (
        f"# Manual\n\n{tracking.START_MARKER}\nantigo\n"
        f"{tracking.END_MARKER}\n\nNarrativa preservada.\n"
    )
    document.write_text(original, encoding="utf-8")

    with pytest.raises(tracking.TrackingOutOfDate):
        tracking.synchronize(document, "atual\n", check=True)

    assert document.read_text(encoding="utf-8") == original
    assert tracking.synchronize(document, "atual\n", check=False) is True
    assert tracking.synchronize(document, "atual\n", check=False) is False
    assert tracking.synchronize(document, "atual\n", check=True) is False
    assert document.read_text(encoding="utf-8").endswith(
        f"{tracking.END_MARKER}\n\nNarrativa preservada.\n"
    )


def test_synchronize_replaces_changed_document_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "tracking.md"
    document.write_text(
        f"{tracking.START_MARKER}\nantigo\n{tracking.END_MARKER}\n",
        encoding="utf-8",
    )
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, target: str | Path) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(tracking.os, "replace", recording_replace)

    assert tracking.synchronize(document, "novo\n", check=False) is True
    assert len(replacements) == 1
    assert replacements[0][1] == document
    assert not replacements[0][0].exists()


def test_find_broken_local_links_ignores_external_and_anchor_links(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "exists.json").write_text("{}", encoding="utf-8")
    (docs / "a(b).md").write_text("ok", encoding="utf-8")
    markdown = docs / "tracking.md"
    text = "\n".join(
        [
            "[existe](exists.json)",
            "[parênteses](a(b).md)",
            "[âncora](#seção)",
            "[externo](https://example.com)",
            "[outro esquema](ftp://example.com/file)",
            "[ausente](missing.json)",
            "![imagem ausente](missing.png)",
            "[referência]: missing-reference.json",
            "[referência multilinha]:",
            "  missing-multiline.json",
            r"\[escapado](missing-escaped.json)",
            "`[código](missing-inline-code.json)`",
            "```md",
            "[bloco de código](missing-fenced-code.json)",
            "```",
        ]
    )

    assert tracking.find_broken_local_links(markdown, text) == [
        "missing.json",
        "missing.png",
        "missing-reference.json",
        "missing-multiline.json",
    ]


def test_find_broken_local_links_validates_heading_fragments(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "processo.md"
    target.write_text(
        "# Processo\n\n## Limites e próximos passos\n",
        encoding="utf-8",
    )
    markdown = docs / "tracking.md"
    text = "\n".join(
        [
            "# Painel",
            "[seção local válida](#painel)",
            "[seção externa válida](processo.md#limites-e-próximos-passos)",
            "[seção local ausente](#estado-ausente)",
            "[seção externa ausente](processo.md#incidente-ausente)",
        ]
    )

    assert tracking.find_broken_local_links(markdown, text) == [
        "#estado-ausente",
        "processo.md#incidente-ausente",
    ]


def test_render_status_names_operational_and_scientific_meanings() -> None:
    rendered = tracking.render_status(tracking.collect_snapshot(REPO_ROOT))

    assert "Estado operacional" in rendered
    assert "Conclusão científica" in rendered
    assert "inconclusiva" in rendered
    assert "observacional" in rendered
    assert "causal local" in rendered
    assert "0,344086 / 0,548387 / 0,596774" in rendered
    assert "Qwen2.5-Coder-7B-Instruct" in rendered
    assert "causal_gate_failed" in rendered
    assert "selectividade insuficiente" in rendered
    assert "0,263825" in rendered
    assert "Follow-up Qwen" in rendered
    assert "20%" in rendered
    assert "50%" in rendered
    assert "40,62%" in rendered
    assert "21,88%" in rendered
    assert "revisão humana pendente" in rendered
    assert "não foi deduplicado" in rendered
    assert "KL alta" in rendered
    assert "não é uma sandbox completa" in rendered


def test_render_status_puts_blank_line_between_gemma_and_qwen_blocks() -> None:
    rendered = tracking.render_status(tracking.collect_snapshot(REPO_ROOT))

    separator = (
        "../projects/refusal_direction/results/final_evaluation.json).\n\n"
        "**Direção de Recusa (piloto Qwen).**"
    )
    assert separator in rendered


def test_render_status_derives_qwen_conclusion_from_scientific_status() -> None:
    snapshot = tracking.collect_snapshot(REPO_ROOT)
    snapshot["qwen_refusal_pilot"]["scientific_status"] = "causal_gate_failed"

    rendered = tracking.render_status(snapshot)

    assert "`causal_gate_failed`" in rendered
    assert "selectividade insuficiente" in rendered
    assert "falha no gate causal" not in rendered


def test_render_status_does_not_keep_causal_label_when_metrics_change() -> None:
    snapshot = tracking.collect_snapshot(REPO_ROOT)
    snapshot["refusal_direction"]["harmful_refusal_rates"]["ablation"] = 1.0

    rendered = tracking.render_status(snapshot)

    assert "causal local" not in rendered
    assert "reavaliação" in rendered


def test_render_status_requires_all_refusal_stages_for_causal_label() -> None:
    snapshot = tracking.collect_snapshot(REPO_ROOT)
    snapshot["refusal_direction"]["completed_stages"].remove("evaluate")

    rendered = tracking.render_status(snapshot)

    assert "causal local" not in rendered
    assert "reavaliação" in rendered


def test_render_status_rejects_tiny_refusal_addition_effect() -> None:
    snapshot = tracking.collect_snapshot(REPO_ROOT)
    snapshot["refusal_direction"]["addition_harmless_refusal_rate"] = 0.01

    rendered = tracking.render_status(snapshot)

    assert "causal local" not in rendered
    assert "reavaliação" in rendered


def test_collect_snapshot_rejects_wrong_json_field_type(tmp_path: Path) -> None:
    artifact_paths = [
        "projects/jspace_llama/results/execution_manifest.json",
        "projects/jspace_llama/results/final_evaluation.json",
        "projects/jspace_llama/results/jspace_observation_manifest.json",
        "projects/jspace_llama/results/jspace_observation.json",
        "projects/refusal_direction/results/execution_manifest.json",
        "projects/refusal_direction/results/final_evaluation.json",
    ]
    for relative in artifact_paths:
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    manifest_path = (
        tmp_path / "projects/jspace_llama/results/execution_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["model"]["id"] = ["não", "é", "texto"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(tracking.TrackingError, match="esperado texto"):
        tracking.collect_snapshot(tmp_path)


def test_collect_snapshot_rejects_non_boolean_operational_status(
    tmp_path: Path,
) -> None:
    artifact_paths = [
        "projects/jspace_llama/results/execution_manifest.json",
        "projects/jspace_llama/results/final_evaluation.json",
        "projects/jspace_llama/results/jspace_observation_manifest.json",
        "projects/jspace_llama/results/jspace_observation.json",
        "projects/refusal_direction/results/execution_manifest.json",
        "projects/refusal_direction/results/final_evaluation.json",
        "projects/refusal_direction/results/qwen25_coder_7b/pilot_evaluation.json",
        "projects/refusal_direction/results/qwen25_coder_7b/final_evaluation.json",
    ]
    for relative in artifact_paths:
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    pilot_path = (
        tmp_path
        / "projects/refusal_direction/results/qwen25_coder_7b/pilot_evaluation.json"
    )
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot["operational_success"] = "false"
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")

    with pytest.raises(tracking.TrackingError, match="esperado booleano"):
        tracking.collect_snapshot(tmp_path)
