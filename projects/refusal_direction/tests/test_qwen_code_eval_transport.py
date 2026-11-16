from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    CODE_EVAL_REMOTE_ARCHIVE,
    REMOTE_CODE_EVAL_DIRECTION_STAGING,
    REMOTE_CODE_EVAL_LENS_STAGING,
    atomic_install_code_eval_sources,
    collect_qwen_code_eval_pack_entries,
    extract_qwen_code_eval_archive,
    is_allowed_code_eval_fetch_member,
    pack_qwen_code_eval_results,
    require_local_code_eval_sources,
)


ROOT = Path(__file__).parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
RUN_SCRIPT = ROOT / "scripts" / "qwen_run_code_eval.py"
STATUS_SCRIPT = ROOT / "scripts" / "qwen_run_code_eval_status.py"
PACK_SCRIPT = ROOT / "scripts" / "colab_pack_qwen_code_eval_results.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_qwen_code_eval.py"
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_code_eval.py"


def _orchestrator():
    spec = importlib.util.spec_from_file_location("run_refusal_code_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_code_eval_fetch_allowlist_separates_profile_and_allows_raw_artifacts() -> None:
    allowed = (
        "results/qwen25_coder_7b_code_eval/00_prepare.json",
        "results/qwen25_coder_7b_code_eval/code_eval_evaluation.json",
        "artifacts/qwen25_coder_7b_code_eval/status.json",
        "artifacts/qwen25_coder_7b_code_eval/events.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/datasets/mitre.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/logits/mitre__baseline.pt",
        "artifacts/qwen25_coder_7b_code_eval/evalplus/results__baseline.json",
        "artifacts/qwen25_coder_7b_code_eval/readout/mitre__baseline.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/manual_review_blind.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/manual_review_key.json",
    )
    assert all(is_allowed_code_eval_fetch_member(path) for path in allowed)
    assert not is_allowed_code_eval_fetch_member(
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    )
    assert not is_allowed_code_eval_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert not is_allowed_code_eval_fetch_member(
        "artifacts/qwen25_coder_7b_code_eval/evalplus/temp__baseline/evil.py"
    )


def test_code_eval_source_pair_installs_atomically_and_is_confined(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        project
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    assert require_local_code_eval_sources(project) == (direction, lens)

    staged_direction = tmp_path / "direction.pt"
    staged_lens = tmp_path / "lens.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_lens.write_bytes(b"new-lens")
    installed = atomic_install_code_eval_sources(
        staged_direction=staged_direction,
        staged_lens=staged_lens,
        project=project,
    )
    assert installed == (direction, lens)
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"

    staged_direction.write_bytes(b"rollback-direction")
    staged_lens.unlink()
    with pytest.raises(FileNotFoundError):
        atomic_install_code_eval_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            project=project,
        )
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"


def test_pack_extract_keeps_raw_ignored_artifacts_inside_profile(tmp_path: Path) -> None:
    root = tmp_path / "root"
    results = root / "results/qwen25_coder_7b_code_eval"
    artifacts = root / "artifacts/qwen25_coder_7b_code_eval"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    compact = {
        "schema_version": 1,
        "experiment": "code_eval_prepare",
        "config": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        "results": {"counts": {"mitre": 32}},
        "provenance": {
            "run_id": "run",
            "implementation_commit": "a" * 40,
        },
    }
    (results / "00_prepare.json").write_text(json.dumps(compact), encoding="utf-8")
    (artifacts / "status.json").write_text(
        json.dumps({"state": "running"}), encoding="utf-8"
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps({"event": "progress"}) + "\n", encoding="utf-8"
    )
    raw = artifacts / "generations/mitre__baseline.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps({"prompt": "raw only", "response": "raw only"}) + "\n",
        encoding="utf-8",
    )
    entries = collect_qwen_code_eval_pack_entries(root)
    names = {str(path.relative_to(root)).replace("\\", "/") for path in entries}
    assert "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl" in names
    archive = tmp_path / "code-eval.tar.gz"
    pack_qwen_code_eval_results(root, archive)
    destination = tmp_path / "destination"
    destination.mkdir()
    extract_qwen_code_eval_archive(archive, destination)
    assert (
        destination
        / "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl"
    ).is_file()


def test_colab_profile_routes_sync_bootstrap_run_status_and_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    assert module.resolve_session("qwen25-coder-code-eval", None) == "refusal-qwen"
    assert module.fetch_pack_script("qwen25-coder-code-eval") == PACK_SCRIPT
    assert module.code_eval_command("refusal-qwen", timeout=14400)[-2:] == [
        "--timeout",
        "14400",
    ]
    assert module.status_command(
        "refusal-qwen", profile="qwen25-coder-code-eval"
    )[module.status_command(
        "refusal-qwen", profile="qwen25-coder-code-eval"
    ).index("-f") + 1] == str(STATUS_SCRIPT)
    assert INSTALL_SCRIPT in module.bootstrap_scripts("qwen25-coder-code-eval")

    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        project
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(module, "current_implementation_commit", lambda cwd=None: "a" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-code-eval"],
    )
    module.main()
    uploaded = [item for command in commands for item in command]
    assert REMOTE_CODE_EVAL_DIRECTION_STAGING in uploaded
    assert REMOTE_CODE_EVAL_LENS_STAGING in uploaded


def test_code_eval_scripts_exist_and_pin_dependencies() -> None:
    for path in (RUN_SCRIPT, STATUS_SCRIPT, PACK_SCRIPT, VALIDATE_SCRIPT, INSTALL_SCRIPT):
        assert path.is_file()
        assert "/content/refusal_direction/src" in path.read_text(encoding="utf-8")
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e" in install
    assert "581d398613e5602a5af361e1c34d3a92ea82ba8e" in install
    assert "--no-deps" in install
    for dependency in (
        "psutil",
        "termcolor",
        "tree_sitter",
        "tree-sitter-python",
    ):
        assert dependency in install
    assert CODE_EVAL_REMOTE_ARCHIVE == Path("/content/qwen_code_eval_results.tar.gz")
