from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES,
    MAX_MEMBER_BYTES,
    collect_qwen_jlens_pack_entries,
    extract_qwen_jlens_archive,
    is_allowed_jlens_fetch_member,
    pack_qwen_jlens_results,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
JLENS_STATUS = ROOT / "scripts" / "qwen_run_jlens_status.py"
JLENS_RUN = ROOT / "scripts" / "qwen_run_jlens.py"
JLENS_PACK = ROOT / "scripts" / "colab_pack_qwen_jlens_results.py"
JLENS_INSTALL = ROOT / "scripts" / "colab_install_jlens.py"


def load_run_refusal():
    spec = importlib.util.spec_from_file_location("run_refusal_jlens", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_REFUSAL = load_run_refusal()


def _compact(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "experiment": "jlens_prepare",
        "config": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        "results": {"fit_count": 100, "holdout_count": 20},
        "provenance": {"run_id": "jlens-run-1", "implementation_commit": "a" * 40},
    }
    payload.update(overrides)
    return payload


def test_jlens_allowlist_accepts_compact_status_and_tensors() -> None:
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/00_jlens_prepare.json")
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/01_jlens_probe.json")
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/02_jlens_fit.json")
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/03_jlens_holdout.json")
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/jlens_evaluation.json")
    assert is_allowed_jlens_fetch_member("results/qwen25_coder_7b_jlens/execution_manifest.json")
    assert is_allowed_jlens_fetch_member("artifacts/qwen25_coder_7b_jlens/status.json")
    assert is_allowed_jlens_fetch_member("artifacts/qwen25_coder_7b_jlens/events.jsonl")
    assert is_allowed_jlens_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/jlens_fit_checkpoint.pt"
    )
    assert is_allowed_jlens_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert not is_allowed_jlens_fetch_member("artifacts/qwen25_coder_7b_jlens/corpus.jsonl")
    assert not is_allowed_jlens_fetch_member("results/qwen25_coder_7b/final_evaluation.json")
    assert not is_allowed_jlens_fetch_member(
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    )
    assert not is_allowed_jlens_fetch_member("results/final_evaluation.json")
    assert not is_allowed_jlens_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/nested/x.pt"
    )


def test_jlens_total_archive_limit_is_raised_narrowly() -> None:
    assert MAX_MEMBER_BYTES == 2 * 1024**3
    assert MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES == 5 * 1024**3
    assert MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES > 4 * 1024**3


def test_pack_and_extract_are_confined_and_reject_leakage(tmp_path: Path) -> None:
    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b_jlens"
    artifacts = root / "artifacts" / "qwen25_coder_7b_jlens"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (results / "00_jlens_prepare.json").write_text(
        json.dumps(_compact()) + "\n", encoding="utf-8"
    )
    (results / "01_jlens_probe.json").write_text(
        json.dumps(_compact(experiment="jlens_probe")) + "\n", encoding="utf-8"
    )
    (results / "02_jlens_fit.json").write_text(
        json.dumps(_compact(experiment="jlens_fit")) + "\n", encoding="utf-8"
    )
    (results / "03_jlens_holdout.json").write_text(
        json.dumps(_compact(experiment="jlens_holdout")) + "\n", encoding="utf-8"
    )
    (results / "jlens_evaluation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "run_id": "jlens-run-1",
                "implementation_commit": "a" * 40,
                "source_artifact_hashes": {},
                "schema_version": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results / "execution_manifest.json").write_text(
        json.dumps(_compact(experiment="execution_manifest")) + "\n", encoding="utf-8"
    )
    (artifacts / "status.json").write_text(
        json.dumps({"state": "completed", "completed_units": 100, "total_units": 100})
        + "\n",
        encoding="utf-8",
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps({"event": "probe_choice", "dim_batch": 32}) + "\n",
        encoding="utf-8",
    )
    (artifacts / "jlens_fit_checkpoint.pt").write_bytes(b"ckpt")
    (artifacts / "qwen25_coder_7b_jacobian_lens.pt").write_bytes(b"lens")
    (artifacts / "corpus.jsonl").write_text('{"fit_text": "def f(): pass"}\n', encoding="utf-8")
    (root / "results" / "qwen25_coder_7b").mkdir(parents=True)
    (root / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").write_text(
        "{}", encoding="utf-8"
    )
    entries = collect_qwen_jlens_pack_entries(root)
    names = {str(path.relative_to(root)).replace("\\", "/") for path in entries}
    assert "artifacts/qwen25_coder_7b_jlens/corpus.jsonl" not in names
    assert "results/qwen25_coder_7b/pilot_evaluation.json" not in names
    assert "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt" in names
    archive = tmp_path / "jlens.tar.gz"
    pack_qwen_jlens_results(root, archive)
    destination = tmp_path / "dest"
    (destination / "results" / "qwen25_coder_7b").mkdir(parents=True)
    (destination / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").write_text(
        "keep", encoding="utf-8"
    )
    extract_qwen_jlens_archive(archive, destination)
    assert (
        destination / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    ).read_text(encoding="utf-8") == "keep"
    assert (
        destination / "artifacts" / "qwen25_coder_7b_jlens" / "qwen25_coder_7b_jacobian_lens.pt"
    ).read_bytes() == b"lens"
    leaked = tmp_path / "leaked.json"
    leaked.write_text(
        json.dumps(_compact(results={"fit_text": "def secret(): pass"})) + "\n",
        encoding="utf-8",
    )
    (results / "00_jlens_prepare.json").write_text(
        leaked.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="leak"):
        collect_qwen_jlens_pack_entries(root)


def test_extract_rolls_back_and_rejects_other_namespaces(tmp_path: Path) -> None:
    payload = json.dumps(_compact()).encode("utf-8")
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("results/qwen25_coder_7b/pilot_evaluation.json")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        extract_qwen_jlens_archive(archive, dest)


def test_profile_session_action_and_bootstrap_pin() -> None:
    assert RUN_REFUSAL.resolve_session("qwen25-coder-jlens", None) == "refusal-qwen"
    assert RUN_REFUSAL.fetch_pack_script("qwen25-coder-jlens") == JLENS_PACK
    assert RUN_REFUSAL.stamps_implementation_commit("qwen25-coder-jlens")
    command = RUN_REFUSAL.jlens_command("refusal-qwen", timeout=14400)
    assert command[command.index("-f") + 1] == str(JLENS_RUN)
    status = RUN_REFUSAL.status_command("refusal-qwen", profile="qwen25-coder-jlens")
    assert status[status.index("-f") + 1] == str(JLENS_STATUS)
    scripts = RUN_REFUSAL.bootstrap_scripts("qwen25-coder-jlens")
    assert scripts[-1] == JLENS_INSTALL
    assert scripts[0] == ROOT / "scripts" / "colab_bootstrap.py"


def test_jlens_action_is_profile_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run_refusal.py", "jlens", "--profile", "qwen25-coder"]
    )
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()
    monkeypatch.setattr(
        sys, "argv", ["run_refusal.py", "dose", "--profile", "qwen25-coder-jlens"]
    )
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()
    monkeypatch.setattr(
        sys, "argv", ["run_refusal.py", "pilot", "--profile", "qwen25-coder-jlens"]
    )
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()


def test_gemma_and_dose_routing_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUN_REFUSAL.resolve_session("gemma", None) == "refusal-gemma"
    assert RUN_REFUSAL.resolve_session("qwen25-coder-dose", None) == "refusal-qwen"
    assert RUN_REFUSAL.fetch_pack_script("qwen25-coder-dose") == (
        ROOT / "scripts" / "colab_pack_qwen_dose_results.py"
    )
    assert RUN_REFUSAL.bootstrap_scripts("gemma") == [
        ROOT / "scripts" / "colab_bootstrap.py"
    ]
    assert RUN_REFUSAL.bootstrap_scripts("qwen25-coder") == [
        ROOT / "scripts" / "colab_bootstrap.py"
    ]


def test_remote_scripts_prepend_src_and_bootstrap_pins_no_deps() -> None:
    for path in (JLENS_RUN, JLENS_STATUS, JLENS_PACK, JLENS_INSTALL):
        source = path.read_text(encoding="utf-8")
        assert "/content/refusal_direction/src" in source
        assert source.index("sys.path.insert") < source.index(
            "refusal_direction_replication"
        )
    install = JLENS_INSTALL.read_text(encoding="utf-8")
    assert "--no-deps" in install
    assert "581d398613e5602a5af361e1c34d3a92ea82ba8e" in install
    assert "anthropics/jacobian-lens" in install


def test_jlens_sync_does_not_require_dose_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(RUN_REFUSAL, "PROJECT", project)
    monkeypatch.setattr(RUN_REFUSAL, "ROOT", tmp_path)
    monkeypatch.setattr(
        RUN_REFUSAL, "current_implementation_commit", lambda cwd=None: "a" * 40
    )
    monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-jlens"],
    )
    RUN_REFUSAL.main()
    uploaded = [item for command in commands for item in command]
    assert "/content/refusal_direction.tar.gz" in uploaded
    assert "/content/dose_source_direction.pt" not in uploaded
