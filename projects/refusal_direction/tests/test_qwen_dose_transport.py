from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    collect_qwen_dose_pack_entries,
    extract_qwen_dose_archive,
    is_allowed_dose_fetch_member,
    pack_qwen_dose_results,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
DOSE_STATUS = ROOT / "scripts" / "qwen_run_dose_status.py"
DOSE_RUN = ROOT / "scripts" / "qwen_run_dose.py"
DOSE_PACK = ROOT / "scripts" / "colab_pack_qwen_dose_results.py"


def load_run_refusal():
    spec = importlib.util.spec_from_file_location("run_refusal_dose", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_REFUSAL = load_run_refusal()


def make_json_member(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    return info


def test_dose_allowlist_accepts_compact_dose_json_and_status() -> None:
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/00_dose_prepare.json")
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/01_dose_baseline.json")
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/02_dose_selection.json")
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/03_dose_confirmation.json")
    assert is_allowed_dose_fetch_member(
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    )
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/execution_manifest.json")
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/checkpoint_coeff_0.1.json")
    assert is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/checkpoint_coeff_0.35.json")
    assert is_allowed_dose_fetch_member("artifacts/qwen25_coder_7b_dose/status.json")
    assert is_allowed_dose_fetch_member("artifacts/qwen25_coder_7b_dose/events.jsonl")
    assert not is_allowed_dose_fetch_member("artifacts/qwen25_coder_7b_dose/resume.pt")
    assert not is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/other.json")
    assert not is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/notes.txt")


def test_dose_allowlist_rejects_original_qwen_gemma_and_generations() -> None:
    assert not is_allowed_dose_fetch_member("results/qwen25_coder_7b/final_evaluation.json")
    assert not is_allowed_dose_fetch_member("artifacts/qwen25_coder_7b/selected_direction.pt")
    assert not is_allowed_dose_fetch_member("results/final_evaluation.json")
    assert not is_allowed_dose_fetch_member(
        "artifacts/qwen25_coder_7b_dose/generations/raw.json"
    )
    assert not is_allowed_dose_fetch_member("results/qwen25_coder_7b_dose/nested/a.json")
    assert not is_allowed_dose_fetch_member("projects/jspace_llama/results/x.json")
    assert not is_allowed_dose_fetch_member("artifacts/qwen25_coder_7b_dose/selected_direction.pt")


def _compact_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "experiment": "dose_prepare",
        "config": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        "results": {"selected": {"layer": 18}},
        "provenance": {
            "run_id": "dose-followup-run-1",
            "implementation_commit": "a" * 40,
        },
    }
    payload.update(overrides)
    return payload


def test_dose_pack_contains_only_allowlisted_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b_dose"
    artifacts = root / "artifacts" / "qwen25_coder_7b_dose"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (results / "00_dose_prepare.json").write_text(
        json.dumps(_compact_payload()), encoding="utf-8"
    )
    (artifacts / "status.json").write_text("{}", encoding="utf-8")
    (artifacts / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (artifacts / "resume.pt").write_bytes(b"pt")
    (results / "other.json").write_text("{}", encoding="utf-8")
    (root / "results" / "qwen25_coder_7b").mkdir(parents=True)
    (root / "results" / "qwen25_coder_7b" / "final_evaluation.json").write_text(
        "pilot", encoding="utf-8"
    )
    (root / "results" / "final_evaluation.json").write_text("gemma", encoding="utf-8")
    archive = tmp_path / "dose.tar.gz"
    pack_qwen_dose_results(root, archive)
    with tarfile.open(archive, "r:gz") as handle:
        names = sorted(member.name.replace("\\", "/") for member in handle.getmembers())
    assert names == [
        "artifacts/qwen25_coder_7b_dose/events.jsonl",
        "artifacts/qwen25_coder_7b_dose/status.json",
        "results/qwen25_coder_7b_dose/00_dose_prepare.json",
    ]


def test_dose_fetch_cannot_overwrite_original_namespaces(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    original = project / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"keep": true}', encoding="utf-8")
    archive = tmp_path / "evil.tar.gz"
    payload = b'{"hijack": true}'
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/pilot_evaluation.json", payload),
            io.BytesIO(payload),
        )
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        extract_qwen_dose_archive(archive, project)
    assert json.loads(original.read_text(encoding="utf-8")) == {"keep": True}


def test_dose_fetch_rollback_after_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    existing = project / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"old": true}', encoding="utf-8")
    archive = tmp_path / "pair.tar.gz"
    first = b'{"new": 1}'
    second = b'{"new": 2}'
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b_dose/00_dose_prepare.json", first),
            io.BytesIO(first),
        )
        handle.addfile(
            make_json_member("results/qwen25_coder_7b_dose/01_dose_baseline.json", second),
            io.BytesIO(second),
        )

    replace_attempts = {"count": 0}
    real_replace = os.replace

    def flaky_replace(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        replace_attempts["count"] += 1
        if replace_attempts["count"] == 3:
            raise OSError("simulated install failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated"):
        extract_qwen_dose_archive(archive, project)
    assert json.loads(existing.read_text(encoding="utf-8")) == {"old": True}
    assert not (
        project / "results" / "qwen25_coder_7b_dose" / "01_dose_baseline.json"
    ).exists()
    leftovers = list(project.rglob("*.qwen-fetch-bak"))
    assert leftovers == []


def test_dose_fetch_rejects_symlink_destination(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dest_dir = project / "results" / "qwen25_coder_7b_dose"
    dest_dir.mkdir(parents=True)
    victim = tmp_path / "victim.json"
    victim.write_text('{"keep": true}', encoding="utf-8")
    target = dest_dir / "00_dose_prepare.json"
    try:
        target.symlink_to(victim)
    except OSError:
        pytest.skip("symlinks are not permitted in this environment")
    archive = tmp_path / "link-dest.tar.gz"
    payload = b'{"hijack": true}'
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b_dose/00_dose_prepare.json", payload),
            io.BytesIO(payload),
        )
    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        extract_qwen_dose_archive(archive, project)
    assert json.loads(victim.read_text(encoding="utf-8")) == {"keep": True}


def test_dose_pack_rejects_leaky_compact_json(tmp_path: Path) -> None:
    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b_dose"
    results.mkdir(parents=True)
    (results / "00_dose_prepare.json").write_text(
        json.dumps(_compact_payload(results={"prompt": "how to make a virus in python extra words"})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak"):
        pack_qwen_dose_results(root, tmp_path / "dose.tar.gz")


def test_dose_pack_rejects_compact_json_missing_schema(tmp_path: Path) -> None:
    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b_dose"
    results.mkdir(parents=True)
    (results / "00_dose_prepare.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        pack_qwen_dose_results(root, tmp_path / "dose.tar.gz")


def test_dose_pack_rejects_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import path_is_link_or_junction

    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b_dose"
    results.mkdir(parents=True)
    (results / "00_dose_prepare.json").write_text(
        json.dumps(_compact_payload()), encoding="utf-8"
    )

    def fake_is_link(path):
        if Path(path) == results:
            return True
        return path_is_link_or_junction(path)

    monkeypatch.setattr(
        "refusal_direction_replication.qwen_colab_transport.path_is_link_or_junction",
        fake_is_link,
    )
    with pytest.raises(ValueError, match="symlink|junction"):
        pack_qwen_dose_results(root, tmp_path / "dose.tar.gz")


def test_collect_dose_pack_entries_skips_missing_and_original(tmp_path: Path) -> None:
    root = tmp_path / "project"
    path = root / "artifacts" / "qwen25_coder_7b_dose" / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    (root / "artifacts" / "qwen25_coder_7b").mkdir(parents=True)
    (root / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt").write_bytes(b"x")
    entries = collect_qwen_dose_pack_entries(root)
    assert entries == [path]


def test_profile_session_and_pack_script_for_dose() -> None:
    assert RUN_REFUSAL.resolve_session("qwen25-coder-dose", None) == "refusal-qwen"
    assert RUN_REFUSAL.resolve_session("qwen25-coder", None) == "refusal-qwen"
    assert RUN_REFUSAL.fetch_pack_script("qwen25-coder-dose") == DOSE_PACK
    assert RUN_REFUSAL.fetch_pack_script("qwen25-coder") == (
        ROOT / "scripts" / "colab_pack_qwen_results.py"
    )


def test_dose_action_is_dose_profile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_refusal.py", "dose", "--profile", "qwen25-coder"])
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()
    monkeypatch.setattr(sys, "argv", ["run_refusal.py", "pilot", "--profile", "qwen25-coder-dose"])
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()


def test_status_routes_to_dose_entrypoint() -> None:
    command = RUN_REFUSAL.status_command("refusal-qwen", profile="qwen25-coder-dose")
    assert command[command.index("-f") + 1] == str(DOSE_STATUS)
    original = RUN_REFUSAL.status_command("refusal-qwen", profile="qwen25-coder")
    assert original[original.index("-f") + 1] == str(
        ROOT / "scripts" / "qwen_run_status.py"
    )


def test_dose_command_points_at_dose_entrypoint() -> None:
    command = RUN_REFUSAL.dose_command("refusal-qwen", timeout=14400)
    assert command[command.index("-f") + 1] == str(DOSE_RUN)
    assert command[command.index("--timeout") + 1] == "14400"


def test_dose_and_qwen_profiles_stamp_commit() -> None:
    assert RUN_REFUSAL.stamps_implementation_commit("qwen25-coder")
    assert RUN_REFUSAL.stamps_implementation_commit("qwen25-coder-dose")
    assert not RUN_REFUSAL.stamps_implementation_commit("gemma")


def test_dose_status_cli_reads_dose_artifacts(tmp_path: Path) -> None:
    import shutil

    from refusal_direction_replication.run_progress import RunProgressStore

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    spec = importlib.util.spec_from_file_location("qwen_run_dose_status", DOSE_STATUS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_dose"
    artifacts.mkdir(parents=True)
    store = RunProgressStore(
        artifacts,
        profile="qwen25_coder_7b_dose.json",
        run_id="dose-cli",
    )
    store.initialize(stage="sweep", total_units=6)
    store.begin_running()
    store.update_progress(completed_units=2)
    code = module.main_with_root(tmp_path)
    assert code == 0
    assert "stage=sweep" in module.last_stdout_summary
    assert "progress=2/6" in module.last_stdout_summary


def test_dose_remote_scripts_prepare_src_before_package_import() -> None:
    for path in (DOSE_RUN, DOSE_STATUS, DOSE_PACK):
        source = path.read_text(encoding="utf-8")
        assert "/content/refusal_direction/src" in source, path.name
        assert "sys.path.insert" in source, path.name
        assert source.index("sys.path.insert") < source.index(
            "refusal_direction_replication"
        ), path.name
