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
    collect_qwen_pack_entries,
    extract_qwen_archive,
    is_allowed_fetch_member,
    normalize_member_name,
    pack_qwen_results,
    validate_qwen_archive,
)
from refusal_direction_replication.run_progress import (
    RunProgressStore,
    STATUS_FILENAME,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
STATUS_CLI = ROOT / "scripts" / "qwen_run_status.py"


def load_run_refusal():
    spec = importlib.util.spec_from_file_location("run_refusal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_REFUSAL = load_run_refusal()


def load_status_module():
    spec = importlib.util.spec_from_file_location("qwen_run_status", STATUS_CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STATUS_MODULE = load_status_module()


def make_json_member(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    return info


def test_invalid_state_transition_is_rejected(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="extract", total_units=10)
    store.begin_running()
    with pytest.raises(ValueError, match="transition"):
        store.update_state("pending")


def test_out_of_range_progress_is_rejected(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="extract", total_units=5)
    store.begin_running()
    with pytest.raises(ValueError, match="completed_units"):
        store.update_progress(completed_units=6, total_units=5)


def test_status_is_written_atomically(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=3)
    status_path = tmp_path / STATUS_FILENAME
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["profile"] == "qwen25_coder_7b.json"
    assert payload["run_id"] == "run-1"
    assert payload["current_stage"] == "baseline"
    assert payload["state"] == "pending"
    assert payload["completed_units"] == 0
    assert payload["total_units"] == 3
    assert "updated_at_utc" in payload


def test_events_append_json_lines_with_run_id(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=1)
    store.begin_running(message="started")
    events_path = tmp_path / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["run_id"] == "run-1"
    assert json.loads(lines[1])["event"] == "state_changed"


def test_completed_stage_is_skipped_on_resume(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=1)
    store.begin_running()
    store.complete_stage()
    assert store.should_skip_stage("baseline")
    assert not store.should_skip_stage("interventions")


def test_resume_does_not_inherit_other_run_id(tmp_path: Path) -> None:
    prior = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-old")
    prior.initialize(stage="baseline", total_units=1)
    prior.begin_running()
    prior.complete_stage()
    fresh = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-new")
    assert not fresh.should_skip_stage("baseline")


def test_terminal_state_rejects_progress_mutation(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=1)
    store.begin_running()
    store.complete_stage()
    with pytest.raises(ValueError, match="terminal"):
        store.update_progress(completed_units=1)


def test_qwen_pack_contains_only_allowlisted_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    results = root / "results" / "qwen25_coder_7b"
    artifacts = root / "artifacts" / "qwen25_coder_7b"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (results / "final_evaluation.json").write_text("{}", encoding="utf-8")
    (artifacts / "status.json").write_text("{}", encoding="utf-8")
    (artifacts / "candidates.pt").write_bytes(b"pt")
    (root / "results" / "final_evaluation.json").write_text("gemma", encoding="utf-8")
    archive = tmp_path / "qwen.tar.gz"
    pack_qwen_results(root, archive)
    with tarfile.open(archive, "r:gz") as handle:
        names = sorted(member.name.replace("\\", "/") for member in handle.getmembers())
    assert names == [
        "artifacts/qwen25_coder_7b/candidates.pt",
        "artifacts/qwen25_coder_7b/status.json",
        "results/qwen25_coder_7b/final_evaluation.json",
    ]


def test_normalize_rejects_raw_dotdot_without_lstrip_sanitization() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        normalize_member_name("../results/qwen25_coder_7b/evil.json")
    with pytest.raises(ValueError, match="unsafe"):
        normalize_member_name("results/qwen25_coder_7b/../../evil.json")


def test_qwen_fetch_rejects_absolute_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "evil.tar.gz"
    payload = b"{}"
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("/results/qwen25_coder_7b/evil.json", payload),
            io.BytesIO(payload),
        )
    with pytest.raises(ValueError):
        extract_qwen_archive(archive, project)
    assert not (project / "results").exists()


def test_qwen_fetch_rejects_symlink_member(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "link.tar.gz"
    info = tarfile.TarInfo(name="results/qwen25_coder_7b/link.json")
    info.type = tarfile.SYMTYPE
    info.linkname = "results/qwen25_coder_7b/ok.json"
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(info)
    with pytest.raises(ValueError, match="link"):
        extract_qwen_archive(archive, project)


def test_qwen_fetch_rejects_directory_member(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "dir.tar.gz"
    info = tarfile.TarInfo(name="results/qwen25_coder_7b/")
    info.type = tarfile.DIRTYPE
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(info)
    with pytest.raises(ValueError, match="directory"):
        extract_qwen_archive(archive, project)


def test_qwen_fetch_rejects_duplicate_members(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "dup.tar.gz"
    payload = b"{}"
    with tarfile.open(archive, "w:gz") as handle:
        info = make_json_member("results/qwen25_coder_7b/a.json", payload)
        handle.addfile(info, io.BytesIO(payload))
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="duplicate"):
        extract_qwen_archive(archive, project)


def test_qwen_fetch_rejects_oversized_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "big.tar.gz"
    payload = b"x" * 32
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/big.json", payload),
            io.BytesIO(payload),
        )
    import refusal_direction_replication.qwen_colab_transport as transport

    monkeypatch.setattr(transport, "MAX_MEMBER_BYTES", 16)
    with pytest.raises(ValueError, match="member"):
        extract_qwen_archive(archive, project)


def test_qwen_fetch_rejects_total_uncompressed_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "total.tar.gz"
    first = b"a" * 20
    second = b"b" * 20
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/one.json", first),
            io.BytesIO(first),
        )
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/two.json", second),
            io.BytesIO(second),
        )
    members = tarfile.open(archive, "r:gz").getmembers()
    import refusal_direction_replication.qwen_colab_transport as transport

    monkeypatch.setattr(transport, "MAX_TOTAL_UNCOMPRESSED_BYTES", 30)
    with pytest.raises(ValueError, match="total uncompressed size"):
        validate_qwen_archive(members)


def test_qwen_fetch_all_or_nothing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "mixed.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        good = b'{"ok": true}'
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/ok.json", good),
            io.BytesIO(good),
        )
        bad_info = make_json_member("results/final_evaluation.json", b"gemm")
        handle.addfile(bad_info, io.BytesIO(b"gemm"))
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        extract_qwen_archive(archive, project)
    assert not (project / "results" / "qwen25_coder_7b").exists()


def test_qwen_fetch_rollback_after_install_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    existing = project / "results" / "qwen25_coder_7b" / "existing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"old": true}', encoding="utf-8")
    archive = tmp_path / "pair.tar.gz"
    first = b'{"new": 1}'
    second = b'{"new": 2}'
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/existing.json", first),
            io.BytesIO(first),
        )
        handle.addfile(
            make_json_member("results/qwen25_coder_7b/other.json", second),
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
        extract_qwen_archive(archive, project)
    assert json.loads(existing.read_text(encoding="utf-8")) == {"old": True}
    assert not (project / "results" / "qwen25_coder_7b" / "other.json").exists()


def test_profile_defaults_session_and_pack_script() -> None:
    assert RUN_REFUSAL.resolve_session("gemma", None) == "refusal-gemma"
    assert RUN_REFUSAL.resolve_session("qwen25-coder", None) == "refusal-qwen"
    assert RUN_REFUSAL.resolve_session("qwen25-coder", "custom") == "custom"
    assert RUN_REFUSAL.fetch_pack_script("gemma") == (
        ROOT / "scripts" / "colab_pack_results.py"
    )
    assert RUN_REFUSAL.fetch_pack_script("qwen25-coder") == (
        ROOT / "scripts" / "colab_pack_qwen_results.py"
    )


def test_status_action_runs_qwen_status_script() -> None:
    command = RUN_REFUSAL.status_command("refusal-qwen")
    script_index = command.index("-f") + 1
    assert command[script_index] == str(ROOT / "scripts" / "qwen_run_status.py")


def test_gemma_start_command_unchanged() -> None:
    assert RUN_REFUSAL.start_command("refusal-gemma", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-gemma",
        "--gpu",
        "L4",
    ]


def test_run_refusal_does_not_import_qwen_transport_at_load() -> None:
    module = load_run_refusal()
    assert "extract_qwen_archive" not in module.__dict__
    assert "extract_qwen_results_archive" in module.__dict__


def test_qwen_status_cli_reports_missing_with_nonzero_exit(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    code = STATUS_MODULE.main_with_root(tmp_path, profile="qwen25_coder_7b.json")
    assert code != 0


def test_qwen_status_cli_prints_human_readable_success(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b"
    artifacts.mkdir(parents=True)
    store = RunProgressStore(
        artifacts,
        profile="qwen25_coder_7b.json",
        run_id="run-cli",
    )
    store.initialize(stage="baseline", total_units=4)
    store.begin_running(message="warming up")
    store.update_progress(completed_units=2)
    code = STATUS_MODULE.main_with_root(tmp_path, profile="qwen25_coder_7b.json")
    assert code == 0
    assert STATUS_MODULE.last_stdout_summary == (
        "profile=qwen25_coder_7b.json | run_id=run-cli | stage=baseline | "
        "state=running | progress=2/4 | message=warming up"
    )


def test_qwen_status_main_uses_remote_root_for_ipykernel_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.setattr(STATUS_MODULE, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "/content/refusal_direction/scripts/qwen_run_status.py",
            "-f",
            "/usr/local/share/jupyter/kernels/python3/kernel.json",
            "--",
        ],
    )
    assert STATUS_MODULE.remote_main() == 2
    assert STATUS_MODULE.REMOTE_ROOT == tmp_path


def test_is_allowed_fetch_member_accepts_results_json() -> None:
    assert is_allowed_fetch_member("results/qwen25_coder_7b/a.json")
    assert not is_allowed_fetch_member("results/qwen25_coder_7b/nested/a.json")
    assert not is_allowed_fetch_member("artifacts/gemma/status.json")


def test_collect_qwen_pack_entries_only_returns_existing_allowlisted_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    path = root / "artifacts" / "qwen25_coder_7b" / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    entries = collect_qwen_pack_entries(root)
    assert entries == [path]
    (root / "artifacts" / "qwen25_coder_7b" / "candidates.pt").write_bytes(b"x")
    assert collect_qwen_pack_entries(root) == [
        root / "artifacts" / "qwen25_coder_7b" / "candidates.pt",
        path,
    ]