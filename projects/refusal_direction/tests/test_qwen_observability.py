from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.run_progress import (
    ALLOWED_TRANSITIONS,
    RunProgressStore,
    STATUS_FILENAME,
)
from refusal_direction_replication.qwen_colab_transport import (
    QWEN_FETCH_ALLOWLIST,
    collect_qwen_pack_entries,
    extract_qwen_archive,
    is_allowed_fetch_member,
    pack_qwen_results,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
SPEC = importlib.util.spec_from_file_location("run_refusal", SCRIPT)
RUN_REFUSAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN_REFUSAL)

STATUS_CLI = ROOT / "scripts" / "qwen_run_status.py"
STATUS_SPEC = importlib.util.spec_from_file_location("qwen_run_status", STATUS_CLI)
STATUS_MODULE = importlib.util.module_from_spec(STATUS_SPEC)
assert STATUS_SPEC.loader is not None
STATUS_SPEC.loader.exec_module(STATUS_MODULE)


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


def test_events_append_json_lines(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=1)
    store.begin_running(message="started")
    events_path = tmp_path / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "initialized"
    second = json.loads(lines[1])
    assert second["event"] == "state_changed"
    assert second["state"] == "running"


def test_completed_stage_is_skipped_on_resume(tmp_path: Path) -> None:
    store = RunProgressStore(tmp_path, profile="qwen25_coder_7b.json", run_id="run-1")
    store.initialize(stage="baseline", total_units=1)
    store.begin_running()
    store.complete_stage()
    assert store.should_skip_stage("baseline")
    assert not store.should_skip_stage("interventions")


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


def test_qwen_fetch_rejects_traversal_and_extracts_nothing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"{}"
        info = tarfile.TarInfo(name="/results/qwen25_coder_7b/evil.json")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError):
        extract_qwen_archive(archive, project)
    assert not (project / "results").exists()


def test_qwen_fetch_rejects_oversized_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "big.tar.gz"
    payload = b"x" * 32
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(name="results/qwen25_coder_7b/big.json")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    import refusal_direction_replication.qwen_colab_transport as transport

    monkeypatch.setattr(transport, "MAX_MEMBER_BYTES", 16)
    with pytest.raises(ValueError, match="member"):
        extract_qwen_archive(archive, project)


def test_qwen_fetch_all_or_nothing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "mixed.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        good = b'{"ok": true}'
        good_info = tarfile.TarInfo(name="results/qwen25_coder_7b/ok.json")
        good_info.size = len(good)
        handle.addfile(good_info, io.BytesIO(good))
        bad_info = tarfile.TarInfo(name="results/final_evaluation.json")
        bad_info.size = 4
        handle.addfile(bad_info, io.BytesIO(b"gemm"))
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        extract_qwen_archive(archive, project)
    assert not (project / "results" / "qwen25_coder_7b").exists()


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
    assert command == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "refusal-qwen",
        "-f",
        str(ROOT / "scripts" / "qwen_run_status.py"),
        "--timeout",
        "120",
    ]


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


def test_qwen_status_cli_reports_missing_with_nonzero_exit(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    code = STATUS_MODULE.main_with_root(tmp_path, profile="qwen25_coder_7b.json")
    assert code != 0


def test_is_allowed_fetch_member_accepts_results_json() -> None:
    assert is_allowed_fetch_member("results/qwen25_coder_7b/a.json")
    assert not is_allowed_fetch_member("results/qwen25_coder_7b/nested/a.json")
    assert not is_allowed_fetch_member("artifacts/gemma/status.json")


def test_collect_qwen_pack_entries_matches_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "project"
    path = root / "artifacts" / "qwen25_coder_7b" / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    entries = collect_qwen_pack_entries(root)
    assert entries == [path]
    assert str(QWEN_FETCH_ALLOWLIST)