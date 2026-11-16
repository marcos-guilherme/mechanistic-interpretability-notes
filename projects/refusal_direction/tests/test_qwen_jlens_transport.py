from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    MAX_JLENS_JSON_BYTES,
    MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES,
    MAX_MEMBER_BYTES,
    REMOTE_JLENS_CHECKPOINT_STAGING,
    REMOTE_JLENS_EVENTS_STAGING,
    REMOTE_JLENS_STATUS_STAGING,
    atomic_install_jlens_checkpoint,
    atomic_install_jlens_resume,
    collect_qwen_jlens_pack_entries,
    extract_qwen_jlens_archive,
    is_allowed_jlens_fetch_member,
    jlens_member_byte_limit,
    optional_local_jlens_checkpoint,
    optional_local_jlens_resume_bundle,
    pack_qwen_jlens_results,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
JLENS_STATUS = ROOT / "scripts" / "qwen_run_jlens_status.py"
JLENS_RUN = ROOT / "scripts" / "qwen_run_jlens.py"
JLENS_PACK = ROOT / "scripts" / "colab_pack_qwen_jlens_results.py"
JLENS_INSTALL = ROOT / "scripts" / "colab_install_jlens.py"
JLENS_CHECKPOINT_INSTALL = ROOT / "scripts" / "colab_install_jlens_checkpoint.py"


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
    assert is_allowed_jlens_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/jlens_resume_bundle.json"
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
    assert scripts[0] == ROOT / "scripts" / "colab_bootstrap.py"
    assert JLENS_INSTALL in scripts
    assert any(path.name == "colab_install_jlens_checkpoint.py" for path in scripts)


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
    for path in (JLENS_RUN, JLENS_STATUS, JLENS_PACK, JLENS_INSTALL, JLENS_CHECKPOINT_INSTALL):
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
    assert REMOTE_JLENS_CHECKPOINT_STAGING not in uploaded


def test_jlens_member_limits_are_stricter_for_json() -> None:
    assert jlens_member_byte_limit("results/qwen25_coder_7b_jlens/00_jlens_prepare.json") == MAX_JLENS_JSON_BYTES
    assert MAX_JLENS_JSON_BYTES < MAX_MEMBER_BYTES
    assert jlens_member_byte_limit(
        "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    ) == MAX_MEMBER_BYTES


def test_extract_revalidates_json_and_rejects_oversized_json(tmp_path: Path) -> None:
    leaky = json.dumps(_compact(results={"fit_text": "def secret(): pass"})).encode()
    archive = tmp_path / "leaky.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("results/qwen25_coder_7b_jlens/00_jlens_prepare.json")
        info.size = len(leaky)
        handle.addfile(info, io.BytesIO(leaky))
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="leak"):
        extract_qwen_jlens_archive(archive, dest)

    payload = json.dumps(_compact()).encode()
    huge = tmp_path / "huge.tar.gz"
    with tarfile.open(huge, "w:gz") as handle:
        info = tarfile.TarInfo("results/qwen25_coder_7b_jlens/00_jlens_prepare.json")
        info.size = MAX_JLENS_JSON_BYTES + 1
        handle.addfile(info, io.BytesIO(payload + b" " * (MAX_JLENS_JSON_BYTES + 1 - len(payload))))
    with pytest.raises(ValueError, match="size"):
        extract_qwen_jlens_archive(huge, dest)


def test_optional_checkpoint_upload_and_validated_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    project = tmp_path / "refusal_direction"
    artifacts = project / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    checkpoint = artifacts / "jlens_fit_checkpoint.pt"
    torch.save(
        {
            "jacobian_sum": {0: torch.eye(4), 1: torch.eye(4), 2: torch.eye(4)},
            "n_done": 8,
            "next_idx": 8,
            "source_layers": [0, 1, 2],
            "target_layer": 3,
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": "c" * 40,
            "implementation_commit": "a" * 40,
            "run_id": "jlens-run-1",
            "dim_batch": 16,
        },
        checkpoint,
    )
    (artifacts / "status.json").write_text(
        json.dumps(
            {
                "run_id": "jlens-run-1",
                "state": "running",
                "current_stage": "fit",
                "completed_units": 8,
                "total_units": 100,
                "fit_completed_units": 8,
                "fit_total_units": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps({"event": "checkpoint", "run_id": "jlens-run-1", "n_done": 8}) + "\n",
        encoding="utf-8",
    )
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
    assert REMOTE_JLENS_CHECKPOINT_STAGING in uploaded
    assert REMOTE_JLENS_STATUS_STAGING in uploaded
    assert REMOTE_JLENS_EVENTS_STAGING in uploaded
    assert str(checkpoint) in uploaded
    archive = tmp_path / ".colab-staging" / "refusal_direction.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        names = [name.replace("\\", "/") for name in handle.getnames()]
    assert not any("jlens_fit_checkpoint.pt" in name for name in names)

    fresh = tmp_path / "fresh"
    staged = tmp_path / "staged.pt"
    staged.write_bytes(checkpoint.read_bytes())
    installed = atomic_install_jlens_checkpoint(staged=staged, project=fresh)
    assert installed == fresh / "artifacts" / "qwen25_coder_7b_jlens" / "jlens_fit_checkpoint.pt"
    assert installed.is_file()
    assert optional_local_jlens_checkpoint(fresh) == installed

    scripts = RUN_REFUSAL.bootstrap_scripts("qwen25-coder-jlens")
    assert any(path.name == "colab_install_jlens_checkpoint.py" for path in scripts)


def test_checkpoint_install_rejects_corrupt_payload(tmp_path: Path) -> None:
    staged = tmp_path / "bad.pt"
    staged.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        atomic_install_jlens_checkpoint(staged=staged, project=tmp_path / "proj")


def test_checkpoint_install_streams_bytes() -> None:
    import inspect

    source = inspect.getsource(atomic_install_jlens_checkpoint)
    assert "read_bytes()" not in source
    assert "copyfileobj" in source


def test_resume_bundle_requires_checkpoint_status_and_events(tmp_path: Path) -> None:
    import torch

    project = tmp_path / "project"
    artifacts = project / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    torch.save(
        {
            "jacobian_sum": {0: torch.eye(4)},
            "n_done": 1,
            "next_idx": 1,
            "source_layers": [0],
            "target_layer": 1,
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": "c" * 40,
            "implementation_commit": "a" * 40,
            "run_id": "jlens-run-1",
            "dim_batch": 16,
        },
        artifacts / "jlens_fit_checkpoint.pt",
    )
    with pytest.raises(ValueError, match="status.json"):
        optional_local_jlens_resume_bundle(project)


def _write_resume_members(
    directory: Path,
    *,
    run_id: str = "jlens-run-1",
    n_done: int = 8,
    dim_batch: int = 16,
) -> tuple[Path, Path, Path]:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "jlens_fit_checkpoint.pt"
    torch.save(
        {
            "jacobian_sum": {0: torch.eye(4), 1: torch.eye(4), 2: torch.eye(4)},
            "n_done": n_done,
            "next_idx": n_done,
            "source_layers": [0, 1, 2],
            "target_layer": 3,
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": "c" * 40,
            "implementation_commit": "a" * 40,
            "run_id": run_id,
            "dim_batch": dim_batch,
        },
        checkpoint,
    )
    status = directory / "status.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "qwen25_coder_7b_jlens.json",
                "run_id": run_id,
                "current_stage": "fit",
                "state": "running",
                "completed_units": n_done,
                "total_units": 100,
                "fit_completed_units": n_done,
                "fit_total_units": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = directory / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "checkpoint",
                "run_id": run_id,
                "n_done": n_done,
                "stage": "fit",
                "state": "running",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, status, events


def test_resume_bundle_rejects_mismatched_run_and_progress(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        validate_jlens_resume_bundle,
    )

    checkpoint, status, events = _write_resume_members(tmp_path / "ok")
    validate_jlens_resume_bundle(checkpoint=checkpoint, status=status, events=events)
    other_status = tmp_path / "other_status.json"
    other_status.write_text(
        json.dumps(
            {
                "run_id": "other-run",
                "current_stage": "fit",
                "completed_units": 8,
                "fit_completed_units": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run"):
        validate_jlens_resume_bundle(
            checkpoint=checkpoint, status=other_status, events=events
        )
    progress_status = tmp_path / "progress.json"
    progress_status.write_text(
        json.dumps(
            {
                "run_id": "jlens-run-1",
                "current_stage": "fit",
                "completed_units": 3,
                "fit_completed_units": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="progress"):
        validate_jlens_resume_bundle(
            checkpoint=checkpoint, status=progress_status, events=events
        )


def test_resume_install_rolls_back_writes_marker_and_consumes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from refusal_direction_replication import qwen_colab_transport as transport

    project = tmp_path / "project"
    old_ckpt, old_status, old_events = _write_resume_members(
        project / "artifacts" / "qwen25_coder_7b_jlens",
        run_id="old-run",
        n_done=1,
    )
    staged_dir = tmp_path / "staging"
    staged_ckpt, staged_status, staged_events = _write_resume_members(
        staged_dir, run_id="new-run", n_done=8
    )

    original = transport._stream_install_regular_file

    def boom(**kwargs):
        if kwargs["destination"].name == "events.jsonl":
            raise OSError("disk full")
        return original(**kwargs)

    monkeypatch.setattr(transport, "_stream_install_regular_file", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_install_jlens_resume(
            checkpoint=staged_ckpt,
            status=staged_status,
            events=staged_events,
            project=project,
        )
    restored = torch.load(old_ckpt, map_location="cpu", weights_only=True)
    assert restored["run_id"] == "old-run"
    assert restored["n_done"] == 1
    assert json.loads(old_status.read_text(encoding="utf-8"))["run_id"] == "old-run"
    assert "old-run" in old_events.read_text(encoding="utf-8")

    monkeypatch.setattr(transport, "_stream_install_regular_file", original)
    staged_ckpt, staged_status, staged_events = _write_resume_members(
        staged_dir, run_id="new-run", n_done=8
    )
    installed = atomic_install_jlens_resume(
        checkpoint=staged_ckpt,
        status=staged_status,
        events=staged_events,
        project=project,
    )
    assert installed is not None
    marker = project / "artifacts" / "qwen25_coder_7b_jlens" / "jlens_resume_bundle.json"
    assert marker.is_file()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["run_id"] == "new-run"
    assert payload["n_done"] == 8
    assert not staged_ckpt.exists()
    assert not staged_status.exists()
    assert not staged_events.exists()


def test_status_reader_shows_fit_progress(tmp_path: Path) -> None:
    import shutil

    spec = importlib.util.spec_from_file_location("qwen_run_jlens_status", JLENS_STATUS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    (artifacts / "status.json").write_text(
        json.dumps(
            {
                "profile": "qwen25_coder_7b_jlens.json",
                "run_id": "jlens-run-1",
                "current_stage": "holdout",
                "state": "completed",
                "completed_units": 20,
                "total_units": 20,
                "fit_completed_units": 100,
                "fit_total_units": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert module.main_with_root(tmp_path) == 0
    assert "progress=20/20" in module.last_stdout_summary
    assert "fit_progress=100/100" in module.last_stdout_summary
