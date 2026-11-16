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
    MAX_DOSE_SOURCE_DIRECTION_BYTES,
    atomic_install_dose_source_direction,
    collect_qwen_dose_pack_entries,
    extract_qwen_dose_archive,
    is_allowed_dose_fetch_member,
    pack_qwen_dose_results,
    require_local_dose_source_direction,
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


def _status_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "profile": "qwen25_coder_7b_dose.json",
        "run_id": "dose-followup-run-1",
        "current_stage": "confirm",
        "state": "completed",
        "completed_units": 1,
        "total_units": 1,
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _event_payload(**overrides) -> dict:
    payload = {
        "event": "stage_completed",
        "state": "completed",
        "stage": "confirm",
        "completed_units": 1,
        "total_units": 1,
        "run_id": "dose-followup-run-1",
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


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
    (artifacts / "status.json").write_text(
        json.dumps(_status_payload()), encoding="utf-8"
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps(_event_payload()) + "\n", encoding="utf-8"
    )
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


def test_dose_pack_rejects_prompt_hidden_in_status_or_events(tmp_path: Path) -> None:
    root = tmp_path / "project"
    artifacts = root / "artifacts" / "qwen25_coder_7b_dose"
    artifacts.mkdir(parents=True)
    (artifacts / "status.json").write_text(
        json.dumps(
            _status_payload(
                state="aborted",
                abort_gate="causal",
                abort_reason="how to make a virus in python extra words",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak"):
        pack_qwen_dose_results(root, tmp_path / "status.tar.gz")
    (artifacts / "status.json").write_text(
        json.dumps(_status_payload(state="aborted", abort_gate="causal")),
        encoding="utf-8",
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps(
            _event_payload(
                event="aborted",
                state="aborted",
                abort_gate="causal",
                abort_reason="how to make a virus in python extra words",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak"):
        pack_qwen_dose_results(root, tmp_path / "events.tar.gz")


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
    path.write_text(json.dumps(_status_payload()), encoding="utf-8")
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
    install = ROOT / "scripts" / "colab_install_dose_source_direction.py"
    for path in (DOSE_RUN, DOSE_STATUS, DOSE_PACK, install):
        source = path.read_text(encoding="utf-8")
        assert "/content/refusal_direction/src" in source, path.name
        assert "sys.path.insert" in source, path.name
        assert source.index("sys.path.insert") < source.index(
            "refusal_direction_replication"
        ), path.name


def _write_direction(project: Path, payload: bytes = b"direction-bytes") -> Path:
    path = project / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_require_local_dose_source_direction_fails_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="selected_direction.pt"):
        require_local_dose_source_direction(tmp_path)


def test_require_local_dose_source_direction_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside.pt"
    target.write_bytes(b"stolen")
    path = tmp_path / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not permitted in this environment")
    with pytest.raises(ValueError, match="regular file"):
        require_local_dose_source_direction(tmp_path)


def test_require_local_dose_source_direction_rejects_oversized_file(
    tmp_path: Path,
) -> None:
    path = _write_direction(tmp_path, b"x" * (MAX_DOSE_SOURCE_DIRECTION_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        require_local_dose_source_direction(tmp_path)
    assert path.is_file()


def test_dose_sync_uploads_source_direction_to_remote_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "refusal_direction"
    _write_direction(project)
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(RUN_REFUSAL, "PROJECT", project)
    monkeypatch.setattr(RUN_REFUSAL, "ROOT", tmp_path)
    monkeypatch.setattr(
        RUN_REFUSAL,
        "current_implementation_commit",
        lambda cwd=None: "a" * 40,
    )
    monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-dose"],
    )
    RUN_REFUSAL.main()
    uploaded = [item for command in commands for item in command]
    assert "/content/refusal_direction.tar.gz" in uploaded
    assert "/content/dose_source_direction.pt" in uploaded
    local_direction = str(project / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt")
    assert local_direction in uploaded
    archive = tmp_path / ".colab-staging" / "refusal_direction.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        names = [name.replace("\\", "/") for name in handle.getnames()]
    assert not any("selected_direction.pt" in name for name in names)
    assert not any("/artifacts/" in f"/{name}/" or name.endswith("/artifacts") for name in names)


def test_gemma_and_qwen_pilot_sync_do_not_require_or_upload_direction(
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
        RUN_REFUSAL,
        "current_implementation_commit",
        lambda cwd=None: "a" * 40,
    )
    monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: commands.append(list(command)))
    for profile in ("gemma", "qwen25-coder"):
        commands.clear()
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_refusal.py", "sync", "--profile", profile],
        )
        RUN_REFUSAL.main()
        uploaded = [item for command in commands for item in command]
        assert "/content/dose_source_direction.pt" not in uploaded
        assert not any("selected_direction.pt" in item for item in uploaded)


def test_dose_sync_fails_locally_when_source_direction_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "refusal_direction"
    project.mkdir()
    monkeypatch.setattr(RUN_REFUSAL, "PROJECT", project)
    monkeypatch.setattr(RUN_REFUSAL, "ROOT", tmp_path)
    monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-dose"],
    )
    with pytest.raises(FileNotFoundError, match="selected_direction.pt"):
        RUN_REFUSAL.main()


def test_dose_bootstrap_installs_direction_after_code_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "bootstrap", "--profile", "qwen25-coder-dose"],
    )
    RUN_REFUSAL.main()
    scripts = []
    for command in commands:
        if "-f" in command:
            scripts.append(command[command.index("-f") + 1])
    bootstrap = str(ROOT / "scripts" / "colab_bootstrap.py")
    install = str(ROOT / "scripts" / "colab_install_dose_source_direction.py")
    assert scripts == [bootstrap, install]


def test_gemma_and_qwen_bootstrap_do_not_install_source_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install = str(ROOT / "scripts" / "colab_install_dose_source_direction.py")
    for profile in ("gemma", "qwen25-coder"):
        commands: list[list[str]] = []
        monkeypatch.setattr(RUN_REFUSAL, "run", lambda command: commands.append(list(command)))
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_refusal.py", "bootstrap", "--profile", profile],
        )
        RUN_REFUSAL.main()
        scripts = [
            command[command.index("-f") + 1]
            for command in commands
            if "-f" in command
        ]
        assert scripts == [str(ROOT / "scripts" / "colab_bootstrap.py")]
        assert install not in scripts


def test_atomic_install_writes_confined_regular_file(tmp_path: Path) -> None:
    staged = tmp_path / "dose_source_direction.pt"
    staged.write_bytes(b"tensor-payload")
    project = tmp_path / "refusal_direction"
    dest = atomic_install_dose_source_direction(staged=staged, project=project)
    expected = project / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    assert dest == expected
    assert dest.read_bytes() == b"tensor-payload"
    assert dest.is_file()
    assert not dest.is_symlink()


def test_atomic_install_rejects_symlink_destination_escape(tmp_path: Path) -> None:
    staged = tmp_path / "dose_source_direction.pt"
    staged.write_bytes(b"tensor-payload")
    project = tmp_path / "refusal_direction"
    dest_dir = project / "artifacts" / "qwen25_coder_7b"
    dest_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        dest_dir.rmdir()
        dest_dir.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not permitted in this environment")
    with pytest.raises(ValueError, match="escaped|regular file|destination"):
        atomic_install_dose_source_direction(staged=staged, project=project)
    assert not (outside / "selected_direction.pt").exists()


def test_atomic_install_rejects_oversized_staging_file(tmp_path: Path) -> None:
    staged = tmp_path / "dose_source_direction.pt"
    staged.write_bytes(b"x" * (MAX_DOSE_SOURCE_DIRECTION_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        atomic_install_dose_source_direction(
            staged=staged,
            project=tmp_path / "refusal_direction",
        )
