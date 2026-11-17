from pathlib import Path
import importlib.util
import tarfile

import pytest


SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
SPEC = importlib.util.spec_from_file_location("run_refusal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_start_command_names_session_and_gpu() -> None:
    assert MODULE.start_command("refusal-gemma", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-gemma",
        "--gpu",
        "L4",
    ]


def test_stop_command_is_always_explicit() -> None:
    assert MODULE.stop_command("refusal-gemma") == [
        "colab",
        "--auth=adc",
        "stop",
        "-s",
        "refusal-gemma",
    ]


def test_exec_command_sets_remote_timeout() -> None:
    assert MODULE.exec_command(
        "refusal-gemma",
        Path("stage.py"),
        timeout=300,
    ) == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "refusal-gemma",
        "-f",
        "stage.py",
        "--timeout",
        "300",
    ]


def test_huggingface_token_uses_standard_cache_path(tmp_path) -> None:
    assert MODULE.huggingface_token_path(tmp_path) == (
        tmp_path / ".cache" / "huggingface" / "token"
    )


def _tiny_project(root) -> Path:
    project = root / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    return project


def test_gemma_archive_does_not_stamp_implementation_commit(tmp_path) -> None:
    project = _tiny_project(tmp_path)
    archive = MODULE.build_archive(
        stamp_commit=False,
        project=project,
        staging=tmp_path / "staging",
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/IMPLEMENTATION_COMMIT" not in names
    assert "refusal_direction/hello.py" in names


def test_archive_excludes_artifacts_even_when_source_direction_exists(
    tmp_path,
) -> None:
    project = _tiny_project(tmp_path)
    direction = project / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    direction.parent.mkdir(parents=True)
    direction.write_bytes(b"tensor")
    (project / "artifacts" / "other.pt").write_bytes(b"nope")
    archive = MODULE.build_archive(
        stamp_commit=False,
        project=project,
        staging=tmp_path / "staging",
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/hello.py" in names
    assert not any("artifacts" in name.replace("\\", "/") for name in names)
    assert not any(name.endswith("selected_direction.pt") for name in names)


def test_qwen_archive_stamps_exact_implementation_commit(tmp_path) -> None:
    project = _tiny_project(tmp_path)
    commit = "e" * 40
    archive = MODULE.build_archive(
        stamp_commit=True,
        project=project,
        staging=tmp_path / "staging",
        commit=commit,
    )
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.extractfile("refusal_direction/IMPLEMENTATION_COMMIT")
        assert member is not None
        assert member.read().decode("utf-8").strip() == commit


def test_lab_profile_session_and_bootstrap_are_dedicated() -> None:
    assert MODULE.PROFILE_SESSIONS["qwen25-coder-lab"] == "refusal-qwen-lab"
    assert MODULE.resolve_session("qwen25-coder-lab", None) == "refusal-qwen-lab"
    scripts = MODULE.bootstrap_scripts("qwen25-coder-lab")
    install = Path(__file__).parents[1] / "scripts" / "colab_install_lab.py"
    status = Path(__file__).parents[1] / "scripts" / "qwen_run_lab_status.py"
    assert install in scripts
    assert status == Path(
        MODULE.status_command("refusal-qwen-lab", profile="qwen25-coder-lab")[
            MODULE.status_command(
                "refusal-qwen-lab", profile="qwen25-coder-lab"
            ).index("-f")
            + 1
        ]
    )


def test_lab_start_forces_a100_without_changing_other_profiles(monkeypatch) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "qwen25-coder-lab", "--gpu", "T4"],
    )
    MODULE.main()
    assert commands[-1][-1] == "A100"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-qwen-lab"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "gemma", "--gpu", "L4"],
    )
    MODULE.main()
    assert commands[-1][-1] == "L4"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-gemma"


def test_lab_fetch_is_refused_and_does_not_download(monkeypatch) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "fetch", "--profile", "qwen25-coder-lab"],
    )
    with pytest.raises(SystemExit):
        MODULE.main()
    assert commands == []


def test_legacy_qwen_fetch_still_downloads(monkeypatch, tmp_path) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "refusal_direction")
    monkeypatch.setattr(MODULE, "extract_qwen_results_archive", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "fetch", "--profile", "qwen25-coder"],
    )
    MODULE.main()
    joined = " ".join(" ".join(item) for item in commands)
    assert "download" in joined
    assert "qwen_results.tar.gz" in joined


# --- jlens v2 profile ------------------------------------------------------

V2 = "qwen25-coder-jlens-v2"
V2_SESSION = "refusal-qwen-jlens-v2"
V2_REMOTE_ARTIFACTS = "/content/refusal_direction/artifacts/qwen25_coder_7b_jlens_v2"
SCRIPTS = Path(__file__).parents[1] / "scripts"


def _record_runs(monkeypatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    return commands


def _main(monkeypatch, *argv: str) -> int:
    import sys

    monkeypatch.setattr(sys, "argv", ["run_refusal.py", *argv])
    return MODULE.main()


def _exec_script(command: list[str]) -> Path:
    return Path(command[command.index("-f") + 1])


def _timeout(command: list[str]) -> str:
    return command[command.index("--timeout") + 1]


def _v2_remote_triple(directory: Path, *, n_done: int, run_id: str = "v2-run"):
    import json

    import torch

    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "jacobian_sum": {0: torch.eye(2)},
            "n_done": n_done,
            "next_idx": n_done,
            "source_layers": [0],
            "target_layer": 1,
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": "c" * 40,
            "implementation_commit": "a" * 40,
            "run_id": run_id,
            "dim_batch": 16,
            "fit_count": 200,
        },
        directory / "jlens_fit_checkpoint.pt",
    )
    (directory / "status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "state": "running",
                "fit_completed_units": n_done,
                "fit_total_units": 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "events.jsonl").write_text(
        json.dumps({"event": "checkpoint", "run_id": run_id, "n_done": n_done}) + "\n",
        encoding="utf-8",
    )
    return directory


def _fake_file_api(monkeypatch, remote: Path, *, fail_on: str | None = None):
    """Serve remote v2 artifacts through a fake ``colab download``."""

    calls: list[tuple[list[str], float]] = []

    def download(command: list[str], *, timeout: float) -> bool:
        calls.append((list(command), timeout))
        assert command[2] == "download"
        name = Path(command[-2]).name
        local = Path(command[-1])
        if name == fail_on:
            local.write_bytes(b"partial")
            raise RuntimeError("connection reset")
        source = remote / name
        if not source.is_file():
            return False
        local.write_bytes(source.read_bytes())
        return True

    monkeypatch.setattr(MODULE, "download_optional", download)
    return calls


def _local_v2(project: Path) -> Path:
    return project / "artifacts" / "qwen25_coder_7b_jlens_v2"


def test_v2_start_forces_a100(monkeypatch) -> None:
    commands = _record_runs(monkeypatch)
    _main(monkeypatch, "start", "--profile", V2, "--gpu", "T4")
    assert commands[-1][-1] == "A100"
    assert commands[-1][commands[-1].index("-s") + 1] == V2_SESSION


def test_v2_sync_uploads_only_v2_staging_names(monkeypatch, tmp_path) -> None:
    commands = _record_runs(monkeypatch)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    archive = tmp_path / "refusal_direction.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(MODULE, "build_archive", lambda **_kwargs: archive)
    lens = tmp_path / "v1_lens.pt"
    lens.write_bytes(b"0123456789")
    monkeypatch.setattr(MODULE, "require_jlens_v2_source_v1_lens", lambda _p: lens)
    monkeypatch.setattr(MODULE, "JLENS_V2_LENS_CHUNK_BYTES", 4)
    resume = [
        (tmp_path / "c.pt", "/content/jlens_v2_fit_checkpoint.pt"),
        (tmp_path / "s.json", "/content/jlens_v2_status.json"),
        (tmp_path / "e.jsonl", "/content/jlens_v2_events.jsonl"),
    ]
    monkeypatch.setattr(MODULE, "optional_jlens_v2_resume_bundle", lambda _p: resume)

    _main(monkeypatch, "sync", "--profile", V2)

    assert all(command[2] == "upload" for command in commands)
    remotes = [command[-1] for command in commands]
    assert remotes == [
        "/content/refusal_direction.tar.gz",
        "/content/jlens_v2_source_v1_lens.pt.manifest.json",
        "/content/jlens_v2_source_v1_lens.pt.part000",
        "/content/jlens_v2_source_v1_lens.pt.part001",
        "/content/jlens_v2_source_v1_lens.pt.part002",
        "/content/jlens_v2_fit_checkpoint.pt",
        "/content/jlens_v2_status.json",
        "/content/jlens_v2_events.jsonl",
    ]
    assert not any("qwen25_coder_7b_jlens" in remote for remote in remotes)
    assert not any(remote.startswith("/content/refusal_direction/") for remote in remotes)


def test_v2_bootstrap_installs_pinned_upstream_then_v2_sources() -> None:
    assert MODULE.bootstrap_scripts(V2) == [
        SCRIPTS / "colab_bootstrap.py",
        SCRIPTS / "colab_install_jlens.py",
        SCRIPTS / "colab_install_jlens_v2_checkpoint.py",
    ]


def test_v2_jlens_and_status_use_v2_scripts_and_long_timeout(monkeypatch) -> None:
    commands = _record_runs(monkeypatch)
    _main(monkeypatch, "jlens", "--profile", V2)
    _main(monkeypatch, "jlens", "--profile", V2, "--timeout", "900")
    _main(monkeypatch, "status", "--profile", V2)

    assert _exec_script(commands[0]) == SCRIPTS / "qwen_run_jlens_v2.py"
    assert _timeout(commands[0]) == "46800"
    assert _timeout(commands[1]) == "900"
    assert _exec_script(commands[2]) == SCRIPTS / "qwen_run_jlens_v2_status.py"
    assert all(command[command.index("-s") + 1] == V2_SESSION for command in commands)


def test_v2_checkpoint_backup_downloads_status_events_then_checkpoint(
    monkeypatch, tmp_path
) -> None:
    commands = _record_runs(monkeypatch)
    project = tmp_path / "refusal_direction"
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", project)
    remote = _v2_remote_triple(tmp_path / "remote", n_done=7)
    calls = _fake_file_api(monkeypatch, remote)

    assert _main(monkeypatch, "checkpoint-backup", "--profile", V2) == 0

    assert commands == []  # no exec, no pack, no stop
    assert [call[0][-2] for call in calls] == [
        f"{V2_REMOTE_ARTIFACTS}/status.json",
        f"{V2_REMOTE_ARTIFACTS}/events.jsonl",
        f"{V2_REMOTE_ARTIFACTS}/jlens_fit_checkpoint.pt",
    ]
    staging = {Path(call[0][-1]).parent for call in calls}
    assert len(staging) == 1 and not staging.pop().exists()
    local = _local_v2(project)
    for name in ("jlens_fit_checkpoint.pt", "status.json", "events.jsonl"):
        assert (local / name).read_bytes() == (remote / name).read_bytes()


def test_v2_checkpoint_backup_is_clean_noop_without_checkpoint(
    monkeypatch, tmp_path
) -> None:
    project = tmp_path / "refusal_direction"
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", project)
    remote = _v2_remote_triple(tmp_path / "remote", n_done=0)
    (remote / "jlens_fit_checkpoint.pt").unlink()
    _fake_file_api(monkeypatch, remote)

    assert _main(monkeypatch, "checkpoint-backup", "--profile", V2) == 0
    assert not _local_v2(project).exists()


@pytest.mark.parametrize("failure", ["download", "invalid"])
def test_v2_checkpoint_backup_failure_preserves_previous_bundle(
    monkeypatch, tmp_path, failure: str
) -> None:
    project = tmp_path / "refusal_direction"
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", project)
    local = _v2_remote_triple(_local_v2(project), n_done=3, run_id="old")
    before = {path.name: path.read_bytes() for path in local.iterdir()}
    remote = _v2_remote_triple(tmp_path / "remote", n_done=9, run_id="new")
    if failure == "invalid":
        (remote / "status.json").write_text(
            '{"run_id": "other", "fit_completed_units": 9}\n', encoding="utf-8"
        )
    _fake_file_api(
        monkeypatch,
        remote,
        fail_on="jlens_fit_checkpoint.pt" if failure == "download" else None,
    )

    with pytest.raises(Exception):
        _main(monkeypatch, "checkpoint-backup", "--profile", V2)

    assert {path.name: path.read_bytes() for path in local.iterdir()} == before
    assert not any((tmp_path / ".colab-staging").glob("jlens-v2-backup-*"))


def test_v2_stop_backs_up_through_file_api_before_stopping(
    monkeypatch, tmp_path
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        MODULE, "run", lambda command: order.append(command[2])
    )
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "refusal_direction")
    remote = _v2_remote_triple(tmp_path / "remote", n_done=5)
    calls = _fake_file_api(monkeypatch, remote)

    def download(command, *, timeout):
        order.append("download")
        return calls_download(command, timeout=timeout)

    calls_download = MODULE.download_optional
    monkeypatch.setattr(MODULE, "download_optional", download)

    assert _main(monkeypatch, "stop", "--profile", V2) == 0
    assert order == ["download", "download", "download", "stop"]
    assert all(timeout <= MODULE.STOP_BACKUP_TIMEOUT_SECONDS for _, timeout in calls)


def test_v2_stop_still_stops_and_reports_when_backup_fails(
    monkeypatch, tmp_path, capsys
) -> None:
    commands = _record_runs(monkeypatch)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    def unreachable(*_args, **_kwargs):
        raise TimeoutError("file API timed out")

    monkeypatch.setattr(MODULE, "backup_jlens_v2_checkpoint", unreachable)

    assert _main(monkeypatch, "stop", "--profile", V2) == 1
    assert commands == [MODULE.stop_command(V2_SESSION)]
    assert "backup failed" in capsys.readouterr().err


def test_v2_checkpoint_backup_passes_explicit_zero_timeout(
    monkeypatch, tmp_path
) -> None:
    captured: list[float] = []

    def record_backup(*_args, **kwargs):
        captured.append(kwargs["timeout"])
        return None

    monkeypatch.setattr(MODULE, "backup_jlens_v2_checkpoint", record_backup)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "refusal_direction")

    assert _main(monkeypatch, "checkpoint-backup", "--profile", V2, "--timeout", "0") == 0
    assert captured == [0]


def test_v2_stop_retries_stop_until_success(monkeypatch, tmp_path) -> None:
    commands: list[list[str]] = []
    attempts = {"n": 0}

    def flaky_run(command: list[str]) -> None:
        commands.append(list(command))
        if command[2] != "stop":
            return
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("colab stop transient")

    import time

    monkeypatch.setattr(MODULE, "run", flaky_run)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    remote = _v2_remote_triple(tmp_path / "remote", n_done=2)
    _fake_file_api(monkeypatch, remote)

    assert _main(monkeypatch, "stop", "--profile", V2) == 0
    stop_commands = [c for c in commands if c[2] == "stop"]
    assert stop_commands == [MODULE.stop_command(V2_SESSION)] * 3


def test_v2_stop_alerts_when_stop_never_succeeds(
    monkeypatch, tmp_path, capsys
) -> None:
    def always_fail(command: list[str]) -> None:
        if command[2] == "stop":
            raise RuntimeError("colab stop down")

    import time

    monkeypatch.setattr(MODULE, "run", always_fail)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    remote = _v2_remote_triple(tmp_path / "remote", n_done=2)
    _fake_file_api(monkeypatch, remote)

    assert _main(monkeypatch, "stop", "--profile", V2) == 1
    err = capsys.readouterr().err
    assert "A100 MAY STILL BE RUNNING" in err
    assert (
        "python tools/colab/run_refusal.py stop --profile qwen25-coder-jlens-v2" in err
    )


def test_v2_fetch_uses_v2_archive_and_extraction(monkeypatch, tmp_path) -> None:
    commands = _record_runs(monkeypatch)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    extracted: list[Path] = []
    monkeypatch.setattr(
        MODULE,
        "extract_qwen_jlens_v2_results_archive",
        lambda archive, _project: extracted.append(archive),
    )

    _main(monkeypatch, "fetch", "--profile", V2)

    assert _exec_script(commands[0]) == SCRIPTS / "colab_pack_qwen_jlens_v2_results.py"
    assert commands[1][2] == "download"
    assert commands[1][-2] == "/content/qwen_jlens_v2_results.tar.gz"
    assert extracted == [tmp_path / ".colab-staging" / "qwen_jlens_v2_results.tar.gz"]


def test_v1_and_lab_commands_are_unchanged_by_v2(monkeypatch) -> None:
    commands = _record_runs(monkeypatch)
    _main(monkeypatch, "jlens", "--profile", "qwen25-coder-jlens")
    _main(monkeypatch, "stop", "--profile", "qwen25-coder-jlens")
    _main(monkeypatch, "start", "--profile", "qwen25-coder-jlens", "--gpu", "L4")

    assert _exec_script(commands[0]) == SCRIPTS / "qwen_run_jlens.py"
    assert _timeout(commands[0]) == "7200"
    assert commands[1] == MODULE.stop_command("refusal-qwen")
    assert commands[2][-1] == "L4"
    assert MODULE.bootstrap_scripts("qwen25-coder-jlens") == [
        SCRIPTS / "colab_bootstrap.py",
        SCRIPTS / "colab_install_jlens.py",
        SCRIPTS / "colab_install_jlens_checkpoint.py",
    ]
    assert MODULE.bootstrap_scripts("qwen25-coder-lab") == [
        SCRIPTS / "colab_install_lab.py"
    ]
    assert MODULE.PROFILE_SESSIONS["qwen25-coder-lab"] == "refusal-qwen-lab"
