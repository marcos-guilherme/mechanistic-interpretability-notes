from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from refusal_direction_replication.qwen_dose import sha256_file
from refusal_direction_replication.qwen_lab_app import (
    FROZEN_DIRECTION_SHA256,
    GRADIO_PIN_VERSION,
)
from refusal_direction_replication.qwen_lab_readout import FROZEN_LENS_SHA256
from refusal_direction_replication.qwen_jlens import JLENS_UPSTREAM_COMMIT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_lab.py"
STATUS_SCRIPT = ROOT / "scripts" / "qwen_run_lab_status.py"
START_SCRIPT = ROOT / "scripts" / "colab_start_lab.py"
RUN_SCRIPT = ROOT / "scripts" / "qwen_run_lab.py"
LAB_CONFIG = ROOT / "configs" / "qwen25_coder_7b_lab.json"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def _orchestrator():
    spec = importlib.util.spec_from_file_location("run_refusal_lab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_lab_config_pins_model_artifacts_and_omits_datasets_and_fetch() -> None:
    assert LAB_CONFIG.is_file()
    config = json.loads(LAB_CONFIG.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["model_revision"] == "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    assert config["dtype"] == "bfloat16"
    assert config["gpu"] == "A100"
    assert config["session"] == "refusal-qwen-lab"
    assert config["gradio_version"] == GRADIO_PIN_VERSION == "5.49.1"
    assert config["jlens_upstream_commit"] == JLENS_COMMIT == JLENS_UPSTREAM_COMMIT
    assert config["source_direction"] == "artifacts/qwen25_coder_7b/selected_direction.pt"
    assert (
        config["source_lens"]
        == "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert config["direction_sha256"] == FROZEN_DIRECTION_SHA256
    assert config["lens_sha256"] == FROZEN_LENS_SHA256
    assert config["artifacts_dir"] == "artifacts/qwen25_coder_7b_lab"
    assert "results_dir" not in config
    assert "dataset_repo_id" not in config
    assert "dataset_revision" not in config
    assert "datasets" not in config
    blob = json.dumps(config)
    assert "mitre" not in blob
    assert "code_search_net" not in blob
    assert "humaneval" not in blob


def test_lab_profile_uses_dedicated_session_and_forces_a100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    assert module.resolve_session("qwen25-coder-lab", None) == "refusal-qwen-lab"
    assert module.PROFILE_SESSIONS["qwen25-coder"] == "refusal-qwen"
    assert module.PROFILE_SESSIONS["qwen25-coder-code-eval"] == "refusal-qwen"
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_refusal.py",
            "start",
            "--profile",
            "qwen25-coder-lab",
            "--gpu",
            "L4",
        ],
    )
    module.main()
    assert commands[-1] == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-qwen-lab",
        "--gpu",
        "A100",
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "qwen25-coder-dose", "--gpu", "L4"],
    )
    module.main()
    assert commands[-1][-1] == "L4"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-qwen"


def test_lab_status_uses_safe_script_and_never_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "status", "--profile", "qwen25-coder-lab"],
    )
    module.main()
    joined = " ".join(" ".join(item) for item in commands)
    assert "download" not in joined
    assert str(STATUS_SCRIPT) in commands[-1]
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-qwen-lab"


def test_lab_fetch_is_explicitly_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "fetch", "--profile", "qwen25-coder-lab"],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert commands == []


def test_chunk_roundtrip_preserves_hash_for_oversized_lens(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_colab_transport import split_artifact_file

    source = tmp_path / "lens.pt"
    payload = os.urandom(64 * 1024 * 2 + 91)
    source.write_bytes(payload)
    parts = split_artifact_file(
        source,
        tmp_path / "parts",
        chunk_bytes=64 * 1024,
    )
    assert len(parts) == 3
    rebuilt = b"".join(part.read_bytes() for part in parts)
    assert hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert all(part.name.startswith("lens.part") for part in parts)


def test_code_eval_chunk_helper_still_writes_local_lens_parts(tmp_path: Path) -> None:
    module = _orchestrator()
    source = tmp_path / "source.pt"
    source.write_bytes(b"abc" * 10)
    parts = module.split_code_eval_lens(source, tmp_path / "staging", chunk_bytes=8)
    assert [part.name for part in parts] == ["lens.part000", "lens.part001", "lens.part002", "lens.part003"]
    assert b"".join(part.read_bytes() for part in parts) == source.read_bytes()


def test_lab_sync_uploads_direction_and_lens_parts_without_tarring_lens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        REMOTE_LAB_DIRECTION_STAGING,
        REMOTE_LAB_LENS_STAGING,
    )

    module = _orchestrator()
    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = project / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    _write_bytes(direction, b"direction-bytes")
    _write_bytes(lens, os.urandom(128 * 1024 + 3))
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(module, "current_implementation_commit", lambda cwd=None: "b" * 40)
    monkeypatch.setattr(
        module,
        "require_lab_sources",
        lambda _project: (direction, lens),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-lab"],
    )
    module.main()
    uploaded = [item for command in commands for item in command]
    assert REMOTE_LAB_DIRECTION_STAGING in uploaded
    assert any(
        str(item).startswith(REMOTE_LAB_LENS_STAGING + ".part") for item in uploaded
    )
    archive = tmp_path / ".colab-staging" / "refusal_direction.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/hello.py" in names
    assert not any("jacobian_lens.pt" in name for name in names)
    assert not any("selected_direction.pt" in name for name in names)
    assert not any("artifacts" in name.replace("\\", "/") for name in names)


def test_require_local_lab_sources_rejects_link_wrong_hash_and_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_colab_transport as transport
    from refusal_direction_replication.qwen_colab_transport import require_local_lab_sources

    project = tmp_path / "project"
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = project / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    payload_dir = b"ok-direction"
    payload_lens = b"ok-lens"
    _write_bytes(direction, payload_dir)
    _write_bytes(lens, payload_lens)
    assert require_local_lab_sources(
        project,
        expected_direction_sha256=sha256_file(direction),
        expected_lens_sha256=sha256_file(lens),
    ) == (direction, lens)

    with pytest.raises(ValueError, match="sha256"):
        require_local_lab_sources(
            project,
            expected_direction_sha256="0" * 64,
            expected_lens_sha256=sha256_file(lens),
        )

    monkeypatch.setattr(transport, "path_is_link_or_junction", lambda path: path == direction)
    with pytest.raises(ValueError, match="regular"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
        )
    monkeypatch.setattr(transport, "path_is_link_or_junction", lambda path: False)

    monkeypatch.setattr(transport, "MAX_LAB_LENS_BYTES", 1)
    _write_bytes(lens, b"too-big")
    with pytest.raises(ValueError, match="size"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
        )


def test_atomic_lab_install_verifies_hashes_and_rolls_back(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        atomic_install_lab_sources,
        require_local_lab_sources,
    )

    project = tmp_path / "project"
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = project / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    _write_bytes(direction, b"old-direction")
    _write_bytes(lens, b"old-lens")
    staged_direction = tmp_path / "direction.pt"
    staged_lens = tmp_path / "lens.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_lens.write_bytes(b"new-lens")
    installed = atomic_install_lab_sources(
        staged_direction=staged_direction,
        staged_lens=staged_lens,
        project=project,
        expected_direction_sha256=sha256_file(staged_direction),
        expected_lens_sha256=sha256_file(staged_lens),
    )
    assert installed == (direction, lens)
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"

    staged_direction.write_bytes(b"rollback-direction")
    staged_lens.write_bytes(b"rollback-lens")
    with pytest.raises(ValueError, match="sha256"):
        atomic_install_lab_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            project=project,
            expected_direction_sha256=sha256_file(staged_direction),
            expected_lens_sha256="f" * 64,
        )
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"
    assert require_local_lab_sources(
        project,
        expected_direction_sha256=sha256_file(direction),
        expected_lens_sha256=sha256_file(lens),
    ) == (direction, lens)


def test_lab_installer_reassembles_parts_atomically_and_pins_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert INSTALL_SCRIPT.is_file()
    install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert GRADIO_PIN_VERSION in install_text
    assert "gradio==5.49.1" in install_text
    assert JLENS_COMMIT in install_text
    assert "--no-deps" in install_text
    assert "fetch_upstream_splits" not in install_text
    assert "/content/refusal_direction/src" in install_text

    module = _load_script(INSTALL_SCRIPT, "install_lab")
    destination = tmp_path / "lab_source_lens.pt"
    monkeypatch.setattr(module, "REMOTE_LAB_LENS_STAGING", str(destination))
    (tmp_path / "lab_source_lens.pt.part000").write_bytes(b"alpha")
    (tmp_path / "lab_source_lens.pt.part001").write_bytes(b"beta")
    assert module.assemble_staged_lens() == destination
    assert destination.read_bytes() == b"alphabeta"
    assert not list(tmp_path.glob("lab_source_lens.pt.part*"))


def test_lab_installer_rejects_noncontiguous_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(INSTALL_SCRIPT, "install_lab_gap")
    destination = tmp_path / "lab_source_lens.pt"
    monkeypatch.setattr(module, "REMOTE_LAB_LENS_STAGING", str(destination))
    (tmp_path / "lab_source_lens.pt.part000").write_bytes(b"alpha")
    (tmp_path / "lab_source_lens.pt.part002").write_bytes(b"gap")
    with pytest.raises(ValueError, match="contiguous"):
        module.assemble_staged_lens()
    assert not destination.exists()


def test_write_and_consume_auth_file_uses_restricted_mode_and_deletes(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        consume_lab_auth_file,
        write_lab_auth_file,
    )

    path = tmp_path / "qwen_lab_auth.json"
    write_lab_auth_file(path, username="alice", password="s3cret-value")
    assert path.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"username": "alice", "password": "s3cret-value"}
    user, password = consume_lab_auth_file(path)
    assert (user, password) == ("alice", "s3cret-value")
    assert not path.exists()


def test_lab_action_prints_credentials_only_locally_and_deletes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(list(command))

    def fake_run_captured(command: list[str]) -> str:
        commands.append(list(command))
        return "https://demo-lab.gradio.live\n"

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "run_captured", fake_run_captured)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PROJECT", tmp_path / "refusal_direction")
    monkeypatch.setattr(
        module,
        "generate_lab_credentials",
        lambda: ("lab_user_x", "lab_password_y"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "lab", "--profile", "qwen25-coder-lab"],
    )
    module.main()
    captured = capsys.readouterr()
    output = captured.out
    assert "https://demo-lab.gradio.live" in output
    assert "lab_user_x" in output
    assert "lab_password_y" in output
    assert "stop" in output.lower()
    assert "relay" in output.lower() or "extern" in output.lower()
    joined = " ".join(" ".join(item) for item in commands)
    assert "lab_password_y" not in joined
    assert "colab_start_lab.py" in joined
    assert "/content/qwen_lab_auth.json" in joined
    assert not (tmp_path / ".colab-staging" / "qwen_lab_auth.json").exists()
    assert "lab_password_y" not in captured.err


def test_lab_action_deletes_local_auth_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()

    def boom(_command: list[str]) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(module, "run", boom)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PROJECT", tmp_path / "refusal_direction")
    monkeypatch.setattr(
        module,
        "generate_lab_credentials",
        lambda: ("lab_user_x", "lab_password_y"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "lab", "--profile", "qwen25-coder-lab"],
    )
    with pytest.raises(RuntimeError, match="upload failed"):
        module.main()
    assert not (tmp_path / ".colab-staging" / "qwen_lab_auth.json").exists()


def test_stop_cleans_lab_credential_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    staging = tmp_path / ".colab-staging"
    auth = staging / "qwen_lab_auth.json"
    parts = staging / "lab-lens-parts" / "lens.part000"
    auth.parent.mkdir(parents=True)
    auth.write_text('{"password": "leftover"}', encoding="utf-8")
    parts.parent.mkdir(parents=True)
    parts.write_bytes(b"part")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "stop", "--profile", "qwen25-coder-lab"],
    )
    module.main()
    assert commands[-1][-1] == "refusal-qwen-lab"
    assert not auth.exists()
    assert not parts.exists()


def test_qwen_run_lab_consumes_auth_file_before_launch_and_stays_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from importlib.machinery import SourceFileLoader
    from refusal_direction_replication.qwen_colab_transport import write_lab_auth_file

    module = SourceFileLoader("qwen_run_lab_auth", str(RUN_SCRIPT)).load_module()
    auth = tmp_path / "auth.json"
    write_lab_auth_file(auth, username="file-user", password="file-secret")
    launch = MagicMock()
    with patch.object(
        module,
        "bootstrap_intervention_lab",
        return_value=SimpleNamespace(
            engine=SimpleNamespace(),
            preflight_summary={"device_name": "A100"},
        ),
    ), patch.object(
        module,
        "build_gradio_app",
        return_value=SimpleNamespace(launch=launch),
    ):
        module.main(
            [
                "--no-share",
                "--server-name",
                "127.0.0.1",
                "--auth-file",
                str(auth),
            ]
        )
    assert not auth.exists()
    captured = capsys.readouterr().out + capsys.readouterr().err
    assert "file-secret" not in captured
    launch.assert_called_once()
    assert launch.call_args.kwargs["auth"] == ("file-user", "file-secret")

    missing = tmp_path / "missing.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.main(["--share", "--auth-file", str(missing)])


def test_partial_auth_file_is_deleted_and_rejected(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_colab_transport import consume_lab_auth_file

    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"username": "only"}), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        consume_lab_auth_file(path)
    assert not path.exists()


def test_lab_status_script_prints_operational_fields_without_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert STATUS_SCRIPT.is_file()
    text = STATUS_SCRIPT.read_text(encoding="utf-8")
    assert "download" not in text
    assert "fetch" not in text
    module = _load_script(STATUS_SCRIPT, "qwen_run_lab_status")
    root = tmp_path / "project"
    config_dir = root / "configs"
    artifacts = root / "artifacts/qwen25_coder_7b_lab"
    config_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (config_dir / "qwen25_coder_7b_lab.json").write_text(
        json.dumps(
            {
                "artifacts_dir": "artifacts/qwen25_coder_7b_lab",
                "gpu": "A100",
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "status.json").write_text(
        json.dumps(
            {
                "profile": "qwen25_coder_7b_lab.json",
                "state": "running",
                "gpu": "A100",
                "share_ready": True,
                "public_url": "https://demo-lab.gradio.live",
                "direction_sha256": "0b1cff5e",
                "lens_sha256": "b700a78c",
            }
        ),
        encoding="utf-8",
    )
    assert module.main_with_root(root) == 0
    printed = capsys.readouterr().out
    assert "running" in printed
    assert "A100" in printed
    assert "https://demo-lab.gradio.live" in printed
    assert "password" not in printed.lower()

    (artifacts / "status.json").write_text(
        json.dumps({"state": "running", "password": "nope", "prompt": "secret prompt"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        module.main_with_root(root)


def test_colab_start_lab_does_not_put_secret_in_command_or_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert START_SCRIPT.is_file()
    start_text = START_SCRIPT.read_text(encoding="utf-8")
    assert "--auth-file" in start_text
    assert "QWEN_LAB_AUTH_PASSWORD" not in start_text
    module = _load_script(START_SCRIPT, "colab_start_lab")
    auth = tmp_path / "qwen_lab_auth.json"
    auth.write_text(
        json.dumps({"username": "u", "password": "hidden-pass"}),
        encoding="utf-8",
    )
    log_path = tmp_path / "server.log"
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(module, "REMOTE_AUTH_FILE", auth)
    monkeypatch.setattr(module, "SERVER_LOG", log_path)
    monkeypatch.setattr(module, "STATUS_PATH", status_path)
    monkeypatch.setattr(module, "RUN_SCRIPT", RUN_SCRIPT)
    popped: list[list[str]] = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        popped.append(list(command))
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("Running on public URL: https://started.gradio.live\n")
            stdout.flush()
        else:
            log_path.write_text(
                "Running on public URL: https://started.gradio.live\n",
                encoding="utf-8",
            )
        return FakeProc()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    url = module.start_lab_and_return_url()
    assert url == "https://started.gradio.live"
    assert popped[0][1] == "-u"
    assert popped[0][-2:] == ["--auth-file", str(auth)]
    assert "hidden-pass" not in " ".join(popped[0])
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert "password" not in status
    assert "hidden-pass" not in json.dumps(status)
    assert status["public_url"] == url


def test_wait_for_public_url_fails_when_server_exits(
    tmp_path: Path,
) -> None:
    module = _load_script(START_SCRIPT, "colab_start_lab_early_exit")
    log_path = tmp_path / "server.log"
    log_path.write_text("startup failed safely\n", encoding="utf-8")

    class ExitedProc:
        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match="exited before publishing"):
        module.wait_for_public_url(log_path, process=ExitedProc(), timeout=1.0)


def test_colab_start_lab_terminates_child_after_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(START_SCRIPT, "colab_start_lab_timeout_cleanup")
    monkeypatch.setattr(module, "REMOTE_AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(module, "SERVER_LOG", tmp_path / "server.log")
    monkeypatch.setattr(module, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(module, "RUN_SCRIPT", RUN_SCRIPT)
    events: list[str] = []

    class RunningProc:
        pid = 4242

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, *, timeout):
            events.append(f"wait:{timeout}")
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: RunningProc())
    monkeypatch.setattr(
        module,
        "wait_for_public_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    with pytest.raises(TimeoutError, match="not ready"):
        module.start_lab_and_return_url()
    assert events == ["terminate", "wait:10"]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error"


def test_existing_orchestrator_actions_remain_intact() -> None:
    module = _orchestrator()
    assert module.PROFILE_SESSIONS["gemma"] == "refusal-gemma"
    assert module.fetch_pack_script("qwen25-coder") == ROOT / "scripts" / "colab_pack_qwen_results.py"
    assert module.fetch_pack_script("qwen25-coder-dose") == ROOT / "scripts" / "colab_pack_qwen_dose_results.py"
    parser_source = SCRIPT.read_text(encoding="utf-8")
    for action in (
        "probe",
        "start",
        "sync",
        "bootstrap",
        "auth-hf",
        "execute",
        "fetch",
        "status",
        "pilot",
        "dose",
        "jlens",
        "code-eval",
        "stop",
        "lab",
    ):
        assert f'"{action}"' in parser_source or f"'{action}'" in parser_source
    assert INSTALL_SCRIPT in module.bootstrap_scripts("qwen25-coder-lab")
    assert module.stamps_implementation_commit("qwen25-coder-lab") is True
