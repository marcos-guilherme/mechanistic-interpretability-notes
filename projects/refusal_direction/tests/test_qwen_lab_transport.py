from __future__ import annotations

import hashlib
import importlib.util
import io
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
from refusal_direction_replication.qwen_lab_readout import (
    FROZEN_LENS_SHA256,
    FROZEN_LENS_V2_SHA256,
)
from refusal_direction_replication.qwen_jlens import JLENS_UPSTREAM_COMMIT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_lab.py"
STATUS_SCRIPT = ROOT / "scripts" / "qwen_run_lab_status.py"
START_SCRIPT = ROOT / "scripts" / "colab_start_lab.py"
RUN_SCRIPT = ROOT / "scripts" / "qwen_run_lab.py"
CLEAN_CHUNKS_SCRIPT = ROOT / "scripts" / "colab_clean_lab_chunks.py"
LAB_CONFIG = ROOT / "configs" / "qwen25_coder_7b_lab.json"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
VALID_SHA256 = "0" * 64
RUNTIME_COMPATIBILITY_SMOKE = (
    "2026-11-16-qwen7b-hooks-jlens-transport-unembed"
)


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


def _seed_lab_sources(project: Path) -> tuple[Path, Path, Path]:
    direction = _write_bytes(
        project / "artifacts/qwen25_coder_7b/selected_direction.pt",
        b"ok-direction",
    )
    lens = _write_bytes(
        project / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt",
        b"ok-lens",
    )
    lens_v2 = _write_bytes(
        project
        / "artifacts/qwen25_coder_7b_jlens_v2"
        / "qwen25_coder_7b_jacobian_lens_v2.pt",
        b"ok-lens-v2",
    )
    return direction, lens, lens_v2


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
    assert config["source_lens_v2"] == (
        "artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt"
    )
    assert config["direction_sha256"] == FROZEN_DIRECTION_SHA256
    assert config["lens_sha256"] == FROZEN_LENS_SHA256
    assert config["lens_v2_sha256"] == FROZEN_LENS_V2_SHA256
    assert config["default_jlens_version"] == "v2"
    assert config["artifacts_dir"] == "artifacts/qwen25_coder_7b_lab"
    assert "results_dir" not in config
    assert "dataset_repo_id" not in config
    assert "dataset_revision" not in config
    assert "datasets" not in config
    blob = json.dumps(config)
    assert "mitre" not in blob
    assert "code_search_net" not in blob
    assert "humaneval" not in blob


def test_lab_transport_pins_frozen_lens_v2_path_hash_and_staging() -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        LAB_LENS_V2_RELATIVE,
        REMOTE_LAB_LENS_V2_MANIFEST,
        REMOTE_LAB_LENS_V2_STAGING,
    )

    assert LAB_LENS_V2_RELATIVE == Path(
        "artifacts/qwen25_coder_7b_jlens_v2/qwen25_coder_7b_jacobian_lens_v2.pt"
    )
    assert REMOTE_LAB_LENS_V2_STAGING == "/content/lab_source_lens_v2.pt"
    assert REMOTE_LAB_LENS_V2_MANIFEST == "/content/lab_source_lens_v2.pt.manifest.json"
    assert FROZEN_LENS_V2_SHA256 == (
        "b80c084a5ce083d2fc8589a62c9a9b2542b6a9dc6b5c006018cc4956eef37a7a"
    )
    run_text = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "--lens-v2-path" in run_text
    install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "lab_source_lens_v2" in install_text
    assert "expected_lens_v2_sha256" in install_text
    assert "lens_v2_sha256" in install_text
    clean_text = CLEAN_CHUNKS_SCRIPT.read_text(encoding="utf-8")
    assert "lab_source_lens_v2" in clean_text


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
        REMOTE_LAB_LENS_V2_STAGING,
    )

    module = _orchestrator()
    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = project / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    lens_v2 = (
        project
        / "artifacts/qwen25_coder_7b_jlens_v2"
        / "qwen25_coder_7b_jacobian_lens_v2.pt"
    )
    _write_bytes(direction, b"direction-bytes")
    _write_bytes(lens, os.urandom(128 * 1024 + 3))
    _write_bytes(lens_v2, os.urandom(64 * 1024 + 7))
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(module, "current_implementation_commit", lambda cwd=None: "b" * 40)
    monkeypatch.setattr(
        module,
        "require_lab_sources",
        lambda _project: (direction, lens, lens_v2),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-lab"],
    )
    module.main()
    uploaded = [item for command in commands for item in command]
    from refusal_direction_replication.qwen_colab_transport import (
        LAB_LENS_CHUNK_BYTES,
        REMOTE_LAB_LENS_MANIFEST,
        REMOTE_LAB_LENS_V2_MANIFEST,
    )

    assert module.CODE_EVAL_LENS_CHUNK_BYTES == 64 * 1024**2
    assert LAB_LENS_CHUNK_BYTES == 16 * 1024**2
    assert module.LAB_LENS_CHUNK_BYTES == 16 * 1024**2
    clean_indexes = [
        index
        for index, command in enumerate(commands)
        if any("colab_clean_lab_chunks.py" in str(item) for item in command)
    ]
    part_indexes = [
        index
        for index, command in enumerate(commands)
        if any(str(item).startswith(REMOTE_LAB_LENS_STAGING + ".part") for item in command)
    ]
    assert clean_indexes, "lab sync must remove stale remote parts before upload"
    assert part_indexes
    assert clean_indexes[0] < part_indexes[0]
    assert REMOTE_LAB_DIRECTION_STAGING in uploaded
    assert REMOTE_LAB_LENS_MANIFEST in uploaded
    assert REMOTE_LAB_LENS_V2_MANIFEST in uploaded
    assert any(
        str(item).startswith(REMOTE_LAB_LENS_STAGING + ".part") for item in uploaded
    )
    assert any(
        str(item).startswith(REMOTE_LAB_LENS_V2_STAGING + ".part") for item in uploaded
    )
    archive = tmp_path / ".colab-staging" / "refusal_direction.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/hello.py" in names
    assert not any("jacobian_lens.pt" in name for name in names)
    assert not any("jacobian_lens_v2.pt" in name for name in names)
    assert not any("selected_direction.pt" in name for name in names)
    assert not any("artifacts" in name.replace("\\", "/") for name in names)


def test_require_local_lab_sources_rejects_link_wrong_hash_and_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication import qwen_colab_transport as transport
    from refusal_direction_replication.qwen_colab_transport import require_local_lab_sources

    project = tmp_path / "project"
    direction, lens, lens_v2 = _seed_lab_sources(project)
    assert require_local_lab_sources(
        project,
        expected_direction_sha256=sha256_file(direction),
        expected_lens_sha256=sha256_file(lens),
        expected_lens_v2_sha256=sha256_file(lens_v2),
    ) == (direction, lens, lens_v2)

    with pytest.raises(ValueError, match="sha256"):
        require_local_lab_sources(
            project,
            expected_direction_sha256="0" * 64,
            expected_lens_sha256=sha256_file(lens),
            expected_lens_v2_sha256=sha256_file(lens_v2),
        )

    with pytest.raises(ValueError, match="sha256"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
            expected_lens_v2_sha256="0" * 64,
        )

    monkeypatch.setattr(transport, "path_is_link_or_junction", lambda path: path == direction)
    with pytest.raises(ValueError, match="regular"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
            expected_lens_v2_sha256=sha256_file(lens_v2),
        )
    monkeypatch.setattr(transport, "path_is_link_or_junction", lambda path: False)

    monkeypatch.setattr(transport, "MAX_LAB_LENS_BYTES", 1)
    _write_bytes(lens, b"too-big")
    with pytest.raises(ValueError, match="size"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
            expected_lens_v2_sha256=sha256_file(lens_v2),
        )

    monkeypatch.setattr(transport, "MAX_LAB_LENS_BYTES", 800 * 1024**2)
    lens_v2.unlink()
    with pytest.raises(FileNotFoundError, match="lens"):
        require_local_lab_sources(
            project,
            expected_direction_sha256=sha256_file(direction),
            expected_lens_sha256=sha256_file(lens),
            expected_lens_v2_sha256="0" * 64,
        )


def test_atomic_lab_install_verifies_hashes_and_rolls_back(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        atomic_install_lab_sources,
        require_local_lab_sources,
    )

    project = tmp_path / "project"
    direction, lens, lens_v2 = _seed_lab_sources(project)
    _write_bytes(direction, b"old-direction")
    _write_bytes(lens, b"old-lens")
    _write_bytes(lens_v2, b"old-lens-v2")
    staged_direction = tmp_path / "direction.pt"
    staged_lens = tmp_path / "lens.pt"
    staged_lens_v2 = tmp_path / "lens_v2.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_lens.write_bytes(b"new-lens")
    staged_lens_v2.write_bytes(b"new-lens-v2")
    installed = atomic_install_lab_sources(
        staged_direction=staged_direction,
        staged_lens=staged_lens,
        staged_lens_v2=staged_lens_v2,
        project=project,
        expected_direction_sha256=sha256_file(staged_direction),
        expected_lens_sha256=sha256_file(staged_lens),
        expected_lens_v2_sha256=sha256_file(staged_lens_v2),
    )
    assert installed == (direction, lens, lens_v2)
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"
    assert lens_v2.read_bytes() == b"new-lens-v2"

    staged_direction.write_bytes(b"rollback-direction")
    staged_lens.write_bytes(b"rollback-lens")
    staged_lens_v2.write_bytes(b"rollback-lens-v2")
    with pytest.raises(ValueError, match="sha256"):
        atomic_install_lab_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            staged_lens_v2=staged_lens_v2,
            project=project,
            expected_direction_sha256=sha256_file(staged_direction),
            expected_lens_sha256="f" * 64,
            expected_lens_v2_sha256=sha256_file(staged_lens_v2),
        )
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"
    assert lens_v2.read_bytes() == b"new-lens-v2"

    with pytest.raises(ValueError, match="sha256"):
        atomic_install_lab_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            staged_lens_v2=staged_lens_v2,
            project=project,
            expected_direction_sha256=sha256_file(staged_direction),
            expected_lens_sha256=sha256_file(staged_lens),
            expected_lens_v2_sha256="f" * 64,
        )
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"
    assert lens_v2.read_bytes() == b"new-lens-v2"
    assert require_local_lab_sources(
        project,
        expected_direction_sha256=sha256_file(direction),
        expected_lens_sha256=sha256_file(lens),
        expected_lens_v2_sha256=sha256_file(lens_v2),
    ) == (direction, lens, lens_v2)


def test_atomic_lab_install_rolls_back_when_v2_stream_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import refusal_direction_replication.qwen_colab_transport as transport

    project = tmp_path / "project"
    direction, lens, lens_v2 = _seed_lab_sources(project)
    original = (direction.read_bytes(), lens.read_bytes(), lens_v2.read_bytes())
    staged_direction = tmp_path / "direction.pt"
    staged_lens = tmp_path / "lens.pt"
    staged_lens_v2 = tmp_path / "lens_v2.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_lens.write_bytes(b"new-lens")
    staged_lens_v2.write_bytes(b"new-lens-v2")
    real_stream = transport._stream_install_regular_file

    def fail_v2_stream(**kwargs):
        destination = kwargs["destination"]
        if destination.name == "qwen25_coder_7b_jacobian_lens_v2.pt":
            raise OSError("disk full")
        return real_stream(**kwargs)

    monkeypatch.setattr(transport, "_stream_install_regular_file", fail_v2_stream)
    with pytest.raises(OSError, match="disk full"):
        transport.atomic_install_lab_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            staged_lens_v2=staged_lens_v2,
            project=project,
            expected_direction_sha256=sha256_file(staged_direction),
            expected_lens_sha256=sha256_file(staged_lens),
            expected_lens_v2_sha256=sha256_file(staged_lens_v2),
        )
    assert (direction.read_bytes(), lens.read_bytes(), lens_v2.read_bytes()) == original


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
    assert "lab_source_lens_v2" in install_text
    assert "expected_lens_v2_sha256" in install_text
    assert "lens_v2_sha256" in install_text

    module = _load_script(INSTALL_SCRIPT, "install_lab")
    destination = tmp_path / "lab_source_lens.pt"
    monkeypatch.setattr(module, "REMOTE_LAB_LENS_STAGING", str(destination))
    part0 = tmp_path / "lab_source_lens.pt.part000"
    part1 = tmp_path / "lab_source_lens.pt.part001"
    part0.write_bytes(b"alpha")
    part1.write_bytes(b"beta")
    from refusal_direction_replication.qwen_colab_transport import (
        write_artifact_chunk_manifest,
    )

    source = tmp_path / "source.pt"
    source.write_bytes(b"alphabeta")
    write_artifact_chunk_manifest(
        source,
        [part0, part1],
        tmp_path / "lab_source_lens.pt.manifest.json",
    )
    assert module.assemble_staged_lens() == destination
    assert destination.read_bytes() == b"alphabeta"
    assert not list(tmp_path.glob("lab_source_lens.pt.part*"))

    destination_v2 = tmp_path / "lab_source_lens_v2.pt"
    monkeypatch.setattr(module, "REMOTE_LAB_LENS_V2_STAGING", str(destination_v2))
    part0_v2 = tmp_path / "lab_source_lens_v2.pt.part000"
    part1_v2 = tmp_path / "lab_source_lens_v2.pt.part001"
    part0_v2.write_bytes(b"gamma")
    part1_v2.write_bytes(b"delta")
    source_v2 = tmp_path / "source_v2.pt"
    source_v2.write_bytes(b"gammadelta")
    write_artifact_chunk_manifest(
        source_v2,
        [part0_v2, part1_v2],
        tmp_path / "lab_source_lens_v2.pt.manifest.json",
    )
    assert module.assemble_staged_lens_v2() == destination_v2
    assert destination_v2.read_bytes() == b"gammadelta"
    assert not list(tmp_path.glob("lab_source_lens_v2.pt.part*"))


def test_lab_installer_rejects_noncontiguous_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(INSTALL_SCRIPT, "install_lab_gap")
    destination = tmp_path / "lab_source_lens.pt"
    monkeypatch.setattr(module, "REMOTE_LAB_LENS_STAGING", str(destination))
    part0 = tmp_path / "lab_source_lens.pt.part000"
    part2 = tmp_path / "lab_source_lens.pt.part002"
    part0.write_bytes(b"alpha")
    part2.write_bytes(b"gap")
    from refusal_direction_replication.qwen_colab_transport import (
        write_artifact_chunk_manifest,
    )

    source = tmp_path / "source.pt"
    source.write_bytes(b"alphagap")
    expected = tmp_path / "lab_source_lens.pt.part001"
    expected.write_bytes(b"gap")
    write_artifact_chunk_manifest(
        source,
        [part0, expected],
        tmp_path / "lab_source_lens.pt.manifest.json",
        part_names=[
            "lab_source_lens.pt.part000",
            "lab_source_lens.pt.part001",
        ],
    )
    expected.unlink()
    with pytest.raises(ValueError, match="missing|extra|contiguous"):
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


def test_qwen_run_lab_forwards_lens_v2_path_to_bootstrap(tmp_path: Path) -> None:
    module = _load_script(RUN_SCRIPT, "qwen_run_lab_lens_v2")
    captured: dict[str, object] = {}
    launch = MagicMock()
    direction = tmp_path / "direction.pt"
    lens = tmp_path / "lens.pt"
    lens_v2 = tmp_path / "lens_v2.pt"
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    lens_v2.write_bytes(b"lens-v2")

    def fake_bootstrap(root, **kwargs):
        del root
        captured.update(kwargs)
        return SimpleNamespace(
            engine=SimpleNamespace(),
            preflight_summary={"device_name": "A100"},
        )

    with patch.object(module, "bootstrap_intervention_lab", fake_bootstrap), patch.object(
        module,
        "build_gradio_app",
        return_value=SimpleNamespace(launch=launch),
    ):
        module.main(
            [
                "--no-share",
                "--server-name",
                "127.0.0.1",
                "--direction-path",
                str(direction),
                "--lens-path",
                str(lens),
                "--lens-v2-path",
                str(lens_v2),
            ]
        )
    assert Path(captured["direction_path"]) == direction
    assert Path(captured["lens_path"]) == lens
    assert Path(captured["lens_v2_path"]) == lens_v2
    launch.assert_called_once()


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert STATUS_SCRIPT.is_file()
    text = STATUS_SCRIPT.read_text(encoding="utf-8")
    assert "download" not in text
    assert "fetch" not in text
    module = _load_script(STATUS_SCRIPT, "qwen_run_lab_status")
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
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
                "pid": 4242,
                "direction_sha256": VALID_SHA256,
                "lens_sha256": "b" * 64,
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
    assert "message=" not in printed

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
    assert auth.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(auth.stat().st_mode) == 0o600
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
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps({"username": "u", "password": "hidden-pass"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REMOTE_AUTH_FILE", auth)
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
    assert not auth.exists()
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error"
    assert "message" not in status


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


def _add_tar_member(
    handle: tarfile.TarFile,
    name: str,
    payload: bytes = b"",
    *,
    member_type: bytes = tarfile.REGTYPE,
    linkname: str = "",
) -> None:
    info = tarfile.TarInfo(name=name)
    info.type = member_type
    info.linkname = linkname
    if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
        return
    info.size = 0
    handle.addfile(info)


def _write_lab_archive(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as handle:
        for name, payload in members:
            member_type = tarfile.DIRTYPE if name.endswith("/") else tarfile.REGTYPE
            _add_tar_member(
                handle,
                name.rstrip("/") if member_type is tarfile.DIRTYPE else name,
                payload,
                member_type=member_type,
            )
    return path


def test_lab_safe_extract_rejects_links_specials_and_extra_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(INSTALL_SCRIPT, "install_lab_extract_reject")
    destination = tmp_path / "content"
    destination.mkdir()
    archive = tmp_path / "refusal_direction.tar.gz"
    remote_root = destination / "refusal_direction"
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "DESTINATION", destination)
    monkeypatch.setattr(module, "REMOTE_ROOT", remote_root)

    with tarfile.open(archive, "w:gz") as handle:
        _add_tar_member(handle, "refusal_direction", member_type=tarfile.DIRTYPE)
        _add_tar_member(handle, "refusal_direction/ok.py", b"print(1)\n")
        _add_tar_member(
            handle,
            "refusal_direction/link.py",
            member_type=tarfile.SYMTYPE,
            linkname="/etc/passwd",
        )
    with pytest.raises(ValueError, match="link|symlink"):
        module.safe_extract()
    assert not remote_root.exists()
    assert list(destination.glob(".lab-extract-*")) == []

    with tarfile.open(archive, "w:gz") as handle:
        _add_tar_member(handle, "refusal_direction", member_type=tarfile.DIRTYPE)
        _add_tar_member(handle, "refusal_direction/ok.py", b"print(1)\n")
        _add_tar_member(
            handle,
            "refusal_direction/hard.py",
            member_type=tarfile.LNKTYPE,
            linkname="refusal_direction/ok.py",
        )
    with pytest.raises(ValueError, match="link"):
        module.safe_extract()

    with tarfile.open(archive, "w:gz") as handle:
        _add_tar_member(handle, "refusal_direction", member_type=tarfile.DIRTYPE)
        _add_tar_member(
            handle,
            "refusal_direction/fifo",
            member_type=tarfile.FIFOTYPE,
        )
    with pytest.raises(ValueError, match="regular|special"):
        module.safe_extract()

    with tarfile.open(archive, "w:gz") as handle:
        _add_tar_member(handle, "refusal_direction", member_type=tarfile.DIRTYPE)
        _add_tar_member(handle, "refusal_direction/ok.py", b"print(1)\n")
        _add_tar_member(handle, "other_root/evil.py", b"nope")
    with pytest.raises(ValueError, match="top-level|unsafe"):
        module.safe_extract()
    assert not (destination / "other_root").exists()


def test_lab_safe_extract_swaps_atomically_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(INSTALL_SCRIPT, "install_lab_extract_swap")
    destination = tmp_path / "content"
    destination.mkdir()
    remote_root = destination / "refusal_direction"
    remote_root.mkdir()
    marker = remote_root / "old.py"
    marker.write_text("keep-me\n", encoding="utf-8")
    archive = tmp_path / "refusal_direction.tar.gz"
    _write_lab_archive(
        archive,
        [
            ("refusal_direction/", b""),
            ("refusal_direction/new.py", b"fresh\n"),
        ],
    )
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "DESTINATION", destination)
    monkeypatch.setattr(module, "REMOTE_ROOT", remote_root)

    real_replace = os.replace

    def fail_final_swap(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        source = Path(src)
        if Path(dst) == remote_root and any(
            part.startswith(".lab-extract-") for part in source.parts
        ):
            raise OSError("swap failed")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_final_swap)
    with pytest.raises(OSError, match="swap failed"):
        module.safe_extract()
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "keep-me\n"
    assert not (remote_root / "new.py").exists()
    assert archive.is_file()
    assert list(destination.glob(".lab-extract-*")) == []

    monkeypatch.setattr(os, "replace", real_replace)
    module.safe_extract()
    assert (remote_root / "new.py").read_bytes() == b"fresh\n"
    assert not marker.exists()
    assert not archive.exists()
    assert list(destination.glob(".lab-extract-*")) == []


def test_lab_status_schema_rejects_free_text_nested_and_bad_url(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        validate_lab_operational_status,
    )

    valid = {
        "profile": "qwen25_coder_7b_lab.json",
        "state": "running",
        "gpu": "A100",
        "share_ready": True,
        "public_url": "https://demo-lab.gradio.live",
        "pid": 7,
        "direction_sha256": VALID_SHA256,
        "lens_sha256": "a" * 64,
    }
    assert validate_lab_operational_status(valid)["pid"] == 7
    with pytest.raises(ValueError, match="message|field"):
        validate_lab_operational_status({**valid, "message": "free text"})
    with pytest.raises(ValueError, match="nested|field"):
        validate_lab_operational_status({**valid, "extra": {"nested": True}})
    with pytest.raises(ValueError, match="list|field|type"):
        validate_lab_operational_status({**valid, "gpu": ["A100"]})
    with pytest.raises(ValueError, match="url|gradio"):
        validate_lab_operational_status(
            {**valid, "public_url": "https://evil.example.com"}
        )
    with pytest.raises(ValueError, match="sha256|hash"):
        validate_lab_operational_status({**valid, "direction_sha256": "0b1cff5e"})
    with pytest.raises(ValueError, match="pid|type"):
        validate_lab_operational_status({**valid, "pid": True})


def test_lab_status_reports_stopped_when_running_pid_is_dead(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(STATUS_SCRIPT, "qwen_run_lab_status_dead_pid")

    def missing_pid(_pid: int, _sig: int) -> None:
        raise ProcessLookupError()

    monkeypatch.setattr(os, "kill", missing_pid)
    root = tmp_path / "project"
    artifacts = root / "artifacts/qwen25_coder_7b_lab"
    artifacts.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs" / "qwen25_coder_7b_lab.json").write_text(
        json.dumps({"artifacts_dir": "artifacts/qwen25_coder_7b_lab"}),
        encoding="utf-8",
    )
    status_path = artifacts / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "profile": "qwen25_coder_7b_lab.json",
                "state": "running",
                "gpu": "A100",
                "share_ready": True,
                "public_url": "https://demo-lab.gradio.live",
                "pid": 999001,
            }
        ),
        encoding="utf-8",
    )
    assert module.main_with_root(root) == 0
    printed = capsys.readouterr().out
    assert "state=stopped" in printed
    assert "share_ready=False" in printed
    assert "message=" not in printed
    assert "password" not in printed.lower()
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "running"
    assert persisted["share_ready"] is True


def test_colab_start_lab_chmods_auth_before_popen_and_wipes_on_popen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(START_SCRIPT, "colab_start_lab_popen_fail")
    auth = tmp_path / "qwen_lab_auth.json"
    auth.write_text(
        json.dumps({"username": "u", "password": "hidden-pass"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REMOTE_AUTH_FILE", auth)
    monkeypatch.setattr(module, "SERVER_LOG", tmp_path / "server.log")
    monkeypatch.setattr(module, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(module, "RUN_SCRIPT", RUN_SCRIPT)
    events: list[str] = []
    real_chmod = os.chmod

    def tracked_chmod(path: str | os.PathLike[str], mode: int) -> None:
        if Path(path) == auth:
            events.append(f"chmod:{mode:o}")
        real_chmod(path, mode)

    def boom(*_args, **_kwargs):
        events.append("popen")
        if os.name != "nt":
            assert stat.S_IMODE(auth.stat().st_mode) == 0o600
        raise OSError("popen failed")

    monkeypatch.setattr(module.os, "chmod", tracked_chmod)
    monkeypatch.setattr(module.subprocess, "Popen", boom)
    with pytest.raises(OSError, match="popen failed"):
        module.start_lab_and_return_url()
    assert events[0].startswith("chmod:")
    assert events[1] == "popen"
    assert not auth.exists()


def test_lab_chunk_manifest_roundtrip_and_rejects_corrupt_parts(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        assemble_chunked_file,
        split_artifact_file,
        write_artifact_chunk_manifest,
    )

    source = tmp_path / "lens.pt"
    source.write_bytes(os.urandom(90))
    parts = split_artifact_file(
        source,
        tmp_path,
        chunk_bytes=40,
        part_prefix="lens.pt.part",
    )
    manifest = write_artifact_chunk_manifest(
        source,
        parts,
        tmp_path / "lens.pt.manifest.json",
    )
    assembled = assemble_chunked_file(
        tmp_path / "lens.pt",
        manifest_path=manifest,
        require_manifest=True,
    )
    payload = source.read_bytes()
    assert assembled.read_bytes() == payload
    assert hashlib.sha256(assembled.read_bytes()).hexdigest() == hashlib.sha256(
        payload
    ).hexdigest()
    assert not list(tmp_path.glob("lens.pt.part*"))

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    corrupt_source = corrupt / "payload.pt"
    corrupt_source.write_bytes(payload)
    parts = split_artifact_file(
        corrupt_source,
        corrupt,
        chunk_bytes=40,
        part_prefix="out.pt.part",
    )
    corrupt_manifest = write_artifact_chunk_manifest(
        corrupt_source,
        parts,
        corrupt / "out.pt.manifest.json",
    )
    parts[0].write_bytes(b"tampered-bytes-not-matching-sha")
    destination = corrupt / "out.pt"
    with pytest.raises(ValueError, match="sha256|altered"):
        assemble_chunked_file(
            destination,
            manifest_path=corrupt_manifest,
            require_manifest=True,
        )
    assert not destination.exists()

    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra_source = extra_dir / "payload.pt"
    extra_source.write_bytes(payload)
    parts = split_artifact_file(
        extra_source,
        extra_dir,
        chunk_bytes=40,
        part_prefix="out.pt.part",
    )
    extra_manifest = write_artifact_chunk_manifest(
        extra_source,
        parts,
        extra_dir / "out.pt.manifest.json",
    )
    extra = extra_dir / "out.pt.part099"
    extra.write_bytes(b"extra")
    destination = extra_dir / "out.pt"
    with pytest.raises(ValueError, match="extra"):
        assemble_chunked_file(
            destination,
            manifest_path=extra_manifest,
            require_manifest=True,
        )
    extra.unlink()
    parts[1].unlink()
    with pytest.raises(ValueError, match="missing"):
        assemble_chunked_file(
            destination,
            manifest_path=extra_manifest,
            require_manifest=True,
        )


def test_lab_chunk_cleanup_unlinks_without_printing_contents(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert CLEAN_CHUNKS_SCRIPT.is_file()
    module = _load_script(CLEAN_CHUNKS_SCRIPT, "clean_lab_chunks")
    destination = tmp_path / "lab_source_lens.pt"
    secret = b"SECRET-LENS-BYTES"
    (tmp_path / "lab_source_lens.pt.part000").write_bytes(secret)
    (tmp_path / "lab_source_lens.pt.part001").write_bytes(secret)
    (tmp_path / "lab_source_lens.pt.manifest.json").write_text("{}", encoding="utf-8")
    assembling = tmp_path / "lab_source_lens.pt.assembling"
    assembling.write_bytes(secret)
    destination.write_bytes(secret)
    module.clean_lab_chunk_files(destination)
    assert list(tmp_path.glob("lab_source_lens.pt*")) == []
    captured = capsys.readouterr()
    assert "SECRET-LENS-BYTES" not in captured.out
    assert "SECRET-LENS-BYTES" not in captured.err


def test_jlens_no_deps_keeps_project_transformers_and_records_smoke_id() -> None:
    from refusal_direction_replication.qwen_jlens import (
        RUNTIME_COMPATIBILITY_SMOKE,
        assert_pinned_transformers_runtime,
        parse_supported_transformers_versions,
    )

    config = json.loads(LAB_CONFIG.read_text(encoding="utf-8"))
    assert config["jlens_install_no_deps"] is True
    assert config["runtime_compatibility_smoke"] == RUNTIME_COMPATIBILITY_SMOKE
    assert RUNTIME_COMPATIBILITY_SMOKE == (
        "2026-11-16-qwen7b-hooks-jlens-transport-unembed"
    )
    supported = parse_supported_transformers_versions(ROOT / "pyproject.toml")
    assert supported == frozenset({"4.44.2", "4.48.3"})
    assert assert_pinned_transformers_runtime(
        ROOT / "pyproject.toml",
        installed="4.44.2",
    ) == "4.44.2"
    with pytest.raises(ValueError, match="runtime_compatibility_smoke"):
        assert_pinned_transformers_runtime(
            ROOT / "pyproject.toml",
            installed="5.0.0",
        )
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "--no-deps" in install
    assert "deliberate" in install.lower() or "projeto" in install.lower()
    assert "assert_pinned_transformers_runtime" in install
    docs = (ROOT / "LABORATORIO_QWEN.md").read_text(encoding="utf-8")
    assert RUNTIME_COMPATIBILITY_SMOKE in docs
    assert "--no-deps" in docs
    compatibility_section = docs.lower()
    smoke_index = compatibility_section.index("runtime_compatibility_smoke")
    window = compatibility_section[max(0, smoke_index - 400) : smoke_index + 400]
    assert "não é uma declaração de compatibilidade geral" in window
    assert "a100" not in window
    assert "qwen25_coder_7b_jacobian_lens_v2.pt" in docs
    assert FROZEN_LENS_V2_SHA256 in docs
    assert "v2 (contexto longo)" in docs
    assert "v1 (100 amostras)" in docs
    assert "--lens-v2-path" in docs
    assert "lab_source_lens_v2" in docs
    assert "source_lens_v2" in docs
