from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "run_refusal.py"
SPEC = importlib.util.spec_from_file_location("sparse_run_refusal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SPARSE = "qwen25-coder-sparse-refusal"
SPARSE_SESSION = "refusal-qwen-sparse-refusal"


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run_refusal.py", *argv])
    return MODULE.main()


def test_sparse_profile_requires_a100_and_has_unique_session() -> None:
    profile = MODULE.PROFILES[SPARSE]
    assert profile.session_name == SPARSE_SESSION
    assert profile.gpu_type == "A100"
    assert profile.remote_root == "/content/refusal_direction"
    assert profile.bootstrap_notebook.endswith("colab_install_sparse_refusal.py")
    assert profile.run_notebook.endswith("colab_run_sparse_refusal.py")
    assert MODULE.PROFILE_SESSIONS[SPARSE] == SPARSE_SESSION
    assert SPARSE in MODULE.A100_PROFILES


def test_sparse_actions_are_isolated_to_sparse_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        MODULE, "run", lambda command: commands.append(list(command))
    )
    with pytest.raises(SystemExit):
        _main(monkeypatch, "sparse-pilot", "--profile", "qwen25-coder")
    assert commands == []


@pytest.mark.parametrize(
    "marker",
    ("Traceback", "RuntimeError", "CUDA out of memory", "SystemExit: 1"),
)
def test_sparse_output_failure_markers_are_terminal(marker: str) -> None:
    assert MODULE.sparse_output_has_failure(f"remote output\n{marker}\n")


def test_sparse_failure_fetches_checkpoint_before_stopping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "refusal_direction")
    monkeypatch.setattr(
        MODULE,
        "run_sparse_remote_action",
        lambda *_args, **_kwargs: (0, "Traceback: remote failure"),
    )
    monkeypatch.setattr(
        MODULE,
        "fetch_sparse_checkpoint",
        lambda *_args, **_kwargs: order.append("checkpoint"),
    )
    monkeypatch.setattr(
        MODULE,
        "try_stop_session",
        lambda _session: order.append("stop") or True,
    )

    assert _main(monkeypatch, "sparse-run", "--profile", SPARSE) == 1
    assert order == ["checkpoint", "stop"]


def test_sparse_start_forces_a100(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        MODULE, "run", lambda command: commands.append(list(command))
    )
    assert _main(
        monkeypatch,
        "start",
        "--profile",
        SPARSE,
        "--gpu",
        "T4",
    ) == 0
    assert commands == [MODULE.start_command(SPARSE_SESSION, "A100")]


def test_sparse_bootstrap_detects_notebook_failure_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "run_sparse_installer",
        lambda *_args, **_kwargs: (0, "SystemExit: 1"),
    )
    monkeypatch.setattr(
        MODULE,
        "fetch_sparse_checkpoint",
        lambda *_args, **_kwargs: order.append("checkpoint"),
    )
    monkeypatch.setattr(
        MODULE,
        "try_stop_session",
        lambda _session: order.append("stop") or True,
    )

    assert _main(monkeypatch, "bootstrap", "--profile", SPARSE) == 1
    assert order == ["checkpoint", "stop"]


def test_sparse_checkpoint_fetch_rolls_back_partial_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = "artifacts/qwen25_coder_sparse_refusal"
    allowlist = frozenset(
        {
            f"{prefix}/sae_checkpoint.pt",
            f"{prefix}/training_status.json",
        }
    )
    archive = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in (
            (f"{prefix}/sae_checkpoint.pt", b"new-checkpoint"),
            (f"{prefix}/training_status.json", b'{"stage":"new"}\n'),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    project = tmp_path / "project"
    destination = project / prefix
    destination.mkdir(parents=True)
    checkpoint = destination / "sae_checkpoint.pt"
    status = destination / "training_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    status.write_text('{"stage":"old"}\n', encoding="utf-8")
    real_replace = MODULE.os.replace

    def fail_status(source, target):
        if (
            Path(target) == status
            and Path(source).name == "training_status.json.incoming"
        ):
            raise OSError("simulated fetch failure")
        return real_replace(source, target)

    monkeypatch.setattr(MODULE.os, "replace", fail_status)
    with pytest.raises(OSError, match="simulated"):
        MODULE._extract_sparse_archive(
            archive,
            project,
            allowlist=allowlist,
        )

    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert status.read_text(encoding="utf-8") == '{"stage":"old"}\n'
