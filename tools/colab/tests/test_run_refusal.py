from __future__ import annotations

import importlib.util
import hashlib
import io
import json
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


def _checkpoint_status(checkpoint: bytes, run_id: str) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "run_id": run_id,
                    "implementation_commit": "implementation",
                    "config_sha256": "c" * 64,
                    "corpus_sha256": "d" * 64,
                    "model_revision": "revision",
                },
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()


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


def test_sparse_run_fetches_blind_review_when_remote_pauses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "_upload_sparse_review_labels",
        lambda _session: order.append("labels"),
    )
    monkeypatch.setattr(
        MODULE,
        "run_sparse_remote_action",
        lambda *_args, **_kwargs: (
            order.append("run")
            or (0, "stage=review_export status=review_pending\n")
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "fetch_sparse_review",
        lambda *_args, **_kwargs: order.append("fetch"),
    )

    assert _main(monkeypatch, "sparse-run", "--profile", SPARSE) == 0
    assert order == ["labels", "run", "fetch"]


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
            f"{prefix}/checkpoint_status.json",
        }
    )
    archive = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in (
            (f"{prefix}/sae_checkpoint.pt", b"new-checkpoint"),
            (
                f"{prefix}/checkpoint_status.json",
                _checkpoint_status(b"new-checkpoint", "new"),
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    project = tmp_path / "project"
    destination = project / prefix
    destination.mkdir(parents=True)
    checkpoint = destination / "sae_checkpoint.pt"
    status = destination / "checkpoint_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    old_status = _checkpoint_status(b"old-checkpoint", "old")
    status.write_bytes(old_status)
    real_replace = MODULE.os.replace

    def fail_status(source, target):
        if (
            Path(target) == status
            and Path(source).name == "checkpoint_status.json.incoming"
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
    assert status.read_bytes() == old_status


def test_sparse_checkpoint_extract_rejects_incomplete_pair_without_mutation(
    tmp_path: Path,
) -> None:
    prefix = "artifacts/qwen25_coder_sparse_refusal"
    allowlist = frozenset(
        {
            f"{prefix}/sae_checkpoint.pt",
            f"{prefix}/checkpoint_status.json",
        }
    )
    archive = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"new-checkpoint"
        info = tarfile.TarInfo(f"{prefix}/sae_checkpoint.pt")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "project" / prefix
    destination.mkdir(parents=True)
    checkpoint = destination / "sae_checkpoint.pt"
    status = destination / "checkpoint_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    old_status = _checkpoint_status(b"old-checkpoint", "old")
    status.write_bytes(old_status)

    with pytest.raises(ValueError, match="checkpoint/status pair"):
        MODULE._extract_sparse_archive(
            archive,
            tmp_path / "project",
            allowlist=allowlist,
        )
    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert status.read_bytes() == old_status


def test_sparse_checkpoint_extract_rejects_mismatched_hash_without_mutation(
    tmp_path: Path,
) -> None:
    prefix = "artifacts/qwen25_coder_sparse_refusal"
    allowlist = frozenset(
        {
            f"{prefix}/sae_checkpoint.pt",
            f"{prefix}/checkpoint_status.json",
        }
    )
    archive = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in (
            (f"{prefix}/sae_checkpoint.pt", b"new-checkpoint"),
            (
                f"{prefix}/checkpoint_status.json",
                _checkpoint_status(b"different-checkpoint", "new"),
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "project" / prefix
    destination.mkdir(parents=True)
    checkpoint = destination / "sae_checkpoint.pt"
    status = destination / "checkpoint_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    old_status = _checkpoint_status(b"old-checkpoint", "old")
    status.write_bytes(old_status)

    with pytest.raises(ValueError, match="SHA-256"):
        MODULE._extract_sparse_archive(
            archive,
            tmp_path / "project",
            allowlist=allowlist,
        )
    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert status.read_bytes() == old_status


def test_sparse_fetch_review_is_allowlisted_and_excludes_private_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "project")

    def fake_fetch(_session, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(MODULE, "_fetch_sparse_archive", fake_fetch)
    assert _main(
        monkeypatch,
        "sparse-fetch-review",
        "--profile",
        SPARSE,
    ) == 0
    assert observed["allowlist"] == frozenset(
        {
            "artifacts/qwen25_coder_sparse_refusal/review/"
            "blind_review.jsonl"
        }
    )
    assert "private_manifest" not in repr(observed)


def test_sparse_fetch_results_uses_compact_results_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_fetch(_session, **kwargs):
        observed.update(kwargs)
        return ()

    monkeypatch.setattr(MODULE, "_fetch_sparse_archive", fake_fetch)
    assert MODULE.fetch_sparse_results(
        SPARSE_SESSION,
        project=tmp_path,
    ) == ()
    assert observed["action"] == "pack-results"
    allowlist = observed["allowlist"]
    assert "results/qwen25_coder_sparse_refusal/12_finalize.json" in allowlist
    assert (
        "results/qwen25_coder_sparse_refusal/execution_manifest.json"
        in allowlist
    )
    assert not any(path.startswith("artifacts/") for path in allowlist)


def test_sparse_stop_fetches_checkpoint_then_always_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def failing_fetch(*_args, **_kwargs):
        order.append("fetch")
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(MODULE, "fetch_sparse_checkpoint", failing_fetch)
    monkeypatch.setattr(
        MODULE,
        "try_stop_session",
        lambda _session: order.append("stop") or True,
    )
    assert _main(monkeypatch, "stop", "--profile", SPARSE) == 1
    assert order == ["fetch", "stop"]
