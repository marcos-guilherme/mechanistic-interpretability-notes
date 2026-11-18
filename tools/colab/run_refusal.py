from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "refusal_direction"
AUTH = ["colab", "--auth=adc"]
SPARSE_REFUSAL = "qwen25-coder-sparse-refusal"
PROFILE_SESSIONS = {
    "gemma": "refusal-gemma",
    "qwen25-coder": "refusal-qwen",
    "qwen25-coder-dose": "refusal-qwen",
    "qwen25-coder-jlens": "refusal-qwen",
    "qwen25-coder-code-eval": "refusal-qwen",
    "qwen25-coder-lab": "refusal-qwen-lab",
    "qwen25-coder-jlens-v2": "refusal-qwen-jlens-v2",
    "qwen25-coder-causal-jlens": "refusal-qwen-causal-jlens",
    SPARSE_REFUSAL: "refusal-qwen-sparse-refusal",
}
JLENS_V2 = "qwen25-coder-jlens-v2"
CAUSAL_JLENS = "qwen25-coder-causal-jlens"


class ColabProfile(NamedTuple):
    session_name: str
    gpu_type: str
    remote_root: str
    bootstrap_notebook: str
    run_notebook: str
    result_archive: str


PROFILES = {
    SPARSE_REFUSAL: ColabProfile(
        session_name="refusal-qwen-sparse-refusal",
        gpu_type="A100",
        remote_root="/content/refusal_direction",
        bootstrap_notebook=(
            "projects/refusal_direction/scripts/"
            "colab_install_sparse_refusal.py"
        ),
        run_notebook=(
            "projects/refusal_direction/scripts/colab_run_sparse_refusal.py"
        ),
        result_archive="/content/qwen_sparse_refusal_results.tar.gz",
    ),
}
EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
}
CODE_EVAL_LENS_CHUNK_BYTES = 64 * 1024**2
ARTIFACT_CHUNK_BYTES = CODE_EVAL_LENS_CHUNK_BYTES
LAB_LENS_CHUNK_BYTES = 16 * 1024**2
REMOTE_CODE_EVAL_LENS_STAGING = "/content/code_eval_source_lens.pt"
REMOTE_LAB_DIRECTION_STAGING = "/content/lab_source_direction.pt"
REMOTE_LAB_LENS_STAGING = "/content/lab_source_lens.pt"
REMOTE_LAB_LENS_MANIFEST = "/content/lab_source_lens.pt.manifest.json"
REMOTE_LAB_AUTH_FILE = "/content/qwen_lab_auth.json"
A100_PROFILES = frozenset(
    {
        "qwen25-coder-code-eval",
        "qwen25-coder-lab",
        JLENS_V2,
        CAUSAL_JLENS,
        SPARSE_REFUSAL,
    }
)
DEFAULT_EXEC_TIMEOUT = 7200
JLENS_V2_EXEC_TIMEOUT = 46800
JLENS_V2_LENS_CHUNK_BYTES = LAB_LENS_CHUNK_BYTES
REMOTE_JLENS_V2_LENS_STAGING = "/content/jlens_v2_source_v1_lens.pt"
REMOTE_JLENS_V2_LENS_MANIFEST = "/content/jlens_v2_source_v1_lens.pt.manifest.json"
REMOTE_JLENS_V2_ARTIFACTS = "/content/refusal_direction/artifacts/qwen25_coder_7b_jlens_v2"
REMOTE_CAUSAL_JLENS_DIRECTION_STAGING = "/content/causal_jlens_source_direction.pt"
REMOTE_CAUSAL_JLENS_V1_LENS_STAGING = "/content/causal_jlens_source_v1_lens.pt"
REMOTE_CAUSAL_JLENS_V2_LENS_STAGING = "/content/causal_jlens_source_v2_lens.pt"
REMOTE_SPARSE_DIRECTION_STAGING = "/content/sparse_refusal_direction.pt"
REMOTE_SPARSE_REVIEWED_PAIRS = "/content/sparse_refusal_reviewed_pairs.jsonl"
REMOTE_SPARSE_REVIEWED_PAIRS_DIGEST = (
    "/content/sparse_refusal_reviewed_pairs.sha256.json"
)
REMOTE_SPARSE_CHECKPOINT_STAGING = "/content/sparse_refusal_checkpoint.pt"
REMOTE_SPARSE_TRAINING_STATUS = "/content/sparse_refusal_training_status.json"
REMOTE_SPARSE_COMMAND = "/content/qwen_sparse_refusal_command.json"
REMOTE_SPARSE_CHECKPOINT_ARCHIVE = (
    "/content/qwen_sparse_refusal_checkpoint.tar.gz"
)
REMOTE_SPARSE_RESULTS_ARCHIVE = "/content/qwen_sparse_refusal_results.tar.gz"
SPARSE_CHUNK_BYTES = LAB_LENS_CHUNK_BYTES
SPARSE_FAILURE_MARKERS = (
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "SystemExit: 1",
)
CAUSAL_JLENS_LENS_CHUNK_BYTES = LAB_LENS_CHUNK_BYTES
# Status/events before the checkpoint: the fit replaces the checkpoint before it
# advances status/events, so this order can never yield status ahead of it.
JLENS_V2_BACKUP_ORDER = ("status.json", "events.jsonl", "jlens_fit_checkpoint.pt")
JLENS_V2_BACKUP_TIMEOUT_SECONDS = 1800
STOP_BACKUP_TIMEOUT_SECONDS = 600
STOP_RETRY_ATTEMPTS = 3
STOP_RETRY_BACKOFF_SECONDS = (0.25, 0.5, 1.0)
REMOTE_FILE_MISSING_MARKER = "file or directory not found"


def huggingface_token_path(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home()
    return base / ".cache" / "huggingface" / "token"


def split_artifact_file(
    source: Path,
    staging: Path,
    *,
    chunk_bytes: int = ARTIFACT_CHUNK_BYTES,
    part_prefix: str = "lens.part",
) -> list[Path]:
    sys.path.insert(0, str(PROJECT / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        split_artifact_file as split_remote_artifact,
    )

    return split_remote_artifact(
        source,
        staging,
        chunk_bytes=chunk_bytes,
        part_prefix=part_prefix,
    )


def split_code_eval_lens(
    source: Path,
    staging: Path,
    *,
    chunk_bytes: int = CODE_EVAL_LENS_CHUNK_BYTES,
) -> list[Path]:
    return split_artifact_file(
        source,
        staging,
        chunk_bytes=chunk_bytes,
        part_prefix="lens.part",
    )


def start_command(session: str, gpu: str) -> list[str]:
    return [*AUTH, "new", "-s", session, "--gpu", gpu]


def stop_command(session: str) -> list[str]:
    return [*AUTH, "stop", "-s", session]


def format_stop_rerun_command(*, profile: str, session: str | None) -> str:
    parts = ["python", "tools/colab/run_refusal.py", "stop", "--profile", profile]
    if session is not None:
        parts.extend(["--session", session])
    return " ".join(parts)


def try_stop_session(session: str) -> bool:
    for attempt in range(STOP_RETRY_ATTEMPTS):
        try:
            run(stop_command(session))
            return True
        except KeyboardInterrupt:
            raise
        except Exception:
            if attempt + 1 < STOP_RETRY_ATTEMPTS:
                time.sleep(STOP_RETRY_BACKOFF_SECONDS[attempt])
    return False


def upload_command(session: str, local: Path, remote: str) -> list[str]:
    return [
        *AUTH,
        "upload",
        "-s",
        session,
        str(local),
        remote,
    ]


def exec_command(
    session: str,
    file: Path,
    *,
    timeout: int,
) -> list[str]:
    return [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(file),
        "--timeout",
        str(timeout),
    ]


def download_command(session: str, remote: str, local: Path) -> list[str]:
    return [
        *AUTH,
        "download",
        "-s",
        session,
        remote,
        str(local),
    ]


def resolve_session(profile: str, session: str | None) -> str:
    if session is not None:
        return session
    return PROFILE_SESSIONS[profile]


def fetch_pack_script(profile: str) -> Path:
    if profile == CAUSAL_JLENS:
        return PROJECT / "scripts" / "colab_pack_qwen_causal_jlens_results.py"
    if profile == JLENS_V2:
        return PROJECT / "scripts" / "colab_pack_qwen_jlens_v2_results.py"
    if profile == "qwen25-coder-code-eval":
        return PROJECT / "scripts" / "colab_pack_qwen_code_eval_results.py"
    if profile == "qwen25-coder-jlens":
        return PROJECT / "scripts" / "colab_pack_qwen_jlens_results.py"
    if profile == "qwen25-coder-dose":
        return PROJECT / "scripts" / "colab_pack_qwen_dose_results.py"
    if profile == "qwen25-coder":
        return PROJECT / "scripts" / "colab_pack_qwen_results.py"
    return PROJECT / "scripts" / "colab_pack_results.py"


def fetch_pack_timeout(profile: str) -> int:
    if profile in ("qwen25-coder-code-eval", JLENS_V2, CAUSAL_JLENS):
        return 1800
    return 300


def stamps_implementation_commit(profile: str) -> bool:
    return profile.startswith("qwen25-coder")


def status_command(session: str, *, profile: str = "qwen25-coder") -> list[str]:
    script = PROJECT / "scripts" / "qwen_run_status.py"
    if profile == "qwen25-coder-lab":
        script = PROJECT / "scripts" / "qwen_run_lab_status.py"
    elif profile == "qwen25-coder-code-eval":
        script = PROJECT / "scripts" / "qwen_run_code_eval_status.py"
    elif profile == "qwen25-coder-jlens":
        script = PROJECT / "scripts" / "qwen_run_jlens_status.py"
    elif profile == JLENS_V2:
        script = PROJECT / "scripts" / "qwen_run_jlens_v2_status.py"
    elif profile == CAUSAL_JLENS:
        script = PROJECT / "scripts" / "qwen_run_causal_jlens_status.py"
    elif profile == "qwen25-coder-dose":
        script = PROJECT / "scripts" / "qwen_run_dose_status.py"
    return [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(script),
        "--timeout",
        "120",
    ]


def pilot_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_pilot.py",
        timeout=timeout,
    )


def dose_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_dose.py",
        timeout=timeout,
    )


def jlens_command(
    session: str, *, timeout: int, profile: str = "qwen25-coder-jlens"
) -> list[str]:
    script = "qwen_run_jlens_v2.py" if profile == JLENS_V2 else "qwen_run_jlens.py"
    return exec_command(
        session,
        PROJECT / "scripts" / script,
        timeout=timeout,
    )


def code_eval_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_code_eval.py",
        timeout=timeout,
    )


def lab_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "colab_start_lab.py",
        timeout=timeout,
    )


def causal_jlens_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_causal_jlens.py",
        timeout=timeout,
    )


def extract_qwen_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_archive,
    )

    extract_qwen_archive(archive, project)


def extract_qwen_dose_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_dose_archive,
    )

    extract_qwen_dose_archive(archive, project)


def extract_qwen_jlens_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_jlens_archive,
    )

    extract_qwen_jlens_archive(archive, project)


def extract_qwen_jlens_v2_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_jlens_v2_archive,
    )

    extract_qwen_jlens_v2_archive(archive, project)


def extract_qwen_code_eval_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_code_eval_archive,
    )

    extract_qwen_code_eval_archive(archive, project)


def extract_qwen_causal_jlens_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_causal_jlens_archive,
    )

    extract_qwen_causal_jlens_archive(archive, project)


def require_dose_source_direction(project: Path) -> Path:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_dose_source_direction,
    )

    return require_local_dose_source_direction(project)


def optional_jlens_checkpoint(project: Path) -> Path | None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        optional_local_jlens_checkpoint,
    )

    return optional_local_jlens_checkpoint(project)


def optional_jlens_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        optional_local_jlens_resume_bundle,
    )

    return optional_local_jlens_resume_bundle(project)


def optional_jlens_v2_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        optional_local_jlens_v2_resume_bundle,
    )

    return optional_local_jlens_v2_resume_bundle(project)


def require_jlens_v2_source_v1_lens(project: Path) -> Path:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.pipeline import load_config  # noqa: WPS433
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        path_is_link_or_junction,
    )
    from refusal_direction_replication.qwen_jlens import JLENS_V2_PROFILE  # noqa: WPS433
    from refusal_direction_replication.qwen_lab_readout import (  # noqa: WPS433
        FROZEN_LENS_SHA256,
        verify_frozen_lens_hash,
    )

    config = load_config(project, profile=JLENS_V2_PROFILE)
    lens = project / str(config["source_v1_lens"])
    if path_is_link_or_junction(lens) or not lens.is_file():
        raise FileNotFoundError(f"frozen v1 lens not found: {config['source_v1_lens']}")
    verify_frozen_lens_hash(lens, FROZEN_LENS_SHA256)
    return lens


def download_optional(command: list[str], *, timeout: float) -> bool:
    """Run ``colab download``; False only when the remote file is absent."""

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode == 0:
        return True
    output = f"{completed.stdout or ''}{completed.stderr or ''}"
    if REMOTE_FILE_MISSING_MARKER in output.casefold():
        return False
    raise subprocess.CalledProcessError(
        completed.returncode, command, output=completed.stdout, stderr=completed.stderr
    )


def backup_jlens_v2_checkpoint(
    session: str,
    *,
    project: Path,
    staging_root: Path,
    timeout: float,
) -> dict | None:
    """Back up the remote v2 resume bundle through the Colab file API only.

    Never uses ``colab exec`` (which would queue behind a running fit) and never
    stops the session. The previous local bundle is replaced only after the
    exact downloaded triple validates; any failure leaves it untouched.
    """

    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        atomic_install_jlens_v2_resume,
        validate_jlens_v2_resume_bundle,
    )

    deadline = time.monotonic() + timeout
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="jlens-v2-backup-", dir=staging_root))
    try:
        present: dict[str, bool] = {}
        for name in JLENS_V2_BACKUP_ORDER:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"v2 backup exceeded {timeout:g}s before {name}")
            present[name] = download_optional(
                download_command(
                    session, f"{REMOTE_JLENS_V2_ARTIFACTS}/{name}", staging / name
                ),
                timeout=remaining,
            )
        status, events, checkpoint = (staging / name for name in JLENS_V2_BACKUP_ORDER)
        if not present["jlens_fit_checkpoint.pt"]:
            print("v2 checkpoint-backup: no remote checkpoint yet; nothing to back up")
            return None
        if not (present["status.json"] and present["events.jsonl"]):
            raise ValueError("remote v2 checkpoint exists without status.json/events.jsonl")
        meta = validate_jlens_v2_resume_bundle(
            checkpoint=checkpoint, status=status, events=events
        )
        atomic_install_jlens_v2_resume(
            checkpoint=checkpoint, status=status, events=events, project=project
        )
        print(
            "v2 checkpoint-backup: "
            f"run_id={meta['run_id']} n_done={meta['n_done']}/{meta['fit_count']}"
        )
        return meta
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def require_code_eval_sources(project: Path) -> tuple[Path, Path]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_code_eval_sources,
    )

    return require_local_code_eval_sources(project)


def require_lab_sources(project: Path) -> tuple[Path, Path]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_lab_sources,
    )

    return require_local_lab_sources(project)


def require_causal_jlens_sources(project: Path) -> tuple[Path, Path, Path]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_causal_jlens_sources,
    )

    return require_local_causal_jlens_sources(project)


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} not found as a regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def require_sparse_sources(
    project: Path,
) -> tuple[Path, Path, tuple[Path, Path] | None]:
    config_path = project / "configs" / "qwen25_coder_sparse_refusal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    direction = _require_regular_file(
        project / str(config["direction_path"]),
        "sparse frozen direction",
    )
    expected_direction = str(config["direction_sha256"])
    actual_direction = hashlib.sha256(direction.read_bytes()).hexdigest()
    if actual_direction != expected_direction:
        raise ValueError("sparse frozen direction SHA-256 differs from profile")
    pairs = _require_regular_file(
        project
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "pairs"
        / "reviewed.jsonl",
        "sparse reviewed pairs",
    )
    checkpoint = (
        project
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "sae_checkpoint.pt"
    )
    status = checkpoint.with_name("training_status.json")
    checkpoint_exists = checkpoint.exists() or checkpoint.is_symlink()
    status_exists = status.exists() or status.is_symlink()
    if checkpoint_exists != status_exists:
        raise ValueError(
            "sparse checkpoint and training status must be present together"
        )
    resume = (
        (
            _require_regular_file(checkpoint, "sparse checkpoint"),
            _require_regular_file(status, "sparse training status"),
        )
        if checkpoint_exists
        else None
    )
    return direction, pairs, resume


def _write_sparse_pairs_digest(source: Path, destination: Path) -> Path:
    payload = {
        "path": Path(REMOTE_SPARSE_REVIEWED_PAIRS).name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def upload_sparse_sources(
    session: str,
    *,
    project: Path = PROJECT,
    staging_root: Path | None = None,
) -> None:
    direction, pairs, resume = require_sparse_sources(project)
    staging = staging_root or (ROOT / ".colab-staging")
    returncode, output = run_sparse_remote_action(
        session,
        "clean-staging",
        timeout=60,
    )
    if returncode != 0 or sparse_output_has_failure(output):
        raise RuntimeError("sparse remote staging cleanup failed")
    upload_chunked_source(
        session,
        direction,
        REMOTE_SPARSE_DIRECTION_STAGING,
        staging / "sparse-direction-parts",
        chunk_bytes=SPARSE_CHUNK_BYTES,
    )
    digest = _write_sparse_pairs_digest(
        pairs,
        staging / "sparse-reviewed-pairs.sha256.json",
    )
    run(upload_command(session, pairs, REMOTE_SPARSE_REVIEWED_PAIRS))
    run(
        upload_command(
            session,
            digest,
            REMOTE_SPARSE_REVIEWED_PAIRS_DIGEST,
        )
    )
    if resume is not None:
        checkpoint, status = resume
        upload_chunked_source(
            session,
            checkpoint,
            REMOTE_SPARSE_CHECKPOINT_STAGING,
            staging / "sparse-checkpoint-parts",
            chunk_bytes=SPARSE_CHUNK_BYTES,
        )
        run(
            upload_command(
                session,
                status,
                REMOTE_SPARSE_TRAINING_STATUS,
            )
        )


def _write_sparse_remote_action(action: str, destination: Path) -> Path:
    if action not in {
        "clean-staging",
        "pilot",
        "run",
        "pack-checkpoint",
        "pack-results",
    }:
        raise ValueError(f"unknown sparse remote action: {action}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"action": action}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_sparse_remote_action(
    session: str,
    action: str,
    *,
    timeout: int,
) -> tuple[int, str]:
    command_file = _write_sparse_remote_action(
        action,
        ROOT / ".colab-staging" / "qwen_sparse_refusal_command.json",
    )
    run(upload_command(session, command_file, REMOTE_SPARSE_COMMAND))
    command = exec_command(
        session,
        PROJECT / "scripts" / "colab_run_sparse_refusal.py",
        timeout=timeout,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return (
        completed.returncode,
        f"{completed.stdout or ''}{completed.stderr or ''}",
    )


def run_sparse_installer(
    session: str,
    *,
    timeout: int,
) -> tuple[int, str]:
    command = exec_command(
        session,
        PROJECT / "scripts" / "colab_install_sparse_refusal.py",
        timeout=timeout,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return (
        completed.returncode,
        f"{completed.stdout or ''}{completed.stderr or ''}",
    )


def sparse_output_has_failure(output: str) -> bool:
    return any(marker.casefold() in output.casefold() for marker in SPARSE_FAILURE_MARKERS)


def _extract_sparse_archive(
    archive: Path,
    project: Path,
    *,
    allowlist: frozenset[str],
) -> tuple[Path, ...]:
    installed: list[Path] = []
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = [member.name.replace("\\", "/") for member in members]
        if len(names) != len(set(names)):
            raise ValueError("sparse archive contains duplicate members")
        for member, name in zip(members, names, strict=True):
            if (
                name not in allowlist
                or member.issym()
                or member.islnk()
                or not member.isfile()
            ):
                raise ValueError(f"sparse archive member is not allowlisted: {name}")
        staging = Path(tempfile.mkdtemp(prefix="sparse-fetch-", dir=archive.parent))
        try:
            for member, name in zip(members, names, strict=True):
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"sparse archive member is unreadable: {name}")
                destination = staging / Path(name).name
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            incoming: dict[Path, Path] = {}
            backups: dict[Path, Path] = {}
            try:
                for member, name in zip(members, names, strict=True):
                    source = staging / Path(name).name
                    destination = project / Path(name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        destination.name + ".incoming"
                    )
                    incoming[destination] = temporary
                    shutil.copyfile(source, temporary)
                for destination in incoming:
                    if destination.exists():
                        backup = destination.with_name(
                            destination.name + ".sparse-fetch-backup"
                        )
                        if backup.exists():
                            backup.unlink()
                        os.replace(destination, backup)
                        backups[destination] = backup
                for destination, temporary in incoming.items():
                    os.replace(temporary, destination)
                    installed.append(destination)
            except Exception:
                for destination in installed:
                    if destination.exists():
                        destination.unlink()
                for destination, backup in backups.items():
                    if backup.exists():
                        os.replace(backup, destination)
                installed.clear()
                raise
            finally:
                for path in (*incoming.values(), *backups.values()):
                    if path.exists():
                        path.unlink()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return tuple(installed)


def _fetch_sparse_archive(
    session: str,
    *,
    action: str,
    remote_archive: str,
    local_name: str,
    allowlist: frozenset[str],
    project: Path,
    timeout: int,
) -> tuple[Path, ...]:
    returncode, output = run_sparse_remote_action(
        session,
        action,
        timeout=timeout,
    )
    if returncode != 0 or sparse_output_has_failure(output):
        raise RuntimeError(f"sparse {action} failed")
    local = ROOT / ".colab-staging" / local_name
    local.parent.mkdir(parents=True, exist_ok=True)
    run(download_command(session, remote_archive, local))
    return _extract_sparse_archive(local, project, allowlist=allowlist)


def fetch_sparse_checkpoint(
    session: str,
    *,
    project: Path = PROJECT,
    timeout: int = 600,
) -> tuple[Path, ...]:
    prefix = "artifacts/qwen25_coder_sparse_refusal"
    return _fetch_sparse_archive(
        session,
        action="pack-checkpoint",
        remote_archive=REMOTE_SPARSE_CHECKPOINT_ARCHIVE,
        local_name="qwen_sparse_refusal_checkpoint.tar.gz",
        allowlist=frozenset(
            {
                f"{prefix}/sae_checkpoint.pt",
                f"{prefix}/training_status.json",
            }
        ),
        project=project,
        timeout=timeout,
    )


def fetch_sparse_results(
    session: str,
    *,
    project: Path = PROJECT,
    timeout: int = 600,
) -> tuple[Path, ...]:
    prefix = "results/qwen25_coder_sparse_refusal"
    stages = (
        "prepare",
        "hardware_pilot",
        "sae_fit",
        "sae_quality",
        "decompose",
        "discover",
        "select",
        "test",
        "review_export",
        "review_merge",
        "external_transfer",
        "code_preservation",
        "finalize",
    )
    allowlist = {
        f"{prefix}/{index:02d}_{stage}.json"
        for index, stage in enumerate(stages)
    }
    allowlist.update(
        {
            f"{prefix}/selection_freeze.json",
            f"{prefix}/operational_error.json",
            f"{prefix}/execution_manifest.json",
        }
    )
    return _fetch_sparse_archive(
        session,
        action="pack-results",
        remote_archive=REMOTE_SPARSE_RESULTS_ARCHIVE,
        local_name="qwen_sparse_refusal_results.tar.gz",
        allowlist=frozenset(allowlist),
        project=project,
        timeout=timeout,
    )


def _upload_sparse_review_labels(session: str, project: Path = PROJECT) -> None:
    labels = (
        project
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "review"
        / "responses_reviewed.jsonl"
    )
    if labels.is_file() and not labels.is_symlink():
        run(
            upload_command(
                session,
                labels,
                (
                    "/content/refusal_direction/artifacts/"
                    "qwen25_coder_sparse_refusal/review/"
                    "responses_reviewed.jsonl"
                ),
            )
        )


def _handle_sparse_failure(session: str) -> int:
    try:
        fetch_sparse_checkpoint(session)
    except Exception:
        print(
            "WARNING: sparse checkpoint fetch failed; stopping session",
            file=sys.stderr,
        )
    stopped = try_stop_session(session)
    if not stopped:
        print("A100 MAY STILL BE RUNNING", file=sys.stderr)
    status = (
        PROJECT
        / "artifacts"
        / "qwen25_coder_sparse_refusal"
        / "training_status.json"
    )
    if status.is_file():
        try:
            payload = json.loads(status.read_text(encoding="utf-8"))
            stage = payload.get("last_completed_stage") or payload.get("stage")
            if isinstance(stage, str):
                print(f"last_completed_stage={stage}", file=sys.stderr)
        except (OSError, ValueError):
            pass
    return 1


def upload_chunked_source(
    session: str,
    source: Path,
    remote_staging: str,
    parts_dir: Path,
    *,
    chunk_bytes: int,
) -> None:
    sys.path.insert(0, str(PROJECT / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        write_artifact_chunk_manifest,
    )

    staged_name = Path(remote_staging).name
    lens_parts = split_artifact_file(
        source,
        parts_dir,
        chunk_bytes=chunk_bytes,
        part_prefix=staged_name + ".part",
    )
    manifest_path = write_artifact_chunk_manifest(
        source,
        lens_parts,
        parts_dir / (staged_name + ".manifest.json"),
    )
    run(upload_command(session, manifest_path, remote_staging + ".manifest.json"))
    for index, part in enumerate(lens_parts):
        run(
            upload_command(
                session,
                part,
                f"{remote_staging}.part{index:03d}",
            )
        )


def bootstrap_scripts(profile: str) -> list[Path]:
    if profile == "qwen25-coder-lab":
        return [PROJECT / "scripts" / "colab_install_lab.py"]
    if profile == SPARSE_REFUSAL:
        return [PROJECT / "scripts" / "colab_install_sparse_refusal.py"]
    scripts = [PROJECT / "scripts" / "colab_bootstrap.py"]
    if profile == "qwen25-coder-dose":
        scripts.append(PROJECT / "scripts" / "colab_install_dose_source_direction.py")
    if profile == "qwen25-coder-jlens":
        scripts.append(PROJECT / "scripts" / "colab_install_jlens.py")
        scripts.append(PROJECT / "scripts" / "colab_install_jlens_checkpoint.py")
    if profile == JLENS_V2:
        scripts.append(PROJECT / "scripts" / "colab_install_jlens.py")
        scripts.append(PROJECT / "scripts" / "colab_install_jlens_v2_checkpoint.py")
    if profile == "qwen25-coder-code-eval":
        scripts.append(PROJECT / "scripts" / "colab_install_code_eval.py")
    if profile == CAUSAL_JLENS:
        scripts.append(PROJECT / "scripts" / "colab_install_causal_jlens.py")
    return scripts


def generate_lab_credentials() -> tuple[str, str]:
    return "lab_" + secrets.token_urlsafe(9), secrets.token_urlsafe(24)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_captured(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return f"{completed.stdout or ''}{completed.stderr or ''}"


def cleanup_lab_local_staging(root: Path) -> None:
    staging = root / ".colab-staging"
    auth = staging / "qwen_lab_auth.json"
    if auth.exists() or os.path.lexists(auth):
        try:
            auth.unlink()
        except FileNotFoundError:
            pass
    parts_dir = staging / "lab-lens-parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir, ignore_errors=True)


def extract_public_lab_url(text: str) -> str:
    sys.path.insert(0, str(PROJECT / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_lab_public_url,
    )

    url = extract_lab_public_url(text)
    if not url:
        raise ValueError("lab public URL missing from remote output")
    return url


def write_session_auth_file(path: Path, username: str, password: str) -> Path:
    sys.path.insert(0, str(PROJECT / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        write_lab_auth_file,
    )

    return write_lab_auth_file(path, username=username, password=password)


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return None
    if info.name.endswith("_output.ipynb"):
        return None
    return info


def current_implementation_commit(cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else ""
    if not commit or commit.casefold() == "unknown":
        raise ValueError("implementation commit is missing")
    return commit


def build_archive(
    *,
    stamp_commit: bool = False,
    project: Path | None = None,
    staging: Path | None = None,
    commit: str | None = None,
) -> Path:
    project = project or PROJECT
    staging = staging or (ROOT / ".colab-staging")
    staging.mkdir(exist_ok=True)
    archive = staging / "refusal_direction.tar.gz"

    def filter_info(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        filtered = _archive_filter(info)
        if filtered is None:
            return None
        if stamp_commit and Path(info.name).name == "IMPLEMENTATION_COMMIT":
            return None
        return filtered

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(
            project,
            arcname="refusal_direction",
            filter=filter_info,
        )
        if stamp_commit:
            hashed = commit or current_implementation_commit()
            payload = (hashed + "\n").encode("utf-8")
            info = tarfile.TarInfo(name="refusal_direction/IMPLEMENTATION_COMMIT")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
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
            "lab",
            "causal-jlens",
            "checkpoint-backup",
            "sparse-prepare",
            "sparse-pilot",
            "sparse-run",
            "sparse-fetch-checkpoint",
            "sparse-fetch-results",
            "stop",
        ),
    )
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SESSIONS),
        default="gemma",
    )
    parser.add_argument(
        "--gpu",
        choices=("T4", "L4", "A100"),
        default="L4",
    )
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()
    session = resolve_session(args.profile, args.session)
    explicit_timeout = args.timeout
    if args.timeout is None:
        args.timeout = (
            JLENS_V2_EXEC_TIMEOUT
            if args.action == "jlens" and args.profile == JLENS_V2
            else DEFAULT_EXEC_TIMEOUT
        )

    if args.action == "probe":
        run([*AUTH, "sessions"])
    elif args.action == "start":
        gpu = "A100" if args.profile in A100_PROFILES else args.gpu
        run(start_command(session, gpu))
    elif args.action == "sync":
        direction = None
        code_eval_sources = None
        lab_sources = None
        causal_sources = None
        sparse_sources = args.profile == SPARSE_REFUSAL
        if args.profile == "qwen25-coder-dose":
            direction = require_dose_source_direction(PROJECT)
        elif args.profile == "qwen25-coder-code-eval":
            code_eval_sources = require_code_eval_sources(PROJECT)
        elif args.profile == "qwen25-coder-lab":
            lab_sources = require_lab_sources(PROJECT)
        elif args.profile == CAUSAL_JLENS:
            causal_sources = require_causal_jlens_sources(PROJECT)
        archive = build_archive(
            stamp_commit=stamps_implementation_commit(args.profile)
        )
        run(
            upload_command(
                session,
                archive,
                "/content/refusal_direction.tar.gz",
            )
        )
        if direction is not None:
            run(
                upload_command(
                    session,
                    direction,
                    "/content/dose_source_direction.pt",
                )
            )
        if code_eval_sources is not None:
            source_direction, source_lens = code_eval_sources
            run(
                upload_command(
                    session,
                    source_direction,
                    "/content/code_eval_source_direction.pt",
                )
            )
            lens_parts = split_code_eval_lens(
                source_lens,
                ROOT / ".colab-staging" / "code-eval-lens-parts",
            )
            for index, part in enumerate(lens_parts):
                run(
                    upload_command(
                        session,
                        part,
                        f"{REMOTE_CODE_EVAL_LENS_STAGING}.part{index:03d}",
                    )
                )
        if lab_sources is not None:
            source_direction, source_lens = lab_sources
            run(
                exec_command(
                    session,
                    PROJECT / "scripts" / "colab_clean_lab_chunks.py",
                    timeout=60,
                )
            )
            run(
                upload_command(
                    session,
                    source_direction,
                    REMOTE_LAB_DIRECTION_STAGING,
                )
            )
            part_prefix = Path(REMOTE_LAB_LENS_STAGING).name + ".part"
            lens_parts = split_artifact_file(
                source_lens,
                ROOT / ".colab-staging" / "lab-lens-parts",
                chunk_bytes=LAB_LENS_CHUNK_BYTES,
                part_prefix=part_prefix,
            )
            sys.path.insert(0, str(PROJECT / "src"))
            from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
                write_artifact_chunk_manifest,
            )

            manifest_path = (
                ROOT
                / ".colab-staging"
                / "lab-lens-parts"
                / (Path(REMOTE_LAB_LENS_STAGING).name + ".manifest.json")
            )
            write_artifact_chunk_manifest(source_lens, lens_parts, manifest_path)
            run(
                upload_command(
                    session,
                    manifest_path,
                    REMOTE_LAB_LENS_MANIFEST,
                )
            )
            for index, part in enumerate(lens_parts):
                run(
                    upload_command(
                        session,
                        part,
                        f"{REMOTE_LAB_LENS_STAGING}.part{index:03d}",
                    )
                )
        if args.profile == "qwen25-coder-jlens":
            for local_path, remote in optional_jlens_resume_bundle(PROJECT):
                run(upload_command(session, local_path, remote))
        if args.profile == JLENS_V2:
            sys.path.insert(0, str(PROJECT / "src"))
            from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
                write_artifact_chunk_manifest,
            )

            source_lens = require_jlens_v2_source_v1_lens(PROJECT)
            parts_dir = ROOT / ".colab-staging" / "jlens-v2-lens-parts"
            staged_name = Path(REMOTE_JLENS_V2_LENS_STAGING).name
            lens_parts = split_artifact_file(
                source_lens,
                parts_dir,
                chunk_bytes=JLENS_V2_LENS_CHUNK_BYTES,
                part_prefix=staged_name + ".part",
            )
            manifest_path = write_artifact_chunk_manifest(
                source_lens,
                lens_parts,
                parts_dir / (staged_name + ".manifest.json"),
            )
            run(upload_command(session, manifest_path, REMOTE_JLENS_V2_LENS_MANIFEST))
            for index, part in enumerate(lens_parts):
                run(
                    upload_command(
                        session,
                        part,
                        f"{REMOTE_JLENS_V2_LENS_STAGING}.part{index:03d}",
                    )
                )
            for local_path, remote in optional_jlens_v2_resume_bundle(PROJECT):
                run(upload_command(session, local_path, remote))
        if causal_sources is not None:
            source_direction, source_v1, source_v2 = causal_sources
            run(
                exec_command(
                    session,
                    PROJECT / "scripts" / "colab_clean_causal_jlens_chunks.py",
                    timeout=60,
                )
            )
            upload_chunked_source(
                session,
                source_direction,
                REMOTE_CAUSAL_JLENS_DIRECTION_STAGING,
                ROOT / ".colab-staging" / "causal-direction-parts",
                chunk_bytes=CAUSAL_JLENS_LENS_CHUNK_BYTES,
            )
            upload_chunked_source(
                session,
                source_v1,
                REMOTE_CAUSAL_JLENS_V1_LENS_STAGING,
                ROOT / ".colab-staging" / "causal-v1-lens-parts",
                chunk_bytes=CAUSAL_JLENS_LENS_CHUNK_BYTES,
            )
            upload_chunked_source(
                session,
                source_v2,
                REMOTE_CAUSAL_JLENS_V2_LENS_STAGING,
                ROOT / ".colab-staging" / "causal-v2-lens-parts",
                chunk_bytes=CAUSAL_JLENS_LENS_CHUNK_BYTES,
            )
        if sparse_sources:
            upload_sparse_sources(session)
    elif args.action == "bootstrap":
        if args.profile == SPARSE_REFUSAL:
            returncode, output = run_sparse_installer(
                session,
                timeout=600,
            )
            if returncode != 0 or sparse_output_has_failure(output):
                return _handle_sparse_failure(session)
        else:
            for script in bootstrap_scripts(args.profile):
                run(
                    exec_command(
                        session,
                        script,
                        timeout=600,
                    )
                )
    elif args.action == "sparse-prepare":
        if args.profile != SPARSE_REFUSAL:
            parser.error(
                f"sparse-prepare is only available for --profile {SPARSE_REFUSAL}"
            )
        upload_sparse_sources(session)
        returncode, output = run_sparse_installer(
            session,
            timeout=args.timeout,
        )
        if returncode != 0 or sparse_output_has_failure(output):
            return _handle_sparse_failure(session)
    elif args.action == "auth-hf":
        token = huggingface_token_path()
        if not token.is_file():
            parser.error(
                "Hugging Face token not found; run `uv run hf auth login`"
            )
        run(
            upload_command(
                session,
                token,
                "/content/hf_token",
            )
        )
        run(
            exec_command(
                session,
                PROJECT / "scripts" / "colab_prepare_hf_auth.py",
                timeout=60,
            )
        )
    elif args.action == "execute":
        if args.notebook is None:
            parser.error("--notebook is required for execute")
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            parser.error(f"notebook does not exist: {notebook}")
        run(
            exec_command(
                session,
                notebook,
                timeout=args.timeout,
            )
        )
    elif args.action == "fetch":
        if args.profile == "qwen25-coder-lab":
            parser.error("fetch is not available for --profile qwen25-coder-lab")
        if args.profile == "qwen25-coder-code-eval":
            remote_archive = "/content/qwen_code_eval_results.tar.gz"
            local_name = "qwen_code_eval_results.tar.gz"
        elif args.profile == "qwen25-coder-jlens":
            remote_archive = "/content/qwen_jlens_results.tar.gz"
            local_name = "qwen_jlens_results.tar.gz"
        elif args.profile == JLENS_V2:
            remote_archive = "/content/qwen_jlens_v2_results.tar.gz"
            local_name = "qwen_jlens_v2_results.tar.gz"
        elif args.profile == CAUSAL_JLENS:
            remote_archive = "/content/qwen_causal_jlens_results.tar.gz"
            local_name = "qwen_causal_jlens_results.tar.gz"
        elif args.profile == "qwen25-coder-dose":
            remote_archive = "/content/qwen_dose_results.tar.gz"
            local_name = "qwen_dose_results.tar.gz"
        elif args.profile == "qwen25-coder":
            remote_archive = "/content/qwen_results.tar.gz"
            local_name = "qwen_results.tar.gz"
        else:
            remote_archive = "/content/refusal_results.tar.gz"
            local_name = "refusal_results.tar.gz"
        run(
            exec_command(
                session,
                fetch_pack_script(args.profile),
                timeout=fetch_pack_timeout(args.profile),
            )
        )
        local = ROOT / ".colab-staging" / local_name
        local.parent.mkdir(exist_ok=True)
        run(
            download_command(
                session,
                remote_archive,
                local,
            )
        )
        if args.profile == "qwen25-coder-code-eval":
            extract_qwen_code_eval_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder-jlens":
            extract_qwen_jlens_results_archive(local, PROJECT)
        elif args.profile == JLENS_V2:
            extract_qwen_jlens_v2_results_archive(local, PROJECT)
        elif args.profile == CAUSAL_JLENS:
            extract_qwen_causal_jlens_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder-dose":
            extract_qwen_dose_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder":
            extract_qwen_results_archive(local, PROJECT)
        else:
            with tarfile.open(local, "r:gz") as handle:
                handle.extractall(PROJECT)
    elif args.action == "status":
        if args.profile not in (
            "qwen25-coder",
            "qwen25-coder-dose",
            "qwen25-coder-jlens",
            "qwen25-coder-code-eval",
            "qwen25-coder-lab",
            JLENS_V2,
            CAUSAL_JLENS,
        ):
            parser.error("status is only available for Qwen profiles")
        run(status_command(session, profile=args.profile))
    elif args.action == "pilot":
        if args.profile != "qwen25-coder":
            parser.error("pilot is only available for --profile qwen25-coder")
        run(pilot_command(session, timeout=args.timeout))
    elif args.action == "dose":
        if args.profile != "qwen25-coder-dose":
            parser.error("dose is only available for --profile qwen25-coder-dose")
        run(dose_command(session, timeout=args.timeout))
    elif args.action == "jlens":
        if args.profile not in ("qwen25-coder-jlens", JLENS_V2):
            parser.error(
                "jlens is only available for --profile qwen25-coder-jlens or "
                f"{JLENS_V2}"
            )
        run(jlens_command(session, timeout=args.timeout, profile=args.profile))
    elif args.action == "code-eval":
        if args.profile != "qwen25-coder-code-eval":
            parser.error(
                "code-eval is only available for --profile qwen25-coder-code-eval"
            )
        run(code_eval_command(session, timeout=args.timeout))
    elif args.action == "causal-jlens":
        if args.profile != CAUSAL_JLENS:
            parser.error(
                "causal-jlens is only available for --profile "
                "qwen25-coder-causal-jlens"
            )
        run(causal_jlens_command(session, timeout=args.timeout))
    elif args.action in ("sparse-pilot", "sparse-run"):
        if args.profile != SPARSE_REFUSAL:
            parser.error(
                f"{args.action} is only available for --profile {SPARSE_REFUSAL}"
            )
        if args.action == "sparse-run":
            _upload_sparse_review_labels(session)
        returncode, output = run_sparse_remote_action(
            session,
            "pilot" if args.action == "sparse-pilot" else "run",
            timeout=args.timeout,
        )
        if returncode != 0 or sparse_output_has_failure(output):
            return _handle_sparse_failure(session)
    elif args.action == "sparse-fetch-checkpoint":
        if args.profile != SPARSE_REFUSAL:
            parser.error(
                "sparse-fetch-checkpoint is only available for --profile "
                f"{SPARSE_REFUSAL}"
            )
        fetch_sparse_checkpoint(session, timeout=args.timeout)
    elif args.action == "sparse-fetch-results":
        if args.profile != SPARSE_REFUSAL:
            parser.error(
                "sparse-fetch-results is only available for --profile "
                f"{SPARSE_REFUSAL}"
            )
        fetch_sparse_results(session, timeout=args.timeout)
    elif args.action == "lab":
        if args.profile != "qwen25-coder-lab":
            parser.error("lab is only available for --profile qwen25-coder-lab")
        auth_path = ROOT / ".colab-staging" / "qwen_lab_auth.json"
        try:
            username, password = generate_lab_credentials()
            write_session_auth_file(auth_path, username, password)
            run(
                upload_command(
                    session,
                    auth_path,
                    REMOTE_LAB_AUTH_FILE,
                )
            )
            remote_output = run_captured(
                lab_command(session, timeout=args.timeout)
            )
            public_url = extract_public_lab_url(remote_output)
            print(f"Public URL: {public_url}")
            print(f"Username: {username}")
            print(f"Password: {password}")
            print(
                "WARNING: This Gradio share link uses an external relay. "
                "Prompts and responses transit that service during the session."
            )
            print("Always stop the A100 when finished:")
            print(
                "  python tools/colab/run_refusal.py stop --profile qwen25-coder-lab"
            )
        finally:
            if auth_path.exists() or os.path.lexists(auth_path):
                try:
                    auth_path.unlink()
                except FileNotFoundError:
                    pass
    elif args.action == "checkpoint-backup":
        if args.profile != JLENS_V2:
            parser.error(f"checkpoint-backup is only available for --profile {JLENS_V2}")
        backup_jlens_v2_checkpoint(
            session,
            project=PROJECT,
            staging_root=ROOT / ".colab-staging",
            timeout=(
                explicit_timeout
                if explicit_timeout is not None
                else JLENS_V2_BACKUP_TIMEOUT_SECONDS
            ),
        )
    elif args.action == "stop":
        if args.profile == "qwen25-coder-lab":
            cleanup_lab_local_staging(ROOT)
        backup_failed = False
        stop_ok = True
        try:
            if args.profile == JLENS_V2:
                try:
                    backup_jlens_v2_checkpoint(
                        session,
                        project=PROJECT,
                        staging_root=ROOT / ".colab-staging",
                        timeout=STOP_BACKUP_TIMEOUT_SECONDS,
                    )
                except Exception as exc:  # noqa: BLE001 - stop must still run
                    backup_failed = True
                    print(
                        "WARNING: final v2 checkpoint backup failed "
                        f"({type(exc).__name__}: {exc}); the previous local "
                        "bundle was kept. Stopping the session anyway.",
                        file=sys.stderr,
                    )
        finally:
            if args.profile in {JLENS_V2, CAUSAL_JLENS, SPARSE_REFUSAL}:
                stop_ok = try_stop_session(session)
            else:
                run(stop_command(session))
        if not stop_ok:
            print("A100 MAY STILL BE RUNNING", file=sys.stderr)
            print(
                format_stop_rerun_command(profile=args.profile, session=args.session),
                file=sys.stderr,
            )
            return 1
        if backup_failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())