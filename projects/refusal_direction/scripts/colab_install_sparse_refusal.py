"""Install the isolated sparse-refusal project and private run inputs.

The module is import-safe on local and Windows hosts. Remote work happens only
from ``main``. Large inputs use manifests with explicit chunk names, sizes and
SHA-256 hashes; reviewed text is copied privately and never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping


ARCHIVE = Path("/content/refusal_direction.tar.gz")
CONTENT_ROOT = Path("/content")
REMOTE_ROOT = CONTENT_ROOT / "refusal_direction"
REMOTE_DIRECTION_STAGING = Path("/content/sparse_refusal_direction.pt")
REMOTE_DIRECTION_MANIFEST = Path(
    "/content/sparse_refusal_direction.pt.manifest.json"
)
REMOTE_REVIEWED_PAIRS = Path("/content/sparse_refusal_reviewed_pairs.jsonl")
REMOTE_REVIEWED_PAIRS_DIGEST = Path(
    "/content/sparse_refusal_reviewed_pairs.sha256.json"
)
REMOTE_CHECKPOINT_STAGING = Path("/content/sparse_refusal_checkpoint.pt")
REMOTE_CHECKPOINT_MANIFEST = Path(
    "/content/sparse_refusal_checkpoint.pt.manifest.json"
)
REMOTE_CHECKPOINT_STATUS = Path("/content/sparse_refusal_checkpoint_status.json")
DIRECTION_RELATIVE = Path(
    "artifacts/qwen25_coder_7b/selected_direction.pt"
)
SPARSE_ARTIFACTS_RELATIVE = Path(
    "artifacts/qwen25_coder_sparse_refusal"
)
REVIEWED_PAIRS_RELATIVE = (
    SPARSE_ARTIFACTS_RELATIVE / "pairs" / "reviewed.jsonl"
)
CHECKPOINT_RELATIVE = SPARSE_ARTIFACTS_RELATIVE / "sae_checkpoint.pt"
CHECKPOINT_STATUS_RELATIVE = SPARSE_ARTIFACTS_RELATIVE / "checkpoint_status.json"
EXPECTED_DIRECTION_SHA256 = (
    "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
)
COPY_BUFFER_BYTES = 8 * 1024**2
PINNED_UV_VERSION = "0.12.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("chunk manifest must be an object")
    if payload.get("version") != 1:
        raise ValueError("chunk manifest version is unsupported")
    entries = payload.get("parts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("chunk manifest parts must be nonempty")
    if payload.get("part_count") != len(entries):
        raise ValueError("chunk manifest part_count differs from parts")
    _require_sha256(payload.get("sha256"), "chunk manifest sha256")
    return payload


def assemble_verified_chunks(
    destination: Path,
    manifest_path: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Assemble only manifest-named chunks and verify each byte identity."""

    destination = Path(destination)
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    entries = list(manifest["parts"])
    expected_names = [str(entry.get("name", "")) for entry in entries]
    if any(
        not name
        or Path(name).name != name
        or name in {".", ".."}
        for name in expected_names
    ):
        raise ValueError("chunk manifest contains an unsafe part name")
    present = sorted(
        path.name
        for path in destination.parent.iterdir()
        if path.name.startswith(destination.name + ".part")
    )
    if present != expected_names:
        raise ValueError("chunk parts differ from the explicit manifest")
    temporary = destination.with_name(destination.name + ".assembling")
    try:
        with temporary.open("wb") as output:
            for entry, name in zip(entries, expected_names, strict=True):
                part = destination.parent / name
                expected_size = entry.get("size")
                if (
                    isinstance(expected_size, bool)
                    or not isinstance(expected_size, int)
                    or expected_size <= 0
                    or part.stat().st_size != expected_size
                ):
                    raise ValueError(f"chunk size differs for {name}")
                expected_part_hash = _require_sha256(
                    entry.get("sha256"), f"{name} sha256"
                )
                if sha256_file(part) != expected_part_hash:
                    raise ValueError(f"chunk sha256 differs for {name}")
                with part.open("rb") as incoming:
                    shutil.copyfileobj(
                        incoming,
                        output,
                        length=COPY_BUFFER_BYTES,
                    )
            output.flush()
            os.fsync(output.fileno())
        expected_size = manifest.get("total_size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or temporary.stat().st_size != expected_size
        ):
            raise ValueError("assembled size differs from manifest")
        manifest_hash = _require_sha256(
            manifest.get("sha256"), "assembled sha256"
        )
        actual_hash = sha256_file(temporary)
        if actual_hash != manifest_hash:
            raise ValueError("assembled sha256 differs from manifest")
        if expected_sha256 is not None and actual_hash != _require_sha256(
            expected_sha256, "expected sha256"
        ):
            raise ValueError("assembled sha256 differs from expected sha256")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    for name in expected_names:
        (destination.parent / name).unlink()
    manifest_path.unlink()
    return destination


def _atomic_copy(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            with source.open("rb") as incoming:
                shutil.copyfileobj(
                    incoming,
                    output,
                    length=COPY_BUFFER_BYTES,
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _copy_to_temporary(
    source: Path,
    destination: Path,
    *,
    kind: str,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{kind}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with Path(source).open("rb") as incoming:
                shutil.copyfileobj(
                    incoming,
                    output,
                    length=COPY_BUFFER_BYTES,
                )
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def atomic_install_checkpoint_pair(
    checkpoint_source: Path,
    status_source: Path,
    *,
    project: Path,
) -> tuple[Path, Path]:
    checkpoint_destination = Path(project) / CHECKPOINT_RELATIVE
    status_destination = Path(project) / CHECKPOINT_STATUS_RELATIVE
    checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    incoming_checkpoint: Path | None = None
    incoming_status: Path | None = None
    try:
        incoming_checkpoint = _copy_to_temporary(
            checkpoint_source,
            checkpoint_destination,
            kind="incoming",
        )
        incoming_status = _copy_to_temporary(
            status_source,
            status_destination,
            kind="incoming",
        )
    except Exception:
        if incoming_checkpoint is not None and incoming_checkpoint.exists():
            incoming_checkpoint.unlink()
        raise
    assert incoming_checkpoint is not None and incoming_status is not None
    destinations = (checkpoint_destination, status_destination)
    incoming = (incoming_checkpoint, incoming_status)
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    try:
        for destination in destinations:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.backup.{os.getpid()}"
                )
                if backup.exists():
                    backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in zip(incoming, destinations, strict=True):
            os.replace(source, destination)
            installed.add(destination)
    except Exception:
        for destination in installed:
            if destination.exists():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for path in (*incoming, *backups.values()):
            if path.exists():
                path.unlink()
    return checkpoint_destination, status_destination


def install_sparse_inputs(
    *,
    project: Path,
    direction_staging: Path,
    direction_manifest: Path,
    reviewed_pairs_staging: Path,
    reviewed_pairs_sha256: str,
    checkpoint_staging: Path | None = None,
    checkpoint_manifest: Path | None = None,
    checkpoint_status_staging: Path | None = None,
    expected_direction_sha256: str = EXPECTED_DIRECTION_SHA256,
) -> dict[str, object]:
    """Install the frozen direction, reviewed pairs and optional resume pair."""

    project = Path(project)
    direction = assemble_verified_chunks(
        Path(direction_staging),
        Path(direction_manifest),
        expected_sha256=expected_direction_sha256,
    )
    direction_destination = _atomic_copy(
        direction,
        project / DIRECTION_RELATIVE,
    )
    reviewed_pairs_staging = Path(reviewed_pairs_staging)
    expected_pairs_hash = _require_sha256(
        reviewed_pairs_sha256, "reviewed pairs sha256"
    )
    actual_pairs_hash = sha256_file(reviewed_pairs_staging)
    if actual_pairs_hash != expected_pairs_hash:
        raise ValueError("reviewed pairs sha256 differs from expected sha256")
    pairs_destination = _atomic_copy(
        reviewed_pairs_staging,
        project / REVIEWED_PAIRS_RELATIVE,
    )
    checkpoint_hash: str | None = None
    if any(
        value is not None
        for value in (
            checkpoint_staging,
            checkpoint_manifest,
            checkpoint_status_staging,
        )
    ):
        if (
            checkpoint_staging is None
            or checkpoint_manifest is None
            or checkpoint_status_staging is None
        ):
            raise ValueError(
                "checkpoint, manifest and checkpoint status must travel together"
            )
        status_payload = json.loads(
            Path(checkpoint_status_staging).read_text(encoding="utf-8")
        )
        if not isinstance(status_payload, dict):
            raise ValueError("checkpoint status must be a JSON object")
        if set(status_payload) != {
            "schema_version",
            "identity",
            "checkpoint_sha256",
        }:
            raise ValueError("checkpoint status schema is invalid")
        if status_payload["schema_version"] != 1:
            raise ValueError("checkpoint status version is unsupported")
        identity = status_payload["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "run_id",
            "implementation_commit",
            "config_sha256",
            "corpus_sha256",
            "model_revision",
        }:
            raise ValueError("checkpoint status identity is incomplete")
        checkpoint = assemble_verified_chunks(
            Path(checkpoint_staging),
            Path(checkpoint_manifest),
        )
        checkpoint_hash = sha256_file(checkpoint)
        if status_payload["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("checkpoint status SHA-256 mismatch")
        source = str(project / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from refusal_direction_replication.qwen_sparse_training import (
            load_training_checkpoint,
        )

        load_training_checkpoint(checkpoint, identity)
        atomic_install_checkpoint_pair(
            checkpoint,
            Path(checkpoint_status_staging),
            project=project,
        )
    return {
        "direction_path": DIRECTION_RELATIVE.as_posix(),
        "direction_sha256": sha256_file(direction_destination),
        "reviewed_pairs_path": REVIEWED_PAIRS_RELATIVE.as_posix(),
        "reviewed_pairs_sha256": sha256_file(pairs_destination),
        "checkpoint_present": checkpoint_hash is not None,
        "checkpoint_sha256": checkpoint_hash,
    }


def safe_extract_repository(
    archive: Path = ARCHIVE,
    destination: Path = CONTENT_ROOT,
) -> None:
    destination = Path(destination)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError("repository archive links are not allowed")
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"unsupported repository member: {member.name}"
                )
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe repository member: {member.name}")
        handle.extractall(destination)


def install_locked_project(project: Path = REMOTE_ROOT) -> None:
    """Install the archived project from its immutable dependency lock."""

    pyproject = Path(project) / "pyproject.toml"
    lock = Path(project) / "uv.lock"
    if not pyproject.is_file():
        raise FileNotFoundError(f"project metadata missing: {pyproject}")
    if not lock.is_file():
        raise FileNotFoundError(f"dependency lock missing: {lock}")
    version_command = [sys.executable, "-m", "uv", "--version"]
    try:
        version = subprocess.run(
            version_command,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                f"uv=={PINNED_UV_VERSION}",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        version = subprocess.run(
            version_command,
            check=True,
            text=True,
            capture_output=True,
        )
    if not str(version.stdout).startswith(f"uv {PINNED_UV_VERSION} "):
        raise RuntimeError(
            f"bootstrap requires uv {PINNED_UV_VERSION}"
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--project",
            str(project),
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def _reviewed_pairs_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"path", "sha256"}:
        raise ValueError("reviewed-pairs digest manifest is invalid")
    if payload["path"] != REMOTE_REVIEWED_PAIRS.name:
        raise ValueError("reviewed-pairs digest path is invalid")
    return _require_sha256(payload["sha256"], "reviewed pairs sha256")


def main() -> int:
    safe_extract_repository()
    install_locked_project()
    checkpoint_present = (
        REMOTE_CHECKPOINT_MANIFEST.is_file()
        or REMOTE_CHECKPOINT_STAGING.is_file()
        or REMOTE_CHECKPOINT_STATUS.is_file()
    )
    summary = install_sparse_inputs(
        project=REMOTE_ROOT,
        direction_staging=REMOTE_DIRECTION_STAGING,
        direction_manifest=REMOTE_DIRECTION_MANIFEST,
        reviewed_pairs_staging=REMOTE_REVIEWED_PAIRS,
        reviewed_pairs_sha256=_reviewed_pairs_hash(
            REMOTE_REVIEWED_PAIRS_DIGEST
        ),
        checkpoint_staging=(
            REMOTE_CHECKPOINT_STAGING if checkpoint_present else None
        ),
        checkpoint_manifest=(
            REMOTE_CHECKPOINT_MANIFEST if checkpoint_present else None
        ),
        checkpoint_status_staging=(
            REMOTE_CHECKPOINT_STATUS if checkpoint_present else None
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
