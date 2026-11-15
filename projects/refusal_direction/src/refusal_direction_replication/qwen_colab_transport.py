"""Safe Qwen-only Colab result packing and local extraction."""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

QWEN_RESULTS_DIR = "results/qwen25_coder_7b"
QWEN_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b"
REMOTE_ARCHIVE = Path("/content/qwen_results.tar.gz")

QWEN_FETCH_ALLOWLIST = frozenset(
    {
        f"{QWEN_ARTIFACTS_DIR}/status.json",
        f"{QWEN_ARTIFACTS_DIR}/events.jsonl",
        f"{QWEN_ARTIFACTS_DIR}/candidates.pt",
        f"{QWEN_ARTIFACTS_DIR}/selected_direction.pt",
        f"{QWEN_ARTIFACTS_DIR}/jlens_fit_checkpoint.pt",
        f"{QWEN_ARTIFACTS_DIR}/qwen25_coder_7b_jacobian_lens.pt",
    }
)

RESULT_JSON_PATTERN = re.compile(
    r"^results/qwen25_coder_7b/[^/]+\.json$"
)

MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 4 * 1024**3


def normalize_member_name(name: str) -> str:
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/"):
        raise ValueError(f"unsafe archive path: {name}")
    if ".." in PurePosixPath(cleaned).parts:
        raise ValueError(f"unsafe archive path: {name}")
    trimmed = cleaned
    while trimmed.startswith("./"):
        trimmed = trimmed[2:]
    if not trimmed:
        raise ValueError(f"unsafe archive path: {name}")
    return str(PurePosixPath(trimmed))


def is_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    if not normalized.startswith(
        ("results/qwen25_coder_7b/", "artifacts/qwen25_coder_7b/")
    ):
        return True
    return False


def is_allowed_fetch_member(name: str) -> bool:
    normalized = normalize_member_name(name)
    if is_unsafe_member_name(normalized):
        return False
    if normalized in QWEN_FETCH_ALLOWLIST:
        return True
    return bool(RESULT_JSON_PATTERN.fullmatch(normalized))


def collect_qwen_pack_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    results_dir = root / "results" / "qwen25_coder_7b"
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json")):
            if path.is_file() and not path.is_symlink():
                entries.append(path)
    for relative in sorted(QWEN_FETCH_ALLOWLIST):
        if relative.startswith(QWEN_RESULTS_DIR):
            continue
        path = root / Path(relative)
        if path.is_file() and not path.is_symlink():
            entries.append(path)
    return entries


def pack_qwen_results(root: Path, archive: Path) -> None:
    entries = collect_qwen_pack_entries(root)
    if not entries:
        raise FileNotFoundError("no Qwen artifacts available to pack")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for path in entries:
            arcname = normalize_member_name(
                str(path.relative_to(root)).replace("\\", "/")
            )
            handle.add(path, arcname=arcname)


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    if member.issym() or member.islnk():
        raise ValueError(f"archive member is a link: {member.name}")
    if member.isdir():
        raise ValueError(f"archive member is a directory: {member.name}")
    if not member.isreg():
        raise ValueError(f"archive member is not a regular file: {member.name}")
    normalized = normalize_member_name(member.name)
    if is_unsafe_member_name(normalized):
        raise ValueError(f"unsafe archive path: {member.name}")
    if not is_allowed_fetch_member(normalized):
        raise ValueError(f"archive member outside allowlist: {member.name}")
    if member.size > MAX_MEMBER_BYTES:
        raise ValueError(f"archive member exceeds size limit: {member.name}")


def validate_qwen_archive(members: list[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    seen: set[str] = set()
    total_size = 0
    file_members: list[tarfile.TarInfo] = []
    for member in members:
        _validate_tar_member(member)
        normalized = normalize_member_name(member.name)
        if normalized in seen:
            raise ValueError(f"duplicate archive member: {member.name}")
        seen.add(normalized)
        total_size += member.size
        file_members.append(member)
    if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError("archive exceeds total uncompressed size limit")
    return file_members


@dataclass
class InstallRecord:
    target: Path
    backup: Path | None
    had_original: bool


def _rollback_install_records(records: list[InstallRecord]) -> None:
    for record in reversed(records):
        if record.backup is not None and record.backup.exists():
            if record.target.exists():
                record.target.unlink()
            os.replace(record.backup, record.target)
        elif record.had_original and record.target.exists():
            record.target.unlink()
        elif not record.had_original and record.target.exists():
            record.target.unlink()


def extract_qwen_archive(archive: Path, project: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        file_members = validate_qwen_archive(members)
        staging = Path(tempfile.mkdtemp(prefix="qwen-fetch-", dir=project.parent))
        records: list[InstallRecord] = []
        try:
            for member in file_members:
                destination = staging / normalize_member_name(member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to extract member: {member.name}")
                with destination.open("wb") as out:
                    shutil.copyfileobj(extracted, out)

            for member in file_members:
                relative = normalize_member_name(member.name)
                target = project / relative
                source = staging / relative
                backup: Path | None = None
                had_original = target.exists()
                if had_original:
                    backup = target.with_suffix(target.suffix + ".qwen-fetch-bak")
                    os.replace(target, backup)
                records.append(
                    InstallRecord(target=target, backup=backup, had_original=had_original)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)

            for record in records:
                if record.backup is not None and record.backup.exists():
                    record.backup.unlink()
        except Exception:
            _rollback_install_records(records)
            shutil.rmtree(staging, ignore_errors=True)
            raise
        else:
            shutil.rmtree(staging, ignore_errors=True)