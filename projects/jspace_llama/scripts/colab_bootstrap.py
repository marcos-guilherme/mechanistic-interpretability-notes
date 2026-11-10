from __future__ import annotations

import importlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ARCHIVE = Path("/content/jspace_llama.tar.gz")
DESTINATION = Path("/content")
ARCHIVE_ROOT = "jspace_llama"
PROJECT = DESTINATION / "jspace_llama_project"
LEGACY_PROJECT = DESTINATION / "jspace_llama"
IMPORT_PACKAGE = "jspace_llama"
LOCKED_REQUIREMENTS = Path("/tmp/jspace-locked-requirements.txt")
SUCCESS_MARKER = "JSPACE_BOOTSTRAP_SUCCESS"
SUCCESS_NONCE_ENV = "JSPACE_SUCCESS_NONCE"


def consume_success_nonce() -> str:
    nonce = os.environ.pop(SUCCESS_NONCE_ENV, None)
    if nonce is None or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise RuntimeError("missing or invalid success nonce")
    return nonce


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        for line in completed.stdout.splitlines():
            print(f"[child stdout] {line}")
    if completed.stderr:
        for line in completed.stderr.splitlines():
            print(f"[child stderr] {line}", file=sys.stderr)
    completed.check_returncode()


def _validate_archive_metadata(path: Path, metadata: object) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"project archive is not a regular file: {path}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"project archive is hard-linked: {path}")


def _open_validated_archive(path: Path, metadata: object):
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        actual = os.fstat(descriptor)
        if (
            metadata.st_dev != actual.st_dev
            or metadata.st_ino != actual.st_ino
            or stat.S_IFMT(metadata.st_mode) != stat.S_IFMT(actual.st_mode)
            or getattr(metadata, "st_nlink", 1)
            != getattr(actual, "st_nlink", 1)
            or metadata.st_size != actual.st_size
        ):
            raise ValueError(f"project archive changed after validation: {path}")
        _validate_archive_metadata(path, actual)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _validated_member_path(
    member: tarfile.TarInfo,
    destination: Path,
) -> Path:
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError("project archive must contain regular files only")
    if "\\" in member.name:
        raise ValueError(f"unsafe archive member: {member.name}")
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute():
        raise ValueError(f"unsafe archive member: {member.name}")
    parts = member_path.parts
    if len(parts) < 2 or parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"unsafe archive member: {member.name}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member: {member.name}")

    root = destination.resolve()
    target = (destination / Path(*parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unsafe archive member: {member.name}") from error
    return target


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive_metadata = os.lstat(archive)
    _validate_archive_metadata(archive, archive_metadata)
    with _open_validated_archive(archive, archive_metadata) as archive_source:
        with tarfile.open(fileobj=archive_source, mode="r:gz") as handle:
            members = handle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive member")
            targets = [
                (member, _validated_member_path(member, destination))
                for member in members
            ]
            target_paths = [target for _, target in targets]
            if len(target_paths) != len(set(target_paths)):
                raise ValueError("duplicate archive member destination")
            for member, target in targets:
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"cannot read archive member: {member.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(stat.S_IMODE(member.mode))
    return destination / ARCHIVE_ROOT


def _is_reparse_point(metadata: object) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _remove_old_project(project: Path) -> None:
    if not os.path.lexists(project):
        return
    metadata = os.lstat(project)
    if stat.S_ISLNK(metadata.st_mode):
        project.unlink()
    elif _is_reparse_point(metadata):
        os.rmdir(project)
    elif stat.S_ISDIR(metadata.st_mode):
        shutil.rmtree(project)
    else:
        project.unlink()


def _validate_real_directory(
    path: Path,
    metadata: object,
    *,
    context: str,
) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{context} is a symlink: {path}")
    if _is_reparse_point(metadata):
        raise ValueError(f"{context} is a reparse point: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{context} is not a directory: {path}")


def validate_project_source(
    project: Path,
    *,
    project_metadata: object | None = None,
    source_metadata: object | None = None,
) -> Path:
    project_info = (
        project_metadata
        if project_metadata is not None
        else os.lstat(project)
    )
    _validate_real_directory(
        project,
        project_info,
        context="active project",
    )
    source = project / "src"
    source_info = (
        source_metadata
        if source_metadata is not None
        else os.lstat(source)
    )
    _validate_real_directory(
        source,
        source_info,
        context="active project source",
    )
    resolved_project = project.resolve(strict=True)
    resolved_source = source.resolve(strict=True)
    try:
        resolved_source.relative_to(resolved_project)
    except ValueError as error:
        raise ValueError(
            f"active project source is outside project: {source}"
        ) from error
    return resolved_source


def _resolved_sys_path_entry(entry: object) -> Path | None:
    if not isinstance(entry, str):
        return None
    candidate = Path(entry if entry else os.getcwd())
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def migrate_same_session(
    project: Path,
    legacy_project: Path,
    package_name: str = IMPORT_PACKAGE,
) -> None:
    source = validate_project_source(project)
    legacy_source = (legacy_project / "src").resolve(strict=False)
    _remove_old_project(legacy_project)
    excluded_paths = {source, legacy_source}
    retained_paths = [
        entry
        for entry in sys.path
        if _resolved_sys_path_entry(entry) not in excluded_paths
    ]
    sys.path[:] = [str(source), *retained_paths]
    package_prefix = f"{package_name}."
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(package_prefix):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def replace_project_tree(
    archive: Path,
    destination: Path,
    project: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".jspace-extract-", dir=destination)
    )
    try:
        extracted_project = safe_extract(archive, temporary)
        if not extracted_project.is_dir():
            raise ValueError(
                f"project archive did not create {ARCHIVE_ROOT}"
            )
        _remove_old_project(project)
        extracted_project.replace(project)
        return project
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def dependency_commands(
    project: Path,
    requirements: Path,
) -> list[list[str]]:
    return [
        [sys.executable, "-m", "pip", "install", "uv==0.12.1"],
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--project",
            str(project),
            "--output-file",
            str(requirements),
        ],
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--requirement",
            str(requirements),
        ],
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "--editable",
            str(project),
        ],
    ]


def print_runtime_diagnostics(torch_module: Any) -> None:
    cuda_available = torch_module.cuda.is_available()
    print(f"Python: {sys.version}")
    print(f"Torch: {torch_module.__version__}")
    print(f"Torch CUDA build: {torch_module.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    if not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable; start a GPU-backed Colab session before bootstrap"
        )
    print(f"CUDA device count: {torch_module.cuda.device_count()}")
    print(f"CUDA device 0: {torch_module.cuda.get_device_name(0)}")


def main() -> int:
    nonce = consume_success_nonce()
    project = replace_project_tree(ARCHIVE, DESTINATION, PROJECT)
    for command in dependency_commands(project, LOCKED_REQUIREMENTS):
        run_checked(command)

    import torch

    print_runtime_diagnostics(torch)
    migrate_same_session(project, LEGACY_PROJECT)
    print(f"{SUCCESS_MARKER}:{nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
