from __future__ import annotations

import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


ARCHIVE = Path("/content/jspace_llama.tar.gz")
DESTINATION = Path("/content")
PROJECT = DESTINATION / "jspace_llama"
LOCKED_REQUIREMENTS = Path("/tmp/jspace-locked-requirements.txt")
SUCCESS_MARKER = "JSPACE_BOOTSTRAP_SUCCESS"


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    completed.check_returncode()


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
    if len(parts) < 2 or parts[0] != "jspace_llama":
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
    with tarfile.open(archive, "r:gz") as handle:
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
                raise ValueError(f"cannot read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(stat.S_IMODE(member.mode))
    return destination / "jspace_llama"


def _remove_old_project(project: Path) -> None:
    if not os.path.lexists(project):
        return
    metadata = os.lstat(project)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_reparse = bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    if stat.S_ISLNK(metadata.st_mode):
        project.unlink()
    elif is_reparse:
        os.rmdir(project)
    elif stat.S_ISDIR(metadata.st_mode):
        shutil.rmtree(project)
    else:
        project.unlink()


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
            raise ValueError("project archive did not create jspace_llama")
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
    project = replace_project_tree(ARCHIVE, DESTINATION, PROJECT)
    for command in dependency_commands(project, LOCKED_REQUIREMENTS):
        run_checked(command)

    import torch

    print_runtime_diagnostics(torch)
    print(SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
