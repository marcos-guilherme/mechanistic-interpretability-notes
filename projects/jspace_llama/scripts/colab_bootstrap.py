from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any


ARCHIVE = Path("/content/jspace_llama.tar.gz")
DESTINATION = Path("/content")
PROJECT = DESTINATION / "jspace_llama"
LOCKED_REQUIREMENTS = Path("/tmp/jspace-locked-requirements.txt")


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


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def dependency_commands(
    project: Path,
    requirements: Path,
) -> list[list[str]]:
    return [
        [sys.executable, "-m", "pip", "install", "uv"],
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
    safe_extract(ARCHIVE, DESTINATION)
    for command in dependency_commands(PROJECT, LOCKED_REQUIREMENTS):
        run_checked(command)

    import torch

    print_runtime_diagnostics(torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
