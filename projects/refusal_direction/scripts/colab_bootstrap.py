from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tarfile


ARCHIVE = Path("/content/refusal_direction.tar.gz")
DESTINATION = Path("/content")
PROJECT = DESTINATION / "refusal_direction"


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


def safe_extract() -> None:
    with tarfile.open(ARCHIVE, "r:gz") as handle:
        root = DESTINATION.resolve()
        for member in handle.getmembers():
            target = (DESTINATION / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(DESTINATION)


def main() -> int:
    safe_extract()
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(PROJECT),
        ],
    )
    run_checked(
        [
            sys.executable,
            str(PROJECT / "scripts" / "fetch_upstream_splits.py"),
        ],
        cwd=PROJECT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
