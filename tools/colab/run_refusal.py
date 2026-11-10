from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "refusal_direction"
AUTH = ["colab", "--auth=adc"]
EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
}


def start_command(session: str, gpu: str) -> list[str]:
    return [*AUTH, "new", "-s", session, "--gpu", gpu]


def stop_command(session: str) -> list[str]:
    return [*AUTH, "stop", "-s", session]


def upload_command(session: str, local: Path, remote: str) -> list[str]:
    return [
        *AUTH,
        "upload",
        "-s",
        session,
        str(local),
        remote,
    ]


def exec_command(session: str, file: Path) -> list[str]:
    return [*AUTH, "exec", "-s", session, "-f", str(file)]


def download_command(session: str, remote: str, local: Path) -> list[str]:
    return [
        *AUTH,
        "download",
        "-s",
        session,
        remote,
        str(local),
    ]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return None
    if info.name.endswith("_output.ipynb"):
        return None
    return info


def build_archive() -> Path:
    staging = ROOT / ".colab-staging"
    staging.mkdir(exist_ok=True)
    archive = staging / "refusal_direction.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(
            PROJECT,
            arcname="refusal_direction",
            filter=_archive_filter,
        )
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
            "execute",
            "fetch",
            "stop",
        ),
    )
    parser.add_argument("--session", default="refusal-gemma")
    parser.add_argument(
        "--gpu",
        choices=("T4", "L4", "A100"),
        default="L4",
    )
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()

    if args.action == "probe":
        run([*AUTH, "sessions"])
    elif args.action == "start":
        run(start_command(args.session, args.gpu))
    elif args.action == "sync":
        archive = build_archive()
        run(
            upload_command(
                args.session,
                archive,
                "/content/refusal_direction.tar.gz",
            )
        )
    elif args.action == "bootstrap":
        run(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_bootstrap.py",
            )
        )
    elif args.action == "execute":
        if args.notebook is None:
            parser.error("--notebook is required for execute")
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            parser.error(f"notebook does not exist: {notebook}")
        run(exec_command(args.session, notebook))
    elif args.action == "fetch":
        run(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_pack_results.py",
            )
        )
        local = ROOT / ".colab-staging" / "refusal_results.tar.gz"
        local.parent.mkdir(exist_ok=True)
        run(
            download_command(
                args.session,
                "/content/refusal_results.tar.gz",
                local,
            )
        )
        with tarfile.open(local, "r:gz") as handle:
            handle.extractall(PROJECT)
    elif args.action == "stop":
        run(stop_command(args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
