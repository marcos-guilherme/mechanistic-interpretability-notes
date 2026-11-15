from __future__ import annotations

import argparse
import io
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "refusal_direction"
AUTH = ["colab", "--auth=adc"]
PROFILE_SESSIONS = {
    "gemma": "refusal-gemma",
    "qwen25-coder": "refusal-qwen",
}
EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
}


def huggingface_token_path(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home()
    return base / ".cache" / "huggingface" / "token"


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
    if profile == "qwen25-coder":
        return PROJECT / "scripts" / "colab_pack_qwen_results.py"
    return PROJECT / "scripts" / "colab_pack_results.py"


def status_command(session: str) -> list[str]:
    return [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(PROJECT / "scripts" / "qwen_run_status.py"),
        "--timeout",
        "120",
    ]


def pilot_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_pilot.py",
        timeout=timeout,
    )


def extract_qwen_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_archive,
    )

    extract_qwen_archive(archive, project)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


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
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    session = resolve_session(args.profile, args.session)

    if args.action == "probe":
        run([*AUTH, "sessions"])
    elif args.action == "start":
        run(start_command(session, args.gpu))
    elif args.action == "sync":
        archive = build_archive(stamp_commit=args.profile == "qwen25-coder")
        run(
            upload_command(
                session,
                archive,
                "/content/refusal_direction.tar.gz",
            )
        )
    elif args.action == "bootstrap":
        run(
            exec_command(
                session,
                PROJECT / "scripts" / "colab_bootstrap.py",
                timeout=600,
            )
        )
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
        if args.profile == "qwen25-coder":
            remote_archive = "/content/qwen_results.tar.gz"
            local_name = "qwen_results.tar.gz"
        else:
            remote_archive = "/content/refusal_results.tar.gz"
            local_name = "refusal_results.tar.gz"
        run(
            exec_command(
                session,
                fetch_pack_script(args.profile),
                timeout=300,
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
        if args.profile == "qwen25-coder":
            extract_qwen_results_archive(local, PROJECT)
        else:
            with tarfile.open(local, "r:gz") as handle:
                handle.extractall(PROJECT)
    elif args.action == "status":
        if args.profile != "qwen25-coder":
            parser.error("status is only available for --profile qwen25-coder")
        run(status_command(session))
    elif args.action == "pilot":
        if args.profile != "qwen25-coder":
            parser.error("pilot is only available for --profile qwen25-coder")
        run(pilot_command(session, timeout=args.timeout))
    elif args.action == "stop":
        run(stop_command(session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())