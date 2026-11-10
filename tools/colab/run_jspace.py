from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "jspace_llama"
STAGING = ROOT / ".colab-staging"
AUTH = ["colab", "--auth=adc"]
REMOTE_ARCHIVE = "/content/jspace_llama.tar.gz"
REMOTE_RESULTS_ARCHIVE = "/content/jspace_results.tar.gz"
RESULT_ALLOWLIST = frozenset(
    {
        "results/execution_manifest.json",
        "results/final_evaluation.json",
    }
)
ARCHIVE_ROOT_FILES = frozenset({"README.md", "pyproject.toml", "uv.lock"})
ARCHIVE_ROOT_DIRECTORIES = frozenset({"notebooks", "scripts", "src", "tests"})
EXCLUDED_PARTS = frozenset(
    {
        ".cache",
        ".git",
        ".huggingface",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "cache",
        "lenses",
        "models",
        "results",
    }
)
EXCLUDED_SUFFIXES = frozenset({".bin", ".ckpt", ".pt", ".pth", ".safetensors"})
SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "hf_token",
        "token",
    }
)


def huggingface_token_path(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home()
    return base / ".cache" / "huggingface" / "token"


def start_command(session: str, gpu: str) -> list[str]:
    return [*AUTH, "new", "-s", session, "--gpu", gpu]


def stop_command(session: str) -> list[str]:
    return [*AUTH, "stop", "-s", session]


def upload_command(session: str, local: Path, remote: str) -> list[str]:
    return [*AUTH, "upload", "-s", session, str(local), remote]


def exec_command(session: str, file: Path, *, timeout: int) -> list[str]:
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
    return [*AUTH, "download", "-s", session, remote, str(local)]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _is_sensitive_filename(name: str) -> bool:
    lower_name = name.lower()
    stem = PurePosixPath(lower_name).stem
    return (
        lower_name in SENSITIVE_FILENAMES
        or lower_name.startswith(".env.")
        or "credential" in stem
        or stem == "token"
        or stem.startswith("token_")
        or stem.endswith("_token")
    )


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if info.issym() or info.islnk():
        return None

    parts = PurePosixPath(info.name.replace("\\", "/")).parts
    relative_parts = parts[1:]
    if not relative_parts:
        return info
    if any(part in EXCLUDED_PARTS for part in relative_parts):
        return None

    root_name = relative_parts[0]
    if len(relative_parts) == 1:
        if info.isdir() and root_name in ARCHIVE_ROOT_DIRECTORIES:
            return info
        if info.isfile() and root_name in ARCHIVE_ROOT_FILES:
            return info
        return None
    if root_name not in ARCHIVE_ROOT_DIRECTORIES:
        return None

    name = relative_parts[-1]
    if _is_sensitive_filename(name):
        return None
    if name.endswith("_output.ipynb"):
        return None
    if PurePosixPath(name).suffix.lower() in EXCLUDED_SUFFIXES:
        return None
    return info


def build_archive(
    *,
    project: Path = PROJECT,
    staging: Path = STAGING,
) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "jspace_llama.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(project, arcname="jspace_llama", filter=_archive_filter)
    return archive


def extract_results_archive(archive: Path, project: Path = PROJECT) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = {member.name for member in members}
        unexpected = names - RESULT_ALLOWLIST
        if unexpected:
            joined = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected result archive member: {joined}")
        missing = RESULT_ALLOWLIST - names
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing result archive member: {joined}")
        if any(
            not member.isfile() or member.issym() or member.islnk()
            for member in members
        ):
            raise ValueError("result archive must contain regular files only")
        handle.extractall(project)


def positive_timeout(value: str) -> int:
    timeout = int(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def build_parser() -> argparse.ArgumentParser:
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
            "stop",
        ),
    )
    parser.add_argument("--session", default="jspace-llama")
    parser.add_argument("--gpu", choices=("T4", "L4", "A100"), default="L4")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--timeout", type=positive_timeout)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "execute":
        if args.notebook is None:
            parser.error("--notebook is required for execute")
        if args.timeout is None:
            parser.error("--timeout is required for execute")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "probe":
        run([*AUTH, "sessions"])
    elif args.action == "start":
        run(start_command(args.session, args.gpu))
    elif args.action == "sync":
        archive = build_archive()
        run(upload_command(args.session, archive, REMOTE_ARCHIVE))
    elif args.action == "bootstrap":
        run(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_bootstrap.py",
                timeout=900,
            )
        )
    elif args.action == "auth-hf":
        token = huggingface_token_path()
        if not token.is_file():
            build_parser().error(
                "Hugging Face token not found; run `uv run hf auth login`"
            )
        run(upload_command(args.session, token, "/content/hf_token"))
        run(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_prepare_hf_auth.py",
                timeout=60,
            )
        )
    elif args.action == "execute":
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            build_parser().error(f"notebook does not exist: {notebook}")
        run(exec_command(args.session, notebook, timeout=args.timeout))
    elif args.action == "fetch":
        run(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_pack_results.py",
                timeout=300,
            )
        )
        STAGING.mkdir(parents=True, exist_ok=True)
        local = STAGING / "jspace_results.tar.gz"
        run(download_command(args.session, REMOTE_RESULTS_ARCHIVE, local))
        extract_results_archive(local)
    elif args.action == "stop":
        run(stop_command(args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
