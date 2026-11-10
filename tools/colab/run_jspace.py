from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "jspace_llama"
STAGING = ROOT / ".colab-staging"
AUTH = ["colab", "--auth=adc"]
REMOTE_ARCHIVE = "/content/jspace_llama.tar.gz"
REMOTE_RESULTS_ARCHIVE = "/content/jspace_results.tar.gz"
BOOTSTRAP_SUCCESS_MARKER = "JSPACE_BOOTSTRAP_SUCCESS"
AUTH_SUCCESS_MARKER = "JSPACE_HF_AUTH_SUCCESS"
PACK_SUCCESS_MARKER = "JSPACE_RESULTS_PACK_SUCCESS"
MAX_RESULT_BYTES = 1_048_576
RESULT_ALLOWLIST = frozenset(
    {
        "results/execution_manifest.json",
        "results/final_evaluation.json",
    }
)
ARCHIVE_ROOT_FILES = frozenset({"README.md", "pyproject.toml", "uv.lock"})
ARCHIVE_SUFFIXES = {
    "notebooks": frozenset({".ipynb"}),
    "scripts": frozenset({".py"}),
    "src": frozenset({".py"}),
    "tests": frozenset({".py"}),
}
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


def run_captured(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()
    return completed


def run_remote_script(command: list[str], success_marker: str) -> None:
    completed = run_captured(command)
    output = f"{completed.stdout}\n{completed.stderr}"
    output_lines = {line.strip() for line in output.splitlines()}
    if success_marker not in output_lines:
        raise RuntimeError(
            "remote script did not emit expected success marker: "
            f"{success_marker}"
        )


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


def _is_reparse_point(metadata: object) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def validate_archive_file(
    path: Path,
    project: Path,
    *,
    metadata: object | None = None,
) -> object:
    file_metadata = metadata if metadata is not None else os.lstat(path)
    mode = file_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"archive source is a symlink: {path}")
    if _is_reparse_point(file_metadata):
        raise ValueError(f"archive source is a reparse point: {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"archive source is not a regular file: {path}")

    resolved_project = project.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_project)
    except ValueError as error:
        raise ValueError(f"archive source resolves outside project: {path}") from error
    return file_metadata


def validate_archive_directory(
    path: Path,
    *,
    metadata: object | None = None,
) -> object:
    directory_metadata = metadata if metadata is not None else os.lstat(path)
    mode = directory_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"archive source is a symlink: {path}")
    if _is_reparse_point(directory_metadata):
        raise ValueError(f"archive source is a reparse point: {path}")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"archive source is not a directory: {path}")
    return directory_metadata


def _archive_candidates(project: Path) -> list[Path]:
    validate_archive_directory(project)
    candidates: list[Path] = []
    for root_name, suffixes in ARCHIVE_SUFFIXES.items():
        root = project / root_name
        if not os.path.lexists(root):
            continue
        validate_archive_directory(root)

        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            retained_directories: list[str] = []
            for directory_name in sorted(directories):
                directory = current_path / directory_name
                if directory_name in EXCLUDED_PARTS:
                    continue
                validate_archive_directory(directory)
                retained_directories.append(directory_name)
            directories[:] = retained_directories

            for filename in sorted(filenames):
                path = current_path / filename
                if filename == ".gitkeep":
                    continue
                if _is_sensitive_filename(filename):
                    continue
                if filename.endswith("_output.ipynb"):
                    continue
                suffix = path.suffix.lower()
                if suffix in EXCLUDED_SUFFIXES:
                    continue
                if suffix not in suffixes:
                    raise ValueError(f"archive source has unapproved suffix: {path}")
                validate_archive_file(path, project)
                candidates.append(path)

    for filename in sorted(ARCHIVE_ROOT_FILES):
        path = project / filename
        if not path.exists():
            raise FileNotFoundError(f"required archive input is missing: {path}")
        validate_archive_file(path, project)
        candidates.append(path)
    return sorted(candidates, key=lambda path: path.relative_to(project).as_posix())


def _add_regular_file(
    handle: tarfile.TarFile,
    path: Path,
    project: Path,
) -> None:
    metadata = validate_archive_file(path, project)
    relative = path.relative_to(project).as_posix()
    info = tarfile.TarInfo(f"jspace_llama/{relative}")
    info.size = metadata.st_size
    info.mode = stat.S_IMODE(metadata.st_mode)
    info.mtime = int(getattr(metadata, "st_mtime", 0))
    with path.open("rb") as source:
        handle.addfile(info, source)


def build_archive(
    *,
    project: Path = PROJECT,
    staging: Path = STAGING,
) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "jspace_llama.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in _archive_candidates(project):
            _add_regular_file(handle, path, project)
    return archive


def _validate_regular_metadata(
    path: Path,
    metadata: object,
    *,
    context: str,
) -> None:
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"{context} is a symlink: {path}")
    if _is_reparse_point(metadata):
        raise ValueError(f"{context} is a reparse point: {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"{context} is not a regular file: {path}")


def validate_result_destination(
    path: Path,
    root: Path,
    *,
    metadata: object | None = None,
) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"result destination is outside project: {path}") from error
    if metadata is not None:
        _validate_regular_metadata(
            path,
            metadata,
            context="result destination",
        )


def validate_result_archive(
    archive: Path,
    *,
    metadata: object | None = None,
) -> None:
    archive_metadata = metadata if metadata is not None else os.lstat(archive)
    _validate_regular_metadata(
        archive,
        archive_metadata,
        context="result archive",
    )


def _validate_json_bytes(name: str, payload: bytes) -> None:
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"result artifact is not valid JSON: {name}") from error


def extract_results_archive(archive: Path, project: Path = PROJECT) -> None:
    validate_result_archive(archive)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("duplicate result archive member")
        names = set(member_names)
        unexpected = names - RESULT_ALLOWLIST
        if unexpected:
            joined = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected result archive member: {joined}")
        missing = RESULT_ALLOWLIST - names
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing result archive member: {joined}")

        payloads: dict[str, bytes] = {}
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("result archive must contain regular files only")
            if member.size < 0 or member.size > MAX_RESULT_BYTES:
                raise ValueError(f"result artifact is too large: {member.name}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read result artifact: {member.name}")
            payload = source.read(MAX_RESULT_BYTES + 1)
            if len(payload) > MAX_RESULT_BYTES:
                raise ValueError(f"result artifact is too large: {member.name}")
            _validate_json_bytes(member.name, payload)
            payloads[member.name] = payload

    results = project / "results"
    for directory in (project, results):
        if os.path.lexists(directory):
            metadata = os.lstat(directory)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"result destination is a symlink: {directory}")
            if _is_reparse_point(metadata):
                raise ValueError(
                    f"result destination is a reparse point: {directory}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"result destination is not a directory: {directory}"
                )
    results.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        destination = project / PurePosixPath(name)
        validate_result_destination(destination, project)
        if os.path.lexists(destination):
            validate_result_destination(
                destination,
                project,
                metadata=os.lstat(destination),
            )
        destination.write_bytes(payload)


def notebook_output_path(notebook: Path) -> Path:
    return notebook.with_name(f"{notebook.stem}_output{notebook.suffix}")


def execute_notebook(session: str, notebook: Path, *, timeout: int) -> Path:
    output_notebook = notebook_output_path(notebook)
    output_notebook.unlink(missing_ok=True)
    run(exec_command(session, notebook, timeout=timeout))
    if not output_notebook.is_file():
        raise FileNotFoundError(
            f"Colab did not create fresh output notebook: {output_notebook}"
        )

    try:
        payload = json.loads(output_notebook.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Colab output notebook is invalid JSON: {output_notebook}"
        ) from error
    for cell_index, cell in enumerate(payload.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                name = output.get("ename", "RemoteError")
                value = output.get("evalue", "")
                raise RuntimeError(
                    f"notebook cell {cell_index} failed: {name}: {value}"
                )
    return output_notebook


def positive_timeout(value: str) -> int:
    timeout = int(value)
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("probe")

    start = actions.add_parser("start")
    start.add_argument("--session", default="jspace-llama")
    start.add_argument("--gpu", choices=("T4", "L4", "A100"), default="L4")

    for action in ("sync", "bootstrap", "auth-hf", "fetch", "stop"):
        action_parser = actions.add_parser(action)
        action_parser.add_argument("--session", default="jspace-llama")

    execute = actions.add_parser("execute")
    execute.add_argument("--session", default="jspace-llama")
    execute.add_argument("--notebook", type=Path, required=True)
    execute.add_argument("--timeout", type=positive_timeout, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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
        run_remote_script(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_bootstrap.py",
                timeout=900,
            ),
            BOOTSTRAP_SUCCESS_MARKER,
        )
    elif args.action == "auth-hf":
        token = huggingface_token_path()
        if not token.is_file():
            build_parser().error(
                "Hugging Face token not found; run `uv run hf auth login`"
            )
        run(upload_command(args.session, token, "/content/hf_token"))
        run_remote_script(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_prepare_hf_auth.py",
                timeout=60,
            ),
            AUTH_SUCCESS_MARKER,
        )
    elif args.action == "execute":
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            build_parser().error(f"notebook does not exist: {notebook}")
        execute_notebook(args.session, notebook, timeout=args.timeout)
    elif args.action == "fetch":
        run_remote_script(
            exec_command(
                args.session,
                PROJECT / "scripts" / "colab_pack_results.py",
                timeout=300,
            ),
            PACK_SUCCESS_MARKER,
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
