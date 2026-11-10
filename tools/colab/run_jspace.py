from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
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
SUCCESS_NONCE_ENV = "JSPACE_SUCCESS_NONCE"
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
    "schemas": frozenset({".json"}),
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


def exec_command(
    session: str,
    file: Path,
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> list[str]:
    command = [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(file),
        "--timeout",
        str(timeout),
    ]
    for name, value in sorted((environment or {}).items()):
        command.extend(["--env", f"{name}={value}"])
    return command


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


def run_remote_operation(
    session: str,
    file: Path,
    *,
    timeout: int,
    operation_marker: str,
) -> None:
    nonce = secrets.token_hex(32)
    expected_marker = f"{operation_marker}:{nonce}"
    command = exec_command(
        session,
        file,
        timeout=timeout,
        environment={SUCCESS_NONCE_ENV: nonce},
    )
    run_remote_script(command, expected_marker)


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


def _validate_link_count(path: Path, metadata: object, *, context: str) -> None:
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"{context} is hard-linked: {path}")


def _same_file_identity(expected: object, actual: object) -> bool:
    return (
        expected.st_dev == actual.st_dev
        and expected.st_ino == actual.st_ino
    )


def open_validated_file(path: Path, metadata: object):
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        actual = os.fstat(descriptor)
        if (
            not _same_file_identity(metadata, actual)
            or stat.S_IFMT(metadata.st_mode) != stat.S_IFMT(actual.st_mode)
            or getattr(metadata, "st_nlink", 1)
            != getattr(actual, "st_nlink", 1)
            or metadata.st_size != actual.st_size
        ):
            raise ValueError(f"file changed after validation: {path}")
        if not stat.S_ISREG(actual.st_mode):
            raise ValueError(f"opened path is not a regular file: {path}")
        _validate_link_count(path, actual, context="opened file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


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
    _validate_link_count(path, file_metadata, context="archive source")

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


def validate_staging_directory(
    staging: Path,
    root: Path,
    *,
    metadata: object | None = None,
) -> object:
    directory_metadata = metadata if metadata is not None else os.lstat(staging)
    mode = directory_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"staging directory is a symlink: {staging}")
    if _is_reparse_point(directory_metadata):
        raise ValueError(f"staging directory is a reparse point: {staging}")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"staging path is not a directory: {staging}")
    resolved_root = root.resolve(strict=True)
    resolved_staging = staging.resolve(strict=True)
    try:
        resolved_staging.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"staging directory is outside repository: {staging}"
        ) from error
    return directory_metadata


def _ensure_staging_directory(staging: Path, root: Path) -> object:
    if not os.path.lexists(staging):
        staging.mkdir(mode=0o700, parents=False)
    return validate_staging_directory(staging, root)


def _same_directory_identity(expected: object, actual: object) -> bool:
    return (
        expected.st_dev == actual.st_dev
        and expected.st_ino == actual.st_ino
        and stat.S_IFMT(expected.st_mode) == stat.S_IFMT(actual.st_mode)
    )


def _validate_private_download_directory(
    directory: Path,
    staging: Path,
    expected_metadata: object,
) -> None:
    actual = validate_staging_directory(directory, staging)
    if not _same_directory_identity(expected_metadata, actual):
        raise ValueError(f"private download directory changed: {directory}")


def _cleanup_private_download_directory(
    directory: Path,
    staging: Path,
    root: Path,
    staging_metadata: object,
    private_metadata: object,
) -> None:
    if not os.path.lexists(directory):
        return
    current_staging = validate_staging_directory(staging, root)
    if not _same_directory_identity(staging_metadata, current_staging):
        raise RuntimeError("staging directory changed during fetch")

    current = os.lstat(directory)
    if stat.S_ISLNK(current.st_mode):
        directory.unlink()
        return
    if _is_reparse_point(current):
        os.rmdir(directory)
        return
    if not stat.S_ISDIR(current.st_mode):
        directory.unlink()
        return
    if not _same_directory_identity(private_metadata, current):
        raise RuntimeError("private download directory changed during fetch")
    shutil.rmtree(directory)


def fetch_results(
    session: str,
    *,
    staging: Path = STAGING,
    project: Path = PROJECT,
    root: Path = ROOT,
) -> None:
    staging_metadata = _ensure_staging_directory(staging, root)
    private_directory = Path(
        tempfile.mkdtemp(prefix="jspace-fetch-", dir=staging)
    )
    current_staging = validate_staging_directory(staging, root)
    if not _same_directory_identity(staging_metadata, current_staging):
        raise RuntimeError("staging directory changed during fetch")
    private_metadata = os.lstat(private_directory)
    _validate_private_download_directory(
        private_directory,
        staging,
        private_metadata,
    )
    private_directory.chmod(0o700)
    local = private_directory / "jspace_results.tar.gz"
    if os.path.lexists(local):
        raise RuntimeError(f"download target unexpectedly exists: {local}")
    try:
        run(download_command(session, REMOTE_RESULTS_ARCHIVE, local))
        _validate_private_download_directory(
            private_directory,
            staging,
            private_metadata,
        )
        extract_results_archive(local, project)
    finally:
        _cleanup_private_download_directory(
            private_directory,
            staging,
            root,
            staging_metadata,
            private_metadata,
        )


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
    with open_validated_file(path, metadata) as source:
        handle.addfile(info, source)


def capture_source_provenance(repository_root: Path = ROOT) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("git returned an invalid source commit")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "schema_version": 1,
        "commit": revision,
        "dirty": bool(status.strip()),
    }


def _add_source_provenance(
    handle: tarfile.TarFile,
    provenance: dict[str, object],
) -> None:
    payload = (
        json.dumps(
            provenance,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    info = tarfile.TarInfo("jspace_llama/source_provenance.json")
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    handle.addfile(info, io.BytesIO(payload))


def build_archive(
    *,
    project: Path = PROJECT,
    staging: Path = STAGING,
    repository_root: Path = ROOT,
) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "jspace_llama.tar.gz"
    if os.path.lexists(archive):
        _validate_regular_metadata(
            archive,
            os.lstat(archive),
            context="project archive",
        )
        archive.unlink()
    with tarfile.open(archive, "x:gz") as handle:
        for path in _archive_candidates(project):
            _add_regular_file(handle, path, project)
        _add_source_provenance(
            handle,
            capture_source_provenance(repository_root),
        )
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
    _validate_link_count(path, metadata, context=context)


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
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"result artifact is not valid JSON: {name}") from error


def extract_results_archive(archive: Path, project: Path = PROJECT) -> None:
    archive_metadata = os.lstat(archive)
    validate_result_archive(archive, metadata=archive_metadata)
    with open_validated_file(archive, archive_metadata) as archive_source:
        with tarfile.open(fileobj=archive_source, mode="r:gz") as handle:
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
                raise FileNotFoundError(
                    f"missing result archive member: {joined}"
                )

            payloads: dict[str, bytes] = {}
            for member in members:
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(
                        "result archive must contain regular files only"
                    )
                if member.size < 0 or member.size > MAX_RESULT_BYTES:
                    raise ValueError(
                        f"result artifact is too large: {member.name}"
                    )
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"cannot read result artifact: {member.name}"
                    )
                payload = source.read(MAX_RESULT_BYTES + 1)
                if len(payload) > MAX_RESULT_BYTES:
                    raise ValueError(
                        f"result artifact is too large: {member.name}"
                    )
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


def _validate_notebook_structure(document: object, notebook: Path) -> None:
    if not isinstance(document, dict):
        raise ValueError(f"notebook root must be an object: {notebook}")
    if document.get("nbformat") != 4:
        raise ValueError(f"notebook must use nbformat 4: {notebook}")
    if not isinstance(document.get("nbformat_minor"), int):
        raise ValueError(f"notebook nbformat_minor must be an integer: {notebook}")
    if not isinstance(document.get("metadata"), dict):
        raise ValueError(f"notebook metadata must be an object: {notebook}")
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"notebook cells must be an array: {notebook}")
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"notebook cell {index} must be an object: {notebook}")
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            raise ValueError(
                f"notebook cell {index} has invalid cell_type: {notebook}"
            )
        if not isinstance(cell.get("metadata"), dict):
            raise ValueError(
                f"notebook cell {index} metadata must be an object: {notebook}"
            )
        source = cell.get("source")
        if not (
            isinstance(source, str)
            or (
                isinstance(source, list)
                and all(isinstance(line, str) for line in source)
            )
        ):
            raise ValueError(
                f"notebook cell {index} source must be text: {notebook}"
            )
        if cell["cell_type"] == "code":
            if not isinstance(cell.get("outputs"), list):
                raise ValueError(
                    f"notebook cell {index} outputs must be an array: {notebook}"
                )
            execution_count = cell.get("execution_count")
            if execution_count is not None and not isinstance(
                execution_count,
                int,
            ):
                raise ValueError(
                    f"notebook cell {index} execution_count is invalid: "
                    f"{notebook}"
                )


def validate_notebook(notebook: Path) -> None:
    if notebook.suffix.lower() != ".ipynb":
        raise ValueError(f"notebook must use the .ipynb extension: {notebook}")
    if not os.path.lexists(notebook):
        raise FileNotFoundError(f"notebook does not exist: {notebook}")
    metadata = os.lstat(notebook)
    _validate_regular_metadata(notebook, metadata, context="notebook")
    try:
        with open_validated_file(notebook, metadata) as source:
            payload = source.read().decode("utf-8")
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"notebook is malformed: {notebook}") from error
    _validate_notebook_structure(document, notebook)


def execute_notebook(session: str, notebook: Path, *, timeout: int) -> Path:
    validate_notebook(notebook)
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
        run_remote_operation(
            args.session,
            PROJECT / "scripts" / "colab_bootstrap.py",
            timeout=900,
            operation_marker=BOOTSTRAP_SUCCESS_MARKER,
        )
    elif args.action == "auth-hf":
        token = huggingface_token_path()
        if not token.is_file():
            build_parser().error(
                "Hugging Face token not found; run `uv run hf auth login`"
            )
        run(upload_command(args.session, token, "/content/hf_token"))
        run_remote_operation(
            args.session,
            PROJECT / "scripts" / "colab_prepare_hf_auth.py",
            timeout=60,
            operation_marker=AUTH_SUCCESS_MARKER,
        )
    elif args.action == "execute":
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            build_parser().error(f"notebook does not exist: {notebook}")
        execute_notebook(args.session, notebook, timeout=args.timeout)
    elif args.action == "fetch":
        run_remote_operation(
            args.session,
            PROJECT / "scripts" / "colab_pack_results.py",
            timeout=300,
            operation_marker=PACK_SUCCESS_MARKER,
        )
        fetch_results(args.session)
    elif args.action == "stop":
        run(stop_command(args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
