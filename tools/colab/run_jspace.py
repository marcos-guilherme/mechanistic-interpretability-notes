from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
from typing import Sequence
import xml.etree.ElementTree as ET
import zlib


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
RESULT_SET_ENV = "JSPACE_RESULT_SET"
MAX_RESULT_BYTES = 1_048_576
MAX_OBSERVATION_RAW_BYTES = 5 * 1_048_576
MAX_OBSERVATION_JSON_BYTES = 5 * 1_048_576
MAX_FIGURE_BYTES = 5 * 1_048_576
MAX_ARCHIVE_BYTES = 40 * 1_048_576
MAX_IMAGE_DIMENSION = 10_000
MAX_TAR_HEADER_OVERHEAD_BYTES = 1_048_576
TAR_BLOCK_BYTES = 512
OBSERVATION_RAW = "artifacts/jspace_observation_raw.json"
RESULT_SETS = {
    "causal": frozenset(
        {
            "results/execution_manifest.json",
            "results/final_evaluation.json",
        }
    ),
    "observation": frozenset(
        {
            OBSERVATION_RAW,
            "results/jspace_observation_manifest.json",
            "results/jspace_observation.json",
            "results/jspace_observation_pass_at_k.png",
            "results/jspace_observation_pass_at_k.svg",
            "results/jspace_observation_rank_heatmap.png",
            "results/jspace_observation_rank_heatmap.svg",
            "results/jspace_observation_panel.png",
            "results/jspace_observation_panel.svg",
        }
    ),
}
RESULT_ALLOWLIST = RESULT_SETS["causal"]
ARCHIVE_ROOT_FILES = frozenset({"README.md", "pyproject.toml", "uv.lock"})
ARCHIVE_SUFFIXES = {
    "data": frozenset({"", ".json", ".md"}),
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


def result_allowlist(result_set: str) -> set[str]:
    try:
        return set(RESULT_SETS[result_set])
    except KeyError as error:
        raise ValueError(f"unknown result set: {result_set}") from error


def result_size_limit(name: str) -> int:
    if name == OBSERVATION_RAW:
        return MAX_OBSERVATION_RAW_BYTES
    if name == "results/jspace_observation.json":
        return MAX_OBSERVATION_JSON_BYTES
    if name.endswith((".png", ".svg")):
        return MAX_FIGURE_BYTES
    return MAX_RESULT_BYTES


def decompressed_tar_limit(result_set: str) -> int:
    allowlist = result_allowlist(result_set)
    padded_payload_bytes = sum(
        (
            (result_size_limit(name) + TAR_BLOCK_BYTES - 1)
            // TAR_BLOCK_BYTES
        )
        * TAR_BLOCK_BYTES
        for name in allowlist
    )
    standard_header_bytes = len(allowlist) * TAR_BLOCK_BYTES
    return (
        padded_payload_bytes
        + standard_header_bytes
        + MAX_TAR_HEADER_OVERHEAD_BYTES
    )


class _BoundedDecompressedReader:
    def __init__(self, source, limit: int) -> None:
        self._source = source
        self._limit = limit
        self._count = 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        remaining = self._limit - self._count
        requested = remaining + 1 if size < 0 else min(size, remaining + 1)
        payload = self._source.read(requested)
        if len(payload) > remaining:
            raise ValueError(
                "decompressed TAR exceeds selected result profile limit"
            )
        self._count += len(payload)
        return payload


@contextmanager
def open_bounded_result_tar(
    archive_source,
    result_set: str,
):
    with gzip.GzipFile(fileobj=archive_source, mode="rb") as decompressed:
        bounded = _BoundedDecompressedReader(
            decompressed,
            decompressed_tar_limit(result_set),
        )
        with tarfile.open(fileobj=bounded, mode="r|") as handle:
            yield handle


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
        encoding="utf-8",
        errors="replace",
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
    environment: dict[str, str] | None = None,
) -> None:
    nonce = secrets.token_hex(32)
    expected_marker = f"{operation_marker}:{nonce}"
    remote_environment = dict(environment or {})
    remote_environment[SUCCESS_NONCE_ENV] = nonce
    command = exec_command(
        session,
        file,
        timeout=timeout,
        environment=remote_environment,
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
    result_set: str = "causal",
    staging: Path = STAGING,
    project: Path = PROJECT,
    root: Path = ROOT,
) -> None:
    result_allowlist(result_set)
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
        if result_set == "causal":
            extract_results_archive(local, project)
        else:
            extract_results_archive(local, project, result_set)
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
                relative = path.relative_to(project).as_posix()
                if (
                    suffix == ""
                    and relative != "data/anthropic-jlens/LICENSE"
                ):
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


def _runtime_critical_files(
    project: Path,
    repository_root: Path,
) -> list[tuple[str, Path]]:
    candidates = [
        path
        for path in _archive_candidates(project)
        if (
            path.relative_to(project).parts[0]
            in {"data", "src", "notebooks", "schemas", "scripts"}
            or path.name in {"pyproject.toml", "uv.lock"}
        )
        and "tests" not in path.relative_to(project).parts
    ]
    runner = repository_root / "tools" / "colab" / "run_jspace.py"
    if runner.is_file():
        _validate_regular_metadata(
            runner,
            os.lstat(runner),
            context="local runner",
        )
        candidates.append(runner)

    labeled: list[tuple[str, Path]] = []
    for path in candidates:
        if path.resolve() == runner.resolve():
            label = "tools/colab/run_jspace.py"
        else:
            relative = path.relative_to(project).as_posix()
            label = f"projects/jspace_llama/{relative}"
        labeled.append((label, path))
    return sorted(labeled, key=lambda item: item[0])


def runtime_source_sha256(
    project: Path,
    repository_root: Path,
) -> str:
    digest = hashlib.sha256()
    files = _runtime_critical_files(project, repository_root)
    if not files:
        raise ValueError("no runtime-critical source files found")
    for label, path in files:
        metadata = os.lstat(path)
        _validate_regular_metadata(path, metadata, context="runtime source")
        with open_validated_file(path, metadata) as source:
            content = source.read()
        label_bytes = label.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _runtime_git_pathspecs(
    project: Path,
    repository_root: Path,
) -> list[str]:
    try:
        project_relative = project.resolve().relative_to(
            repository_root.resolve()
        ).as_posix()
    except ValueError:
        project_relative = "projects/jspace_llama"
    return [
        f"{project_relative}/data",
        f"{project_relative}/src",
        f"{project_relative}/notebooks",
        f"{project_relative}/schemas",
        f"{project_relative}/scripts",
        f"{project_relative}/pyproject.toml",
        f"{project_relative}/uv.lock",
        "tools/colab/run_jspace.py",
    ]


def capture_source_provenance(
    repository_root: Path = ROOT,
    project: Path = PROJECT,
) -> dict[str, object]:
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
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            *_runtime_git_pathspecs(project, repository_root),
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    scoped_dirty = bool(status.strip())
    return {
        "schema_version": 1,
        "commit": revision,
        "dirty": scoped_dirty,
        "scoped_dirty": scoped_dirty,
        "archive_sha256": runtime_source_sha256(
            project,
            repository_root,
        ),
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
            capture_source_provenance(repository_root, project),
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
    if getattr(archive_metadata, "st_size", 0) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"result archive is too large: {archive}")


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

    def reject_nonfinite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON value: {value}")
        return parsed

    try:
        json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=reject_nonfinite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"result artifact is not valid JSON: {name}") from error


def _validate_png_bytes(name: str, payload: bytes) -> None:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"result artifact is not a valid PNG: {name}")
    if len(payload) < 33:
        raise ValueError(f"result PNG has an invalid IHDR: {name}")
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13 or payload[12:16] != b"IHDR":
        raise ValueError(f"result PNG has an invalid IHDR: {name}")
    ihdr = payload[16:29]
    expected_crc = struct.unpack(">I", payload[29:33])[0]
    actual_crc = zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise ValueError(f"result PNG has an invalid IHDR checksum: {name}")
    width, height = struct.unpack(">II", ihdr[:8])
    if not (
        0 < width <= MAX_IMAGE_DIMENSION
        and 0 < height <= MAX_IMAGE_DIMENSION
    ):
        raise ValueError(f"result PNG has invalid dimensions: {name}")
    bit_depth, color_type, compression, filter_method, interlace = ihdr[8:]
    valid_bit_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if (
        bit_depth not in valid_bit_depths.get(color_type, set())
        or compression != 0
        or filter_method != 0
        or interlace not in {0, 1}
    ):
        raise ValueError(f"result PNG has invalid IHDR fields: {name}")


def _local_xml_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].lower()


SAFE_SVG_ELEMENTS = frozenset(
    {
        "agent",
        "clippath",
        "creator",
        "date",
        "defs",
        "format",
        "g",
        "image",
        "metadata",
        "path",
        "rdf",
        "rect",
        "style",
        "svg",
        "title",
        "type",
        "use",
        "work",
    }
)
SAFE_SVG_ATTRIBUTES = frozenset(
    {
        "clip-path",
        "d",
        "height",
        "href",
        "id",
        "resource",
        "style",
        "transform",
        "type",
        "version",
        "viewbox",
        "width",
        "x",
        "y",
    }
)
SAFE_STYLE_PROPERTIES = frozenset(
    {
        "fill",
        "opacity",
        "stroke",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "stroke-width",
    }
)
SAFE_METADATA_RESOURCE = "http://purl.org/dc/dcmitype/StillImage"
CSS_ESCAPE = re.compile(
    r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|([^0-9a-fA-F\r\n\f])|(\r\n|[\r\n\f]))"
)
SVG_FRAGMENT = re.compile(r"#[A-Za-z_][A-Za-z0-9_.:-]*\Z")
SVG_CLIP_PATH = re.compile(
    r"url\(\s*#[A-Za-z_][A-Za-z0-9_.:-]*\s*\)\Z",
    flags=re.IGNORECASE,
)
SAFE_STYLE_ELEMENT = "*{stroke-linejoin:round;stroke-linecap:butt}"


def _decode_css_escapes(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hexadecimal, escaped, newline = match.groups()
        if hexadecimal is not None:
            codepoint = int(hexadecimal, 16)
            if (
                codepoint == 0
                or codepoint > 0x10FFFF
                or 0xD800 <= codepoint <= 0xDFFF
            ):
                return "\N{REPLACEMENT CHARACTER}"
            return chr(codepoint)
        if escaped is not None:
            return escaped
        if newline is not None:
            return ""
        raise AssertionError("unreachable CSS escape")

    return CSS_ESCAPE.sub(replace, value)


def _validate_style_attribute(name: str, value: str) -> None:
    normalized = _decode_css_escapes(value)
    lowered = normalized.lower()
    if "url" in lowered or "@import" in lowered:
        raise ValueError(f"result SVG contains external content: {name}")
    if "\\" in normalized or "@" in normalized:
        raise ValueError(f"result SVG contains unsafe CSS: {name}")
    for declaration in normalized.split(";"):
        if not declaration.strip():
            continue
        if ":" not in declaration:
            raise ValueError(f"result SVG contains unsafe CSS: {name}")
        property_name, property_value = declaration.split(":", 1)
        if property_name.strip().lower() not in SAFE_STYLE_PROPERTIES:
            raise ValueError(f"result SVG contains unsafe CSS: {name}")
        if re.fullmatch(
            r"[-#A-Za-z0-9.,%+ ]+",
            property_value.strip(),
        ) is None:
            raise ValueError(f"result SVG contains unsafe CSS: {name}")


def _validate_embedded_png(name: str, value: str) -> None:
    prefix = "data:image/png;base64,"
    if not value.lower().startswith(prefix):
        raise ValueError(f"result SVG contains external content: {name}")
    encoded = "".join(value[len(prefix) :].split())
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"result SVG contains invalid embedded image: {name}") from error
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"result SVG embedded image is not PNG: {name}")


def _validate_svg_bytes(name: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"result artifact is not valid UTF-8 SVG: {name}") from error
    upper = text.upper()
    if "<!DOCTYPE" in upper:
        raise ValueError(f"result SVG contains DOCTYPE: {name}")
    if "<!ENTITY" in upper:
        raise ValueError(f"result SVG contains ENTITY: {name}")
    if "<?XML-STYLESHEET" in upper:
        raise ValueError(f"result SVG contains active content: {name}")
    without_declaration = re.sub(
        r"^\s*<\?xml[^?]*\?>",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if "<?" in without_declaration:
        raise ValueError(f"result SVG contains active content: {name}")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError(f"result artifact is not valid XML SVG: {name}") from error
    if _local_xml_name(root.tag) != "svg":
        raise ValueError(f"result SVG root is not svg: {name}")
    for element in root.iter():
        element_name = _local_xml_name(element.tag)
        if element_name not in SAFE_SVG_ELEMENTS:
            raise ValueError(
                f"result SVG contains active unsafe element "
                f"{element_name}: {name}"
            )
        if element_name == "style":
            decoded_style = _decode_css_escapes(element.text or "")
            lowered_style = decoded_style.lower()
            if "url" in lowered_style or "@import" in lowered_style:
                raise ValueError(
                    f"result SVG contains external content: {name}"
                )
            normalized_style = re.sub(
                r"\s+",
                "",
                decoded_style,
            ).lower()
            if normalized_style != SAFE_STYLE_ELEMENT:
                raise ValueError(f"result SVG contains unsafe CSS: {name}")
        for attribute, value in element.attrib.items():
            local_attribute = _local_xml_name(attribute)
            if local_attribute not in SAFE_SVG_ATTRIBUTES:
                raise ValueError(
                    f"result SVG contains unsafe attribute "
                    f"{local_attribute}: {name}"
                )
            normalized_value = _decode_css_escapes(value.strip())
            if local_attribute == "style":
                _validate_style_attribute(name, normalized_value)
            elif local_attribute == "clip-path":
                if SVG_CLIP_PATH.fullmatch(normalized_value) is None:
                    raise ValueError(
                        f"result SVG contains external content: {name}"
                    )
            elif local_attribute == "href":
                if element_name == "use":
                    if SVG_FRAGMENT.fullmatch(normalized_value) is None:
                        raise ValueError(
                            f"result SVG contains external content: {name}"
                        )
                elif element_name == "image":
                    _validate_embedded_png(name, normalized_value)
                else:
                    raise ValueError(
                        f"result SVG contains external content: {name}"
                    )
            elif local_attribute == "resource":
                if normalized_value != SAFE_METADATA_RESOURCE:
                    raise ValueError(
                        f"result SVG contains external content: {name}"
                    )
            elif local_attribute == "type" and normalized_value != "text/css":
                raise ValueError(f"result SVG contains unsafe type: {name}")


def _validate_result_bytes(name: str, payload: bytes) -> None:
    if name.endswith(".json"):
        _validate_json_bytes(name, payload)
    elif name.endswith(".png"):
        _validate_png_bytes(name, payload)
    elif name.endswith(".svg"):
        _validate_svg_bytes(name, payload)
    else:
        raise ValueError(f"unsupported result artifact: {name}")


def _normalized_result_member(name: str) -> str:
    if "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe result archive member: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe result archive member: {name}")
    normalized = path.as_posix()
    if name != normalized:
        raise ValueError(f"unsafe result archive member: {name}")
    if (
        not normalized.startswith("results/")
        and normalized != OBSERVATION_RAW
    ):
        raise ValueError(f"unsafe result archive member: {name}")
    return normalized


def _validate_result_directories(
    project: Path,
    payloads: dict[str, bytes],
) -> Path:
    results = project / "results"
    directories = {
        project,
        *{
            project / PurePosixPath(name).parent
            for name in payloads
        },
    }
    for directory in sorted(directories, key=lambda path: len(path.parts)):
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
    for directory in sorted(directories, key=lambda path: len(path.parts)):
        directory.mkdir(parents=True, exist_ok=True)
    return results


def _replace_result_payloads(
    payloads: dict[str, bytes],
    project: Path,
) -> None:
    results = _validate_result_directories(project, payloads)
    destinations: dict[str, Path] = {}
    metadata: dict[str, object | None] = {}
    for name in sorted(payloads):
        destination = project / PurePosixPath(name)
        validate_result_destination(destination, project)
        destinations[name] = destination
        if os.path.lexists(destination):
            current = os.lstat(destination)
            validate_result_destination(
                destination,
                project,
                metadata=current,
            )
            metadata[name] = current
        else:
            metadata[name] = None

    with tempfile.TemporaryDirectory(
        prefix=".jspace-result-transaction-",
        dir=results,
    ) as transaction_name:
        transaction = Path(transaction_name)
        prepared = transaction / "prepared"
        backups = transaction / "backups"
        prepared.mkdir()
        backups.mkdir()
        prepared_paths: dict[str, Path] = {}
        backup_paths: dict[str, Path | None] = {}
        for index, name in enumerate(sorted(payloads)):
            prepared_path = prepared / str(index)
            prepared_path.write_bytes(payloads[name])
            prepared_paths[name] = prepared_path
            expected = metadata[name]
            if expected is None:
                backup_paths[name] = None
                continue
            current = os.lstat(destinations[name])
            if not _same_file_identity(expected, current):
                raise ValueError(
                    f"result destination changed after validation: "
                    f"{destinations[name]}"
                )
            backup = backups / str(index)
            with open_validated_file(destinations[name], current) as source:
                with backup.open("xb") as target:
                    shutil.copyfileobj(source, target)
            backup_paths[name] = backup

        replaced: list[str] = []
        try:
            for name in sorted(payloads):
                os.replace(prepared_paths[name], destinations[name])
                replaced.append(name)
        except BaseException:
            rollback_errors: list[BaseException] = []
            for name in reversed(replaced):
                try:
                    backup = backup_paths[name]
                    if backup is None:
                        destinations[name].unlink(missing_ok=True)
                    else:
                        os.replace(backup, destinations[name])
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    "result replacement failed and rollback was incomplete"
                ) from rollback_errors[0]
            raise


def _scan_result_tar_members(
    handle: tarfile.TarFile,
    allowlist: set[str],
) -> list[tuple[str, int]]:
    members: list[tuple[str, int]] = []
    names: set[str] = set()
    total_size = 0
    profile_payload_limit = sum(
        result_size_limit(name) for name in allowlist
    )
    for member in handle:
        name = _normalized_result_member(member.name)
        if name in names:
            raise ValueError("duplicate result archive member")
        if name not in allowlist:
            raise ValueError(f"unexpected result archive member: {name}")
        if len(members) >= len(allowlist):
            raise ValueError("too many result archive members")
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError(
                "result archive must contain regular files only"
            )
        size_limit = result_size_limit(name)
        if member.size < 0 or member.size > size_limit:
            raise ValueError(f"result artifact is too large: {name}")
        total_size += member.size
        if total_size > profile_payload_limit:
            raise ValueError("result archive expands beyond size limit")
        names.add(name)
        members.append((name, member.size))

    missing = allowlist - names
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"missing result archive member: {joined}"
        )
    return members


def _read_validated_result_tar(
    handle: tarfile.TarFile,
    expected: list[tuple[str, int]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    index = 0
    for member in handle:
        if index >= len(expected):
            raise ValueError("result archive changed after validation")
        expected_name, expected_size = expected[index]
        name = _normalized_result_member(member.name)
        if (
            name != expected_name
            or member.size != expected_size
            or not member.isfile()
            or member.issym()
            or member.islnk()
        ):
            raise ValueError("result archive changed after validation")
        size_limit = result_size_limit(name)
        source = handle.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read result artifact: {name}")
        payload = source.read(size_limit + 1)
        if len(payload) > size_limit:
            raise ValueError(f"result artifact is too large: {name}")
        _validate_result_bytes(name, payload)
        payloads[name] = payload
        index += 1
    if index != len(expected):
        raise ValueError("result archive changed after validation")
    return payloads


def extract_results_archive(
    archive: Path,
    project: Path = PROJECT,
    result_set: str = "causal",
) -> None:
    allowlist = result_allowlist(result_set)
    archive_metadata = os.lstat(archive)
    validate_result_archive(archive, metadata=archive_metadata)
    with open_validated_file(archive, archive_metadata) as archive_source:
        with open_bounded_result_tar(
            archive_source,
            result_set,
        ) as handle:
            expected = _scan_result_tar_members(handle, allowlist)

    with open_validated_file(archive, archive_metadata) as archive_source:
        with open_bounded_result_tar(
            archive_source,
            result_set,
        ) as handle:
            payloads = _read_validated_result_tar(handle, expected)

    _replace_result_payloads(payloads, project)


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

    for action in ("sync", "bootstrap", "auth-hf", "stop"):
        action_parser = actions.add_parser(action)
        action_parser.add_argument("--session", default="jspace-llama")

    fetch = actions.add_parser("fetch")
    fetch.add_argument("--session", default="jspace-llama")
    fetch.add_argument(
        "--result-set",
        choices=tuple(RESULT_SETS),
        default="causal",
    )

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
            environment={RESULT_SET_ENV: args.result_set},
        )
        fetch_results(args.session, result_set=args.result_set)
    elif args.action == "stop":
        run(stop_command(args.session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
