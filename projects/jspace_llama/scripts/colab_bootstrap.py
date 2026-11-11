from __future__ import annotations

from contextlib import contextmanager
import gzip
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
UPGRADED_MODULE_ROOTS = frozenset({"matplotlib"})
MAX_PROJECT_ARCHIVE_BYTES = 16 * 1_048_576
MAX_PROJECT_MEMBER_BYTES = 8 * 1_048_576
MAX_PROJECT_TOTAL_BYTES = 16 * 1_048_576
MAX_PROJECT_MEMBERS = 1_024
MAX_TAR_METADATA_OVERHEAD_BYTES = 1_048_576
TAR_BLOCK_BYTES = 512
MAX_DECOMPRESSED_TAR_BYTES = (
    MAX_PROJECT_TOTAL_BYTES
    + MAX_PROJECT_MEMBERS * TAR_BLOCK_BYTES
    + MAX_TAR_METADATA_OVERHEAD_BYTES
)
MATPLOTLIB_SMOKE_SCRIPT = """\
import importlib.metadata
import io

import matplotlib

matplotlib.use("svg")

from matplotlib import ft2font
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.text import Text

installed_version = importlib.metadata.version("matplotlib")
if matplotlib.__version__ != installed_version:
    raise RuntimeError(
        f"Matplotlib metadata mismatch: "
        f"{matplotlib.__version__} != {installed_version}"
    )
_ = Text.get_fontfeatures
_ = ft2font.CharacterCodeType
figure = Figure()
canvas = FigureCanvasSVG(figure)
figure.text(0.5, 0.5, "bootstrap smoke")
output = io.BytesIO()
canvas.print_svg(output)
if not output.getvalue():
    raise RuntimeError("Matplotlib SVG smoke render was empty")
"""


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
    if getattr(metadata, "st_size", 0) > MAX_PROJECT_ARCHIVE_BYTES:
        raise ValueError(f"project archive is too large: {path}")


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
                "decompressed TAR exceeds project archive limit"
            )
        self._count += len(payload)
        return payload


@contextmanager
def _open_bounded_project_tar(archive_source):
    with gzip.GzipFile(fileobj=archive_source, mode="rb") as decompressed:
        bounded = _BoundedDecompressedReader(
            decompressed,
            MAX_DECOMPRESSED_TAR_BYTES,
        )
        with tarfile.open(fileobj=bounded, mode="r|") as handle:
            yield handle


def _validated_member_path(
    member: tarfile.TarInfo,
    destination: Path,
) -> Path:
    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        raise ValueError("project archive must contain regular files only")
    if "\x00" in member.name or "\\" in member.name:
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


def _scan_project_tar_members(
    handle: tarfile.TarFile,
    destination: Path,
) -> list[tuple[str, int, int]]:
    members: list[tuple[str, int, int]] = []
    names: set[str] = set()
    targets: set[Path] = set()
    total_size = 0
    for member in handle:
        if len(members) >= MAX_PROJECT_MEMBERS:
            raise ValueError("too many project archive members")
        if member.name in names:
            raise ValueError("duplicate archive member")
        target = _validated_member_path(member, destination)
        if target in targets:
            raise ValueError("duplicate archive member destination")
        normalized = PurePosixPath(member.name).as_posix()
        if member.name != normalized:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.size < 0 or member.size > MAX_PROJECT_MEMBER_BYTES:
            raise ValueError(
                f"project archive member is too large: {member.name}"
            )
        total_size += member.size
        if total_size > MAX_PROJECT_TOTAL_BYTES:
            raise ValueError("project archive expands beyond size limit")
        names.add(member.name)
        targets.add(target)
        members.append(
            (
                member.name,
                member.size,
                stat.S_IMODE(member.mode),
            )
        )
    if not members:
        raise ValueError("project archive is empty")
    return members


def _copy_member_payload(
    source,
    output,
    declared_size: int,
    name: str,
) -> None:
    remaining = declared_size
    while remaining:
        payload = source.read(min(64 * 1_024, remaining))
        if not payload or len(payload) > remaining:
            raise ValueError(
                f"project archive member differs from declared size: {name}"
            )
        output.write(payload)
        remaining -= len(payload)
    if source.read(1):
        raise ValueError(
            f"project archive member differs from declared size: {name}"
        )


def _extract_project_tar_members(
    handle: tarfile.TarFile,
    destination: Path,
    expected: list[tuple[str, int, int]],
) -> None:
    index = 0
    for member in handle:
        if index >= len(expected):
            raise ValueError("project archive changed after validation")
        expected_name, expected_size, expected_mode = expected[index]
        target = _validated_member_path(member, destination)
        if (
            member.name != expected_name
            or member.size != expected_size
            or stat.S_IMODE(member.mode) != expected_mode
        ):
            raise ValueError("project archive changed after validation")
        source = handle.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            _copy_member_payload(
                source,
                output,
                expected_size,
                expected_name,
            )
        target.chmod(expected_mode)
        index += 1
    if index != len(expected):
        raise ValueError("project archive changed after validation")


def safe_extract(archive: Path, destination: Path) -> Path:
    archive_metadata = os.lstat(archive)
    _validate_archive_metadata(archive, archive_metadata)
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".jspace-extract-", dir=destination)
    )
    final_project = destination / ARCHIVE_ROOT
    try:
        with _open_validated_archive(
            archive,
            archive_metadata,
        ) as archive_source:
            with _open_bounded_project_tar(archive_source) as handle:
                expected = _scan_project_tar_members(handle, staging)
        with _open_validated_archive(
            archive,
            archive_metadata,
        ) as archive_source:
            with _open_bounded_project_tar(archive_source) as handle:
                _extract_project_tar_members(handle, staging, expected)
        extracted_project = staging / ARCHIVE_ROOT
        if not extracted_project.is_dir():
            raise ValueError(
                f"project archive did not create {ARCHIVE_ROOT}"
            )
        if os.path.lexists(final_project):
            raise FileExistsError(
                f"project archive destination exists: {final_project}"
            )
        extracted_project.replace(final_project)
        return final_project
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
    module_roots = {*UPGRADED_MODULE_ROOTS, package_name}
    for name in tuple(sys.modules):
        if any(
            name == root or name.startswith(f"{root}.")
            for root in module_roots
        ):
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
            "--reinstall-package",
            "matplotlib",
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
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "--yes",
            "torchaudio",
        ],
    ]


def matplotlib_smoke_command() -> list[str]:
    return [sys.executable, "-c", MATPLOTLIB_SMOKE_SCRIPT]


def run_matplotlib_smoke_check() -> None:
    run_checked(matplotlib_smoke_command())


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
    run_matplotlib_smoke_check()
    print(f"{SUCCESS_MARKER}:{nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
