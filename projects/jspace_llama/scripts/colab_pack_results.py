from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile


ROOT = Path("/content/jspace_llama")
ARCHIVE = Path("/content/jspace_results.tar.gz")
SUCCESS_MARKER = "JSPACE_RESULTS_PACK_SUCCESS"
SUCCESS_NONCE_ENV = "JSPACE_SUCCESS_NONCE"
MAX_RESULT_BYTES = 1_048_576
RESULT_ALLOWLIST = (
    "execution_manifest.json",
    "final_evaluation.json",
)


def _is_reparse_point(metadata: object) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def consume_success_nonce() -> str:
    nonce = os.environ.pop(SUCCESS_NONCE_ENV, None)
    if nonce is None or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise RuntimeError("missing or invalid success nonce")
    return nonce


def validate_result_file(
    path: Path,
    *,
    metadata: object | None = None,
) -> object:
    file_metadata = metadata if metadata is not None else os.lstat(path)
    mode = file_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"result artifact is a symlink: {path}")
    if _is_reparse_point(file_metadata):
        raise ValueError(f"result artifact is a reparse point: {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"result artifact is not a regular file: {path}")
    if getattr(file_metadata, "st_nlink", 1) != 1:
        raise ValueError(f"result artifact is hard-linked: {path}")
    if file_metadata.st_size > MAX_RESULT_BYTES:
        raise ValueError(f"result artifact is too large: {path}")
    return file_metadata


def validate_result_directory(
    path: Path,
    *,
    metadata: object | None = None,
) -> object:
    directory_metadata = metadata if metadata is not None else os.lstat(path)
    mode = directory_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"result path is a symlink: {path}")
    if _is_reparse_point(directory_metadata):
        raise ValueError(f"result path is a reparse point: {path}")
    if not stat.S_ISDIR(mode):
        raise ValueError(f"result path is not a directory: {path}")
    return directory_metadata


def validate_archive_path(path: Path, metadata: object) -> None:
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"result archive is a symlink: {path}")
    if _is_reparse_point(metadata):
        raise ValueError(f"result archive is a reparse point: {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"result archive is not a regular file: {path}")
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"result archive is hard-linked: {path}")


def open_validated_file(path: Path, metadata: object):
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
            raise ValueError(f"file changed after validation: {path}")
        validate_result_file(path, metadata=actual)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_json_artifact(path: Path) -> tuple[object, bytes]:
    metadata = validate_result_file(path)
    with open_validated_file(path, metadata) as source:
        payload = source.read(MAX_RESULT_BYTES + 1)
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError(f"result artifact is too large: {path}")
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"result artifact is not valid JSON: {path}") from error
    return metadata, payload


def pack_results(root: Path, archive: Path) -> Path:
    if os.path.lexists(archive):
        validate_archive_path(archive, os.lstat(archive))
        archive.unlink()
    results = root / "results"
    validate_result_directory(root)
    missing = [
        name
        for name in RESULT_ALLOWLIST
        if not os.path.lexists(results / name)
    ]
    if missing:
        raise FileNotFoundError(
            "missing required result artifact(s): " + ", ".join(missing)
        )

    validate_result_directory(results)
    resolved_results = results.resolve(strict=True)
    artifacts: list[tuple[str, object, bytes]] = []
    for name in RESULT_ALLOWLIST:
        path = results / name
        try:
            path.resolve(strict=True).relative_to(resolved_results)
        except ValueError as error:
            raise ValueError(
                f"result artifact resolves outside results: {path}"
            ) from error
        metadata, payload = _read_json_artifact(path)
        artifacts.append((name, metadata, payload))

    with tarfile.open(archive, "x:gz") as handle:
        for name, metadata, payload in artifacts:
            info = tarfile.TarInfo(f"results/{name}")
            info.size = len(payload)
            info.mode = stat.S_IMODE(metadata.st_mode)
            info.mtime = int(getattr(metadata, "st_mtime", 0))
            handle.addfile(info, fileobj=io.BytesIO(payload))
    return archive


def main() -> int:
    nonce = consume_success_nonce()
    pack_results(ROOT, ARCHIVE)
    print(f"{SUCCESS_MARKER}:{nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
