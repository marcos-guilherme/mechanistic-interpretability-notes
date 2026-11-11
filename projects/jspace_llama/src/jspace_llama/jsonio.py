"""Deterministic hashing and atomic JSON persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


class StrictJSONError(ValueError):
    pass


class BoundedFileError(ValueError):
    pass


def read_regular_file_bounded(
    path: str | Path,
    *,
    max_bytes: int,
) -> bytes:
    source = Path(path)
    before = os.lstat(source)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BoundedFileError("file is not a regular file")
    if before.st_size > max_bytes:
        raise BoundedFileError("file is too large")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise BoundedFileError("file changed during validation")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
        if len(payload) > max_bytes or after.st_size > max_bytes:
            raise BoundedFileError("file is too large")
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_size != len(payload)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise BoundedFileError("file changed during validation")
        return payload
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON value: {value}")


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    try:
        text = (
            bytes(payload).decode("utf-8")
            if isinstance(payload, (bytes, bytearray))
            else payload
        )
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictJSONError(str(error)) from error


def canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def document_sha256(document: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def atomic_write_json(path: str | Path, document: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                document,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
