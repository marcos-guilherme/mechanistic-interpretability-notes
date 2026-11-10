from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import tarfile


ROOT = Path("/content/jspace_llama")
ARCHIVE = Path("/content/jspace_results.tar.gz")
SUCCESS_MARKER = "JSPACE_RESULTS_PACK_SUCCESS"
MAX_RESULT_BYTES = 1_048_576
RESULT_ALLOWLIST = (
    "execution_manifest.json",
    "final_evaluation.json",
)


def _is_reparse_point(metadata: object) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


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


def _read_json_artifact(path: Path) -> tuple[object, bytes]:
    metadata = validate_result_file(path)
    with path.open("rb") as source:
        payload = source.read(MAX_RESULT_BYTES + 1)
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError(f"result artifact is too large: {path}")
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"result artifact is not valid JSON: {path}") from error
    return metadata, payload


def pack_results(root: Path, archive: Path) -> Path:
    archive.unlink(missing_ok=True)
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

    with tarfile.open(archive, "w:gz") as handle:
        for name, metadata, payload in artifacts:
            info = tarfile.TarInfo(f"results/{name}")
            info.size = len(payload)
            info.mode = stat.S_IMODE(metadata.st_mode)
            info.mtime = int(getattr(metadata, "st_mtime", 0))
            handle.addfile(info, fileobj=io.BytesIO(payload))
    return archive


def main() -> int:
    pack_results(ROOT, ARCHIVE)
    print(SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
