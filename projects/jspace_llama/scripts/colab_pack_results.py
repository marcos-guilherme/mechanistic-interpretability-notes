from __future__ import annotations

import io
import math
import os
from pathlib import Path
import re
import stat
import struct
import tarfile
import xml.etree.ElementTree as ET
import zlib

from jspace_llama.jsonio import StrictJSONError, strict_json_loads


ROOT = Path("/content/jspace_llama_project")
ARCHIVE = Path("/content/jspace_results.tar.gz")
SUCCESS_MARKER = "JSPACE_RESULTS_PACK_SUCCESS"
SUCCESS_NONCE_ENV = "JSPACE_SUCCESS_NONCE"
RESULT_SET_ENV = "JSPACE_RESULT_SET"
MAX_RESULT_BYTES = 1_048_576
MAX_OBSERVATION_JSON_BYTES = 5 * 1_048_576
MAX_FIGURE_BYTES = 5 * 1_048_576
MAX_ARCHIVE_BYTES = 40 * 1_048_576
MAX_IMAGE_DIMENSION = 10_000
RESULT_SETS = {
    "causal": frozenset(
        {
            "results/execution_manifest.json",
            "results/final_evaluation.json",
        }
    ),
    "observation": frozenset(
        {
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
OBSERVATION_FIGURE_SOURCES = {
    (
        f"results/jspace_observation_{stem}{suffix}"
    ): f"results/figures/observation_{stem}{suffix}"
    for stem in ("pass_at_k", "rank_heatmap", "panel")
    for suffix in (".png", ".svg")
}


def result_allowlist(result_set: str) -> set[str]:
    try:
        return set(RESULT_SETS[result_set])
    except KeyError as error:
        raise ValueError(f"unknown result set: {result_set}") from error


def result_size_limit(name: str) -> int:
    if name == "results/jspace_observation.json":
        return MAX_OBSERVATION_JSON_BYTES
    if name.endswith((".png", ".svg")):
        return MAX_FIGURE_BYTES
    return MAX_RESULT_BYTES


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
    size_limit: int = MAX_RESULT_BYTES,
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
    if file_metadata.st_size > size_limit:
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


def open_validated_file(
    path: Path,
    metadata: object,
    *,
    size_limit: int = MAX_RESULT_BYTES,
):
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
        validate_result_file(path, metadata=actual, size_limit=size_limit)
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _validate_png(name: str, payload: bytes) -> None:
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


def _validate_svg(name: str, payload: bytes) -> None:
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
    if "@IMPORT" in upper:
        raise ValueError(f"result SVG contains external content: {name}")
    for match in re.finditer(
        r"url\(\s*(['\"]?)(.*?)\1\s*\)",
        text,
        flags=re.IGNORECASE,
    ):
        if not match.group(2).strip().startswith("#"):
            raise ValueError(f"result SVG contains external content: {name}")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError(f"result artifact is not valid XML SVG: {name}") from error
    if _local_xml_name(root.tag) != "svg":
        raise ValueError(f"result SVG root is not svg: {name}")
    for element in root.iter():
        if _local_xml_name(element.tag) in {"script", "foreignobject"}:
            raise ValueError(f"result SVG contains active content: {name}")
        for attribute, value in element.attrib.items():
            local_attribute = _local_xml_name(attribute)
            normalized_value = value.strip()
            lowered_value = normalized_value.lower()
            if local_attribute.startswith("on"):
                raise ValueError(f"result SVG contains active content: {name}")
            if local_attribute in {"href", "src"} and (
                not normalized_value.startswith("#")
            ):
                raise ValueError(f"result SVG contains external content: {name}")
            if "@import" in lowered_value:
                raise ValueError(f"result SVG contains external content: {name}")


def _reject_nonfinite_numbers(document: object) -> None:
    if isinstance(document, float) and not math.isfinite(document):
        raise ValueError("non-finite JSON number")
    if isinstance(document, dict):
        for value in document.values():
            _reject_nonfinite_numbers(value)
    elif isinstance(document, list):
        for value in document:
            _reject_nonfinite_numbers(value)


def _validate_artifact(name: str, payload: bytes) -> None:
    if name.endswith(".json"):
        try:
            document = strict_json_loads(payload)
            _reject_nonfinite_numbers(document)
        except (StrictJSONError, ValueError) as error:
            raise ValueError(f"result artifact is not valid JSON: {name}") from error
    elif name.endswith(".png"):
        _validate_png(name, payload)
    elif name.endswith(".svg"):
        _validate_svg(name, payload)
    else:
        raise ValueError(f"unsupported result artifact: {name}")


def _read_artifact(path: Path, name: str) -> tuple[object, bytes]:
    size_limit = result_size_limit(name)
    metadata = validate_result_file(path, size_limit=size_limit)
    with open_validated_file(
        path,
        metadata,
        size_limit=size_limit,
    ) as source:
        payload = source.read(size_limit + 1)
    if len(payload) > size_limit:
        raise ValueError(f"result artifact is too large: {path}")
    _validate_artifact(name, payload)
    return metadata, payload


def _read_json_artifact(path: Path) -> tuple[object, bytes]:
    """Preserve the causal packer's strict JSON helper contract."""
    return _read_artifact(path, f"results/{path.name}")


def _result_source(root: Path, name: str) -> Path:
    alternate = OBSERVATION_FIGURE_SOURCES.get(name)
    if alternate is not None and os.path.lexists(root / alternate):
        return root / alternate
    return root / name


def pack_results(
    root: Path,
    archive: Path,
    result_set: str | None = None,
) -> Path:
    selected = (
        result_set
        if result_set is not None
        else os.environ.get(RESULT_SET_ENV, "causal")
    )
    allowlist = result_allowlist(selected)
    if os.path.lexists(archive):
        validate_archive_path(archive, os.lstat(archive))
        archive.unlink()
    results = root / "results"
    validate_result_directory(root)
    missing = [
        name
        for name in sorted(allowlist)
        if not os.path.lexists(_result_source(root, name))
    ]
    if missing:
        raise FileNotFoundError(
            "missing required result artifact(s): " + ", ".join(missing)
        )

    validate_result_directory(results)
    resolved_results = results.resolve(strict=True)
    artifacts: list[tuple[str, object, bytes]] = []
    for name in sorted(allowlist):
        path = _result_source(root, name)
        try:
            path.resolve(strict=True).relative_to(resolved_results)
        except ValueError as error:
            raise ValueError(
                f"result artifact resolves outside results: {path}"
            ) from error
        metadata, payload = _read_artifact(path, name)
        artifacts.append((name, metadata, payload))

    try:
        with tarfile.open(archive, "x:gz") as handle:
            for name, metadata, payload in artifacts:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = stat.S_IMODE(metadata.st_mode)
                info.mtime = int(getattr(metadata, "st_mtime", 0))
                handle.addfile(info, fileobj=io.BytesIO(payload))
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError(f"result archive is too large: {archive}")
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive


def main() -> int:
    nonce = consume_success_nonce()
    pack_results(
        ROOT,
        ARCHIVE,
        os.environ.get(RESULT_SET_ENV, "causal"),
    )
    print(f"{SUCCESS_MARKER}:{nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
