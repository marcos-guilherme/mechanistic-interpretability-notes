from __future__ import annotations

import argparse
import base64
import binascii
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
from typing import Any
import xml.etree.ElementTree as ET
import zlib

import jsonschema


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jspace_llama.jsonio import (
    BoundedFileError,
    StrictJSONError,
    document_sha256,
    read_regular_file_bounded,
    strict_json_loads,
)
from jspace_llama.observation_data import (
    DATASET_SHA256,
    EXPECTED_ITEM_COUNT,
    PANEL_NAMES,
)


class ObservationValidationError(ValueError):
    pass


METHODS = ("jlens", "logit_lens")
SCOPES = ("band", "all_layers")
KS = (1, 5, 10)
FORBIDDEN_KEYS = {
    "activation",
    "activations",
    "logit",
    "logits",
    "continuation",
    "tensor",
    "tensors",
    "credential",
    "credentials",
    "secret",
    "password",
    "api_key",
    "cache_path",
    "model_path",
    "lens_path",
}
ABSOLUTE_PATHS = (
    re.compile(r"(?i)\bfile://"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]"),
    re.compile(r"\\\\[^\\/\s]+[\\/][^\\/\s]+"),
    re.compile(r"(?<![:/\w])/(?!/)[^\s\"'<>]+"),
)
REASON_MESSAGES = {
    "incomplete_observation": "observational run did not produce every required record",
    "incomplete_provenance": "reproducible provenance is incomplete",
    "cuda_required": "a CUDA runtime is required",
    "asset_load_failed": "model or lens asset loading failed",
    "resource_exhausted": "GPU resources were exhausted",
    "tokenization_failed": "observational tokenization failed",
    "dimension_mismatch": "model and lens dimensions do not match",
    "layer_band_invalid": "the selected layer band is invalid",
    "lens_integrity_failed": "lens integrity verification failed",
    "dataset_invalid": "the pinned observational dataset is invalid",
    "readout_failed": "observational readout failed",
    "unexpected_error": "unexpected programming error; see traceback",
}
FAILURE_ERROR_TYPES = {
    "incomplete_observation": frozenset({"ObservationDataError"}),
    "cuda_required": frozenset({"CudaRequiredError"}),
    "asset_load_failed": frozenset({"AssetLoadError"}),
    "resource_exhausted": frozenset(
        {"ResourceExhaustionError", "OutOfMemoryError"}
    ),
    "tokenization_failed": frozenset({"TokenizationError"}),
    "dimension_mismatch": frozenset({"DimensionMismatchError"}),
    "layer_band_invalid": frozenset({"LayerBandError"}),
    "lens_integrity_failed": frozenset({"LensIntegrityError"}),
    "dataset_invalid": frozenset({"DatasetValidationError"}),
    "readout_failed": frozenset({"ObservationReadoutError"}),
}
MAX_METADATA_JSON_BYTES = 2_000_000
MAX_OBSERVATION_JSON_BYTES = 5 * 1_048_576
MAX_FIGURE_BYTES = 5 * 1_048_576
MAX_IMAGE_DIMENSION = 10_000
MAX_PNG_CHUNKS = 4_096
MAX_EMBEDDED_PNG_BYTES = 4 * 1_048_576
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FIGURE_NAMES = tuple(
    f"jspace_observation_{stem}{suffix}"
    for stem in ("pass_at_k", "rank_heatmap", "panel")
    for suffix in (".png", ".svg")
)
AUDIT_ITEM_FIELDS = {
    "name",
    "prompt",
    "prompt_token_ids",
    "prompt_token_count",
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "intermediates",
}
AUDIT_INTERMEDIATE_FIELDS = {
    "text",
    "token_id",
    "token_ids",
    "variant",
    "single_token",
    "prompt_sha256",
    "token_ids_sha256",
}


def _read_json(
    path: Path,
    *,
    max_bytes: int = MAX_METADATA_JSON_BYTES,
) -> dict[str, Any]:
    try:
        payload = read_regular_file_bounded(path, max_bytes=max_bytes)
        document = strict_json_loads(payload)
    except (BoundedFileError, OSError, StrictJSONError) as error:
        raise ObservationValidationError(
            f"invalid JSON file {path.name}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ObservationValidationError(
            f"JSON root is not an object: {path.name}"
        )
    return document


def _validate_schema(
    document: dict[str, Any],
    schema_path: Path,
) -> None:
    schema = _read_json(schema_path)
    try:
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        ).validate(document)
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        raise ObservationValidationError(
            f"schema validation failed for {schema_path.name}"
            f"{' at ' + location if location else ''}: {error.message}"
        ) from error


def _is_decoded_token_path(path: tuple[str | int, ...]) -> bool:
    method_names = {"jlens", "logit_lens"}
    if len(path) == 8:
        return (
            path[0] == "panel"
            and isinstance(path[1], int)
            and path[2] == "top_tokens"
            and path[3] in method_names
            and isinstance(path[4], int)
            and path[5] == "tokens"
            and isinstance(path[6], int)
            and path[7] == "token"
        )
    if len(path) == 7:
        return (
            path[0] == "panel"
            and isinstance(path[1], int)
            and path[2] == "pursuit"
            and isinstance(path[3], int)
            and path[4] == "tokens"
            and isinstance(path[5], int)
            and path[6] == "token"
        )
    if len(path) == 10:
        return (
            path[0] == "raw"
            and path[1] in {"direct", "panel"}
            and path[2] == "items"
            and isinstance(path[3], int)
            and path[4] == "top_tokens"
            and path[5] in method_names
            and isinstance(path[6], int)
            and path[7] == "tokens"
            and isinstance(path[8], int)
            and path[9] == "token"
        )
    if len(path) == 9:
        return (
            path[0:3] == ("raw", "panel", "items")
            and isinstance(path[3], int)
            and path[4] == "pursuit"
            and isinstance(path[5], int)
            and path[6] == "tokens"
            and isinstance(path[7], int)
            and path[8] == "token"
        )
    return False


def _reject_forbidden(
    value: Any,
    location: str = "$",
    *,
    path: tuple[str | int, ...] = (),
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_KEYS or normalized.endswith(
                ("_credential", "_secret", "_password", "_api_key")
            ):
                raise ObservationValidationError(
                    f"forbidden field at {location}.{key}"
                )
            _reject_forbidden(
                child,
                f"{location}.{key}",
                path=(*path, str(key)),
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(
                child,
                f"{location}[{index}]",
                path=(*path, index),
            )
    elif isinstance(value, float) and not math.isfinite(value):
        raise ObservationValidationError(
            f"non-finite value at {location}"
        )
    elif (
        isinstance(value, str)
        and not _is_decoded_token_path(path)
        and any(pattern.search(value) for pattern in ABSOLUTE_PATHS)
    ):
        raise ObservationValidationError(
            f"absolute/local path at {location}"
        )


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _provenance_complete(provenance: dict[str, Any]) -> bool:
    try:
        source = provenance["source"]
        model = provenance["model"]
        lens = provenance["lens"]
        software = provenance["software"]
        hardware = provenance["hardware"]
        return all(
            (
                _is_hex(source["commit"], 40),
                isinstance(source["dirty"], bool),
                source["scoped_dirty"] is False,
                _is_hex(source["runtime_source_sha256"], 64),
                model["id"] == "meta-llama/Llama-3.1-8B-Instruct",
                _is_hex(model["resolved_revision"], 40),
                model["layer_count"] == 32,
                lens["repo_id"] == "neuronpedia/jacobian-lens",
                lens["filename"].endswith(
                    "Llama-3.1-8B-Instruct_jacobian_lens.pt"
                ),
                _is_hex(lens["resolved_revision"], 40),
                _is_hex(lens["sha256"], 64),
                isinstance(lens["size_bytes"], int),
                not isinstance(lens["size_bytes"], bool),
                lens["size_bytes"] > 0,
                lens["integrity_verified"] is True,
                software["jspace_commit"]
                == "04c822a255fe0a10d84e08ec499e5e09f4b7d37a",
                all(
                    isinstance(software[key], str) and software[key]
                    for key in (
                        "python",
                        "pytorch",
                        "transformers",
                        "huggingface_hub",
                        "jspace",
                    )
                ),
                isinstance(hardware["gpu_name"], str)
                and bool(hardware["gpu_name"]),
                isinstance(hardware["cuda_version"], str)
                and bool(hardware["cuda_version"]),
                hardware["dtype"] == "bfloat16",
            )
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _runtime_git_pathspecs(project: Path, repository: Path) -> list[str]:
    try:
        relative = project.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise ObservationValidationError(
            "project root is outside its source repository"
        ) from error
    return [
        f"{relative}/data",
        f"{relative}/src",
        f"{relative}/notebooks",
        f"{relative}/schemas",
        f"{relative}/scripts",
        f"{relative}/pyproject.toml",
        f"{relative}/uv.lock",
        "tools/colab/run_jspace.py",
    ]


def _is_runtime_git_path(path: str, project_relative: str) -> bool:
    relative = path.removeprefix(f"{project_relative}/")
    if relative == path:
        return path == "tools/colab/run_jspace.py"
    if relative in {"pyproject.toml", "uv.lock"}:
        return True
    if relative.startswith("data/"):
        suffix = Path(relative).suffix.lower()
        return (
            suffix in {".json", ".md"}
            or relative == "data/anthropic-jlens/LICENSE"
        )
    if relative.startswith(("src/", "scripts/")):
        return relative.endswith(".py")
    if relative.startswith("schemas/"):
        return relative.endswith(".json")
    if relative.startswith("notebooks/"):
        return (
            relative.endswith(".ipynb")
            and not relative.endswith("_output.ipynb")
        )
    return False


def _git_bytes(
    repository: Path,
    command: list[str],
    *,
    error_message: str,
) -> bytes:
    completed = subprocess.run(
        command,
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ObservationValidationError(error_message)
    return completed.stdout


@lru_cache(maxsize=8)
def _runtime_source_sha256_at_commit(
    project: Path,
    commit: str,
) -> str:
    repository = project.resolve().parents[1]
    project_relative = project.resolve().relative_to(repository).as_posix()
    tree = _git_bytes(
        repository,
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            *_runtime_git_pathspecs(project, repository),
        ],
        error_message="source commit is unavailable from local repository",
    )
    paths = [
        path
        for path in tree.decode("utf-8").splitlines()
        if _is_runtime_git_path(path, project_relative)
    ]
    if not paths:
        raise ObservationValidationError(
            "source commit contains no runtime-critical files"
        )
    digest = hashlib.sha256()
    for label in sorted(paths):
        content = _git_bytes(
            repository,
            ["git", "show", f"{commit}:{label}"],
            error_message=f"source commit cannot read runtime file: {label}",
        )
        label_bytes = label.encode("utf-8")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_source_provenance(
    source: dict[str, Any],
    *,
    project: Path,
) -> None:
    commit = source.get("commit")
    digest = source.get("runtime_source_sha256")
    if commit is None and digest is None:
        return
    if not _is_hex(commit, 40) or not _is_hex(digest, 64):
        raise ObservationValidationError(
            "source commit and runtime digest must be concrete together"
        )
    actual = _runtime_source_sha256_at_commit(project, commit)
    if digest != actual:
        raise ObservationValidationError(
            "runtime source SHA-256 does not match source commit"
        )


def _validate_png(name: str, payload: bytes) -> None:
    if len(payload) > MAX_FIGURE_BYTES or not payload.startswith(PNG_SIGNATURE):
        raise ObservationValidationError(f"invalid PNG figure: {name}")
    offset = len(PNG_SIGNATURE)
    chunks = 0
    seen_ihdr = seen_plte = seen_idat = seen_iend = False
    idat_finished = False
    idat_bytes = 0
    bit_depth = color_type = -1
    plte_entries = 0
    while offset < len(payload):
        if seen_iend or len(payload) - offset < 12:
            raise ObservationValidationError(f"truncated/trailing PNG: {name}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        if length > MAX_FIGURE_BYTES or length > len(payload) - offset - 12:
            raise ObservationValidationError(f"invalid PNG chunk length: {name}")
        if (
            not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in chunk_type)
            or not 65 <= chunk_type[2] <= 90
        ):
            raise ObservationValidationError(f"invalid PNG chunk type: {name}")
        start = offset + 8
        end = start + length
        data = payload[start:end]
        expected_crc = struct.unpack(">I", payload[end : end + 4])[0]
        if expected_crc != zlib.crc32(chunk_type + data) & 0xFFFFFFFF:
            raise ObservationValidationError(f"invalid PNG checksum: {name}")
        offset = end + 4
        chunks += 1
        if chunks > MAX_PNG_CHUNKS or (chunks == 1 and chunk_type != b"IHDR"):
            raise ObservationValidationError(f"invalid PNG chunk order: {name}")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunks != 1 or length != 13:
                raise ObservationValidationError(f"invalid PNG IHDR: {name}")
            seen_ihdr = True
            width, height = struct.unpack(">II", data[:8])
            if not (
                0 < width <= MAX_IMAGE_DIMENSION
                and 0 < height <= MAX_IMAGE_DIMENSION
            ):
                raise ObservationValidationError(f"invalid PNG dimensions: {name}")
            bit_depth, color_type, compression, filter_method, interlace = data[8:]
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                bit_depth not in valid_depths.get(color_type, set())
                or compression != 0
                or filter_method != 0
                or interlace not in {0, 1}
            ):
                raise ObservationValidationError(f"invalid PNG IHDR fields: {name}")
        elif chunk_type == b"PLTE":
            if (
                not seen_ihdr
                or seen_plte
                or seen_idat
                or length == 0
                or length % 3
                or length > 768
            ):
                raise ObservationValidationError(f"invalid PNG PLTE: {name}")
            seen_plte = True
            plte_entries = length // 3
        elif chunk_type == b"IDAT":
            if not seen_ihdr or idat_finished:
                raise ObservationValidationError(f"misordered PNG IDAT: {name}")
            seen_idat = True
            idat_bytes += length
        elif chunk_type == b"IEND":
            if not seen_ihdr or not seen_idat or length != 0 or offset != len(payload):
                raise ObservationValidationError(f"invalid PNG IEND: {name}")
            seen_iend = True
        elif 65 <= chunk_type[0] <= 90:
            raise ObservationValidationError(f"unknown critical PNG chunk: {name}")
        if seen_idat and chunk_type not in {b"IDAT", b"IEND"}:
            idat_finished = True
    if not (seen_ihdr and seen_idat and idat_bytes and seen_iend):
        raise ObservationValidationError(f"incomplete PNG: {name}")
    if color_type == 3 and (not seen_plte or plte_entries > 2**bit_depth):
        raise ObservationValidationError(f"invalid indexed PNG palette: {name}")
    if color_type in {0, 4} and seen_plte:
        raise ObservationValidationError(f"forbidden PNG palette: {name}")


SAFE_SVG_ELEMENTS = frozenset(
    {
        "agent", "clippath", "creator", "date", "defs", "format", "g",
        "image", "metadata", "path", "rdf", "rect", "style", "svg",
        "title", "type", "use", "work",
    }
)
SAFE_SVG_ATTRIBUTES = frozenset(
    {
        "clip-path", "d", "height", "href", "id", "resource", "style",
        "transform", "type", "version", "viewbox", "width", "x", "y",
    }
)
SAFE_STYLE_PROPERTIES = frozenset(
    {
        "fill", "opacity", "stroke", "stroke-linecap", "stroke-linejoin",
        "stroke-opacity", "stroke-width",
    }
)
CSS_ESCAPE = re.compile(
    r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|([^0-9a-fA-F\r\n\f])|(\r\n|[\r\n\f]))"
)
SVG_FRAGMENT = re.compile(r"#[A-Za-z_][A-Za-z0-9_.:-]*\Z")
SVG_CLIP_PATH = re.compile(
    r"url\(\s*#[A-Za-z_][A-Za-z0-9_.:-]*\s*\)\Z",
    flags=re.IGNORECASE,
)


def _decode_css_escapes(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        hexadecimal, escaped, newline = match.groups()
        if hexadecimal is not None:
            codepoint = int(hexadecimal, 16)
            if codepoint == 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                return "\N{REPLACEMENT CHARACTER}"
            return chr(codepoint)
        if escaped is not None:
            return escaped
        if newline is not None:
            return ""
        raise AssertionError("unreachable CSS escape")

    return CSS_ESCAPE.sub(replace, value)


def _validate_style(name: str, value: str) -> None:
    normalized = _decode_css_escapes(value)
    if (
        "url" in normalized.lower()
        or "@import" in normalized.lower()
        or "\\" in normalized
        or "@" in normalized
    ):
        raise ObservationValidationError(f"unsafe SVG CSS: {name}")
    for declaration in normalized.split(";"):
        if not declaration.strip():
            continue
        if ":" not in declaration:
            raise ObservationValidationError(f"unsafe SVG CSS: {name}")
        prop, prop_value = declaration.split(":", 1)
        if (
            prop.strip().lower() not in SAFE_STYLE_PROPERTIES
            or re.fullmatch(r"[-#A-Za-z0-9.,%+ ]+", prop_value.strip()) is None
        ):
            raise ObservationValidationError(f"unsafe SVG CSS: {name}")


def _validate_embedded_png(name: str, value: str) -> None:
    prefix = "data:image/png;base64,"
    if not value.lower().startswith(prefix):
        raise ObservationValidationError(f"external SVG content: {name}")
    encoded = "".join(value[len(prefix) :].split())
    if len(encoded) > ((MAX_EMBEDDED_PNG_BYTES + 2) // 3) * 4:
        raise ObservationValidationError(f"embedded PNG too large: {name}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ObservationValidationError(f"invalid embedded PNG: {name}") from error
    if len(payload) > MAX_EMBEDDED_PNG_BYTES:
        raise ObservationValidationError(f"embedded PNG too large: {name}")
    _validate_png(name, payload)


def _local_xml_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].lower()


def _validate_svg(name: str, payload: bytes) -> None:
    if len(payload) > MAX_FIGURE_BYTES:
        raise ObservationValidationError(f"SVG figure is too large: {name}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ObservationValidationError(f"invalid UTF-8 SVG: {name}") from error
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper or "<?XML-STYLESHEET" in upper:
        raise ObservationValidationError(f"active SVG content: {name}")
    without_declaration = re.sub(
        r"^\s*<\?xml[^?]*\?>", "", text, count=1, flags=re.IGNORECASE
    )
    if "<?" in without_declaration:
        raise ObservationValidationError(f"active SVG content: {name}")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ObservationValidationError(f"invalid XML SVG: {name}") from error
    if _local_xml_name(root.tag) != "svg":
        raise ObservationValidationError(f"SVG root is not svg: {name}")
    for element in root.iter():
        element_name = _local_xml_name(element.tag)
        if element_name not in SAFE_SVG_ELEMENTS:
            raise ObservationValidationError(f"unsafe SVG element: {name}")
        if element_name == "style":
            normalized = re.sub(
                r"\s+", "", _decode_css_escapes(element.text or "")
            ).lower()
            if normalized != "*{stroke-linejoin:round;stroke-linecap:butt}":
                raise ObservationValidationError(f"unsafe SVG style: {name}")
        for attribute, value in element.attrib.items():
            local = _local_xml_name(attribute)
            normalized = _decode_css_escapes(value.strip())
            if local not in SAFE_SVG_ATTRIBUTES or local.startswith("on"):
                raise ObservationValidationError(f"unsafe SVG attribute: {name}")
            if local == "style":
                _validate_style(name, normalized)
            elif local == "clip-path" and SVG_CLIP_PATH.fullmatch(normalized) is None:
                raise ObservationValidationError(f"external SVG content: {name}")
            elif local == "href":
                if element_name == "use":
                    if SVG_FRAGMENT.fullmatch(normalized) is None:
                        raise ObservationValidationError(f"external SVG content: {name}")
                elif element_name == "image":
                    _validate_embedded_png(name, normalized)
                else:
                    raise ObservationValidationError(f"external SVG content: {name}")
            elif local == "resource" and normalized != "http://purl.org/dc/dcmitype/StillImage":
                raise ObservationValidationError(f"external SVG content: {name}")
            elif local == "type" and normalized != "text/css":
                raise ObservationValidationError(f"unsafe SVG type: {name}")


def _validate_figures(results_dir: Path, *, success: bool) -> None:
    existing = {
        path.name
        for path in results_dir.iterdir()
        if path.name.startswith("jspace_observation_")
        and path.suffix.lower() in {".png", ".svg"}
    }
    expected = set(FIGURE_NAMES)
    if not success:
        if existing & expected:
            raise ObservationValidationError(
                "non-success observation must not publish success figures"
            )
        return
    if existing != expected:
        raise ObservationValidationError(
            "canonical observation figure set is incomplete or substituted"
        )
    for name in FIGURE_NAMES:
        path = results_dir / name
        try:
            payload = read_regular_file_bounded(
                path,
                max_bytes=MAX_FIGURE_BYTES,
            )
        except (BoundedFileError, OSError) as error:
            raise ObservationValidationError(
                f"invalid figure file {name}: {error}"
            ) from error
        if path.suffix == ".png":
            _validate_png(name, payload)
        else:
            _validate_svg(name, payload)


def _metric(
    records: list[dict[str, Any]],
    method: str,
    scope: str,
) -> dict[str, float | int]:
    rank_key = "best_rank" if scope == "band" else "all_layers_best_rank"
    totals = {k: 0.0 for k in KS}
    n_intermediates = 0
    for record in records:
        intermediates = record["intermediates"]
        n_intermediates += len(intermediates)
        for k in KS:
            totals[k] += sum(
                int(intermediate[method][rank_key]) <= k
                for intermediate in intermediates
            ) / len(intermediates)
    return {
        "n_items": len(records),
        "n_intermediates": n_intermediates,
        **{
            f"pass@{k}": totals[k] / len(records) if records else 0.0
            for k in KS
        },
    }


def _validate_counts_and_claims(result: dict[str, Any]) -> bool:
    records = result["per_item_best_ranks"]
    panel = result["panel"]
    complete = (
        len(records) == EXPECTED_ITEM_COUNT
        and len({record["name"] for record in records})
        == EXPECTED_ITEM_COUNT
        and [item["name"] for item in panel] == list(PANEL_NAMES)
        and result["panel_names"] == list(PANEL_NAMES)
    )
    summary = result["tokenization_summary"]
    intermediate_count = sum(
        len(record["intermediates"]) for record in records
    )
    single_count = sum(
        intermediate["single_token"] is True
        for record in records
        for intermediate in record["intermediates"]
    )
    expected_summary = {
        "item_count": len(records),
        "intermediate_count": intermediate_count,
        "single_token_count": single_count,
        "multi_token_count": intermediate_count - single_count,
    }
    if summary != expected_summary:
        raise ObservationValidationError(
            "tokenization counts do not recompute"
        )
    for method in METHODS:
        for scope in SCOPES:
            expected = _metric(records, method, scope)
            claimed = result["metrics"][method][scope]
            if claimed != expected:
                raise ObservationValidationError(
                    f"pass@k claims do not recompute for {method}/{scope}"
                )
    band = result["layers"]["band"]
    all_layers = result["layers"]["all_layers"]
    if complete and (not band or not all_layers or not set(band) <= set(all_layers)):
        raise ObservationValidationError("layer scopes are incomplete")
    for record in records:
        prompt_ids = record["prompt_token_ids"]
        if (
            not isinstance(record["prompt"], str)
            or not record["prompt"].endswith((" ", ' "'))
            or not isinstance(prompt_ids, list)
            or not prompt_ids
            or record["prompt_token_count"] != len(prompt_ids)
            or record["prompt_sha256"] != _text_sha256(record["prompt"])
            or record["prompt_token_ids_sha256"]
            != _token_ids_sha256(prompt_ids)
        ):
            raise ObservationValidationError(
                "compact prompt tokenization audit is invalid"
            )
        if not record["intermediates"]:
            raise ObservationValidationError("item has no intermediates")
        for intermediate in record["intermediates"]:
            if (
                not isinstance(intermediate["token_id"], int)
                or isinstance(intermediate["token_id"], bool)
                or intermediate["token_id"] < 0
            ):
                raise ObservationValidationError(
                    "compact intermediate token ID is invalid"
                )
            for method in METHODS:
                detail = intermediate[method]
                if [rank["layer"] for rank in detail["ranks"]] != all_layers:
                    raise ObservationValidationError(
                        "rank layers differ from all-layer scope"
                    )
                by_layer = {
                    rank["layer"]: rank["rank"] for rank in detail["ranks"]
                }
                band_best = min(
                    ((by_layer[layer], layer) for layer in band),
                    key=lambda pair: pair[0],
                )
                all_best = min(
                    ((by_layer[layer], layer) for layer in all_layers),
                    key=lambda pair: pair[0],
                )
                if (
                    detail["best_rank"],
                    detail["best_layer"],
                ) != band_best or (
                    detail["all_layers_best_rank"],
                    detail["all_layers_best_layer"],
                ) != all_best:
                    raise ObservationValidationError(
                        "best-rank claims do not recompute"
                    )
    for item in panel:
        for method in METHODS:
            if [
                slot["layer"] for slot in item["top_tokens"][method]
            ] != band:
                raise ObservationValidationError(
                    "top-token detail is not band-only"
                )
        if [slot["layer"] for slot in item["pursuit"]] != band:
            raise ObservationValidationError(
                "pursuit detail differs from workspace band"
            )
    return complete


def _validate_raw(
    manifest: dict[str, Any],
    *,
    raw_path: Path,
) -> dict[str, Any]:
    claimed = manifest["raw_artifact_sha256"]
    try:
        payload = read_regular_file_bounded(
            raw_path,
            max_bytes=MAX_OBSERVATION_JSON_BYTES,
        )
    except FileNotFoundError as error:
        raise ObservationValidationError("raw artifact is required") from error
    except (BoundedFileError, OSError) as error:
        raise ObservationValidationError(
            f"raw artifact cannot be read: {error}"
        ) from error
    if hashlib.sha256(payload).hexdigest() != claimed:
        raise ObservationValidationError(
            "raw artifact SHA-256 does not match manifest"
        )
    try:
        raw = strict_json_loads(payload)
    except StrictJSONError as error:
        raise ObservationValidationError("raw artifact is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ObservationValidationError("raw artifact root is not an object")
    return raw


def _raw_tokenization_summary(
    audit: list[dict[str, Any]],
) -> dict[str, int]:
    intermediates = [
        intermediate
        for item in audit
        for intermediate in item["intermediates"]
    ]
    single = sum(
        intermediate["single_token"] is True
        for intermediate in intermediates
    )
    return {
        "item_count": len(audit),
        "intermediate_count": len(intermediates),
        "single_token_count": single,
        "multi_token_count": len(intermediates) - single,
    }


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    canonical = "[" + ",".join(str(token_id) for token_id in token_ids) + "]"
    return _text_sha256(canonical)


def _validate_tokenization_audit(
    audit: Any,
    direct_records: list[dict[str, Any]],
) -> None:
    if not isinstance(audit, list) or len(audit) != len(direct_records):
        raise ObservationValidationError(
            "raw tokenization audit count is invalid"
        )
    for audit_item, direct_item in zip(audit, direct_records, strict=True):
        if (
            not isinstance(audit_item, dict)
            or set(audit_item) != AUDIT_ITEM_FIELDS
            or audit_item["name"] != direct_item["name"]
            or audit_item["prompt"] != direct_item["prompt"]
            or not isinstance(audit_item["prompt_token_ids"], list)
            or not audit_item["prompt_token_ids"]
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in audit_item["prompt_token_ids"]
            )
            or audit_item["prompt_token_count"]
            != len(audit_item["prompt_token_ids"])
            or audit_item["prompt_sha256"]
            != _text_sha256(audit_item["prompt"])
            or audit_item["prompt_token_ids_sha256"]
            != _token_ids_sha256(audit_item["prompt_token_ids"])
            or not isinstance(audit_item["intermediates"], list)
            or len(audit_item["intermediates"])
            != len(direct_item["intermediates"])
        ):
            raise ObservationValidationError(
                "raw tokenization audit item is invalid"
            )
        for audited, direct in zip(
            audit_item["intermediates"],
            direct_item["intermediates"],
            strict=True,
        ):
            if (
                not isinstance(audited, dict)
                or set(audited) != AUDIT_INTERMEDIATE_FIELDS
                or not isinstance(audited["text"], str)
                or not audited["text"]
                or not isinstance(audited["token_id"], int)
                or isinstance(audited["token_id"], bool)
                or audited["token_id"] < 0
                or not isinstance(audited["token_ids"], list)
                or not audited["token_ids"]
                or any(
                    not isinstance(token_id, int)
                    or isinstance(token_id, bool)
                    or token_id < 0
                    for token_id in audited["token_ids"]
                )
                or audited["variant"] not in {"leading_space", "raw"}
                or not isinstance(audited["single_token"], bool)
            ):
                raise ObservationValidationError(
                    "raw tokenization audit fields are invalid"
                )
            prompt = (
                " " + audited["text"].lstrip()
                if audited["variant"] == "leading_space"
                else audited["text"]
            )
            if (
                audited["token_id"] != audited["token_ids"][0]
                or audited["single_token"]
                is not (len(audited["token_ids"]) == 1)
                or audited["prompt_sha256"] != _text_sha256(prompt)
                or audited["token_ids_sha256"]
                != _token_ids_sha256(audited["token_ids"])
                or (
                    audited["text"],
                    audited["token_id"],
                    audited["single_token"],
                )
                != (
                    direct["text"],
                    direct["token_id"],
                    direct["single_token"],
                )
            ):
                raise ObservationValidationError(
                    "raw tokenization audit hashes or claims do not recompute"
                )


def _project_raw_records(
    records: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": record["name"],
            "prompt": audited["prompt"],
            "prompt_token_ids": audited["prompt_token_ids"],
            "prompt_token_count": audited["prompt_token_count"],
            "prompt_sha256": audited["prompt_sha256"],
            "prompt_token_ids_sha256": audited[
                "prompt_token_ids_sha256"
            ],
            "intermediates": [
                {
                    "text": intermediate["text"],
                    "token_id": intermediate["token_id"],
                    "single_token": intermediate["single_token"],
                    **{
                        method: intermediate[method]
                        for method in METHODS
                    },
                }
                for intermediate in record["intermediates"]
            ],
        }
        for record, audited in zip(records, audit, strict=True)
    ]


def _reconcile_raw(
    raw: dict[str, Any],
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if set(raw) != {
        "provenance",
        "tokenization_audit",
        "direct",
        "panel",
        "layers",
    }:
        raise ObservationValidationError(
            "raw observation has unexpected fields"
        )
    if raw["provenance"] != manifest["provenance"]:
        raise ObservationValidationError(
            "raw provenance differs from compact manifest"
        )
    if raw["layers"] != result["layers"]:
        raise ObservationValidationError(
            "raw layer scopes differ from compact result"
        )
    _validate_tokenization_audit(
        raw["tokenization_audit"],
        raw["direct"]["items"],
    )
    projected = _project_raw_records(
        raw["direct"]["items"],
        raw["tokenization_audit"],
    )
    if projected != result["per_item_best_ranks"]:
        raise ObservationValidationError(
            "raw direct records differ from compact result"
        )
    if raw["panel"]["items"] != result["panel"]:
        raise ObservationValidationError(
            "raw panel content differs from compact result"
        )
    direct_by_name = {
        entry["name"]: entry for entry in raw["direct"]["items"]
    }
    for panel_item in raw["panel"]["items"]:
        direct_item = direct_by_name.get(panel_item["name"])
        if direct_item is None or {
            key: direct_item[key]
            for key in (
                "name",
                "prompt",
                "target",
                "intermediates",
                "top_tokens",
            )
        } != {
            key: panel_item[key]
            for key in (
                "name",
                "prompt",
                "target",
                "intermediates",
                "top_tokens",
            )
        }:
            raise ObservationValidationError(
                "raw panel differs from raw direct records"
            )
    if (
        _raw_tokenization_summary(raw["tokenization_audit"])
        != result["tokenization_summary"]
    ):
        raise ObservationValidationError(
            "raw tokenization differs from compact result"
        )
    for method in METHODS:
        for scope in SCOPES:
            recomputed = _metric(projected, method, scope)
            if (
                raw["direct"]["metrics"][method][scope] != recomputed
                or result["metrics"][method][scope] != recomputed
            ):
                raise ObservationValidationError(
                    "raw metrics differ from compact result"
                )
            single_records = [
                {
                    "intermediates": [
                        intermediate
                        for intermediate in record["intermediates"]
                        if intermediate["single_token"]
                    ]
                }
                for record in projected
                if any(
                    intermediate["single_token"]
                    for intermediate in record["intermediates"]
                )
            ]
            if (
                raw["direct"]["single_token_metrics"][method][scope]
                != _metric(single_records, method, scope)
            ):
                raise ObservationValidationError(
                    "raw single-token metrics do not recompute"
                )


def _reconcile_raw_checked(
    raw: dict[str, Any],
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> None:
    try:
        _reconcile_raw(raw, manifest, result)
    except ObservationValidationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ObservationValidationError(
            "raw observation structure is invalid"
        ) from error


def _validate_failure_raw(
    raw: dict[str, Any],
    manifest: dict[str, Any],
    reason: dict[str, Any],
    result: dict[str, Any],
) -> None:
    failure = raw.get("failure")
    if (
        set(raw) != {"provenance", "failure"}
        or raw.get("provenance") != manifest["provenance"]
        or not isinstance(failure, dict)
        or set(failure) != {"reason_code", "error_type"}
        or failure["reason_code"] != reason["code"]
        or not isinstance(failure["error_type"], str)
        or not failure["error_type"]
        or bool(result["per_item_best_ranks"])
        or bool(result["panel"])
        or bool(result["layers"]["band"])
        or bool(result["layers"]["all_layers"])
        or result["tokenization_summary"]["item_count"] != 0
    ):
        raise ObservationValidationError(
            "raw failure differs from compact result"
        )
    expected_error_types = FAILURE_ERROR_TYPES.get(reason["code"])
    if expected_error_types is None:
        raise ObservationValidationError(
            "failure raw is not allowed for this observation reason"
        )
    if failure["error_type"] not in expected_error_types:
        raise ObservationValidationError(
            "raw failure error type differs from the pipeline mapping"
        )


def validate_documents(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    schema_dir: str | Path,
    raw_path: str | Path,
) -> None:
    schemas = Path(schema_dir)
    _reject_forbidden(manifest)
    _reject_forbidden(result)
    _validate_schema(
        manifest,
        schemas / "jspace_observation_manifest.schema.json",
    )
    _validate_schema(
        result,
        schemas / "jspace_observation.schema.json",
    )
    if manifest["run_id"] != result["run_id"]:
        raise ObservationValidationError("run_id differs across artifacts")
    _validate_source_provenance(
        manifest["provenance"]["source"],
        project=schemas.parent,
    )
    if result["manifest_sha256"] != document_sha256(manifest):
        raise ObservationValidationError(
            "manifest_sha256 does not match manifest"
        )
    dataset = manifest["dataset"]
    if (
        dataset["sha256"] != DATASET_SHA256
        or dataset["item_count"] != EXPECTED_ITEM_COUNT
    ):
        raise ObservationValidationError("dataset claim is not pinned")
    raw = _validate_raw(
        manifest,
        raw_path=Path(raw_path),
    )
    _reject_forbidden(raw, "$raw", path=("raw",))
    complete = _validate_counts_and_claims(result)
    provenance_complete = _provenance_complete(manifest["provenance"])
    expected_status = (
        "success" if complete and provenance_complete else "inconclusive"
    )
    if manifest["status"] == "error":
        raise ObservationValidationError(
            "unexpected errors must surface instead of validating"
        )
    if manifest["status"] != result["status"]:
        raise ObservationValidationError("status differs across artifacts")
    if result["status"] != expected_status:
        raise ObservationValidationError("status claim does not recompute")
    if manifest["reason"] != result["reason"]:
        raise ObservationValidationError("reason differs across artifacts")
    reason = result["reason"]
    if reason is not None and reason["message"] != REASON_MESSAGES[reason["code"]]:
        raise ObservationValidationError("reason is not from the allowlist")
    if result["status"] == "success":
        if reason is not None or manifest["raw_artifact_sha256"] is None:
            raise ObservationValidationError(
                "successful observation needs raw hash and no reason"
            )
        try:
            if (
                len(raw["direct"]["items"]) != EXPECTED_ITEM_COUNT
                or len(raw["tokenization_audit"]) != EXPECTED_ITEM_COUNT
                or [item["name"] for item in raw["panel"]["items"]]
                != list(PANEL_NAMES)
            ):
                raise ObservationValidationError(
                    "raw observation is incomplete"
                )
            _reconcile_raw_checked(raw, manifest, result)
        except (IndexError, KeyError, TypeError):
            raise ObservationValidationError(
                "raw observation is incomplete"
            ) from None
    elif reason is None:
        raise ObservationValidationError(
            "inconclusive observation requires a reason"
        )
    elif reason["code"] == "incomplete_provenance":
        if not complete or "failure" in raw:
            raise ObservationValidationError(
                "incomplete provenance requires measurement raw data"
            )
        _reconcile_raw_checked(raw, manifest, result)
    elif reason["code"] in FAILURE_ERROR_TYPES:
        _validate_failure_raw(raw, manifest, reason, result)
    else:
        raise ObservationValidationError(
            "inconclusive reason is not produced by the observation pipeline"
        )


def validate_result_files(
    results_dir: Path,
    schema_dir: Path,
    raw_path: Path,
) -> None:
    manifest = _read_json(results_dir / "jspace_observation_manifest.json")
    result = _read_json(
        results_dir / "jspace_observation.json",
        max_bytes=MAX_OBSERVATION_JSON_BYTES,
    )
    validate_documents(
        manifest,
        result,
        schema_dir=schema_dir,
        raw_path=raw_path,
    )
    _validate_figures(
        results_dir,
        success=result["status"] == "success",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--schema-dir", type=Path, default=PROJECT / "schemas")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT / "artifacts" / "jspace_observation_raw.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_result_files(args.results_dir, args.schema_dir, args.raw_path)
    except (ObservationValidationError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    print("PASS: observation schemas, hashes, counts, and claims are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
