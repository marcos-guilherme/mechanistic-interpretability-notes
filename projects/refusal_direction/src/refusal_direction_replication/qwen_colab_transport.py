"""Safe Qwen-only Colab result packing and local extraction."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from .qwen_dose import metadata_contains_leak, sha256_file
from .qwen_code_eval import compact_contains_leak
from .qwen_jlens import (
    jlens_metadata_contains_leak,
    load_fit_checkpoint,
    validate_checkpoint_structure,
    V2_FIT_COUNT_CANDIDATES,
)

QWEN_RESULTS_DIR = "results/qwen25_coder_7b"
QWEN_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b"
REMOTE_ARCHIVE = Path("/content/qwen_results.tar.gz")

QWEN_FETCH_ALLOWLIST = frozenset(
    {
        f"{QWEN_ARTIFACTS_DIR}/status.json",
        f"{QWEN_ARTIFACTS_DIR}/events.jsonl",
        f"{QWEN_ARTIFACTS_DIR}/candidates.pt",
        f"{QWEN_ARTIFACTS_DIR}/selected_direction.pt",
        f"{QWEN_ARTIFACTS_DIR}/jlens_fit_checkpoint.pt",
        f"{QWEN_ARTIFACTS_DIR}/qwen25_coder_7b_jacobian_lens.pt",
    }
)

DOSE_RESULTS_DIR = "results/qwen25_coder_7b_dose"
DOSE_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_dose"
DOSE_REMOTE_ARCHIVE = Path("/content/qwen_dose_results.tar.gz")

DOSE_FETCH_ALLOWLIST = frozenset(
    {
        f"{DOSE_ARTIFACTS_DIR}/status.json",
        f"{DOSE_ARTIFACTS_DIR}/events.jsonl",
        f"{DOSE_RESULTS_DIR}/00_dose_prepare.json",
        f"{DOSE_RESULTS_DIR}/01_dose_baseline.json",
        f"{DOSE_RESULTS_DIR}/02_dose_selection.json",
        f"{DOSE_RESULTS_DIR}/03_dose_confirmation.json",
        f"{DOSE_RESULTS_DIR}/dose_followup_evaluation.json",
        f"{DOSE_RESULTS_DIR}/execution_manifest.json",
    }
)
DOSE_TENSOR_ALLOWLIST = frozenset()
DOSE_SOURCE_DIRECTION_RELATIVE = Path("artifacts") / "qwen25_coder_7b" / "selected_direction.pt"
REMOTE_DOSE_SOURCE_DIRECTION_STAGING = "/content/dose_source_direction.pt"
MAX_DOSE_SOURCE_DIRECTION_BYTES = 16 * 1024 * 1024

RESULT_JSON_PATTERN = re.compile(
    r"^results/qwen25_coder_7b/[^/]+\.json$"
)
DOSE_CHECKPOINT_JSON_PATTERN = re.compile(
    r"^results/qwen25_coder_7b_dose/checkpoint_coeff_[0-9]+(?:\.[0-9]+)?\.json$"
)

MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES = 5 * 1024**3
MAX_JLENS_JSON_BYTES = 16 * 1024 * 1024
MAX_JLENS_JSONL_BYTES = 32 * 1024 * 1024
JLENS_CHECKPOINT_RELATIVE = Path("artifacts") / "qwen25_coder_7b_jlens" / "jlens_fit_checkpoint.pt"
JLENS_STATUS_RELATIVE = Path("artifacts") / "qwen25_coder_7b_jlens" / "status.json"
JLENS_EVENTS_RELATIVE = Path("artifacts") / "qwen25_coder_7b_jlens" / "events.jsonl"
JLENS_RESUME_MARKER_RELATIVE = (
    Path("artifacts") / "qwen25_coder_7b_jlens" / "jlens_resume_bundle.json"
)
REMOTE_JLENS_CHECKPOINT_STAGING = "/content/jlens_fit_checkpoint.pt"
REMOTE_JLENS_STATUS_STAGING = "/content/jlens_status.json"
REMOTE_JLENS_EVENTS_STAGING = "/content/jlens_events.jsonl"
MAX_JLENS_CHECKPOINT_BYTES = MAX_MEMBER_BYTES
COMPACT_SCHEMA_FIELDS = (
    "schema_version",
    "experiment",
    "config",
    "results",
    "provenance",
)
EVALUATION_SCHEMA_FIELDS = (
    "status",
    "run_id",
    "implementation_commit",
    "source_artifact_hashes",
    "source_pilot_run_id",
)
JLENS_RESULTS_DIR = "results/qwen25_coder_7b_jlens"
JLENS_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_jlens"
JLENS_REMOTE_ARCHIVE = Path("/content/qwen_jlens_results.tar.gz")
JLENS_FETCH_ALLOWLIST = frozenset(
    {
        f"{JLENS_RESULTS_DIR}/00_jlens_prepare.json",
        f"{JLENS_RESULTS_DIR}/01_jlens_probe.json",
        f"{JLENS_RESULTS_DIR}/02_jlens_fit.json",
        f"{JLENS_RESULTS_DIR}/03_jlens_holdout.json",
        f"{JLENS_RESULTS_DIR}/jlens_evaluation.json",
        f"{JLENS_RESULTS_DIR}/execution_manifest.json",
        f"{JLENS_ARTIFACTS_DIR}/status.json",
        f"{JLENS_ARTIFACTS_DIR}/events.jsonl",
        f"{JLENS_ARTIFACTS_DIR}/jlens_fit_checkpoint.pt",
        f"{JLENS_ARTIFACTS_DIR}/qwen25_coder_7b_jacobian_lens.pt",
        f"{JLENS_ARTIFACTS_DIR}/jlens_resume_bundle.json",
    }
)
JLENS_V2_RESULTS_DIR = "results/qwen25_coder_7b_jlens_v2"
JLENS_V2_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_jlens_v2"
JLENS_V2_REMOTE_ARCHIVE = Path("/content/qwen_jlens_v2_results.tar.gz")
REMOTE_JLENS_V2_CHECKPOINT_STAGING = "/content/jlens_v2_fit_checkpoint.pt"
REMOTE_JLENS_V2_STATUS_STAGING = "/content/jlens_v2_status.json"
REMOTE_JLENS_V2_EVENTS_STAGING = "/content/jlens_v2_events.jsonl"
JLENS_V2_FINAL_LENS_NAME = "qwen25_coder_7b_jacobian_lens_v2.pt"
JLENS_V2_CHECKPOINT_RELATIVE = (
    Path("artifacts") / "qwen25_coder_7b_jlens_v2" / "jlens_fit_checkpoint.pt"
)
JLENS_V2_STATUS_RELATIVE = Path("artifacts") / "qwen25_coder_7b_jlens_v2" / "status.json"
JLENS_V2_EVENTS_RELATIVE = Path("artifacts") / "qwen25_coder_7b_jlens_v2" / "events.jsonl"
JLENS_V2_RESUME_MARKER_RELATIVE = (
    Path("artifacts") / "qwen25_coder_7b_jlens_v2" / "jlens_resume_bundle.json"
)
JLENS_V2_FETCH_ALLOWLIST = frozenset(
    {
        *(
            f"{JLENS_V2_RESULTS_DIR}/{name}"
            for name in (
                "00_jlens_prepare.json",
                "01_jlens_probe.json",
                "02_jlens_fit.json",
                "03_jlens_holdout.json",
                "04_jlens_compare.json",
                "jlens_evaluation.json",
                "execution_manifest.json",
            )
        ),
        *(
            f"{JLENS_V2_ARTIFACTS_DIR}/{name}"
            for name in (
                "status.json",
                "events.jsonl",
                "jlens_fit_checkpoint.pt",
                JLENS_V2_FINAL_LENS_NAME,
                "jlens_resume_bundle.json",
            )
        ),
    }
)
JLENS_EVALUATION_SCHEMA_FIELDS = (
    "status",
    "run_id",
    "implementation_commit",
)
CODE_EVAL_RESULTS_DIR = "results/qwen25_coder_7b_code_eval"
CODE_EVAL_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_code_eval"
CODE_EVAL_REMOTE_ARCHIVE = Path("/content/qwen_code_eval_results.tar.gz")
CODE_EVAL_DIRECTION_RELATIVE = Path("artifacts/qwen25_coder_7b/selected_direction.pt")
CODE_EVAL_LENS_RELATIVE = (
    Path("artifacts")
    / "qwen25_coder_7b_jlens"
    / "qwen25_coder_7b_jacobian_lens.pt"
)
REMOTE_CODE_EVAL_DIRECTION_STAGING = "/content/code_eval_source_direction.pt"
REMOTE_CODE_EVAL_LENS_STAGING = "/content/code_eval_source_lens.pt"
LAB_PROFILE = "qwen25_coder_7b_lab.json"
LAB_ARTIFACTS_DIR = "artifacts/qwen25_coder_7b_lab"
LAB_DIRECTION_RELATIVE = CODE_EVAL_DIRECTION_RELATIVE
LAB_LENS_RELATIVE = CODE_EVAL_LENS_RELATIVE
REMOTE_LAB_DIRECTION_STAGING = "/content/lab_source_direction.pt"
REMOTE_LAB_LENS_STAGING = "/content/lab_source_lens.pt"
REMOTE_LAB_LENS_MANIFEST = "/content/lab_source_lens.pt.manifest.json"
REMOTE_LAB_AUTH_FILE = "/content/qwen_lab_auth.json"
ARTIFACT_CHUNK_BYTES = 64 * 1024**2
LAB_LENS_CHUNK_BYTES = 16 * 1024**2
CHUNK_MANIFEST_VERSION = 1
MAX_LAB_DIRECTION_BYTES = MAX_DOSE_SOURCE_DIRECTION_BYTES
MAX_LAB_LENS_BYTES = 800 * 1024**2
LAB_STATUS_FIELDS = frozenset(
    {
        "profile",
        "state",
        "gpu",
        "share_ready",
        "public_url",
        "pid",
        "direction_sha256",
        "lens_sha256",
    }
)
LAB_STATUS_REQUIRED_FIELDS = ("profile", "state", "gpu", "share_ready")
LAB_STATUS_STATES = frozenset({"running", "stopped", "error"})
LAB_STATUS_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CODE_EVAL_COMPACT_NAMES = frozenset(
    {
        "00_prepare.json",
        "01_generate.json",
        "02_evaluate.json",
        "03_readout.json",
        "04_finalize.json",
        "code_eval_evaluation.json",
        "execution_manifest.json",
    }
)
CODE_EVAL_DATASETS = r"(?:mitre|mitre_frr|humaneval_plus)"
CODE_EVAL_CONDITIONS = r"(?:baseline|ablation|addition|random_control)"
CODE_EVAL_RAW_PATTERNS = (
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/datasets/{CODE_EVAL_DATASETS}\.jsonl$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/generations/"
        rf"{CODE_EVAL_DATASETS}__{CODE_EVAL_CONDITIONS}\.jsonl$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/logits/"
        rf"{CODE_EVAL_DATASETS}__{CODE_EVAL_CONDITIONS}\.pt$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/evalplus/"
        rf"(?:selected_problems\.jsonl|samples__{CODE_EVAL_CONDITIONS}\.jsonl|"
        rf"(?:results|identity)__{CODE_EVAL_CONDITIONS}\.json)$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/readout/"
        rf"{CODE_EVAL_DATASETS}__{CODE_EVAL_CONDITIONS}\.jsonl$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/"
        r"(?:manual_review_blind\.jsonl|manual_review_key\.json|"
        r"refusal_lexicon\.json)$"
    ),
)
MAX_CODE_EVAL_JSON_BYTES = 64 * 1024**2
MAX_CODE_EVAL_JSONL_BYTES = 512 * 1024**2
MAX_CODE_EVAL_TOTAL_BYTES = 4 * 1024**3


def normalize_member_name(name: str) -> str:
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/"):
        raise ValueError(f"unsafe archive path: {name}")
    if ".." in PurePosixPath(cleaned).parts:
        raise ValueError(f"unsafe archive path: {name}")
    trimmed = cleaned
    while trimmed.startswith("./"):
        trimmed = trimmed[2:]
    if not trimmed:
        raise ValueError(f"unsafe archive path: {name}")
    return str(PurePosixPath(trimmed))


def is_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    if not normalized.startswith(
        ("results/qwen25_coder_7b/", "artifacts/qwen25_coder_7b/")
    ):
        return True
    return False


def is_allowed_fetch_member(name: str) -> bool:
    normalized = normalize_member_name(name)
    if is_unsafe_member_name(normalized):
        return False
    if normalized in QWEN_FETCH_ALLOWLIST:
        return True
    return bool(RESULT_JSON_PATTERN.fullmatch(normalized))


def is_dose_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    if not normalized.startswith((f"{DOSE_RESULTS_DIR}/", f"{DOSE_ARTIFACTS_DIR}/")):
        return True
    return False


def path_is_link_or_junction(path: Path) -> bool:
    if not os.path.lexists(path):
        return False
    try:
        if path.is_symlink():
            return True
    except OSError:
        pass
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def dose_source_direction_path(project: Path) -> Path:
    return project / DOSE_SOURCE_DIRECTION_RELATIVE


def _validate_regular_direction_file(path: Path, *, origin: str) -> None:
    if not os.path.lexists(path):
        raise FileNotFoundError(
            f"dose profile requires artifacts/qwen25_coder_7b/selected_direction.pt "
            f"({origin} missing: {path})"
        )
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(
            f"dose source direction must be a regular file ({origin}: {path})"
        )
    size = path.stat().st_size
    if size <= 0 or size > MAX_DOSE_SOURCE_DIRECTION_BYTES:
        raise ValueError(
            f"dose source direction size {size} exceeds "
            f"{MAX_DOSE_SOURCE_DIRECTION_BYTES} bytes ({origin}: {path})"
        )


def _assert_confined_direction_destination(project: Path, destination: Path) -> Path:
    project_root = project.resolve()
    expected = (project_root / DOSE_SOURCE_DIRECTION_RELATIVE).resolve()
    current = destination
    for _ in range(len(destination.parts) + 1):
        if path_is_link_or_junction(current):
            raise ValueError(
                "dose source direction destination escaped "
                "artifacts/qwen25_coder_7b/selected_direction.pt"
            )
        if current == project or current.parent == current:
            break
        current = current.parent
    resolved = destination.resolve()
    if resolved != expected or resolved.name != "selected_direction.pt":
        raise ValueError(
            "dose source direction destination must remain "
            "artifacts/qwen25_coder_7b/selected_direction.pt"
        )
    try:
        resolved.relative_to(project_root / "artifacts" / "qwen25_coder_7b")
    except ValueError as error:
        raise ValueError(
            "dose source direction destination escaped "
            "artifacts/qwen25_coder_7b/selected_direction.pt"
        ) from error
    return destination


def require_local_dose_source_direction(project: Path) -> Path:
    path = dose_source_direction_path(project)
    _validate_regular_direction_file(path, origin="local")
    return _assert_confined_direction_destination(project, path)


def atomic_install_dose_source_direction(
    *,
    staged: Path,
    project: Path,
) -> Path:
    _validate_regular_direction_file(staged, origin="staging")
    destination = project / DOSE_SOURCE_DIRECTION_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined_direction_destination(project, destination)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".selected_direction.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(staged.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        if tmp_path.stat().st_size > MAX_DOSE_SOURCE_DIRECTION_BYTES:
            raise ValueError("dose source direction size exceeds limit after copy")
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    _validate_regular_direction_file(destination, origin="destination")
    return _assert_confined_direction_destination(project, destination)


def _validate_code_eval_source(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> None:
    if not os.path.lexists(path):
        raise FileNotFoundError(f"code-eval {label} source missing: {path}")
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(f"code-eval {label} source must be a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(
            f"code-eval {label} source size {size} exceeds {max_bytes} bytes"
        )


def _assert_code_eval_source_destination(
    project: Path,
    destination: Path,
    relative: Path,
) -> Path:
    expected = (project.resolve() / relative).resolve()
    current = destination
    while current != project and current.parent != current:
        if path_is_link_or_junction(current):
            raise ValueError("code-eval source destination is a link or junction")
        current = current.parent
    if destination.resolve() != expected:
        raise ValueError("code-eval source destination escaped its fixed path")
    return destination


def require_local_code_eval_sources(project: Path) -> tuple[Path, Path]:
    direction = project / CODE_EVAL_DIRECTION_RELATIVE
    lens = project / CODE_EVAL_LENS_RELATIVE
    _validate_code_eval_source(
        direction,
        label="direction",
        max_bytes=MAX_DOSE_SOURCE_DIRECTION_BYTES,
    )
    _validate_code_eval_source(
        lens,
        label="lens",
        max_bytes=MAX_MEMBER_BYTES,
    )
    return (
        _assert_code_eval_source_destination(
            project, direction, CODE_EVAL_DIRECTION_RELATIVE
        ),
        _assert_code_eval_source_destination(
            project, lens, CODE_EVAL_LENS_RELATIVE
        ),
    )


def atomic_install_code_eval_sources(
    *,
    staged_direction: Path,
    staged_lens: Path,
    project: Path,
) -> tuple[Path, Path]:
    """Install the frozen direction and lens as one rollback-safe pair."""

    _validate_code_eval_source(
        staged_direction,
        label="direction staging",
        max_bytes=MAX_DOSE_SOURCE_DIRECTION_BYTES,
    )
    _validate_code_eval_source(
        staged_lens,
        label="lens staging",
        max_bytes=MAX_MEMBER_BYTES,
    )
    destinations = (
        project / CODE_EVAL_DIRECTION_RELATIVE,
        project / CODE_EVAL_LENS_RELATIVE,
    )
    relatives = (CODE_EVAL_DIRECTION_RELATIVE, CODE_EVAL_LENS_RELATIVE)
    for destination, relative in zip(destinations, relatives):
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_code_eval_source_destination(project, destination, relative)
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".code_eval_sources_backup.",
            dir=project,
        )
    )
    records: list[InstallRecord] = []
    try:
        for destination, relative in zip(destinations, relatives):
            had_original = destination.exists()
            backup = backup_root / relative if had_original else None
            if backup is not None:
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            records.append(
                InstallRecord(
                    target=destination,
                    backup=backup,
                    had_original=had_original,
                )
            )
        _stream_install_regular_file(
            staged=staged_direction,
            destination=destinations[0],
            max_bytes=MAX_DOSE_SOURCE_DIRECTION_BYTES,
            prefix=".code_eval_direction.",
        )
        _stream_install_regular_file(
            staged=staged_lens,
            destination=destinations[1],
            max_bytes=MAX_MEMBER_BYTES,
            prefix=".code_eval_lens.",
        )
        return require_local_code_eval_sources(project)
    except Exception:
        _rollback_install_records(records)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def split_artifact_file(
    source: Path,
    staging: Path,
    *,
    chunk_bytes: int = ARTIFACT_CHUNK_BYTES,
    part_prefix: str = "lens.part",
) -> list[Path]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob(f"{part_prefix}*"):
        old.unlink()
    parts: list[Path] = []
    with source.open("rb") as incoming:
        index = 0
        while payload := incoming.read(chunk_bytes):
            part = staging / f"{part_prefix}{index:03d}"
            part.write_bytes(payload)
            parts.append(part)
            index += 1
    if not parts:
        raise ValueError("artifact is empty")
    return parts


def write_artifact_chunk_manifest(
    source: Path,
    parts: Sequence[Path],
    destination: Path,
    *,
    part_names: Sequence[str] | None = None,
) -> Path:
    names = list(part_names) if part_names is not None else [part.name for part in parts]
    if len(names) != len(parts):
        raise ValueError("part names do not match parts")
    payload = {
        "version": CHUNK_MANIFEST_VERSION,
        "total_size": source.stat().st_size,
        "sha256": sha256_file(source),
        "part_count": len(parts),
        "parts": [
            {
                "name": name,
                "size": part.stat().st_size,
                "sha256": sha256_file(part),
            }
            for part, name in zip(parts, names)
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _discover_chunk_parts(destination: Path) -> list[Path]:
    return sorted(destination.parent.glob(destination.name + ".part*"))


def _validate_parts_against_manifest(
    parts: Sequence[Path],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("version") != CHUNK_MANIFEST_VERSION:
        raise ValueError("chunk manifest version is unsupported")
    expected_entries = list(manifest.get("parts") or [])
    expected_count = int(manifest.get("part_count") or 0)
    if expected_count != len(expected_entries):
        raise ValueError("chunk manifest part_count does not match parts")
    actual_names = [part.name for part in parts]
    expected_names = [str(entry.get("name") or "") for entry in expected_entries]
    extra = sorted(set(actual_names) - set(expected_names))
    missing = [name for name in expected_names if name not in set(actual_names)]
    if extra:
        raise ValueError(f"extra chunk parts: {', '.join(extra)}")
    if missing:
        raise ValueError(f"missing chunk parts: {', '.join(missing)}")
    if actual_names != expected_names:
        raise ValueError("chunk parts do not match manifest order")
    for part, entry in zip(parts, expected_entries):
        expected_size = int(entry.get("size") or 0)
        if not part.is_file() or part.stat().st_size != expected_size:
            raise ValueError(f"altered chunk part size: {part.name}")
        actual_hash = sha256_file(part)
        if actual_hash != entry.get("sha256"):
            raise ValueError(f"altered chunk part sha256: {part.name}")


def assemble_chunked_file(
    destination: Path,
    *,
    manifest_path: Path | None = None,
    require_manifest: bool = False,
) -> Path:
    parts = _discover_chunk_parts(destination)
    resolved_manifest = manifest_path
    if require_manifest and resolved_manifest is None:
        resolved_manifest = destination.with_name(destination.name + ".manifest.json")
    if require_manifest and (
        resolved_manifest is None or not Path(resolved_manifest).is_file()
    ):
        raise FileNotFoundError("chunk manifest is required")
    manifest: Mapping[str, Any] | None = None
    if resolved_manifest is not None and Path(resolved_manifest).is_file():
        manifest = json.loads(Path(resolved_manifest).read_text(encoding="utf-8"))
        if not parts:
            raise FileNotFoundError("lab lens staging parts are missing")
        _validate_parts_against_manifest(parts, manifest)
    elif not parts:
        if destination.is_file():
            return destination
        raise FileNotFoundError("lab lens staging parts are missing")
    else:
        expected = [
            destination.name + f".part{index:03d}" for index in range(len(parts))
        ]
        if [part.name for part in parts] != expected:
            raise ValueError("lab lens staging parts are not contiguous")
        for part in parts:
            if not part.is_file() or part.stat().st_size <= 0:
                raise ValueError(f"invalid lab lens part: {part.name}")
    temporary = destination.with_name(destination.name + ".assembling")
    try:
        with temporary.open("wb") as output:
            for part in parts:
                if not part.is_file() or part.stat().st_size <= 0:
                    raise ValueError(f"invalid lab lens part: {part.name}")
                with part.open("rb") as incoming:
                    while chunk := incoming.read(8 * 1024**2):
                        output.write(chunk)
        if manifest is not None:
            if temporary.stat().st_size != int(manifest["total_size"]):
                raise ValueError("assembled size does not match manifest")
            if sha256_file(temporary) != manifest["sha256"]:
                raise ValueError("assembled sha256 does not match manifest")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
    for part in parts:
        part.unlink()
    if resolved_manifest is not None:
        try:
            Path(resolved_manifest).unlink()
        except FileNotFoundError:
            pass
    return destination


def load_lab_profile_config(project: Path) -> dict[str, Any]:
    path = project / "configs" / LAB_PROFILE
    return json.loads(path.read_text(encoding="utf-8"))


def _lab_expected_hashes(
    project: Path,
    *,
    expected_direction_sha256: str | None,
    expected_lens_sha256: str | None,
) -> tuple[str, str]:
    if expected_direction_sha256 and expected_lens_sha256:
        return expected_direction_sha256, expected_lens_sha256
    config = load_lab_profile_config(project)
    return (
        expected_direction_sha256 or str(config["direction_sha256"]),
        expected_lens_sha256 or str(config["lens_sha256"]),
    )


def _validate_lab_source(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> None:
    if not os.path.lexists(path):
        raise FileNotFoundError(f"lab {label} source missing: {path}")
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(f"lab {label} source must be a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(
            f"lab {label} source size {size} exceeds {max_bytes} bytes"
        )
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"lab {label} sha256 {actual!r} != expected {expected_sha256!r}"
            )


def _assert_lab_source_destination(
    project: Path,
    destination: Path,
    relative: Path,
) -> Path:
    expected = (project.resolve() / relative).resolve()
    current = destination
    while current != project and current.parent != current:
        if path_is_link_or_junction(current):
            raise ValueError("lab source destination is a link or junction")
        current = current.parent
    if destination.resolve() != expected:
        raise ValueError("lab source destination escaped its fixed path")
    return destination


def require_local_lab_sources(
    project: Path,
    *,
    expected_direction_sha256: str | None = None,
    expected_lens_sha256: str | None = None,
) -> tuple[Path, Path]:
    direction_hash, lens_hash = _lab_expected_hashes(
        project,
        expected_direction_sha256=expected_direction_sha256,
        expected_lens_sha256=expected_lens_sha256,
    )
    direction = project / LAB_DIRECTION_RELATIVE
    lens = project / LAB_LENS_RELATIVE
    _validate_lab_source(
        direction,
        label="direction",
        max_bytes=MAX_LAB_DIRECTION_BYTES,
        expected_sha256=direction_hash,
    )
    _validate_lab_source(
        lens,
        label="lens",
        max_bytes=MAX_LAB_LENS_BYTES,
        expected_sha256=lens_hash,
    )
    return (
        _assert_lab_source_destination(project, direction, LAB_DIRECTION_RELATIVE),
        _assert_lab_source_destination(project, lens, LAB_LENS_RELATIVE),
    )


def atomic_install_lab_sources(
    *,
    staged_direction: Path,
    staged_lens: Path,
    project: Path,
    expected_direction_sha256: str | None = None,
    expected_lens_sha256: str | None = None,
) -> tuple[Path, Path]:
    direction_hash, lens_hash = _lab_expected_hashes(
        project,
        expected_direction_sha256=expected_direction_sha256,
        expected_lens_sha256=expected_lens_sha256,
    )
    _validate_lab_source(
        staged_direction,
        label="direction staging",
        max_bytes=MAX_LAB_DIRECTION_BYTES,
        expected_sha256=direction_hash,
    )
    _validate_lab_source(
        staged_lens,
        label="lens staging",
        max_bytes=MAX_LAB_LENS_BYTES,
        expected_sha256=lens_hash,
    )
    destinations = (
        project / LAB_DIRECTION_RELATIVE,
        project / LAB_LENS_RELATIVE,
    )
    relatives = (LAB_DIRECTION_RELATIVE, LAB_LENS_RELATIVE)
    for destination, relative in zip(destinations, relatives):
        destination.parent.mkdir(parents=True, exist_ok=True)
        _assert_lab_source_destination(project, destination, relative)
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".lab_sources_backup.",
            dir=project,
        )
    )
    records: list[InstallRecord] = []
    try:
        for destination, relative in zip(destinations, relatives):
            had_original = destination.exists()
            backup = backup_root / relative if had_original else None
            if backup is not None:
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            records.append(
                InstallRecord(
                    target=destination,
                    backup=backup,
                    had_original=had_original,
                )
            )
        _stream_install_regular_file(
            staged=staged_direction,
            destination=destinations[0],
            max_bytes=MAX_LAB_DIRECTION_BYTES,
            prefix=".lab_direction.",
        )
        _stream_install_regular_file(
            staged=staged_lens,
            destination=destinations[1],
            max_bytes=MAX_LAB_LENS_BYTES,
            prefix=".lab_lens.",
        )
        return require_local_lab_sources(
            project,
            expected_direction_sha256=direction_hash,
            expected_lens_sha256=lens_hash,
        )
    except Exception:
        _rollback_install_records(records)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def write_lab_auth_file(
    path: Path,
    *,
    username: str,
    password: str,
) -> Path:
    user = str(username).strip()
    secret = str(password).strip()
    if not user or not secret:
        raise ValueError("auth requires both username and password")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"username": user, "password": secret}, indent=None)
    encoded = payload.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if path.exists() or os.path.lexists(path):
        if path_is_link_or_junction(path):
            raise ValueError("auth file must be a regular file")
        path.unlink()
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def consume_lab_auth_file(path: Path) -> tuple[str, str]:
    try:
        if not os.path.lexists(path):
            raise FileNotFoundError(f"auth file missing: {path}")
        if path_is_link_or_junction(path) or not path.is_file():
            raise ValueError("auth file must be a regular file")
        os.chmod(path, 0o600)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("auth file must contain a JSON object")
        user = str(payload.get("username") or payload.get("user") or "").strip()
        password = str(payload.get("password") or "").strip()
        if not user or not password:
            raise ValueError("auth requires both username and password")
        return user, password
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def validate_lab_operational_status(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("lab status must be a JSON object")
    extra = [key for key in payload if key not in LAB_STATUS_FIELDS]
    if extra:
        raise ValueError(f"lab status has unknown field: {extra[0]}")
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            raise ValueError("lab status must not contain nested or list values")
    missing = [field for field in LAB_STATUS_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"lab status missing field: {missing[0]}")
    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("lab status profile is invalid")
    state = payload.get("state")
    if state not in LAB_STATUS_STATES:
        raise ValueError("lab status state is invalid")
    gpu = payload.get("gpu")
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("lab status gpu is invalid")
    share_ready = payload.get("share_ready")
    if not isinstance(share_ready, bool):
        raise ValueError("lab status share_ready is invalid")
    if "public_url" in payload:
        url = payload["public_url"]
        if not isinstance(url, str) or extract_lab_public_url(url) != url:
            raise ValueError("lab status public_url is not a Gradio URL")
    if "pid" in payload:
        pid = payload["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("lab status pid is invalid")
    for hash_key in ("direction_sha256", "lens_sha256"):
        if hash_key not in payload:
            continue
        digest = payload[hash_key]
        if not isinstance(digest, str) or LAB_STATUS_SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"lab status {hash_key} is invalid")
    return {key: payload[key] for key in payload}


def extract_lab_public_url(text: str) -> str | None:
    match = re.search(
        r"https://[A-Za-z0-9._-]+\.gradio\.(?:live|space)(?:/[^\s]*)?",
        text,
    )
    if match is None:
        return None
    return match.group(0).rstrip(").,")


def jlens_checkpoint_path(project: Path) -> Path:
    return project / JLENS_CHECKPOINT_RELATIVE


def _validate_regular_jlens_checkpoint(path: Path, *, origin: str) -> None:
    if not os.path.lexists(path):
        raise FileNotFoundError(
            f"jlens checkpoint missing ({origin}: {path})"
        )
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(
            f"jlens checkpoint must be a regular file ({origin}: {path})"
        )
    size = path.stat().st_size
    if size <= 0 or size > MAX_JLENS_CHECKPOINT_BYTES:
        raise ValueError(
            f"jlens checkpoint size {size} exceeds "
            f"{MAX_JLENS_CHECKPOINT_BYTES} bytes ({origin}: {path})"
        )


def _assert_confined_checkpoint(
    project: Path, destination: Path, relative: Path
) -> Path:
    shown = relative.as_posix()
    project_root = project.resolve()
    expected = (project_root / relative).resolve()
    current = destination
    for _ in range(len(destination.parts) + 1):
        if path_is_link_or_junction(current):
            raise ValueError(f"jlens checkpoint destination escaped {shown}")
        if current == project or current.parent == current:
            break
        current = current.parent
    resolved = destination.resolve()
    if resolved != expected or resolved.name != "jlens_fit_checkpoint.pt":
        raise ValueError(f"jlens checkpoint destination must remain {shown}")
    try:
        resolved.relative_to(project_root / relative.parent)
    except ValueError as error:
        raise ValueError(f"jlens checkpoint destination escaped {shown}") from error
    return destination


def _assert_confined_jlens_checkpoint(project: Path, destination: Path) -> Path:
    return _assert_confined_checkpoint(project, destination, JLENS_CHECKPOINT_RELATIVE)


def validate_jlens_v2_checkpoint_payload(payload: Mapping[str, Any]) -> int:
    """Validate a v2 checkpoint against its own stored selected fit count."""

    if "fit_count" not in payload:
        raise ValueError("v2 checkpoint is missing fit_count")
    stored_count = payload["fit_count"]
    if type(stored_count) is not int:
        raise ValueError("v2 checkpoint fit_count is not an integer")
    if stored_count not in V2_FIT_COUNT_CANDIDATES:
        raise ValueError(
            f"v2 checkpoint fit_count {stored_count} is not one of "
            f"{V2_FIT_COUNT_CANDIDATES}"
        )
    validate_checkpoint_structure(payload, fit_count=stored_count)
    return stored_count


@dataclass(frozen=True)
class _JLensTransportProfile:
    checkpoint: Path
    status: Path
    events: Path
    marker: Path
    validate_payload: Callable[[Mapping[str, Any]], Any]
    v2: bool


_JLENS_V1_TRANSPORT = _JLensTransportProfile(
    checkpoint=JLENS_CHECKPOINT_RELATIVE,
    status=JLENS_STATUS_RELATIVE,
    events=JLENS_EVENTS_RELATIVE,
    marker=JLENS_RESUME_MARKER_RELATIVE,
    validate_payload=validate_checkpoint_structure,
    v2=False,
)
_JLENS_V2_TRANSPORT = _JLensTransportProfile(
    checkpoint=JLENS_V2_CHECKPOINT_RELATIVE,
    status=JLENS_V2_STATUS_RELATIVE,
    events=JLENS_V2_EVENTS_RELATIVE,
    marker=JLENS_V2_RESUME_MARKER_RELATIVE,
    validate_payload=validate_jlens_v2_checkpoint_payload,
    v2=True,
)


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _optional_local_checkpoint(
    profile: _JLensTransportProfile, project: Path
) -> Path | None:
    path = project / profile.checkpoint
    if not os.path.lexists(path):
        return None
    _validate_regular_jlens_checkpoint(path, origin="local")
    profile.validate_payload(load_fit_checkpoint(path))
    return _assert_confined_checkpoint(project, path, profile.checkpoint)


def optional_local_jlens_checkpoint(project: Path) -> Path | None:
    return _optional_local_checkpoint(_JLENS_V1_TRANSPORT, project)


def _copy_validate_replace_checkpoint(
    profile: _JLensTransportProfile,
    *,
    staged: Path,
    destination: Path,
    prefix: str,
) -> None:
    """Copy, fsync, validate that exact temporary file, then ``os.replace``.

    The destination is untouched on any failure and the temporary is removed.
    """

    fd, tmp_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with staged.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_regular_jlens_checkpoint(tmp_path, origin="temporary")
        profile.validate_payload(load_fit_checkpoint(tmp_path))
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_install_checkpoint(
    profile: _JLensTransportProfile,
    *,
    staged: Path,
    project: Path,
    missing_ok: bool = False,
) -> Path | None:
    if not os.path.lexists(staged):
        if missing_ok:
            return None
        raise FileNotFoundError(f"jlens checkpoint missing (staging: {staged})")
    _validate_regular_jlens_checkpoint(staged, origin="staging")
    destination = project / profile.checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined_checkpoint(project, destination, profile.checkpoint)
    # Only the exact temporary copy is loaded and validated (no separate staged
    # load), so what is validated is what gets installed.
    _copy_validate_replace_checkpoint(
        profile,
        staged=staged,
        destination=destination,
        prefix=".jlens_fit_checkpoint.",
    )
    _validate_regular_jlens_checkpoint(destination, origin="destination")
    return _assert_confined_checkpoint(project, destination, profile.checkpoint)


def atomic_install_jlens_checkpoint(
    *,
    staged: Path,
    project: Path,
    missing_ok: bool = False,
) -> Path | None:
    # Kept verbatim from v1: its streaming contract is asserted by source
    # inspection in test_qwen_jlens_transport.py. v2 uses the profile helper.
    if not os.path.lexists(staged):
        if missing_ok:
            return None
        raise FileNotFoundError(f"jlens checkpoint missing (staging: {staged})")
    _validate_regular_jlens_checkpoint(staged, origin="staging")
    payload = load_fit_checkpoint(staged)
    validate_checkpoint_structure(payload)
    destination = project / JLENS_CHECKPOINT_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined_jlens_checkpoint(project, destination)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".jlens_fit_checkpoint.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with staged.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if tmp_path.stat().st_size > MAX_JLENS_CHECKPOINT_BYTES:
            raise ValueError("jlens checkpoint size exceeds limit after copy")
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    _validate_regular_jlens_checkpoint(destination, origin="destination")
    installed = load_fit_checkpoint(destination)
    validate_checkpoint_structure(installed)
    return _assert_confined_jlens_checkpoint(project, destination)


def _stream_install_regular_file(
    *,
    staged: Path,
    destination: Path,
    max_bytes: int,
    prefix: str,
) -> Path:
    if path_is_link_or_junction(staged) or not staged.is_file():
        raise ValueError(f"jlens resume member must be a regular file: {staged}")
    size = staged.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"jlens resume member size {size} exceeds {max_bytes} bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with staged.open("rb") as incoming:
                shutil.copyfileobj(incoming, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if tmp_path.stat().st_size > max_bytes:
            raise ValueError("jlens resume member size exceeds limit after copy")
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return destination


def _optional_local_resume_bundle(
    profile: _JLensTransportProfile,
    project: Path,
    staging_names: tuple[str, str, str],
) -> list[tuple[Path, str]]:
    checkpoint = _optional_local_checkpoint(profile, project)
    if checkpoint is None:
        return []
    status = project / profile.status
    events = project / profile.events
    if not status.is_file():
        raise ValueError("jlens resume requires status.json with the checkpoint")
    if not events.is_file():
        raise ValueError("jlens resume requires events.jsonl with the checkpoint")
    if path_is_link_or_junction(status) or path_is_link_or_junction(events):
        raise ValueError("jlens resume status/events must be regular files")
    if status.stat().st_size > MAX_JLENS_JSON_BYTES:
        raise ValueError("jlens status.json exceeds size limit")
    if events.stat().st_size > MAX_JLENS_JSONL_BYTES:
        raise ValueError("jlens events.jsonl exceeds size limit")
    _validate_resume_bundle(
        profile, checkpoint=checkpoint, status=status, events=events
    )
    return [
        (checkpoint, staging_names[0]),
        (status, staging_names[1]),
        (events, staging_names[2]),
    ]


def optional_local_jlens_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    return _optional_local_resume_bundle(
        _JLENS_V1_TRANSPORT,
        project,
        (
            REMOTE_JLENS_CHECKPOINT_STAGING,
            REMOTE_JLENS_STATUS_STAGING,
            REMOTE_JLENS_EVENTS_STAGING,
        ),
    )


def validate_jlens_resume_bundle(
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
) -> dict[str, Any]:
    return _validate_resume_bundle(
        _JLENS_V1_TRANSPORT, checkpoint=checkpoint, status=status, events=events
    )


def _validate_resume_bundle(
    profile: _JLensTransportProfile,
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
) -> dict[str, Any]:
    if path_is_link_or_junction(status) or not status.is_file():
        raise ValueError("jlens resume requires status.json with the checkpoint")
    if path_is_link_or_junction(events) or not events.is_file():
        raise ValueError("jlens resume requires events.jsonl with the checkpoint")
    if status.stat().st_size > MAX_JLENS_JSON_BYTES:
        raise ValueError("jlens status.json exceeds size limit")
    if events.stat().st_size > MAX_JLENS_JSONL_BYTES:
        raise ValueError("jlens events.jsonl exceeds size limit")
    _validate_regular_jlens_checkpoint(checkpoint, origin="resume")
    payload = load_fit_checkpoint(checkpoint)
    profile.validate_payload(payload)
    validate_jlens_pack_json(status, _posix(profile.status))
    validate_jlens_pack_json(events, _posix(profile.events))
    status_payload = json.loads(status.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not run_id or status_payload.get("run_id") != run_id:
        raise ValueError("jlens resume run IDs do not match")
    n_done = int(payload["n_done"])
    status_progress = status_payload.get("fit_completed_units")
    if status_progress is None:
        status_progress = status_payload.get("completed_units", 0)
    if int(status_progress or 0) > n_done:
        raise ValueError("jlens resume status is ahead of checkpoint")
    event_done = 0
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("run_id") != run_id:
            raise ValueError("jlens resume events run IDs do not match")
        if event.get("event") == "checkpoint" and event.get("n_done") is not None:
            event_done = max(event_done, int(event["n_done"]))
    if event_done > n_done:
        raise ValueError("jlens resume events are ahead of checkpoint")
    meta: dict[str, Any] = {
        "run_id": str(run_id),
        "n_done": n_done,
        "dim_batch": int(payload["dim_batch"]),
    }
    if profile.v2:
        meta["fit_count"] = int(payload["fit_count"])
        fit_total = status_payload.get("fit_total_units")
        if fit_total is not None and (
            type(fit_total) is not int or fit_total != meta["fit_count"]
        ):
            raise ValueError(
                f"jlens resume status fit_total_units {fit_total!r} != "
                f"checkpoint fit_count {meta['fit_count']}"
            )
    return meta


def atomic_install_jlens_resume(
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
    project: Path,
    missing_ok: bool = False,
    consume_staging: bool = True,
) -> Path | None:
    return _atomic_install_resume(
        _JLENS_V1_TRANSPORT,
        checkpoint=checkpoint,
        status=status,
        events=events,
        project=project,
        missing_ok=missing_ok,
        consume_staging=consume_staging,
    )


def _atomic_install_resume(
    profile: _JLensTransportProfile,
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
    project: Path,
    missing_ok: bool = False,
    consume_staging: bool = True,
) -> Path | None:
    if not os.path.lexists(checkpoint):
        if missing_ok:
            return None
        raise FileNotFoundError(f"jlens checkpoint missing (staging: {checkpoint})")
    if not os.path.lexists(status) or not os.path.lexists(events):
        raise ValueError("jlens resume requires status.json and events.jsonl")
    meta = _validate_resume_bundle(
        profile, checkpoint=checkpoint, status=status, events=events
    )
    destinations = {
        "checkpoint": project / profile.checkpoint,
        "status": project / profile.status,
        "events": project / profile.events,
        "marker": project / profile.marker,
    }
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined_checkpoint(project, destinations["checkpoint"], profile.checkpoint)
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=".jlens_resume_backup.",
            dir=destinations["checkpoint"].parent,
        )
    )
    records: list[InstallRecord] = []
    try:
        for destination in destinations.values():
            had_original = destination.exists()
            backup: Path | None = None
            if had_original:
                backup = backup_root / destination.name
                os.replace(destination, backup)
            records.append(
                InstallRecord(
                    target=destination, backup=backup, had_original=had_original
                )
            )
        installed = _atomic_install_checkpoint(
            profile, staged=checkpoint, project=project
        )
        _stream_install_regular_file(
            staged=status,
            destination=destinations["status"],
            max_bytes=MAX_JLENS_JSON_BYTES,
            prefix=".jlens_status.",
        )
        validate_jlens_pack_json(destinations["status"], _posix(profile.status))
        _stream_install_regular_file(
            staged=events,
            destination=destinations["events"],
            max_bytes=MAX_JLENS_JSONL_BYTES,
            prefix=".jlens_events.",
        )
        validate_jlens_pack_json(destinations["events"], _posix(profile.events))
        marker_payload = {
            "run_id": meta["run_id"],
            "n_done": meta["n_done"],
            "dim_batch": meta["dim_batch"],
            "checkpoint": _posix(profile.checkpoint),
            "status": _posix(profile.status),
            "events": _posix(profile.events),
        }
        if profile.v2:
            marker_payload["fit_count"] = meta["fit_count"]
        destinations["marker"].write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_jlens_pack_json(destinations["marker"], _posix(profile.marker))
        if consume_staging:
            for staged in (checkpoint, status, events):
                if staged.exists() and staged.is_file():
                    staged.unlink()
        return installed
    except Exception:
        _rollback_install_records(records)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def is_allowed_dose_fetch_member(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return False
    if is_dose_unsafe_member_name(normalized):
        return False
    if normalized in DOSE_FETCH_ALLOWLIST:
        return True
    if normalized in DOSE_TENSOR_ALLOWLIST:
        return True
    return bool(DOSE_CHECKPOINT_JSON_PATTERN.fullmatch(normalized))


def collect_qwen_pack_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    results_dir = root / "results" / "qwen25_coder_7b"
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json")):
            if path.is_file() and not path.is_symlink():
                entries.append(path)
    for relative in sorted(QWEN_FETCH_ALLOWLIST):
        if relative.startswith(QWEN_RESULTS_DIR):
            continue
        path = root / Path(relative)
        if path.is_file() and not path.is_symlink():
            entries.append(path)
    return entries


def pack_qwen_results(root: Path, archive: Path) -> None:
    _pack_entries(collect_qwen_pack_entries(root), root, archive, kind="Qwen")


def _assert_pack_path_safe(root: Path, path: Path) -> None:
    current = path
    while True:
        if path_is_link_or_junction(current):
            raise ValueError(f"pack path is a symlink or junction: {current}")
        if current == root or current.parent == current:
            break
        current = current.parent


def validate_dose_pack_json(path: Path, relative: str) -> None:
    name = Path(relative).name
    if name == "events.jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if metadata_contains_leak(payload):
                raise ValueError(f"compact json leakage in {relative}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if metadata_contains_leak(payload):
        raise ValueError(f"compact json leakage in {relative}")
    if name == "status.json":
        return
    fields = (
        EVALUATION_SCHEMA_FIELDS
        if name == "dose_followup_evaluation.json"
        else COMPACT_SCHEMA_FIELDS
    )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(
            f"compact json schema missing {', '.join(missing)} in {relative}"
        )


def collect_qwen_dose_pack_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    candidates: list[Path] = []
    results_dir = root / "results" / "qwen25_coder_7b_dose"
    if results_dir.is_dir():
        candidates.extend(sorted(results_dir.glob("*.json")))
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b_dose"
    if artifacts_dir.is_dir():
        candidates.extend(
            artifacts_dir / name for name in ("status.json", "events.jsonl")
        )
        candidates.extend(sorted(artifacts_dir.glob("*.pt")))
    for path in candidates:
        if not path.is_file() or path.is_symlink() or path_is_link_or_junction(path):
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if not is_allowed_dose_fetch_member(relative):
            continue
        _assert_pack_path_safe(root, path)
        if relative.endswith((".json", ".jsonl")):
            validate_dose_pack_json(path, relative)
        entries.append(path)
    return entries


def pack_qwen_dose_results(root: Path, archive: Path) -> None:
    _pack_entries(
        collect_qwen_dose_pack_entries(root),
        root,
        archive,
        kind="Qwen dose",
    )


def _pack_entries(
    entries: list[Path],
    root: Path,
    archive: Path,
    *,
    kind: str,
) -> None:
    if not entries:
        raise FileNotFoundError(f"no {kind} artifacts available to pack")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        for path in entries:
            arcname = normalize_member_name(
                str(path.relative_to(root)).replace("\\", "/")
            )
            handle.add(path, arcname=arcname)


def jlens_member_byte_limit(name: str) -> int:
    normalized = name.replace("\\", "/")
    if normalized.endswith(".pt"):
        return MAX_MEMBER_BYTES
    if normalized.endswith(".jsonl"):
        return MAX_JLENS_JSONL_BYTES
    return MAX_JLENS_JSON_BYTES


def _validate_tar_member(
    member: tarfile.TarInfo,
    *,
    is_unsafe: Callable[[str], bool],
    is_allowed: Callable[[str], bool],
    max_member: int | None = None,
    member_limit_for: Callable[[str], int] | None = None,
) -> None:
    if member.issym() or member.islnk():
        raise ValueError(f"archive member is a link: {member.name}")
    if member.isdir():
        raise ValueError(f"archive member is a directory: {member.name}")
    if not member.isreg():
        raise ValueError(f"archive member is not a regular file: {member.name}")
    normalized = normalize_member_name(member.name)
    if is_unsafe(normalized):
        raise ValueError(f"unsafe archive path: {member.name}")
    if not is_allowed(normalized):
        raise ValueError(f"archive member outside allowlist: {member.name}")
    if member_limit_for is not None:
        limit = member_limit_for(normalized)
    else:
        limit = MAX_MEMBER_BYTES if max_member is None else max_member
    if member.size > limit:
        raise ValueError(f"archive member exceeds size limit: {member.name}")


def _validate_archive(
    members: list[tarfile.TarInfo],
    *,
    is_unsafe: Callable[[str], bool],
    is_allowed: Callable[[str], bool],
    max_total: int | None = None,
    max_member: int | None = None,
    member_limit_for: Callable[[str], int] | None = None,
) -> list[tarfile.TarInfo]:
    seen: set[str] = set()
    total_size = 0
    file_members: list[tarfile.TarInfo] = []
    member_limit = MAX_MEMBER_BYTES if max_member is None else max_member
    total_limit = (
        MAX_TOTAL_UNCOMPRESSED_BYTES if max_total is None else max_total
    )
    for member in members:
        _validate_tar_member(
            member,
            is_unsafe=is_unsafe,
            is_allowed=is_allowed,
            max_member=member_limit,
            member_limit_for=member_limit_for,
        )
        normalized = normalize_member_name(member.name)
        if normalized in seen:
            raise ValueError(f"duplicate archive member: {member.name}")
        seen.add(normalized)
        total_size += member.size
        file_members.append(member)
    if total_size > total_limit:
        raise ValueError("archive exceeds total uncompressed size limit")
    return file_members


def validate_qwen_archive(members: list[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    return _validate_archive(
        members,
        is_unsafe=is_unsafe_member_name,
        is_allowed=is_allowed_fetch_member,
    )


def validate_qwen_dose_archive(members: list[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    return _validate_archive(
        members,
        is_unsafe=is_dose_unsafe_member_name,
        is_allowed=is_allowed_dose_fetch_member,
    )


@dataclass
class InstallRecord:
    target: Path
    backup: Path | None
    had_original: bool


def _rollback_install_records(records: list[InstallRecord]) -> None:
    for record in reversed(records):
        if record.backup is not None and record.backup.exists():
            if record.target.exists():
                record.target.unlink()
            os.replace(record.backup, record.target)
        elif record.had_original and record.target.exists():
            record.target.unlink()
        elif not record.had_original and record.target.exists():
            record.target.unlink()


def extract_qwen_archive(archive: Path, project: Path) -> None:
    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_archive,
        prefix="qwen-fetch-",
    )


def extract_qwen_dose_archive(archive: Path, project: Path) -> None:
    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_dose_archive,
        prefix="qwen-dose-fetch-",
    )


def is_jlens_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    if not normalized.startswith((f"{JLENS_RESULTS_DIR}/", f"{JLENS_ARTIFACTS_DIR}/")):
        return True
    return False


def is_allowed_jlens_fetch_member(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return False
    if is_jlens_unsafe_member_name(normalized):
        return False
    return normalized in JLENS_FETCH_ALLOWLIST


def validate_jlens_pack_json(path: Path, relative: str) -> None:
    name = Path(relative).name
    if name == "events.jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if jlens_metadata_contains_leak(payload):
                raise ValueError(f"compact json leakage in {relative}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if jlens_metadata_contains_leak(payload):
        raise ValueError(f"compact json leakage in {relative}")
    if name in {"status.json", "jlens_resume_bundle.json"}:
        return
    fields = (
        JLENS_EVALUATION_SCHEMA_FIELDS
        if name == "jlens_evaluation.json"
        else COMPACT_SCHEMA_FIELDS
    )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(
            f"compact json schema missing {', '.join(missing)} in {relative}"
        )


def collect_qwen_jlens_pack_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    candidates: list[Path] = []
    results_dir = root / "results" / "qwen25_coder_7b_jlens"
    if results_dir.is_dir():
        candidates.extend(sorted(results_dir.glob("*.json")))
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b_jlens"
    if artifacts_dir.is_dir():
        candidates.extend(
            artifacts_dir / name
            for name in (
                "status.json",
                "events.jsonl",
                "jlens_fit_checkpoint.pt",
                "qwen25_coder_7b_jacobian_lens.pt",
                "jlens_resume_bundle.json",
            )
        )
    for path in candidates:
        if not path.is_file() or path.is_symlink() or path_is_link_or_junction(path):
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if not is_allowed_jlens_fetch_member(relative):
            continue
        _assert_pack_path_safe(root, path)
        if relative.endswith((".json", ".jsonl")):
            validate_jlens_pack_json(path, relative)
        entries.append(path)
    return entries


def pack_qwen_jlens_results(root: Path, archive: Path) -> None:
    _pack_entries(
        collect_qwen_jlens_pack_entries(root),
        root,
        archive,
        kind="Qwen jlens",
    )


def validate_qwen_jlens_archive(members: list[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    return _validate_archive(
        members,
        is_unsafe=is_jlens_unsafe_member_name,
        is_allowed=is_allowed_jlens_fetch_member,
        max_total=MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES,
        member_limit_for=jlens_member_byte_limit,
    )


def _revalidate_staged_jlens_members(
    staging: Path,
    members: list[tarfile.TarInfo],
) -> None:
    for member in members:
        relative = normalize_member_name(member.name)
        path = staging / relative
        if path.stat().st_size > jlens_member_byte_limit(relative):
            raise ValueError(f"archive member exceeds size limit: {member.name}")
        if relative.endswith((".json", ".jsonl")):
            validate_jlens_pack_json(path, relative)


def extract_qwen_jlens_archive(archive: Path, project: Path) -> None:
    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_jlens_archive,
        prefix="qwen-jlens-fetch-",
        post_stage_validate=_revalidate_staged_jlens_members,
    )


def jlens_v2_checkpoint_path(project: Path) -> Path:
    return project / JLENS_V2_CHECKPOINT_RELATIVE


def optional_local_jlens_v2_checkpoint(project: Path) -> Path | None:
    return _optional_local_checkpoint(_JLENS_V2_TRANSPORT, project)


def atomic_install_jlens_v2_checkpoint(
    *,
    staged: Path,
    project: Path,
    missing_ok: bool = False,
) -> Path | None:
    return _atomic_install_checkpoint(
        _JLENS_V2_TRANSPORT, staged=staged, project=project, missing_ok=missing_ok
    )


def optional_local_jlens_v2_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    return _optional_local_resume_bundle(
        _JLENS_V2_TRANSPORT,
        project,
        (
            REMOTE_JLENS_V2_CHECKPOINT_STAGING,
            REMOTE_JLENS_V2_STATUS_STAGING,
            REMOTE_JLENS_V2_EVENTS_STAGING,
        ),
    )


def validate_jlens_v2_resume_bundle(
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
) -> dict[str, Any]:
    return _validate_resume_bundle(
        _JLENS_V2_TRANSPORT, checkpoint=checkpoint, status=status, events=events
    )


def atomic_install_jlens_v2_resume(
    *,
    checkpoint: Path,
    status: Path,
    events: Path,
    project: Path,
    missing_ok: bool = False,
    consume_staging: bool = True,
) -> Path | None:
    return _atomic_install_resume(
        _JLENS_V2_TRANSPORT,
        checkpoint=checkpoint,
        status=status,
        events=events,
        project=project,
        missing_ok=missing_ok,
        consume_staging=consume_staging,
    )


def atomic_replace_validated_jlens_v2_backup(staged: Path, destination: Path) -> Path:
    """Replace ``destination`` with ``staged`` only after full validation.

    The previous destination is never touched until a complete, fsynced copy
    next to it has been loaded and validated; the swap is one ``os.replace``.
    """

    _validate_regular_jlens_checkpoint(staged, origin="backup staging")
    if os.path.lexists(destination) and (
        path_is_link_or_junction(destination) or not destination.is_file()
    ):
        raise ValueError(f"jlens backup destination must be a regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_validate_replace_checkpoint(
        _JLENS_V2_TRANSPORT,
        staged=staged,
        destination=destination,
        prefix=".jlens_v2_backup.",
    )
    return destination


def is_jlens_v2_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    return not normalized.startswith(
        (f"{JLENS_V2_RESULTS_DIR}/", f"{JLENS_V2_ARTIFACTS_DIR}/")
    )


def is_allowed_jlens_v2_fetch_member(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return False
    if is_jlens_v2_unsafe_member_name(normalized):
        return False
    return normalized in JLENS_V2_FETCH_ALLOWLIST


def collect_qwen_jlens_v2_pack_entries(root: Path) -> list[Path]:
    entries: list[Path] = []
    for relative in sorted(JLENS_V2_FETCH_ALLOWLIST):
        path = root / Path(relative)
        if not path.is_file() or path.is_symlink() or path_is_link_or_junction(path):
            continue
        _assert_pack_path_safe(root, path)
        if relative.endswith((".json", ".jsonl")):
            validate_jlens_pack_json(path, relative)
        entries.append(path)
    return entries


def pack_qwen_jlens_v2_results(root: Path, archive: Path) -> None:
    _pack_entries(
        collect_qwen_jlens_v2_pack_entries(root),
        root,
        archive,
        kind="Qwen jlens v2",
    )


def validate_qwen_jlens_v2_archive(
    members: list[tarfile.TarInfo],
) -> list[tarfile.TarInfo]:
    return _validate_archive(
        members,
        is_unsafe=is_jlens_v2_unsafe_member_name,
        is_allowed=is_allowed_jlens_v2_fetch_member,
        max_total=MAX_JLENS_TOTAL_UNCOMPRESSED_BYTES,
        member_limit_for=jlens_member_byte_limit,
    )


def _validate_staged_jlens_v2_members(
    staging: Path,
    members: list[tarfile.TarInfo],
) -> None:
    """Validate staged v2 members before any project file is replaced."""

    _revalidate_staged_jlens_members(staging, members)
    artifacts = staging / JLENS_V2_ARTIFACTS_DIR
    checkpoint = artifacts / "jlens_fit_checkpoint.pt"
    if not checkpoint.is_file():
        return
    status = artifacts / "status.json"
    events = artifacts / "events.jsonl"
    if status.is_file() and events.is_file():
        _validate_resume_bundle(
            _JLENS_V2_TRANSPORT, checkpoint=checkpoint, status=status, events=events
        )
        return
    _validate_regular_jlens_checkpoint(checkpoint, origin="archive")
    validate_jlens_v2_checkpoint_payload(load_fit_checkpoint(checkpoint))


def extract_qwen_jlens_v2_archive(archive: Path, project: Path) -> None:
    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_jlens_v2_archive,
        prefix="qwen-jlens-v2-fetch-",
        post_stage_validate=_validate_staged_jlens_v2_members,
    )


def is_code_eval_unsafe_member_name(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return True
    return not normalized.startswith(
        (f"{CODE_EVAL_RESULTS_DIR}/", f"{CODE_EVAL_ARTIFACTS_DIR}/")
    )


def is_allowed_code_eval_fetch_member(name: str) -> bool:
    try:
        normalized = normalize_member_name(name)
    except ValueError:
        return False
    if is_code_eval_unsafe_member_name(normalized):
        return False
    if normalized in {
        f"{CODE_EVAL_ARTIFACTS_DIR}/status.json",
        f"{CODE_EVAL_ARTIFACTS_DIR}/events.jsonl",
    }:
        return True
    if normalized.startswith(f"{CODE_EVAL_RESULTS_DIR}/"):
        return Path(normalized).name in CODE_EVAL_COMPACT_NAMES
    return any(pattern.fullmatch(normalized) for pattern in CODE_EVAL_RAW_PATTERNS)


def code_eval_member_byte_limit(name: str) -> int:
    normalized = normalize_member_name(name)
    if normalized.endswith(".pt"):
        return MAX_MEMBER_BYTES
    if normalized.endswith(".jsonl"):
        return MAX_CODE_EVAL_JSONL_BYTES
    return MAX_CODE_EVAL_JSON_BYTES


def _validate_code_eval_compact(path: Path, relative: str) -> None:
    if not relative.startswith(f"{CODE_EVAL_RESULTS_DIR}/"):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if compact_contains_leak(payload):
        raise ValueError(f"compact json leakage in {relative}")


def collect_qwen_code_eval_pack_entries(root: Path) -> list[Path]:
    candidates: list[Path] = []
    results_dir = root / CODE_EVAL_RESULTS_DIR
    artifacts_dir = root / CODE_EVAL_ARTIFACTS_DIR
    if results_dir.is_dir():
        candidates.extend(sorted(results_dir.glob("*.json")))
    if artifacts_dir.is_dir():
        candidates.extend(
            path for path in sorted(artifacts_dir.rglob("*")) if path.is_file()
        )
    entries: list[Path] = []
    for path in candidates:
        if path.is_symlink() or path_is_link_or_junction(path):
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if not is_allowed_code_eval_fetch_member(relative):
            continue
        _assert_pack_path_safe(root, path)
        if path.stat().st_size > code_eval_member_byte_limit(relative):
            raise ValueError(f"code-eval member exceeds size limit: {relative}")
        _validate_code_eval_compact(path, relative)
        entries.append(path)
    return entries


def pack_qwen_code_eval_results(root: Path, archive: Path) -> None:
    _pack_entries(
        collect_qwen_code_eval_pack_entries(root),
        root,
        archive,
        kind="Qwen code eval",
    )


def validate_qwen_code_eval_archive(
    members: list[tarfile.TarInfo],
) -> list[tarfile.TarInfo]:
    return _validate_archive(
        members,
        is_unsafe=is_code_eval_unsafe_member_name,
        is_allowed=is_allowed_code_eval_fetch_member,
        max_total=MAX_CODE_EVAL_TOTAL_BYTES,
        member_limit_for=code_eval_member_byte_limit,
    )


def _revalidate_code_eval_members(
    staging: Path,
    members: list[tarfile.TarInfo],
) -> None:
    for member in members:
        relative = normalize_member_name(member.name)
        path = staging / relative
        if path.stat().st_size > code_eval_member_byte_limit(relative):
            raise ValueError(f"code-eval member exceeds size limit: {relative}")
        _validate_code_eval_compact(path, relative)


def _code_eval_tree_identity(root: Path) -> dict[str, Any] | None:
    results_dir = root / CODE_EVAL_RESULTS_DIR
    identities: list[dict[str, Any]] = []

    def add_identity(source: dict[str, Any]) -> None:
        fields = ("run_id", "implementation_commit", "source_artifact_hashes")
        if not all(field in source for field in fields):
            return
        identities.append(
            {
                "run_id": source["run_id"],
                "implementation_commit": source["implementation_commit"],
                "source_artifact_hashes": source["source_artifact_hashes"],
            }
        )

    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            add_identity(payload.get("provenance") or payload)
    artifacts_dir = root / CODE_EVAL_ARTIFACTS_DIR
    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                add_identity(payload.get("identity") or payload)
            elif path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            add_identity(json.loads(line))
            elif path.suffix == ".pt" and path.parent.name == "logits":
                payload = torch.load(path, map_location="cpu", weights_only=True)
                add_identity(payload)
    if not identities:
        return None
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise ValueError("code-eval tree contains conflicting identities")
    return first


def _code_eval_destination_exists(project: Path) -> bool:
    return any(
        any(candidate.is_file() for candidate in path.rglob("*"))
        for path in (
            project / CODE_EVAL_RESULTS_DIR,
            project / CODE_EVAL_ARTIFACTS_DIR,
        )
        if path.is_dir()
    )


def _validate_code_eval_fetch_identity(staging: Path, project: Path) -> None:
    incoming = _code_eval_tree_identity(staging)
    if not _code_eval_destination_exists(project):
        return
    existing = _code_eval_tree_identity(project)
    if incoming is None or existing is None:
        raise ValueError("code-eval identity conflict: identity is unavailable")
    if incoming != existing:
        raise ValueError(
            "code-eval identity conflict: run, commit, or source hashes differ"
        )


def extract_qwen_code_eval_archive(archive: Path, project: Path) -> None:
    def validate_staged(staging: Path, members: list[tarfile.TarInfo]) -> None:
        _revalidate_code_eval_members(staging, members)
        _validate_code_eval_fetch_identity(staging, project)

    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_code_eval_archive,
        prefix="qwen-code-eval-fetch-",
        post_stage_validate=validate_staged,
    )


def _extract_allowlisted_archive(
    archive: Path,
    project: Path,
    *,
    validator: Callable[[list[tarfile.TarInfo]], list[tarfile.TarInfo]],
    prefix: str,
    post_stage_validate: Callable[[Path, list[tarfile.TarInfo]], None] | None = None,
) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        file_members = validator(members)
        staging = Path(tempfile.mkdtemp(prefix=prefix, dir=project.parent))
        records: list[InstallRecord] = []
        try:
            for member in file_members:
                destination = staging / normalize_member_name(member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to extract member: {member.name}")
                with destination.open("wb") as out:
                    shutil.copyfileobj(extracted, out)
            if post_stage_validate is not None:
                post_stage_validate(staging, file_members)

            backup_root = staging / ".backups"
            for member in file_members:
                relative = normalize_member_name(member.name)
                target = project / relative
                source = staging / relative
                current = project
                for part in Path(relative).parts:
                    current = current / part
                    if path_is_link_or_junction(current):
                        raise ValueError(
                            f"result destination is a symlink or junction: {current}"
                        )
                backup: Path | None = None
                had_original = target.exists()
                if had_original:
                    backup = backup_root / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                records.append(
                    InstallRecord(target=target, backup=backup, had_original=had_original)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        except Exception:
            _rollback_install_records(records)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)