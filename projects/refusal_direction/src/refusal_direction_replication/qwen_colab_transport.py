"""Safe Qwen-only Colab result packing and local extraction."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .qwen_dose import metadata_contains_leak
from .qwen_code_eval import compact_contains_leak
from .qwen_jlens import (
    jlens_metadata_contains_leak,
    load_fit_checkpoint,
    validate_checkpoint_structure,
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
        rf"{CODE_EVAL_DATASETS}__baseline\.pt$"
    ),
    re.compile(
        rf"^{CODE_EVAL_ARTIFACTS_DIR}/evalplus/"
        rf"(?:selected_problems\.jsonl|samples__{CODE_EVAL_CONDITIONS}\.jsonl|"
        rf"results__{CODE_EVAL_CONDITIONS}\.json)$"
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


def _assert_confined_jlens_checkpoint(project: Path, destination: Path) -> Path:
    project_root = project.resolve()
    expected = (project_root / JLENS_CHECKPOINT_RELATIVE).resolve()
    current = destination
    for _ in range(len(destination.parts) + 1):
        if path_is_link_or_junction(current):
            raise ValueError(
                "jlens checkpoint destination escaped "
                "artifacts/qwen25_coder_7b_jlens/jlens_fit_checkpoint.pt"
            )
        if current == project or current.parent == current:
            break
        current = current.parent
    resolved = destination.resolve()
    if resolved != expected or resolved.name != "jlens_fit_checkpoint.pt":
        raise ValueError(
            "jlens checkpoint destination must remain "
            "artifacts/qwen25_coder_7b_jlens/jlens_fit_checkpoint.pt"
        )
    try:
        resolved.relative_to(project_root / "artifacts" / "qwen25_coder_7b_jlens")
    except ValueError as error:
        raise ValueError(
            "jlens checkpoint destination escaped "
            "artifacts/qwen25_coder_7b_jlens/jlens_fit_checkpoint.pt"
        ) from error
    return destination


def optional_local_jlens_checkpoint(project: Path) -> Path | None:
    path = jlens_checkpoint_path(project)
    if not os.path.lexists(path):
        return None
    _validate_regular_jlens_checkpoint(path, origin="local")
    payload = load_fit_checkpoint(path)
    validate_checkpoint_structure(payload)
    return _assert_confined_jlens_checkpoint(project, path)


def atomic_install_jlens_checkpoint(
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


def optional_local_jlens_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    checkpoint = optional_local_jlens_checkpoint(project)
    if checkpoint is None:
        return []
    status = project / JLENS_STATUS_RELATIVE
    events = project / JLENS_EVENTS_RELATIVE
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
    validate_jlens_resume_bundle(checkpoint=checkpoint, status=status, events=events)
    return [
        (checkpoint, REMOTE_JLENS_CHECKPOINT_STAGING),
        (status, REMOTE_JLENS_STATUS_STAGING),
        (events, REMOTE_JLENS_EVENTS_STAGING),
    ]


def validate_jlens_resume_bundle(
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
    validate_checkpoint_structure(payload)
    validate_jlens_pack_json(status, str(JLENS_STATUS_RELATIVE).replace("\\", "/"))
    validate_jlens_pack_json(events, str(JLENS_EVENTS_RELATIVE).replace("\\", "/"))
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
    return {
        "run_id": str(run_id),
        "n_done": n_done,
        "dim_batch": int(payload["dim_batch"]),
    }


def atomic_install_jlens_resume(
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
    meta = validate_jlens_resume_bundle(
        checkpoint=checkpoint, status=status, events=events
    )
    destinations = {
        "checkpoint": project / JLENS_CHECKPOINT_RELATIVE,
        "status": project / JLENS_STATUS_RELATIVE,
        "events": project / JLENS_EVENTS_RELATIVE,
        "marker": project / JLENS_RESUME_MARKER_RELATIVE,
    }
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_confined_jlens_checkpoint(project, destinations["checkpoint"])
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
        installed = atomic_install_jlens_checkpoint(staged=checkpoint, project=project)
        _stream_install_regular_file(
            staged=status,
            destination=destinations["status"],
            max_bytes=MAX_JLENS_JSON_BYTES,
            prefix=".jlens_status.",
        )
        validate_jlens_pack_json(
            destinations["status"], str(JLENS_STATUS_RELATIVE).replace("\\", "/")
        )
        _stream_install_regular_file(
            staged=events,
            destination=destinations["events"],
            max_bytes=MAX_JLENS_JSONL_BYTES,
            prefix=".jlens_events.",
        )
        validate_jlens_pack_json(
            destinations["events"], str(JLENS_EVENTS_RELATIVE).replace("\\", "/")
        )
        marker_payload = {
            "run_id": meta["run_id"],
            "n_done": meta["n_done"],
            "dim_batch": meta["dim_batch"],
            "checkpoint": str(JLENS_CHECKPOINT_RELATIVE).replace("\\", "/"),
            "status": str(JLENS_STATUS_RELATIVE).replace("\\", "/"),
            "events": str(JLENS_EVENTS_RELATIVE).replace("\\", "/"),
        }
        destinations["marker"].write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_jlens_pack_json(
            destinations["marker"],
            str(JLENS_RESUME_MARKER_RELATIVE).replace("\\", "/"),
        )
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


def extract_qwen_code_eval_archive(archive: Path, project: Path) -> None:
    _extract_allowlisted_archive(
        archive,
        project,
        validator=validate_qwen_code_eval_archive,
        prefix="qwen-code-eval-fetch-",
        post_stage_validate=_revalidate_code_eval_members,
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