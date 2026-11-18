"""Remote entrypoint for sparse-refusal stages and allowlisted archives."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable


REMOTE_ROOT = Path("/content/refusal_direction")
REMOTE_COMMAND = Path("/content/qwen_sparse_refusal_command.json")
REMOTE_CHECKPOINT_ARCHIVE = Path(
    "/content/qwen_sparse_refusal_checkpoint.tar.gz"
)
REMOTE_RESULTS_ARCHIVE = Path("/content/qwen_sparse_refusal_results.tar.gz")
SPARSE_PROFILE = "qwen25_coder_sparse_refusal.json"
SPARSE_RESULTS_DIR = "results/qwen25_coder_sparse_refusal"
SPARSE_ARTIFACTS_DIR = "artifacts/qwen25_coder_sparse_refusal"
SPARSE_CHECKPOINT_ALLOWLIST = (
    f"{SPARSE_ARTIFACTS_DIR}/sae_checkpoint.pt",
    f"{SPARSE_ARTIFACTS_DIR}/training_status.json",
)
SPARSE_RESULTS_ALLOWLIST = (
    *(f"{SPARSE_RESULTS_DIR}/{index:02d}_{stage}.json" for index, stage in enumerate(
        (
            "prepare",
            "hardware_pilot",
            "sae_fit",
            "sae_quality",
            "decompose",
            "discover",
            "select",
            "test",
            "review_export",
            "review_merge",
            "external_transfer",
            "code_preservation",
            "finalize",
        )
    )),
    f"{SPARSE_RESULTS_DIR}/selection_freeze.json",
    f"{SPARSE_RESULTS_DIR}/operational_error.json",
    f"{SPARSE_RESULTS_DIR}/execution_manifest.json",
)
REMOTE_ACTIONS = frozenset(
    {"clean-staging", "pilot", "run", "pack-checkpoint", "pack-results"}
)


def _archive_command(
    archive: Path,
    allowlist: tuple[str, ...],
) -> str:
    arguments = [
        "tar",
        "-czf",
        archive.as_posix(),
        "-C",
        REMOTE_ROOT.as_posix(),
        *allowlist,
    ]
    return shlex.join(arguments)


def sparse_checkpoint_archive_command() -> str:
    return _archive_command(
        REMOTE_CHECKPOINT_ARCHIVE,
        SPARSE_CHECKPOINT_ALLOWLIST,
    )


def sparse_results_archive_command() -> str:
    return _archive_command(
        REMOTE_RESULTS_ARCHIVE,
        SPARSE_RESULTS_ALLOWLIST,
    )


def _safe_allowlisted_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe sparse archive member: {relative}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relative_path).resolve()
    if root_resolved not in candidate.parents:
        raise ValueError(f"sparse archive member escapes root: {relative}")
    return candidate


def _pack_allowlisted(
    root: Path,
    archive: Path,
    allowlist: tuple[str, ...],
) -> Path:
    existing: list[tuple[Path, str]] = []
    for relative in allowlist:
        source = _safe_allowlisted_path(root, relative)
        if source.is_symlink():
            raise ValueError(f"sparse archive source is a link: {relative}")
        if source.is_file():
            existing.append((source, relative))
    if not existing:
        raise FileNotFoundError("no allowlisted sparse files are available")
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz") as handle:
            for source, relative in existing:
                handle.add(source, arcname=relative, recursive=False)
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive


def pack_checkpoint_archive(
    root: Path = REMOTE_ROOT,
    archive: Path = REMOTE_CHECKPOINT_ARCHIVE,
) -> Path:
    return _pack_allowlisted(
        root,
        archive,
        SPARSE_CHECKPOINT_ALLOWLIST,
    )


def pack_results_archive(
    root: Path = REMOTE_ROOT,
    archive: Path = REMOTE_RESULTS_ARCHIVE,
) -> Path:
    return _pack_allowlisted(
        root,
        archive,
        SPARSE_RESULTS_ALLOWLIST,
    )


def clean_sparse_staging(content_root: Path = Path("/content")) -> None:
    exact_names = {
        "sparse_refusal_direction.pt",
        "sparse_refusal_direction.pt.manifest.json",
        "sparse_refusal_direction.pt.assembling",
        "sparse_refusal_reviewed_pairs.jsonl",
        "sparse_refusal_reviewed_pairs.sha256.json",
        "sparse_refusal_checkpoint.pt",
        "sparse_refusal_checkpoint.pt.manifest.json",
        "sparse_refusal_checkpoint.pt.assembling",
        "sparse_refusal_training_status.json",
    }
    prefixes = (
        "sparse_refusal_direction.pt.part",
        "sparse_refusal_checkpoint.pt.part",
    )
    for path in Path(content_root).iterdir():
        if path.name in exact_names or path.name.startswith(prefixes):
            if path.is_dir():
                raise ValueError(f"sparse staging path is a directory: {path}")
            path.unlink()


def run_sparse_action(
    action: str,
    *,
    root: Path = REMOTE_ROOT,
    run_function: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if action not in {"pilot", "run"}:
        raise ValueError(f"unsupported sparse stage action: {action}")
    if run_function is None:
        source = str(Path(root) / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from refusal_direction_replication.qwen_sparse_runtime import (
            run_qwen_sparse_refusal,
        )

        run_function = run_qwen_sparse_refusal
    return run_function(
        Path(root),
        SPARSE_PROFILE,
        stop_after="hardware_pilot" if action == "pilot" else None,
    )


def write_training_status(
    payload: dict[str, Any],
    *,
    root: Path = REMOTE_ROOT,
) -> Path:
    experiment = str(payload.get("experiment", ""))
    stage = experiment.removeprefix("qwen_sparse_refusal_")
    results = payload.get("results")
    status = (
        str(results.get("status", "unknown"))
        if isinstance(results, dict)
        else str(payload.get("status", "unknown"))
    )
    provenance = payload.get("provenance")
    run_id = (
        provenance.get("run_id")
        if isinstance(provenance, dict)
        else None
    )
    compact = {
        "last_completed_stage": stage,
        "run_id": str(run_id or "unknown"),
        "status": status,
    }
    destination = (
        Path(root)
        / SPARSE_ARTIFACTS_DIR
        / "training_status.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".training_status.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(compact, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def preserve_last_completed_status(root: Path = REMOTE_ROOT) -> Path | None:
    results_root = Path(root) / SPARSE_RESULTS_DIR
    latest: dict[str, Any] | None = None
    for relative in SPARSE_RESULTS_ALLOWLIST:
        name = Path(relative).name
        if not name[:2].isdigit():
            continue
        path = results_root / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            latest = payload
    return (
        write_training_status(latest, root=root)
        if latest is not None
        else None
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_execution_manifest(root: Path = REMOTE_ROOT) -> Path:
    results_root = Path(root) / SPARSE_RESULTS_DIR
    files: list[dict[str, str]] = []
    provenance: dict[str, Any] = {}
    for relative in SPARSE_RESULTS_ALLOWLIST:
        name = Path(relative).name
        if name == "execution_manifest.json":
            continue
        path = results_root / name
        if not path.is_file():
            continue
        files.append({"path": name, "sha256": _sha256_file(path)})
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(
            payload.get("provenance"), dict
        ):
            provenance = dict(payload["provenance"])
    if not files:
        raise FileNotFoundError("no compact sparse stages exist for manifest")
    manifest = {
        "schema_version": 1,
        "experiment": "qwen_sparse_refusal_execution_manifest",
        "results": {"files": files},
        "provenance": provenance,
    }
    destination = results_root / "execution_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".execution_manifest.",
        suffix=".tmp",
        dir=results_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _read_remote_action(path: Path = REMOTE_COMMAND) -> str:
    if not Path(path).is_file():
        return "run"
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        raise ValueError("sparse remote command must contain only action")
    action = payload["action"]
    if action not in REMOTE_ACTIONS:
        raise ValueError(f"unknown sparse remote action: {action}")
    return str(action)


def remote_main() -> int:
    action = _read_remote_action()
    if action == "clean-staging":
        clean_sparse_staging()
        print("sparse_staging=clean")
        return 0
    if action == "pack-checkpoint":
        path = pack_checkpoint_archive()
        print(f"checkpoint_archive={path}")
        return 0
    if action == "pack-results":
        path = pack_results_archive()
        print(f"results_archive={path}")
        return 0
    try:
        payload = run_sparse_action(action)
    except BaseException:
        preserve_last_completed_status()
        try:
            write_execution_manifest()
        except FileNotFoundError:
            pass
        raise
    write_training_status(payload)
    write_execution_manifest()
    experiment = str(payload.get("experiment", "unknown"))
    results = payload.get("results")
    status = (
        str(results.get("status", "unknown"))
        if isinstance(results, dict)
        else str(payload.get("status", "unknown"))
    )
    stage = experiment.removeprefix("qwen_sparse_refusal_")
    print(f"stage={stage} status={status}")
    return 1 if status == "operational_error" else 0


if __name__ == "__main__":
    raise SystemExit(remote_main())
