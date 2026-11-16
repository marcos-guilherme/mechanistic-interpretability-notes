"""Install pinned lab dependencies and frozen direction/J-lens artifacts.

Jacobian-lens is installed with pip --no-deps deliberately so the project's
Transformers pins stay. The recorded runtime_compatibility_smoke confirms
Qwen 7B hooks plus J-lens transport/unembed on that pinned runtime; it is
not a general compatibility claim.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REMOTE_ROOT = Path("/content/refusal_direction")
ARCHIVE = Path("/content/refusal_direction.tar.gz")
DESTINATION = Path("/content")
ARCHIVE_ROOT = "refusal_direction"
_SRC = "/content/refusal_direction/src"
REMOTE_LAB_DIRECTION_STAGING = "/content/lab_source_direction.pt"
REMOTE_LAB_LENS_STAGING = "/content/lab_source_lens.pt"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
GRADIO_SPEC = "gradio==5.49.1"
JLENS_SPEC = (
    "git+https://github.com/anthropics/jacobian-lens.git@" + JLENS_COMMIT
)


def _install(spec: str, *, no_deps: bool = False) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        command.append("--no-deps")
    command.append(spec)
    subprocess.run(command, check=True)


def _normalized_lab_member(name: str) -> str:
    cleaned = name.replace("\\", "/").rstrip("/")
    if not cleaned:
        raise ValueError(f"unsafe archive member: {name}")
    if cleaned != str(PurePosixPath(cleaned)):
        raise ValueError(f"unsafe archive member: {name}")
    path = PurePosixPath(cleaned)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member: {name}")
    if not path.parts or path.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"unsafe archive member top-level: {name}")
    return str(path)


def _validate_archive_member(member: tarfile.TarInfo) -> str:
    if member.issym() or member.islnk():
        raise ValueError(f"archive member is a link: {member.name}")
    if member.isdir():
        return _normalized_lab_member(member.name)
    if not member.isreg():
        raise ValueError(f"archive member is not a regular file: {member.name}")
    return _normalized_lab_member(member.name)


def safe_extract() -> None:
    if not ARCHIVE.is_file():
        if not REMOTE_ROOT.is_dir():
            raise FileNotFoundError("lab source archive is missing")
        return
    DESTINATION.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".lab-extract-", dir=DESTINATION))
    backup: Path | None = None
    try:
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            seen: set[str] = set()
            file_members: list[tuple[tarfile.TarInfo, str]] = []
            for member in handle.getmembers():
                normalized = _validate_archive_member(member)
                if normalized in seen:
                    raise ValueError(f"duplicate archive member: {member.name}")
                seen.add(normalized)
                if member.isdir():
                    (staging / normalized).mkdir(parents=True, exist_ok=True)
                else:
                    file_members.append((member, normalized))
            for member, normalized in file_members:
                target = staging / normalized
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to extract member: {member.name}")
                with target.open("wb") as out:
                    shutil.copyfileobj(extracted, out)
        extras = [path for path in staging.iterdir() if path.name != ARCHIVE_ROOT]
        if extras:
            raise ValueError("archive extracted extra top-level paths")
        extracted_root = staging / ARCHIVE_ROOT
        if not extracted_root.is_dir():
            raise ValueError("archive did not contain refusal_direction/")
        if os.path.lexists(REMOTE_ROOT):
            fd, backup_name = tempfile.mkstemp(
                prefix=".lab-root-backup-",
                dir=DESTINATION,
            )
            os.close(fd)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(REMOTE_ROOT, backup)
        os.replace(extracted_root, REMOTE_ROOT)
    except Exception:
        if backup is not None and os.path.lexists(backup):
            if os.path.lexists(REMOTE_ROOT):
                if REMOTE_ROOT.is_dir():
                    shutil.rmtree(REMOTE_ROOT)
                else:
                    REMOTE_ROOT.unlink()
            os.replace(backup, REMOTE_ROOT)
        raise
    else:
        if backup is not None and os.path.lexists(backup):
            if backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)
        try:
            ARCHIVE.unlink()
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def assemble_staged_lens() -> Path:
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from refusal_direction_replication.qwen_colab_transport import (
        assemble_chunked_file,
    )

    destination = Path(REMOTE_LAB_LENS_STAGING)
    return assemble_chunked_file(
        destination,
        manifest_path=destination.with_name(destination.name + ".manifest.json"),
        require_manifest=True,
    )


def _assert_gradio_pin() -> None:
    from refusal_direction_replication.qwen_lab_app import GRADIO_PIN_VERSION

    installed = importlib.metadata.version("gradio")
    if GRADIO_SPEC != f"gradio=={GRADIO_PIN_VERSION}":
        raise ValueError("Gradio install spec differs from GRADIO_PIN_VERSION")
    if installed != GRADIO_PIN_VERSION:
        raise ValueError(
            f"installed Gradio {installed!r} != pinned {GRADIO_PIN_VERSION!r}"
        )


def main() -> int:
    safe_extract()
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from refusal_direction_replication.qwen_colab_transport import (
        atomic_install_lab_sources,
        load_lab_profile_config,
    )
    from refusal_direction_replication.qwen_jlens import (
        JLENS_UPSTREAM_SPEC,
        RUNTIME_COMPATIBILITY_SMOKE,
        assert_jlens_contract,
        assert_pinned_transformers_runtime,
    )

    if JLENS_UPSTREAM_SPEC.rsplit("@", 1)[-1] != JLENS_COMMIT:
        raise ValueError("Jacobian-lens install spec differs from the pinned commit")
    if JLENS_SPEC != JLENS_UPSTREAM_SPEC:
        raise ValueError("Jacobian-lens install spec differs from the pinned commit")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REMOTE_ROOT)],
        check=True,
    )
    _install(JLENS_SPEC, no_deps=True)
    assert_pinned_transformers_runtime(REMOTE_ROOT / "pyproject.toml")
    _install(GRADIO_SPEC)
    assert_jlens_contract()
    _assert_gradio_pin()
    config = load_lab_profile_config(REMOTE_ROOT)
    if config.get("jlens_install_no_deps") is not True:
        raise ValueError("jlens --no-deps exception is not recorded in the lab config")
    if config.get("runtime_compatibility_smoke") != RUNTIME_COMPATIBILITY_SMOKE:
        raise ValueError("runtime_compatibility_smoke does not match the recorded identifier")
    staged_lens = assemble_staged_lens()
    atomic_install_lab_sources(
        staged_direction=Path(REMOTE_LAB_DIRECTION_STAGING),
        staged_lens=staged_lens,
        project=REMOTE_ROOT,
        expected_direction_sha256=str(config["direction_sha256"]),
        expected_lens_sha256=str(config["lens_sha256"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
