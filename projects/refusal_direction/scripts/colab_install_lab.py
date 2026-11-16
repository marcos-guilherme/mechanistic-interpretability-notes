"""Install pinned lab dependencies and frozen direction/J-lens artifacts."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tarfile
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
ARCHIVE = Path("/content/refusal_direction.tar.gz")
DESTINATION = Path("/content")
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


def safe_extract() -> None:
    if not ARCHIVE.is_file():
        if not REMOTE_ROOT.is_dir():
            raise FileNotFoundError("lab source archive is missing")
        return
    with tarfile.open(ARCHIVE, "r:gz") as handle:
        root = DESTINATION.resolve()
        for member in handle.getmembers():
            target = (DESTINATION / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(DESTINATION)


def assemble_staged_lens() -> Path:
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from refusal_direction_replication.qwen_colab_transport import (
        assemble_chunked_file,
    )

    return assemble_chunked_file(Path(REMOTE_LAB_LENS_STAGING))


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
        assert_jlens_contract,
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
    _install(GRADIO_SPEC)
    assert_jlens_contract()
    _assert_gradio_pin()
    config = load_lab_profile_config(REMOTE_ROOT)
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
