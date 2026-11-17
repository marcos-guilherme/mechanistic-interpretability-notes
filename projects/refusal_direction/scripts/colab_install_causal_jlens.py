"""Install pinned causal-jlens dependencies and the three frozen source tensors.

Direction, v1 lens and v2 lens arrive as named chunks plus manifests. Each is
assembled and hash/size verified in staging before a single rollback-safe
install of the triple. Jacobian-lens is installed with pip --no-deps so the
project Transformers pins stay. PurpleLlama is checked out at the pinned
commit used to rehydrate the frozen MITRE cohort.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    MAX_CAUSAL_JLENS_DIRECTION_BYTES,
    MAX_CAUSAL_JLENS_LENS_BYTES,
    REMOTE_CAUSAL_JLENS_DIRECTION_STAGING,
    REMOTE_CAUSAL_JLENS_V1_LENS_STAGING,
    REMOTE_CAUSAL_JLENS_V2_LENS_STAGING,
    assemble_chunked_file,
    atomic_install_causal_jlens_sources,
    path_is_link_or_junction,
    sha256_file,
)
from refusal_direction_replication.qwen_jlens import (
    JLENS_UPSTREAM_SPEC,
    assert_jlens_contract,
)

JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
PURPLE_LLAMA_COMMIT = "0654501399e7cbd560db9501965dc09c567a5563"
SOURCE_ROOT = Path("/content/qwen_code_eval_sources")


def _install(spec: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", spec],
        check=True,
    )


def _install_purplellama_source() -> None:
    destination = SOURCE_ROOT / "PurpleLlama"
    if not (destination / ".git").is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=destination, check=True)
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/meta-llama/PurpleLlama.git",
            ],
            cwd=destination,
            check=True,
        )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", PURPLE_LLAMA_COMMIT],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=destination,
        check=True,
    )
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != PURPLE_LLAMA_COMMIT:
        raise ValueError("PurpleLlama checkout differs from the pinned commit")


def clean_source_staging(staged: Path, manifest: Path) -> None:
    for path in [*staged.parent.glob(staged.name + ".part*"), manifest, staged]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    assembling = staged.with_name(staged.name + ".assembling")
    if assembling.exists():
        assembling.unlink()


def _require_source(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None,
    origin: str,
) -> None:
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(f"causal-jlens {label} {origin} must be a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(
            f"causal-jlens {label} {origin} size {size} exceeds {max_bytes} bytes"
        )
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"causal-jlens {label} {origin} sha256 {actual!r} != expected {expected_sha256!r}"
            )


def _assemble_source(
    staged: Path,
    manifest: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None,
) -> Path:
    assembled = assemble_chunked_file(
        staged, manifest_path=manifest, require_manifest=True
    )
    _require_source(
        assembled,
        label=label,
        max_bytes=max_bytes,
        expected_sha256=expected_sha256,
        origin="staging",
    )
    return assembled


def install_causal_jlens_sources(
    *,
    project: Path,
    direction_staging: Path,
    direction_manifest: Path,
    v1_staging: Path,
    v1_manifest: Path,
    v2_staging: Path,
    v2_manifest: Path,
    expected_direction_sha256: str | None = None,
    expected_v1_sha256: str | None = None,
    expected_v2_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    try:
        staged_direction = _assemble_source(
            direction_staging,
            direction_manifest,
            label="direction",
            max_bytes=MAX_CAUSAL_JLENS_DIRECTION_BYTES,
            expected_sha256=expected_direction_sha256,
        )
        staged_v1 = _assemble_source(
            v1_staging,
            v1_manifest,
            label="v1 lens",
            max_bytes=MAX_CAUSAL_JLENS_LENS_BYTES,
            expected_sha256=expected_v1_sha256,
        )
        staged_v2 = _assemble_source(
            v2_staging,
            v2_manifest,
            label="v2 lens",
            max_bytes=MAX_CAUSAL_JLENS_LENS_BYTES,
            expected_sha256=expected_v2_sha256,
        )
        return atomic_install_causal_jlens_sources(
            staged_direction=staged_direction,
            staged_v1_lens=staged_v1,
            staged_v2_lens=staged_v2,
            project=project,
            expected_direction_sha256=expected_direction_sha256,
            expected_v1_sha256=expected_v1_sha256,
            expected_v2_sha256=expected_v2_sha256,
        )
    finally:
        clean_source_staging(direction_staging, direction_manifest)
        clean_source_staging(v1_staging, v1_manifest)
        clean_source_staging(v2_staging, v2_manifest)


def main() -> int:
    if JLENS_UPSTREAM_SPEC.rsplit("@", 1)[-1] != JLENS_COMMIT:
        raise ValueError("Jacobian-lens install spec differs from the pinned commit")
    _install(JLENS_UPSTREAM_SPEC)
    _install_purplellama_source()
    assert_jlens_contract()
    install_causal_jlens_sources(
        project=REMOTE_ROOT,
        direction_staging=Path(REMOTE_CAUSAL_JLENS_DIRECTION_STAGING),
        direction_manifest=Path(REMOTE_CAUSAL_JLENS_DIRECTION_STAGING + ".manifest.json"),
        v1_staging=Path(REMOTE_CAUSAL_JLENS_V1_LENS_STAGING),
        v1_manifest=Path(REMOTE_CAUSAL_JLENS_V1_LENS_STAGING + ".manifest.json"),
        v2_staging=Path(REMOTE_CAUSAL_JLENS_V2_LENS_STAGING),
        v2_manifest=Path(REMOTE_CAUSAL_JLENS_V2_LENS_STAGING + ".manifest.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
