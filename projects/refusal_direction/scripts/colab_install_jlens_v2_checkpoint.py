"""Install the frozen v1 lens and an optional v2 resume bundle on Colab.

The v1 lens arrives as v2-named chunks plus a manifest. It is assembled and
SHA-256 verified in staging, then copied next to its source path and verified
again before a single ``os.replace``; the v1 path is never written earlier.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    MAX_LAB_LENS_BYTES,
    REMOTE_JLENS_V2_CHECKPOINT_STAGING,
    REMOTE_JLENS_V2_EVENTS_STAGING,
    REMOTE_JLENS_V2_STATUS_STAGING,
    assemble_chunked_file,
    atomic_install_jlens_v2_resume,
    path_is_link_or_junction,
    sha256_file,
)
from refusal_direction_replication.qwen_lab_readout import FROZEN_LENS_SHA256

REMOTE_V1_LENS_STAGING = Path("/content/jlens_v2_source_v1_lens.pt")
REMOTE_V1_LENS_MANIFEST = Path("/content/jlens_v2_source_v1_lens.pt.manifest.json")
V1_LENS_RELATIVE = Path("artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt")


def clean_lens_staging(staged: Path, manifest: Path) -> None:
    for path in [*staged.parent.glob(staged.name + ".part*"), manifest, staged]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    assembling = staged.with_name(staged.name + ".assembling")
    if assembling.exists():
        assembling.unlink()


def _require_lens(path: Path, expected_sha256: str, *, origin: str) -> None:
    if path_is_link_or_junction(path) or not path.is_file():
        raise ValueError(f"v1 lens {origin} must be a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_LAB_LENS_BYTES:
        raise ValueError(f"v1 lens {origin} size {size} exceeds {MAX_LAB_LENS_BYTES} bytes")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"v1 lens {origin} sha256 {actual!r} != expected {expected_sha256!r}")


def install_frozen_v1_lens(
    *,
    project: Path,
    lens_staging: Path,
    lens_manifest: Path,
    expected_sha256: str = FROZEN_LENS_SHA256,
    relative: Path = V1_LENS_RELATIVE,
) -> Path:
    destination = project / relative
    staged_parts = list(lens_staging.parent.glob(lens_staging.name + ".part*"))
    if not staged_parts and not os.path.lexists(lens_manifest) and destination.is_file():
        _require_lens(destination, expected_sha256, origin="installed copy")
        return destination
    try:
        assembled = assemble_chunked_file(
            lens_staging, manifest_path=lens_manifest, require_manifest=True
        )
        _require_lens(assembled, expected_sha256, origin="staging")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination) and path_is_link_or_junction(destination):
            raise ValueError(f"v1 lens destination is a link or junction: {destination}")
        fd, tmp_name = tempfile.mkstemp(
            prefix=".jlens_v2_v1_lens.", suffix=".tmp", dir=destination.parent
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                with assembled.open("rb") as incoming:
                    shutil.copyfileobj(incoming, handle)
                handle.flush()
                os.fsync(handle.fileno())
            _require_lens(tmp_path, expected_sha256, origin="temporary copy")
            os.replace(tmp_path, destination)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return destination
    finally:
        clean_lens_staging(lens_staging, lens_manifest)


def install_v2_sources(
    *,
    project: Path,
    lens_staging: Path,
    lens_manifest: Path,
    checkpoint: Path,
    status: Path,
    events: Path,
    expected_lens_sha256: str = FROZEN_LENS_SHA256,
) -> Path:
    lens = install_frozen_v1_lens(
        project=project,
        lens_staging=lens_staging,
        lens_manifest=lens_manifest,
        expected_sha256=expected_lens_sha256,
    )
    atomic_install_jlens_v2_resume(
        checkpoint=checkpoint,
        status=status,
        events=events,
        project=project,
        missing_ok=True,
    )
    return lens


def main() -> int:
    from refusal_direction_replication.qwen_jlens import JLENS_V2_PROFILE
    from refusal_direction_replication.qwen_jlens_runtime import load_config

    config = load_config(ROOT, profile=JLENS_V2_PROFILE)
    if Path(str(config["source_v1_lens"])) != V1_LENS_RELATIVE:
        raise ValueError("v2 profile source_v1_lens differs from the installer path")
    install_v2_sources(
        project=ROOT,
        lens_staging=REMOTE_V1_LENS_STAGING,
        lens_manifest=REMOTE_V1_LENS_MANIFEST,
        checkpoint=Path(REMOTE_JLENS_V2_CHECKPOINT_STAGING),
        status=Path(REMOTE_JLENS_V2_STATUS_STAGING),
        events=Path(REMOTE_JLENS_V2_EVENTS_STAGING),
    )
    print(f"installed frozen v1 lens at {V1_LENS_RELATIVE.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
