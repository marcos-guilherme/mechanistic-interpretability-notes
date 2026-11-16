"""Remove leftover lab lens chunks without reading or printing contents."""

from __future__ import annotations

from pathlib import Path

REMOTE_LAB_LENS_STAGING = Path("/content/lab_source_lens.pt")


def clean_lab_chunk_files(destination: Path) -> None:
    parent = destination.parent
    stem = destination.name
    for path in list(parent.glob(stem + ".part*")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for suffix in (".assembling", ".manifest.json"):
        extra = destination.with_name(stem + suffix)
        try:
            extra.unlink()
        except FileNotFoundError:
            pass
    try:
        destination.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    clean_lab_chunk_files(REMOTE_LAB_LENS_STAGING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
