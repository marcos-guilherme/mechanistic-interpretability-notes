"""Remove leftover causal source chunks without reading or printing contents."""

from __future__ import annotations

from pathlib import Path

REMOTE_CAUSAL_STAGING = (
    Path("/content/causal_jlens_source_direction.pt"),
    Path("/content/causal_jlens_source_v1_lens.pt"),
    Path("/content/causal_jlens_source_v2_lens.pt"),
)


def clean_causal_jlens_chunk_files(*destinations: Path) -> None:
    for destination in destinations:
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
    clean_causal_jlens_chunk_files(*REMOTE_CAUSAL_STAGING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
