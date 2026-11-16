from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import tempfile
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactEnvelope:
    experiment: str
    config: dict[str, Any]
    results: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: int = 1


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_artifact(path: Path, artifact: ArtifactEnvelope) -> None:
    atomic_write_json(path, asdict(artifact))
