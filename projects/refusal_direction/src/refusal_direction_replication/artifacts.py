from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactEnvelope:
    experiment: str
    config: dict[str, Any]
    results: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: int = 1


def write_artifact(path: Path, artifact: ArtifactEnvelope) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        asdict(artifact),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="utf-8")
