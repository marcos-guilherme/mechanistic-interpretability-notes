import json

from refusal_direction_replication.artifacts import (
    ArtifactEnvelope,
    write_artifact,
)


def test_write_artifact_is_sorted_and_terminated(tmp_path) -> None:
    artifact = ArtifactEnvelope(
        experiment="unit",
        config={"seed": 17},
        results={"value": 1.0},
        provenance={"upstream_commit": "abc"},
    )
    path = tmp_path / "artifact.json"
    write_artifact(path, artifact)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["experiment"] == "unit"
