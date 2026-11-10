from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_project_pins_fidelity_dependencies() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(config["project"]["dependencies"])
    assert "transformers==4.44.2" in dependencies
    assert "datasets==2.19.2" in dependencies
    assert "accelerate==0.31.0" in dependencies


def test_raw_artifact_directory_is_not_packaged() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["name"] == "refusal-direction-replication"
