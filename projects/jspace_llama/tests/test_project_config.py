import hashlib
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]
JSPACE_COMMIT = "04c822a255fe0a10d84e08ec499e5e09f4b7d37a"
DATASET = (
    ROOT
    / "data"
    / "anthropic-jlens"
    / "evaluations"
    / "lens-eval-multihop.json"
)
DATASET_SHA256 = (
    "2cd5a47aa3d7627708843bc61f03ab1d"
    "10eda4184f357c5cbf93c97184e32c9c"
)


def load_config() -> dict:
    return tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )


def test_project_configuration_is_reproducible() -> None:
    config = load_config()
    project = config["project"]
    dependencies = set(project["dependencies"])

    assert project["name"] == "jspace-llama"
    assert project["requires-python"] == ">=3.11,<3.14"
    assert (
        "jspace @ "
        f"git+https://github.com/haohww/jspace@{JSPACE_COMMIT}"
        in dependencies
    )
    assert any(dependency.startswith("torch") for dependency in dependencies)
    assert any(
        dependency.startswith("transformers")
        for dependency in dependencies
    )
    assert any(
        dependency.startswith("huggingface-hub")
        for dependency in dependencies
    )
    assert any(
        dependency.startswith("jsonschema")
        for dependency in dependencies
    )


def test_notebook_and_test_tools_are_development_dependencies() -> None:
    config = load_config()
    runtime = set(config["project"]["dependencies"])
    dev = set(config["dependency-groups"]["dev"])

    assert {"ipykernel>=6.29", "jupyter>=1.1", "pytest>=9.2"} <= dev
    assert {"ipykernel>=6.29", "jupyter>=1.1", "pytest>=9.2"}.isdisjoint(
        runtime
    )


def test_observation_dataset_is_pinned_and_complete() -> None:
    payload = DATASET.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == DATASET_SHA256
    document = json.loads(payload)
    assert len(document["items"]) == 93
    assert {item["name"] for item in document["items"]} >= {
        "spider-legs",
        "amazon-language",
        "nhop-primary-planet",
    }


def test_static_figure_dependency_is_runtime_locked() -> None:
    dependencies = set(load_config()["project"]["dependencies"])
    assert any(item.startswith("matplotlib") for item in dependencies)
