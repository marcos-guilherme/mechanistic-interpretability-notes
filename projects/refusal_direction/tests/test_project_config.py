from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_project_pins_fidelity_dependencies() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(config["project"]["dependencies"])
    assert (
        "transformers==4.44.2; python_version < '3.13'"
        in dependencies
    )
    assert (
        "transformers==4.48.3; python_version >= '3.13'"
        in dependencies
    )
    assert "datasets==2.19.2" in dependencies
    assert "accelerate==0.31.0" in dependencies


def test_raw_artifact_directory_is_not_packaged() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["name"] == "refusal-direction-replication"


def test_project_supports_current_colab_python() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["requires-python"] == ">=3.11,<3.14"


def test_notebook_tools_do_not_replace_colab_kernel_packages() -> None:
    config = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = set(config["project"]["dependencies"])
    dev = set(config["dependency-groups"]["dev"])
    assert "ipykernel>=6.29" not in dependencies
    assert {"ipykernel>=6.29", "jupyter>=1.1"} <= dev
