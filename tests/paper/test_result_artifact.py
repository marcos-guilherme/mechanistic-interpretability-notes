import json
from pathlib import Path


RESULT_PATH = (
    Path(__file__).parents[2]
    / "results"
    / "paper"
    / "baseline_seed_42_n_256.json"
)


def test_baseline_result_contains_reproducibility_metadata() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    config = result["config"]

    assert config["model_name"] == "gpt2"
    assert config["dtype"] in {"float16", "float32", "bfloat16"}
    assert len(config["dataset_sha256"]) == 64
    assert len(config["git_commit"]) == 40
    assert 1 <= config["n_groups"] <= 30
    assert "torch_seed" not in config


def test_baseline_result_reports_standard_errors() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert set(result["standard_error"]) == set(result["observed"])
    assert all(value >= 0 for value in result["standard_error"].values())
