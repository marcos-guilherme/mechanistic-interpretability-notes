import json
from pathlib import Path

from ioi_replication.runtime import compatibility_mode_config


ROOT = Path(__file__).parents[2]
RESULT_PATH = (
    ROOT
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


def test_all_paper_artifacts_record_actual_compatibility_mode() -> None:
    result_paths = (
        RESULT_PATH,
        ROOT / "results/paper/06_circuit_discovery.json",
        ROOT / "results/paper/07_path_patching.json",
        ROOT / "results/paper/08_backup_name_movers.json",
        ROOT / "results/paper/09_faithfulness_completeness_minimality.json",
        ROOT / "results/paper/10_adversarial_examples.json",
    )

    for path in result_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["config"]["transformer_lens_compatibility_mode"] == (
            compatibility_mode_config()
        )
