import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "08_backup_name_movers.ipynb"
RESULT = ROOT / "results" / "paper" / "08_backup_name_movers.json"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_08_runs_primary_knockout_and_separates_compensation() -> None:
    source = _notebook_source()

    assert "cache_grouped_mean_activations" in source
    assert "measure_backup_name_movers" in source
    assert "PUBLISHED_CIRCUIT" in source
    assert "compensação" in source
    assert "contribuição basal" in source
    assert "08_backup_name_movers.json" in source
    assert "N = 16" in source or "N = 32" in source


def test_notebook_08_artifact_has_reproducibility_and_uncertainty() -> None:
    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    config = artifact["config"]

    assert config["n_examples"] >= 16
    assert len(config["dataset_sha256"]) == 64
    assert config["model_name"] == "gpt2"
    assert config["dtype"] in {"float16", "float32", "bfloat16"}
    assert config["device"] in {"cuda", "cpu"}
    assert "gpu" in config
    assert set(artifact["logit_difference"]) == {
        "baseline",
        "primary_name_mover_knockout",
        "delta",
    }
    assert len(artifact["backup_heads"]) == 8
    assert len(artifact["ranking_by_delta_dla"]) == 8
    assert len(artifact["ranking_by_delta_attention_io_minus_s"]) == 8
    for metrics in artifact["backup_heads"].values():
        assert "baseline_dla" in metrics
        assert "delta_dla" in metrics
        assert "standard_error" in metrics["delta_dla"]
        assert "baseline_attention_io" in metrics
        assert "delta_attention_io" in metrics
