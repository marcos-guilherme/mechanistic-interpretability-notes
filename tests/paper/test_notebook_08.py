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
    assert "build_balanced_paired_dataset" in source
    assert "compensação" in source
    assert "contribuição basal" in source
    assert "08_backup_name_movers.json" in source
    assert "N_TARGET = 16" in source or "N_TARGET = 32" in source
    assert "EXAMPLES_PER_REFERENCE_GROUP = 10" in source
    assert "BATCH_SIZE_REFERENCE" in source
    assert "delta_result_fixed_scale" in source
    assert "delta_scale" in source
    assert "END→S2" in source
    assert "DLA basal primárias − DLA das substituições pABC" in source


def test_notebook_08_artifact_has_reproducibility_and_uncertainty() -> None:
    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    config = artifact["config"]

    assert config["n_target"] >= 16
    assert config["n_reference"] >= 300
    assert len(config["target_dataset_sha256"]) == 64
    assert len(config["reference_dataset_sha256"]) == 64
    assert config["model_name"] == "gpt2"
    assert config["dtype"] in {"float16", "float32", "bfloat16"}
    assert config["device"] in {"cuda", "cpu"}
    assert "gpu" in config
    reference = artifact["reference"]
    assert len(reference["group_counts"]) == 30
    assert min(reference["group_counts"].values()) >= 5
    traceability = artifact["traceability"]
    assert len(traceability["source_commit"]) == 40
    assert len(traceability["source_sha256"]) >= 3
    assert all(len(value) == 64 for value in traceability["source_sha256"].values())
    assert traceability["torch_version"]
    assert traceability["transformer_lens_version"]
    assert traceability["timestamp_utc"].endswith("Z")
    assert set(artifact["logit_difference"]) == {
        "baseline",
        "primary_name_mover_knockout",
        "delta",
    }
    assert len(artifact["backup_heads"]) == 8
    assert len(artifact["ranking_by_delta_result_fixed_scale"]) == 6
    assert all(
        not row["head"].startswith("9.")
        for row in artifact["ranking_by_delta_result_fixed_scale"]
    )
    assert len(artifact["removed_primary_name_movers"]) == 3
    assert len(artifact["primary_name_mover_substitutions"]) == 3
    assert len(artifact["effectively_removed_primary_contribution"]) == 3
    assert "compensation_fraction" in artifact
    assert (
        "baseline primary DLA minus pABC substitution DLA"
        in artifact["compensation_fraction"]["definition"]
    )
    for metrics in artifact["backup_heads"].values():
        assert "baseline_dla" in metrics
        assert "delta_dla" in metrics
        assert "standard_error" in metrics["delta_dla"]
        assert "delta_result_fixed_scale" in metrics
        assert "delta_scale" in metrics
        assert "baseline_attention_io" in metrics
        assert "delta_attention_io" in metrics
        assert "baseline_attention_s1" in metrics
        assert "baseline_attention_s2" in metrics
