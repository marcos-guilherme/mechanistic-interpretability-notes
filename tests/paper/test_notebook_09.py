import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "09_faithfulness_completeness_minimality.ipynb"
RESULT = (
    ROOT
    / "results"
    / "paper"
    / "09_faithfulness_completeness_minimality.json"
)


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_09_runs_all_paper_validation_definitions() -> None:
    source = _notebook_source()

    assert "N_TARGET = 24" in source
    assert "EXAMPLES_PER_REFERENCE_GROUP = 10" in source
    assert "RANDOM_SUBSETS = 10" in source
    assert "GREEDY_STEPS = 10" in source
    assert "GREEDY_CANDIDATES = 10" in source
    assert "evaluate_faithfulness" in source
    assert "evaluate_completeness_study" in source
    assert "evaluate_all_minimality" in source
    assert "positions=reference.positions" in source
    assert "PAPER_MODEL_PERFORMANCE" in source
    assert "PAPER_FAITHFULNESS_GAP" in source
    assert "PAPER_RETAINED_FRACTION" in source
    assert "add_hline(y=1.0" in source
    assert "09_faithfulness_completeness_minimality.json" in source


def test_notebook_09_artifact_is_complete_traceable_and_finite() -> None:
    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    config = artifact["config"]

    assert config["n_target"] >= 24
    assert config["n_reference"] >= 300
    assert config["reference_min_group_size"] >= 10
    assert config["random_subsets"] == 10
    assert config["greedy_steps"] == 10
    assert config["greedy_candidate_limit"] == 10
    assert len(config["target_dataset_sha256"]) == 64
    assert len(config["reference_dataset_sha256"]) == 64
    assert len(artifact["reference"]["group_counts"]) == 30
    assert min(artifact["reference"]["group_counts"].values()) >= 10

    faithfulness = artifact["faithfulness"]
    assert math.isfinite(faithfulness["model"]["mean"])
    assert math.isfinite(faithfulness["circuit"]["mean"])
    assert math.isfinite(faithfulness["absolute_gap"])
    assert math.isfinite(faithfulness["retained_fraction"])
    assert faithfulness["paper_targets"] == {
        "model_performance": 3.56,
        "absolute_gap": 0.46,
        "retained_fraction": 0.87,
    }

    completeness = artifact["completeness"]
    assert len(completeness["classes"]) == 7
    assert len(completeness["random"]) == 10
    assert len(completeness["greedy"]) == 1
    assert len(completeness["greedy"][0]["curve"]) == 10
    all_completeness = [
        completeness["empty"],
        *completeness["classes"],
        *completeness["random"],
        *completeness["greedy"][0]["curve"],
    ]
    assert all(
        math.isfinite(item["incompleteness"])
        for item in all_completeness
    )
    assert all("removed" in item for item in all_completeness)

    minimality = artifact["minimality"]
    assert len(minimality) == 26
    assert len({item["node"] for item in minimality}) == 26
    assert all("context" in item for item in minimality)
    assert all(math.isfinite(item["raw_effect"]) for item in minimality)
    assert all(math.isfinite(item["normalized_effect"]) for item in minimality)
    assert all(item["threshold_fraction"] == 0.01 for item in minimality)

    traceability = artifact["traceability"]
    assert len(traceability["source_commit"]) == 40
    assert traceability["git_dirty"] in {True, False}
    assert all(
        len(value) == 64
        for value in traceability["source_sha256"].values()
    )
    assert traceability["timestamp_utc"].endswith("Z")
