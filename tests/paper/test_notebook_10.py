import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "10_adversarial_examples.ipynb"
RESULT = ROOT / "results" / "paper" / "10_adversarial_examples.json"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_10_runs_matched_historical_adversarial_experiment() -> None:
    source = _notebook_source()

    assert "N = 500" in source
    assert "BATCH_SIZE = 16" in source
    assert "build_adversarial_dataset" in source
    assert "summarize_adversarial_values" in source
    assert "DOUBLE_ADVERSARIAL_TEMPLATES" in source
    assert "extra_io_positions" in source
    assert "adversarial_positions" in source
    assert "IO−S" in source
    assert "S>IO" in source
    assert "1.23" in source
    assert "23.4%" in source
    assert "não prova o circuito" in source
    assert "Duplicate Token" in source
    assert "S-Inhibition" in source
    assert "10_adversarial_examples.json" in source


def test_notebook_10_artifact_is_traceable_paired_and_finite() -> None:
    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    config = artifact["config"]

    assert config["seed"] == 510
    assert config["n"] == 500
    assert config["batch_size"] == 16
    assert config["model_name"] == "gpt2"
    assert config["device"] in {"cuda", "cpu"}
    assert config["dtype"] in {"float16", "float32", "bfloat16"}
    assert len(config["dataset_sha256"]) == 64
    assert config["prepend_bos"] is False
    assert config["padding_side"] == "right"

    metrics = artifact["metrics"]
    assert set(metrics) == {
        "logit_difference_io_minus_s",
        "s_wins_error_rate",
        "io_probability",
    }
    for metric in metrics.values():
        assert set(metric) == {"baseline", "adversarial", "delta"}
        for estimate in metric.values():
            assert math.isfinite(estimate["mean"])
            assert estimate["standard_error"] >= 0
            assert len(estimate["ci95"]) == 2
            assert all(math.isfinite(value) for value in estimate["ci95"])
    for arm in ("baseline", "adversarial"):
        assert 0 <= metrics["s_wins_error_rate"][arm]["mean"] <= 1
        assert 0 <= metrics["io_probability"][arm]["mean"] <= 1

    assert artifact["paper_targets"] == {
        "adversarial_logit_difference": 1.23,
        "adversarial_s_wins_error_rate": 0.234,
    }
    assert set(artifact["observed_minus_paper"]) == {
        "adversarial_logit_difference",
        "adversarial_s_wins_error_rate",
    }
    assert artifact["methodology"]["comparison_scope"].startswith(
        "Same five clauses, ordered-pair sampling"
    )
    assert "seeded" in artifact["methodology"]["differences"][0]

    traceability = artifact["traceability"]
    assert len(traceability["source_commit"]) == 40
    assert traceability["git_dirty"] is False
    assert traceability["timestamp_utc"].endswith("Z")
    assert traceability["historical_source_commit"] == (
        "ea15315dd24481e9e2ac5c3ef335d82907a1dc34"
    )
    assert traceability["historical_source_url"].endswith(
        "/easy_transformer/advex.py"
    )
    assert all(
        len(value) == 64
        for value in traceability["source_sha256"].values()
    )
    assert set(traceability["source_sha256"]) >= {
        "paper/adversarial.py",
        "paper/dataset.py",
        "paper/metrics.py",
        "runtime.py",
        "notebook",
    }
    assert traceability["versions"]["torch"]
    assert traceability["versions"]["transformer_lens"]
