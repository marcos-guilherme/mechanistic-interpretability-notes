import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "10_adversarial_examples.ipynb"
RESULT = ROOT / "results" / "paper" / "10_adversarial_examples.json"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_10_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{NOTEBOOK.name}:cell-{index}",
                "exec",
            )


def test_notebook_10_runs_matched_historical_adversarial_experiment() -> None:
    source = _notebook_source()

    assert "N = 500" in source
    assert "BATCH_SIZE = 16" in source
    assert "BASE_SEED = 510" in source
    assert "PAPER_ATTACK_SEED = 511" in source
    assert "S_CONTROL_ATTACK_SEED = 513" in source
    assert "ADVEX_ATTACK_SEED = 512" in source
    assert "ALLOWED_TEMPLATE_IDS = tuple(range(7))" in source
    assert "build_adversarial_dataset" in source
    assert "summarize_adversarial_values" in source
    assert "DOUBLE_ADVERSARIAL_TEMPLATES" in source
    assert 'role="IO"' in source
    assert 'role="S"' in source
    assert 'clause_count="single"' in source
    assert 'clause_count="double"' in source
    assert "inserted_positions" in source
    assert "adversarial_positions" in source
    assert "convert_ids_to_tokens" in source
    assert "IO−S" in source
    assert "S>IO" in source
    assert "Duplicate Token" in source
    assert "S-Inhibition" in source
    assert "BOS" in source
    assert "punct" in source
    assert "git_blob" in source
    assert "baseline" in source and "controle S" in source
    assert "hashlib.sha256(path.read_bytes())" not in source
    assert "10_adversarial_examples.json" in source


def test_notebook_10_artifact_is_traceable_paired_and_finite() -> None:
    artifact = json.loads(RESULT.read_text(encoding="utf-8"))
    config = artifact["config"]

    assert config["base_seed"] == 510
    assert config["attack_seeds"] == {
        "paper_io_single": 511,
        "paper_s_control": 513,
        "advex_double_io": 512,
    }
    assert config["n"] == 500
    assert config["batch_size"] == 16
    assert config["model_name"] == "gpt2"
    assert config["device"] in {"cuda", "cpu"}
    assert config["dtype"] in {"float16", "float32", "bfloat16"}
    assert set(config["dataset_sha256"]) == {
        "paper_io_single",
        "paper_s_control",
        "advex_double_io",
    }
    assert all(
        len(value) == 64 for value in config["dataset_sha256"].values()
    )
    assert config["allowed_template_ids"] == list(range(7))
    assert config["mixed_population"] == {
        "ABBA": list(range(7)),
        "BABA": list(range(7)),
    }
    assert config["prepend_bos"] is False
    assert config["padding_side"] == "right"

    assert set(artifact["protocols"]) == {
        "baseline",
        "paper_io_single",
        "paper_s_control",
        "advex_double_io",
    }
    baseline = artifact["protocols"]["baseline"]
    assert baseline["attack"] is None
    assert set(baseline["metrics"]) == {
        "logit_difference_io_minus_s",
        "s_wins_error_rate",
        "io_probability",
    }
    for estimate in baseline["metrics"].values():
        assert math.isfinite(estimate["mean"])
        assert estimate["standard_error"] >= 0
        assert len(estimate["ci95"]) == 2

    expected_attacks = {
        "paper_io_single": ("IO", "single", True),
        "paper_s_control": ("S", "single", True),
        "advex_double_io": ("IO", "double", False),
    }
    for name, (role, clause_count, paper_comparable) in expected_attacks.items():
        protocol = artifact["protocols"][name]
        assert protocol["attack"] == {
            "role": role,
            "clause_count": clause_count,
        }
        assert protocol["paper_figure_8_protocol"] is paper_comparable
        assert set(protocol["metrics"]) == {
            "logit_difference_io_minus_s",
            "s_wins_error_rate",
            "io_probability",
        }
        assert set(protocol["delta_from_baseline"]) == set(
            protocol["metrics"]
        )
        for estimate in [
            *protocol["metrics"].values(),
            *protocol["delta_from_baseline"].values(),
        ]:
            assert math.isfinite(estimate["mean"])
            assert estimate["standard_error"] >= 0
            assert len(estimate["ci95"]) == 2
            assert all(math.isfinite(value) for value in estimate["ci95"])
        error = protocol["metrics"]["s_wins_error_rate"]
        assert error["ci_method"] == "wilson"
        assert 0 <= error["ci95"][0] <= error["ci95"][1] <= 1
        assert protocol["delta_from_baseline"][
            "s_wins_error_rate"
        ]["ci_method"] == "normal_paired_1.96_se"

    assert artifact["paper_figure_8_reference"]["paper_io_single"] == {
        "logit_difference": 1.23,
        "io_probability": 0.36,
        "s_wins_error_rate": 0.234,
    }
    assert set(artifact["comparison_to_paper"]) == {
        "baseline",
        "paper_io_single",
        "paper_s_control",
    }
    assert set(artifact["comparison_to_paper"]["paper_io_single"]) == {
        "logit_difference_io_minus_s",
        "s_wins_error_rate",
        "io_probability",
    }
    assert "advex_double_io" not in artifact["comparison_to_paper"]
    assert artifact["methodology"]["comparison_scope"].startswith(
        "Figure 8 targets are compared with baseline"
    )
    assert "single-token" in " ".join(
        artifact["methodology"]["differences"]
    )
    assert "BOS" in artifact["methodology"]["advex_source_inconsistency"]
    assert "punct" in artifact["methodology"]["advex_source_inconsistency"]
    assert "intentional" in artifact["methodology"]["position_policy"]

    traceability = artifact["traceability"]
    assert len(traceability["source_commit"]) == 40
    assert traceability["timestamp_utc"].endswith("Z")
    assert traceability["historical_source_commit"] == (
        "ea15315dd24481e9e2ac5c3ef335d82907a1dc34"
    )
    assert traceability["historical_source_url"].endswith(
        "/easy_transformer/advex.py"
    )
    assert traceability["hash_method"] == (
        "git rev-parse <source_commit>:<path> (normalized Git blob IDs)"
    )
    assert all(
        len(value) == 40
        for value in traceability["source_blobs"].values()
    )
    assert set(traceability["source_blobs"]) >= {
        "paper/adversarial.py",
        "paper/dataset.py",
        "paper/metrics.py",
        "runtime.py",
        "notebook",
    }
    source_paths = {
        "paper/adversarial.py": "src/ioi_replication/paper/adversarial.py",
        "paper/dataset.py": "src/ioi_replication/paper/dataset.py",
        "paper/metrics.py": "src/ioi_replication/paper/metrics.py",
        "runtime.py": "src/ioi_replication/runtime.py",
        "notebook": "notebooks/10_adversarial_examples.ipynb",
    }
    for name, relative_path in source_paths.items():
        expected = subprocess.check_output(
            [
                "git",
                "rev-parse",
                f"{traceability['source_commit']}:{relative_path}",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        assert traceability["source_blobs"][name] == expected
    assert traceability["versions"]["torch"]
    assert traceability["versions"]["transformer_lens"]
