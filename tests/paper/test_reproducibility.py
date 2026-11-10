from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]
RESULTS = ROOT / "results" / "paper"
SUMMARY = RESULTS / "summary.json"
RUNNER = ROOT / "scripts" / "run_paper_notebooks.py"
SUMMARY_BUILDER = ROOT / "scripts" / "build_paper_summary.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_paper_notebooks",
    RUNNER,
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)
finalize_manifest = RUNNER_MODULE.finalize_manifest


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_summary_schema_and_values_are_derived_from_artifacts() -> None:
    summary = _load("summary.json")
    baseline = _load("baseline_seed_42_n_256.json")
    validation = _load("09_faithfulness_completeness_minimality.json")
    adversarial = _load("10_adversarial_examples.json")
    path_patching = _load("07_path_patching.json")

    assert summary["schema_version"] == 1
    assert set(summary["sources"]) == {
        "baseline",
        "circuit",
        "path_patching",
        "backup",
        "validation",
        "adversarial",
    }
    assert all(
        (ROOT / source["path"]).is_file()
        for source in summary["sources"].values()
    )
    assert summary["sources"]["circuit"]["classification"] == (
        "diagnostic-validation"
    )
    assert summary["sources"]["path_patching"]["classification"] == "path-grid"
    assert summary["key_results"]["baseline"] == baseline["observed"]
    assert summary["key_results"]["faithfulness"] == {
        "model": validation["faithfulness"]["model"]["mean"],
        "circuit": validation["faithfulness"]["circuit"]["mean"],
        "absolute_gap": validation["faithfulness"]["absolute_gap"],
        "retained_fraction": validation["faithfulness"]["retained_fraction"],
    }
    assert summary["key_results"]["completeness_greedy_max"] == max(
        item["incompleteness"]
        for item in validation["completeness"]["greedy_top5"]
    )
    assert summary["key_results"]["minimality"] == {
        "point_above_1_percent": sum(
            item["passes_threshold"] for item in validation["minimality"]
        ),
        "ci95_above_1_percent": sum(
            item["ci_supports_threshold"] for item in validation["minimality"]
        ),
        "total": len(validation["minimality"]),
    }
    assert summary["key_results"]["adversarial"][
        "paper_io_single_logit_delta"
    ] == adversarial["protocols"]["paper_io_single"]["delta_from_baseline"][
        "logit_difference_io_minus_s"
    ]["mean"]
    path_summary = summary["key_results"]["path_patching"]
    assert path_summary["primary_method"] == "historical_four_pass_semantic"
    assert path_summary["historical_four_pass_semantic"] == {
        mode: {
            name: values["effect"]
            for name, values in mode_values.items()
        }
        for mode, mode_values in path_patching[
            "historical_four_pass_semantic"
        ].items()
    }
    assert path_summary["transformerlens_direct_path_all_positions"] == {
        name: values["effect"]
        for name, values in path_patching[
            "transformerlens_direct_path_all_positions"
        ].items()
    }
    assert path_summary["published_path_grid"]["n_edges"] == len(
        path_patching["published_path_grid"]["rows"]
    )


def test_summary_builder_check_detects_no_drift() -> None:
    completed = subprocess.run(
        [sys.executable, str(SUMMARY_BUILDER), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_notebook_runner_has_safe_order_timeout_and_fail_fast_manifest() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "05_paper_dataset_and_baseline.ipynb" in source
    assert "10_adversarial_examples.ipynb" in source
    assert "check=False" in source
    assert "break" in source
    assert "ExecutePreprocessor.timeout" in source
    assert "ioi-replication" in source
    assert "execution_manifest.json" in source
    assert "IOI_SOURCE_COMMIT" in source

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runner_only_marks_success_after_summary_succeeds() -> None:
    failed_manifest = {"status": "running"}
    assert finalize_manifest(
        failed_manifest,
        summary_return_code=3,
        finished_at_utc="2026-01-01T00:00:00Z",
    ) == 3
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["summary"] == {
        "exit_status": "failed",
        "return_code": 3,
    }

    successful_manifest = {"status": "running"}
    assert finalize_manifest(
        successful_manifest,
        summary_return_code=0,
        finished_at_utc="2026-01-01T00:00:00Z",
    ) == 0
    assert successful_manifest["status"] == "success"
