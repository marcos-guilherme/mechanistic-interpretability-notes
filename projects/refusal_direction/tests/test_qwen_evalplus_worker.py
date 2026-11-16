from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "qwen_evalplus_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("qwen_evalplus_worker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_aligns_exactly_one_solution_per_selected_task(tmp_path: Path) -> None:
    worker = _load_worker()
    problems = tmp_path / "problems.jsonl"
    samples = tmp_path / "samples.jsonl"
    problems.write_text(
        "\n".join(
            json.dumps({"task_id": f"HumanEval/{index}", "prompt": "def f():\n"})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    samples.write_text(
        "\n".join(
            json.dumps({"task_id": f"HumanEval/{index}", "solution": "def f():\n pass"})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    problem_map, sample_map = worker.load_aligned(problems, samples)
    assert set(problem_map) == {"HumanEval/0", "HumanEval/1"}
    assert set(sample_map) == set(problem_map)

    samples.write_text(
        json.dumps({"task_id": "HumanEval/0", "solution": "def f(): pass"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly"):
        worker.load_aligned(problems, samples)


def test_worker_rejects_duplicate_tasks_and_non_gibibyte_limit(tmp_path: Path) -> None:
    worker = _load_worker()
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "HumanEval/0", "prompt": "x"}),
                json.dumps({"task_id": "HumanEval/0", "prompt": "y"}),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        worker.read_unique_jsonl(duplicate)
    with pytest.raises(ValueError, match="1 GiB"):
        worker.require_memory_limit(2 * 1024**3)
    assert worker.require_memory_limit(1024**3) == 1024**3


def test_worker_uses_evalplus_sanitize_check_and_reliability_guard() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from evalplus.sanitize import sanitize" in source
    assert "check_correctness" in source
    assert "reliability_guard" in source
    assert "EvalPlus is not a complete security sandbox" in source


def test_worker_identity_requires_exact_task_ids() -> None:
    worker = _load_worker()
    task_ids = ["HumanEval/0", "HumanEval/1"]
    identity = {
        "run_id": "run",
        "implementation_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "source_artifact_hashes": {"direction": "c" * 64},
        "condition": "addition",
        "expected_task_ids": task_ids,
        "expected_task_ids_sha256": worker.task_ids_sha256(task_ids),
    }
    assert worker.validate_identity(identity, set(task_ids)) == identity
    with pytest.raises(ValueError, match="task IDs"):
        worker.validate_identity(identity, {"HumanEval/0", "HumanEval/2"})
