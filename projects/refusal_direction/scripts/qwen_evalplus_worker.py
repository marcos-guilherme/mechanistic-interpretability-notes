"""Evaluate selected HumanEval+ solutions in a resource-limited subprocess.

EvalPlus applies ``reliability_guard`` and per-test time limits inside another
process. Its own documentation is explicit that this is not a complete
security sandbox, so this worker must only run inside the ephemeral Colab VM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ONE_GIB = 1024**3


def task_ids_sha256(task_ids: list[str]) -> str:
    encoded = json.dumps(sorted(task_ids), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_identity(
    identity: dict[str, Any],
    observed_task_ids: set[str],
) -> dict[str, Any]:
    required = {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "source_artifact_hashes",
        "condition",
        "expected_task_ids",
        "expected_task_ids_sha256",
    }
    if not required.issubset(identity):
        raise ValueError("EvalPlus identity is incomplete")
    expected = [str(item) for item in identity["expected_task_ids"]]
    if len(expected) != len(set(expected)) or set(expected) != observed_task_ids:
        raise ValueError("EvalPlus identity task IDs do not match selected problems")
    if identity["expected_task_ids_sha256"] != task_ids_sha256(expected):
        raise ValueError("EvalPlus identity task IDs hash mismatch")
    return identity


def read_unique_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record["task_id"])
        if task_id in records:
            raise ValueError(f"duplicate task_id: {task_id}")
        records[task_id] = record
    return records


def load_aligned(
    problems_path: Path,
    samples_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    problems = read_unique_jsonl(problems_path)
    samples = read_unique_jsonl(samples_path)
    if set(problems) != set(samples):
        raise ValueError("EvalPlus requires exactly one solution per selected task")
    return problems, samples


def require_memory_limit(value: int) -> int:
    if int(value) != ONE_GIB:
        raise ValueError("EvalPlus memory limit must be exactly 1 GiB")
    return ONE_GIB


def apply_process_memory_limit(memory_bytes: int) -> None:
    """Apply an inherited address-space limit on Linux before importing EvalPlus."""

    require_memory_limit(memory_bytes)
    if os.name != "posix":
        raise RuntimeError("the limited EvalPlus worker is only supported on Linux")
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def evaluate_selected(
    *,
    problems_path: Path,
    samples_path: Path,
    output_path: Path,
    identity_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval import PASS
    from evalplus.eval.utils import reliability_guard
    from evalplus.sanitize import sanitize

    if not callable(reliability_guard):
        raise RuntimeError("EvalPlus reliability_guard is unavailable")
    problems, samples = load_aligned(problems_path, samples_path)
    identity = validate_identity(
        json.loads(identity_path.read_text(encoding="utf-8")),
        set(problems),
    )
    corpus_hash = hashlib.sha256(problems_path.read_bytes()).hexdigest()
    expected = get_groundtruth(problems, corpus_hash, [])
    records = []
    for task_id in sorted(problems):
        problem = problems[task_id]
        raw_solution = str(samples[task_id]["solution"])
        solution = sanitize(raw_solution, entrypoint=str(problem["entry_point"]))
        result = check_correctness(
            "humaneval",
            0,
            problem,
            solution,
            expected[task_id],
            base_only=False,
            fast_check=True,
            identifier=task_id,
            min_time_limit=1.0,
            gt_time_limit_factor=float(timeout_seconds),
        )
        base_status = result["base"][0]
        plus_status = result["plus"][0]
        records.append(
            {
                "task_id": task_id,
                "base_status": base_status,
                "plus_status": plus_status,
                "base_pass": base_status == PASS,
                "plus_pass": base_status == plus_status == PASS,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": "humaneval_plus",
        "identity": identity,
        "records": records,
        "n": len(records),
        "pass_count": sum(record["plus_pass"] for record in records),
        "pass_at_1": (
            sum(record["plus_pass"] for record in records) / len(records)
            if records
            else None
        ),
        "reliability_guard": True,
        "security_limitation": "EvalPlus is not a complete security sandbox",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    apply_process_memory_limit(args.memory_bytes)
    evaluate_selected(
        problems_path=args.problems,
        samples_path=args.samples,
        output_path=args.output,
        identity_path=args.identity,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
