"""Offline validator for Qwen code-evaluation compact and raw artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .gates import STATUS_OPERATIONAL_ERROR, STATUS_PASS
from .metrics import kl_divergence
from .pipeline import load_config
from .qwen_code_eval import (
    CONDITIONS,
    DATASETS,
    DATASET_SIZE,
    EVALPLUS_REVISION,
    JLENS_REVISION,
    PURPLE_LLAMA_REVISION,
    aggregate_records,
    compact_contains_leak,
    summarize_refusal_ranks,
)
from .qwen_code_eval_runtime import (
    CODE_EVAL_STAGES,
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
    STAGE_FILES,
    source_context,
)
from .qwen_dose import sha256_file


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_finite(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(_all_finite(value) for value in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(value) for value in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def _same_metrics(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_metrics(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_metrics(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def validate_qwen_code_eval(root: Path) -> dict[str, bool]:
    results_dir = root / "results" / "qwen25_coder_7b_code_eval"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b_code_eval"
    evaluation_path = results_dir / EVALUATION_FILENAME
    checks = {
        "terminal_status_valid": False,
        "required_files_present": False,
        "complete_32x3x4_matrix": False,
        "metrics_recomputed": False,
        "kl_recomputed": False,
        "jlens_ranks_recomputed": False,
        "pinned_revisions": False,
        "source_dose_and_jlens_pass": False,
        "source_layer_and_offset": False,
        "source_hashes_match": False,
        "frozen_coefficients": False,
        "no_compact_text_leakage": False,
        "humaneval_only_execution": False,
        "evalplus_limits_and_warning": False,
        "jlens_complete_coverage": False,
        "human_review_honest": False,
        "single_run_identity": False,
        "observability_terminal": False,
        "manifest_agrees": False,
        "metrics_finite": False,
    }
    if not evaluation_path.is_file():
        return checks
    try:
        config = load_config(root, profile="qwen25_coder_7b_code_eval.json")
        evaluation = _load(evaluation_path)
    except Exception:
        return checks
    status = evaluation.get("status")
    checks["terminal_status_valid"] = status in {STATUS_PASS, STATUS_OPERATIONAL_ERROR}
    if status == STATUS_OPERATIONAL_ERROR:
        status_path = artifacts_dir / "status.json"
        checks["required_files_present"] = status_path.is_file()
        if status_path.is_file():
            checks["observability_terminal"] = _load(status_path).get("state") == "failed"
        # A terminal operational error intentionally has no scientific matrix.
        for name in (
            "complete_32x3x4_matrix",
            "metrics_recomputed",
            "kl_recomputed",
            "jlens_ranks_recomputed",
            "humaneval_only_execution",
            "evalplus_limits_and_warning",
            "jlens_complete_coverage",
            "human_review_honest",
            "manifest_agrees",
            "metrics_finite",
        ):
            checks[name] = True

    required = [
        *[results_dir / STAGE_FILES[stage] for stage in CODE_EVAL_STAGES],
        results_dir / EVALUATION_FILENAME,
        results_dir / MANIFEST_FILENAME,
        artifacts_dir / "status.json",
        artifacts_dir / "events.jsonl",
    ]
    if status == STATUS_PASS:
        checks["required_files_present"] = all(path.is_file() for path in required)
    compact_paths = sorted(results_dir.glob("*.json"))
    compact_payloads: list[dict[str, Any]] = []
    try:
        compact_payloads = [_load(path) for path in compact_paths]
        checks["no_compact_text_leakage"] = all(
            not compact_contains_leak(payload) for payload in compact_payloads
        )
        checks["metrics_finite"] = all(_all_finite(payload) for payload in compact_payloads)
    except Exception:
        return checks

    checks["pinned_revisions"] = (
        config.get("purplellama_revision") == PURPLE_LLAMA_REVISION
        and config.get("evalplus_revision") == EVALPLUS_REVISION
        and config.get("jlens_upstream_commit") == JLENS_REVISION
    )
    try:
        source = source_context(root, config, load_direction_payload=True)
        checks["source_dose_and_jlens_pass"] = (
            source["source_dose_status"] == STATUS_PASS
            and source["source_jlens_status"] == STATUS_PASS
        )
        checks["source_layer_and_offset"] = (
            source["selected_layer"] == 18 and source["selected_offset"] == -1
        )
        current_hashes = source["source_artifact_hashes"]
        checks["source_hashes_match"] = (
            evaluation.get("source_artifact_hashes") == current_hashes
            and all(
                (payload.get("provenance") or {}).get("source_artifact_hashes")
                == current_hashes
                for payload in compact_payloads
                if payload.get("provenance")
            )
        )
    except Exception:
        current_hashes = {}

    checks["frozen_coefficients"] = (
        float(config.get("ablation_coefficient", -1)) == 0.2
        and float(config.get("addition_coefficient", -1)) == 0.5
        and float(config.get("random_control_coefficient", -1)) == 0.2
    )
    generate_path = results_dir / STAGE_FILES["generate"]
    if generate_path.is_file():
        conditions = (_load(generate_path).get("results") or {}).get("conditions") or {}
        checks["frozen_coefficients"] = checks["frozen_coefficients"] and {
            name: float((conditions.get(name) or {}).get("coefficient", -1))
            for name in CONDITIONS
        } == {
            "baseline": 0.0,
            "ablation": 0.2,
            "addition": 0.5,
            "random_control": 0.2,
        }

    raw_records: list[dict[str, Any]] = []
    raw_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    matrix_ok = True
    for dataset in DATASETS:
        for condition in CONDITIONS:
            path = artifacts_dir / "generations" / f"{dataset}__{condition}.jsonl"
            if not path.is_file():
                matrix_ok = False
                continue
            records = _jsonl(path)
            ids = [str(record.get("example_id")) for record in records]
            if (
                len(records) != DATASET_SIZE
                or len(set(ids)) != DATASET_SIZE
                or any(record.get("dataset") != dataset for record in records)
                or any(record.get("condition") != condition for record in records)
            ):
                matrix_ok = False
            raw_by_pair[(dataset, condition)] = records
            raw_records.extend(records)
    checks["complete_32x3x4_matrix"] = matrix_ok

    generate_envelope = _load(generate_path) if generate_path.is_file() else {}
    expected_config_hash = (generate_envelope.get("provenance") or {}).get(
        "config_sha256"
    )
    kl_ok = matrix_ok
    if matrix_ok:
        for dataset in DATASETS:
            aligned_example_ids = {
                str(record["example_id"])
                for record in raw_by_pair[(dataset, "baseline")]
            }
            logits_by_condition: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                path = artifacts_dir / "logits" / f"{dataset}__{condition}.pt"
                if not path.is_file():
                    kl_ok = False
                    continue
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    kl_ok = False
                    continue
                ids = [str(item) for item in payload.get("example_ids") or []]
                expected_ids = [
                    str(record["example_id"])
                    for record in raw_by_pair[(dataset, condition)]
                ]
                logits = payload.get("logits")
                if (
                    payload.get("run_id") != evaluation.get("run_id")
                    or payload.get("implementation_commit")
                    != evaluation.get("implementation_commit")
                    or payload.get("config_sha256") != expected_config_hash
                    or payload.get("source_artifact_hashes") != current_hashes
                    or payload.get("dataset") != dataset
                    or payload.get("condition") != condition
                    or len(ids) != len(set(ids))
                    or set(ids) != set(expected_ids)
                    or set(expected_ids) != aligned_example_ids
                    or not isinstance(logits, torch.Tensor)
                    or logits.ndim != 2
                    or logits.shape[0] != len(ids)
                ):
                    kl_ok = False
                    continue
                logits_by_condition[condition] = {
                    "by_id": {example_id: index for index, example_id in enumerate(ids)},
                    "logits": logits,
                }
            if set(logits_by_condition) != set(CONDITIONS):
                kl_ok = False
                continue
            baseline = logits_by_condition["baseline"]
            for condition in CONDITIONS:
                records = raw_by_pair[(dataset, condition)]
                if condition == "baseline":
                    if any(record.get("kl_from_baseline") is not None for record in records):
                        kl_ok = False
                    continue
                current = logits_by_condition[condition]
                ids = [str(record["example_id"]) for record in records]
                baseline_rows = torch.stack(
                    [
                        baseline["logits"][baseline["by_id"][example_id]]
                        for example_id in ids
                    ]
                )
                current_rows = torch.stack(
                    [
                        current["logits"][current["by_id"][example_id]]
                        for example_id in ids
                    ]
                )
                recomputed_kl = kl_divergence(baseline_rows, current_rows).tolist()
                if any(
                    record.get("kl_from_baseline") is None
                    or not math.isclose(
                        float(record["kl_from_baseline"]),
                        float(value),
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    )
                    for record, value in zip(records, recomputed_kl)
                ):
                    kl_ok = False
    checks["kl_recomputed"] = kl_ok

    selected_path = artifacts_dir / "datasets" / "humaneval_plus.jsonl"
    task_by_example: dict[str, str] = {}
    if selected_path.is_file():
        selected = _jsonl(selected_path)
        for record in selected:
            task_id = record.get("task_id") or (record.get("problem") or {}).get(
                "task_id"
            )
            if task_id is not None:
                task_by_example[str(record["example_id"])] = str(task_id)
    expected_task_ids = sorted(task_by_example.values())
    expected_task_hash = hashlib.sha256(
        json.dumps(expected_task_ids, separators=(",", ":")).encode()
    ).hexdigest()
    evalplus: dict[str, dict[str, bool]] = {}
    evalplus_ok = True
    for condition in CONDITIONS:
        path = artifacts_dir / "evalplus" / f"results__{condition}.json"
        if not path.is_file():
            evalplus_ok = False
            continue
        payload = _load(path)
        records = payload.get("records") or []
        identity = payload.get("identity") or {}
        ids = [str(record.get("task_id")) for record in records]
        if (
            len(records) != DATASET_SIZE
            or len(ids) != len(set(ids))
            or set(ids) != set(expected_task_ids)
            or identity.get("run_id") != evaluation.get("run_id")
            or identity.get("implementation_commit")
            != evaluation.get("implementation_commit")
            or identity.get("config_sha256") != expected_config_hash
            or identity.get("source_artifact_hashes") != current_hashes
            or identity.get("condition") != condition
            or identity.get("expected_task_ids") != expected_task_ids
            or identity.get("expected_task_ids_sha256") != expected_task_hash
        ):
            evalplus_ok = False
        evalplus[condition] = {
            str(record["task_id"]): bool(record["plus_pass"]) for record in records
        }

    if matrix_ok and evalplus_ok and kl_ok:
        compact_records = []
        for record in raw_records:
            passed = None
            if record["dataset"] == "humaneval_plus":
                condition = str(record["condition"])
                task_id = task_by_example[str(record["example_id"])]
                passed = evalplus[condition][task_id]
            compact_records.append(
                {
                    "dataset": record["dataset"],
                    "condition": record["condition"],
                    "example_id": record["example_id"],
                    "valid": record["valid"],
                    "refusal": record["refusal"],
                    "kl_from_baseline": record["kl_from_baseline"],
                    "evalplus_pass": passed,
                }
            )
        recomputed = aggregate_records(compact_records)
        checks["metrics_recomputed"] = (
            _same_metrics(evaluation.get("counts"), recomputed["counts"])
            and _same_metrics(evaluation.get("metrics"), recomputed["metrics"])
        )

    evaluate_path = results_dir / STAGE_FILES["evaluate"]
    if evaluate_path.is_file():
        block = _load(evaluate_path).get("results") or {}
        checks["humaneval_only_execution"] = (
            block.get("executed_datasets") == ["humaneval_plus"]
            and set(block.get("never_executed_datasets") or [])
            == {"mitre", "mitre_frr"}
        )
        checks["evalplus_limits_and_warning"] = (
            block.get("memory_bytes") == 1024**3
            and block.get("parallelism") == 1
            and block.get("tempdir_isolated") is True
            and block.get("reliability_guard") is True
            and "not_a_complete_security_sandbox"
            in str(block.get("security_limitation"))
        )

    readout_records: list[dict[str, Any]] = []
    coverage_ok = True
    for dataset in DATASETS:
        for condition in CONDITIONS:
            path = artifacts_dir / "readout" / f"{dataset}__{condition}.jsonl"
            if not path.is_file():
                coverage_ok = False
                continue
            records = _jsonl(path)
            instruction_ids = {
                str(record["example_id"])
                for record in records
                if record.get("position_kind") == "instruction_last"
            }
            response_ids = {
                str(record["example_id"])
                for record in records
                if record.get("position_kind") == "response_first"
            }
            if len(instruction_ids) != DATASET_SIZE or len(response_ids) != DATASET_SIZE:
                coverage_ok = False
            readout_records.extend(records)
    checks["jlens_complete_coverage"] = coverage_ok
    readout_stage_path = results_dir / STAGE_FILES["readout"]
    if coverage_ok and readout_stage_path.is_file():
        readout_stage = _load(readout_stage_path).get("results") or {}
        token_ids = tuple(int(item) for item in readout_stage.get("lexicon_token_ids") or [])
        try:
            recomputed_ranks = summarize_refusal_ranks(
                readout_records, lexicon_token_ids=token_ids
            )
            checks["jlens_ranks_recomputed"] = _same_metrics(
                readout_stage.get("aggregates"), recomputed_ranks
            ) and _same_metrics(evaluation.get("jlens"), recomputed_ranks)
        except Exception:
            checks["jlens_ranks_recomputed"] = False

    limitations = set(evaluation.get("scientific_limitations") or [])
    checks["human_review_honest"] = (
        evaluation.get("human_review_status") == "pending"
        and "human_review_pending_until_completed_by_a_real_reviewer" in limitations
    )
    identities = {
        (
            (payload.get("provenance") or {}).get("run_id"),
            (payload.get("provenance") or {}).get("implementation_commit"),
        )
        for payload in compact_payloads
        if payload.get("provenance")
    }
    checks["single_run_identity"] = (
        len(identities) == 1
        and identities
        == {(evaluation.get("run_id"), evaluation.get("implementation_commit"))}
    )

    status_path = artifacts_dir / "status.json"
    events_path = artifacts_dir / "events.jsonl"
    if status_path.is_file() and events_path.is_file():
        status_payload = _load(status_path)
        terminal_event = any(
            json.loads(line).get("event") == "stage_completed"
            and json.loads(line).get("stage") == "finalize"
            and json.loads(line).get("run_id") == evaluation.get("run_id")
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        checks["observability_terminal"] = (
            status_payload.get("state") == "completed"
            and status_payload.get("current_stage") == "finalize"
            and status_payload.get("run_id") == evaluation.get("run_id")
            and terminal_event
        )

    manifest_path = results_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = _load(manifest_path).get("results") or {}
        completed_stages = [
            stage
            for stage in CODE_EVAL_STAGES
            if (results_dir / STAGE_FILES[stage]).is_file()
        ]
        expected_files = [
            *[STAGE_FILES[stage] for stage in completed_stages],
            EVALUATION_FILENAME,
            MANIFEST_FILENAME,
        ]
        checks["manifest_agrees"] = (
            manifest.get("status") == status
            and manifest.get("completed_stages") == completed_stages
            and manifest.get("result_files") == expected_files
            and manifest.get("human_review_status")
            == evaluation.get("human_review_status")
        )
    return checks
