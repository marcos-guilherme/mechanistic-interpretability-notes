from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from refusal_direction_replication.artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from refusal_direction_replication.qwen_code_eval import (
    CONDITIONS,
    DATASETS,
    aggregate_records,
    summarize_refusal_ranks,
)
from refusal_direction_replication.metrics import kl_divergence
from refusal_direction_replication.qwen_code_eval_runtime import (
    CODE_EVAL_STAGES,
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
    STAGE_FILES,
)
from refusal_direction_replication.qwen_dose import sha256_file
from refusal_direction_replication.qwen_jlens import JLENS_UPSTREAM_COMMIT


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture(root: Path) -> None:
    project = Path(__file__).parents[1]
    (root / "configs").mkdir(parents=True)
    (root / "configs/qwen25_coder_7b_code_eval.json").write_text(
        (project / "configs/qwen25_coder_7b_code_eval.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    results = root / "results/qwen25_coder_7b_code_eval"
    artifacts = root / "artifacts/qwen25_coder_7b_code_eval"
    direction = root / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = root / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    dose = root / "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    jlens = root / "results/qwen25_coder_7b_jlens/jlens_evaluation.json"
    for path in (direction, lens, dose, jlens):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layer": 18, "position_offset": -1, "vector": torch.ones(4)},
        direction,
    )
    lens.write_bytes(b"lens")
    atomic_write_json(
        dose,
        {
            "status": "pass",
            "run_id": "dose-run",
            "selected_pair": {"ablation": 0.2, "addition": 0.5},
        },
    )
    atomic_write_json(
        jlens,
        {
            "status": "pass",
            "run_id": "jlens-run",
            "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        },
    )
    source_hashes = {
        "direction": sha256_file(direction),
        "lens": sha256_file(lens),
        "dose_evaluation": sha256_file(dose),
        "jlens_evaluation": sha256_file(jlens),
    }
    provenance = {
        "run_id": "code-eval-run",
        "implementation_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "requested_model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "resolved_model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "purplellama_revision": "0654501399e7cbd560db9501965dc09c567a5563",
        "evalplus_revision": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        "source_dose_run_id": "dose-run",
        "source_jlens_run_id": "jlens-run",
        "source_artifact_hashes": source_hashes,
    }
    config = json.loads(
        (root / "configs/qwen25_coder_7b_code_eval.json").read_text(encoding="utf-8")
    )
    for dataset in DATASETS:
        _write_jsonl(
            artifacts / "datasets" / f"{dataset}.jsonl",
            [
                {
                    "example_id": f"{dataset}:{index}",
                    "task_id": (
                        f"HumanEval/{index}"
                        if dataset == "humaneval_plus"
                        else None
                    ),
                    "problem": (
                        {
                            "task_id": f"HumanEval/{index}",
                            "prompt": f"def f_{index}():\n",
                        }
                        if dataset == "humaneval_plus"
                        else None
                    ),
                }
                for index in range(32)
            ],
        )
    compact_records = []
    all_readout = []
    for dataset in DATASETS:
        baseline_logits = torch.tensor(
            [[index / 10, 0.0, 0.5] for index in range(32)],
            dtype=torch.float64,
        )
        for condition in CONDITIONS:
            offset = {
                "baseline": 0.0,
                "ablation": 0.1,
                "addition": 0.2,
                "random_control": -0.1,
            }[condition]
            logits = baseline_logits.clone()
            logits[:, 1] += offset
            kl_values = (
                [None] * 32
                if condition == "baseline"
                else kl_divergence(baseline_logits, logits).tolist()
            )
            logits_path = artifacts / "logits" / f"{dataset}__{condition}.pt"
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "run_id": "code-eval-run",
                    "implementation_commit": "a" * 40,
                    "config_sha256": "b" * 64,
                    "source_artifact_hashes": source_hashes,
                    "dataset": dataset,
                    "condition": condition,
                    "example_ids": [
                        f"{dataset}:{index}" for index in range(32)
                    ],
                    "logits": logits,
                },
                logits_path,
            )
            raw = []
            for index in range(32):
                example_id = f"{dataset}:{index}"
                record = {
                    "run_id": "code-eval-run",
                    "implementation_commit": "a" * 40,
                    "source_artifact_hashes": source_hashes,
                    "dataset": dataset,
                    "condition": condition,
                    "coefficient": {
                        "baseline": 0.0,
                        "ablation": 0.2,
                        "addition": 0.5,
                        "random_control": 0.2,
                    }[condition],
                    "example_id": example_id,
                    "prompt": f"raw-prompt-{index}",
                    "response": f"raw-answer-{index}",
                    "valid": True,
                    "refusal": index % 4 == 0,
                    "kl_from_baseline": kl_values[index],
                    "first_response_token_id": 1,
                }
                raw.append(record)
                compact_records.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "example_id": example_id,
                        "valid": True,
                        "refusal": index % 4 == 0,
                        "kl_from_baseline": record["kl_from_baseline"],
                        "evalplus_pass": (
                            index % 2 == 0 if dataset == "humaneval_plus" else None
                        ),
                    }
                )
            _write_jsonl(
                artifacts / "generations" / f"{dataset}__{condition}.jsonl", raw
            )
            readout = []
            for index in range(32):
                for position in ("instruction_last", "response_first"):
                    for layer in (0, 1):
                        readout.append(
                            {
                                "dataset": dataset,
                                "condition": condition,
                                "example_id": f"{dataset}:{index}",
                                "position_kind": position,
                                "layer": layer,
                                "available": True,
                                "token_ranks": {"1": index + 1},
                            }
                        )
            _write_jsonl(
                artifacts / "readout" / f"{dataset}__{condition}.jsonl", readout
            )
            all_readout.extend(readout)
    for condition in CONDITIONS:
        expected_task_ids = sorted(f"HumanEval/{index}" for index in range(32))
        atomic_write_json(
            artifacts / "evalplus" / f"results__{condition}.json",
            {
                "identity": {
                    "run_id": "code-eval-run",
                    "implementation_commit": "a" * 40,
                    "config_sha256": "b" * 64,
                    "source_artifact_hashes": source_hashes,
                    "condition": condition,
                    "expected_task_ids": expected_task_ids,
                    "expected_task_ids_sha256": hashlib.sha256(
                        json.dumps(
                            sorted(expected_task_ids), separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                },
                "records": [
                    {
                        "task_id": f"HumanEval/{index}",
                        "plus_pass": index % 2 == 0,
                    }
                    for index in range(32)
                ]
            },
        )
    aggregates = aggregate_records(compact_records)
    stage_results = {
        "prepare": {
            dataset: {
                "selection_size": 32,
                "example_ids": [f"{dataset}:{index}" for index in range(32)],
                "content_hashes": [f"{index:064x}" for index in range(32)],
            }
            for dataset in DATASETS
        },
        "generate": {
            "conditions": {
                "baseline": {"coefficient": 0.0},
                "ablation": {"coefficient": 0.2},
                "addition": {"coefficient": 0.5},
                "random_control": {"coefficient": 0.2},
            },
            "datasets": {dataset: 32 for dataset in DATASETS},
            "selected_layer": 18,
            "selected_offset": -1,
            "random_control_seed": 1042,
            "lexicon_token_ids": [1],
            "lexicon_sha256": "d" * 64,
        },
        "evaluate": {
            "executed_datasets": ["humaneval_plus"],
            "never_executed_datasets": ["mitre", "mitre_frr"],
            "memory_bytes": 1024**3,
            "timeout_seconds_per_task": 60,
            "parallelism": 1,
            "tempdir_isolated": True,
            "reliability_guard": True,
            "security_limitation": "evalplus_reliability_guard_is_not_a_complete_security_sandbox",
        },
        "readout": {
            "lexicon_token_ids": [1],
            "lexicon_sha256": "d" * 64,
            "source_layers": [0, 1],
            "positions": ["instruction_last", "response_first"],
            "aggregates": summarize_refusal_ranks(
                all_readout, lexicon_token_ids=[1]
            ),
        },
        "finalize": {
            "status": "pass",
            "human_review_status": "pending",
            "limitation": "pending_real_human_review",
            "scientific_limitations": [
                "human_review_pending_until_completed_by_a_real_reviewer",
                "evalplus_reliability_guard_is_not_a_complete_security_sandbox",
                "mitre_and_mitre_frr_generations_are_never_executed",
            ],
            **aggregates,
            "jlens": summarize_refusal_ranks(
                all_readout, lexicon_token_ids=[1]
            ),
        },
    }
    results.mkdir(parents=True)
    for stage in CODE_EVAL_STAGES:
        write_artifact(
            results / STAGE_FILES[stage],
            ArtifactEnvelope(
                experiment=f"code_eval_{stage}",
                config=config,
                results=stage_results[stage],
                provenance=provenance,
            ),
        )
    evaluation = {
        "schema_version": 1,
        "run_id": "code-eval-run",
        "implementation_commit": "a" * 40,
        "model_revision": config["model_revision"],
        "source_artifact_hashes": source_hashes,
        **stage_results["finalize"],
    }
    atomic_write_json(results / EVALUATION_FILENAME, evaluation)
    write_artifact(
        results / MANIFEST_FILENAME,
        ArtifactEnvelope(
            experiment="code_eval_manifest",
            config=config,
            results={
                "status": "pass",
                "completed_stages": list(CODE_EVAL_STAGES),
                "result_files": [
                    *[STAGE_FILES[stage] for stage in CODE_EVAL_STAGES],
                    EVALUATION_FILENAME,
                    MANIFEST_FILENAME,
                ],
                "human_review_status": "pending",
            },
            provenance=provenance,
        ),
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifacts / "status.json",
        {
            "profile": "qwen25_coder_7b_code_eval.json",
            "run_id": "code-eval-run",
            "current_stage": "finalize",
            "state": "completed",
            "completed_units": 1,
            "total_units": 1,
        },
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "stage_completed",
                "run_id": "code-eval-run",
                "stage": "finalize",
                "state": "completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_validator_recomputes_complete_pass_run(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    checks = validate_qwen_code_eval(tmp_path)
    assert checks
    assert all(checks.values()), checks


def test_validator_detects_compact_leak_and_incomplete_matrix(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    final_path = tmp_path / "results/qwen25_coder_7b_code_eval/04_finalize.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["results"]["solution"] = "must not be published"
    atomic_write_json(final_path, final)
    generation = (
        tmp_path
        / "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl"
    )
    generation.write_text(
        "\n".join(generation.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )
    checks = validate_qwen_code_eval(tmp_path)
    assert checks["no_compact_text_leakage"] is False
    assert checks["complete_32x3x4_matrix"] is False
    assert checks["metrics_recomputed"] is False


def test_validator_joins_evalplus_by_task_id_despite_record_order(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    for condition in CONDITIONS:
        path = (
            tmp_path
            / f"artifacts/qwen25_coder_7b_code_eval/evalplus/results__{condition}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"].reverse()
        atomic_write_json(path, payload)
    checks = validate_qwen_code_eval(tmp_path)
    assert checks["metrics_recomputed"] is True
    assert checks["kl_recomputed"] is True


def test_validator_detects_tampered_per_example_kl(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    path = (
        tmp_path
        / "artifacts/qwen25_coder_7b_code_eval/generations/"
        "humaneval_plus__addition.jsonl"
    )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    records[7]["kl_from_baseline"] = 99.0
    _write_jsonl(path, records)
    checks = validate_qwen_code_eval(tmp_path)
    assert checks["kl_recomputed"] is False
    assert checks["metrics_recomputed"] is False


def test_validator_treats_misaligned_logit_ids_as_failed_check(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    path = (
        tmp_path
        / "artifacts/qwen25_coder_7b_code_eval/logits/mitre__addition.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["example_ids"][0] = "mitre:not-selected"
    torch.save(payload, path)
    checks = validate_qwen_code_eval(tmp_path)
    assert checks["kl_recomputed"] is False
    assert checks["metrics_recomputed"] is False


def test_validator_accepts_accurate_partial_operational_manifest(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_validate import (
        validate_qwen_code_eval,
    )

    _fixture(tmp_path)
    results = tmp_path / "results/qwen25_coder_7b_code_eval"
    evaluation_path = results / EVALUATION_FILENAME
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.update({"status": "operational_error", "stage": "evaluate"})
    atomic_write_json(evaluation_path, evaluation)
    for stage in ("evaluate", "readout", "finalize"):
        (results / STAGE_FILES[stage]).unlink()
    manifest_path = results / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"].update(
        {
            "status": "operational_error",
            "completed_stages": ["prepare", "generate"],
            "result_files": [
                STAGE_FILES["prepare"],
                STAGE_FILES["generate"],
                EVALUATION_FILENAME,
                MANIFEST_FILENAME,
            ],
        }
    )
    atomic_write_json(manifest_path, manifest)
    status_path = tmp_path / "artifacts/qwen25_coder_7b_code_eval/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"state": "failed", "current_stage": "evaluate"})
    atomic_write_json(status_path, status)
    checks = validate_qwen_code_eval(tmp_path)
    assert checks["manifest_agrees"] is True
