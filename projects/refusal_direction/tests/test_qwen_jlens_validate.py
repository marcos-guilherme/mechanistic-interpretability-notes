from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from refusal_direction_replication.artifacts import ArtifactEnvelope, write_artifact
from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_jlens import (
    JLENS_PROFILE,
    JLENS_UPSTREAM_COMMIT,
    QWEN_REVISION,
    SOURCE_DOSE_RUN_ID,
    summarize_holdout_metrics,
)
from refusal_direction_replication.qwen_jlens_validate import validate_qwen_jlens


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "jlens-run-validate"
COMMIT = "a" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _envelope(experiment: str, config: dict, results: dict) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        experiment=experiment,
        config=config,
        results=results,
        provenance={
            "run_id": RUN_ID,
            "implementation_commit": COMMIT,
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "requested_model_revision": QWEN_REVISION,
            "resolved_model_revision": QWEN_REVISION,
            "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
            "dataset_repo_id": config["dataset_repo_id"],
            "dataset_config": config["dataset_config"],
            "dataset_split": config["dataset_split"],
            "dataset_revision": config["dataset_revision"],
            "source_dose_run_id": SOURCE_DOSE_RUN_ID,
        },
    )


def _fit_holdout_meta() -> tuple[list[dict], list[dict]]:
    fit = [
        {
            "example_id": f"fit-{index:03d}",
            "content_hash": f"{index:064x}"[:64],
            "token_count": 128 + index,
            "source_index": index,
        }
        for index in range(100)
    ]
    holdout = [
        {
            "example_id": f"hold-{index:03d}",
            "content_hash": f"{1000 + index:064x}"[:64],
            "token_count": 130,
            "source_index": 200 + index,
        }
        for index in range(20)
    ]
    return fit, holdout


def write_jlens_tree(root: Path, *, tiny: bool = True) -> Path:
    import shutil

    shutil.copytree(ROOT / "configs", root / "configs")
    config = load_config(root, profile=JLENS_PROFILE)
    if tiny:
        config = dict(config)
        config["n_layers"] = 4
        config["d_model"] = 8
        config["source_layers"] = [0, 1, 2]
        config["target_layer"] = 3
        (root / "configs" / JLENS_PROFILE).write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    results = root / "results" / "qwen25_coder_7b_jlens"
    artifacts = root / "artifacts" / "qwen25_coder_7b_jlens"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    fit, holdout = _fit_holdout_meta()
    prepare_results = {
        "dataset_repo_id": config["dataset_repo_id"],
        "dataset_revision": config["dataset_revision"],
        "fit_count": 100,
        "holdout_count": 20,
        "n_selected": 120,
        "exclusions_available": False,
        "n_excluded": 0,
        "benchmark_deduplicated": False,
        "fit": fit,
        "holdout": holdout,
        "fit_ids": [item["example_id"] for item in fit],
        "holdout_ids": [item["example_id"] for item in holdout],
        "fit_hashes": [item["content_hash"] for item in fit],
        "holdout_hashes": [item["content_hash"] for item in holdout],
        "corpus_hash": "c" * 64,
        "config_hash": "d" * 64,
    }
    write_artifact(
        results / "00_jlens_prepare.json",
        _envelope("jlens_prepare", config, prepare_results),
    )
    write_artifact(
        results / "01_jlens_probe.json",
        _envelope(
            "jlens_probe",
            config,
            {
                "probe_order": [32, 16, 8],
                "attempts": [{"dim_batch": 32, "outcome": "ok", "wall_seconds": 1.0}],
                "selected_dim_batch": 32,
                "wall_seconds": 1.0,
                "projected_seconds": 100.0,
                "projected_limit_seconds": 43200,
                "exceeds_limit": False,
            },
        ),
    )
    write_artifact(
        results / "02_jlens_fit.json",
        _envelope(
            "jlens_fit",
            config,
            {
                "n_done": 100,
                "selected_dim_batch": 32,
                "source_layers": config["source_layers"],
                "target_layer": config["target_layer"],
                "skip_first": 16,
                "corpus_hash": "c" * 64,
                "config_hash": "d" * 64,
            },
        ),
    )
    records = []
    for item in holdout:
        for layer in config["source_layers"]:
            records.append(
                {
                    "example_id": item["example_id"],
                    "content_hash": item["content_hash"],
                    "position": 16,
                    "layer": layer,
                    "jlens_rank": 2,
                    "logit_lens_rank": 3,
                }
            )
    metrics = summarize_holdout_metrics(records, source_layers=config["source_layers"])
    write_artifact(
        results / "03_jlens_holdout.json",
        _envelope("jlens_holdout", config, metrics),
    )
    evaluation = {
        "schema_version": 1,
        "status": "pass",
        "operational_success": True,
        "run_id": RUN_ID,
        "implementation_commit": COMMIT,
        "model_revision": QWEN_REVISION,
        "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
        "dataset_repo_id": config["dataset_repo_id"],
        "dataset_config": config["dataset_config"],
        "dataset_split": config["dataset_split"],
        "dataset_revision": config["dataset_revision"],
        "source_dose_run_id": SOURCE_DOSE_RUN_ID,
        "source_dose_status": "pass",
        "selected_dim_batch": 32,
        "n_fit": 100,
        "n_holdout": 20,
    }
    _write_json(results / "jlens_evaluation.json", evaluation)
    write_artifact(
        results / "execution_manifest.json",
        _envelope(
            "execution_manifest",
            config,
            {
                "status": "pass",
                "result_files": [
                    "00_jlens_prepare.json",
                    "01_jlens_probe.json",
                    "02_jlens_fit.json",
                    "03_jlens_holdout.json",
                    "jlens_evaluation.json",
                    "execution_manifest.json",
                ],
            },
        ),
    )
    _write_json(
        artifacts / "status.json",
        {
            "schema_version": 1,
            "profile": JLENS_PROFILE,
            "run_id": RUN_ID,
            "current_stage": "holdout",
            "state": "completed",
            "completed_units": 20,
            "total_units": 20,
            "fit_completed_units": 100,
            "fit_total_units": 100,
            "updated_at_utc": "2026-01-01T00:00:00+00:00",
        },
    )
    events = [
        {"event": "initialized", "run_id": RUN_ID, "stage": "prepare", "state": "pending"},
        {"event": "initialized", "run_id": RUN_ID, "stage": "probe", "state": "pending"},
        {
            "event": "probe_choice",
            "run_id": RUN_ID,
            "dim_batch": 32,
            "state": "running",
            "stage": "probe",
        },
        {"event": "initialized", "run_id": RUN_ID, "stage": "fit", "state": "pending"},
    ]
    events.extend(
        {
            "event": "checkpoint",
            "run_id": RUN_ID,
            "n_done": index,
            "state": "running",
            "stage": "fit",
        }
        for index in range(1, 101)
    )
    events.append(
        {"event": "initialized", "run_id": RUN_ID, "stage": "holdout", "state": "pending"}
    )
    events.append(
        {
            "event": "stage_completed",
            "run_id": RUN_ID,
            "stage": "holdout",
            "state": "completed",
        }
    )
    (artifacts / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )
    layers = config["source_layers"]
    d_model = int(config["d_model"])
    torch.save(
        {
            "J": {layer: torch.eye(d_model) for layer in layers},
            "n_prompts": 100,
            "source_layers": layers,
            "d_model": d_model,
        },
        artifacts / "qwen25_coder_7b_jacobian_lens.pt",
    )
    torch.save(
        {
            "jacobian_sum": {layer: torch.eye(d_model) for layer in layers},
            "n_done": 100,
            "next_idx": 100,
            "source_layers": layers,
            "target_layer": config["target_layer"],
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": QWEN_REVISION,
            "implementation_commit": COMMIT,
            "run_id": RUN_ID,
            "dim_batch": 32,
        },
        artifacts / "jlens_fit_checkpoint.pt",
    )
    lens_path = artifacts / "qwen25_coder_7b_jacobian_lens.pt"
    lens_digest = hashlib.sha256(lens_path.read_bytes()).hexdigest()
    lens_bytes = lens_path.stat().st_size
    fit_path = results / "02_jlens_fit.json"
    fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_payload["results"].update(
        {
            "lens_sha256": lens_digest,
            "lens_bytes": lens_bytes,
            "jlens_upstream_commit": JLENS_UPSTREAM_COMMIT,
            "implementation_commit": COMMIT,
            "run_id": RUN_ID,
            "resolved_model_revision": QWEN_REVISION,
        }
    )
    fit_path.write_text(json.dumps(fit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluation_path = results / "jlens_evaluation.json"
    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_payload["lens_sha256"] = lens_digest
    evaluation_payload["lens_bytes"] = lens_bytes
    evaluation_payload["corpus_hash"] = "c" * 64
    evaluation_payload["config_hash"] = "d" * 64
    evaluation_path.write_text(
        json.dumps(evaluation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dose = root / "results" / "qwen25_coder_7b_dose"
    dose.mkdir(parents=True)
    _write_json(
        dose / "dose_followup_evaluation.json",
        {
            "status": "pass",
            "run_id": SOURCE_DOSE_RUN_ID,
            "implementation_commit": "10587e1db6ecf53f4e422c1c4a351edfd3647204",
            "model_revision": QWEN_REVISION,
        },
    )
    return root


def test_validator_accepts_complete_tiny_tree(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    checks = validate_qwen_jlens(root)
    failed = [name for name, passed in checks.items() if not passed]
    assert failed == ["original_pilot_validator", "original_dose_validator"]


def test_validator_rejects_overlapping_hashes_and_leakage(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    prepare_path = root / "results" / "qwen25_coder_7b_jlens" / "00_jlens_prepare.json"
    payload = json.loads(prepare_path.read_text(encoding="utf-8"))
    payload["results"]["holdout_hashes"][0] = payload["results"]["fit_hashes"][0]
    payload["results"]["fit_text"] = "def leaked(): pass"
    prepare_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["fit_holdout_disjoint"] is False
    assert checks["no_raw_text_leakage"] is False


def test_validator_rejects_probe_order_and_holdout_arithmetic(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    probe_path = root / "results" / "qwen25_coder_7b_jlens" / "01_jlens_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["results"]["probe_order"] = [8, 16, 32]
    probe["results"]["exceeds_limit"] = True
    probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    holdout_path = root / "results" / "qwen25_coder_7b_jlens" / "03_jlens_holdout.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout["results"]["overall"]["jlens"]["pass@1"] = 1.0
    holdout_path.write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["probe_order_and_projection"] is False
    assert checks["holdout_metric_arithmetic"] is False


def test_production_config_keeps_qwen_dimensions() -> None:
    config = load_config(ROOT, profile=JLENS_PROFILE)
    assert config["n_layers"] == 28
    assert config["d_model"] == 3584
    assert config["source_layers"] == list(range(27))
    assert config["target_layer"] == 27
    assert config["jlens_upstream_commit"] == JLENS_UPSTREAM_COMMIT


def test_validator_rejects_duplicate_internal_hashes(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    prepare_path = root / "results" / "qwen25_coder_7b_jlens" / "00_jlens_prepare.json"
    payload = json.loads(prepare_path.read_text(encoding="utf-8"))
    payload["results"]["fit_hashes"][1] = payload["results"]["fit_hashes"][0]
    prepare_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["fit_holdout_disjoint"] is False


def test_validator_rejects_probe_math_mismatch(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    probe_path = root / "results" / "qwen25_coder_7b_jlens" / "01_jlens_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["results"]["wall_seconds"] = 2.0
    probe["results"]["projected_seconds"] = 100.0
    probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["probe_order_and_projection"] is False


def test_validator_requires_one_hundred_checkpoint_events(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    events_path = root / "artifacts" / "qwen25_coder_7b_jlens" / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "event": "probe_choice",
                "run_id": RUN_ID,
                "dim_batch": 32,
                "state": "running",
                "stage": "probe",
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "checkpoint",
                "run_id": RUN_ID,
                "n_done": 1,
                "state": "running",
                "stage": "fit",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checks = validate_qwen_jlens(root)
    assert checks["run_status_events_manifest"] is False


def test_validator_does_not_skip_original_validators_on_tiny_tree(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    checks = validate_qwen_jlens(root)
    assert checks["original_pilot_validator"] is False
    assert checks["original_dose_validator"] is False


def test_validator_runs_original_validators_on_project_root() -> None:
    checks = validate_qwen_jlens(ROOT)
    assert checks["original_pilot_validator"] is True
    assert checks["original_dose_validator"] is True


def test_validator_rejects_missing_provenance_and_non_sha256(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    prepare_path = root / "results" / "qwen25_coder_7b_jlens" / "00_jlens_prepare.json"
    payload = json.loads(prepare_path.read_text(encoding="utf-8"))
    del payload["provenance"]["model_name"]
    payload["results"]["corpus_hash"] = "not-a-hash"
    prepare_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["model_dataset_upstream_revisions"] is False


def test_validator_rejects_holdout_order_zero_positions_and_layer_mismatch(
    tmp_path: Path,
) -> None:
    root = write_jlens_tree(tmp_path)
    holdout_path = root / "results" / "qwen25_coder_7b_jlens" / "03_jlens_holdout.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout["results"]["holdout_ids"] = list(reversed(holdout["results"]["holdout_ids"]))
    first_id = next(iter(holdout["results"]["by_item"]))
    first_layer = next(iter(holdout["results"]["by_item"][first_id]))
    holdout["results"]["by_item"][first_id][first_layer]["n_positions"] = 0
    holdout["results"]["by_layer"][first_layer]["jlens"]["mean_rank"] = 99.0
    holdout_path.write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["holdout_complete"] is False
    assert checks["holdout_metric_arithmetic"] is False


def test_validator_accepts_resume_event_history(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    events_path = root / "artifacts" / "qwen25_coder_7b_jlens" / "events.jsonl"
    extra = [
        {"event": "initialized", "run_id": RUN_ID, "stage": "prepare", "state": "pending"},
        {"event": "initialized", "run_id": RUN_ID, "stage": "fit", "state": "pending"},
        {
            "event": "checkpoint",
            "run_id": RUN_ID,
            "n_done": 100,
            "state": "running",
            "stage": "fit",
        },
        {
            "event": "initialized",
            "run_id": RUN_ID,
            "stage": "holdout",
            "state": "pending",
        },
        {
            "event": "stage_completed",
            "run_id": RUN_ID,
            "stage": "holdout",
            "state": "completed",
        },
    ]
    events_path.write_text(
        events_path.read_text(encoding="utf-8")
        + "".join(json.dumps(item) + "\n" for item in extra),
        encoding="utf-8",
    )
    checks = validate_qwen_jlens(root)
    assert checks["run_status_events_manifest"] is True


def test_validator_rejects_empty_manifest_and_incomplete_checkpoint(
    tmp_path: Path,
) -> None:
    root = write_jlens_tree(tmp_path)
    manifest_path = root / "results" / "qwen25_coder_7b_jlens" / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["result_files"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = root / "artifacts" / "qwen25_coder_7b_jlens" / "jlens_fit_checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["n_done"] = 99
    torch.save(payload, checkpoint_path)
    evaluation_path = root / "results" / "qwen25_coder_7b_jlens" / "jlens_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["lens_bytes"] = 1
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["run_status_events_manifest"] is False
    assert checks["checkpoint_and_lens_identity"] is False


def test_validator_recomputes_lens_hash_and_holdout_item_stats(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    fit_path = root / "results" / "qwen25_coder_7b_jlens" / "02_jlens_fit.json"
    payload = json.loads(fit_path.read_text(encoding="utf-8"))
    payload["results"]["lens_sha256"] = "0" * 64
    payload["results"]["lens_bytes"] = 1
    fit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["checkpoint_and_lens_identity"] is False


def test_validator_rejects_config_dataset_drift(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    config_path = root / "configs" / JLENS_PROFILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_revision"] = "0" * 40
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["model_dataset_upstream_revisions"] is False


def test_validator_rejects_missing_dataset_config_and_split(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    prepare_path = root / "results" / "qwen25_coder_7b_jlens" / "00_jlens_prepare.json"
    payload = json.loads(prepare_path.read_text(encoding="utf-8"))
    payload["provenance"].pop("dataset_config", None)
    payload["provenance"].pop("dataset_split", None)
    prepare_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["model_dataset_upstream_revisions"] is False


def test_validator_rejects_foreign_run_events(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    events_path = root / "artifacts" / "qwen25_coder_7b_jlens" / "events.jsonl"
    extra = {
        "event": "checkpoint",
        "run_id": "other-run",
        "n_done": 50,
        "stage": "fit",
        "state": "running",
    }
    events_path.write_text(
        events_path.read_text(encoding="utf-8") + json.dumps(extra) + "\n",
        encoding="utf-8",
    )
    checks = validate_qwen_jlens(root)
    assert checks["run_status_events_manifest"] is False


def test_validator_rejects_dim_batch_and_projection_rule_mismatch(
    tmp_path: Path,
) -> None:
    root = write_jlens_tree(tmp_path)
    fit_path = root / "results" / "qwen25_coder_7b_jlens" / "02_jlens_fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["results"]["selected_dim_batch"] = 16
    fit_path.write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["probe_order_and_projection"] is False
    root = write_jlens_tree(tmp_path / "projection")
    probe_path = root / "results" / "qwen25_coder_7b_jlens" / "01_jlens_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["results"]["projected_limit_seconds"] = 99999
    probe["results"]["exceeds_limit"] = False
    probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["probe_order_and_projection"] is False


def test_validator_rejects_missing_evaluation_hashes(tmp_path: Path) -> None:
    root = write_jlens_tree(tmp_path)
    evaluation_path = root / "results" / "qwen25_coder_7b_jlens" / "jlens_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("corpus_hash", None)
    evaluation.pop("config_hash", None)
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["checkpoint_and_lens_identity"] is False
    assert checks["model_dataset_upstream_revisions"] is False


def test_validator_rejects_negative_and_nonmonotonic_hits(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_jlens_validate import (
        _metric_from_counts,
        _recompose_by_layer,
    )

    root = write_jlens_tree(tmp_path)
    holdout_path = root / "results" / "qwen25_coder_7b_jlens" / "03_jlens_holdout.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    first_id = next(iter(holdout["results"]["by_item"]))
    first_layer = next(iter(holdout["results"]["by_item"][first_id]))
    block = holdout["results"]["by_item"][first_id][first_layer]
    block["jlens_hits@1"] = 4
    block["jlens_hits@5"] = 2
    block["jlens_hits@10"] = -1
    source_layers = [int(key) for key in holdout["results"]["by_layer"]]
    holdout["results"]["by_layer"] = _recompose_by_layer(
        holdout["results"]["by_item"], source_layers
    )
    for method, prefix in (("jlens", "jlens"), ("logit_lens", "logit_lens")):
        n_pos = rank_sum = hits1 = hits5 = hits10 = 0
        for layers in holdout["results"]["by_item"].values():
            for cell in layers.values():
                n_pos += int(cell.get("n_positions") or 0)
                rank_sum += int(cell.get(f"{prefix}_rank_sum") or 0)
                hits1 += int(cell.get(f"{prefix}_hits@1") or 0)
                hits5 += int(cell.get(f"{prefix}_hits@5") or 0)
                hits10 += int(cell.get(f"{prefix}_hits@10") or 0)
        holdout["results"]["overall"][method] = _metric_from_counts(
            n_pos=n_pos, rank_sum=rank_sum, hits1=hits1, hits5=hits5, hits10=hits10
        )
    holdout_path.write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    checks = validate_qwen_jlens(root)
    assert checks["holdout_metric_arithmetic"] is False
