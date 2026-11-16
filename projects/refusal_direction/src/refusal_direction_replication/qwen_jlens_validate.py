"""Offline validator for compact Qwen Jacobian-lens artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .pipeline import load_config
from .qwen_dose_validate import validate_qwen_dose
from .qwen_dose import sha256_file
from .qwen_jlens import (
    DIM_BATCH_PROBE_ORDER,
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    EXPECTED_SOURCE_LAYERS,
    EXPECTED_TARGET_LAYER,
    FIT_COUNT,
    HOLDOUT_COUNT,
    JLENS_PROFILE,
    JLENS_UPSTREAM_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
    SKIP_FIRST,
    SOURCE_DOSE_RUN_ID,
    jlens_metadata_contains_leak,
    validate_fitted_lens,
)
from .qwen_pilot_validate import _commit_ok, validate_qwen_pilot
from .run_progress import EVENTS_FILENAME, STATUS_FILENAME

PASS_FILES = (
    "00_jlens_prepare.json",
    "01_jlens_probe.json",
    "02_jlens_fit.json",
    "03_jlens_holdout.json",
    "jlens_evaluation.json",
    "execution_manifest.json",
)
ABORTED_FILES = (
    "00_jlens_prepare.json",
    "01_jlens_probe.json",
    "jlens_evaluation.json",
    "execution_manifest.json",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_mean(by_layer: dict[str, Any], method: str, field: str) -> float:
    total = 0.0
    weight = 0
    for payload in by_layer.values():
        block = payload[method]
        n_obs = int(block["n_observations"])
        total += float(block[field]) * n_obs
        weight += n_obs
    return total / weight if weight else 0.0


class _LoadedLens:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.jacobians = {
            int(layer): tensor for layer, tensor in payload["J"].items()
        }
        self.source_layers = [int(layer) for layer in payload["source_layers"]]
        self.n_prompts = int(payload["n_prompts"])
        self.d_model = int(payload["d_model"])


def validate_qwen_jlens(root: Path) -> dict[str, bool]:
    results_dir = root / "results" / "qwen25_coder_7b_jlens"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b_jlens"
    evaluation_path = results_dir / "jlens_evaluation.json"
    status_path = artifacts_dir / STATUS_FILENAME
    checks = {
        "required_stages_present": False,
        "model_dataset_upstream_revisions": True,
        "fit_holdout_counts": True,
        "fit_holdout_disjoint": True,
        "no_raw_text_leakage": True,
        "probe_order_and_projection": True,
        "checkpoint_and_lens_identity": True,
        "holdout_complete": True,
        "holdout_metric_arithmetic": True,
        "run_status_events_manifest": True,
        "source_dose_still_pass": True,
        "original_pilot_validator": False,
        "original_dose_validator": False,
        "production_layer_contract": True,
    }
    try:
        config = load_config(root, profile=JLENS_PROFILE)
    except Exception:
        checks["model_dataset_upstream_revisions"] = False
        return checks

    if config.get("model_name") != QWEN_MODEL:
        checks["model_dataset_upstream_revisions"] = False
    if config.get("model_revision") != QWEN_REVISION:
        checks["model_dataset_upstream_revisions"] = False
    if config.get("jlens_upstream_commit") != JLENS_UPSTREAM_COMMIT:
        checks["model_dataset_upstream_revisions"] = False
    package_config_path = Path(__file__).resolve().parents[2] / "configs" / JLENS_PROFILE
    if package_config_path.is_file():
        package_config = json.loads(package_config_path.read_text(encoding="utf-8"))
        if (
            int(package_config.get("n_layers", -1)) != EXPECTED_N_LAYERS
            or int(package_config.get("d_model", -1)) != EXPECTED_D_MODEL
            or list(package_config.get("source_layers") or []) != EXPECTED_SOURCE_LAYERS
            or int(package_config.get("target_layer", -1)) != EXPECTED_TARGET_LAYER
        ):
            checks["production_layer_contract"] = False
    else:
        checks["production_layer_contract"] = False

    dose_path = root / config.get(
        "source_dose_evaluation",
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json",
    )
    if dose_path.is_file():
        dose = _load(dose_path)
        if dose.get("status") != "pass" or dose.get("run_id") != SOURCE_DOSE_RUN_ID:
            checks["source_dose_still_pass"] = False
    else:
        checks["source_dose_still_pass"] = False

    try:
        original = validate_qwen_pilot(root)
        checks["original_pilot_validator"] = all(original.values())
    except Exception:
        checks["original_pilot_validator"] = False
    try:
        dose = validate_qwen_dose(root)
        checks["original_dose_validator"] = all(dose.values())
    except Exception:
        checks["original_dose_validator"] = False

    if not evaluation_path.is_file():
        return checks
    evaluation = _load(evaluation_path)
    status = str(evaluation.get("status"))
    required = ABORTED_FILES if status == "aborted" else PASS_FILES
    checks["required_stages_present"] = all(
        (results_dir / name).is_file() for name in required
    ) and status_path.is_file()

    compact_files = [
        path for path in sorted(results_dir.glob("*.json")) if path.is_file()
    ]
    payloads = [_load(path) for path in compact_files]
    run_ids = {evaluation.get("run_id")}
    for payload in payloads:
        if jlens_metadata_contains_leak(payload):
            checks["no_raw_text_leakage"] = False
        provenance = payload.get("provenance") or {}
        if provenance.get("model_name") not in {None, QWEN_MODEL}:
            checks["model_dataset_upstream_revisions"] = False
        if provenance.get("requested_model_revision") not in {None, QWEN_REVISION}:
            checks["model_dataset_upstream_revisions"] = False
        if provenance.get("jlens_upstream_commit") not in {None, JLENS_UPSTREAM_COMMIT}:
            checks["model_dataset_upstream_revisions"] = False
        if provenance.get("dataset_revision") not in {
            None,
            config.get("dataset_revision"),
        }:
            checks["model_dataset_upstream_revisions"] = False
        if not _commit_ok(provenance.get("implementation_commit") or evaluation.get("implementation_commit")):
            if provenance.get("implementation_commit"):
                checks["run_status_events_manifest"] = False
        run_ids.add(provenance.get("run_id") or evaluation.get("run_id"))

    prepare_path = results_dir / "00_jlens_prepare.json"
    if prepare_path.is_file():
        prepare = _load(prepare_path)
        results = prepare.get("results") or {}
        fit_ids = list(results.get("fit_ids") or [])
        holdout_ids = list(results.get("holdout_ids") or [])
        fit_hashes = list(results.get("fit_hashes") or [])
        holdout_hashes = list(results.get("holdout_hashes") or [])
        if results.get("fit_count") != FIT_COUNT or len(fit_ids) != FIT_COUNT:
            checks["fit_holdout_counts"] = False
        if results.get("holdout_count") != HOLDOUT_COUNT or len(holdout_ids) != HOLDOUT_COUNT:
            checks["fit_holdout_counts"] = False
        all_hashes = list(fit_hashes) + list(holdout_hashes)
        if set(fit_ids) & set(holdout_ids) or set(fit_hashes) & set(holdout_hashes):
            checks["fit_holdout_disjoint"] = False
        if len(set(fit_ids)) != len(fit_ids) or len(set(holdout_ids)) != len(holdout_ids):
            checks["fit_holdout_disjoint"] = False
        if len(set(fit_hashes)) != len(fit_hashes) or len(set(holdout_hashes)) != len(
            holdout_hashes
        ):
            checks["fit_holdout_disjoint"] = False
        if len(all_hashes) != FIT_COUNT + HOLDOUT_COUNT or len(set(all_hashes)) != len(
            all_hashes
        ):
            checks["fit_holdout_disjoint"] = False
        if results.get("exclusions_available") is False and results.get(
            "benchmark_deduplicated"
        ):
            checks["no_raw_text_leakage"] = False

    probe_path = results_dir / "01_jlens_probe.json"
    if probe_path.is_file():
        probe = (_load(probe_path).get("results") or {})
        if list(probe.get("probe_order") or []) != list(DIM_BATCH_PROBE_ORDER):
            checks["probe_order_and_projection"] = False
        selected = probe.get("selected_dim_batch")
        abort_gate = evaluation.get("abort_gate")
        if status == "aborted" and abort_gate == "dim_batch_oom":
            if selected not in {None}:
                checks["probe_order_and_projection"] = False
        elif selected not in DIM_BATCH_PROBE_ORDER:
            checks["probe_order_and_projection"] = False
        if probe.get("exceeds_limit") and status == "pass":
            checks["probe_order_and_projection"] = False
        if (
            not probe.get("exceeds_limit")
            and float(probe.get("projected_seconds") or 0) > float(
                probe.get("projected_limit_seconds") or 0
            )
        ):
            checks["probe_order_and_projection"] = False
        attempts = list(probe.get("attempts") or [])
        order = list(probe.get("probe_order") or [])
        if not attempts or len(attempts) > len(order):
            checks["probe_order_and_projection"] = False
        else:
            for index, attempt in enumerate(attempts):
                if int(attempt.get("dim_batch") or -1) != int(order[index]):
                    checks["probe_order_and_projection"] = False
            if status == "pass":
                if attempts[-1].get("outcome") != "ok":
                    checks["probe_order_and_projection"] = False
                if any(item.get("outcome") != "cuda_oom" for item in attempts[:-1]):
                    checks["probe_order_and_projection"] = False
                if int(attempts[-1].get("dim_batch") or -1) != int(selected):
                    checks["probe_order_and_projection"] = False
            if status == "aborted" and abort_gate == "dim_batch_oom":
                if [item.get("outcome") for item in attempts] != ["cuda_oom"] * len(
                    order
                ):
                    checks["probe_order_and_projection"] = False
        wall = float(probe.get("wall_seconds") or 0)
        projected = float(probe.get("projected_seconds") or 0)
        if abs(projected - wall * FIT_COUNT) > 1e-6:
            checks["probe_order_and_projection"] = False

    holdout_path = results_dir / "03_jlens_holdout.json"
    if holdout_path.is_file():
        holdout = _load(holdout_path).get("results") or {}
        source_layers = [int(layer) for layer in config["source_layers"]]
        if int(holdout.get("n_items") or 0) != HOLDOUT_COUNT:
            checks["holdout_complete"] = False
        by_layer = holdout.get("by_layer") or {}
        if {int(key) for key in by_layer} != set(source_layers):
            checks["holdout_complete"] = False
        overall = holdout.get("overall") or {}
        by_item = holdout.get("by_item") or {}
        if len(by_item) != HOLDOUT_COUNT:
            checks["holdout_complete"] = False
        for item_id, layers in by_item.items():
            if {int(key) for key in layers} != set(source_layers):
                checks["holdout_complete"] = False
        if len(holdout.get("holdout_ids") or []) != HOLDOUT_COUNT:
            checks["holdout_complete"] = False
        for method in ("jlens", "logit_lens"):
            expected_pass1 = _weighted_mean(by_layer, method, "pass@1")
            expected_pass5 = _weighted_mean(by_layer, method, "pass@5")
            expected_pass10 = _weighted_mean(by_layer, method, "pass@10")
            expected_rank = _weighted_mean(by_layer, method, "mean_rank")
            observed = overall.get(method) or {}
            if abs(float(observed.get("pass@1") or 0) - expected_pass1) > 1e-9:
                checks["holdout_metric_arithmetic"] = False
            if abs(float(observed.get("pass@5") or 0) - expected_pass5) > 1e-9:
                checks["holdout_metric_arithmetic"] = False
            if abs(float(observed.get("pass@10") or 0) - expected_pass10) > 1e-9:
                checks["holdout_metric_arithmetic"] = False
            if abs(float(observed.get("mean_rank") or 0) - expected_rank) > 1e-9:
                checks["holdout_metric_arithmetic"] = False
            prefix = "jlens" if method == "jlens" else "logit_lens"
            n_pos = 0
            rank_sum = 0
            hits1 = 0
            hits5 = 0
            hits10 = 0
            for layers in by_item.values():
                for layer in source_layers:
                    block = (layers or {}).get(str(int(layer))) or {}
                    n_pos += int(block.get("n_positions") or 0)
                    rank_sum += int(block.get(f"{prefix}_rank_sum") or 0)
                    hits1 += int(block.get(f"{prefix}_hits@1") or 0)
                    hits5 += int(block.get(f"{prefix}_hits@5") or 0)
                    hits10 += int(block.get(f"{prefix}_hits@10") or 0)
            if n_pos:
                if abs(float(observed.get("mean_rank") or 0) - rank_sum / n_pos) > 1e-9:
                    checks["holdout_metric_arithmetic"] = False
                if abs(float(observed.get("pass@1") or 0) - hits1 / n_pos) > 1e-9:
                    checks["holdout_metric_arithmetic"] = False
                if abs(float(observed.get("pass@5") or 0) - hits5 / n_pos) > 1e-9:
                    checks["holdout_metric_arithmetic"] = False
                if abs(float(observed.get("pass@10") or 0) - hits10 / n_pos) > 1e-9:
                    checks["holdout_metric_arithmetic"] = False

    lens_path = artifacts_dir / "qwen25_coder_7b_jacobian_lens.pt"
    checkpoint_path = artifacts_dir / "jlens_fit_checkpoint.pt"
    if lens_path.is_file():
        try:
            payload = torch.load(lens_path, map_location="cpu", weights_only=True)
            lens = _LoadedLens(payload)
            validate_fitted_lens(
                lens,
                source_layers=config["source_layers"],
                d_model=int(config["d_model"]),
                n_prompts=FIT_COUNT,
            )
            if int(payload.get("d_model")) != int(config["d_model"]):
                checks["checkpoint_and_lens_identity"] = False
            actual_hash = sha256_file(lens_path)
            actual_size = lens_path.stat().st_size
            fit_path = results_dir / "02_jlens_fit.json"
            fit_results = (
                _load(fit_path).get("results") if fit_path.is_file() else {}
            ) or {}
            if fit_results.get("lens_sha256") != actual_hash:
                checks["checkpoint_and_lens_identity"] = False
            if int(fit_results.get("lens_bytes") or -1) != actual_size:
                checks["checkpoint_and_lens_identity"] = False
            if evaluation.get("lens_sha256") not in {None, actual_hash}:
                checks["checkpoint_and_lens_identity"] = False
            if status == "pass" and evaluation.get("lens_sha256") != actual_hash:
                checks["checkpoint_and_lens_identity"] = False
            for field_name in (
                "corpus_hash",
                "config_hash",
                "jlens_upstream_commit",
                "implementation_commit",
                "run_id",
            ):
                if not fit_results.get(field_name):
                    checks["checkpoint_and_lens_identity"] = False
        except Exception:
            checks["checkpoint_and_lens_identity"] = False
    elif status == "pass":
        checks["checkpoint_and_lens_identity"] = False
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if checkpoint.get("run_id") != evaluation.get("run_id"):
                checks["checkpoint_and_lens_identity"] = False
            if checkpoint.get("model_revision") != QWEN_REVISION:
                checks["checkpoint_and_lens_identity"] = False
            if int(checkpoint.get("skip_first")) != SKIP_FIRST:
                checks["checkpoint_and_lens_identity"] = False
            if [int(layer) for layer in checkpoint.get("source_layers") or []] != [
                int(layer) for layer in config["source_layers"]
            ]:
                checks["checkpoint_and_lens_identity"] = False
        except Exception:
            checks["checkpoint_and_lens_identity"] = False

    if status_path.is_file():
        status_payload = _load(status_path)
        if jlens_metadata_contains_leak(status_payload):
            checks["no_raw_text_leakage"] = False
        if status_payload.get("run_id") != evaluation.get("run_id"):
            checks["run_status_events_manifest"] = False
        if status == "pass":
            if int(status_payload.get("completed_units") or -1) != int(
                status_payload.get("total_units") or -2
            ):
                checks["run_status_events_manifest"] = False
        events_path = artifacts_dir / EVENTS_FILENAME
        if events_path.is_file():
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            names = {item.get("event") for item in events}
            if status == "pass" and not {"probe_choice", "checkpoint", "stage_completed"} <= names:
                checks["run_status_events_manifest"] = False
            checkpoint_events = [
                item for item in events if item.get("event") == "checkpoint"
            ]
            if status == "pass" and len(checkpoint_events) != FIT_COUNT:
                checks["run_status_events_manifest"] = False
            initialized_stages = [
                item.get("stage")
                for item in events
                if item.get("event") == "initialized"
            ]
            if status == "pass" and initialized_stages != [
                "prepare",
                "probe",
                "fit",
                "holdout",
            ]:
                checks["run_status_events_manifest"] = False
            if status == "pass":
                if status_payload.get("current_stage") != "holdout":
                    checks["run_status_events_manifest"] = False
                if status_payload.get("state") != "completed":
                    checks["run_status_events_manifest"] = False
            if any(jlens_metadata_contains_leak(item) for item in events):
                checks["no_raw_text_leakage"] = False
        else:
            checks["run_status_events_manifest"] = False
    else:
        checks["run_status_events_manifest"] = False

    manifest_path = results_dir / "execution_manifest.json"
    if manifest_path.is_file():
        manifest = _load(manifest_path)
        if (manifest.get("results") or {}).get("status") != status:
            checks["run_status_events_manifest"] = False
        listed = list((manifest.get("results") or {}).get("result_files") or [])
        expected_files = list(ABORTED_FILES if status == "aborted" else PASS_FILES)
        if listed and set(listed) != set(expected_files):
            checks["run_status_events_manifest"] = False
        actual = {path.name for path in results_dir.glob("*.json") if path.is_file()}
        if expected_files and not set(expected_files).issubset(actual):
            checks["run_status_events_manifest"] = False
        if len(run_ids - {None}) > 1:
            checks["run_status_events_manifest"] = False

    return checks
