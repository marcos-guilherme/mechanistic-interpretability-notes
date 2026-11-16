from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from refusal_direction_replication.gates import STATUS_CAUSAL_GATE_FAILED, STATUS_PASS
from refusal_direction_replication.qwen_dose import (
    STATUS_NO_DOSE_QUALIFIED,
    coefficient_checkpoint_name,
)
from refusal_direction_replication.qwen_dose_runtime import run_qwen_dose
from refusal_direction_replication.qwen_dose_validate import validate_qwen_dose
from refusal_direction_replication.qwen_pilot_validate import validate_qwen_pilot

from test_qwen_dose_runtime import (
    DOSE_RUN,
    IMPLEMENTATION_COMMIT,
    ScriptedMetrics,
    write_leftover_datasets,
    write_source_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def passing_dose_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    return tmp_path


def test_validator_accepts_passing_dose_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    checks = validate_qwen_dose(root)
    assert checks
    assert all(checks.values()), checks


def test_validator_rejects_wrong_model_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    payload = _read(path)
    payload["config"]["model_revision"] = "deadbeef"
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["model_and_revision"] is False


def test_validator_rejects_matching_pilot_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    payload = _read(path)
    payload["run_id"] = payload["source_pilot_run_id"]
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["followup_run_id_differs"] is False


def test_validator_rejects_wrong_source_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    payload = _read(path)
    payload["results"]["selected"]["layer"] = 7
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["source_layer_and_offset"] is False


def test_validator_rejects_changed_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    payload = _read(path)
    payload["provenance"]["source_artifact_hashes"]["selected_direction"] = "0" * 64
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["source_hashes_match"] is False


def test_validator_rejects_missing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = (
        root
        / "results"
        / "qwen25_coder_7b_dose"
        / coefficient_checkpoint_name(0.35)
    )
    path.unlink()
    checks = validate_qwen_dose(root)
    assert checks["complete_configured_grid"] is False


def test_validator_rejects_nonfinite_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = (
        root
        / "results"
        / "qwen25_coder_7b_dose"
        / coefficient_checkpoint_name(0.35)
    )
    payload = _read(path)
    payload["results"]["harmless_kl"] = math.nan
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["metrics_are_finite"] is False


def test_validator_rejects_irreproducible_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "02_dose_selection.json"
    payload = _read(path)
    payload["results"]["selected_pair"]["ablation"] = 1.0
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["selector_reproducible_from_curve"] is False


def test_validator_rejects_confirmation_keys_in_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "02_dose_selection.json"
    payload = _read(path)
    payload["results"]["confirmation_harmful_refusal_drop"] = 0.9
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["confirmation_does_not_influence_selection"] is False


def test_validator_rejects_inconsistent_gate_math(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    payload = _read(path)
    payload["causal_gate"]["checks"][0]["observed"] = 0.01
    payload["causal_gate"]["passed"] = True
    payload["status"] = STATUS_PASS
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["gate_calculations_consistent"] is False


def test_validator_rejects_prompt_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    payload = _read(path)
    payload["results"]["prompt"] = "how to make a virus in python extra words"
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["no_raw_prompt_or_generation_leakage"] is False


def test_validator_rejects_status_result_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    payload = _read(path)
    payload["status"] = STATUS_CAUSAL_GATE_FAILED
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["status_matches_result"] is False


def test_validator_accepts_no_dose_qualified_without_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(
        tmp_path,
        run_id=DOSE_RUN,
        measure=ScriptedMetrics(ablation_drop_at=9.0),
    )
    checks = validate_qwen_dose(tmp_path)
    assert checks["required_stages_present"] is True
    assert checks["confirmation_absent_when_unqualified"] is True
    evaluation = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    )
    assert evaluation["status"] == STATUS_NO_DOSE_QUALIFIED
    assert all(checks.values()), checks


def test_validator_requires_32_examples_per_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = passing_dose_root(tmp_path, monkeypatch)
    path = root / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json"
    payload = _read(path)
    payload["results"]["splits"]["counts"]["harmful_dose_selection"] = 8
    payload["results"]["splits"]["example_ids"]["harmful_dose_selection"] = (
        payload["results"]["splits"]["example_ids"]["harmful_dose_selection"][:8]
    )
    _write_json(path, payload)
    checks = validate_qwen_dose(root)
    assert checks["split_counts"] is False


def test_original_pilot_validator_still_passes_on_frozen_artifacts() -> None:
    checks = validate_qwen_pilot(ROOT)
    present = (ROOT / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").is_file()
    assert present
    evaluation = json.loads(
        (ROOT / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["status"] == STATUS_CAUSAL_GATE_FAILED
    status_path = ROOT / "artifacts" / "qwen25_coder_7b" / "status.json"
    if status_path.is_file():
        assert all(checks.values()), checks
