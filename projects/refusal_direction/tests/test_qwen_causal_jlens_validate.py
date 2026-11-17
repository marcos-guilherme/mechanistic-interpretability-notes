"""Tests for the offline causal J-lens validator."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from refusal_direction_replication.artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_causal_jlens import (
    BOOTSTRAP_CI_LOWER_INDEX,
    BOOTSTRAP_CI_UPPER_INDEX,
    CAUSAL_JLENS_PROFILE,
    CONDITIONS,
    DATASETS,
    LENS_VERSIONS,
    PRIMARY_LAYER,
    PRODUCTION_DATASET_SIZE,
    READOUT_LAYERS,
    READOUT_STEPS,
    aggregate_fidelity,
    cluster_bootstrap_difference,
    cohort_digest,
    compact_contains_text_leak,
    summarize_behavior_changes,
)
from refusal_direction_replication.qwen_causal_jlens_runtime import (
    EVALUATION_FILENAME,
    MANIFEST_FILENAME,
    STAGE_FILES,
    checkpoint_identity,
    expected_matrix,
    source_context,
)
from refusal_direction_replication.qwen_causal_jlens_validate import (
    validate_qwen_causal_jlens,
)
from test_qwen_causal_jlens_runtime import _seed_causal_project


PROJECT = Path(__file__).resolve().parents[1]
PACKAGED_PROFILE = PROJECT / "configs" / CAUSAL_JLENS_PROFILE
SCRIPT = PROJECT / "scripts" / "validate_qwen_causal_jlens.py"
RUN_ID = "causal-validate-run"
COMMIT = "a" * 40
EARLY_STOP_DATASET = "mitre"
EARLY_STOP_CONDITION = "ablation"
EOS_TOKEN_ID = 151645
MAX_NEW_TOKENS = 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _res(root: Path, name: str) -> Path:
    return root / "results" / "qwen25_coder_7b_causal_jlens" / name


def _art(root: Path, name: str) -> Path:
    return root / "artifacts" / "qwen25_coder_7b_causal_jlens" / name


def _rank_for(condition: str) -> int:
    return {
        "baseline": 1,
        "ablation": 3,
        "addition": 6,
        "random_control": 2,
    }[condition]


def _generated_token_ids(count: int, *, eos: bool) -> list[int]:
    if count < 1:
        return []
    if eos:
        return [2] * (count - 1) + [EOS_TOKEN_ID]
    ids = [2] * count
    if ids[-1] == EOS_TOKEN_ID:
        ids[-1] = 3
    return ids


def _is_early_stop(dataset: str, example_id: str, condition: str, records: dict[str, list[dict[str, Any]]]) -> bool:
    first_id = records[EARLY_STOP_DATASET][0]["example_id"]
    return (
        dataset == EARLY_STOP_DATASET
        and example_id == first_id
        and condition == EARLY_STOP_CONDITION
    )


def _build_generation_rows(
    records: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    responses: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for dataset in DATASETS:
        baseline_refusal = dataset == "mitre"
        for item in records[dataset]:
            for condition in CONDITIONS:
                early = _is_early_stop(dataset, item["example_id"], condition, records)
                token_count = 3 if early else MAX_NEW_TOKENS
                readout = min(READOUT_STEPS, token_count)
                responses.append(
                    {
                        "dataset": dataset,
                        "example_id": item["example_id"],
                        "content_hash": item["content_hash"],
                        "condition": condition,
                        "prompt": item["text"],
                        "response": f"generated {dataset} {condition}",
                        "refusal": baseline_refusal or condition == "addition",
                        "valid": True,
                        "generated_token_ids": _generated_token_ids(
                            token_count, eos=early
                        ),
                        "generated_token_count": token_count,
                        "readout_steps": readout,
                        "terminal_reason": "eos" if early else "max_tokens",
                    }
                )
                for step in range(1, readout + 1):
                    for layer in READOUT_LAYERS:
                        for version in LENS_VERSIONS:
                            observations.append(
                                {
                                    "dataset": dataset,
                                    "example_id": item["example_id"],
                                    "condition": condition,
                                    "generation_step": step,
                                    "layer": int(layer),
                                    "lens_version": version,
                                    "rank": _rank_for(condition),
                                    "target_token_id": 2,
                                }
                            )
    return responses, observations


def _compact_responses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leak_keys = {"prompt", "response", "response_text", "generated_text", "instruction", "content"}
    return [
        {
            key: value
            for key, value in row.items()
            if str(key).casefold() not in leak_keys and key != "generated_token_ids"
        }
        for row in rows
    ]


def _refusal_rates(responses: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for condition in CONDITIONS:
            items = [
                row
                for row in responses
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            refusals = [bool(row["refusal"]) for row in items]
            output[dataset][condition] = {
                "prompt_count": len(items),
                "refusal_count": sum(refusals),
                "refusal_rate": (sum(refusals) / len(refusals) if refusals else None),
            }
    return output


def _primary_from_bootstrap(bootstrap: dict[str, Any], margin: float) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    for key, payload in bootstrap.items():
        dataset, condition, lens_version = str(key).split("/")
        primary.setdefault(lens_version, {}).setdefault(dataset, {})[condition] = {
            "estimate": payload["estimate"],
            "ci_lower": payload["ci_lower"],
            "ci_upper": payload["ci_upper"],
            "prompt_count": payload["prompt_count"],
            "equivalent": (
                float(payload["ci_lower"]) >= -margin
                and float(payload["ci_upper"]) <= margin
            ),
        }
    return primary


def _seed_complete_tree(root: Path) -> None:
    seeded = _seed_causal_project(root, dataset_size=2)
    config = load_config(root, profile=CAUSAL_JLENS_PROFILE)
    source = source_context(root, config)
    records = seeded["_records"]
    prepare_results = {
        dataset: {
            "dataset": dataset,
            "example_ids": [item["example_id"] for item in records[dataset]],
            "content_hashes": [item["content_hash"] for item in records[dataset]],
            "source_indices": [item["source_index"] for item in records[dataset]],
            "selection_size": len(records[dataset]),
        }
        for dataset in DATASETS
    }
    source_prepare = json.loads(
        (root / str(config["source_prepare"])).read_text(encoding="utf-8")
    )
    prepare_results["source_files"] = (source_prepare.get("results") or {}).get(
        "source_files"
    ) or {}
    provenance = {
        "run_id": RUN_ID,
        "implementation_commit": COMMIT,
        "config_sha256": checkpoint_identity(
            run_id=RUN_ID,
            implementation_commit=COMMIT,
            config=config,
            source=source,
            prepare=prepare_results,
        )["config_hash"],
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": config["model_revision"],
        "purplellama_revision": config["purplellama_revision"],
        "jlens_upstream_commit": config["jlens_upstream_commit"],
        "source_dose_run_id": source.get("source_dose_run_id"),
        "source_artifact_hashes": dict(source["source_artifact_hashes"]),
        "direction_layer": source["direction_layer"],
        "direction_offset": source["direction_offset"],
        "eos_token_id": int(config.get("eos_token_id", EOS_TOKEN_ID)),
        "cohort_digest": cohort_digest(source_prepare),
    }
    responses, observations = _build_generation_rows(records)
    compact = _compact_responses(responses)
    fidelity = aggregate_fidelity(observations)
    behavior = summarize_behavior_changes(compact, observations)
    bootstrap: dict[str, Any] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            if condition == "baseline":
                continue
            for lens_version in LENS_VERSIONS:
                key = f"{dataset}/{condition}/{lens_version}"
                bootstrap[key] = cluster_bootstrap_difference(
                    observations,
                    condition=condition,
                    k=int(config["primary_k"]),
                    layer=int(config["primary_layer"]),
                    lens_version=lens_version,
                    dataset=dataset,
                    resamples=int(config["bootstrap_resamples"]),
                    seed=int(config["bootstrap_seed"]),
                )
    analysis_results = {
        "fidelity": fidelity,
        "bootstrap": bootstrap,
        "behavior_changes": behavior,
        "refusal_rates": _refusal_rates(compact),
        "observation_count": len(observations),
        "response_count": len(responses),
    }
    matrix = expected_matrix(prepare_results)
    panel_results = {
        "response_count": len(responses),
        "observation_count": len(observations),
        "conditions": list(CONDITIONS),
        "matrix_size": len(matrix),
        "terminal_reasons": [row["terminal_reason"] for row in responses],
    }
    pilot_results = {
        "response_count": 16,
        "observation_count": sum(
            1
            for row in observations
            if row["example_id"]
            in {
                records[dataset][index]["example_id"]
                for dataset in DATASETS
                for index in range(2)
            }
        ),
        "pilot_status": "pass",
        "conditions": list(CONDITIONS),
        "pilot_ids": {
            dataset: [item["example_id"] for item in records[dataset][:2]]
            for dataset in DATASETS
        },
        "terminal_reasons": [row["terminal_reason"] for row in responses],
    }
    evaluation = {
        "schema_version": 1,
        "status": "pass",
        "pilot_status": "pass",
        "response_count": len(responses),
        "observation_count": len(observations),
        "prepare_ids": {
            dataset: prepare_results[dataset]["example_ids"] for dataset in DATASETS
        },
        "results": {
            "primary": _primary_from_bootstrap(
                bootstrap, float(config["equivalence_margin"])
            ),
            "refusal_rates": analysis_results["refusal_rates"],
            "behavior_change_analysis": behavior,
            "fidelity": fidelity,
        },
        "run_id": RUN_ID,
        "implementation_commit": COMMIT,
        "model_revision": config["model_revision"],
        "source_artifact_hashes": dict(source["source_artifact_hashes"]),
        "config": dict(config),
        "provenance": dict(provenance),
    }
    results_dir = root / str(config["results_dir"])
    artifacts_dir = root / str(config["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stage_payloads = {
        "prepare": prepare_results,
        "pilot": pilot_results,
        "panel": panel_results,
        "analysis": analysis_results,
    }
    for stage, results in stage_payloads.items():
        write_artifact(
            results_dir / STAGE_FILES[stage],
            ArtifactEnvelope(
                experiment=f"causal_jlens_{stage}",
                config=dict(config),
                results=results,
                provenance=dict(provenance),
            ),
        )
    atomic_write_json(results_dir / EVALUATION_FILENAME, evaluation)
    write_artifact(
        results_dir / MANIFEST_FILENAME,
        ArtifactEnvelope(
            experiment="causal_jlens_manifest",
            config=dict(config),
            results={
                "status": "pass",
                "completed_stages": ["prepare", "pilot", "panel", "analysis", "finalize"],
                "result_files": [
                    STAGE_FILES["prepare"],
                    STAGE_FILES["pilot"],
                    STAGE_FILES["panel"],
                    STAGE_FILES["analysis"],
                    EVALUATION_FILENAME,
                    MANIFEST_FILENAME,
                ],
            },
            provenance=dict(provenance),
        ),
    )
    _write_jsonl(artifacts_dir / "responses.jsonl", responses)
    _write_jsonl(artifacts_dir / "observations.jsonl", observations)
    identity = checkpoint_identity(
        run_id=RUN_ID,
        implementation_commit=COMMIT,
        config=config,
        source=source,
        prepare=prepare_results,
    )
    atomic_write_json(
        artifacts_dir / "checkpoint.json",
        {
            "identity": identity,
            "completed": [list(item) for item in matrix],
        },
    )
    atomic_write_json(
        artifacts_dir / "status.json",
        {
            "schema_version": 1,
            "profile": CAUSAL_JLENS_PROFILE,
            "run_id": RUN_ID,
            "current_stage": "finalize",
            "state": "completed",
            "completed_units": 1,
            "total_units": 1,
        },
    )
    events = [
        {
            "event": "initialized",
            "stage": stage,
            "run_id": RUN_ID,
            "state": "pending",
        }
        for stage in ("prepare", "pilot", "panel", "analysis", "finalize")
    ]
    events.append(
        {
            "event": "stage_completed",
            "stage": "finalize",
            "state": "completed",
            "run_id": RUN_ID,
            "completed_units": 1,
            "total_units": 1,
        }
    )
    _write_jsonl(artifacts_dir / "events.jsonl", events)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mutate(root: Path, mutation: str) -> None:
    if mutation == "duplicate_observation":
        path = _art(root, "observations.jsonl")
        rows = _jsonl(path)
        rows.append(dict(rows[0]))
        _write_jsonl(path, rows)
        return
    if mutation == "duplicate_exploratory_response":
        path = _art(root, "responses.jsonl")
        rows = _jsonl(path)
        extra = dict(rows[0])
        extra["refusal"] = not bool(extra["refusal"])
        rows.append(extra)
        _write_jsonl(path, rows)
        return
    if mutation == "wrong_rank_metric":
        _rewrite_json(
            _res(root, "03_analysis.json"),
            lambda payload: payload["results"]["fidelity"]["mitre"]["baseline"]["v1"][
                "25"
            ].update(pass_at_5=0.0),
        )
        return
    if mutation == "raw_response_in_results":
        _rewrite_json(
            _res(root, "03_analysis.json"),
            lambda payload: payload["results"].update(response="I cannot help with that"),
        )
        return
    if mutation == "missing_early_stop_marker":
        path = _art(root, "responses.jsonl")
        rows = _jsonl(path)
        found = False
        for row in rows:
            if int(row["generated_token_count"]) < READOUT_STEPS:
                row.pop("terminal_reason", None)
                found = True
        assert found
        _write_jsonl(path, rows)
        return
    if mutation == "wrong_bootstrap_seed":
        _rewrite_json(
            _res(root, "03_analysis.json"),
            lambda payload: [
                block.update(seed=1)
                for block in payload["results"]["bootstrap"].values()
            ],
        )
        return
    if mutation == "wrong_source_hash":
        _rewrite_json(
            _res(root, EVALUATION_FILENAME),
            lambda payload: payload["source_artifact_hashes"].update(direction="0" * 64),
        )
        return
    if mutation == "nested_list_leak":
        _rewrite_json(
            _res(root, "03_analysis.json"),
            lambda payload: payload["results"].update(
                samples=["Write a harmful payload for this prompt"]
            ),
        )
        return
    if mutation == "token_list_count_mismatch":
        path = _art(root, "responses.jsonl")
        rows = _jsonl(path)
        for row in rows:
            if row["terminal_reason"] == "eos":
                row["generated_token_ids"] = list(row["generated_token_ids"])[:-1]
                break
        else:
            raise AssertionError("no eos row")
        _write_jsonl(path, rows)
        return
    if mutation == "invalid_terminal_semantics":
        path = _art(root, "responses.jsonl")
        rows = _jsonl(path)
        for row in rows:
            if row["terminal_reason"] == "max_tokens":
                row["terminal_reason"] = "length"
                break
        else:
            raise AssertionError("no max_tokens row")
        _write_jsonl(path, rows)
        return
    if mutation == "fractional_count":
        path = _art(root, "responses.jsonl")
        rows = _jsonl(path)
        rows[0]["generated_token_count"] = 3.5
        _write_jsonl(path, rows)
        return
    if mutation == "evaluation_model_revision":
        _rewrite_json(
            _res(root, EVALUATION_FILENAME),
            lambda payload: payload.update(model_revision="0" * 40),
        )
        return
    if mutation == "evaluation_prepare_ids":
        _rewrite_json(
            _res(root, EVALUATION_FILENAME),
            lambda payload: payload["prepare_ids"]["mitre"].reverse(),
        )
        return
    if mutation == "evaluation_pilot_status":
        _rewrite_json(
            _res(root, EVALUATION_FILENAME),
            lambda payload: payload.update(pilot_status="fail"),
        )
        return
    if mutation == "manifest_completed_stages":
        _rewrite_json(
            _res(root, MANIFEST_FILENAME),
            lambda payload: payload["results"].update(completed_stages=["prepare"]),
        )
        return
    if mutation == "manifest_config":
        _rewrite_json(
            _res(root, MANIFEST_FILENAME),
            lambda payload: payload["config"].update(max_new_tokens=8),
        )
        return
    raise AssertionError(mutation)


def test_validator_accepts_complete_recomputable_tree(tmp_path: Path) -> None:
    _seed_complete_tree(tmp_path)
    checks = validate_qwen_causal_jlens(tmp_path, allow_fixture_contract=True)
    assert checks
    assert all(checks.values())
    assert {
        "exact_ids_and_pair_matrix",
        "unique_observation_keys",
        "early_termination",
        "metrics_recomputed",
        "bootstrap_recomputed",
        "equivalence_flags",
        "behavior_changes",
        "no_raw_text",
        "shared_identity",
        "frozen_cohort_and_lens_provenance",
    } <= set(checks)
    analysis = json.loads(_res(tmp_path, "03_analysis.json").read_text(encoding="utf-8"))
    sample = next(iter(analysis["results"]["bootstrap"].values()))
    assert sample["ci_lower_index"] == BOOTSTRAP_CI_LOWER_INDEX == 249
    assert sample["ci_upper_index"] == BOOTSTRAP_CI_UPPER_INDEX == 9749
    assert sample["resamples"] == 10_000
    assert sample["seed"] == 2028
    assert not compact_contains_text_leak(analysis)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_observation", "duplicate"),
        ("wrong_rank_metric", "metric"),
        ("raw_response_in_results", "raw text"),
        ("missing_early_stop_marker", "termination"),
        ("wrong_bootstrap_seed", "bootstrap"),
        ("wrong_source_hash", "hash"),
    ],
)
def test_validator_rejects_scientific_corruption(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _seed_complete_tree(tmp_path)
    _mutate(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        validate_qwen_causal_jlens(tmp_path, allow_fixture_contract=True)


def test_validator_rejects_duplicate_exploratory_response_rows(tmp_path: Path) -> None:
    _seed_complete_tree(tmp_path)
    _mutate(tmp_path, "duplicate_exploratory_response")
    with pytest.raises(ValueError, match="duplicate"):
        validate_qwen_causal_jlens(tmp_path, allow_fixture_contract=True)


def test_strict_validation_rejects_a_fixture_profile_by_default(tmp_path: Path) -> None:
    _seed_complete_tree(tmp_path)
    with pytest.raises(ValueError, match="production contract"):
        validate_qwen_causal_jlens(tmp_path)


def test_production_contract_requires_packaged_profile_exactly(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    payload = json.loads(PACKAGED_PROFILE.read_text(encoding="utf-8"))
    assert payload["dataset_size"] == PRODUCTION_DATASET_SIZE
    payload["dataset_size"] = 31
    (tmp_path / "configs" / CAUSAL_JLENS_PROFILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="production contract"):
        validate_qwen_causal_jlens(tmp_path)


def test_strict_validation_accepts_the_packaged_profile_contract(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / CAUSAL_JLENS_PROFILE).write_text(
        PACKAGED_PROFILE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError) as caught:
        validate_qwen_causal_jlens(tmp_path)
    assert "production contract" not in str(caught.value)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "validate_qwen_causal_jlens_script", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_is_strict_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script()
    _seed_complete_tree(tmp_path)
    assert module.main(tmp_path, allow_fixture_contract=True) == 0
    assert "PASS" in capsys.readouterr().out
    assert module.main(tmp_path) == 1
    assert "production contract" in capsys.readouterr().out
    source = SCRIPT.read_text(encoding="utf-8")
    assert "allow_fixture_contract" not in source.split("if __name__", 1)[1]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nested_list_leak", "raw text"),
        ("token_list_count_mismatch", "token"),
        ("invalid_terminal_semantics", "terminal"),
        ("fractional_count", "count"),
        ("evaluation_model_revision", "revision"),
        ("evaluation_prepare_ids", "prepare"),
        ("evaluation_pilot_status", "pilot"),
        ("manifest_completed_stages", "completed_stages"),
        ("manifest_config", "config"),
    ],
)
def test_validator_rejects_review_corruption(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _seed_complete_tree(tmp_path)
    _mutate(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        validate_qwen_causal_jlens(tmp_path, allow_fixture_contract=True)


def _edit_profile(root: Path, **overrides: Any) -> None:
    path = root / "configs" / CAUSAL_JLENS_PROFILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_fixture_mode_rejects_non_allowlisted_profile_fields(tmp_path: Path) -> None:
    _seed_complete_tree(tmp_path)
    _edit_profile(tmp_path, max_new_tokens=32)
    with pytest.raises(ValueError, match="allowlist"):
        validate_qwen_causal_jlens(tmp_path, allow_fixture_contract=True)
