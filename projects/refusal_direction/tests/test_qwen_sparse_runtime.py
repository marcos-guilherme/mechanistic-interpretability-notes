from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import refusal_direction_replication.qwen_sparse_runtime as sparse_runtime
from refusal_direction_replication.qwen_sparse_runtime import (
    CONDITIONS,
    SCIENTIFIC_STATUSES,
    SPARSE_STAGES,
    SparseRuntime,
    SparseStageStore,
    build_code_preservation_jobs,
    compact_contains_leak,
    expected_test_matrix,
    freeze_selection,
    sparse_condition_hook_specs,
)


def _config() -> dict[str, object]:
    return {
        "model_revision": "model-commit",
        "direction_sha256": "d" * 64,
        "dose_grid": [0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        "cardinality_grid": [8, 16, 32, 64],
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_test_pairs() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "pair_id": f"pair-{index:04d}",
            "offensive_sha256": _sha(f"offensive-{index}"),
            "defensive_sha256": _sha(f"defensive-{index}"),
        }
        for index in range(64)
    )


def _write_panel_prepare(path: Path) -> Path:
    results = {}
    for dataset in ("mitre", "mitre_frr", "humaneval_plus"):
        results[dataset] = {
            "example_ids": [f"{dataset}-{index}" for index in range(32)],
            "content_hashes": [_sha(f"{dataset}-{index}") for index in range(32)],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path


def _freeze_fixture(
    vectors: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    aliases = {
        "sparse_unfiltered": "unfiltered",
        "sparse_filtered": "filtered",
        "sparse_random_control": "random",
    }
    rows = []
    for vector, alias in aliases.items():
        vector_hash = (
            sparse_runtime.tensor_sha256(vectors[alias])
            if vectors
            else _sha(vector)
        )
        vector_norm = float(vectors[alias].norm()) if vectors else 1.0
        for intervention in ("addition", "ablation"):
            for cardinality in (8, 16, 32, 64):
                for dose in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                    rows.append(
                        {
                            "vector": vector,
                            "intervention": intervention,
                            "cardinality": cardinality,
                            "dose": dose,
                            "dense_desired_effect": 0.8,
                            "desired_effect": 0.74 if dose == 0.1 else 0.76,
                            "defensive_false_refusal": cardinality / 1000,
                            "defensive_kl": cardinality / 100,
                            "feature_indices": list(range(cardinality)),
                            "coefficients": [1.0] * cardinality,
                            "vector_sha256": vector_hash,
                            "vector_norm": vector_norm,
                            "dense_vector_sha256": (
                                sparse_runtime.tensor_sha256(vectors["dense"])
                                if vectors
                                else "d" * 64
                            ),
                            "dense_vector_norm": (
                                float(vectors["dense"].norm()) if vectors else 1.0
                            ),
                            "dense_dose": 0.5,
                        }
                    )
    return freeze_selection(rows)


def _store(
    tmp_path: Path,
    *,
    run_id: str = "run-1",
    prepare_path: Path | None = None,
    expected_prepare_sha256: str | None = None,
) -> SparseStageStore:
    identity_kwargs = (
        {
            "prepare_path": prepare_path,
            "expected_prepare_sha256": expected_prepare_sha256,
        }
        if prepare_path is not None
        else {}
    )
    return SparseStageStore(
        tmp_path / "results",
        config=_config(),
        run_id=run_id,
        implementation_commit="impl-1",
        corpus_sha256="c" * 64,
        pair_split_sha256="p" * 64,
        test_pairs=_frozen_test_pairs(),
        **identity_kwargs,
    )


def _valid_stage_result(stage: str) -> dict[str, object]:
    if stage == "sae_quality":
        return {
            "status": "pass",
            "explained_variance": 0.8,
            "mean_cosine": 0.9,
            "nonfinite_count": 0,
            "mean_l0": 64.0,
            "dead_feature_fraction": 0.2,
            "norm_drift": 0.2,
        }
    if stage == "decompose":
        return {
            "status": "pass",
            "cosine": 0.95,
            "relative_l2": 0.35,
            "norm": 1.0,
        }
    return {"status": "completed"}


def _complete_through_select(store: SparseStageStore) -> None:
    for stage in SPARSE_STAGES[:6]:
        store.write_stage(stage, _valid_stage_result(stage))
    store.write_stage("select", _freeze_fixture())
    assert store.selection_freeze_path.is_file()


def _complete_through_review(store: SparseStageStore) -> None:
    _complete_through_select(store)
    store.write_stage(
        "test",
        {
            "status": "completed",
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "test_split_sha256": store.test_split_sha256,
            "observations": expected_test_matrix(_frozen_test_pairs()),
        },
    )
    store.write_stage(
        "review_export",
        {
            "status": "review_pending",
            "review_count": 512,
            "test_split_sha256": store.test_split_sha256,
            "blind_review_sha256": "b" * 64,
            "private_manifest_sha256": "a" * 64,
        },
    )
    store.write_stage(
        "review_merge",
        {
            "schema_version": 1,
            "status": "review_complete",
            "review_count": 512,
            "manifest_seed": 2031,
            "blind_review_sha256": "b" * 64,
            "private_manifest_sha256": "a" * 64,
            "test_split_sha256": store.test_split_sha256,
            "reviews": [],
            "label_counts": {},
            "aggregate_counts": {},
            "automatic_agreement": 1.0,
        },
    )


def _external_receipt(
    identity: dict[str, object],
    freeze: dict[str, object],
) -> dict[str, object]:
    observations = []
    for dataset in ("mitre", "mitre_frr"):
        for item in identity["datasets"][dataset]:
            for condition in sparse_runtime.EXTERNAL_CONDITIONS:
                observations.append(
                    {
                        "dataset": dataset,
                        **item,
                        "condition": condition,
                        "dose": sparse_runtime.condition_dose(
                            condition,
                            freeze,
                        ),
                        "refused": int(item["example_id"].rsplit("-", 1)[1])
                        % 2
                        == 0,
                        "kl_from_baseline": 0.1,
                        "invalid_generation": False,
                    }
                )
    return {
        "status": "completed",
        "prepare_sha256": identity["prepare_sha256"],
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "observations": observations,
        "metrics": {
            dataset: {
                condition: {
                    "n": 32,
                    "refusal_rate": 0.5,
                    "mean_kl": 0.1,
                    "invalid_generation_rate": 0.0,
                }
                for condition in sparse_runtime.EXTERNAL_CONDITIONS
            }
            for dataset in ("mitre", "mitre_frr")
        },
    }


def _code_receipt(
    tmp_path: Path,
    identity: dict[str, object],
    freeze: dict[str, object],
) -> dict[str, object]:
    jobs = build_code_preservation_jobs(
        worker=tmp_path / "worker.py",
        problems=tmp_path / "problems.jsonl",
        sample_paths={
            condition: tmp_path / f"{condition}.jsonl"
            for condition in ("baseline", "dense", "filtered")
        },
        output_dir=tmp_path / "output",
        identity_dir=tmp_path / "identity",
        temp_root=tmp_path / "temp",
        timeout_seconds=60,
        remote_vm=True,
        identity=identity,
        freeze=freeze,
    )
    outputs = {
        condition: [
            {
                **item,
                "plus_pass": True,
                "refused": False,
                "kl_from_baseline": 0.0,
                "invalid_generation": False,
            }
            for item in identity["items"]
        ]
        for condition in ("baseline", "dense", "filtered")
    }
    return {
        "status": "completed",
        "prepare_sha256": identity["prepare_sha256"],
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "conditions": ["baseline", "dense", "filtered"],
        "doses": {"baseline": 0.0, "dense": 0.5, "filtered": 0.2},
        "memory_bytes": 1024**3,
        "timeout_seconds": 60,
        "parallelism": 1,
        "remote_vm": True,
        "evalplus_revision": sparse_runtime.EVALPLUS_REVISION,
        "job_manifest_sha256": jobs["job_manifest_sha256"],
        "worker_outputs": outputs,
        "metrics": {
            condition: {
                "n": 32,
                "pass_at_1": 1.0,
                "refusal_rate": 0.0,
                "mean_kl": 0.0,
                "invalid_generation_rate": 0.0,
            }
            for condition in ("baseline", "dense", "filtered")
        },
    }


def test_stage_machine_is_exact_and_test_requires_frozen_selection(
    tmp_path: Path,
) -> None:
    assert SPARSE_STAGES == (
        "prepare",
        "hardware_pilot",
        "sae_fit",
        "sae_quality",
        "decompose",
        "discover",
        "select",
        "test",
        "review_export",
        "review_merge",
        "external_transfer",
        "code_preservation",
        "finalize",
    )
    assert SCIENTIFIC_STATUSES == {
        "pass_addition",
        "pass_ablation",
        "fail_selectivity_addition",
        "fail_selectivity_ablation",
        "resource_projection_failed",
        "sae_quality_failed",
        "direction_reconstruction_failed",
        "inconclusive_review",
        "operational_error",
    }
    runtime = SparseRuntime(_store(tmp_path))
    with pytest.raises(ValueError, match="selection must be frozen"):
        runtime.run_stage("test", lambda **_kwargs: {})


def test_resume_validates_full_provenance_and_prior_hash_chain(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_stage("prepare", {"status": "completed"})
    store.write_stage("hardware_pilot", {"status": "completed"})
    payload = json.loads(store.stage_path("prepare").read_text(encoding="utf-8"))
    payload["provenance"]["corpus_sha256"] = "x" * 64
    store.stage_path("prepare").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    resumed = _store(tmp_path)
    with pytest.raises(ValueError, match="corpus_sha256"):
        resumed.completed_stages()


def test_selection_freezes_full_grid_with_frozen_tie_breaks() -> None:
    rows: list[dict[str, object]] = []
    for vector in ("sparse_unfiltered", "sparse_filtered", "sparse_random_control"):
        for intervention in ("addition", "ablation"):
            for cardinality in (8, 16, 32, 64):
                for dose in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                    actual_cardinality = (
                        7
                        if vector == "sparse_filtered" and cardinality == 8
                        else cardinality
                    )
                    rows.append(
                        {
                            "vector": vector,
                            "intervention": intervention,
                            "cardinality": cardinality,
                            "dose": dose,
                            "dense_desired_effect": 0.80,
                            "desired_effect": 0.74 if dose == 0.1 else 0.76,
                            "defensive_false_refusal": cardinality / 1000,
                            "defensive_kl": cardinality / 100,
                            "feature_indices": list(range(actual_cardinality)),
                            "coefficients": [1.0] * actual_cardinality,
                            "vector_sha256": _sha(f"{vector}:{cardinality}"),
                            "vector_norm": 1.0,
                            "dense_vector_sha256": "d" * 64,
                            "dense_vector_norm": 1.0,
                            "dense_dose": 0.5,
                        }
                    )
    frozen = freeze_selection(rows)
    assert len(frozen["decision_trace"]) == 3 * 2 * 4 * 6
    for vector in ("sparse_unfiltered", "sparse_filtered", "sparse_random_control"):
        for intervention in ("addition", "ablation"):
            selected = frozen["selections"][vector][intervention]
            assert selected["dose"] == 0.2
            assert selected["cardinality"] == 8
    assert frozen["selections"]["sparse_filtered"]["addition"][
        "actual_cardinality"
    ] == 7
    assert not compact_contains_leak(frozen)

    with pytest.raises(ValueError, match="Cartesian selection grid"):
        freeze_selection(rows[:-1])


def test_hooks_preserve_dense_intervention_semantics() -> None:
    blocks = [torch.nn.Identity() for _ in range(24)]
    attentions = [torch.nn.Identity() for _ in range(24)]
    mlps = [torch.nn.Identity() for _ in range(24)]
    vectors = {
        "dense": torch.tensor([1.0, 0.0]),
        "unfiltered": torch.tensor([0.0, 1.0]),
        "filtered": torch.tensor([1.0, 1.0]),
        "random": torch.tensor([-1.0, 1.0]),
    }
    doses = {
        name: {"addition": 0.2, "ablation": 0.35}
        for name in vectors
    }
    freeze = _freeze_fixture(vectors)

    pre, forward = sparse_condition_hook_specs(
        condition="filtered_addition",
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        vectors=vectors,
        freeze=freeze,
    )
    assert len(pre) == 1 and pre[0][0] is blocks[18]
    assert forward == []
    changed = pre[0][1](blocks[18], (torch.zeros(1, 1, 2),))[0]
    torch.testing.assert_close(changed, torch.tensor([[[0.2, 0.2]]]))

    pre, forward = sparse_condition_hook_specs(
        condition="dense_ablation",
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        vectors=vectors,
        freeze=freeze,
    )
    assert len(pre) == 24
    assert len(forward) == 48

    keyed_vectors = {
        "dense": vectors["dense"],
        "filtered_addition": vectors["filtered"],
    }
    pre, forward = sparse_condition_hook_specs(
        condition="filtered_addition",
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        vectors=keyed_vectors,
        freeze=freeze,
    )
    assert len(pre) == 1
    assert forward == []


def test_compact_test_matrix_has_all_nine_conditions_without_leaks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    observations = expected_test_matrix(_frozen_test_pairs())
    payload = SparseRuntime(store).run_stage(
        "test",
        lambda **_kwargs: {
            "status": "completed",
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "test_split_sha256": store.test_split_sha256,
            "observations": observations,
        },
    )
    keys = {
        (row["pair_id"], row["side"], row["condition"])
        for row in payload["results"]["observations"]
    }
    assert len(keys) == 64 * 2 * 9
    assert {row["condition"] for row in observations} == set(CONDITIONS)
    assert not compact_contains_leak(payload)


def test_test_matrix_rejects_same_sized_wrong_frozen_ids_and_hashes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    wrong = list(expected_test_matrix(_frozen_test_pairs()))
    wrong[0] = {**wrong[0], "pair_id": "other-pair"}
    with pytest.raises(ValueError, match="frozen test split"):
        SparseRuntime(store).run_stage(
            "test",
            lambda **_kwargs: {
                "status": "completed",
                "selection_freeze_sha256": store.selection_freeze_sha256(),
                "test_split_sha256": store.test_split_sha256,
                "observations": wrong,
            },
        )


def test_selection_digest_and_hook_vectors_are_governed_by_freeze(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    freeze = json.loads(store.selection_freeze_path.read_text(encoding="utf-8"))
    freeze["selections"]["sparse_filtered"]["addition"]["dose"] = 1.0
    store.selection_freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_freeze_sha256"):
        store.load_stage("select")


def test_stage_allowlist_rejects_unknown_non_blacklisted_field(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="prepare schema"):
        store.write_stage(
            "prepare",
            {"status": "completed", "secret_payload": "not raw-key named"},
        )


def test_operational_result_and_precondition_errors_are_terminal_and_provenanced(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    result = SparseRuntime(store).run_stage(
        "prepare",
        lambda **_kwargs: {
            "status": "operational_error",
            "error_code": "fixture",
        },
    )
    assert result["status"] == "operational_error"
    assert not store.stage_path("prepare").exists()
    error = json.loads(store.operational_error_path.read_text(encoding="utf-8"))
    assert set(error["provenance"]) == {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "model_revision",
        "direction_sha256",
        "corpus_sha256",
        "pair_split_sha256",
        "test_split_sha256",
        "prior_stage_sha256",
    }

    blocked = _store(tmp_path / "blocked")
    with pytest.raises(ValueError, match="selection must be frozen"):
        SparseRuntime(blocked).run_stage("test", lambda **_kwargs: {})
    assert blocked.operational_error_path.is_file()


def test_operational_error_blocks_same_run_resume(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = SparseRuntime(store).run_stage(
        "prepare",
        lambda **_kwargs: {
            "status": "operational_error",
            "error_code": "fixture",
        },
    )
    called = False

    def must_not_run(**_kwargs):
        nonlocal called
        called = True
        return {"status": "completed"}

    resumed = SparseRuntime(_store(tmp_path)).run_stage("prepare", must_not_run)
    assert resumed == first
    assert called is False

    restarted = SparseRuntime(_store(tmp_path, run_id="run-2")).run_stage(
        "prepare",
        lambda **_kwargs: {"status": "completed"},
    )
    assert restarted["provenance"]["run_id"] == "run-2"


def test_entrypoint_returns_operational_error_before_scanning_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        **_config(),
        "results_dir": "results",
        "artifacts_dir": "artifacts",
    }
    terminal = {"status": "operational_error", "stage": "sae_quality"}
    monkeypatch.setattr(sparse_runtime, "load_sparse_config", lambda _root: config)
    monkeypatch.setattr(
        sparse_runtime,
        "verify_sparse_profile_digest",
        lambda _config: None,
    )
    monkeypatch.setattr(
        sparse_runtime,
        "resolve_run_id",
        lambda _path, requested: requested or "run-1",
    )
    monkeypatch.setattr(
        sparse_runtime,
        "_sha256_file",
        lambda _path: "p" * 64,
    )
    monkeypatch.setattr(
        sparse_runtime,
        "_load_frozen_test_pairs",
        lambda _path: _frozen_test_pairs(),
    )
    monkeypatch.setattr(
        SparseStageStore,
        "load_operational_error",
        lambda _self: terminal,
    )
    monkeypatch.setattr(
        SparseStageStore,
        "completed_stages",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("completed_stages must not run")
        ),
    )

    result = sparse_runtime.run_qwen_sparse_refusal(
        tmp_path,
        run_id="run-1",
        implementation_commit="impl-1",
        corpus_sha256="c" * 64,
        pair_split_sha256="p" * 64,
    )
    assert result is terminal


def test_quality_and_decomposition_contracts_enforce_scientific_gates(
    tmp_path: Path,
) -> None:
    pilot = _store(tmp_path / "pilot")
    pilot.write_stage("prepare", {"status": "completed"})
    with pytest.raises(ValueError, match="hardware_pilot status"):
        pilot.write_stage("hardware_pilot", {"status": "made_up"})

    store = _store(tmp_path)
    for stage in SPARSE_STAGES[:3]:
        store.write_stage(stage, {"status": "completed"})
    with pytest.raises(ValueError, match="sae_quality schema"):
        store.write_stage("sae_quality", {"status": "pass"})
    with pytest.raises(ValueError, match="sae_quality_failed"):
        store.write_stage(
            "sae_quality",
            {
                "status": "pass",
                "explained_variance": 0.79,
                "mean_cosine": 0.95,
                "nonfinite_count": 0,
                "mean_l0": 64.0,
                "dead_feature_fraction": 0.1,
                "norm_drift": 0.1,
            },
        )

    failed = _store(tmp_path / "decompose")
    for stage in SPARSE_STAGES[:4]:
        failed.write_stage(stage, _valid_stage_result(stage))
    with pytest.raises(ValueError, match="direction_reconstruction_failed"):
        failed.write_stage(
            "decompose",
            {
                "status": "pass",
                "cosine": 0.94,
                "relative_l2": 0.2,
                "norm": 1.0,
            },
        )


def test_selection_recomputes_winner_from_decision_trace(tmp_path: Path) -> None:
    freeze = _freeze_fixture()
    freeze["selections"]["sparse_filtered"]["addition"] = dict(
        next(
            row
            for row in freeze["decision_trace"]
            if row["vector"] == "sparse_filtered"
            and row["intervention"] == "addition"
            and row["dose"] == 0.35
            and row["cardinality"] == 8
        )
    )
    freeze["selection_freeze_sha256"] = sparse_runtime.selection_freeze_digest(
        freeze
    )
    store = _store(tmp_path)
    for stage in SPARSE_STAGES[:6]:
        store.write_stage(stage, _valid_stage_result(stage))
    with pytest.raises(ValueError, match="recomputed selection"):
        store.write_stage("select", freeze)


def test_external_transfer_requires_frozen_bijection_and_full_coverage(
    tmp_path: Path,
) -> None:
    prepare = _write_panel_prepare(tmp_path / "00_prepare.json")
    identity = sparse_runtime.frozen_external_transfer_identity(
        prepare,
        expected_prepare_sha256=_file_sha(prepare),
    )
    freeze = _freeze_fixture()
    observations = []
    for dataset in ("mitre", "mitre_frr"):
        for item in identity["datasets"][dataset]:
            for condition in sparse_runtime.EXTERNAL_CONDITIONS:
                observations.append(
                    {
                        "dataset": dataset,
                        **item,
                        "condition": condition,
                        "dose": sparse_runtime.condition_dose(
                            condition,
                            freeze,
                        ),
                        "refused": int(item["example_id"].rsplit("-", 1)[1])
                            % 2
                            == 0,
                        "kl_from_baseline": 0.1,
                        "invalid_generation": False,
                    }
                )
    receipt = {
        "status": "completed",
        "prepare_sha256": identity["prepare_sha256"],
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "observations": observations,
        "metrics": {
            dataset: {
                condition: {
                    "n": 32,
                    "refusal_rate": 0.5,
                    "mean_kl": 0.1,
                    "invalid_generation_rate": 0.0,
                }
                for condition in sparse_runtime.EXTERNAL_CONDITIONS
            }
            for dataset in ("mitre", "mitre_frr")
        },
    }
    sparse_runtime.validate_external_transfer_receipt(receipt, identity, freeze)
    receipt["metrics"]["mitre"]["baseline"]["refusal_rate"] = 0.4
    with pytest.raises(ValueError, match="metrics differ"):
        sparse_runtime.validate_external_transfer_receipt(receipt, identity, freeze)
    receipt["metrics"]["mitre"]["baseline"]["refusal_rate"] = 0.5
    receipt["observations"].pop()
    with pytest.raises(ValueError, match="external transfer coverage"):
        sparse_runtime.validate_external_transfer_receipt(receipt, identity, freeze)


def test_humaneval_stage_validates_frozen_ids_doses_limits_and_worker_outputs(
    tmp_path: Path,
) -> None:
    prepare = _write_panel_prepare(tmp_path / "00_prepare.json")
    identity = sparse_runtime.frozen_humaneval_identity(
        prepare,
        expected_prepare_sha256=_file_sha(prepare),
    )
    freeze = _freeze_fixture()
    jobs = build_code_preservation_jobs(
        worker=tmp_path / "worker.py",
        problems=tmp_path / "problems.jsonl",
        sample_paths={
            condition: tmp_path / f"{condition}.jsonl"
            for condition in ("baseline", "dense", "filtered")
        },
        output_dir=tmp_path / "output",
        identity_dir=tmp_path / "identity",
        temp_root=tmp_path / "temp",
        timeout_seconds=60,
        remote_vm=True,
        identity=identity,
        freeze=freeze,
    )
    outputs = {
        condition: [
            {
                **item,
                "plus_pass": True,
                "refused": False,
                "kl_from_baseline": 0.0,
                "invalid_generation": False,
            }
            for item in identity["items"]
        ]
        for condition in ("baseline", "dense", "filtered")
    }
    receipt = {
        "status": "completed",
        "prepare_sha256": identity["prepare_sha256"],
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "conditions": ["baseline", "dense", "filtered"],
        "doses": {
            "baseline": 0.0,
            "dense": 0.5,
            "filtered": 0.2,
        },
        "memory_bytes": 1024**3,
        "timeout_seconds": 60,
        "parallelism": 1,
        "remote_vm": True,
        "evalplus_revision": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "job_manifest_sha256": jobs["job_manifest_sha256"],
        "worker_outputs": outputs,
        "metrics": {
            condition: {
                "n": 32,
                "pass_at_1": 1.0,
                "refusal_rate": 0.0,
                "mean_kl": 0.0,
                "invalid_generation_rate": 0.0,
            }
            for condition in ("baseline", "dense", "filtered")
        },
    }
    sparse_runtime.validate_code_preservation_receipt(
        receipt,
        identity=identity,
        freeze=freeze,
    )
    receipt["metrics"]["filtered"]["pass_at_1"] = 0.5
    with pytest.raises(ValueError, match="metrics differ"):
        sparse_runtime.validate_code_preservation_receipt(
            receipt,
            identity=identity,
            freeze=freeze,
        )
    receipt["metrics"]["filtered"]["pass_at_1"] = 1.0
    receipt["worker_outputs"]["filtered"].pop()
    with pytest.raises(ValueError, match="HumanEval\\+ coverage"):
        sparse_runtime.validate_code_preservation_receipt(
            receipt,
            identity=identity,
            freeze=freeze,
        )


def test_injected_runners_cannot_bypass_external_or_humaneval_validation(
    tmp_path: Path,
) -> None:
    prepare = _write_panel_prepare(tmp_path / "00_prepare.json")
    prepare_sha256 = _file_sha(prepare)
    freeze = _freeze_fixture()

    external_store = _store(
        tmp_path / "external",
        prepare_path=prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    _complete_through_review(external_store)
    external_identity = sparse_runtime.frozen_external_transfer_identity(
        prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    invalid_external = _external_receipt(external_identity, freeze)
    invalid_external["observations"].pop()
    with pytest.raises(ValueError, match="external transfer coverage"):
        SparseRuntime(external_store).run_stage(
            "external_transfer",
            lambda **_kwargs: invalid_external,
        )

    code_store = _store(
        tmp_path / "code",
        prepare_path=prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    _complete_through_review(code_store)
    code_store.write_stage(
        "external_transfer",
        _external_receipt(external_identity, freeze),
    )
    code_identity = sparse_runtime.frozen_humaneval_identity(
        prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    invalid_code = _code_receipt(tmp_path / "code-inputs", code_identity, freeze)
    invalid_code["worker_outputs"]["filtered"].pop()
    with pytest.raises(ValueError, match="HumanEval\\+ coverage"):
        SparseRuntime(code_store).run_stage(
            "code_preservation",
            lambda **_kwargs: invalid_code,
        )


def test_persisted_scientific_receipts_are_revalidated_on_resume(
    tmp_path: Path,
) -> None:
    prepare = _write_panel_prepare(tmp_path / "00_prepare.json")
    prepare_sha256 = _file_sha(prepare)
    freeze = _freeze_fixture()
    store = _store(
        tmp_path,
        prepare_path=prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    _complete_through_review(store)
    external_identity = sparse_runtime.frozen_external_transfer_identity(
        prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    store.write_stage(
        "external_transfer",
        _external_receipt(external_identity, freeze),
    )
    external_path = store.stage_path("external_transfer")
    external_payload = json.loads(external_path.read_text(encoding="utf-8"))
    external_payload["results"]["metrics"]["mitre"]["baseline"][
        "refusal_rate"
    ] = 0.4
    external_path.write_text(json.dumps(external_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics differ"):
        store.load_stage("external_transfer")
    external_payload["results"]["metrics"]["mitre"]["baseline"][
        "refusal_rate"
    ] = 0.5
    external_path.write_text(json.dumps(external_payload), encoding="utf-8")

    code_identity = sparse_runtime.frozen_humaneval_identity(
        prepare,
        expected_prepare_sha256=prepare_sha256,
    )
    store.write_stage(
        "code_preservation",
        _code_receipt(tmp_path / "code-inputs", code_identity, freeze),
    )

    payload = json.loads(
        store.stage_path("code_preservation").read_text(encoding="utf-8")
    )
    payload["results"]["metrics"]["filtered"]["pass_at_1"] = 0.5
    store.stage_path("code_preservation").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metrics differ"):
        store.load_stage("code_preservation")


def test_alternate_profile_without_allowlisted_digest_is_rejected(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "alternate.json").write_text(
        json.dumps(_config()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approved sparse profile"):
        sparse_runtime.run_qwen_sparse_refusal(
            tmp_path,
            "alternate.json",
            run_id="run",
            implementation_commit="impl",
            corpus_sha256="c" * 64,
            pair_split_sha256="p" * 64,
        )


def test_scientific_halts_and_operational_errors_remain_distinct(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for stage in SPARSE_STAGES[:3]:
        store.write_stage(stage, _valid_stage_result(stage))
    halted = SparseRuntime(store).run_stage(
        "sae_quality",
        lambda **_kwargs: {
            "status": "sae_quality_failed",
            "explained_variance": 0.79,
            "mean_cosine": 0.9,
            "nonfinite_count": 0,
            "mean_l0": 64.0,
            "dead_feature_fraction": 0.2,
            "norm_drift": 0.2,
        },
    )
    assert halted["results"]["status"] == "sae_quality_failed"
    assert not store.stage_path("decompose").exists()
    assert not store.operational_error_path.exists()

    resource = _store(tmp_path / "resource")
    resource.write_stage("prepare", {"status": "completed"})
    projected = SparseRuntime(resource).run_stage(
        "hardware_pilot",
        lambda **_kwargs: {"status": "resource_projection_failed"},
    )
    assert projected["results"]["status"] == "resource_projection_failed"
    assert not resource.stage_path("sae_fit").exists()
    assert not resource.operational_error_path.exists()

    other = _store(tmp_path / "operational")
    with pytest.raises(RuntimeError, match="boom"):
        SparseRuntime(other).run_stage(
            "prepare",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    error = json.loads(other.operational_error_path.read_text(encoding="utf-8"))
    assert error["status"] == "operational_error"
    assert error["operational_success"] is False


def test_external_and_code_evaluation_require_complete_blind_review(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    store.write_stage(
        "test",
        {
            "status": "completed",
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "test_split_sha256": store.test_split_sha256,
            "observations": expected_test_matrix(_frozen_test_pairs()),
        },
    )
    store.write_stage(
        "review_export",
        {
            "status": "review_pending",
            "review_count": 512,
            "test_split_sha256": store.test_split_sha256,
            "blind_review_sha256": "b" * 64,
            "private_manifest_sha256": "a" * 64,
        },
    )
    store.write_stage(
        "review_merge",
        {
            "status": "inconclusive_review",
            "test_split_sha256": store.test_split_sha256,
            "blind_review_sha256": "b" * 64,
            "private_manifest_sha256": "a" * 64,
        },
    )
    runtime = SparseRuntime(store)
    with pytest.raises(ValueError, match="complete blind review"):
        runtime.run_stage("external_transfer", lambda **_kwargs: {})
    terminal = runtime.run_stage("code_preservation", lambda **_kwargs: {})
    assert terminal["status"] == "operational_error"
    assert terminal["stage"] == "external_transfer"


def test_review_merge_must_match_exported_blind_hash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    store.write_stage(
        "test",
        {
            "status": "completed",
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "test_split_sha256": store.test_split_sha256,
            "observations": expected_test_matrix(_frozen_test_pairs()),
        },
    )
    store.write_stage(
        "review_export",
        {
            "status": "review_pending",
            "review_count": 512,
            "test_split_sha256": store.test_split_sha256,
            "blind_review_sha256": "b" * 64,
            "private_manifest_sha256": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="08_review_export"):
        SparseRuntime(store).run_stage(
            "review_merge",
            lambda **_kwargs: {
                "status": "inconclusive_review",
                "test_split_sha256": store.test_split_sha256,
                "blind_review_sha256": "c" * 64,
                "private_manifest_sha256": "a" * 64,
            },
        )


def test_persisted_review_merge_revalidates_frozen_test_split(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_review(store)
    payload = json.loads(
        store.stage_path("review_merge").read_text(encoding="utf-8")
    )
    payload["results"]["test_split_sha256"] = "0" * 64
    store.stage_path("review_merge").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen test split"):
        store.load_stage("review_merge")
    payload["results"]["test_split_sha256"] = store.test_split_sha256
    payload["results"]["blind_review_sha256"] = "c" * 64
    store.stage_path("review_merge").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="08_review_export blind hash"):
        store.load_stage("review_merge")


def test_humaneval_plus_reuses_remote_limited_worker_for_frozen_conditions(
    tmp_path: Path,
) -> None:
    sample_paths = {
        condition: tmp_path / f"{condition}.jsonl"
        for condition in ("baseline", "dense", "filtered")
    }
    jobs = build_code_preservation_jobs(
        worker=tmp_path / "qwen_evalplus_worker.py",
        problems=tmp_path / "problems.jsonl",
        sample_paths=sample_paths,
        output_dir=tmp_path / "output",
        identity_dir=tmp_path / "identity",
        temp_root=tmp_path / "temp",
        timeout_seconds=60,
        remote_vm=True,
        identity={
            "prepare_sha256": "a" * 64,
            "items": tuple(
                {
                    "example_id": f"id-{index}",
                    "content_sha256": _sha(f"id-{index}"),
                }
                for index in range(32)
            ),
        },
        freeze=_freeze_fixture(),
    )
    assert jobs["conditions"] == ("baseline", "dense", "filtered")
    assert jobs["parallelism"] == 1
    assert jobs["memory_bytes"] == 1024**3
    assert set(jobs["jobs"]) == {"baseline", "dense", "filtered"}
    assert all(
        str(tmp_path / "qwen_evalplus_worker.py") in spec["command"]
        for spec in jobs["jobs"].values()
    )
    with pytest.raises(ValueError, match="remote VM"):
        build_code_preservation_jobs(
            worker=tmp_path / "qwen_evalplus_worker.py",
            problems=tmp_path / "problems.jsonl",
            sample_paths=sample_paths,
            output_dir=tmp_path / "output",
            identity_dir=tmp_path / "identity",
            temp_root=tmp_path / "temp",
            timeout_seconds=60,
            remote_vm=False,
            identity={
                "prepare_sha256": "a" * 64,
                "items": tuple(
                    {
                        "example_id": f"id-{index}",
                        "content_sha256": _sha(f"id-{index}"),
                    }
                    for index in range(32)
                ),
            },
            freeze=_freeze_fixture(),
        )
