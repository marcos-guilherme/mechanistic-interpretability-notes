from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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


def _store(tmp_path: Path) -> SparseStageStore:
    return SparseStageStore(
        tmp_path / "results",
        config=_config(),
        run_id="run-1",
        implementation_commit="impl-1",
        corpus_sha256="c" * 64,
        pair_split_sha256="p" * 64,
    )


def _complete_through_select(store: SparseStageStore) -> None:
    for stage in SPARSE_STAGES[:6]:
        store.write_stage(stage, {"status": "completed"})
    store.write_stage(
        "select",
        {
            "status": "selection_frozen",
            "selection_freeze_sha256": "f" * 64,
        },
    )
    assert store.selection_freeze_path.is_file()


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
                            "vector_sha256": f"{vector}:{cardinality}",
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

    pre, forward = sparse_condition_hook_specs(
        condition="filtered_addition",
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        vectors=vectors,
        doses=doses,
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
        doses=doses,
    )
    assert len(pre) == 24
    assert len(forward) == 48


def test_compact_test_matrix_has_all_nine_conditions_without_leaks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _complete_through_select(store)
    pair_ids = [f"pair-{index:04d}" for index in range(64)]
    observations = expected_test_matrix(pair_ids)
    payload = SparseRuntime(store).run_stage(
        "test",
        lambda **_kwargs: {
            "status": "completed",
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


def test_scientific_halts_and_operational_errors_remain_distinct(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    for stage in SPARSE_STAGES[:3]:
        store.write_stage(stage, {"status": "completed"})
    halted = SparseRuntime(store).run_stage(
        "sae_quality",
        lambda **_kwargs: {"status": "sae_quality_failed"},
    )
    assert halted["results"]["status"] == "sae_quality_failed"
    assert not store.stage_path("decompose").exists()
    assert not store.operational_error_path.exists()

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
    store.write_stage("test", {"status": "completed", "observations": []})
    store.write_stage("review_export", {"status": "review_pending", "review_count": 512})
    store.write_stage("review_merge", {"status": "inconclusive_review"})
    runtime = SparseRuntime(store)
    with pytest.raises(ValueError, match="complete blind review"):
        runtime.run_stage("external_transfer", lambda **_kwargs: {})
    with pytest.raises(ValueError, match="complete blind review"):
        runtime.run_stage("code_preservation", lambda **_kwargs: {})


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
        timeout_seconds=10,
        remote_vm=True,
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
            timeout_seconds=10,
            remote_vm=False,
        )
