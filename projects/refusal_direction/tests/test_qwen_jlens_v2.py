from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_jlens import (
    FINAL_LENS_FILENAME,
    CorpusItem,
    JLENS_V2_PROFILE,
    JLensCorpus,
    V2_FINAL_LENS_FILENAME,
    V2_FIT_COUNT_CANDIDATES,
    V2_HOLDOUT_COUNT,
    V2_MAX_FIT_COUNT,
    V2_MAX_SEQ_LEN,
    load_fit_checkpoint,
    jlens_metadata_contains_leak,
    select_fit_count,
    slice_v2_corpus,
    summarize_dual_holdout_metrics,
    summarize_holdout_subset_stability,
    validate_checkpoint_structure,
    write_fit_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def _identity() -> dict[str, Any]:
    return {
        "source_layers": [0, 1, 2],
        "target_layer": 3,
        "skip_first": 16,
        "corpus_hash": "a" * 64,
        "config_hash": "b" * 64,
        "model_revision": QWEN_REVISION,
        "implementation_commit": "c" * 40,
        "run_id": "jlens-run-v2",
    }


def _checkpoint_payload(
    *,
    n_done: int,
    next_idx: int,
    fit_count: int,
    dim_batch: int = 8,
) -> dict[str, Any]:
    payload = {
        "jacobian_sum": {0: torch.eye(4)},
        "n_done": n_done,
        "next_idx": next_idx,
        "dim_batch": dim_batch,
        "fit_count": fit_count,
        **_identity(),
    }
    return payload


def test_v2_profile_is_isolated_from_v1_lab_and_code_eval() -> None:
    v1 = load_config(ROOT, profile="qwen25_coder_7b_jlens.json")
    v2 = load_config(ROOT, profile=JLENS_V2_PROFILE)
    lab = load_config(ROOT, profile="qwen25_coder_7b_lab.json")

    assert v2["max_seq_len"] == V2_MAX_SEQ_LEN == 256
    assert v2["min_tokens_before_truncation"] == V2_MAX_SEQ_LEN
    assert v2["holdout_count"] == V2_HOLDOUT_COUNT == 200
    assert v2["fit_count_candidates"] == V2_FIT_COUNT_CANDIDATES
    assert v2["max_fit_count"] == V2_MAX_FIT_COUNT
    assert "fit_count" not in v2
    assert v2["max_fit_count"] == max(v2["fit_count_candidates"])
    assert all(0 < candidate <= v2["max_fit_count"] for candidate in v2["fit_count_candidates"])
    assert v2["results_dir"] == "results/qwen25_coder_7b_jlens_v2"
    assert v2["artifacts_dir"] == "artifacts/qwen25_coder_7b_jlens_v2"
    assert v2["source_v1_lens"].endswith(
        "qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert V2_FINAL_LENS_FILENAME == "qwen25_coder_7b_jacobian_lens_v2.pt"
    assert V2_FINAL_LENS_FILENAME != FINAL_LENS_FILENAME
    assert v1["artifacts_dir"] != v2["artifacts_dir"]
    assert lab["source_lens"] == v2["source_v1_lens"]


def _corpus_item(split: str, index: int) -> CorpusItem:
    text = f"{split}-function-{index}"
    return CorpusItem(
        example_id=f"test:{split}:{index}",
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=512,
        fit_text=text,
        source_index=index,
    )


def _corpus(*, fit: int, holdout: int) -> JLensCorpus:
    fit_items = tuple(_corpus_item("fit", index) for index in range(fit))
    holdout_items = tuple(_corpus_item("holdout", index) for index in range(holdout))
    return JLensCorpus(
        fit=fit_items,
        holdout=holdout_items,
        dataset_revision="test-revision",
        n_scanned=fit + holdout,
        n_missing_code_or_docstring=0,
        n_too_short=0,
        n_duplicate_dropped=0,
        n_excluded=0,
        exclusions_available=False,
    )


def test_select_fit_count_picks_largest_safe_candidate() -> None:
    selected = select_fit_count(
        probe_seconds=80.0,
        candidates=[100, 150, 200, 250, 300, 400, 600, 800],
        limit_seconds=43_200,
    )
    assert selected.n_prompts == 400
    assert selected.projected_seconds == 32_000.0


def test_select_fit_count_rejects_when_minimum_exceeds_limit() -> None:
    with pytest.raises(ValueError, match="minimum candidate"):
        select_fit_count(
            probe_seconds=500.0,
            candidates=[100, 150],
            limit_seconds=43_200,
        )


def test_v2_checkpoint_accepts_progress_above_v1_limit(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    write_fit_checkpoint(
        path,
        jacobian_sum={0: torch.eye(4)},
        n_done=150,
        next_idx=150,
        dim_batch=8,
        fit_count=200,
        **_identity(),
    )
    payload = load_fit_checkpoint(path)
    validate_checkpoint_structure(payload, fit_count=200)
    assert payload["fit_count"] == 200


def test_v2_checkpoint_rejects_progress_past_selected_fit(tmp_path: Path) -> None:
    payload = _checkpoint_payload(n_done=201, next_idx=201, fit_count=200)
    with pytest.raises(ValueError, match="outside 0..200"):
        validate_checkpoint_structure(payload, fit_count=200)


def test_v2_checkpoint_rejects_mismatched_selected_fit_count() -> None:
    payload = _checkpoint_payload(n_done=50, next_idx=50, fit_count=200)
    with pytest.raises(ValueError, match="fit_count"):
        validate_checkpoint_structure(payload, fit_count=150)


def test_slice_v2_corpus_keeps_holdout_fixed() -> None:
    pool = _corpus(fit=800, holdout=200)
    small = slice_v2_corpus(pool, fit_count=100, max_fit_count=800)
    large = slice_v2_corpus(pool, fit_count=600, max_fit_count=800)
    assert small.fit == pool.fit[:100]
    assert large.fit == pool.fit[:600]
    assert small.holdout == large.holdout == pool.holdout
    assert set(item.content_hash for item in large.fit).isdisjoint(
        item.content_hash for item in large.holdout
    )

def _records(
    method: str,
    *,
    ids: list[str],
    positions: tuple[int, ...] = (16, 17),
    layers: tuple[int, ...] = (0, 1),
    target_offset: int = 0,
    logit_rank: int = 7,
) -> list[dict[str, Any]]:
    jlens_rank = 1 if method == "v2" else 3
    records = []
    for example_id in ids:
        for layer in layers:
            for position in positions:
                records.append(
                    {
                        "example_id": example_id,
                        "content_hash": hashlib.sha256(example_id.encode()).hexdigest(),
                        "position": position,
                        "layer": layer,
                        "target_token_id": 5 + position + target_offset,
                        "jlens_rank": jlens_rank,
                        "logit_lens_rank": logit_rank,
                    }
                )
    return records


def test_dual_holdout_comparison_requires_shared_observations() -> None:
    comparison = summarize_dual_holdout_metrics(
        v2_records=_records("v2", ids=["a", "b"]),
        v1_records=_records("v1", ids=["a", "b"]),
        source_layers=[0, 1],
    )
    assert set(comparison["methods"]) == {"jlens_v2", "jlens_v1", "logit_lens"}
    assert comparison["shared_holdout_ids"] == ["a", "b"]
    assert comparison["n_observations"] == 8
    assert len(comparison["stability"]["halves"]) == 2
    overall = {name: block["overall"] for name, block in comparison["methods"].items()}
    assert overall["jlens_v2"]["mean_rank"] == 1.0
    assert overall["jlens_v1"]["mean_rank"] == 3.0
    assert overall["logit_lens"]["mean_rank"] == 7.0
    assert overall["jlens_v2"]["pass@1"] == 1.0
    assert overall["jlens_v1"]["pass@5"] == 1.0
    assert set(comparison["methods"]["jlens_v2"]["by_layer"]) == {"0", "1"}
    item = comparison["by_item"]["a"]["0"]
    assert item["n_positions"] == 2
    assert item["jlens_v2_rank_sum"] == 2
    assert item["jlens_v1_rank_sum"] == 6
    assert item["logit_lens_rank_sum"] == 14
    assert not jlens_metadata_contains_leak(comparison)


def test_dual_holdout_rejects_different_ids() -> None:
    with pytest.raises(ValueError, match="same holdout"):
        summarize_dual_holdout_metrics(
            v2_records=_records("v2", ids=["a"]),
            v1_records=_records("v1", ids=["b"]),
            source_layers=[0],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"positions": (16, 18)},
        {"layers": (0, 2)},
        {"target_offset": 1},
    ],
    ids=["position", "layer", "target_token_id"],
)
def test_dual_holdout_rejects_mismatched_observation_identity(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="observation identity"):
        summarize_dual_holdout_metrics(
            v2_records=_records("v2", ids=["a", "b"], layers=(0, 1, 2)),
            v1_records=_records("v1", ids=["a", "b"], **{"layers": (0, 1, 2), **kwargs}),
            source_layers=[0, 1, 2],
        )


def test_dual_holdout_rejects_content_hash_and_order_mismatch() -> None:
    v2 = _records("v2", ids=["a", "b"])
    swapped = _records("v1", ids=["a", "b"])
    swapped[0]["content_hash"] = "f" * 64
    with pytest.raises(ValueError, match="observation identity"):
        summarize_dual_holdout_metrics(v2, swapped, [0, 1])
    reordered = list(reversed(_records("v1", ids=["a", "b"])))
    with pytest.raises(ValueError, match="same holdout|observation identity"):
        summarize_dual_holdout_metrics(v2, reordered, [0, 1])


def test_dual_holdout_rejects_disagreeing_logit_lens_reference() -> None:
    with pytest.raises(ValueError, match="logit lens"):
        summarize_dual_holdout_metrics(
            v2_records=_records("v2", ids=["a"], logit_rank=7),
            v1_records=_records("v1", ids=["a"], logit_rank=8),
            source_layers=[0, 1],
        )


def test_dual_holdout_rejects_empty_and_unlisted_layers() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_dual_holdout_metrics([], [], [0])
    with pytest.raises(ValueError, match="source_layers"):
        summarize_dual_holdout_metrics(
            _records("v2", ids=["a"]), _records("v1", ids=["a"]), [0]
        )


def test_subset_stability_splits_ids_into_contiguous_balanced_parts() -> None:
    ids = [f"id-{index}" for index in range(5)]
    stability = summarize_holdout_subset_stability(
        _records("v2", ids=ids), source_layers=[0, 1], n_parts=2
    )
    first, second = stability["halves"]
    assert first["holdout_ids"] == ids[:3]
    assert second["holdout_ids"] == ids[3:]
    assert first["n_items"] == 3 and second["n_items"] == 2
    assert first["overall"]["jlens"]["n_observations"] == 12
    assert second["overall"]["logit_lens"]["mean_rank"] == 7.0


def test_subset_stability_splits_200_ids_into_first_and_second_100() -> None:
    ids = [f"id-{index}" for index in range(200)]
    stability = summarize_holdout_subset_stability(
        _records("v2", ids=ids, positions=(16,), layers=(0,)),
        source_layers=[0],
    )
    first, second = stability["halves"]
    assert first["holdout_ids"] == ids[:100]
    assert second["holdout_ids"] == ids[100:]


def test_subset_stability_rejects_invalid_part_count() -> None:
    with pytest.raises(ValueError, match="n_parts"):
        summarize_holdout_subset_stability(
            _records("v2", ids=["a"]), source_layers=[0, 1], n_parts=2
        )
    with pytest.raises(ValueError, match="n_parts"):
        summarize_holdout_subset_stability(
            _records("v2", ids=["a", "b"]), source_layers=[0, 1], n_parts=0
        )


def test_v2_documentation_names_safety_boundaries() -> None:
    text = (ROOT / "JLENS_V2.md").read_text(encoding="utf-8")
    for required in (
        "512 tokens",
        "200 amostras",
        "checkpoint",
        "12 horas",
        "não substitui",
        "checkpoint-backup",
    ):
        assert required in text
