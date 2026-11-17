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
    select_fit_count,
    slice_v2_corpus,
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

    assert v2["max_seq_len"] == V2_MAX_SEQ_LEN == 512
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