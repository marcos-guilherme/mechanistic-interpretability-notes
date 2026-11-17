from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_jlens import (
    FINAL_LENS_FILENAME,
    JLENS_V2_PROFILE,
    V2_FINAL_LENS_FILENAME,
    V2_FIT_COUNT_CANDIDATES,
    V2_HOLDOUT_COUNT,
    V2_MAX_FIT_COUNT,
    V2_MAX_SEQ_LEN,
)


ROOT = Path(__file__).resolve().parents[1]


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