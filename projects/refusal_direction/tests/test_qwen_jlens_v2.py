from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.pipeline import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_v2_profile_is_isolated_from_v1_lab_and_code_eval() -> None:
    v1 = load_config(ROOT, profile="qwen25_coder_7b_jlens.json")
    v2 = load_config(ROOT, profile="qwen25_coder_7b_jlens_v2.json")
    lab = load_config(ROOT, profile="qwen25_coder_7b_lab.json")

    assert v2["max_seq_len"] == 512
    assert v2["min_tokens_before_truncation"] == 512
    assert v2["holdout_count"] == 200
    assert v2["fit_count_candidates"] == [100, 150, 200, 250, 300, 400, 600, 800]
    assert v2["results_dir"] == "results/qwen25_coder_7b_jlens_v2"
    assert v2["artifacts_dir"] == "artifacts/qwen25_coder_7b_jlens_v2"
    assert v2["source_v1_lens"].endswith(
        "qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert v1["artifacts_dir"] != v2["artifacts_dir"]
    assert lab["source_lens"] == v2["source_v1_lens"]