from copy import deepcopy
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_sparse_protocol import (
    CORPUS_SOURCES,
    EXPECTED_SPARSE_PROFILE_SHA256,
    PAIR_SPLIT_SIZES,
    SPARSE_SCHEMA_VERSION,
    load_sparse_config,
    verify_sparse_profile_digest,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_sparse_profile_is_frozen() -> None:
    config = load_sparse_config(PROJECT_ROOT)
    assert config["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["model_revision"] == (
        "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
    assert config["direction_sha256"] == (
        "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
    )
    assert config["sae"] == {
        "layer": 18,
        "d_in": 3584,
        "d_sae": 16384,
        "k": 64,
        "fit_tokens": 4_750_000,
        "holdout_tokens": 250_000,
        "sequence_length": 256,
        "effective_batch_tokens": 8192,
        "microbatch_candidates": [4096, 2048, 1024, 512],
        "checkpoint_every_tokens": 250_000,
        "max_projected_hours": 12,
    }
    assert config["pair_splits"] == {
        "discovery": 128,
        "selection": 64,
        "test": 64,
        "split_seed": 2029,
    }
    assert config["bootstrap_seed"] == 2029
    assert config["bootstrap_resamples"] == 10_000
    assert SPARSE_SCHEMA_VERSION == 1
    assert PAIR_SPLIT_SIZES == {"discovery": 128, "selection": 64, "test": 64}
    assert len(EXPECTED_SPARSE_PROFILE_SHA256) == 64
    verify_sparse_profile_digest(config)


def test_sparse_profile_digest_rejects_changes() -> None:
    config = deepcopy(load_sparse_config(PROJECT_ROOT))
    config["seed"] = 43
    with pytest.raises(ValueError, match="sparse profile digest changed"):
        verify_sparse_profile_digest(config)


def test_corpus_source_identities_are_frozen() -> None:
    assert CORPUS_SOURCES == (
        {
            "repo_id": "code-search-net/code_search_net",
            "config": "python",
            "split": "train",
            "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
            "target_tokens": 2_750_000,
        },
        {
            "repo_id": "HuggingFaceH4/ultrachat_200k",
            "config": "default",
            "split": "train_sft",
            "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
            "target_tokens": 1_250_000,
        },
        {
            "repo_id": "rufimelo/DeltaSecommits",
            "config": "default",
            "split": "train",
            "revision": "40de1664860b8e37fb2322ab2b2fbd0b248f3598",
            "target_tokens": 1_000_000,
        },
    )
