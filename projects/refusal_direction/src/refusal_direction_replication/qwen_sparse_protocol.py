"""Frozen identities for the Qwen sparse-refusal steering protocol."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


SPARSE_PROFILE = "qwen25_coder_sparse_refusal.json"
SPARSE_SCHEMA_VERSION = 1
EXPECTED_SPARSE_PROFILE_SHA256 = (
    "16a90b88e15dae64fac7f987bb4a6efec9c5714d3bbfaa9bc0d57ec796a785d2"
)
PAIR_SPLIT_SIZES = {"discovery": 128, "selection": 64, "test": 64}
PAIR_SPLIT_SEED = 2029
PURPLELLAMA_REVISION = "0654501399e7cbd560db9501965dc09c567a5563"
CORPUS_SOURCES = (
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


def load_sparse_config(root: Path) -> dict[str, Any]:
    """Load and verify the sole approved sparse-refusal profile."""
    path = Path(root) / "configs" / SPARSE_PROFILE
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("sparse profile must contain a JSON object")
    verify_sparse_profile_digest(config)
    return config


def verify_sparse_profile_digest(config: Mapping[str, Any]) -> None:
    """Reject any semantic change to the frozen profile."""
    actual = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != EXPECTED_SPARSE_PROFILE_SHA256:
        raise ValueError(f"sparse profile digest changed: {actual}")
