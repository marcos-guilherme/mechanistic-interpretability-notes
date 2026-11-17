from __future__ import annotations

from pathlib import Path

import pytest

from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_causal_jlens import (
    CAUSAL_JLENS_PROFILE,
    aggregate_fidelity,
    cluster_bootstrap_difference,
    summarize_behavior_changes,
)

ROOT = Path(__file__).resolve().parents[1]


def _observation(
    *,
    rank: int,
    example_id: str,
    step: int,
    dataset: str = "mitre",
    condition: str = "baseline",
    layer: int = 25,
    lens_version: str = "v1",
) -> dict:
    return {
        "dataset": dataset,
        "example_id": example_id,
        "condition": condition,
        "generation_step": step,
        "layer": layer,
        "lens_version": lens_version,
        "rank": rank,
    }


def _paired_records(*, prompt_count: int, steps: int) -> list[dict]:
    records: list[dict] = []
    for index in range(prompt_count):
        example_id = f"p-{index}"
        for step in range(1, steps + 1):
            records.append(
                _observation(
                    rank=1,
                    example_id=example_id,
                    step=step,
                    condition="baseline",
                )
            )
            fail_rank = 6 if step <= 2 else 1
            records.append(
                _observation(
                    rank=fail_rank,
                    example_id=example_id,
                    step=step,
                    condition="ablation",
                )
            )
    return records


def _responses() -> list[dict]:
    return [
        {
            "dataset": "mitre",
            "example_id": "changed-1",
            "condition": "baseline",
            "refusal": False,
        },
        {
            "dataset": "mitre",
            "example_id": "changed-1",
            "condition": "ablation",
            "refusal": True,
        },
        {
            "dataset": "mitre",
            "example_id": "unchanged-1",
            "condition": "baseline",
            "refusal": True,
        },
        {
            "dataset": "mitre",
            "example_id": "unchanged-1",
            "condition": "ablation",
            "refusal": True,
        },
    ]


def _observations() -> list[dict]:
    rows: list[dict] = []
    for step in range(1, 9):
        rows.append(
            _observation(
                rank=4,
                example_id="changed-1",
                step=step,
                condition="baseline",
            )
        )
        rows.append(
            _observation(
                rank=2,
                example_id="changed-1",
                step=step,
                condition="ablation",
            )
        )
    return rows


def test_profile_freezes_the_approved_protocol() -> None:
    config = load_config(ROOT, profile=CAUSAL_JLENS_PROFILE)
    assert config["datasets"] == ["mitre", "mitre_frr"]
    assert config["dataset_size"] == 32
    assert config["conditions"] == [
        "baseline",
        "ablation",
        "addition",
        "random_control",
    ]
    assert config["max_new_tokens"] == 64
    assert config["readout_steps"] == 8
    assert config["readout_layers"] == [19, 22, 25, 26]
    assert config["bootstrap_resamples"] == 10_000
    assert config["bootstrap_seed"] == 2028
    assert config["equivalence_margin"] == 0.05


def test_aggregate_fidelity_uses_one_based_rank() -> None:
    records = [
        _observation(rank=1, example_id="a", step=1),
        _observation(rank=5, example_id="a", step=2),
        _observation(rank=6, example_id="b", step=1),
    ]
    block = aggregate_fidelity(records)["mitre"]["baseline"]["v1"]["25"]
    assert block == {
        "n": 3,
        "prompt_count": 2,
        "pass_at_1": pytest.approx(1 / 3),
        "pass_at_5": pytest.approx(2 / 3),
        "pass_at_10": 1.0,
        "mean_rank": 4.0,
    }


def test_cluster_bootstrap_resamples_prompts_not_tokens() -> None:
    records = _paired_records(prompt_count=4, steps=8)
    result = cluster_bootstrap_difference(
        records,
        condition="ablation",
        k=5,
        layer=25,
        lens_version="v1",
        dataset="mitre",
        resamples=10_000,
        seed=2028,
    )
    assert result["cluster_unit"] == "example_id"
    assert result["resamples"] == 10_000
    assert result["seed"] == 2028
    assert result["estimate"] == pytest.approx(-0.25)


def test_behavior_change_summary_uses_prompt_level_mean_rank() -> None:
    result = summarize_behavior_changes(_responses(), _observations())
    changed = result["mitre"]["ablation"]["changed"]["v1"]
    assert changed["prompt_count"] == 1
    assert changed["layer"] == 25
    assert changed["mean_rank_delta"] == pytest.approx(-2.0)