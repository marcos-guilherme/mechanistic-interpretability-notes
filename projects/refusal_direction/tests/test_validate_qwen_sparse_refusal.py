from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path
from typing import Any

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "validate_qwen_sparse_refusal.py"
PROFILE = PROJECT / "configs" / "qwen25_coder_sparse_refusal.json"
STAGES = (
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
CONDITIONS = (
    "baseline",
    "dense_addition",
    "dense_ablation",
    "unfiltered_addition",
    "unfiltered_ablation",
    "filtered_addition",
    "filtered_ablation",
    "random_addition",
    "random_ablation",
)
PRIMARY = (
    "dense_addition",
    "filtered_addition",
    "dense_ablation",
    "filtered_ablation",
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("sparse_validator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bootstrap(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    side, orientation = {
        "offensive_delta": ("offensive", 1.0),
        "defensive_improvement": ("defensive", -1.0),
        "defensive_recovery_delta": ("defensive", -1.0),
        "offensive_retention_improvement": ("offensive", 1.0),
    }[metric]
    intervention = "addition" if metric in {
        "offensive_delta",
        "defensive_improvement",
    } else "ablation"
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["side"] == side and row["condition"].endswith(intervention):
            family = row["condition"].split("_", 1)[0]
            grouped.setdefault(row["pair_id"], {})[family] = float(
                row["label"] == "refuse"
                if metric != "defensive_recovery_delta"
                else row["label"] != "answer"
            )
    pair_ids = tuple(sorted(grouped))
    contrasts = {
        pair_id: orientation
        * (values["filtered"] - values["dense"])
        for pair_id, values in grouped.items()
    }
    estimate = statistics.fmean(contrasts.values())
    rng = random.Random(2029)
    draws = sorted(
        statistics.fmean(
            contrasts[pair_id]
            for pair_id in rng.choices(pair_ids, k=len(pair_ids))
        )
        for _ in range(10_000)
    )
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_lower": draws[249],
        "ci_upper": draws[9749],
        "pair_count": 64,
        "resamples": 10_000,
        "seed": 2029,
        "pair_ids": list(pair_ids),
    }


def write_complete_sparse_fixture(root: Path) -> Path:
    results = root / "results"
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    pair_rows = []
    split_sizes = {"discovery": 128, "selection": 64, "test": 64}
    cursor = 0
    for split, size in split_sizes.items():
        for index in range(cursor, cursor + size):
            pair_rows.append(
                {
                    "pair_id": f"pair-{index:04d}",
                    "split": split,
                    "category": f"category-{index % 8}",
                    "source_identity": f"source-{index:04d}",
                    "offensive_sha256": _sha(f"offensive-{index}"),
                    "defensive_sha256": _sha(f"defensive-{index}"),
                    "offensive_normalized_sha256": _sha(
                        f"normalized-offensive-{index}"
                    ),
                    "defensive_normalized_sha256": _sha(
                        f"normalized-defensive-{index}"
                    ),
                }
            )
        cursor += size
    test_pairs = [row for row in pair_rows if row["split"] == "test"]
    observations = [
        {
            "pair_id": pair["pair_id"],
            "side": side,
            "prompt_sha256": pair[f"{side}_sha256"],
            "condition": condition,
        }
        for pair in test_pairs
        for side in ("offensive", "defensive")
        for condition in CONDITIONS
    ]
    reviews = []
    for pair in test_pairs:
        index = int(pair["pair_id"].split("-")[1])
        for side in ("offensive", "defensive"):
            for condition in PRIMARY:
                if side == "offensive":
                    refused = condition.startswith("dense") or index % 8 != 0
                else:
                    refused = condition.startswith("dense") and index % 2 == 0
                review_id = _sha(
                    f"{pair['pair_id']}:{side}:{condition}"
                )[:24]
                reviews.append(
                    {
                        "review_id": review_id,
                        "pair_id": pair["pair_id"],
                        "side": side,
                        "condition": condition,
                        "prompt_sha256": pair[f"{side}_sha256"],
                        "generation_sha256": _sha(f"generation:{review_id}"),
                        "label": "refuse" if refused else "answer",
                    }
                )
    bootstrap = {
        metric: _bootstrap(reviews, metric)
        for metric in (
            "offensive_delta",
            "defensive_improvement",
            "defensive_recovery_delta",
            "offensive_retention_improvement",
        )
    }
    external = [
        {
            "dataset": dataset,
            "example_id": f"{dataset}-{index}",
            "content_sha256": _sha(f"{dataset}-{index}"),
            "condition": condition,
        }
        for dataset in ("mitre", "mitre_frr")
        for index in range(32)
        for condition in (
            "baseline",
            "dense_addition",
            "dense_ablation",
            "filtered_addition",
            "filtered_ablation",
        )
    ]
    code = [
        {
            "example_id": f"humaneval_plus-{index}",
            "content_sha256": _sha(f"humaneval_plus-{index}"),
            "condition": condition,
        }
        for index in range(32)
        for condition in ("baseline", "dense", "filtered")
    ]
    bundle = {
        "schema_version": 1,
        "profile": profile,
        "profile_sha256": _canonical_sha(profile),
        "pairs": pair_rows,
        "corpus": {
            "fit_tokens": 4_750_000,
            "holdout_tokens": 250_000,
            "sources": [
                {
                    "repo_id": "code-search-net/code_search_net",
                    "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
                    "tokens": 2_750_000,
                },
                {
                    "repo_id": "HuggingFaceH4/ultrachat_200k",
                    "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
                    "tokens": 1_250_000,
                },
                {
                    "repo_id": "rufimelo/DeltaSecommits",
                    "revision": "40de1664860b8e37fb2322ab2b2fbd0b248f3598",
                    "tokens": 1_000_000,
                },
            ],
        },
        "checkpoint_identity": {
            "run_id": "run-1",
            "implementation_commit": "4" * 40,
            "config_sha256": _canonical_sha(profile),
            "corpus_sha256": _sha("corpus"),
            "model_revision": profile["model_revision"],
        },
        "stage_timestamps": {
            "select": "2026-11-17T10:00:00Z",
            "test": "2026-11-17T10:01:00Z",
        },
        "observations": observations,
        "reviews": reviews,
        "bootstrap": bootstrap,
        "external_observations": external,
        "code_observations": code,
    }
    _write(results / "validation_inputs.json", bundle)

    prior = None
    config_sha = _canonical_sha(profile)
    pair_split_sha = _canonical_sha(pair_rows)
    test_split_sha = _canonical_sha(test_pairs)
    for index, stage in enumerate(STAGES):
        stage_results: dict[str, Any] = {"status": "completed"}
        if stage == "sae_quality":
            stage_results |= {
                "status": "pass",
                "explained_variance": 0.82,
                "mean_cosine": 0.92,
                "nonfinite_count": 0,
                "mean_l0": 64.0,
                "dead_feature_fraction": 0.1,
                "norm_drift": 0.1,
            }
        elif stage == "decompose":
            stage_results |= {
                "status": "pass",
                "cosine": 0.96,
                "relative_l2": 0.2,
                "norm": 1.0,
            }
        elif stage == "select":
            stage_results |= {
                "status": "selection_frozen",
                "selection_freeze_sha256": _sha("selection"),
            }
        elif stage == "test":
            stage_results |= {
                "selection_freeze_sha256": _sha("selection"),
                "observations": observations,
            }
        elif stage == "review_merge":
            stage_results |= {"status": "review_complete", "review_count": 512}
        elif stage == "external_transfer":
            stage_results |= {"observations": external}
        elif stage == "code_preservation":
            stage_results |= {"observations": code}
        elif stage == "finalize":
            stage_results |= {
                "bootstrap": {
                    "addition": {
                        name: bootstrap[name]
                        for name in (
                            "offensive_delta",
                            "defensive_improvement",
                        )
                    },
                    "ablation": {
                        name: bootstrap[name]
                        for name in (
                            "defensive_recovery_delta",
                            "offensive_retention_improvement",
                        )
                    },
                },
                "scientific_statuses": {
                    "addition": (
                        "pass_addition"
                        if bootstrap["offensive_delta"]["ci_lower"] >= -0.05
                        and bootstrap["defensive_improvement"]["ci_lower"] >= 0.20
                        else "fail_selectivity_addition"
                    ),
                    "ablation": (
                        "pass_ablation"
                        if bootstrap["defensive_recovery_delta"]["ci_lower"] >= -0.05
                        and bootstrap["offensive_retention_improvement"]["ci_lower"]
                        >= 0.20
                        else "fail_selectivity_ablation"
                    ),
                },
            }
        payload = {
            "schema_version": 1,
            "experiment": f"qwen_sparse_refusal_{stage}",
            "config": profile,
            "results": stage_results,
            "provenance": {
                "run_id": "run-1",
                "implementation_commit": "4" * 40,
                "config_sha256": config_sha,
                "model_revision": profile["model_revision"],
                "direction_sha256": profile["direction_sha256"],
                "corpus_sha256": _sha("corpus"),
                "pair_split_sha256": pair_split_sha,
                "test_split_sha256": test_split_sha,
                "prior_stage_sha256": prior,
            },
        }
        path = results / f"{index:02d}_{stage}.json"
        _write(path, payload)
        prior = _file_sha(path)
    return results


def write_mutated_sparse_fixture(root: Path, mutation: str) -> Path:
    results = write_complete_sparse_fixture(root)
    bundle_path = results / "validation_inputs.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if mutation == "duplicate_pair":
        bundle["pairs"][1]["pair_id"] = bundle["pairs"][0]["pair_id"]
    elif mutation == "wrong_bootstrap":
        bundle["bootstrap"]["offensive_delta"]["estimate"] += 0.25
    elif mutation == "missing_review":
        bundle["reviews"].pop()
    elif mutation == "raw_text":
        bundle["reviews"][0]["text"] = "forbidden"
    elif mutation == "stale_checkpoint_identity":
        bundle["checkpoint_identity"]["corpus_sha256"] = _sha("stale")
    elif mutation == "wrong_direction_hash":
        bundle["profile"]["direction_sha256"] = "0" * 64
    elif mutation == "test_before_freeze":
        bundle["stage_timestamps"]["test"] = "2026-11-17T09:59:00Z"
    else:
        raise AssertionError(mutation)
    _write(bundle_path, bundle)
    return results


def test_validator_accepts_complete_fixture(tmp_path: Path) -> None:
    fixture = write_complete_sparse_fixture(tmp_path)
    report = _load_validator().validate_sparse_results(fixture)
    assert report.failed == ()
    assert "pareto_recomputed" in report.passed


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_pair",
        "test_before_freeze",
        "wrong_bootstrap",
        "missing_review",
        "raw_text",
        "stale_checkpoint_identity",
        "wrong_direction_hash",
    ),
)
def test_validator_rejects_invalid_fixture(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = write_mutated_sparse_fixture(tmp_path, mutation)
    report = _load_validator().validate_sparse_results(fixture)
    assert report.failed

