from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pytest

from refusal_direction_replication.qwen_sparse_review import (
    REVIEW_LABELS,
    build_blind_review,
    merge_blind_review,
)
from refusal_direction_replication.qwen_sparse_runtime import (
    compact_contains_leak,
    evaluate_blind_review,
)


PRIMARY_CONDITIONS = (
    "dense_addition",
    "filtered_addition",
    "dense_ablation",
    "filtered_ablation",
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _response_files(tmp_path: Path) -> list[Path]:
    paths = []
    for condition in PRIMARY_CONDITIONS:
        rows = []
        for pair_index in range(64):
            for side in ("offensive", "defensive"):
                rows.append(
                    {
                        "pair_id": f"pair-{pair_index:04d}",
                        "side": side,
                        "condition": condition,
                        "prompt": f"private prompt {pair_index} {side}",
                        "prompt_sha256": _sha(
                            f"private prompt {pair_index} {side}"
                        ),
                        "generation": f"private generation {condition}",
                        "automatic_refusal": pair_index % 2 == 0,
                    }
                )
        path = tmp_path / "responses" / f"{condition}.jsonl"
        _write_jsonl(path, rows)
        paths.append(path)
    return paths


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _frozen_test_pairs() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "pair_id": f"pair-{index:04d}",
            "offensive_sha256": _sha(f"private prompt {index} offensive"),
            "defensive_sha256": _sha(f"private prompt {index} defensive"),
        }
        for index in range(64)
    )


def _blind_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _labels(
    public_path: Path,
    *,
    label: str = "answer",
) -> list[dict[str, object]]:
    digest = _blind_hash(public_path)
    return [
        {
            "review_id": row["review_id"],
            "label": label,
            "blind_review_sha256": digest,
        }
        for row in _read_jsonl(public_path)
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_review_export_contains_exactly_512_rows_and_hides_assignment(
    tmp_path: Path,
) -> None:
    public_path = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    rows = _read_jsonl(public_path)
    assert len(rows) == 512
    assert all(set(row) == {"review_id", "prompt", "generation"} for row in rows)
    assert all(
        not {"condition", "side", "pair_id", "dose", "split"}.intersection(row)
        for row in rows
    )

    manifest = json.loads(
        (tmp_path / "blind" / "private_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["assignments"]) == 512
    assert not any(
        {"prompt", "generation", "response", "text", "content"}.intersection(row)
        for row in manifest["assignments"]
    )


def test_review_merge_requires_exact_bijection_and_emits_compact_metrics(
    tmp_path: Path,
) -> None:
    public_path = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    public = _read_jsonl(public_path)
    reviewed_path = tmp_path / "reviewed.jsonl"
    reviewed = [
        {
            "review_id": row["review_id"],
            "label": "refuse" if index % 2 == 0 else "answer",
            "blind_review_sha256": _blind_hash(public_path),
        }
        for index, row in enumerate(public)
    ]
    _write_jsonl(reviewed_path, reviewed)
    merged = merge_blind_review(
        tmp_path / "blind" / "private_manifest.json",
        reviewed_path,
    )
    assert merged["status"] == "review_complete"
    assert merged["review_count"] == 512
    assert set(merged["label_counts"]) == set(REVIEW_LABELS)
    assert not compact_contains_leak(merged)
    assert all(
        set(row) == {"review_id", "label"}
        for row in merged["reviews"]
    )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unknown", "bad_label"))
def test_review_merge_rejects_malformed_review_ids_or_labels(
    tmp_path: Path,
    mutation: str,
) -> None:
    public_path = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    rows = _labels(public_path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    elif mutation == "unknown":
        rows[-1]["review_id"] = "not-expected"
    else:
        rows[-1]["label"] = "maybe"
    reviewed_path = tmp_path / f"{mutation}.jsonl"
    _write_jsonl(reviewed_path, rows)
    message = "review IDs" if mutation != "bad_label" else "review label"
    with pytest.raises(ValueError, match=message):
        merge_blind_review(
            tmp_path / "blind" / "private_manifest.json",
            reviewed_path,
        )


def test_review_export_rejects_leaks_duplicates_and_wrong_coverage(
    tmp_path: Path,
) -> None:
    paths = _response_files(tmp_path)
    rows = _read_jsonl(paths[0])
    rows[1] = dict(rows[0])
    _write_jsonl(paths[0], rows)
    with pytest.raises(ValueError, match="review coverage"):
        build_blind_review(
            paths,
            tmp_path / "bad",
            frozen_test_pairs=_frozen_test_pairs(),
        )


def test_review_merge_rejects_tampered_private_assignment_coverage(
    tmp_path: Path,
) -> None:
    public_path = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    manifest_path = tmp_path / "blind" / "private_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assignments"][-1]["pair_id"] = manifest["assignments"][0]["pair_id"]
    manifest["assignments"][-1]["side"] = manifest["assignments"][0]["side"]
    manifest["assignments"][-1]["condition"] = manifest["assignments"][0]["condition"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        reviewed_path,
        _labels(public_path),
    )
    with pytest.raises(ValueError, match="review coverage"):
        merge_blind_review(manifest_path, reviewed_path)


def test_final_evaluation_recomputes_separate_addition_and_ablation_gates(
    tmp_path: Path,
) -> None:
    public_path = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    assignments = {
        row["review_id"]: row
        for row in json.loads(
            (tmp_path / "blind" / "private_manifest.json").read_text(
                encoding="utf-8"
            )
        )["assignments"]
    }
    reviewed_path = tmp_path / "reviewed.jsonl"
    _write_jsonl(
        reviewed_path,
        [
            {
                "review_id": row["review_id"],
                "label": (
                    "ambiguous"
                    if assignments[row["review_id"]]["condition"]
                    == "filtered_ablation"
                    and assignments[row["review_id"]]["side"] == "defensive"
                    else "answer"
                ),
                "blind_review_sha256": _blind_hash(public_path),
            }
            for row in _read_jsonl(public_path)
        ],
    )
    evaluation = evaluate_blind_review(
        tmp_path / "blind" / "private_manifest.json",
        reviewed_path,
    )
    assert evaluation["review"]["status"] == "review_complete"
    assert evaluation["scientific_statuses"] == {
        "addition": "fail_selectivity_addition",
        "ablation": "fail_selectivity_ablation",
    }
    assert evaluation["bootstrap"]["addition"]["offensive_delta"]["resamples"] == 10_000
    assert evaluation["bootstrap"]["ablation"][
        "defensive_recovery_delta"
    ]["estimate"] == -1.0
    assert evaluation["bootstrap"]["ablation"][
        "offensive_retention_improvement"
    ]["seed"] == 2029
    assert not compact_contains_leak(evaluation)


def test_review_labels_bind_exact_blind_file_and_generation_replay(
    tmp_path: Path,
) -> None:
    paths = _response_files(tmp_path)
    first = build_blind_review(
        paths,
        tmp_path / "blind-a",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    first_ids = {row["review_id"] for row in _read_jsonl(first)}
    changed = _read_jsonl(paths[0])
    changed[0]["generation"] = "different generation"
    _write_jsonl(paths[0], changed)
    second = build_blind_review(
        paths,
        tmp_path / "blind-b",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    second_ids = {row["review_id"] for row in _read_jsonl(second)}
    assert first_ids != second_ids

    reviewed = tmp_path / "blind-a" / "responses_reviewed.jsonl"
    _write_jsonl(reviewed, _labels(first))
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blind_review_sha256"):
        merge_blind_review(
            tmp_path / "blind-a" / "private_manifest.json",
            reviewed,
        )


def test_review_script_rejects_labels_from_another_blind_package(
    tmp_path: Path,
) -> None:
    public = build_blind_review(
        _response_files(tmp_path),
        tmp_path / "blind",
        frozen_test_pairs=_frozen_test_pairs(),
    )
    output = tmp_path / "blind" / "responses_reviewed.jsonl"
    rows = _labels(public)
    rows[0]["blind_review_sha256"] = "0" * 64
    _write_jsonl(output, rows[:1])
    namespace = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "scripts"
            / "review_qwen_sparse_responses.py"
        )
    )
    with pytest.raises(ValueError, match="blind_review_sha256"):
        namespace["review_responses"](
            public,
            output,
            input_fn=lambda _prompt: "q",
            output_fn=lambda _message: None,
        )
