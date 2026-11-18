"""Blinded response-review export and strict text-free merge helpers.

The public review package and its source responses are private artifacts.  The
returned merge payload is compact and may be copied to versioned results.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .data import prompt_content_hash


PRIMARY_REVIEW_CONDITIONS = (
    "dense_addition",
    "filtered_addition",
    "dense_ablation",
    "filtered_ablation",
)
REVIEW_LABELS = (
    "refuse",
    "answer",
    "ambiguous",
    "invalid_generation",
)
REVIEW_ROW_COUNT = 512
PRIVATE_MANIFEST_FILENAME = "private_manifest.json"
BLIND_REVIEW_FILENAME = "blind_review.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: expected object")
        rows.append(value)
    return rows


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(
    row: Mapping[str, Any],
    field: str,
    identity: str,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{identity}: {field} must be nonempty")
    return value


def _review_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _required_text(row, "pair_id", "<response>"),
        _required_text(row, "side", "<response>"),
        _required_text(row, "condition", "<response>"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _frozen_side_hashes(
    frozen_test_pairs: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], str]:
    if len(frozen_test_pairs) != 64:
        raise ValueError("frozen test split must contain exactly 64 pairs")
    output: dict[tuple[str, str], str] = {}
    for row in frozen_test_pairs:
        if set(row) != {
            "pair_id",
            "offensive_sha256",
            "defensive_sha256",
        }:
            raise ValueError("frozen test split schema changed")
        pair_id = _required_text(row, "pair_id", "<frozen-test>")
        for side in ("offensive", "defensive"):
            digest = _required_text(row, f"{side}_sha256", pair_id)
            if len(digest) != 64:
                raise ValueError("frozen test split hash must be SHA-256")
            key = (pair_id, side)
            if key in output:
                raise ValueError("frozen test split contains duplicate pairs")
            output[key] = digest
    if len(output) != 128:
        raise ValueError("frozen test split identities are incomplete")
    return output


def _validated_response_rows(
    response_paths: Sequence[Path],
    frozen_test_pairs: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = [
        row
        for path in response_paths
        for row in _read_jsonl(Path(path))
    ]
    keys = [_review_key(row) for row in rows]
    frozen_hashes = _frozen_side_hashes(frozen_test_pairs)
    pair_ids = {pair_id for pair_id, _side in frozen_hashes}
    actual_shape = {
        (pair_id, side, condition)
        for pair_id in pair_ids
        for side in ("offensive", "defensive")
        for condition in PRIMARY_REVIEW_CONDITIONS
    }
    if (
        len(rows) != REVIEW_ROW_COUNT
        or len(pair_ids) != 64
        or len(keys) != len(set(keys))
        or set(keys) != actual_shape
    ):
        raise ValueError(
            "review coverage must contain exactly 64 pairs, two sides, "
            "and four primary conditions"
        )
    validated: list[dict[str, Any]] = []
    for row in rows:
        pair_id, side, condition = _review_key(row)
        if side not in {"offensive", "defensive"}:
            raise ValueError("review coverage contains an invalid side")
        if condition not in PRIMARY_REVIEW_CONDITIONS:
            raise ValueError("review coverage contains a non-primary condition")
        prompt = _required_text(row, "prompt", pair_id)
        prompt_sha256 = _required_text(row, "prompt_sha256", pair_id)
        computed_prompt_sha256 = prompt_content_hash(prompt)
        if computed_prompt_sha256 != prompt_sha256:
            raise ValueError(
                f"{pair_id}: computed prompt hash differs from declared hash"
            )
        if prompt_sha256 != frozen_hashes.get((pair_id, side)):
            raise ValueError("review responses differ from frozen test split")
        generation = row.get("generation")
        if not isinstance(generation, str):
            raise ValueError(f"{pair_id}: generation must be a string")
        automatic = row.get("automatic_refusal")
        if not isinstance(automatic, bool):
            raise ValueError(
                f"{pair_id}: automatic_refusal must be boolean"
            )
        validated.append(
            {
                "pair_id": pair_id,
                "side": side,
                "condition": condition,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "generation": generation,
                "generation_sha256": hashlib.sha256(
                    generation.encode("utf-8")
                ).hexdigest(),
                "automatic_refusal": automatic,
            }
        )
    return validated


def build_blind_review(
    response_paths: Sequence[Path],
    output_dir: Path,
    seed: int = 2031,
    *,
    frozen_test_pairs: Sequence[Mapping[str, str]],
) -> Path:
    """Export exactly 512 primary responses with assignment data separated."""
    if len(response_paths) == 0:
        raise ValueError("response_paths must be nonempty")
    rows = _validated_response_rows(response_paths, frozen_test_pairs)
    ordered = sorted(
        rows,
        key=lambda row: (
            row["pair_id"],
            row["side"],
            row["condition"],
        ),
    )
    public: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        material = (
            f"{int(seed)}:{index}:{row['pair_id']}:{row['side']}:"
            f"{row['condition']}:{row['prompt_sha256']}:"
            f"{row['generation_sha256']}"
        )
        review_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        public.append(
            {
                "review_id": review_id,
                "prompt": row["prompt"],
                "generation": row["generation"],
            }
        )
        assignments.append(
            {
                "review_id": review_id,
                "pair_id": row["pair_id"],
                "side": row["side"],
                "condition": row["condition"],
                "prompt_sha256": row["prompt_sha256"],
                "generation_sha256": row["generation_sha256"],
                "automatic_refusal": row["automatic_refusal"],
            }
        )
    random.Random(int(seed)).shuffle(public)
    output = Path(output_dir)
    public_path = output / BLIND_REVIEW_FILENAME
    _atomic_write_jsonl(public_path, public)
    _atomic_write_json(
        output / PRIVATE_MANIFEST_FILENAME,
        {
            "schema_version": 1,
            "seed": int(seed),
            "review_count": REVIEW_ROW_COUNT,
            "blind_review_sha256": _sha256_file(public_path),
            "test_split_sha256": _canonical_sha256(
                list(frozen_test_pairs)
            ),
            "assignments": assignments,
        },
    )
    return public_path


def _validate_manifest(
    private_manifest: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(private_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("private manifest must be an object")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != REVIEW_ROW_COUNT:
        raise ValueError("private manifest must contain 512 assignments")
    by_id: dict[str, dict[str, Any]] = {}
    coverage: list[tuple[str, str, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("private manifest assignment must be an object")
        review_id = _required_text(assignment, "review_id", "<assignment>")
        if review_id in by_id:
            raise ValueError("private manifest has duplicate review IDs")
        if set(assignment) != {
            "review_id",
            "pair_id",
            "side",
            "condition",
            "prompt_sha256",
            "generation_sha256",
            "automatic_refusal",
        }:
            raise ValueError("private manifest assignment schema changed")
        pair_id = _required_text(assignment, "pair_id", review_id)
        side = _required_text(assignment, "side", review_id)
        condition = _required_text(assignment, "condition", review_id)
        for field in ("prompt_sha256", "generation_sha256"):
            if len(_required_text(assignment, field, review_id)) != 64:
                raise ValueError("private assignment hash must be SHA-256")
        if side not in {"offensive", "defensive"}:
            raise ValueError("private review coverage has an invalid side")
        if condition not in PRIMARY_REVIEW_CONDITIONS:
            raise ValueError("private review coverage has an invalid condition")
        if not isinstance(assignment["automatic_refusal"], bool):
            raise ValueError("private automatic refusal must be boolean")
        coverage.append((pair_id, side, condition))
        by_id[review_id] = assignment
    pair_ids = {pair_id for pair_id, _side, _condition in coverage}
    expected_coverage = {
        (pair_id, side, condition)
        for pair_id in pair_ids
        for side in ("offensive", "defensive")
        for condition in PRIMARY_REVIEW_CONDITIONS
    }
    if (
        len(pair_ids) != 64
        or len(coverage) != len(set(coverage))
        or set(coverage) != expected_coverage
    ):
        raise ValueError("private review coverage is not an exact 64x2x4 matrix")
    return payload, by_id


def merge_blind_review(
    private_manifest: Path,
    reviewed_jsonl: Path,
) -> dict[str, Any]:
    """Strictly merge one valid human label for every blinded review ID."""
    manifest, assignments = _validate_manifest(Path(private_manifest))
    blind_path = Path(private_manifest).parent / BLIND_REVIEW_FILENAME
    actual_blind_sha256 = _sha256_file(blind_path)
    if manifest.get("blind_review_sha256") != actual_blind_sha256:
        raise ValueError("blind_review_sha256 does not match blind review file")
    reviewed = _read_jsonl(Path(reviewed_jsonl))
    ids: list[str] = []
    labels: dict[str, str] = {}
    for row in reviewed:
        if set(row) != {"review_id", "label", "blind_review_sha256"}:
            raise ValueError(
                "review rows must contain only review_id and review label"
            )
        review_id = _required_text(row, "review_id", "<review>")
        label = _required_text(row, "label", review_id)
        if row.get("blind_review_sha256") != actual_blind_sha256:
            raise ValueError("review labels blind_review_sha256 mismatch")
        if label not in REVIEW_LABELS:
            raise ValueError(f"invalid review label: {label}")
        ids.append(review_id)
        labels[review_id] = label
    if (
        len(ids) != REVIEW_ROW_COUNT
        or len(ids) != len(set(ids))
        or set(ids) != set(assignments)
    ):
        raise ValueError(
            "review IDs must form an exact one-to-one match with the manifest"
        )

    label_counts = Counter(labels.values())
    aggregate_counts: dict[str, dict[str, dict[str, int]]] = {}
    agreement_total = 0
    agreement_count = 0
    for review_id, assignment in assignments.items():
        condition = str(assignment["condition"])
        side = str(assignment["side"])
        label = labels[review_id]
        block = aggregate_counts.setdefault(condition, {}).setdefault(
            side,
            {item: 0 for item in REVIEW_LABELS},
        )
        block[label] += 1
        if label in {"refuse", "answer"}:
            agreement_total += 1
            human_refusal = label == "refuse"
            agreement_count += (
                human_refusal is bool(assignment["automatic_refusal"])
            )
    reviews = [
        {"review_id": review_id, "label": labels[review_id]}
        for review_id in sorted(labels)
    ]
    return {
        "schema_version": 1,
        "status": "review_complete",
        "review_count": REVIEW_ROW_COUNT,
        "manifest_seed": int(manifest["seed"]),
        "blind_review_sha256": actual_blind_sha256,
        "test_split_sha256": str(manifest["test_split_sha256"]),
        "reviews": reviews,
        "label_counts": {
            label: int(label_counts.get(label, 0))
            for label in REVIEW_LABELS
        },
        "aggregate_counts": aggregate_counts,
        "automatic_agreement": {
            "eligible_count": agreement_total,
            "agreement_count": agreement_count,
            "rate": (
                agreement_count / agreement_total
                if agreement_total
                else None
            ),
        },
    }
