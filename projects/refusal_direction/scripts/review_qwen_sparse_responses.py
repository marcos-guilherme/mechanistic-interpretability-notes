"""Interactive blinded review for the 512 private primary responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from refusal_direction_replication.qwen_sparse_review import (
    BLIND_REVIEW_FILENAME,
    REVIEW_LABELS,
    REVIEW_ROW_COUNT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "qwen25_coder_sparse_refusal"
    / "review"
)
COMMAND_LABELS = {
    "r": "refuse",
    "a": "answer",
    "m": "ambiguous",
    "i": "invalid_generation",
}
REVIEW_COMMANDS = frozenset({*COMMAND_LABELS, "b", "q"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("review package must contain JSON objects")
    return rows


def _atomic_write_labels(
    path: Path,
    review_ids: list[str],
    labels: dict[str, str],
    blind_review_sha256: str,
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
            for review_id in review_ids:
                if review_id in labels:
                    handle.write(
                        json.dumps(
                            {
                                "review_id": review_id,
                                "label": labels[review_id],
                                "blind_review_sha256": blind_review_sha256,
                            },
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


def _load_existing_labels(
    path: Path,
    expected_ids: set[str],
    blind_review_sha256: str,
) -> dict[str, str]:
    if not path.is_file():
        return {}
    labels: dict[str, str] = {}
    for row in _read_jsonl(path):
        if set(row) != {
            "review_id",
            "label",
            "blind_review_sha256",
        }:
            raise ValueError("saved review row schema changed")
        review_id = str(row["review_id"])
        label = str(row["label"])
        if review_id not in expected_ids or review_id in labels:
            raise ValueError("saved review IDs do not match blind package")
        if label not in REVIEW_LABELS:
            raise ValueError("saved review label is invalid")
        if row["blind_review_sha256"] != blind_review_sha256:
            raise ValueError("saved blind_review_sha256 does not match package")
        labels[review_id] = label
    return labels


def review_responses(
    review_path: Path,
    output_path: Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> bool:
    rows = _read_jsonl(review_path)
    if len(rows) != REVIEW_ROW_COUNT:
        raise ValueError("blind review package must contain exactly 512 rows")
    if any(
        set(row) != {"review_id", "prompt", "generation"}
        for row in rows
    ):
        raise ValueError("blind review package schema changed")
    review_ids = [str(row["review_id"]) for row in rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("blind review package contains duplicate review IDs")
    blind_review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    labels = _load_existing_labels(
        output_path,
        set(review_ids),
        blind_review_sha256,
    )
    index = next(
        (
            offset
            for offset, review_id in enumerate(review_ids)
            if review_id not in labels
        ),
        len(rows),
    )
    while index < len(rows):
        row = rows[index]
        review_id = str(row["review_id"])
        output_fn(f"\n[{index + 1}/{len(rows)}] {review_id}")
        output_fn("PROMPT")
        output_fn(str(row["prompt"]))
        output_fn("GENERATION")
        output_fn(str(row["generation"]))
        current = labels.get(review_id)
        if current:
            output_fn(f"Saved label: {current}")
        command = input_fn(
            "r=refuse a=answer m=ambiguous i=invalid "
            "b=back q=save+quit > "
        ).strip().casefold()
        if command not in REVIEW_COMMANDS:
            output_fn("Invalid command.")
            continue
        if command == "q":
            _atomic_write_labels(
                output_path,
                review_ids,
                labels,
                blind_review_sha256,
            )
            return False
        if command == "b":
            index = max(0, index - 1)
            continue
        labels[review_id] = COMMAND_LABELS[command]
        _atomic_write_labels(
            output_path,
            review_ids,
            labels,
            blind_review_sha256,
        )
        index += 1
    _atomic_write_labels(
        output_path,
        review_ids,
        labels,
        blind_review_sha256,
    )
    output_fn(f"Completed {len(labels)} blinded reviews.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW_DIR / BLIND_REVIEW_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REVIEW_DIR / "responses_reviewed.jsonl",
    )
    args = parser.parse_args()
    review_responses(args.review, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
