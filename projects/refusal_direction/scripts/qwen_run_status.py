from __future__ import annotations

import argparse
import sys
from pathlib import Path

from refusal_direction_replication.pipeline import load_config, project_paths
from refusal_direction_replication.run_progress import RunProgressStore, STATUS_FILENAME


PROJECT = Path(__file__).resolve().parents[1]


def summarize_status(payload: dict) -> str:
    parts = [
        f"profile={payload['profile']}",
        f"run_id={payload['run_id']}",
        f"stage={payload['current_stage']}",
        f"state={payload['state']}",
        f"progress={payload['completed_units']}/{payload['total_units']}",
        f"updated_at_utc={payload['updated_at_utc']}",
    ]
    message = payload.get("message")
    if message:
        parts.append(f"message={message}")
    abort_gate = payload.get("abort_gate")
    abort_reason = payload.get("abort_reason")
    if abort_gate:
        parts.append(f"abort_gate={abort_gate}")
    if abort_reason:
        parts.append(f"abort_reason={abort_reason}")
    return " | ".join(parts)


def main_with_root(root: Path, *, profile: str = "qwen25_coder_7b.json") -> int:
    config = load_config(root, profile)
    _, artifacts = project_paths(root, config)
    status_path = artifacts / STATUS_FILENAME
    if not status_path.is_file():
        print(f"missing status: {status_path}", file=sys.stderr)
        return 2
    store = RunProgressStore(
        artifacts,
        profile=profile,
        run_id="unused",
    )
    payload = store.read_status()
    if payload is None:
        print(f"missing status: {status_path}", file=sys.stderr)
        return 2
    print(summarize_status(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="qwen25_coder_7b.json",
        help="profile config filename",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT,
        help="refusal_direction project root",
    )
    args = parser.parse_args()
    return main_with_root(args.root.resolve(), profile=args.profile)


if __name__ == "__main__":
    raise SystemExit(main())