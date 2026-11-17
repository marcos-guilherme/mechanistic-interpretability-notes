"""Remote-safe entrypoint for the sparse-refusal stage machine."""

from __future__ import annotations

import argparse
from pathlib import Path

from refusal_direction_replication.qwen_sparse_protocol import SPARSE_PROFILE
from refusal_direction_replication.qwen_sparse_runtime import (
    SPARSE_STAGES,
    run_qwen_sparse_refusal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--profile", default=SPARSE_PROFILE)
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after", choices=SPARSE_STAGES)
    args = parser.parse_args()
    result = run_qwen_sparse_refusal(
        args.root,
        args.profile,
        run_id=args.run_id,
        stop_after=args.stop_after,
    )
    stage = str(result.get("experiment", "unknown")).removeprefix(
        "qwen_sparse_refusal_"
    )
    status = str((result.get("results") or {}).get("status", "unknown"))
    print(f"{stage}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
