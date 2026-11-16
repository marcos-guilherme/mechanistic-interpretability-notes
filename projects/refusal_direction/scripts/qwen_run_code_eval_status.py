"""Lightweight status reader for the Qwen code evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_PROFILE = "qwen25_coder_7b_code_eval.json"
last_stdout_summary: str | None = None


def main_with_root(root: Path, *, profile: str = DEFAULT_PROFILE) -> int:
    global last_stdout_summary
    config = json.loads((root / "configs" / profile).read_text(encoding="utf-8"))
    status_path = root / config["artifacts_dir"] / "status.json"
    if not status_path.is_file():
        print(f"missing status: {status_path}", file=sys.stderr)
        return 2
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    last_stdout_summary = (
        f"profile={payload['profile']} | run_id={payload['run_id']} | "
        f"stage={payload['current_stage']} | state={payload['state']} | "
        f"progress={payload['completed_units']}/{payload['total_units']}"
    )
    print(last_stdout_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_with_root(REMOTE_ROOT))
