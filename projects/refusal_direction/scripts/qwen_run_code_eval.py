"""Remote-safe entrypoint for the five-stage Qwen code evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_code_eval_runtime import run_qwen_code_eval


def remote_main() -> int:
    payload = run_qwen_code_eval(REMOTE_ROOT)
    print(payload.get("status", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(remote_main())
