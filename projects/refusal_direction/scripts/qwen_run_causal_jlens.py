"""Remote-safe entrypoint for the causal J-lens fidelity run."""

from __future__ import annotations

import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_causal_jlens import CAUSAL_JLENS_PROFILE
from refusal_direction_replication.qwen_causal_jlens_runtime import run_qwen_causal_jlens


def remote_main() -> int:
    payload = run_qwen_causal_jlens(REMOTE_ROOT, profile=CAUSAL_JLENS_PROFILE)
    print(payload.get("status", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(remote_main())
