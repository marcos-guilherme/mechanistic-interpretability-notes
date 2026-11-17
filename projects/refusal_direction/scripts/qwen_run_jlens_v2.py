"""Remote-safe Qwen Jacobian-lens v2 entrypoint for Colab exec.

Colab exec injects extra kernel argv, so this entrypoint ignores process
arguments and always uses the remote working tree at
``/content/refusal_direction`` with the v2 profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_jlens import JLENS_V2_PROFILE
from refusal_direction_replication.qwen_jlens_runtime import run_qwen_jlens


def remote_main() -> int:
    payload = run_qwen_jlens(REMOTE_ROOT, profile=JLENS_V2_PROFILE)
    print(payload.get("status", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(remote_main())
