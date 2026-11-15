"""Remote-safe Qwen causal-pilot entrypoint for Colab exec.

Colab exec injects extra kernel argv, so this entrypoint ignores process
arguments and always uses the remote working tree at
``/content/refusal_direction``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_pilot import run_qwen_pilot


def remote_main() -> int:
    payload = run_qwen_pilot(REMOTE_ROOT)
    print(payload.get("status", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(remote_main())
