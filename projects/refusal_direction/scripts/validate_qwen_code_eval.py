from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REMOTE_SRC = "/content/refusal_direction/src"
if _REMOTE_SRC not in sys.path:
    sys.path.insert(0, _REMOTE_SRC)

from refusal_direction_replication.qwen_code_eval_validate import (
    validate_qwen_code_eval,
)


def main() -> int:
    checks = validate_qwen_code_eval(ROOT)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
