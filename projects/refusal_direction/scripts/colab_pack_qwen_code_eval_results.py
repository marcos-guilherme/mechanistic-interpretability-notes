import sys
from pathlib import Path

ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    CODE_EVAL_REMOTE_ARCHIVE,
    pack_qwen_code_eval_results,
)


def main() -> int:
    pack_qwen_code_eval_results(ROOT, CODE_EVAL_REMOTE_ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
