"""Install a validated J-lens checkpoint for resume in a new Colab session."""

import sys
from pathlib import Path

ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_JLENS_CHECKPOINT_STAGING,
    atomic_install_jlens_checkpoint,
)


def main() -> int:
    atomic_install_jlens_checkpoint(
        staged=Path(REMOTE_JLENS_CHECKPOINT_STAGING),
        project=ROOT,
        missing_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
