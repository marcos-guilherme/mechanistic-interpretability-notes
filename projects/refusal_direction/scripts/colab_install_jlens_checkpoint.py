"""Install a validated J-lens resume bundle in a new Colab session."""

import sys
from pathlib import Path

ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_JLENS_CHECKPOINT_STAGING,
    REMOTE_JLENS_EVENTS_STAGING,
    REMOTE_JLENS_STATUS_STAGING,
    atomic_install_jlens_resume,
)


def main() -> int:
    atomic_install_jlens_resume(
        checkpoint=Path(REMOTE_JLENS_CHECKPOINT_STAGING),
        status=Path(REMOTE_JLENS_STATUS_STAGING),
        events=Path(REMOTE_JLENS_EVENTS_STAGING),
        project=ROOT,
        missing_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
