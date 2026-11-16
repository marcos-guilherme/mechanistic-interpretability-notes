import sys
from pathlib import Path

ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_DOSE_SOURCE_DIRECTION_STAGING,
    atomic_install_dose_source_direction,
)


def main() -> int:
    atomic_install_dose_source_direction(
        staged=Path(REMOTE_DOSE_SOURCE_DIRECTION_STAGING),
        project=ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
