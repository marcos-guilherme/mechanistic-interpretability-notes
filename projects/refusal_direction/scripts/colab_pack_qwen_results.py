from pathlib import Path

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_ARCHIVE,
    pack_qwen_results,
)


ROOT = Path("/content/refusal_direction")


def main() -> int:
    pack_qwen_results(ROOT, REMOTE_ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())