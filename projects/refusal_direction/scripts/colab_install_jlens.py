"""Install the pinned Jacobian-lens upstream without touching Transformers.

Pin: anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e
`--no-deps` is deliberate so the project's Transformers pins stay.
"""

from __future__ import annotations

import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_jlens import (
    JLENS_UPSTREAM_SPEC,
    assert_jlens_contract,
    jlens_install_command,
)


def main() -> int:
    import subprocess

    command = jlens_install_command(sys.executable)
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    completed.check_returncode()
    assert_jlens_contract()
    print(f"installed {JLENS_UPSTREAM_SPEC} with --no-deps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
