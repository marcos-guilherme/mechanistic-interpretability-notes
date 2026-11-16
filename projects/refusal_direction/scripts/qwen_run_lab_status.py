"""Lightweight operational status reader for the Qwen intervention lab."""

from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    validate_lab_operational_status,
)

DEFAULT_PROFILE = "qwen25_coder_7b_lab.json"
last_stdout_summary: str | None = None


def artifacts_dir_for_profile(root: Path, profile: str = DEFAULT_PROFILE) -> Path:
    config_path = root / "configs" / profile
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return root / config["artifacts_dir"]


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if getattr(error, "errno", None) == errno.ESRCH:
            return False
        if getattr(error, "errno", None) == errno.EPERM:
            return True
        return False
    return True


def apply_pid_liveness(payload: dict) -> dict:
    result = dict(payload)
    if result.get("state") != "running":
        return result
    pid = result.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or not pid_is_alive(pid):
        result["state"] = "stopped"
        result["share_ready"] = False
    return result


def summarize_status_for_display(payload: dict) -> str:
    validated = validate_lab_operational_status(payload)
    display = apply_pid_liveness(validated)
    parts = [
        f"profile={display['profile']}",
        f"state={display['state']}",
        f"gpu={display['gpu']}",
        f"share_ready={display['share_ready']}",
    ]
    if display.get("public_url"):
        parts.append(f"public_url={display['public_url']}")
    if "pid" in display:
        parts.append(f"pid={display['pid']}")
    if display.get("direction_sha256"):
        parts.append(f"direction_sha256={display['direction_sha256']}")
    if display.get("lens_sha256"):
        parts.append(f"lens_sha256={display['lens_sha256']}")
    return " | ".join(parts)


def read_status_payload(artifacts_dir: Path) -> dict | None:
    status_path = artifacts_dir / "status.json"
    if not status_path.is_file():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def main_with_root(root: Path, *, profile: str = DEFAULT_PROFILE) -> int:
    global last_stdout_summary
    artifacts = artifacts_dir_for_profile(root, profile)
    payload = read_status_payload(artifacts)
    if payload is None:
        print(f"missing status: {artifacts / 'status.json'}", file=sys.stderr)
        last_stdout_summary = None
        return 2
    summary = summarize_status_for_display(payload)
    print(summary)
    last_stdout_summary = summary
    return 0


def remote_main() -> int:
    return main_with_root(REMOTE_ROOT, profile=DEFAULT_PROFILE)


if __name__ == "__main__":
    raise SystemExit(remote_main())
