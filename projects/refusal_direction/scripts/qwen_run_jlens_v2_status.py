"""Lightweight Qwen Jacobian-lens v2 status reader for Colab exec.

Reads only the v2 status/events paths. Colab runs one execution at a time per
session, so this queues behind a running fit; use ``checkpoint-backup`` for
in-run progress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_jlens import JLENS_V2_PROFILE
from refusal_direction_replication.run_progress import EVENTS_FILENAME, STATUS_FILENAME

last_stdout_summary: str | None = None


def artifacts_dir_for_profile(root: Path, profile: str = JLENS_V2_PROFILE) -> Path:
    config_path = root / "configs" / profile
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return root / config["artifacts_dir"]


def last_event_name(artifacts_dir: Path) -> str | None:
    events_path = artifacts_dir / EVENTS_FILENAME
    if not events_path.is_file():
        return None
    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return str(json.loads(lines[-1]).get("event"))
    except json.JSONDecodeError:
        return None


def summarize_status_for_display(payload: dict, *, last_event: str | None) -> str:
    parts = [
        f"profile={payload['profile']}",
        f"run_id={payload['run_id']}",
        f"stage={payload['current_stage']}",
        f"state={payload['state']}",
        f"progress={payload['completed_units']}/{payload['total_units']}",
    ]
    if "fit_completed_units" in payload and "fit_total_units" in payload:
        parts.append(
            f"fit_progress={payload['fit_completed_units']}/{payload['fit_total_units']}"
        )
    for key in ("message", "abort_gate", "abort_reason"):
        value = payload.get(key)
        if value:
            parts.append(f"{key}={value}")
    if last_event:
        parts.append(f"last_event={last_event}")
    return " | ".join(parts)


def main_with_root(root: Path, *, profile: str = JLENS_V2_PROFILE) -> int:
    global last_stdout_summary
    artifacts = artifacts_dir_for_profile(root, profile)
    status_path = artifacts / STATUS_FILENAME
    if not status_path.is_file():
        print(f"missing status: {status_path}", file=sys.stderr)
        last_stdout_summary = None
        return 2
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    summary = summarize_status_for_display(payload, last_event=last_event_name(artifacts))
    print(summary)
    last_stdout_summary = summary
    return 0


def remote_main() -> int:
    return main_with_root(REMOTE_ROOT)


if __name__ == "__main__":
    raise SystemExit(remote_main())
