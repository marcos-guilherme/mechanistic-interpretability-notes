"""Lightweight Qwen Jacobian-lens status reader for Colab exec."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.run_progress import STATUS_FILENAME

DEFAULT_PROFILE = "qwen25_coder_7b_jlens.json"
last_stdout_summary: str | None = None


def artifacts_dir_for_profile(root: Path, profile: str) -> Path:
    config_path = root / "configs" / profile
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return root / config["artifacts_dir"]


def summarize_status_for_display(payload: dict) -> str:
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
    message = payload.get("message")
    if message:
        parts.append(f"message={message}")
    abort_gate = payload.get("abort_gate")
    abort_reason = payload.get("abort_reason")
    if abort_gate:
        parts.append(f"abort_gate={abort_gate}")
    if abort_reason:
        parts.append(f"abort_reason={abort_reason}")
    return " | ".join(parts)


def read_status_payload(artifacts_dir: Path) -> dict | None:
    status_path = artifacts_dir / STATUS_FILENAME
    if not status_path.is_file():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def main_with_root(root: Path, *, profile: str = DEFAULT_PROFILE) -> int:
    global last_stdout_summary
    artifacts = artifacts_dir_for_profile(root, profile)
    payload = read_status_payload(artifacts)
    if payload is None:
        print(f"missing status: {artifacts / STATUS_FILENAME}", file=sys.stderr)
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
