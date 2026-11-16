"""Lightweight operational status reader for the Qwen intervention lab."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import lab_status_is_safe

DEFAULT_PROFILE = "qwen25_coder_7b_lab.json"
last_stdout_summary: str | None = None


def artifacts_dir_for_profile(root: Path, profile: str = DEFAULT_PROFILE) -> Path:
    config_path = root / "configs" / profile
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return root / config["artifacts_dir"]


def summarize_status_for_display(payload: dict) -> str:
    if not lab_status_is_safe(payload):
        raise ValueError("lab status payload contains forbidden secrets or text")
    parts = [
        f"profile={payload.get('profile', DEFAULT_PROFILE)}",
        f"state={payload.get('state', 'unknown')}",
    ]
    if payload.get("gpu"):
        parts.append(f"gpu={payload['gpu']}")
    if "share_ready" in payload:
        parts.append(f"share_ready={payload['share_ready']}")
    if payload.get("public_url"):
        parts.append(f"public_url={payload['public_url']}")
    if payload.get("direction_sha256"):
        parts.append(f"direction_sha256={payload['direction_sha256']}")
    if payload.get("lens_sha256"):
        parts.append(f"lens_sha256={payload['lens_sha256']}")
    if payload.get("message"):
        parts.append(f"message={payload['message']}")
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
