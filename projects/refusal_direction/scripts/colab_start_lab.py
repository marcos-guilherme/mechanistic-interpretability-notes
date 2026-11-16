"""Start the Gradio lab server in the background and return the public URL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_LAB_AUTH_FILE,
    extract_lab_public_url,
    lab_status_is_safe,
)

RUN_SCRIPT = REMOTE_ROOT / "scripts" / "qwen_run_lab.py"
REMOTE_AUTH_FILE = Path(REMOTE_LAB_AUTH_FILE)
SERVER_LOG = Path("/content/qwen_lab_server.log")
STATUS_PATH = REMOTE_ROOT / "artifacts" / "qwen25_coder_7b_lab" / "status.json"
URL_WAIT_SECONDS = 1800


def write_operational_status(payload: dict) -> None:
    if not lab_status_is_safe(payload):
        raise ValueError("refusing to persist lab status with secrets")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".status.",
        suffix=".tmp",
        dir=STATUS_PATH.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, STATUS_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def wait_for_public_url(
    log_path: Path,
    *,
    process=None,
    timeout: float = URL_WAIT_SECONDS,
) -> str:
    deadline = time.time() + timeout
    while time.time() <= deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            url = extract_lab_public_url(text)
            if url:
                return url
        if process is not None and process.poll() is not None:
            raise RuntimeError("lab server exited before publishing a public URL")
        time.sleep(0.05)
    raise TimeoutError("lab public URL was not published")


def start_lab_and_return_url() -> str:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = SERVER_LOG.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(RUN_SCRIPT),
        "--auth-file",
        str(REMOTE_AUTH_FILE),
    ]
    try:
        proc = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(REMOTE_ROOT),
            start_new_session=True,
        )
    finally:
        log_handle.close()
    try:
        url = wait_for_public_url(SERVER_LOG, process=proc)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        write_operational_status(
            {
                "profile": "qwen25_coder_7b_lab.json",
                "state": "error",
                "gpu": "A100",
                "share_ready": False,
                "pid": proc.pid,
            }
        )
        raise
    payload = {
        "profile": "qwen25_coder_7b_lab.json",
        "state": "running",
        "gpu": "A100",
        "share_ready": True,
        "public_url": url,
        "pid": proc.pid,
    }
    write_operational_status(payload)
    return url


def main() -> int:
    url = start_lab_and_return_url()
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
