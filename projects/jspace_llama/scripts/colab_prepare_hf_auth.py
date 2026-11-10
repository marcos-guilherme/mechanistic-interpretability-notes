from __future__ import annotations

import os
from pathlib import Path
import re


SOURCE = Path("/content/hf_token")
DESTINATION = Path.home() / ".cache" / "huggingface" / "token"
SUCCESS_MARKER = "JSPACE_HF_AUTH_SUCCESS"
SUCCESS_NONCE_ENV = "JSPACE_SUCCESS_NONCE"


def consume_success_nonce() -> str:
    nonce = os.environ.pop(SUCCESS_NONCE_ENV, None)
    if nonce is None or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise RuntimeError("missing or invalid success nonce")
    return nonce


def prepare_token(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    destination.chmod(0o600)
    return destination


def main() -> int:
    nonce = consume_success_nonce()
    prepare_token(SOURCE, DESTINATION)
    print(f"{SUCCESS_MARKER}:{nonce}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
