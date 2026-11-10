from __future__ import annotations

from pathlib import Path


SOURCE = Path("/content/hf_token")
DESTINATION = Path.home() / ".cache" / "huggingface" / "token"


def prepare_token(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    destination.chmod(0o600)
    return destination


def main() -> int:
    prepare_token(SOURCE, DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
