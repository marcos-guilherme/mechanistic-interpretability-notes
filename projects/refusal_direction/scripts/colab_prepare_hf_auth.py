from pathlib import Path


SOURCE = Path("/content/hf_token")
DESTINATION = Path.home() / ".cache" / "huggingface" / "token"


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.replace(DESTINATION)
    DESTINATION.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
