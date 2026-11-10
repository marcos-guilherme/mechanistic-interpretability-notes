from pathlib import Path
import tarfile


ROOT = Path("/content/refusal_direction")
RESULTS = ROOT / "results"
ARCHIVE = Path("/content/refusal_results.tar.gz")


def main() -> int:
    if not RESULTS.exists():
        raise FileNotFoundError(RESULTS)
    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(RESULTS, arcname="results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
