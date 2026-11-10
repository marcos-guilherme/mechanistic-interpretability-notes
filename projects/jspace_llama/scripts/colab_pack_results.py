from __future__ import annotations

from pathlib import Path
import tarfile


ROOT = Path("/content/jspace_llama")
ARCHIVE = Path("/content/jspace_results.tar.gz")
RESULT_ALLOWLIST = (
    "execution_manifest.json",
    "final_evaluation.json",
)


def pack_results(root: Path, archive: Path) -> Path:
    results = root / "results"
    missing = [name for name in RESULT_ALLOWLIST if not (results / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required result artifact(s): " + ", ".join(missing)
        )

    with tarfile.open(archive, "w:gz") as handle:
        for name in RESULT_ALLOWLIST:
            handle.add(results / name, arcname=f"results/{name}")
    return archive


def main() -> int:
    pack_results(ROOT, ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
