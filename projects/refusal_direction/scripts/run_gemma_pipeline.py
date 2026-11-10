from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "baseline": "01_dataset_and_baseline.ipynb",
    "extract": "02_extract_candidate_directions.ipynb",
    "intervene": "03_causal_interventions.ipynb",
    "evaluate": "04_evaluation_and_validation.ipynb",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(NOTEBOOKS))
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    notebook = ROOT / "notebooks" / NOTEBOOKS[args.stage]
    if not notebook.exists():
        parser.error(f"missing notebook: {notebook}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook),
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
            f"--ExecutePreprocessor.timeout={args.timeout}",
        ],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
