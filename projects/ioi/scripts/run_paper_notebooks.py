"""Execute paper notebooks 05-10 in order and write a compact manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
MANIFEST = RESULTS / "execution_manifest.json"
NOTEBOOKS = (
    "05_paper_dataset_and_baseline.ipynb",
    "06_circuit_discovery.ipynb",
    "07_path_patching.ipynb",
    "08_backup_name_movers.ipynb",
    "09_faithfulness_completeness_minimality.ipynb",
    "10_adversarial_examples.ipynb",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _write_manifest(manifest: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(MANIFEST)


def finalize_manifest(
    manifest: dict[str, object],
    *,
    summary_return_code: int,
    finished_at_utc: str,
) -> int:
    """Record summary status and only then mark the full run successful."""
    manifest["finished_at_utc"] = finished_at_utc
    manifest["summary"] = {
        "exit_status": (
            "success" if summary_return_code == 0 else "failed"
        ),
        "return_code": summary_return_code,
    }
    manifest["status"] = (
        "success" if summary_return_code == 0 else "failed"
    )
    return summary_return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--kernel", default="ioi-replication")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit execution without CUDA (GPU is required by default)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit an initially dirty source tree",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        parser.error("commit or clean source changes before fresh execution")
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("CUDA is unavailable; pass --allow-cpu only intentionally")

    source_commit = _git("rev-parse", "HEAD")
    device = {
        "type": "cuda" if torch.cuda.is_available() else "cpu",
        "name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            args.kernel,
            "--display-name",
            "Python (IOI Replication)",
        ],
        cwd=ROOT,
        check=True,
    )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "source_commit": source_commit,
        "initial_git_dirty": dirty,
        "device": device,
        "kernel": args.kernel,
        "timeout_seconds": args.timeout,
        "status": "running",
        "notebooks": [],
    }
    _write_manifest(manifest)
    environment = {
        **os.environ,
        "IOI_SOURCE_COMMIT": source_commit,
    }
    failed = False
    records: list[dict[str, object]] = []
    manifest["notebooks"] = records

    for index, notebook_name in enumerate(NOTEBOOKS):
        notebook_path = ROOT / "notebooks" / notebook_name
        started = _utc_now()
        start = time.perf_counter()
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
                str(notebook_path),
                f"--ExecutePreprocessor.kernel_name={args.kernel}",
                f"--ExecutePreprocessor.timeout={args.timeout}",
                "--ExecutePreprocessor.record_timing=True",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        records.append(
            {
                "notebook": f"notebooks/{notebook_name}",
                "started_at_utc": started,
                "finished_at_utc": _utc_now(),
                "duration_seconds": round(time.perf_counter() - start, 3),
                "exit_status": (
                    "success" if completed.returncode == 0 else "failed"
                ),
                "return_code": completed.returncode,
            }
        )
        _write_manifest(manifest)
        if completed.returncode != 0:
            failed = True
            for skipped_name in NOTEBOOKS[index + 1 :]:
                records.append(
                    {
                        "notebook": f"notebooks/{skipped_name}",
                        "started_at_utc": None,
                        "finished_at_utc": None,
                        "duration_seconds": None,
                        "exit_status": "not_run_after_failure",
                        "return_code": None,
                    }
                )
            break

    if failed:
        manifest["finished_at_utc"] = _utc_now()
        manifest["status"] = "failed"
        manifest["summary"] = {
            "exit_status": "not_run_after_failure",
            "return_code": None,
        }
        _write_manifest(manifest)
        return 1
    summary_return_code = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_paper_summary.py")],
        cwd=ROOT,
        check=False,
    ).returncode
    result = finalize_manifest(
        manifest,
        summary_return_code=summary_return_code,
        finished_at_utc=_utc_now(),
    )
    _write_manifest(manifest)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
