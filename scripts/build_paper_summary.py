"""Build the compact paper-result index from canonical JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
OUTPUT = RESULTS / "summary.json"
ARTIFACTS = {
    "baseline": "baseline_seed_42_n_256.json",
    "circuit": "06_circuit_discovery.json",
    "path_patching": "07_path_patching.json",
    "backup": "08_backup_name_movers.json",
    "validation": "09_faithfulness_completeness_minimality.json",
    "adversarial": "10_adversarial_examples.json",
}


def _read(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _source_commit(
    artifact: dict[str, Any],
    fallback: str,
) -> str:
    return str(
        artifact.get("traceability", {}).get("source_commit")
        or artifact.get("config", {}).get("git_commit")
        or fallback
    )


def build_summary() -> dict[str, Any]:
    artifacts = {
        name: _read(path)
        for name, path in ARTIFACTS.items()
    }
    manifest_path = RESULTS / "execution_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    fallback_commit = str(manifest.get("source_commit") or _git_head())
    validation = artifacts["validation"]
    adversarial = artifacts["adversarial"]
    backup = artifacts["backup"]
    path_patching = artifacts["path_patching"]
    circuit = artifacts["circuit"]

    return {
        "schema_version": 1,
        "paper": "arXiv:2211.00593",
        "sources": {
            name: {
                "path": f"results/paper/{path}",
                "source_commit": _source_commit(
                    artifacts[name],
                    fallback_commit,
                ),
                "config": artifacts[name].get("config", {}),
            }
            for name, path in ARTIFACTS.items()
        },
        "key_results": {
            "baseline": artifacts["baseline"]["observed"],
            "circuit_discovery": {
                "baseline_logit_difference": circuit[
                    "baseline_logit_difference"
                ],
                "published_class_mean_patch_effect": {
                    name: values["mean_patch_effect"]
                    for name, values in circuit["published_classes"].items()
                },
            },
            "path_patching": {
                name: values["effect"]
                for name, values in path_patching[
                    "transformerlens_direct_path_all_positions"
                ].items()
            },
            "backup_compensation_fraction": backup[
                "compensation_fraction"
            ]["value"],
            "faithfulness": {
                "model": validation["faithfulness"]["model"]["mean"],
                "circuit": validation["faithfulness"]["circuit"]["mean"],
                "absolute_gap": validation["faithfulness"]["absolute_gap"],
                "retained_fraction": validation["faithfulness"][
                    "retained_fraction"
                ],
            },
            "completeness_greedy_max": max(
                item["incompleteness"]
                for item in validation["completeness"]["greedy_top5"]
            ),
            "minimality": {
                "point_above_1_percent": sum(
                    item["passes_threshold"]
                    for item in validation["minimality"]
                ),
                "ci95_above_1_percent": sum(
                    item["ci_supports_threshold"]
                    for item in validation["minimality"]
                ),
                "total": len(validation["minimality"]),
            },
            "adversarial": {
                "baseline_logit_difference": adversarial["protocols"][
                    "baseline"
                ]["metrics"]["logit_difference_io_minus_s"]["mean"],
                "paper_io_single_logit_difference": adversarial["protocols"][
                    "paper_io_single"
                ]["metrics"]["logit_difference_io_minus_s"]["mean"],
                "paper_io_single_logit_delta": adversarial["protocols"][
                    "paper_io_single"
                ]["delta_from_baseline"][
                    "logit_difference_io_minus_s"
                ]["mean"],
                "paper_s_control_logit_difference": adversarial["protocols"][
                    "paper_s_control"
                ]["metrics"]["logit_difference_io_minus_s"]["mean"],
            },
        },
    }


def _serialized(summary: dict[str, Any]) -> str:
    return json.dumps(
        summary,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if summary.json differs from the derived content",
    )
    args = parser.parse_args()
    expected = _serialized(build_summary())
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != expected:
            print("summary.json is stale; regenerate it", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
