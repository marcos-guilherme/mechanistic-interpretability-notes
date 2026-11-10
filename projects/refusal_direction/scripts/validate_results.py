from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate(root: Path) -> dict[str, bool]:
    payload = json.loads(
        (root / "results" / "final_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    config = payload["config"]
    provenance = payload["provenance"]
    results = payload["results"]
    baseline = results["baseline"]
    ablation = results["ablation"]
    addition = results["addition"]
    control = results["random_control"]
    tolerance = config["ce_tolerance"]

    ablation_effect = (
        baseline["harmful_refusal_rate"]
        - ablation["harmful_refusal_rate"]
    )
    control_effect = (
        baseline["harmful_refusal_rate"]
        - control["harmful_refusal_rate"]
    )
    return {
        "ran_on_cuda": provenance["runtime"]["device"] == "cuda",
        "selected_direction_is_nonzero": results["selected"]["norm"] > 0,
        "ablation_reduces_harmful_refusal": ablation_effect > 0,
        "addition_increases_false_refusal": (
            addition["harmless_refusal_rate"]
            > baseline["harmless_refusal_rate"]
        ),
        "random_control_is_weaker_than_ablation": (
            abs(control_effect) < abs(ablation_effect)
        ),
        "ablation_preserves_harmless_refusal": (
            ablation["harmless_refusal_rate"]
            <= baseline["harmless_refusal_rate"]
        ),
        "ablation_kl_within_tolerance": (
            ablation["harmless_kl"] <= tolerance
        ),
        "ablation_ce_not_degraded": ablation["ce_delta"] <= tolerance,
    }


def main() -> int:
    checks = validate(ROOT)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
