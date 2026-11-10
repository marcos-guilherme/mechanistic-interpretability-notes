from pathlib import Path
import importlib.util
import json


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_results.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_results(root: Path, *, ablation_rate: float = 0.0) -> None:
    results = root / "results"
    results.mkdir()
    (results / "final_evaluation.json").write_text(
        json.dumps(
            {
                "config": {"ce_tolerance": 0.1},
                "provenance": {"runtime": {"device": "cuda"}},
                "results": {
                    "selected": {"norm": 2.0},
                    "baseline": {
                        "harmful_refusal_rate": 1.0,
                        "harmless_refusal_rate": 0.0,
                    },
                    "ablation": {
                        "harmful_refusal_rate": ablation_rate,
                        "harmless_refusal_rate": 0.0,
                        "ce_delta": -0.2,
                        "harmless_kl": 0.03,
                    },
                    "addition": {"harmless_refusal_rate": 0.9},
                    "random_control": {"harmful_refusal_rate": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )


def test_validate_accepts_expected_causal_pattern(tmp_path) -> None:
    write_results(tmp_path)
    checks = load_validator().validate(tmp_path)
    assert checks
    assert all(checks.values())


def test_validate_rejects_ablation_without_effect(tmp_path) -> None:
    write_results(tmp_path, ablation_rate=1.0)
    checks = load_validator().validate(tmp_path)
    assert checks["ablation_reduces_harmful_refusal"] is False
