from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.qwen_dose_validate import validate_qwen_dose
from refusal_direction_replication.qwen_jlens_validate import validate_qwen_jlens
from refusal_direction_replication.qwen_pilot_validate import validate_qwen_pilot


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    original = validate_qwen_pilot(ROOT)
    dose = validate_qwen_dose(ROOT)
    jlens = validate_qwen_jlens(ROOT)
    print("original Qwen pilot")
    for name, passed in original.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print("Qwen dose follow-up")
    for name, passed in dose.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print("Qwen Jacobian lens")
    for name, passed in jlens.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    ok = all(original.values()) and all(dose.values()) and all(jlens.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
