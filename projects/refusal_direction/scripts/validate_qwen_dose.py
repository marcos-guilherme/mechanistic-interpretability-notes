from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.qwen_dose_validate import validate_qwen_dose
from refusal_direction_replication.qwen_pilot_validate import validate_qwen_pilot


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    original = validate_qwen_pilot(ROOT)
    dose = validate_qwen_dose(ROOT)
    print("original Qwen pilot")
    for name, passed in original.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print("Qwen dose follow-up")
    for name, passed in dose.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    original_ok = all(original.values())
    dose_ok = all(dose.values())
    return 0 if original_ok and dose_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
