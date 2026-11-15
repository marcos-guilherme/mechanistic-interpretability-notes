from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.qwen_pilot_validate import validate_qwen_pilot


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = validate_qwen_pilot(ROOT)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
