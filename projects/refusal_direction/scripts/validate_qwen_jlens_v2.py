from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.qwen_jlens_v2_validate import (
    validate_qwen_jlens_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def main(root: Path = ROOT) -> int:
    print("Qwen Jacobian lens v2 (long context)")
    try:
        checks = validate_qwen_jlens_v2(root)
    except ValueError as exc:
        print(f"FAIL {exc}")
        return 1
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
