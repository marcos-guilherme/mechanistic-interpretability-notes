from __future__ import annotations

from pathlib import Path

from refusal_direction_replication.qwen_causal_jlens_validate import (
    validate_qwen_causal_jlens,
)


ROOT = Path(__file__).resolve().parents[1]


def main(root: Path = ROOT, *, allow_fixture_contract: bool = False) -> int:
    print("Qwen causal J-lens fidelity")
    try:
        checks = validate_qwen_causal_jlens(
            root, allow_fixture_contract=allow_fixture_contract
        )
    except ValueError as exc:
        print(f"FAIL {exc}")
        return 1
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
