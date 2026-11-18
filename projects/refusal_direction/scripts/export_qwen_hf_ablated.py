"""Export the Qwen refusal-ablated Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_PROJECT = _SCRIPT.parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from refusal_direction_replication.qwen_hf_export import (  # noqa: E402
    HF_ABLATED_PROFILE,
    export_qwen_hf_ablated,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the Qwen refusal-ablated Hugging Face checkpoint."
    )
    parser.add_argument("--root", type=Path, default=_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--profile", default=HF_ABLATED_PROFILE)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    result = export_qwen_hf_ablated(
        args.root.resolve(),
        output_dir=output_dir,
        profile=args.profile,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
