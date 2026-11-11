"""Fresh-process entrypoint for deterministic observation figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from jspace_llama.jsonio import strict_json_loads


MAX_COMPACT_RESULT_BYTES = 2_000_000


def _load_compact_result(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("figure input must be a regular file")
    if path.stat().st_size > MAX_COMPACT_RESULT_BYTES:
        raise ValueError("figure input is too large")
    document = strict_json_loads(path.read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("artifact") != "jspace_observation"
        or document.get("status") != "success"
    ):
        raise ValueError("figure input is not a successful compact result")
    return document


def render_from_paths(input_path: Path, output_dir: Path) -> list[Path]:
    if not input_path.is_absolute() or not output_dir.is_absolute():
        raise ValueError("figure renderer paths must be absolute")
    resolved_input = input_path.resolve(strict=True)
    resolved_output = output_dir.resolve()
    if resolved_output.parent != resolved_input.parent:
        raise ValueError("figure renderer paths must share one staging root")
    if os.path.lexists(resolved_output) and resolved_output.is_symlink():
        raise ValueError("figure output directory cannot be a symlink")
    result = _load_compact_result(resolved_input)

    from jspace_llama.observation_figures import write_observation_figures

    return write_observation_figures(result, resolved_output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = render_from_paths(args.input, args.output_dir)
    if len(paths) != 6:
        raise RuntimeError("observational figure set is incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
