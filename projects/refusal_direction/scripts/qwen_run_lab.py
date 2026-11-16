"""Launch the Qwen intervention lab Gradio UI on a GPU host (Colab A100)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_PROJECT = _SCRIPT.parents[1]
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from refusal_direction_replication.qwen_lab_app import (  # noqa: E402
    GRADIO_PIN_VERSION,
    build_gradio_app,
    bootstrap_intervention_lab,
    lab_operational_log,
    require_launch_authentication,
    resolve_auth_credentials,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor do laboratório Qwen.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("QWEN_LAB_ROOT", _PROJECT)).resolve(),
        help="Raiz do projeto refusal_direction.",
    )
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=None,
        help="Caminho explícito da direção congelada.",
    )
    parser.add_argument(
        "--lens-path",
        type=Path,
        default=None,
        help="Caminho explícito da J-lens ajustada.",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    auth = resolve_auth_credentials()
    launch_auth = require_launch_authentication(
        share=bool(args.share),
        server_name=args.server_name,
        auth=auth,
    )
    bundle = bootstrap_intervention_lab(
        args.root,
        direction_path=args.direction_path,
        lens_path=args.lens_path,
    )
    demo = build_gradio_app(
        bundle.engine,
        preflight_summary=bundle.preflight_summary,
        auth=launch_auth if launch_auth != ("", "") else None,
    )
    lab_operational_log(status="launching", share=bool(args.share), gradio=GRADIO_PIN_VERSION)
    loopback = str(args.server_name).strip().lower() in {"127.0.0.1", "localhost", "::1"}
    show_error = loopback and not bool(args.share)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        auth=launch_auth if launch_auth != ("", "") else None,
        show_error=show_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())