"""Install pinned readout/evaluation dependencies and frozen source tensors."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = Path("/content/refusal_direction")
_SRC = "/content/refusal_direction/src"
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from refusal_direction_replication.qwen_colab_transport import (
    REMOTE_CODE_EVAL_DIRECTION_STAGING,
    REMOTE_CODE_EVAL_LENS_STAGING,
    atomic_install_code_eval_sources,
)
from refusal_direction_replication.qwen_jlens import (
    JLENS_UPSTREAM_SPEC,
    assert_jlens_contract,
)

EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
EVALPLUS_SPEC = (
    "git+https://github.com/evalplus/evalplus.git@" + EVALPLUS_COMMIT
)
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
PURPLE_LLAMA_COMMIT = "0654501399e7cbd560db9501965dc09c567a5563"
SOURCE_ROOT = Path("/content/qwen_code_eval_sources")


def _install(spec: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", spec],
        check=True,
    )


def _installed_commit(distribution_name: str) -> str:
    distribution = importlib.metadata.distribution(distribution_name)
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise ValueError(f"{distribution_name} direct_url metadata missing")
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    if not commit:
        raise ValueError(f"{distribution_name} commit metadata missing")
    return str(commit)


def _install_purplellama_source() -> None:
    destination = SOURCE_ROOT / "PurpleLlama"
    if not (destination / ".git").is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=destination, check=True)
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/meta-llama/PurpleLlama.git",
            ],
            cwd=destination,
            check=True,
        )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", PURPLE_LLAMA_COMMIT],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=destination,
        check=True,
    )
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != PURPLE_LLAMA_COMMIT:
        raise ValueError("PurpleLlama checkout differs from the pinned commit")


def main() -> int:
    if JLENS_UPSTREAM_SPEC.rsplit("@", 1)[-1] != JLENS_COMMIT:
        raise ValueError("Jacobian-lens install spec differs from the pinned commit")
    _install(JLENS_UPSTREAM_SPEC)
    _install(EVALPLUS_SPEC)
    _install_purplellama_source()
    assert_jlens_contract()
    if _installed_commit("evalplus") != EVALPLUS_COMMIT:
        raise ValueError("installed EvalPlus commit differs from the pinned commit")
    from evalplus.eval import untrusted_check
    from evalplus.sanitize import sanitize

    if not callable(untrusted_check) or not callable(sanitize):
        raise ValueError("EvalPlus execution contract is unavailable")
    atomic_install_code_eval_sources(
        staged_direction=Path(REMOTE_CODE_EVAL_DIRECTION_STAGING),
        staged_lens=Path(REMOTE_CODE_EVAL_LENS_STAGING),
        project=REMOTE_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
