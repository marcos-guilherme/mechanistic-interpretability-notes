"""Install pinned readout/evaluation dependencies and frozen source tensors."""

from __future__ import annotations

import importlib.metadata
import inspect
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
EVALPLUS_RUNTIME_DEPENDENCIES = (
    "appdirs==1.4.4",
    "fire==0.7.0",
    "multipledispatch==1.0.0",
    "psutil==6.1.1",
    "rich==13.9.4",
    "tempdir==0.7.1",
    "termcolor==2.5.0",
    "tqdm==4.67.1",
    "tree_sitter==0.23.2",
    "tree-sitter-python==0.23.6",
    "wget==3.2",
)


def _install(spec: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", spec],
        check=True,
    )


def _install_evalplus_runtime_dependencies() -> None:
    # EvalPlus declares optional model-provider stacks alongside its execution
    # dependencies. Install only the packages exercised by data loading,
    # sanitization and the local correctness worker.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *EVALPLUS_RUNTIME_DEPENDENCIES,
        ],
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


def smoke_evalplus_contract(
    *,
    sanitize=None,
    check_correctness=None,
    reliability_guard=None,
) -> str:
    """Exercise harmless sanitization and pin the worker-facing EvalPlus API."""
    if sanitize is None or check_correctness is None or reliability_guard is None:
        from evalplus.eval.utils import reliability_guard as imported_guard
        from evalplus.evaluate import check_correctness as imported_check
        from evalplus.sanitize import sanitize as imported_sanitize

        sanitize = imported_sanitize
        check_correctness = imported_check
        reliability_guard = imported_guard
    if not all(callable(item) for item in (sanitize, check_correctness, reliability_guard)):
        raise RuntimeError("EvalPlus execution contract is unavailable")
    required_parameters = {
        "dataset",
        "completion_id",
        "problem",
        "solution",
        "expected_output",
    }
    if not required_parameters.issubset(inspect.signature(check_correctness).parameters):
        raise RuntimeError("EvalPlus check_correctness API differs from the worker contract")
    harmless = "def add(a, b):\n    return a + b\n"
    sanitized = sanitize(harmless, entrypoint="add")
    if not isinstance(sanitized, str) or "def add" not in sanitized:
        raise RuntimeError("EvalPlus sanitize smoke test failed")
    return sanitized


def assemble_staged_lens() -> Path:
    destination = Path(REMOTE_CODE_EVAL_LENS_STAGING)
    parts = sorted(destination.parent.glob(destination.name + ".part*"))
    if not parts:
        if destination.is_file():
            return destination
        raise FileNotFoundError("code-eval lens staging parts are missing")
    expected = [
        destination.name + f".part{index:03d}" for index in range(len(parts))
    ]
    if [part.name for part in parts] != expected:
        raise ValueError("code-eval lens staging parts are not contiguous")
    temporary = destination.with_name(destination.name + ".assembling")
    try:
        with temporary.open("wb") as output:
            for part in parts:
                if not part.is_file() or part.stat().st_size <= 0:
                    raise ValueError(f"invalid code-eval lens part: {part.name}")
                with part.open("rb") as incoming:
                    while chunk := incoming.read(8 * 1024**2):
                        output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    for part in parts:
        part.unlink()
    return destination


def main() -> int:
    if JLENS_UPSTREAM_SPEC.rsplit("@", 1)[-1] != JLENS_COMMIT:
        raise ValueError("Jacobian-lens install spec differs from the pinned commit")
    _install(JLENS_UPSTREAM_SPEC)
    _install_evalplus_runtime_dependencies()
    _install(EVALPLUS_SPEC)
    _install_purplellama_source()
    assert_jlens_contract()
    if _installed_commit("evalplus") != EVALPLUS_COMMIT:
        raise ValueError("installed EvalPlus commit differs from the pinned commit")
    smoke_evalplus_contract()
    staged_lens = assemble_staged_lens()
    atomic_install_code_eval_sources(
        staged_direction=Path(REMOTE_CODE_EVAL_DIRECTION_STAGING),
        staged_lens=staged_lens,
        project=REMOTE_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
