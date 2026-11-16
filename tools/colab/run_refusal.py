from __future__ import annotations

import argparse
import io
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "refusal_direction"
AUTH = ["colab", "--auth=adc"]
PROFILE_SESSIONS = {
    "gemma": "refusal-gemma",
    "qwen25-coder": "refusal-qwen",
    "qwen25-coder-dose": "refusal-qwen",
    "qwen25-coder-jlens": "refusal-qwen",
    "qwen25-coder-code-eval": "refusal-qwen",
}
EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
}


def huggingface_token_path(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home()
    return base / ".cache" / "huggingface" / "token"


def start_command(session: str, gpu: str) -> list[str]:
    return [*AUTH, "new", "-s", session, "--gpu", gpu]


def stop_command(session: str) -> list[str]:
    return [*AUTH, "stop", "-s", session]


def upload_command(session: str, local: Path, remote: str) -> list[str]:
    return [
        *AUTH,
        "upload",
        "-s",
        session,
        str(local),
        remote,
    ]


def exec_command(
    session: str,
    file: Path,
    *,
    timeout: int,
) -> list[str]:
    return [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(file),
        "--timeout",
        str(timeout),
    ]


def download_command(session: str, remote: str, local: Path) -> list[str]:
    return [
        *AUTH,
        "download",
        "-s",
        session,
        remote,
        str(local),
    ]


def resolve_session(profile: str, session: str | None) -> str:
    if session is not None:
        return session
    return PROFILE_SESSIONS[profile]


def fetch_pack_script(profile: str) -> Path:
    if profile == "qwen25-coder-code-eval":
        return PROJECT / "scripts" / "colab_pack_qwen_code_eval_results.py"
    if profile == "qwen25-coder-jlens":
        return PROJECT / "scripts" / "colab_pack_qwen_jlens_results.py"
    if profile == "qwen25-coder-dose":
        return PROJECT / "scripts" / "colab_pack_qwen_dose_results.py"
    if profile == "qwen25-coder":
        return PROJECT / "scripts" / "colab_pack_qwen_results.py"
    return PROJECT / "scripts" / "colab_pack_results.py"


def stamps_implementation_commit(profile: str) -> bool:
    return profile.startswith("qwen25-coder")


def status_command(session: str, *, profile: str = "qwen25-coder") -> list[str]:
    script = PROJECT / "scripts" / "qwen_run_status.py"
    if profile == "qwen25-coder-code-eval":
        script = PROJECT / "scripts" / "qwen_run_code_eval_status.py"
    elif profile == "qwen25-coder-jlens":
        script = PROJECT / "scripts" / "qwen_run_jlens_status.py"
    elif profile == "qwen25-coder-dose":
        script = PROJECT / "scripts" / "qwen_run_dose_status.py"
    return [
        *AUTH,
        "exec",
        "-s",
        session,
        "-f",
        str(script),
        "--timeout",
        "120",
    ]


def pilot_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_pilot.py",
        timeout=timeout,
    )


def dose_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_dose.py",
        timeout=timeout,
    )


def jlens_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_jlens.py",
        timeout=timeout,
    )


def code_eval_command(session: str, *, timeout: int) -> list[str]:
    return exec_command(
        session,
        PROJECT / "scripts" / "qwen_run_code_eval.py",
        timeout=timeout,
    )


def extract_qwen_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_archive,
    )

    extract_qwen_archive(archive, project)


def extract_qwen_dose_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_dose_archive,
    )

    extract_qwen_dose_archive(archive, project)


def extract_qwen_jlens_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_jlens_archive,
    )

    extract_qwen_jlens_archive(archive, project)


def extract_qwen_code_eval_results_archive(archive: Path, project: Path) -> None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        extract_qwen_code_eval_archive,
    )

    extract_qwen_code_eval_archive(archive, project)


def require_dose_source_direction(project: Path) -> Path:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_dose_source_direction,
    )

    return require_local_dose_source_direction(project)


def optional_jlens_checkpoint(project: Path) -> Path | None:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        optional_local_jlens_checkpoint,
    )

    return optional_local_jlens_checkpoint(project)


def optional_jlens_resume_bundle(project: Path) -> list[tuple[Path, str]]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        optional_local_jlens_resume_bundle,
    )

    return optional_local_jlens_resume_bundle(project)


def require_code_eval_sources(project: Path) -> tuple[Path, Path]:
    sys.path.insert(0, str(project / "src"))
    from refusal_direction_replication.qwen_colab_transport import (  # noqa: WPS433
        require_local_code_eval_sources,
    )

    return require_local_code_eval_sources(project)


def bootstrap_scripts(profile: str) -> list[Path]:
    scripts = [PROJECT / "scripts" / "colab_bootstrap.py"]
    if profile == "qwen25-coder-dose":
        scripts.append(PROJECT / "scripts" / "colab_install_dose_source_direction.py")
    if profile == "qwen25-coder-jlens":
        scripts.append(PROJECT / "scripts" / "colab_install_jlens.py")
        scripts.append(PROJECT / "scripts" / "colab_install_jlens_checkpoint.py")
    if profile == "qwen25-coder-code-eval":
        scripts.append(PROJECT / "scripts" / "colab_install_code_eval.py")
    return scripts


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in EXCLUDED_PARTS for part in parts):
        return None
    if info.name.endswith("_output.ipynb"):
        return None
    return info


def current_implementation_commit(cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else ""
    if not commit or commit.casefold() == "unknown":
        raise ValueError("implementation commit is missing")
    return commit


def build_archive(
    *,
    stamp_commit: bool = False,
    project: Path | None = None,
    staging: Path | None = None,
    commit: str | None = None,
) -> Path:
    project = project or PROJECT
    staging = staging or (ROOT / ".colab-staging")
    staging.mkdir(exist_ok=True)
    archive = staging / "refusal_direction.tar.gz"

    def filter_info(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        filtered = _archive_filter(info)
        if filtered is None:
            return None
        if stamp_commit and Path(info.name).name == "IMPLEMENTATION_COMMIT":
            return None
        return filtered

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(
            project,
            arcname="refusal_direction",
            filter=filter_info,
        )
        if stamp_commit:
            hashed = commit or current_implementation_commit()
            payload = (hashed + "\n").encode("utf-8")
            info = tarfile.TarInfo(name="refusal_direction/IMPLEMENTATION_COMMIT")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "probe",
            "start",
            "sync",
            "bootstrap",
            "auth-hf",
            "execute",
            "fetch",
            "status",
            "pilot",
            "dose",
            "jlens",
            "code-eval",
            "stop",
        ),
    )
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_SESSIONS),
        default="gemma",
    )
    parser.add_argument(
        "--gpu",
        choices=("T4", "L4", "A100"),
        default="L4",
    )
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    session = resolve_session(args.profile, args.session)

    if args.action == "probe":
        run([*AUTH, "sessions"])
    elif args.action == "start":
        run(start_command(session, args.gpu))
    elif args.action == "sync":
        direction = None
        code_eval_sources = None
        if args.profile == "qwen25-coder-dose":
            direction = require_dose_source_direction(PROJECT)
        elif args.profile == "qwen25-coder-code-eval":
            code_eval_sources = require_code_eval_sources(PROJECT)
        archive = build_archive(
            stamp_commit=stamps_implementation_commit(args.profile)
        )
        run(
            upload_command(
                session,
                archive,
                "/content/refusal_direction.tar.gz",
            )
        )
        if direction is not None:
            run(
                upload_command(
                    session,
                    direction,
                    "/content/dose_source_direction.pt",
                )
            )
        if code_eval_sources is not None:
            source_direction, source_lens = code_eval_sources
            run(
                upload_command(
                    session,
                    source_direction,
                    "/content/code_eval_source_direction.pt",
                )
            )
            run(
                upload_command(
                    session,
                    source_lens,
                    "/content/code_eval_source_lens.pt",
                )
            )
        if args.profile == "qwen25-coder-jlens":
            for local_path, remote in optional_jlens_resume_bundle(PROJECT):
                run(upload_command(session, local_path, remote))
    elif args.action == "bootstrap":
        for script in bootstrap_scripts(args.profile):
            run(
                exec_command(
                    session,
                    script,
                    timeout=600,
                )
            )
    elif args.action == "auth-hf":
        token = huggingface_token_path()
        if not token.is_file():
            parser.error(
                "Hugging Face token not found; run `uv run hf auth login`"
            )
        run(
            upload_command(
                session,
                token,
                "/content/hf_token",
            )
        )
        run(
            exec_command(
                session,
                PROJECT / "scripts" / "colab_prepare_hf_auth.py",
                timeout=60,
            )
        )
    elif args.action == "execute":
        if args.notebook is None:
            parser.error("--notebook is required for execute")
        notebook = args.notebook.resolve()
        if not notebook.is_file():
            parser.error(f"notebook does not exist: {notebook}")
        run(
            exec_command(
                session,
                notebook,
                timeout=args.timeout,
            )
        )
    elif args.action == "fetch":
        if args.profile == "qwen25-coder-code-eval":
            remote_archive = "/content/qwen_code_eval_results.tar.gz"
            local_name = "qwen_code_eval_results.tar.gz"
        elif args.profile == "qwen25-coder-jlens":
            remote_archive = "/content/qwen_jlens_results.tar.gz"
            local_name = "qwen_jlens_results.tar.gz"
        elif args.profile == "qwen25-coder-dose":
            remote_archive = "/content/qwen_dose_results.tar.gz"
            local_name = "qwen_dose_results.tar.gz"
        elif args.profile == "qwen25-coder":
            remote_archive = "/content/qwen_results.tar.gz"
            local_name = "qwen_results.tar.gz"
        else:
            remote_archive = "/content/refusal_results.tar.gz"
            local_name = "refusal_results.tar.gz"
        run(
            exec_command(
                session,
                fetch_pack_script(args.profile),
                timeout=300,
            )
        )
        local = ROOT / ".colab-staging" / local_name
        local.parent.mkdir(exist_ok=True)
        run(
            download_command(
                session,
                remote_archive,
                local,
            )
        )
        if args.profile == "qwen25-coder-code-eval":
            extract_qwen_code_eval_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder-jlens":
            extract_qwen_jlens_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder-dose":
            extract_qwen_dose_results_archive(local, PROJECT)
        elif args.profile == "qwen25-coder":
            extract_qwen_results_archive(local, PROJECT)
        else:
            with tarfile.open(local, "r:gz") as handle:
                handle.extractall(PROJECT)
    elif args.action == "status":
        if args.profile not in (
            "qwen25-coder",
            "qwen25-coder-dose",
            "qwen25-coder-jlens",
            "qwen25-coder-code-eval",
        ):
            parser.error("status is only available for Qwen profiles")
        run(status_command(session, profile=args.profile))
    elif args.action == "pilot":
        if args.profile != "qwen25-coder":
            parser.error("pilot is only available for --profile qwen25-coder")
        run(pilot_command(session, timeout=args.timeout))
    elif args.action == "dose":
        if args.profile != "qwen25-coder-dose":
            parser.error("dose is only available for --profile qwen25-coder-dose")
        run(dose_command(session, timeout=args.timeout))
    elif args.action == "jlens":
        if args.profile != "qwen25-coder-jlens":
            parser.error("jlens is only available for --profile qwen25-coder-jlens")
        run(jlens_command(session, timeout=args.timeout))
    elif args.action == "code-eval":
        if args.profile != "qwen25-coder-code-eval":
            parser.error(
                "code-eval is only available for --profile qwen25-coder-code-eval"
            )
        run(code_eval_command(session, timeout=args.timeout))
    elif args.action == "stop":
        run(stop_command(session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())