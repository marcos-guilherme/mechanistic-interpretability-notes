from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).parents[3]
PROJECT = ROOT / "projects" / "jspace_llama"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
BOOTSTRAP_PATH = PROJECT / "scripts" / "colab_bootstrap.py"
AUTH_PATH = PROJECT / "scripts" / "colab_prepare_hf_auth.py"
PACK_PATH = PROJECT / "scripts" / "colab_pack_results.py"


def load_module(path: Path, name: str):
    assert path.is_file(), f"missing required module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_commands_use_adc_session_and_explicit_resources(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_commands")
    local = tmp_path / "payload.tar.gz"

    assert runner.start_command("jspace-llama", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "jspace-llama",
        "--gpu",
        "L4",
    ]
    assert runner.stop_command("jspace-llama") == [
        "colab",
        "--auth=adc",
        "stop",
        "-s",
        "jspace-llama",
    ]
    assert runner.exec_command(
        "jspace-llama", Path("stage.py"), timeout=300
    ) == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "jspace-llama",
        "-f",
        "stage.py",
        "--timeout",
        "300",
    ]
    assert runner.upload_command(
        "jspace-llama", local, "/content/payload.tar.gz"
    ) == [
        "colab",
        "--auth=adc",
        "upload",
        "-s",
        "jspace-llama",
        str(local),
        "/content/payload.tar.gz",
    ]
    assert runner.download_command(
        "jspace-llama", "/content/results.tar.gz", local
    ) == [
        "colab",
        "--auth=adc",
        "download",
        "-s",
        "jspace-llama",
        "/content/results.tar.gz",
        str(local),
    ]


def test_cli_defaults_to_jspace_session_and_l4() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_defaults")

    args = runner.parse_args(["start"])

    assert args.session == "jspace-llama"
    assert args.gpu == "L4"


@pytest.mark.parametrize("gpu", ["T4", "L4", "A100"])
def test_cli_accepts_only_planned_gpu_types(gpu: str) -> None:
    runner = load_module(RUNNER_PATH, f"run_jspace_gpu_{gpu}")

    assert runner.parse_args(["start", "--gpu", gpu]).gpu == gpu


def test_execute_requires_notebook_and_timeout() -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_execute_args")

    with pytest.raises(SystemExit):
        runner.parse_args(["execute", "--timeout", "600"])
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "execute",
                "--notebook",
                "projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
            ]
        )

    args = runner.parse_args(
        [
            "execute",
            "--notebook",
            "projects/jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
            "--timeout",
            "600",
        ]
    )
    assert args.timeout == 600


def test_archive_contains_only_remote_project_inputs(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_archive")
    project = tmp_path / "projects" / "jspace_llama"
    files = {
        "pyproject.toml": "project",
        "uv.lock": "lock",
        "README.md": "docs",
        "src/jspace_llama/core.py": "source",
        "tests/test_core.py": "tests",
        "scripts/colab_bootstrap.py": "script",
        "notebooks/01_readout_and_causal_swap.ipynb": "{}",
        ".git/config": "git",
        ".venv/secret.py": "venv",
        "__pycache__/core.pyc": "cache",
        ".pytest_cache/state": "cache",
        ".ipynb_checkpoints/draft.ipynb": "{}",
        "artifacts/raw.json": "raw",
        "results/final_evaluation.json": "{}",
        "notebooks/01_readout_and_causal_swap_output.ipynb": "{}",
        ".env": "HF_TOKEN=secret",
        "credentials.json": "secret",
        "scripts/adc_credentials.json": "secret",
        "src/jspace_llama/api_token.txt": "secret",
        "models/model.bin": "heavy",
        "lenses/lens.pt": "heavy",
        "cache/weights.safetensors": "heavy",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    archive = runner.build_archive(
        project=project,
        staging=tmp_path / ".colab-staging",
    )

    with tarfile.open(archive, "r:gz") as handle:
        archived_files = {
            member.name for member in handle.getmembers() if member.isfile()
        }
    assert archived_files == {
        "jspace_llama/pyproject.toml",
        "jspace_llama/uv.lock",
        "jspace_llama/README.md",
        "jspace_llama/src/jspace_llama/core.py",
        "jspace_llama/tests/test_core.py",
        "jspace_llama/scripts/colab_bootstrap.py",
        "jspace_llama/notebooks/01_readout_and_causal_swap.ipynb",
    }


def test_bootstrap_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_traversal")
    archive = tmp_path / "bad.tar.gz"
    source = tmp_path / "payload.txt"
    source.write_text("unsafe", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="unsafe archive member"):
        bootstrap.safe_extract(archive, tmp_path / "destination")

    assert not (tmp_path / "escaped.txt").exists()


def test_bootstrap_uses_locked_metadata_and_editable_install(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_commands")
    project = tmp_path / "jspace_llama"
    requirements = tmp_path / "requirements.txt"

    commands = bootstrap.dependency_commands(project, requirements)

    assert ["uv", "export", "--locked"] == commands[1][:3]
    assert "--project" in commands[1]
    assert str(project) in commands[1]
    assert ["uv", "pip", "install", "--system"] == commands[2][:4]
    assert commands[3][-2:] == ["--editable", str(project)]


def test_bootstrap_cuda_diagnostics_fail_clearly_without_cuda(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "jspace_bootstrap_cuda")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeVersion:
        cuda = None

    class FakeTorch:
        __version__ = "test-torch"
        cuda = FakeCuda()
        version = FakeVersion()

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        bootstrap.print_runtime_diagnostics(FakeTorch())

    output = capsys.readouterr().out
    assert "Python:" in output
    assert "Torch: test-torch" in output
    assert "CUDA available: False" in output


def test_huggingface_token_uses_standard_path_and_is_never_printed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_hf_path")
    auth = load_module(AUTH_PATH, "jspace_hf_auth")
    source = tmp_path / "hf_token"
    destination = tmp_path / ".cache" / "huggingface" / "token"
    secret = "hf_do_not_disclose"
    source.write_text(secret, encoding="utf-8")

    assert runner.huggingface_token_path(tmp_path) == destination
    assert auth.prepare_token(source, destination) == destination

    assert destination.read_text(encoding="utf-8") == secret
    assert not source.exists()
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_results_pack_contains_only_strict_allowlist(tmp_path: Path) -> None:
    pack = load_module(PACK_PATH, "jspace_pack_results")
    root = tmp_path / "jspace_llama"
    results = root / "results"
    results.mkdir(parents=True)
    (results / "execution_manifest.json").write_text("{}", encoding="utf-8")
    (results / "final_evaluation.json").write_text("{}", encoding="utf-8")
    (results / "raw_generations.json").write_text("secret", encoding="utf-8")
    (results / "tensor.pt").write_text("heavy", encoding="utf-8")
    archive = tmp_path / "jspace_results.tar.gz"

    pack.pack_results(root, archive)

    with tarfile.open(archive, "r:gz") as handle:
        assert {member.name for member in handle.getmembers()} == {
            "results/execution_manifest.json",
            "results/final_evaluation.json",
        }


@pytest.mark.parametrize(
    "missing",
    ["execution_manifest.json", "final_evaluation.json"],
)
def test_results_pack_fails_when_allowlisted_output_is_missing(
    tmp_path: Path,
    missing: str,
) -> None:
    pack = load_module(PACK_PATH, f"jspace_pack_missing_{missing}")
    root = tmp_path / "jspace_llama"
    results = root / "results"
    results.mkdir(parents=True)
    for name in {"execution_manifest.json", "final_evaluation.json"} - {missing}:
        (results / name).write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=missing):
        pack.pack_results(root, tmp_path / "results.tar.gz")


def test_downloaded_results_reject_unexpected_members(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "run_jspace_fetch_validation")
    archive = tmp_path / "results.tar.gz"
    source = tmp_path / "raw_generations.json"
    source.write_text("secret", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="results/raw_generations.json")

    with pytest.raises(ValueError, match="unexpected result archive member"):
        runner.extract_results_archive(archive, tmp_path / "project")


def test_documentation_lists_exact_jspace_workflow_and_gated_auth() -> None:
    colab_docs = (ROOT / "tools" / "colab" / "README.md").read_text(
        encoding="utf-8"
    )
    project_docs = (PROJECT / "README.md").read_text(encoding="utf-8")
    commands = [
        "python tools/colab/run_jspace.py probe",
        "python tools/colab/run_jspace.py start --gpu L4",
        "python tools/colab/run_jspace.py sync",
        "python tools/colab/run_jspace.py bootstrap",
        "python tools/colab/run_jspace.py auth-hf",
        (
            "python tools/colab/run_jspace.py execute "
            "--notebook projects/jspace_llama/notebooks/"
            "01_readout_and_causal_swap.ipynb --timeout 7200"
        ),
        "python tools/colab/run_jspace.py fetch",
        "python tools/colab/run_jspace.py stop",
    ]

    for documentation in (colab_docs, project_docs):
        for command in commands:
            assert command in documentation
        assert "meta-llama/Llama-3.1-8B-Instruct" in documentation
        assert "uv run hf auth login" in documentation
