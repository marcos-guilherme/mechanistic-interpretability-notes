from pathlib import Path
import importlib.util
import tarfile

import pytest


SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
SPEC = importlib.util.spec_from_file_location("run_refusal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_start_command_names_session_and_gpu() -> None:
    assert MODULE.start_command("refusal-gemma", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-gemma",
        "--gpu",
        "L4",
    ]


def test_stop_command_is_always_explicit() -> None:
    assert MODULE.stop_command("refusal-gemma") == [
        "colab",
        "--auth=adc",
        "stop",
        "-s",
        "refusal-gemma",
    ]


def test_exec_command_sets_remote_timeout() -> None:
    assert MODULE.exec_command(
        "refusal-gemma",
        Path("stage.py"),
        timeout=300,
    ) == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "refusal-gemma",
        "-f",
        "stage.py",
        "--timeout",
        "300",
    ]


def test_huggingface_token_uses_standard_cache_path(tmp_path) -> None:
    assert MODULE.huggingface_token_path(tmp_path) == (
        tmp_path / ".cache" / "huggingface" / "token"
    )


def _tiny_project(root) -> Path:
    project = root / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    return project


def test_gemma_archive_does_not_stamp_implementation_commit(tmp_path) -> None:
    project = _tiny_project(tmp_path)
    archive = MODULE.build_archive(
        stamp_commit=False,
        project=project,
        staging=tmp_path / "staging",
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/IMPLEMENTATION_COMMIT" not in names
    assert "refusal_direction/hello.py" in names


def test_archive_excludes_artifacts_even_when_source_direction_exists(
    tmp_path,
) -> None:
    project = _tiny_project(tmp_path)
    direction = project / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    direction.parent.mkdir(parents=True)
    direction.write_bytes(b"tensor")
    (project / "artifacts" / "other.pt").write_bytes(b"nope")
    archive = MODULE.build_archive(
        stamp_commit=False,
        project=project,
        staging=tmp_path / "staging",
    )
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "refusal_direction/hello.py" in names
    assert not any("artifacts" in name.replace("\\", "/") for name in names)
    assert not any(name.endswith("selected_direction.pt") for name in names)


def test_qwen_archive_stamps_exact_implementation_commit(tmp_path) -> None:
    project = _tiny_project(tmp_path)
    commit = "e" * 40
    archive = MODULE.build_archive(
        stamp_commit=True,
        project=project,
        staging=tmp_path / "staging",
        commit=commit,
    )
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.extractfile("refusal_direction/IMPLEMENTATION_COMMIT")
        assert member is not None
        assert member.read().decode("utf-8").strip() == commit


def test_lab_profile_session_and_bootstrap_are_dedicated() -> None:
    assert MODULE.PROFILE_SESSIONS["qwen25-coder-lab"] == "refusal-qwen-lab"
    assert MODULE.resolve_session("qwen25-coder-lab", None) == "refusal-qwen-lab"
    scripts = MODULE.bootstrap_scripts("qwen25-coder-lab")
    install = Path(__file__).parents[1] / "scripts" / "colab_install_lab.py"
    status = Path(__file__).parents[1] / "scripts" / "qwen_run_lab_status.py"
    assert install in scripts
    assert status == Path(
        MODULE.status_command("refusal-qwen-lab", profile="qwen25-coder-lab")[
            MODULE.status_command(
                "refusal-qwen-lab", profile="qwen25-coder-lab"
            ).index("-f")
            + 1
        ]
    )


def test_lab_start_forces_a100_without_changing_other_profiles(monkeypatch) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "qwen25-coder-lab", "--gpu", "T4"],
    )
    MODULE.main()
    assert commands[-1][-1] == "A100"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-qwen-lab"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "gemma", "--gpu", "L4"],
    )
    MODULE.main()
    assert commands[-1][-1] == "L4"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-gemma"


def test_lab_fetch_is_refused_and_does_not_download(monkeypatch) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "fetch", "--profile", "qwen25-coder-lab"],
    )
    with pytest.raises(SystemExit):
        MODULE.main()
    assert commands == []


def test_legacy_qwen_fetch_still_downloads(monkeypatch, tmp_path) -> None:
    import sys

    commands: list[list[str]] = []
    monkeypatch.setattr(MODULE, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "PROJECT", tmp_path / "refusal_direction")
    monkeypatch.setattr(MODULE, "extract_qwen_results_archive", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "fetch", "--profile", "qwen25-coder"],
    )
    MODULE.main()
    joined = " ".join(" ".join(item) for item in commands)
    assert "download" in joined
    assert "qwen_results.tar.gz" in joined
