from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
PROJECT = ROOT / "projects" / "jspace_llama"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
BOOTSTRAP_PATH = PROJECT / "scripts" / "colab_bootstrap.py"
AUTH_PATH = PROJECT / "scripts" / "colab_prepare_hf_auth.py"
PACK_PATH = PROJECT / "scripts" / "colab_pack_results.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_tar_bytes(
    handle: tarfile.TarFile,
    name: str,
    payload: bytes,
    *,
    member_type: bytes | None = None,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    if member_type is not None:
        info.type = member_type
    handle.addfile(info, io.BytesIO(payload))


def write_result_files(root: Path) -> None:
    results = root / "results"
    results.mkdir(parents=True)
    (results / "execution_manifest.json").write_text("{}", encoding="utf-8")
    (results / "final_evaluation.json").write_text("[]", encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "module_name"),
    [
        (BOOTSTRAP_PATH, "review_bootstrap_marker"),
        (AUTH_PATH, "review_auth_marker"),
        (PACK_PATH, "review_pack_marker"),
    ],
)
def test_remote_scripts_define_unique_success_markers(
    path: Path,
    module_name: str,
) -> None:
    module = load_module(path, module_name)

    assert module.SUCCESS_MARKER.startswith("JSPACE_")
    assert module.SUCCESS_MARKER.endswith("_SUCCESS")


def test_remote_success_markers_are_unique() -> None:
    markers = {
        load_module(BOOTSTRAP_PATH, "review_bootstrap_unique").SUCCESS_MARKER,
        load_module(AUTH_PATH, "review_auth_unique").SUCCESS_MARKER,
        load_module(PACK_PATH, "review_pack_unique").SUCCESS_MARKER,
    }

    assert len(markers) == 3


def test_auth_marker_is_emitted_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auth = load_module(AUTH_PATH, "review_auth_main_marker")
    source = tmp_path / "hf_token"
    destination = tmp_path / "cache" / "token"
    source.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(auth, "SOURCE", source)
    monkeypatch.setattr(auth, "DESTINATION", destination)

    assert auth.main() == 0
    assert capsys.readouterr().out.strip() == auth.SUCCESS_MARKER


def test_pack_marker_is_emitted_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = load_module(PACK_PATH, "review_pack_main_marker")
    root = tmp_path / "jspace_llama"
    write_result_files(root)
    monkeypatch.setattr(pack, "ROOT", root)
    monkeypatch.setattr(pack, "ARCHIVE", tmp_path / "results.tar.gz")

    assert pack.main() == 0
    assert capsys.readouterr().out.strip() == pack.SUCCESS_MARKER


def test_bootstrap_marker_is_emitted_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_main_marker")
    project = tmp_path / "jspace_llama"
    monkeypatch.setattr(
        bootstrap,
        "replace_project_tree",
        lambda archive, destination, target: project,
        raising=False,
    )
    monkeypatch.setattr(
        bootstrap,
        "dependency_commands",
        lambda project, requirements: [],
    )
    monkeypatch.setattr(bootstrap, "print_runtime_diagnostics", lambda torch: None)
    monkeypatch.setitem(sys.modules, "torch", object())

    assert bootstrap.main() == 0
    assert capsys.readouterr().out.strip() == bootstrap.SUCCESS_MARKER


def test_local_runner_rejects_missing_remote_success_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "review_runner_marker")
    completed = subprocess.CompletedProcess(
        args=["colab"],
        returncode=0,
        stdout="remote traceback recorded in notebook output\n",
        stderr="",
    )
    monkeypatch.setattr(runner, "run_captured", lambda command: completed)

    with pytest.raises(RuntimeError, match="success marker"):
        runner.run_remote_script(["colab", "exec"], "JSPACE_EXPECTED_SUCCESS")


def test_local_runner_requires_marker_as_exact_output_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "review_runner_exact_marker")
    completed = subprocess.CompletedProcess(
        args=["colab"],
        returncode=0,
        stdout='print("JSPACE_EXPECTED_SUCCESS")\n',
        stderr="",
    )
    monkeypatch.setattr(runner, "run_captured", lambda command: completed)

    with pytest.raises(RuntimeError, match="success marker"):
        runner.run_remote_script(["colab", "exec"], "JSPACE_EXPECTED_SUCCESS")


def test_pack_removes_stale_archive_before_validating_results(
    tmp_path: Path,
) -> None:
    pack = load_module(PACK_PATH, "review_pack_stale")
    root = tmp_path / "jspace_llama"
    (root / "results").mkdir(parents=True)
    archive = tmp_path / "jspace_results.tar.gz"
    archive.write_bytes(b"stale")

    with pytest.raises(FileNotFoundError):
        pack.pack_results(root, archive)

    assert not archive.exists()


def test_notebook_execution_removes_stale_output_and_rejects_cell_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "review_notebook_errors")
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    output = tmp_path / "experiment_output.ipynb"
    output.write_text('{"stale": true}', encoding="utf-8")

    def fake_run(command: list[str]) -> None:
        assert not output.exists()
        output.write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "outputs": [
                                {
                                    "output_type": "error",
                                    "ename": "RuntimeError",
                                    "evalue": "remote failure",
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "run", fake_run)

    with pytest.raises(RuntimeError, match="RuntimeError.*remote failure"):
        runner.execute_notebook("jspace-llama", notebook, timeout=600)

    assert output.exists()


def test_notebook_execution_requires_fresh_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "review_notebook_output")
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "run", lambda command: None)

    with pytest.raises(FileNotFoundError, match="_output.ipynb"):
        runner.execute_notebook("jspace-llama", notebook, timeout=600)


def test_archive_has_only_explicit_regular_files(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_archive_regular")
    project = tmp_path / "jspace_llama"
    for relative, content in {
        "README.md": "readme",
        "pyproject.toml": "project",
        "uv.lock": "lock",
        "src/pkg/core.py": "source",
        "scripts/task.py": "script",
        "tests/test_task.py": "test",
        "notebooks/run.ipynb": "{}",
    }.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    archive = runner.build_archive(
        project=project,
        staging=tmp_path / "staging",
    )

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    assert members
    assert all(member.isfile() for member in members)


def test_archive_rejects_unknown_suffix_in_approved_tree(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_archive_suffix")
    project = tmp_path / "jspace_llama"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "bootstrap.sh").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="suffix"):
        runner.build_archive(project=project, staging=tmp_path / "staging")


def test_archive_rejects_path_resolving_outside_project(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_archive_outside")
    project = tmp_path / "jspace_llama"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("source", encoding="utf-8")
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_file_attributes=0,
        st_size=6,
    )

    with pytest.raises(ValueError, match="outside project"):
        runner.validate_archive_file(outside, project, metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFDIR, 0x400, "reparse"),
        (stat.S_IFREG, 0, "directory"),
    ],
)
def test_archive_rejects_unsafe_approved_directories(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"review_archive_directory_{mode}_{attributes}",
    )
    path = tmp_path / "jspace_llama" / "src"
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
    )

    with pytest.raises(ValueError, match=message):
        runner.validate_archive_directory(path, metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFREG, 0x400, "reparse"),
        (stat.S_IFIFO, 0, "regular"),
    ],
)
def test_archive_rejects_unsafe_source_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"review_archive_metadata_{mode}_{attributes}",
    )
    project = tmp_path / "jspace_llama"
    path = project / "src" / "pkg" / "core.py"
    path.parent.mkdir(parents=True)
    path.write_text("source", encoding="utf-8")
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
        st_size=6,
    )

    with pytest.raises(ValueError, match=message):
        runner.validate_archive_file(path, project, metadata=metadata)


def test_bootstrap_rejects_duplicate_archive_names(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_duplicate")
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(handle, "jspace_llama/src/pkg.py", b"first")
        add_tar_bytes(handle, "jspace_llama/src/pkg.py", b"second")

    with pytest.raises(ValueError, match="duplicate archive member"):
        bootstrap.safe_extract(archive, tmp_path / "destination")


def test_bootstrap_rejects_duplicate_normalized_destinations(
    tmp_path: Path,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_normalized")
    archive = tmp_path / "duplicate-normalized.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(handle, "jspace_llama/src/pkg.py", b"first")
        add_tar_bytes(handle, "jspace_llama/src//pkg.py", b"second")

    with pytest.raises(ValueError, match="duplicate archive member"):
        bootstrap.safe_extract(archive, tmp_path / "destination")


def test_bootstrap_rejects_special_archive_members(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_special")
    archive = tmp_path / "special.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(
            handle,
            "jspace_llama/src/pipe.py",
            b"",
            member_type=tarfile.FIFOTYPE,
        )

    with pytest.raises(ValueError, match="regular files"):
        bootstrap.safe_extract(archive, tmp_path / "destination")


def test_bootstrap_replaces_old_project_tree(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_fresh")
    destination = tmp_path / "content"
    project = destination / "jspace_llama"
    project.mkdir(parents=True)
    (project / "stale.txt").write_text("stale", encoding="utf-8")
    archive = tmp_path / "project.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(handle, "jspace_llama/src/fresh.py", b"fresh")

    bootstrap.replace_project_tree(archive, destination, project)

    assert not (project / "stale.txt").exists()
    assert (project / "src" / "fresh.py").read_bytes() == b"fresh"


def test_pack_rejects_invalid_json(tmp_path: Path) -> None:
    pack = load_module(PACK_PATH, "review_pack_json")
    root = tmp_path / "jspace_llama"
    write_result_files(root)
    (root / "results" / "final_evaluation.json").write_text(
        "not json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid JSON"):
        pack.pack_results(root, tmp_path / "results.tar.gz")


def test_pack_rejects_oversized_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_module(PACK_PATH, "review_pack_size")
    root = tmp_path / "jspace_llama"
    write_result_files(root)
    monkeypatch.setattr(pack, "MAX_RESULT_BYTES", 1)

    with pytest.raises(ValueError, match="too large"):
        pack.pack_results(root, tmp_path / "results.tar.gz")


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFREG, 0x400, "reparse"),
        (stat.S_IFIFO, 0, "regular"),
    ],
)
def test_pack_rejects_unsafe_result_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    pack = load_module(PACK_PATH, f"review_pack_metadata_{mode}_{attributes}")
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
        st_size=2,
    )

    with pytest.raises(ValueError, match=message):
        pack.validate_result_file(path, metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFDIR, 0x400, "reparse"),
        (stat.S_IFREG, 0, "directory"),
    ],
)
def test_pack_rejects_unsafe_results_directory_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    pack = load_module(PACK_PATH, f"review_pack_directory_{mode}_{attributes}")
    path = tmp_path / "results"
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
    )

    with pytest.raises(ValueError, match=message):
        pack.validate_result_directory(path, metadata=metadata)


def make_results_archive(
    archive: Path,
    manifest: bytes = b"{}",
    evaluation: bytes = b"[]",
) -> None:
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(
            handle,
            "results/execution_manifest.json",
            manifest,
        )
        add_tar_bytes(
            handle,
            "results/final_evaluation.json",
            evaluation,
        )


def test_result_extract_rejects_duplicate_names(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_extract_duplicate")
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(handle, "results/execution_manifest.json", b"{}")
        add_tar_bytes(handle, "results/execution_manifest.json", b"{}")
        add_tar_bytes(handle, "results/final_evaluation.json", b"[]")

    with pytest.raises(ValueError, match="duplicate result archive member"):
        runner.extract_results_archive(archive, tmp_path / "project")


def test_result_extract_rejects_invalid_json(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_extract_json")
    archive = tmp_path / "results.tar.gz"
    make_results_archive(archive, evaluation=b"not json")

    with pytest.raises(ValueError, match="valid JSON"):
        runner.extract_results_archive(archive, tmp_path / "project")


def test_result_extract_rejects_oversized_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "review_extract_size")
    archive = tmp_path / "results.tar.gz"
    make_results_archive(archive)
    monkeypatch.setattr(runner, "MAX_RESULT_BYTES", 1)

    with pytest.raises(ValueError, match="too large"):
        runner.extract_results_archive(archive, tmp_path / "project")


def test_result_extract_rejects_special_member(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "review_extract_special")
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(
            handle,
            "results/execution_manifest.json",
            b"",
            member_type=tarfile.SYMTYPE,
        )
        add_tar_bytes(handle, "results/final_evaluation.json", b"[]")

    with pytest.raises(ValueError, match="regular files"):
        runner.extract_results_archive(archive, tmp_path / "project")


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFREG, 0x400, "reparse"),
        (stat.S_IFIFO, 0, "regular"),
    ],
)
def test_result_extract_rejects_unsafe_archive_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"review_result_archive_{mode}_{attributes}",
    )
    archive = tmp_path / "results.tar.gz"
    archive.write_bytes(b"archive")
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
    )

    with pytest.raises(ValueError, match=message):
        runner.validate_result_archive(archive, metadata=metadata)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFREG, 0x400, "reparse"),
        (stat.S_IFIFO, 0, "regular"),
    ],
)
def test_result_extract_rejects_unsafe_destination_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"review_destination_metadata_{mode}_{attributes}",
    )
    root = tmp_path / "project"
    path = root / "results" / "execution_manifest.json"
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
        st_size=2,
    )

    with pytest.raises(ValueError, match=message):
        runner.validate_result_destination(path, root, metadata=metadata)


def test_bootstrap_pins_uv_version(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_uv")

    commands = bootstrap.dependency_commands(
        tmp_path / "jspace_llama",
        tmp_path / "requirements.txt",
    )

    assert commands[0][-1] == "uv==0.12.1"


def test_docs_pin_colab_cli_version() -> None:
    for path in (ROOT / "tools" / "colab" / "README.md", PROJECT / "README.md"):
        documentation = path.read_text(encoding="utf-8")
        assert "google-colab-cli==0.7.5" in documentation
        assert "uv==0.12.1" in documentation


@pytest.mark.parametrize(
    "argv",
    [
        ["probe", "--session", "unused"],
        ["start", "--notebook", "unused.ipynb"],
        ["sync", "--gpu", "T4"],
        ["bootstrap", "--timeout", "60"],
        ["auth-hf", "--gpu", "L4"],
        ["fetch", "--timeout", "300"],
        ["stop", "--gpu", "L4"],
    ],
)
def test_action_subparsers_reject_unrelated_options(argv: list[str]) -> None:
    runner = load_module(RUNNER_PATH, f"review_parser_{argv[0]}")

    with pytest.raises(SystemExit):
        runner.parse_args(argv)
