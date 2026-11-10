from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace

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


def write_valid_notebook(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def create_hard_link_or_skip(source: Path, link: Path) -> None:
    try:
        os.link(source, link)
    except OSError as error:
        pytest.skip(f"filesystem cannot create hard links: {error}")
    if os.stat(source).st_nlink == 1:
        pytest.skip("filesystem did not report the created hard link")


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
    nonce = "a" * 64
    monkeypatch.setenv(auth.SUCCESS_NONCE_ENV, nonce)

    assert auth.main() == 0
    assert capsys.readouterr().out.strip() == f"{auth.SUCCESS_MARKER}:{nonce}"


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
    nonce = "b" * 64
    monkeypatch.setenv(pack.SUCCESS_NONCE_ENV, nonce)

    assert pack.main() == 0
    assert capsys.readouterr().out.strip() == f"{pack.SUCCESS_MARKER}:{nonce}"


def test_bootstrap_marker_is_emitted_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "review_bootstrap_main_marker")
    project = tmp_path / "jspace_llama"
    migration_calls: list[Path] = []
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
    monkeypatch.setattr(
        bootstrap,
        "migrate_same_session",
        lambda legacy: migration_calls.append(legacy),
        raising=False,
    )
    monkeypatch.setattr(bootstrap, "print_runtime_diagnostics", lambda torch: None)
    monkeypatch.setitem(sys.modules, "torch", object())
    nonce = "c" * 64
    monkeypatch.setenv(bootstrap.SUCCESS_NONCE_ENV, nonce)

    assert bootstrap.main() == 0
    assert migration_calls == [bootstrap.LEGACY_PROJECT]
    assert (
        capsys.readouterr().out.strip()
        == f"{bootstrap.SUCCESS_MARKER}:{nonce}"
    )


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
    write_valid_notebook(notebook)
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
    write_valid_notebook(notebook)
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


def test_remote_checkout_basename_does_not_shadow_import_package(
    tmp_path: Path,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "live_namespace_bootstrap")
    pack = load_module(PACK_PATH, "live_namespace_pack")
    package_name = "jspace_llama"
    checkout_basename = bootstrap.PROJECT.name
    content = tmp_path / "content"
    source_package = content / checkout_basename / "src" / package_name
    source_package.mkdir(parents=True)
    (source_package / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util; "
                f"print(importlib.util.find_spec({package_name!r}))"
            ),
        ],
        cwd=content,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "None"
    assert checkout_basename != package_name
    assert pack.ROOT.name == checkout_basename


def test_bootstrap_migrates_legacy_checkout_and_cached_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "same_session_migration")
    package_name = "jspace_llama"
    content = tmp_path / "content"
    legacy = content / package_name
    legacy.mkdir(parents=True)
    (legacy / "stale.py").write_text("STALE = True\n", encoding="utf-8")
    archive = content / "jspace_llama.tar.gz"
    archive.write_bytes(b"archive remains")

    source_root = content / "jspace_llama_project" / "src"
    source_package = source_root / package_name
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    config = source_package / "config.py"
    config.write_text("SOURCE = 'new-project'\n", encoding="utf-8")

    cached_package = ModuleType(package_name)
    cached_package.__path__ = [str(legacy)]  # type: ignore[attr-defined]
    cached_config = ModuleType(f"{package_name}.config")
    unrelated_name = f"{package_name}_tools"
    unrelated = ModuleType(unrelated_name)
    prior_package_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(f"{package_name}.")
    }
    prior_unrelated = sys.modules.get(unrelated_name)
    invalidations = 0
    real_invalidate_caches = importlib.invalidate_caches

    def track_invalidation() -> None:
        nonlocal invalidations
        invalidations += 1
        real_invalidate_caches()

    monkeypatch.setattr(
        importlib,
        "invalidate_caches",
        track_invalidation,
    )
    sys.path.insert(0, str(source_root))
    try:
        sys.modules[package_name] = cached_package
        sys.modules[f"{package_name}.config"] = cached_config
        sys.modules[unrelated_name] = unrelated

        bootstrap.migrate_same_session(legacy)

        assert not os.path.lexists(legacy)
        assert archive.read_bytes() == b"archive remains"
        assert package_name not in sys.modules
        assert f"{package_name}.config" not in sys.modules
        assert sys.modules[unrelated_name] is unrelated
        assert invalidations == 1

        resolved = importlib.import_module(f"{package_name}.config")
        assert resolved.SOURCE == "new-project"
        assert Path(resolved.__file__).resolve() == config.resolve()
    finally:
        sys.path.remove(str(source_root))
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(prior_package_modules)
        if prior_unrelated is None:
            sys.modules.pop(unrelated_name, None)
        else:
            sys.modules[unrelated_name] = prior_unrelated
        real_invalidate_caches()


def test_legacy_reparse_directory_uses_non_recursive_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "legacy_reparse_removal")
    legacy = tmp_path / "jspace_llama"
    legacy.mkdir()
    actual = os.lstat(legacy)
    metadata = SimpleNamespace(
        st_mode=actual.st_mode,
        st_file_attributes=0x400,
    )
    removed: list[Path] = []
    monkeypatch.setattr(bootstrap.os, "lstat", lambda path: metadata)
    monkeypatch.setattr(
        bootstrap.os,
        "rmdir",
        lambda path: removed.append(Path(path)),
    )
    monkeypatch.setattr(
        bootstrap.shutil,
        "rmtree",
        lambda path: pytest.fail(f"recursive removal used for {path}"),
    )

    bootstrap._remove_old_project(legacy)

    assert removed == [legacy]


def test_legacy_symlink_uses_non_recursive_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "legacy_symlink_removal")
    legacy = tmp_path / "jspace_llama"
    legacy.mkdir()
    metadata = SimpleNamespace(
        st_mode=stat.S_IFLNK,
        st_file_attributes=0,
    )
    unlinked: list[Path] = []
    monkeypatch.setattr(bootstrap.os, "lstat", lambda path: metadata)
    monkeypatch.setattr(
        bootstrap.Path,
        "unlink",
        lambda path: unlinked.append(path),
    )
    monkeypatch.setattr(
        bootstrap.shutil,
        "rmtree",
        lambda path: pytest.fail(f"recursive removal used for {path}"),
    )

    bootstrap._remove_old_project(legacy)

    assert unlinked == [legacy]


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
        assert "/content/jspace_llama_project" in documentation


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


def test_archive_rejects_hard_linked_source_file(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "remaining_source_hardlink")
    project = tmp_path / "jspace_llama"
    source = project / "src" / "pkg" / "core.py"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    create_hard_link_or_skip(source, tmp_path / "source-link.py")

    with pytest.raises(ValueError, match="hard-linked"):
        runner.validate_archive_file(source, project)


def test_build_rejects_hard_linked_staging_archive(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "remaining_staging_hardlink")
    project = tmp_path / "jspace_llama"
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        (project / name).write_text(name, encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    archive = staging / "jspace_llama.tar.gz"
    create_hard_link_or_skip(victim, archive)

    with pytest.raises(ValueError, match="hard-linked"):
        runner.build_archive(project=project, staging=staging)

    assert victim.read_text(encoding="utf-8") == "do not overwrite"


def test_bootstrap_rejects_hard_linked_project_archive(tmp_path: Path) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "remaining_bootstrap_hardlink")
    archive = tmp_path / "project.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        add_tar_bytes(handle, "jspace_llama/src/core.py", b"source")
    linked_archive = tmp_path / "linked-project.tar.gz"
    create_hard_link_or_skip(archive, linked_archive)

    with pytest.raises(ValueError, match="hard-linked"):
        bootstrap.safe_extract(linked_archive, tmp_path / "destination")


def test_pack_rejects_hard_linked_result_file(tmp_path: Path) -> None:
    pack = load_module(PACK_PATH, "remaining_result_hardlink")
    root = tmp_path / "jspace_llama"
    write_result_files(root)
    source = root / "results" / "execution_manifest.json"
    create_hard_link_or_skip(source, tmp_path / "manifest-link.json")

    with pytest.raises(ValueError, match="hard-linked"):
        pack.pack_results(root, tmp_path / "results.tar.gz")


def test_pack_rejects_hard_linked_stale_archive(tmp_path: Path) -> None:
    pack = load_module(PACK_PATH, "remaining_pack_archive_hardlink")
    root = tmp_path / "jspace_llama"
    write_result_files(root)
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    archive = tmp_path / "results.tar.gz"
    create_hard_link_or_skip(victim, archive)

    with pytest.raises(ValueError, match="hard-linked"):
        pack.pack_results(root, archive)

    assert victim.read_text(encoding="utf-8") == "do not overwrite"


def test_extract_rejects_hard_linked_results_archive(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "remaining_archive_hardlink")
    archive = tmp_path / "results.tar.gz"
    make_results_archive(archive)
    linked_archive = tmp_path / "linked-results.tar.gz"
    create_hard_link_or_skip(archive, linked_archive)

    with pytest.raises(ValueError, match="hard-linked"):
        runner.extract_results_archive(linked_archive, tmp_path / "project")


def test_source_safe_open_rejects_changed_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_source_identity")
    path = tmp_path / "source.py"
    path.write_text("source", encoding="utf-8")
    metadata = os.lstat(path)
    real_fstat = runner.os.fstat

    def changed_fstat(descriptor: int):
        actual = real_fstat(descriptor)
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(runner.os, "fstat", changed_fstat)

    with pytest.raises(ValueError, match="changed after validation"):
        with runner.open_validated_file(path, metadata):
            pass


def test_result_safe_open_rejects_changed_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_module(PACK_PATH, "remaining_result_identity")
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    metadata = os.lstat(path)
    real_fstat = pack.os.fstat

    def changed_fstat(descriptor: int):
        actual = real_fstat(descriptor)
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(pack.os, "fstat", changed_fstat)

    with pytest.raises(ValueError, match="changed after validation"):
        with pack.open_validated_file(path, metadata):
            pass


def test_exec_command_passes_nonce_environment() -> None:
    runner = load_module(RUNNER_PATH, "remaining_nonce_command")

    command = runner.exec_command(
        "jspace-llama",
        Path("stage.py"),
        timeout=300,
        environment={"JSPACE_SUCCESS_NONCE": "abc123"},
    )

    assert command[-2:] == ["--env", "JSPACE_SUCCESS_NONCE=abc123"]


def test_remote_operation_uses_fresh_nonce_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_fresh_nonce")
    nonces = iter(["a" * 64, "b" * 64])
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(runner.secrets, "token_hex", lambda size: next(nonces))
    monkeypatch.setattr(
        runner,
        "run_remote_script",
        lambda command, expected: calls.append((command, expected)),
    )

    for _ in range(2):
        runner.run_remote_operation(
            "jspace-llama",
            Path("stage.py"),
            timeout=300,
            operation_marker="JSPACE_TEST_SUCCESS",
        )

    assert calls[0][1] == f"JSPACE_TEST_SUCCESS:{'a' * 64}"
    assert calls[1][1] == f"JSPACE_TEST_SUCCESS:{'b' * 64}"
    assert calls[0][1] != calls[1][1]
    assert calls[0][0][-1] == f"JSPACE_SUCCESS_NONCE={'a' * 64}"
    assert calls[1][0][-1] == f"JSPACE_SUCCESS_NONCE={'b' * 64}"


def test_static_or_wrong_nonce_marker_cannot_spoof_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_nonce_spoof")
    completed = subprocess.CompletedProcess(
        args=["colab"],
        returncode=0,
        stdout=(
            "JSPACE_TEST_SUCCESS\n"
            f"JSPACE_TEST_SUCCESS:{'b' * 64}\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(runner, "run_captured", lambda command: completed)

    with pytest.raises(RuntimeError, match="success marker"):
        runner.run_remote_script(
            ["colab", "exec"],
            f"JSPACE_TEST_SUCCESS:{'a' * 64}",
        )


def test_bootstrap_prefixes_child_output_to_prevent_marker_spoof(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = load_module(BOOTSTRAP_PATH, "remaining_child_spoof")
    spoof = f"JSPACE_BOOTSTRAP_SUCCESS:{'a' * 64}"
    completed = subprocess.CompletedProcess(
        args=["child"],
        returncode=0,
        stdout=f"{spoof}\n",
        stderr=f"{spoof}\n",
    )
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: completed)

    bootstrap.run_checked(["child"])

    captured = capsys.readouterr()
    assert spoof not in captured.out.splitlines()
    assert spoof not in captured.err.splitlines()


@pytest.mark.parametrize(
    ("path", "module_name"),
    [
        (BOOTSTRAP_PATH, "remaining_bootstrap_nonce"),
        (AUTH_PATH, "remaining_auth_nonce"),
        (PACK_PATH, "remaining_pack_nonce"),
    ],
)
def test_remote_scripts_require_and_consume_nonce(
    path: Path,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(path, module_name)
    monkeypatch.delenv("JSPACE_SUCCESS_NONCE", raising=False)
    with pytest.raises(RuntimeError, match="nonce"):
        module.consume_success_nonce()

    nonce = "a" * 64
    monkeypatch.setenv("JSPACE_SUCCESS_NONCE", nonce)
    assert module.consume_success_nonce() == nonce
    assert "JSPACE_SUCCESS_NONCE" not in os.environ


def test_execute_rejects_non_notebook_before_remote_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_notebook_extension")
    path = tmp_path / "experiment.py"
    path.write_text("{}", encoding="utf-8")
    remote_called = False

    def fail_if_called(command: list[str]) -> None:
        nonlocal remote_called
        remote_called = True

    monkeypatch.setattr(runner, "run", fail_if_called)

    with pytest.raises(ValueError, match=r"\.ipynb"):
        runner.execute_notebook("jspace-llama", path, timeout=600)

    assert not remote_called


def test_execute_rejects_malformed_notebook_before_remote_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_notebook_malformed")
    path = tmp_path / "experiment.ipynb"
    path.write_text("{}", encoding="utf-8")
    remote_called = False

    def fail_if_called(command: list[str]) -> None:
        nonlocal remote_called
        remote_called = True

    monkeypatch.setattr(runner, "run", fail_if_called)

    with pytest.raises(ValueError, match="notebook"):
        runner.execute_notebook("jspace-llama", path, timeout=600)

    assert not remote_called


def test_execute_rejects_hard_linked_notebook_before_remote_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "remaining_notebook_hardlink")
    path = tmp_path / "experiment.ipynb"
    write_valid_notebook(path)
    create_hard_link_or_skip(path, tmp_path / "notebook-link.ipynb")
    remote_called = False

    def fail_if_called(command: list[str]) -> None:
        nonlocal remote_called
        remote_called = True

    monkeypatch.setattr(runner, "run", fail_if_called)

    with pytest.raises(ValueError, match="hard-linked"):
        runner.execute_notebook("jspace-llama", path, timeout=600)

    assert not remote_called


def test_runner_help_works_without_site_packages() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", str(RUNNER_PATH), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFDIR, 0x400, "reparse"),
        (stat.S_IFREG, 0, "directory"),
        (stat.S_IFIFO, 0, "directory"),
    ],
)
def test_staging_rejects_unsafe_metadata_without_overwrite(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    runner = load_module(
        RUNNER_PATH,
        f"fetch_staging_metadata_{mode}_{attributes}",
    )
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    staging.mkdir()
    sentinel = staging / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
        st_dev=staging.stat().st_dev,
        st_ino=staging.stat().st_ino,
    )

    with pytest.raises(ValueError, match=message):
        runner.validate_staging_directory(
            staging,
            root,
            metadata=metadata,
        )

    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_staging_must_resolve_inside_repository(tmp_path: Path) -> None:
    runner = load_module(RUNNER_PATH, "fetch_staging_outside")
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside repository"):
        runner.validate_staging_directory(outside, root)


def run_fake_fetch(
    runner,
    staging: Path,
    project: Path,
    observed_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str]) -> None:
        local = Path(command[-1])
        assert not os.path.lexists(local)
        assert local.parent != staging
        observed_paths.append(local)
        local.write_bytes(b"downloaded")

    def fake_extract(archive: Path, destination: Path) -> None:
        assert archive == observed_paths[-1]
        assert archive.read_bytes() == b"downloaded"
        assert destination == project

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "extract_results_archive", fake_extract)
    runner.fetch_results(
        "jspace-llama",
        staging=staging,
        project=project,
        root=staging.parent,
    )


def test_fetch_uses_fresh_randomized_private_paths_and_cleans_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_random_paths")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    predictable = staging / "jspace_results.tar.gz"
    staging.mkdir()
    predictable.write_text("pre-existing", encoding="utf-8")
    observed_paths: list[Path] = []

    run_fake_fetch(runner, staging, project, observed_paths, monkeypatch)
    run_fake_fetch(runner, staging, project, observed_paths, monkeypatch)

    assert len(set(observed_paths)) == 2
    assert all(not path.parent.exists() for path in observed_paths)
    assert predictable.read_text(encoding="utf-8") == "pre-existing"


def test_fetch_does_not_overwrite_predictable_hard_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_hard_link")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    staging.mkdir()
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    create_hard_link_or_skip(
        victim,
        staging / "jspace_results.tar.gz",
    )

    run_fake_fetch(runner, staging, project, [], monkeypatch)

    assert victim.read_text(encoding="utf-8") == "untouched"


def test_fetch_does_not_overwrite_predictable_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_symlink")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    staging.mkdir()
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    link = staging / "jspace_results.tar.gz"
    try:
        link.symlink_to(victim)
    except OSError:
        metadata = SimpleNamespace(
            st_mode=stat.S_IFLNK,
            st_file_attributes=0,
            st_dev=staging.stat().st_dev,
            st_ino=staging.stat().st_ino,
        )
        with pytest.raises(ValueError, match="symlink"):
            runner.validate_staging_directory(
                staging,
                root,
                metadata=metadata,
            )
        assert victim.read_text(encoding="utf-8") == "untouched"
        return

    run_fake_fetch(runner, staging, project, [], monkeypatch)

    assert link.is_symlink()
    assert victim.read_text(encoding="utf-8") == "untouched"


def test_fetch_does_not_overwrite_predictable_special_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_special_entry")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    staging.mkdir()
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    special = staging / "jspace_results.tar.gz"
    special.mkdir()
    sentinel = special / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    run_fake_fetch(runner, staging, project, [], monkeypatch)

    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_fetch_cleans_private_download_after_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_cleanup_failure")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    downloaded: list[Path] = []

    def fake_run(command: list[str]) -> None:
        local = Path(command[-1])
        downloaded.append(local)
        local.write_bytes(b"invalid")

    def fail_extract(archive: Path, destination: Path) -> None:
        raise ValueError("invalid downloaded archive")

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "extract_results_archive", fail_extract)

    with pytest.raises(ValueError, match="invalid downloaded archive"):
        runner.fetch_results(
            "jspace-llama",
            staging=staging,
            project=project,
            root=root,
        )

    assert downloaded
    assert not downloaded[0].parent.exists()


def test_fetch_rejects_staging_replacement_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_module(RUNNER_PATH, "fetch_staging_replaced")
    root = tmp_path / "repository"
    root.mkdir()
    staging = root / ".colab-staging"
    staging.mkdir()
    (staging / "sentinel.txt").write_text("untouched", encoding="utf-8")
    project = root / "projects" / "jspace_llama"
    project.mkdir(parents=True)
    original = root / "original-staging"
    download_called = False

    def replace_staging(*, prefix: str, dir: Path) -> str:
        staging.rename(original)
        staging.mkdir()
        private = staging / f"{prefix}attacker"
        private.mkdir()
        return str(private)

    def fake_run(command: list[str]) -> None:
        nonlocal download_called
        download_called = True

    monkeypatch.setattr(runner.tempfile, "mkdtemp", replace_staging)
    monkeypatch.setattr(runner, "run", fake_run)

    with pytest.raises(RuntimeError, match="staging directory changed"):
        runner.fetch_results(
            "jspace-llama",
            staging=staging,
            project=project,
            root=root,
        )

    assert not download_called
    assert (original / "sentinel.txt").read_text(encoding="utf-8") == "untouched"


def test_run_captured_decodes_utf8_ansi_and_preserves_marker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = load_module(RUNNER_PATH, "live_utf8_capture")
    marker = f"JSPACE_BOOTSTRAP_SUCCESS:{'a' * 64}"
    stdout_prefix = b"\x1b[32mremote: "
    stdout_suffix = b"\x1b[0m\n" + marker.encode("ascii") + b"\n"
    stderr_prefix = b"\x1b[33mwarning: "
    stderr_suffix = b"\x1b[0m\n"
    child_code = (
        "import os\n"
        f"os.write(1, {stdout_prefix!r} + b'\\xc2\\x81' + {stdout_suffix!r})\n"
        f"os.write(2, {stderr_prefix!r} + b'\\x81' + {stderr_suffix!r})\n"
    )

    completed = runner.run_captured([sys.executable, "-c", child_code])

    assert isinstance(completed.stdout, str)
    assert isinstance(completed.stderr, str)
    assert marker in completed.stdout.splitlines()
    assert "\x81" in completed.stdout
    assert "\ufffd" in completed.stderr
    captured = capsys.readouterr()
    assert marker in captured.out
    assert "warning:" in captured.err
