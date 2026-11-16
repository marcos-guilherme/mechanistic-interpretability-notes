from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    CODE_EVAL_REMOTE_ARCHIVE,
    REMOTE_CODE_EVAL_DIRECTION_STAGING,
    REMOTE_CODE_EVAL_LENS_STAGING,
    atomic_install_code_eval_sources,
    collect_qwen_code_eval_pack_entries,
    extract_qwen_code_eval_archive,
    is_allowed_code_eval_fetch_member,
    pack_qwen_code_eval_results,
    require_local_code_eval_sources,
)


ROOT = Path(__file__).parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
RUN_SCRIPT = ROOT / "scripts" / "qwen_run_code_eval.py"
STATUS_SCRIPT = ROOT / "scripts" / "qwen_run_code_eval_status.py"
PACK_SCRIPT = ROOT / "scripts" / "colab_pack_qwen_code_eval_results.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_qwen_code_eval.py"
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_code_eval.py"


def _orchestrator():
    spec = importlib.util.spec_from_file_location("run_refusal_code_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _installer():
    spec = importlib.util.spec_from_file_location("install_code_eval", INSTALL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_code_eval_fetch_allowlist_separates_profile_and_allows_raw_artifacts() -> None:
    allowed = (
        "results/qwen25_coder_7b_code_eval/00_prepare.json",
        "results/qwen25_coder_7b_code_eval/code_eval_evaluation.json",
        "artifacts/qwen25_coder_7b_code_eval/status.json",
        "artifacts/qwen25_coder_7b_code_eval/events.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/datasets/mitre.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/logits/mitre__baseline.pt",
        "artifacts/qwen25_coder_7b_code_eval/logits/mitre__addition.pt",
        "artifacts/qwen25_coder_7b_code_eval/evalplus/results__baseline.json",
        "artifacts/qwen25_coder_7b_code_eval/evalplus/identity__baseline.json",
        "artifacts/qwen25_coder_7b_code_eval/readout/mitre__baseline.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/manual_review_blind.jsonl",
        "artifacts/qwen25_coder_7b_code_eval/manual_review_key.json",
    )
    assert all(is_allowed_code_eval_fetch_member(path) for path in allowed)
    assert not is_allowed_code_eval_fetch_member(
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    )
    assert not is_allowed_code_eval_fetch_member(
        "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    assert not is_allowed_code_eval_fetch_member(
        "artifacts/qwen25_coder_7b_code_eval/evalplus/temp__baseline/evil.py"
    )


def test_code_eval_source_pair_installs_atomically_and_is_confined(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        project
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    assert require_local_code_eval_sources(project) == (direction, lens)

    staged_direction = tmp_path / "direction.pt"
    staged_lens = tmp_path / "lens.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_lens.write_bytes(b"new-lens")
    installed = atomic_install_code_eval_sources(
        staged_direction=staged_direction,
        staged_lens=staged_lens,
        project=project,
    )
    assert installed == (direction, lens)
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"

    staged_direction.write_bytes(b"rollback-direction")
    staged_lens.unlink()
    with pytest.raises(FileNotFoundError):
        atomic_install_code_eval_sources(
            staged_direction=staged_direction,
            staged_lens=staged_lens,
            project=project,
        )
    assert direction.read_bytes() == b"new-direction"
    assert lens.read_bytes() == b"new-lens"


def test_pack_extract_keeps_raw_ignored_artifacts_inside_profile(tmp_path: Path) -> None:
    root = tmp_path / "root"
    results = root / "results/qwen25_coder_7b_code_eval"
    artifacts = root / "artifacts/qwen25_coder_7b_code_eval"
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    compact = {
        "schema_version": 1,
        "experiment": "code_eval_prepare",
        "config": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        "results": {"counts": {"mitre": 32}},
        "provenance": {
            "run_id": "run",
            "implementation_commit": "a" * 40,
        },
    }
    (results / "00_prepare.json").write_text(json.dumps(compact), encoding="utf-8")
    (artifacts / "status.json").write_text(
        json.dumps({"state": "running"}), encoding="utf-8"
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps({"event": "progress"}) + "\n", encoding="utf-8"
    )
    raw = artifacts / "generations/mitre__baseline.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps({"prompt": "raw only", "response": "raw only"}) + "\n",
        encoding="utf-8",
    )
    entries = collect_qwen_code_eval_pack_entries(root)
    names = {str(path.relative_to(root)).replace("\\", "/") for path in entries}
    assert "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl" in names
    archive = tmp_path / "code-eval.tar.gz"
    pack_qwen_code_eval_results(root, archive)
    destination = tmp_path / "destination"
    destination.mkdir()
    extract_qwen_code_eval_archive(archive, destination)
    assert (
        destination
        / "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl"
    ).is_file()


def _compact_identity(run_id: str) -> dict:
    return {
        "schema_version": 1,
        "experiment": "code_eval_prepare",
        "config": {},
        "results": {},
        "provenance": {
            "run_id": run_id,
            "implementation_commit": "a" * 40,
            "source_artifact_hashes": {"direction": "b" * 64},
        },
    }


def test_code_eval_fetch_rejects_identity_conflict_and_allows_incremental(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    existing = destination / "results/qwen25_coder_7b_code_eval/00_prepare.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_compact_identity("local-run")), encoding="utf-8")

    conflicting_root = tmp_path / "conflicting"
    conflict = (
        conflicting_root / "results/qwen25_coder_7b_code_eval/01_generate.json"
    )
    conflict.parent.mkdir(parents=True)
    conflict.write_text(json.dumps(_compact_identity("remote-run")), encoding="utf-8")
    archive = tmp_path / "conflict.tar.gz"
    pack_qwen_code_eval_results(conflicting_root, archive)
    before = existing.read_bytes()
    with pytest.raises(ValueError, match="identity conflict"):
        extract_qwen_code_eval_archive(archive, destination)
    assert existing.read_bytes() == before
    assert not (
        destination / "results/qwen25_coder_7b_code_eval/01_generate.json"
    ).exists()

    incremental_root = tmp_path / "incremental"
    incremental = (
        incremental_root / "results/qwen25_coder_7b_code_eval/01_generate.json"
    )
    incremental.parent.mkdir(parents=True)
    incremental.write_text(
        json.dumps(_compact_identity("local-run")), encoding="utf-8"
    )
    archive = tmp_path / "incremental.tar.gz"
    pack_qwen_code_eval_results(incremental_root, archive)
    extract_qwen_code_eval_archive(archive, destination)
    assert existing.read_bytes() == before
    assert (
        destination / "results/qwen25_coder_7b_code_eval/01_generate.json"
    ).is_file()


def test_code_eval_fetch_rejects_internally_conflicting_new_tree(
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "incoming"
    results = incoming / "results/qwen25_coder_7b_code_eval"
    results.mkdir(parents=True)
    (results / "00_prepare.json").write_text(
        json.dumps(_compact_identity("first-run")), encoding="utf-8"
    )
    (results / "01_generate.json").write_text(
        json.dumps(_compact_identity("second-run")), encoding="utf-8"
    )
    archive = tmp_path / "internal-conflict.tar.gz"
    pack_qwen_code_eval_results(incoming, archive)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="conflicting identities"):
        extract_qwen_code_eval_archive(archive, destination)
    assert not (destination / "results/qwen25_coder_7b_code_eval").exists()


def test_code_eval_fetch_rejects_raw_artifact_from_different_run(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    existing = destination / "results/qwen25_coder_7b_code_eval/00_prepare.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")

    incoming = tmp_path / "incoming"
    compact = incoming / "results/qwen25_coder_7b_code_eval/01_generate.json"
    compact.parent.mkdir(parents=True)
    compact.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")
    raw = (
        incoming
        / "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "run_id": "other-run",
                "implementation_commit": "a" * 40,
                "source_artifact_hashes": {"direction": "b" * 64},
                "example_id": "mitre:0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "mixed-raw.tar.gz"
    pack_qwen_code_eval_results(incoming, archive)
    with pytest.raises(ValueError, match="conflicting identities"):
        extract_qwen_code_eval_archive(archive, destination)
    assert not (
        destination / "results/qwen25_coder_7b_code_eval/01_generate.json"
    ).exists()


def test_code_eval_fetch_rolls_back_partial_same_identity_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import refusal_direction_replication.qwen_colab_transport as transport

    destination = tmp_path / "destination"
    result = destination / "results/qwen25_coder_7b_code_eval/00_prepare.json"
    raw = (
        destination
        / "artifacts/qwen25_coder_7b_code_eval/generations/mitre__baseline.jsonl"
    )
    result.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    result.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")
    raw_identity = {
        "run_id": "same-run",
        "implementation_commit": "a" * 40,
        "source_artifact_hashes": {"direction": "b" * 64},
    }
    raw.write_text(
        json.dumps({**raw_identity, "response": "old"}) + "\n",
        encoding="utf-8",
    )

    incoming = tmp_path / "incoming"
    incoming_result = incoming / result.relative_to(destination)
    incoming_raw = incoming / raw.relative_to(destination)
    incoming_result.parent.mkdir(parents=True)
    incoming_raw.parent.mkdir(parents=True)
    incoming_result.write_text(
        json.dumps(_compact_identity("same-run"), indent=2), encoding="utf-8"
    )
    incoming_raw.write_text(
        json.dumps({**raw_identity, "response": "new"}) + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "rollback.tar.gz"
    pack_qwen_code_eval_results(incoming, archive)

    real_replace = transport.os.replace

    def fail_raw_install(source, target):
        if Path(target) == raw and ".backups" not in str(source):
            raise OSError("simulated install failure")
        return real_replace(source, target)

    monkeypatch.setattr(transport.os, "replace", fail_raw_install)
    with pytest.raises(OSError, match="simulated"):
        extract_qwen_code_eval_archive(archive, destination)
    assert json.loads(result.read_text()) == _compact_identity("same-run")
    assert json.loads(raw.read_text(encoding="utf-8"))["response"] == "old"


def test_colab_profile_routes_sync_bootstrap_run_status_and_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    assert module.resolve_session("qwen25-coder-code-eval", None) == "refusal-qwen"
    assert module.fetch_pack_script("qwen25-coder-code-eval") == PACK_SCRIPT
    assert module.fetch_pack_timeout("qwen25-coder-code-eval") == 1800
    assert module.code_eval_command("refusal-qwen", timeout=14400)[-2:] == [
        "--timeout",
        "14400",
    ]
    assert module.status_command(
        "refusal-qwen", profile="qwen25-coder-code-eval"
    )[module.status_command(
        "refusal-qwen", profile="qwen25-coder-code-eval"
    ).index("-f") + 1] == str(STATUS_SCRIPT)
    assert INSTALL_SCRIPT in module.bootstrap_scripts("qwen25-coder-code-eval")

    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    direction = project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        project
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(module, "current_implementation_commit", lambda cwd=None: "a" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", "qwen25-coder-code-eval"],
    )
    module.main()
    uploaded = [item for command in commands for item in command]
    assert REMOTE_CODE_EVAL_DIRECTION_STAGING in uploaded
    assert any(
        str(item).startswith(REMOTE_CODE_EVAL_LENS_STAGING + ".part")
        for item in uploaded
    )


def test_code_eval_installer_reassembles_contiguous_lens_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _installer()
    destination = tmp_path / "lens.pt"
    monkeypatch.setattr(
        module,
        "REMOTE_CODE_EVAL_LENS_STAGING",
        str(destination),
    )
    (tmp_path / "lens.pt.part000").write_bytes(b"first")
    (tmp_path / "lens.pt.part001").write_bytes(b"second")
    assert module.assemble_staged_lens() == destination
    assert destination.read_bytes() == b"firstsecond"
    assert not list(tmp_path.glob("lens.pt.part*"))


def test_code_eval_scripts_exist_and_pin_dependencies() -> None:
    for path in (RUN_SCRIPT, STATUS_SCRIPT, PACK_SCRIPT, VALIDATE_SCRIPT, INSTALL_SCRIPT):
        assert path.is_file()
        assert "/content/refusal_direction/src" in path.read_text(encoding="utf-8")
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e" in install
    assert "581d398613e5602a5af361e1c34d3a92ea82ba8e" in install
    assert "--no-deps" in install
    for dependency in (
        "psutil",
        "termcolor",
        "tree_sitter",
        "tree-sitter-python",
    ):
        assert dependency in install
    assert CODE_EVAL_REMOTE_ARCHIVE == Path("/content/qwen_code_eval_results.tar.gz")


def test_evalplus_bootstrap_smoke_sanitizes_harmless_code_and_checks_api() -> None:
    module = _installer()
    observed: list[tuple[str, str]] = []

    def sanitize(code: str, *, entrypoint: str) -> str:
        observed.append((code, entrypoint))
        return code

    def check_correctness(
        dataset,
        completion_id,
        problem,
        solution,
        expected_output,
        **kwargs,
    ):
        return {}

    assert module.smoke_evalplus_contract(
        sanitize=sanitize,
        check_correctness=check_correctness,
        reliability_guard=lambda: None,
    ) == "def add(a, b):\n    return a + b\n"
    assert observed[0][1] == "add"

    with pytest.raises(RuntimeError, match="sanitize"):
        module.smoke_evalplus_contract(
            sanitize=lambda *_args, **_kwargs: "",
            check_correctness=check_correctness,
            reliability_guard=lambda: None,
        )


def test_code_eval_start_forces_a100_without_changing_other_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_refusal.py",
            "start",
            "--profile",
            "qwen25-coder-code-eval",
            "--gpu",
            "L4",
        ],
    )
    module.main()
    assert commands[-1][-1] == "A100"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", "qwen25-coder-dose", "--gpu", "L4"],
    )
    module.main()
    assert commands[-1][-1] == "L4"
