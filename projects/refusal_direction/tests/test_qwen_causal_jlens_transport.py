from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    CAUSAL_JLENS_ARTIFACTS_DIR,
    CAUSAL_JLENS_FETCH_ALLOWLIST,
    CAUSAL_JLENS_REMOTE_ARCHIVE,
    CAUSAL_JLENS_RESULTS_DIR,
    CODE_EVAL_ARTIFACTS_DIR,
    CODE_EVAL_REMOTE_ARCHIVE,
    CODE_EVAL_RESULTS_DIR,
    JLENS_V2_ARTIFACTS_DIR,
    JLENS_V2_REMOTE_ARCHIVE,
    JLENS_V2_RESULTS_DIR,
    REMOTE_CAUSAL_JLENS_DIRECTION_STAGING,
    REMOTE_CAUSAL_JLENS_V1_LENS_STAGING,
    REMOTE_CAUSAL_JLENS_V2_LENS_STAGING,
    atomic_install_causal_jlens_sources,
    extract_qwen_causal_jlens_archive,
    is_allowed_causal_jlens_fetch_member,
    pack_qwen_causal_jlens_results,
    require_local_causal_jlens_sources,
    validate_qwen_causal_jlens_archive,
)


ROOT = Path(__file__).parents[1]
SCRIPT = Path(__file__).parents[3] / "tools" / "colab" / "run_refusal.py"
RUN_SCRIPT = ROOT / "scripts" / "qwen_run_causal_jlens.py"
STATUS_SCRIPT = ROOT / "scripts" / "qwen_run_causal_jlens_status.py"
PACK_SCRIPT = ROOT / "scripts" / "colab_pack_qwen_causal_jlens_results.py"
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_causal_jlens.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_qwen_causal_jlens.py"

CAUSAL_PROFILE = "qwen25-coder-causal-jlens"
DIRECTION_NAME = "selected_direction.pt"
V1_LENS_NAME = "qwen25_coder_7b_jacobian_lens.pt"
V2_LENS_NAME = "qwen25_coder_7b_jacobian_lens_v2.pt"
SOURCE_DIRECTION = f"artifacts/qwen25_coder_7b/{DIRECTION_NAME}"
SOURCE_V1_LENS = f"artifacts/qwen25_coder_7b_jlens/{V1_LENS_NAME}"
SOURCE_V2_LENS = f"artifacts/qwen25_coder_7b_jlens_v2/{V2_LENS_NAME}"
COMPACT_NAMES = (
    "00_prepare.json",
    "01_pilot.json",
    "02_panel.json",
    "03_analysis.json",
    "causal_jlens_evaluation.json",
    "execution_manifest.json",
)
ARTIFACT_NAMES = (
    "status.json",
    "events.jsonl",
    "checkpoint.json",
    "responses.jsonl",
    "observations.jsonl",
)


def _orchestrator():
    spec = importlib.util.spec_from_file_location("run_refusal_causal_jlens", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _installer():
    spec = importlib.util.spec_from_file_location(
        "install_causal_jlens", INSTALL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _seed_causal_sources(project: Path) -> tuple[Path, Path, Path]:
    direction = _write_bytes(project / SOURCE_DIRECTION, b"direction-bytes")
    v1 = _write_bytes(project / SOURCE_V1_LENS, b"v1-lens-bytes")
    v2 = _write_bytes(project / SOURCE_V2_LENS, b"v2-lens-bytes")
    return direction, v1, v2


def _compact_identity(run_id: str) -> dict:
    return {
        "schema_version": 1,
        "experiment": "causal_jlens_analysis",
        "config": {},
        "results": {},
        "provenance": {
            "run_id": run_id,
            "implementation_commit": "a" * 40,
            "source_artifact_hashes": {
                "direction": "b" * 64,
                "v1_lens": "c" * 64,
                "v2_lens": "d" * 64,
            },
        },
    }


def _raw_identity(run_id: str, **extra: object) -> dict:
    payload = {
        "run_id": run_id,
        "implementation_commit": "a" * 40,
        "source_artifact_hashes": {
            "direction": "b" * 64,
            "v1_lens": "c" * 64,
            "v2_lens": "d" * 64,
        },
    }
    payload.update(extra)
    return payload


def _seed_causal_tree(root: Path, *, run_id: str = "causal-run") -> None:
    results = root / CAUSAL_JLENS_RESULTS_DIR
    artifacts = root / CAUSAL_JLENS_ARTIFACTS_DIR
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    for name in COMPACT_NAMES:
        (results / name).write_text(
            json.dumps(_compact_identity(run_id)) + "\n",
            encoding="utf-8",
        )
    (artifacts / "status.json").write_text(
        json.dumps({"state": "running", **_raw_identity(run_id)}) + "\n",
        encoding="utf-8",
    )
    (artifacts / "events.jsonl").write_text(
        json.dumps({"event": "progress", **_raw_identity(run_id)}) + "\n",
        encoding="utf-8",
    )
    (artifacts / "checkpoint.json").write_text(
        json.dumps(_raw_identity(run_id, completed=[])) + "\n",
        encoding="utf-8",
    )
    (artifacts / "responses.jsonl").write_text(
        json.dumps(
            _raw_identity(run_id, prompt="raw only", response="raw only")
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts / "observations.jsonl").write_text(
        json.dumps(_raw_identity(run_id, rank=1, layer=25)) + "\n",
        encoding="utf-8",
    )
    (artifacts / "datasets" / "mitre.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (artifacts / "datasets" / "mitre.jsonl").write_text(
        json.dumps({"prompt": "must not travel"}) + "\n",
        encoding="utf-8",
    )
    _write_bytes(root / SOURCE_DIRECTION, b"source-direction")
    _write_bytes(root / SOURCE_V1_LENS, b"source-v1")
    _write_bytes(root / SOURCE_V2_LENS, b"source-v2")
    (root / CODE_EVAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (root / CODE_EVAL_RESULTS_DIR / "00_prepare.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (root / JLENS_V2_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (root / JLENS_V2_RESULTS_DIR / "00_jlens_prepare.json").write_text(
        "{}\n", encoding="utf-8"
    )


def _tar_info(name: str, size: int = 8) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    return info


def _stage_chunked(
    staging_dir: Path,
    staged: Path,
    payload: bytes,
    *,
    chunk_bytes: int = 5,
) -> tuple[Path, Path]:
    source = staging_dir / f"{staged.name}.source"
    staging_dir.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    from refusal_direction_replication import qwen_colab_transport as transport

    parts = transport.split_artifact_file(
        source,
        staging_dir,
        chunk_bytes=chunk_bytes,
        part_prefix=staged.name + ".part",
    )
    manifest = staged.with_name(staged.name + ".manifest.json")
    transport.write_artifact_chunk_manifest(source, parts, manifest)
    source.unlink()
    return staged, manifest


def test_causal_profile_requires_three_frozen_sources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _seed_causal_sources(project)
    direction, v1, v2 = require_local_causal_jlens_sources(project)
    assert direction.name == DIRECTION_NAME
    assert v1.name == V1_LENS_NAME
    assert v2.name == V2_LENS_NAME
    assert direction == project / SOURCE_DIRECTION
    assert v1 == project / SOURCE_V1_LENS
    assert v2 == project / SOURCE_V2_LENS


def test_causal_archive_is_confined_and_preserves_raw_audit(tmp_path: Path) -> None:
    _seed_causal_tree(tmp_path)
    archive = tmp_path / "causal.tar.gz"
    pack_qwen_causal_jlens_results(tmp_path, archive)
    extracted = tmp_path / "extracted"
    extract_qwen_causal_jlens_archive(archive, extracted)
    assert (
        extracted / "results/qwen25_coder_7b_causal_jlens/03_analysis.json"
    ).is_file()
    assert (
        extracted / "artifacts/qwen25_coder_7b_causal_jlens/responses.jsonl"
    ).is_file()
    assert (
        extracted / "artifacts/qwen25_coder_7b_causal_jlens/checkpoint.json"
    ).is_file()
    assert (
        extracted / "artifacts/qwen25_coder_7b_causal_jlens/status.json"
    ).is_file()
    assert not (extracted / SOURCE_DIRECTION).exists()
    assert not (extracted / SOURCE_V1_LENS).exists()
    assert not (extracted / SOURCE_V2_LENS).exists()
    assert not (
        extracted / "artifacts/qwen25_coder_7b_causal_jlens/datasets/mitre.jsonl"
    ).exists()
    assert not (extracted / CODE_EVAL_RESULTS_DIR).exists()
    assert not (extracted / JLENS_V2_RESULTS_DIR).exists()


def test_orchestrator_exposes_forced_a100_causal_action() -> None:
    module = _orchestrator()
    assert module.PROFILE_SESSIONS["qwen25-coder-causal-jlens"]
    assert "qwen25-coder-causal-jlens" in module.A100_PROFILES
    command = module.causal_jlens_command("session", timeout=7200)
    assert command[-3].endswith("qwen_run_causal_jlens.py")


def test_causal_namespace_is_disjoint_from_code_eval_and_v2() -> None:
    assert CAUSAL_JLENS_RESULTS_DIR == "results/qwen25_coder_7b_causal_jlens"
    assert CAUSAL_JLENS_ARTIFACTS_DIR == "artifacts/qwen25_coder_7b_causal_jlens"
    assert CAUSAL_JLENS_REMOTE_ARCHIVE != CODE_EVAL_REMOTE_ARCHIVE
    assert CAUSAL_JLENS_REMOTE_ARCHIVE != JLENS_V2_REMOTE_ARCHIVE
    assert CAUSAL_JLENS_FETCH_ALLOWLIST.isdisjoint(
        {
            SOURCE_DIRECTION,
            SOURCE_V1_LENS,
            SOURCE_V2_LENS,
            f"{CODE_EVAL_RESULTS_DIR}/00_prepare.json",
            f"{CODE_EVAL_ARTIFACTS_DIR}/status.json",
            f"{JLENS_V2_RESULTS_DIR}/00_jlens_prepare.json",
            f"{JLENS_V2_ARTIFACTS_DIR}/status.json",
        }
    )
    assert all(
        name.startswith((f"{CAUSAL_JLENS_RESULTS_DIR}/", f"{CAUSAL_JLENS_ARTIFACTS_DIR}/"))
        for name in CAUSAL_JLENS_FETCH_ALLOWLIST
    )
    for name in COMPACT_NAMES:
        assert f"{CAUSAL_JLENS_RESULTS_DIR}/{name}" in CAUSAL_JLENS_FETCH_ALLOWLIST
    for name in ARTIFACT_NAMES:
        assert f"{CAUSAL_JLENS_ARTIFACTS_DIR}/{name}" in CAUSAL_JLENS_FETCH_ALLOWLIST


def test_causal_fetch_allowlist_rejects_sources_and_foreign_trees() -> None:
    allowed = [
        f"{CAUSAL_JLENS_RESULTS_DIR}/{name}" for name in COMPACT_NAMES
    ] + [f"{CAUSAL_JLENS_ARTIFACTS_DIR}/{name}" for name in ARTIFACT_NAMES]
    assert all(is_allowed_causal_jlens_fetch_member(path) for path in allowed)
    rejected = (
        SOURCE_DIRECTION,
        SOURCE_V1_LENS,
        SOURCE_V2_LENS,
        f"{CAUSAL_JLENS_ARTIFACTS_DIR}/datasets/mitre.jsonl",
        f"{CODE_EVAL_RESULTS_DIR}/00_prepare.json",
        f"{JLENS_V2_ARTIFACTS_DIR}/{V2_LENS_NAME}",
        f"/{CAUSAL_JLENS_ARTIFACTS_DIR}/status.json",
        f"{CAUSAL_JLENS_ARTIFACTS_DIR}/../qwen25_coder_7b/{DIRECTION_NAME}",
        "results/qwen25_coder_7b_dose/dose_followup_evaluation.json",
    )
    assert all(not is_allowed_causal_jlens_fetch_member(path) for path in rejected)


@pytest.mark.parametrize(
    "name",
    [
        SOURCE_V1_LENS,
        SOURCE_V2_LENS,
        SOURCE_DIRECTION,
        f"{CODE_EVAL_ARTIFACTS_DIR}/status.json",
        f"{JLENS_V2_RESULTS_DIR}/00_jlens_prepare.json",
        f"/{CAUSAL_JLENS_ARTIFACTS_DIR}/status.json",
        f"{CAUSAL_JLENS_ARTIFACTS_DIR}/../qwen25_coder_7b/{DIRECTION_NAME}",
    ],
)
def test_causal_archive_rejects_source_absolute_and_traversal_paths(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        validate_qwen_causal_jlens_archive([_tar_info(name)])


def test_causal_archive_rejects_links_and_extract_leaves_project_untouched(
    tmp_path: Path,
) -> None:
    link = _tar_info(f"{CAUSAL_JLENS_ARTIFACTS_DIR}/status.json", size=0)
    link.type = tarfile.SYMTYPE
    link.linkname = "../../secret"
    with pytest.raises(ValueError, match="link"):
        validate_qwen_causal_jlens_archive([link])

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = json.dumps({"state": "ok"}).encode()
        info = _tar_info(f"{CAUSAL_JLENS_ARTIFACTS_DIR}/status.json", size=len(payload))
        handle.addfile(info, io.BytesIO(payload))
        extra = b"{}"
        info = _tar_info(SOURCE_V1_LENS, size=len(extra))
        handle.addfile(info, io.BytesIO(extra))
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="allowlist|unsafe"):
        extract_qwen_causal_jlens_archive(archive, project)
    assert not (project / "artifacts").exists()
    assert not (project / "results").exists()


def test_causal_source_triple_installs_atomically_and_rolls_back(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    direction, v1, v2 = _seed_causal_sources(project)
    assert require_local_causal_jlens_sources(project) == (direction, v1, v2)

    staged_direction = tmp_path / "direction.pt"
    staged_v1 = tmp_path / "v1.pt"
    staged_v2 = tmp_path / "v2.pt"
    staged_direction.write_bytes(b"new-direction")
    staged_v1.write_bytes(b"new-v1")
    staged_v2.write_bytes(b"new-v2")
    installed = atomic_install_causal_jlens_sources(
        staged_direction=staged_direction,
        staged_v1_lens=staged_v1,
        staged_v2_lens=staged_v2,
        project=project,
    )
    assert installed == (direction, v1, v2)
    assert direction.read_bytes() == b"new-direction"
    assert v1.read_bytes() == b"new-v1"
    assert v2.read_bytes() == b"new-v2"

    staged_direction.write_bytes(b"rollback-direction")
    staged_v1.write_bytes(b"rollback-v1")
    staged_v2.unlink()
    with pytest.raises(FileNotFoundError):
        atomic_install_causal_jlens_sources(
            staged_direction=staged_direction,
            staged_v1_lens=staged_v1,
            staged_v2_lens=staged_v2,
            project=project,
        )
    assert direction.read_bytes() == b"new-direction"
    assert v1.read_bytes() == b"new-v1"
    assert v2.read_bytes() == b"new-v2"


def test_causal_chunked_install_validates_hash_and_size(
    tmp_path: Path,
) -> None:
    installer = _installer()
    payloads = {
        "direction": b"tiny-direction-payload",
        "v1": b"tiny-v1-lens-payload",
        "v2": b"tiny-v2-lens-payload",
    }
    staging = tmp_path / "content"
    direction_staged, direction_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_DIRECTION_STAGING).name,
        payloads["direction"],
    )
    v1_staged, v1_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_V1_LENS_STAGING).name,
        payloads["v1"],
    )
    v2_staged, v2_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_V2_LENS_STAGING).name,
        payloads["v2"],
    )
    project = tmp_path / "project"
    previous = _seed_causal_sources(project)

    installer.install_causal_jlens_sources(
        project=project,
        direction_staging=direction_staged,
        direction_manifest=direction_manifest,
        v1_staging=v1_staged,
        v1_manifest=v1_manifest,
        v2_staging=v2_staged,
        v2_manifest=v2_manifest,
        expected_direction_sha256=_sha(payloads["direction"]),
        expected_v1_sha256=_sha(payloads["v1"]),
        expected_v2_sha256=_sha(payloads["v2"]),
    )

    installed = require_local_causal_jlens_sources(project)
    assert installed[0].read_bytes() == payloads["direction"]
    assert installed[1].read_bytes() == payloads["v1"]
    assert installed[2].read_bytes() == payloads["v2"]
    assert not list(staging.glob("*.part*"))
    assert not list(staging.glob("*.manifest.json"))

    direction_staged, direction_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_DIRECTION_STAGING).name,
        b"altered-direction",
    )
    v1_staged, v1_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_V1_LENS_STAGING).name,
        payloads["v1"],
    )
    v2_staged, v2_manifest = _stage_chunked(
        staging,
        staging / Path(REMOTE_CAUSAL_JLENS_V2_LENS_STAGING).name,
        payloads["v2"],
    )
    with pytest.raises(ValueError):
        installer.install_causal_jlens_sources(
            project=project,
            direction_staging=direction_staged,
            direction_manifest=direction_manifest,
            v1_staging=v1_staged,
            v1_manifest=v1_manifest,
            v2_staging=v2_staged,
            v2_manifest=v2_manifest,
            expected_direction_sha256=_sha(payloads["direction"]),
            expected_v1_sha256=_sha(payloads["v1"]),
            expected_v2_sha256=_sha(payloads["v2"]),
        )
    assert installed[0].read_bytes() == payloads["direction"]
    assert installed[1].read_bytes() == payloads["v1"]
    assert installed[2].read_bytes() == payloads["v2"]
    assert previous[0].read_bytes() == payloads["direction"]


def test_causal_fetch_rejects_identity_conflict_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    existing = destination / CAUSAL_JLENS_RESULTS_DIR / "00_prepare.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_compact_identity("local-run")), encoding="utf-8")
    raw = destination / CAUSAL_JLENS_ARTIFACTS_DIR / "responses.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(_raw_identity("local-run", response="old")) + "\n",
        encoding="utf-8",
    )

    conflicting_root = tmp_path / "conflicting"
    conflict = conflicting_root / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text(
        json.dumps(_compact_identity("remote-run")), encoding="utf-8"
    )
    archive = tmp_path / "conflict.tar.gz"
    pack_qwen_causal_jlens_results(conflicting_root, archive)
    before = existing.read_bytes()
    raw_before = raw.read_bytes()
    with pytest.raises(ValueError, match="identity conflict"):
        extract_qwen_causal_jlens_archive(archive, destination)
    assert existing.read_bytes() == before
    assert raw.read_bytes() == raw_before
    assert not (
        destination / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json"
    ).exists()

    incremental_root = tmp_path / "incremental"
    incremental = incremental_root / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json"
    incremental.parent.mkdir(parents=True)
    incremental.write_text(
        json.dumps(_compact_identity("local-run")), encoding="utf-8"
    )
    archive = tmp_path / "incremental.tar.gz"
    pack_qwen_causal_jlens_results(incremental_root, archive)
    extract_qwen_causal_jlens_archive(archive, destination)
    assert existing.read_bytes() == before
    assert (destination / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json").is_file()


def test_causal_fetch_rejects_internally_conflicting_new_tree(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    results = incoming / CAUSAL_JLENS_RESULTS_DIR
    results.mkdir(parents=True)
    (results / "00_prepare.json").write_text(
        json.dumps(_compact_identity("first-run")), encoding="utf-8"
    )
    (results / "03_analysis.json").write_text(
        json.dumps(_compact_identity("second-run")), encoding="utf-8"
    )
    archive = tmp_path / "internal-conflict.tar.gz"
    pack_qwen_causal_jlens_results(incoming, archive)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="conflicting identities"):
        extract_qwen_causal_jlens_archive(archive, destination)
    assert not (destination / CAUSAL_JLENS_RESULTS_DIR).exists()


def test_causal_fetch_rejects_raw_artifact_from_different_run(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    existing = destination / CAUSAL_JLENS_RESULTS_DIR / "00_prepare.json"
    existing.parent.mkdir(parents=True)
    existing.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")

    incoming = tmp_path / "incoming"
    compact = incoming / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json"
    compact.parent.mkdir(parents=True)
    compact.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")
    raw = incoming / CAUSAL_JLENS_ARTIFACTS_DIR / "responses.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(_raw_identity("other-run", response="raw only")) + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "mixed-raw.tar.gz"
    pack_qwen_causal_jlens_results(incoming, archive)
    with pytest.raises(ValueError, match="conflicting identities"):
        extract_qwen_causal_jlens_archive(archive, destination)
    assert not (
        destination / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json"
    ).exists()


def test_causal_fetch_rolls_back_partial_same_identity_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import refusal_direction_replication.qwen_colab_transport as transport

    destination = tmp_path / "destination"
    result = destination / CAUSAL_JLENS_RESULTS_DIR / "00_prepare.json"
    raw = destination / CAUSAL_JLENS_ARTIFACTS_DIR / "responses.jsonl"
    result.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    result.write_text(json.dumps(_compact_identity("same-run")), encoding="utf-8")
    raw.write_text(
        json.dumps(_raw_identity("same-run", response="old")) + "\n",
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
        json.dumps(_raw_identity("same-run", response="new")) + "\n",
        encoding="utf-8",
    )
    archive = tmp_path / "rollback.tar.gz"
    pack_qwen_causal_jlens_results(incoming, archive)

    real_replace = transport.os.replace

    def fail_raw_install(source, target):
        if Path(target) == raw and ".backups" not in str(source):
            raise OSError("simulated install failure")
        return real_replace(source, target)

    monkeypatch.setattr(transport.os, "replace", fail_raw_install)
    with pytest.raises(OSError, match="simulated"):
        extract_qwen_causal_jlens_archive(archive, destination)
    assert json.loads(result.read_text()) == _compact_identity("same-run")
    assert json.loads(raw.read_text(encoding="utf-8"))["response"] == "old"


def test_causal_pack_rejects_raw_text_leak_in_compact_results(tmp_path: Path) -> None:
    _seed_causal_tree(tmp_path)
    leaked = _compact_identity("causal-run")
    leaked["results"] = {"response": "should not be public"}
    (tmp_path / CAUSAL_JLENS_RESULTS_DIR / "03_analysis.json").write_text(
        json.dumps(leaked) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak"):
        pack_qwen_causal_jlens_results(tmp_path, tmp_path / "out.tar.gz")


def test_causal_orchestrator_routes_sync_bootstrap_status_fetch_and_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    assert module.resolve_session(CAUSAL_PROFILE, None) == "refusal-qwen-causal-jlens"
    assert module.resolve_session("qwen25-coder-code-eval", None) == "refusal-qwen"
    assert module.fetch_pack_script(CAUSAL_PROFILE) == PACK_SCRIPT
    assert module.fetch_pack_timeout(CAUSAL_PROFILE) == 1800
    assert INSTALL_SCRIPT in module.bootstrap_scripts(CAUSAL_PROFILE)
    status = module.status_command("refusal-qwen-causal-jlens", profile=CAUSAL_PROFILE)
    assert status[status.index("-f") + 1] == str(STATUS_SCRIPT)

    project = tmp_path / "refusal_direction"
    project.mkdir()
    (project / "hello.py").write_text("print('ok')\n", encoding="utf-8")
    _seed_causal_sources(project)
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        module, "current_implementation_commit", lambda cwd=None: "a" * 40
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "sync", "--profile", CAUSAL_PROFILE],
    )
    module.main()
    uploaded = [item for command in commands for item in command]
    assert any(
        str(item).startswith(REMOTE_CAUSAL_JLENS_DIRECTION_STAGING)
        for item in uploaded
    )
    assert any(
        str(item).startswith(REMOTE_CAUSAL_JLENS_V1_LENS_STAGING)
        for item in uploaded
    )
    assert any(
        str(item).startswith(REMOTE_CAUSAL_JLENS_V2_LENS_STAGING)
        for item in uploaded
    )
    assert any(str(item).endswith(".manifest.json") for item in uploaded)
    assert any(".part" in str(item) for item in uploaded)
    assert not any(
        "code_eval_source" in str(item) or "jlens_v2_source" in str(item)
        for item in uploaded
    )


def test_causal_start_forces_a100_and_stop_is_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "start", "--profile", CAUSAL_PROFILE, "--gpu", "L4"],
    )
    module.main()
    assert commands[-1][-1] == "A100"
    assert commands[-1][commands[-1].index("-s") + 1] == "refusal-qwen-causal-jlens"

    commands.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "stop", "--profile", CAUSAL_PROFILE],
    )
    module.main()
    assert commands == [
        [
            "colab",
            "--auth=adc",
            "stop",
            "-s",
            "refusal-qwen-causal-jlens",
        ]
    ]


def test_causal_jlens_action_rejects_other_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _orchestrator()
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "causal-jlens", "--profile", "qwen25-coder-code-eval"],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert commands == []

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "code-eval", "--profile", CAUSAL_PROFILE],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert commands == []

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "causal-jlens", "--profile", CAUSAL_PROFILE],
    )
    module.main()
    assert commands[-1][-3].endswith("qwen_run_causal_jlens.py")


def test_causal_scripts_exist_and_pin_dependencies() -> None:
    assert VALIDATE_SCRIPT.is_file()
    for path in (RUN_SCRIPT, STATUS_SCRIPT, PACK_SCRIPT, INSTALL_SCRIPT):
        assert path.is_file()
        assert "/content/refusal_direction/src" in path.read_text(encoding="utf-8")
    install = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "581d398613e5602a5af361e1c34d3a92ea82ba8e" in install
    assert "--no-deps" in install
    assert "0654501399e7cbd560db9501965dc09c567a5563" in install
    assert "evalplus" not in install.casefold()
    assert CAUSAL_JLENS_REMOTE_ARCHIVE == Path("/content/qwen_causal_jlens_results.tar.gz")
    pack = PACK_SCRIPT.read_text(encoding="utf-8")
    assert "pack_qwen_causal_jlens_results" in pack
