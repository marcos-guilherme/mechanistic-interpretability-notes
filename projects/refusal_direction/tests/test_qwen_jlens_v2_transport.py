from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
import torch

from refusal_direction_replication import qwen_colab_transport as transport
from refusal_direction_replication.qwen_colab_transport import (
    JLENS_ARTIFACTS_DIR,
    JLENS_REMOTE_ARCHIVE,
    JLENS_V2_ARTIFACTS_DIR,
    JLENS_V2_FETCH_ALLOWLIST,
    JLENS_V2_REMOTE_ARCHIVE,
    JLENS_V2_RESULTS_DIR,
    REMOTE_JLENS_CHECKPOINT_STAGING,
    REMOTE_JLENS_EVENTS_STAGING,
    REMOTE_JLENS_STATUS_STAGING,
    REMOTE_JLENS_V2_CHECKPOINT_STAGING,
    REMOTE_JLENS_V2_EVENTS_STAGING,
    REMOTE_JLENS_V2_STATUS_STAGING,
    atomic_install_jlens_v2_checkpoint,
    atomic_install_jlens_v2_resume,
    atomic_replace_validated_jlens_v2_backup,
    extract_qwen_jlens_v2_archive,
    optional_local_jlens_v2_checkpoint,
    optional_local_jlens_v2_resume_bundle,
    pack_qwen_jlens_v2_results,
    validate_jlens_v2_resume_bundle,
    validate_qwen_jlens_v2_archive,
)

V2_ARTIFACTS = "artifacts/qwen25_coder_7b_jlens_v2"
V2_RESULTS = "results/qwen25_coder_7b_jlens_v2"


def _write_valid_checkpoint(
    path: Path,
    *,
    fit_count: int = 200,
    n_done: int = 50,
    run_id: str = "jlens-v2-run-1",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "jacobian_sum": {0: torch.eye(4), 1: torch.eye(4)},
            "n_done": n_done,
            "next_idx": n_done,
            "source_layers": [0, 1],
            "target_layer": 2,
            "skip_first": 16,
            "corpus_hash": "c" * 64,
            "config_hash": "d" * 64,
            "model_revision": "c" * 40,
            "implementation_commit": "a" * 40,
            "run_id": run_id,
            "dim_batch": 16,
            "fit_count": fit_count,
        },
        path,
    )
    return path


def _write_resume_members(
    directory: Path,
    *,
    fit_count: int = 200,
    n_done: int = 50,
    run_id: str = "jlens-v2-run-1",
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = _write_valid_checkpoint(
        directory / "jlens_fit_checkpoint.pt",
        fit_count=fit_count,
        n_done=n_done,
        run_id=run_id,
    )
    status = directory / "status.json"
    status.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current_stage": "fit",
                "state": "running",
                "completed_units": n_done,
                "fit_completed_units": n_done,
                "fit_total_units": fit_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = directory / "events.jsonl"
    events.write_text(
        json.dumps({"event": "checkpoint", "run_id": run_id, "n_done": n_done}) + "\n",
        encoding="utf-8",
    )
    return checkpoint, status, events


def _tar_info(name: str, size: int = 8) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    return info


def _compact(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "experiment": "qwen_jlens_v2",
        "config": {"fit_count": 200},
        "results": {"ok": True},
        "provenance": {"run_id": "jlens-v2-run-1"},
    }
    payload.update(overrides)
    return payload


def _populate_v2_tree(root: Path) -> None:
    results = root / V2_RESULTS
    artifacts = root / V2_ARTIFACTS
    results.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    names = [
        "00_jlens_prepare.json",
        "01_jlens_probe.json",
        "02_jlens_fit.json",
        "03_jlens_holdout.json",
        "04_jlens_compare.json",
        "execution_manifest.json",
    ]
    for name in names:
        (results / name).write_text(json.dumps(_compact()) + "\n", encoding="utf-8")
    (results / "jlens_evaluation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "run_id": "jlens-v2-run-1",
                "implementation_commit": "a" * 40,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_resume_members(artifacts)
    (artifacts / "qwen25_coder_7b_jacobian_lens_v2.pt").write_bytes(b"lens-v2")
    (artifacts / "jlens_resume_bundle.json").write_text("{}\n", encoding="utf-8")
    # Must never travel: raw corpus and arbitrary files.
    (artifacts / "corpus.jsonl").write_text('{"code": "secret"}\n', encoding="utf-8")
    (artifacts / "notes.txt").write_text("arbitrary\n", encoding="utf-8")


# --- backup preservation -------------------------------------------------


def test_v2_backup_keeps_previous_file_when_staging_is_corrupt(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.pt"
    _write_valid_checkpoint(destination, fit_count=200, n_done=50)
    previous = destination.read_bytes()
    staged = tmp_path / "download.pt"
    staged.write_bytes(b"corrupt")

    with pytest.raises(Exception):
        atomic_replace_validated_jlens_v2_backup(staged, destination)

    assert destination.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint.pt",
        "download.pt",
    ]


def test_v2_backup_keeps_previous_file_when_fit_count_is_not_a_candidate(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkpoint.pt"
    _write_valid_checkpoint(destination, fit_count=200, n_done=50)
    previous = destination.read_bytes()
    staged = _write_valid_checkpoint(tmp_path / "download.pt", fit_count=123, n_done=5)

    with pytest.raises(ValueError, match="fit_count"):
        atomic_replace_validated_jlens_v2_backup(staged, destination)

    assert destination.read_bytes() == previous


def test_v2_backup_keeps_previous_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    _write_valid_checkpoint(destination, fit_count=200, n_done=50)
    previous = destination.read_bytes()
    staged = _write_valid_checkpoint(tmp_path / "download.pt", fit_count=400, n_done=120)

    def boom(src, dst):
        raise OSError("disk detached")

    monkeypatch.setattr(transport.os, "replace", boom)
    with pytest.raises(OSError, match="disk detached"):
        atomic_replace_validated_jlens_v2_backup(staged, destination)
    monkeypatch.undo()

    assert destination.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint.pt",
        "download.pt",
    ]


def test_v2_backup_replaces_atomically_and_leaves_no_extra_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    _write_valid_checkpoint(destination, fit_count=200, n_done=50)
    staged = _write_valid_checkpoint(tmp_path / "download.pt", fit_count=400, n_done=120)
    expected = staged.read_bytes()
    calls: list[tuple[Path, Path, bytes]] = []
    real_replace = os.replace

    def spy(src, dst):
        # The destination is still the old copy while the temp file is complete.
        calls.append((Path(src), Path(dst), Path(src).read_bytes()))
        return real_replace(src, dst)

    monkeypatch.setattr(transport.os, "replace", spy)
    result = atomic_replace_validated_jlens_v2_backup(staged, destination)
    monkeypatch.undo()

    assert result == destination
    assert destination.read_bytes() == expected
    assert len(calls) == 1
    src, dst, content = calls[0]
    assert dst == destination and src.parent == destination.parent
    assert content == expected
    # No sidecar logs, temp files or copies of the payload are left behind.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint.pt",
        "download.pt",
    ]


# --- v2 checkpoint install / resume --------------------------------------


def test_v2_checkpoint_above_100_installs_and_resumes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    staged = _write_valid_checkpoint(tmp_path / "staged.pt", fit_count=400, n_done=120)

    installed = atomic_install_jlens_v2_checkpoint(staged=staged, project=project)
    assert installed == project / V2_ARTIFACTS / "jlens_fit_checkpoint.pt"
    assert optional_local_jlens_v2_checkpoint(project) == installed

    # Resume bundle pairs the same checkpoint with v2 status/events staging names.
    _write_resume_members(project / V2_ARTIFACTS, fit_count=400, n_done=120)
    bundle = optional_local_jlens_v2_resume_bundle(project)
    assert [staging for _, staging in bundle] == [
        REMOTE_JLENS_V2_CHECKPOINT_STAGING,
        REMOTE_JLENS_V2_STATUS_STAGING,
        REMOTE_JLENS_V2_EVENTS_STAGING,
    ]

    fresh = tmp_path / "remote"
    ckpt, status, events = _write_resume_members(
        tmp_path / "stage", fit_count=400, n_done=120
    )
    atomic_install_jlens_v2_resume(
        checkpoint=ckpt, status=status, events=events, project=fresh
    )
    marker = json.loads(
        (fresh / V2_ARTIFACTS / "jlens_resume_bundle.json").read_text(encoding="utf-8")
    )
    assert marker["n_done"] == 120
    assert marker["fit_count"] == 400
    assert marker["checkpoint"].startswith(V2_ARTIFACTS)
    assert not ckpt.exists() and not status.exists() and not events.exists()
    assert not (fresh / "artifacts" / "qwen25_coder_7b_jlens").exists()


def test_v2_checkpoint_rejects_fit_count_outside_candidates_or_v1_payload(
    tmp_path: Path,
) -> None:
    odd = _write_valid_checkpoint(tmp_path / "odd.pt", fit_count=123, n_done=5)
    with pytest.raises(ValueError, match="fit_count"):
        atomic_install_jlens_v2_checkpoint(staged=odd, project=tmp_path / "p1")

    v1_like = tmp_path / "v1.pt"
    payload = torch.load(odd, map_location="cpu", weights_only=True)
    del payload["fit_count"]
    torch.save(payload, v1_like)
    with pytest.raises(ValueError, match="fit_count"):
        atomic_install_jlens_v2_checkpoint(staged=v1_like, project=tmp_path / "p2")
    assert not (tmp_path / "p2" / V2_ARTIFACTS / "jlens_fit_checkpoint.pt").exists()


def test_v2_resume_install_rolls_back_previous_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    old_ckpt, old_status, old_events = _write_resume_members(
        project / V2_ARTIFACTS, run_id="old-run", n_done=1
    )
    before = old_ckpt.read_bytes()
    ckpt, status, events = _write_resume_members(
        tmp_path / "stage", run_id="new-run", n_done=60
    )
    original = transport._stream_install_regular_file

    def boom(**kwargs):
        if kwargs["destination"].name == "events.jsonl":
            raise OSError("disk full")
        return original(**kwargs)

    monkeypatch.setattr(transport, "_stream_install_regular_file", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_install_jlens_v2_resume(
            checkpoint=ckpt, status=status, events=events, project=project
        )
    assert old_ckpt.read_bytes() == before
    assert json.loads(old_status.read_text(encoding="utf-8"))["run_id"] == "old-run"
    assert "old-run" in old_events.read_text(encoding="utf-8")


# --- pack / archive / extract --------------------------------------------


def test_v2_namespace_is_disjoint_from_v1() -> None:
    assert JLENS_V2_RESULTS_DIR == V2_RESULTS
    assert JLENS_V2_ARTIFACTS_DIR == V2_ARTIFACTS
    assert JLENS_V2_REMOTE_ARCHIVE != JLENS_REMOTE_ARCHIVE
    assert {
        REMOTE_JLENS_V2_CHECKPOINT_STAGING,
        REMOTE_JLENS_V2_STATUS_STAGING,
        REMOTE_JLENS_V2_EVENTS_STAGING,
    }.isdisjoint(
        {
            REMOTE_JLENS_CHECKPOINT_STAGING,
            REMOTE_JLENS_STATUS_STAGING,
            REMOTE_JLENS_EVENTS_STAGING,
        }
    )
    assert all(
        name.startswith((f"{V2_RESULTS}/", f"{V2_ARTIFACTS}/"))
        for name in JLENS_V2_FETCH_ALLOWLIST
    )
    assert not any("corpus" in name for name in JLENS_V2_FETCH_ALLOWLIST)
    assert f"{JLENS_ARTIFACTS_DIR}/status.json" not in JLENS_V2_FETCH_ALLOWLIST


def test_v2_pack_and_extract_only_carry_allowlisted_members(tmp_path: Path) -> None:
    source = tmp_path / "colab" / "refusal_direction"
    _populate_v2_tree(source)
    archive = tmp_path / "out" / "v2.tar.gz"
    pack_qwen_jlens_v2_results(source, archive)

    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert names
    assert names <= set(JLENS_V2_FETCH_ALLOWLIST)
    assert f"{V2_RESULTS}/04_jlens_compare.json" in names
    assert f"{V2_ARTIFACTS}/qwen25_coder_7b_jacobian_lens_v2.pt" in names
    assert f"{V2_ARTIFACTS}/jlens_fit_checkpoint.pt" in names
    assert not any("corpus" in name or "notes" in name for name in names)

    project = tmp_path / "local" / "refusal_direction"
    project.mkdir(parents=True)
    extract_qwen_jlens_v2_archive(archive, project)
    assert (project / V2_RESULTS / "jlens_evaluation.json").is_file()
    assert (project / V2_ARTIFACTS / "jlens_fit_checkpoint.pt").is_file()
    assert not (project / V2_ARTIFACTS / "corpus.jsonl").exists()
    assert not (project / "artifacts" / "qwen25_coder_7b_jlens").exists()


def test_v2_pack_rejects_raw_text_leak(tmp_path: Path) -> None:
    source = tmp_path / "colab" / "refusal_direction"
    _populate_v2_tree(source)
    (source / V2_RESULTS / "02_jlens_fit.json").write_text(
        json.dumps(_compact(results={"code": "def secret(): ..."})) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leak"):
        pack_qwen_jlens_v2_results(source, tmp_path / "out.tar.gz")


@pytest.mark.parametrize(
    "name",
    [
        f"{V2_ARTIFACTS}/corpus.jsonl",
        "artifacts/qwen25_coder_7b_jlens/status.json",
        "results/qwen25_coder_7b_jlens/00_jlens_prepare.json",
        f"/{V2_ARTIFACTS}/status.json",
        f"{V2_ARTIFACTS}/../qwen25_coder_7b_jlens/status.json",
    ],
)
def test_v2_archive_rejects_corpus_v1_absolute_and_traversal_paths(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe|allowlist"):
        validate_qwen_jlens_v2_archive([_tar_info(name)])


def test_v2_archive_rejects_v1_member_paths() -> None:
    members = [_tar_info("artifacts/qwen25_coder_7b_jlens/status.json")]
    with pytest.raises(ValueError, match="unsafe"):
        validate_qwen_jlens_v2_archive(members)


def test_v2_archive_rejects_links_and_extract_leaves_project_untouched(
    tmp_path: Path,
) -> None:
    link = _tar_info(f"{V2_ARTIFACTS}/status.json", size=0)
    link.type = tarfile.SYMTYPE
    link.linkname = "../../secret"
    with pytest.raises(ValueError, match="link"):
        validate_qwen_jlens_v2_archive([link])

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = json.dumps({"state": "ok"}).encode()
        info = _tar_info(f"{V2_ARTIFACTS}/status.json", size=len(payload))
        handle.addfile(info, io.BytesIO(payload))
        extra = b"{}"
        info = _tar_info(f"{V2_ARTIFACTS}/corpus.jsonl", size=len(extra))
        handle.addfile(info, io.BytesIO(extra))
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="allowlist"):
        extract_qwen_jlens_v2_archive(archive, project)
    assert not (project / "artifacts").exists()


# --- review fixes ----------------------------------------------------------


def test_v2_extract_rejects_invalid_archived_fit_count_before_replacing_local(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    local = _write_valid_checkpoint(
        project / V2_ARTIFACTS / "jlens_fit_checkpoint.pt", fit_count=200, n_done=50
    )
    previous = local.read_bytes()
    members = _write_resume_members(tmp_path / "stage", fit_count=123, n_done=5)
    archive = tmp_path / "bad_fit_count.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for member in members:
            handle.add(member, arcname=f"{V2_ARTIFACTS}/{member.name}")

    with pytest.raises(ValueError, match="fit_count"):
        extract_qwen_jlens_v2_archive(archive, project)

    assert local.read_bytes() == previous
    assert sorted(path.name for path in local.parent.iterdir()) == [
        "jlens_fit_checkpoint.pt"
    ]


@pytest.mark.parametrize("operation", ["backup", "install"])
def test_v2_validates_the_exact_temporary_copy_before_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    project = tmp_path / "project"
    destination = _write_valid_checkpoint(
        project / V2_ARTIFACTS / "jlens_fit_checkpoint.pt", fit_count=200, n_done=50
    )
    previous = destination.read_bytes()
    staged = _write_valid_checkpoint(tmp_path / "download.pt", fit_count=400, n_done=120)

    def corrupt_copy(source, target, *args, **kwargs):
        target.write(b"torn copy")

    monkeypatch.setattr(transport.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(ValueError, match="checkpoint"):
        if operation == "backup":
            atomic_replace_validated_jlens_v2_backup(staged, destination)
        else:
            atomic_install_jlens_v2_checkpoint(staged=staged, project=project)
    monkeypatch.undo()

    assert destination.read_bytes() == previous
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "jlens_fit_checkpoint.pt"
    ]


def test_v2_fit_count_must_be_a_real_int(tmp_path: Path) -> None:
    for index, bad in enumerate((200.0, True, "200")):
        staged = _write_valid_checkpoint(tmp_path / f"bad{index}.pt", n_done=5)
        payload = torch.load(staged, map_location="cpu", weights_only=True)
        payload["fit_count"] = bad
        torch.save(payload, staged)
        with pytest.raises(ValueError, match="integer"):
            atomic_install_jlens_v2_checkpoint(
                staged=staged, project=tmp_path / f"project{index}"
            )


def test_v2_resume_status_fit_total_must_match_checkpoint_fit_count(
    tmp_path: Path,
) -> None:
    checkpoint, status, events = _write_resume_members(
        tmp_path / "stage", fit_count=200, n_done=50
    )
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["fit_total_units"] = 800
    status.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fit_total_units"):
        validate_jlens_v2_resume_bundle(
            checkpoint=checkpoint, status=status, events=events
        )
