from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

from refusal_direction_replication.qwen_colab_transport import (
    split_artifact_file,
    write_artifact_chunk_manifest,
)


ROOT = Path(__file__).parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "colab_install_sparse_refusal.py"
RUN_SCRIPT = ROOT / "scripts" / "colab_run_sparse_refusal.py"
DIRECTION_SHA256 = (
    "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _chunks(source: Path, destination: Path) -> Path:
    parts = split_artifact_file(
        source,
        destination.parent,
        chunk_bytes=3,
        part_prefix=destination.name + ".part",
    )
    manifest = destination.with_name(destination.name + ".manifest.json")
    return write_artifact_chunk_manifest(source, parts, manifest)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sparse_scripts_import_without_remote_side_effects() -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_import")
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_import")
    assert installer.REMOTE_ROOT == Path("/content/refusal_direction")
    assert runner.REMOTE_ROOT == Path("/content/refusal_direction")


def test_sparse_installer_assembles_and_hashes_direction_chunks(
    tmp_path: Path,
) -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_chunks")
    source = tmp_path / "source.pt"
    source.write_bytes(b"frozen-direction")
    destination = tmp_path / "direction.pt"
    manifest = _chunks(source, destination)

    assembled = installer.assemble_verified_chunks(
        destination,
        manifest,
        expected_sha256=_sha(source),
    )

    assert assembled.read_bytes() == b"frozen-direction"
    assert not list(tmp_path.glob("direction.pt.part*"))


def test_sparse_installer_rejects_changed_direction_hash(tmp_path: Path) -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_hash")
    source = tmp_path / "source.pt"
    source.write_bytes(b"altered-direction")
    destination = tmp_path / "direction.pt"
    manifest = _chunks(source, destination)

    with pytest.raises(ValueError, match="sha256"):
        installer.assemble_verified_chunks(
            destination,
            manifest,
            expected_sha256=DIRECTION_SHA256,
        )


def test_sparse_installer_copies_pairs_and_optional_checkpoint_atomically(
    tmp_path: Path,
) -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_inputs")
    project = tmp_path / "project"
    direction_source = tmp_path / "direction-source.pt"
    direction_source.write_bytes(b"direction")
    direction_staging = tmp_path / "direction.pt"
    direction_manifest = _chunks(direction_source, direction_staging)
    pairs = tmp_path / "reviewed.jsonl"
    pairs.write_text(
        json.dumps({"pair_id": "private", "offensive_prompt": "SECRET"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_source = tmp_path / "checkpoint-source.pt"
    checkpoint_source.write_bytes(b"checkpoint")
    checkpoint_staging = tmp_path / "checkpoint.pt"
    checkpoint_manifest = _chunks(checkpoint_source, checkpoint_staging)
    status = tmp_path / "training_status.json"
    status.write_text('{"stage":"sae_fit"}\n', encoding="utf-8")

    summary = installer.install_sparse_inputs(
        project=project,
        direction_staging=direction_staging,
        direction_manifest=direction_manifest,
        reviewed_pairs_staging=pairs,
        reviewed_pairs_sha256=_sha(pairs),
        checkpoint_staging=checkpoint_staging,
        checkpoint_manifest=checkpoint_manifest,
        training_status_staging=status,
        expected_direction_sha256=_sha(direction_source),
    )

    assert (
        project / "artifacts/qwen25_coder_7b/selected_direction.pt"
    ).read_bytes() == b"direction"
    private = (
        project
        / "artifacts/qwen25_coder_sparse_refusal/pairs/reviewed.jsonl"
    )
    assert private.read_bytes() == pairs.read_bytes()
    assert (
        project
        / "artifacts/qwen25_coder_sparse_refusal/sae_checkpoint.pt"
    ).read_bytes() == b"checkpoint"
    assert summary["reviewed_pairs_sha256"] == _sha(pairs)
    assert "SECRET" not in json.dumps(summary)


def test_sparse_checkpoint_pair_rolls_back_if_status_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_rollback")
    project = tmp_path / "project"
    artifacts = (
        project / "artifacts" / "qwen25_coder_sparse_refusal"
    )
    artifacts.mkdir(parents=True)
    checkpoint = artifacts / "sae_checkpoint.pt"
    status = artifacts / "training_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    status.write_text('{"stage":"old"}\n', encoding="utf-8")
    incoming_checkpoint = tmp_path / "new-checkpoint.pt"
    incoming_status = tmp_path / "new-status.json"
    incoming_checkpoint.write_bytes(b"new-checkpoint")
    incoming_status.write_text('{"stage":"new"}\n', encoding="utf-8")
    real_replace = installer.os.replace

    def fail_status(source, destination):
        if (
            Path(destination) == status
            and Path(source).name.startswith(".training_status.json.incoming.")
        ):
            raise OSError("simulated status install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(installer.os, "replace", fail_status)
    with pytest.raises(OSError, match="simulated"):
        installer.atomic_install_checkpoint_pair(
            incoming_checkpoint,
            incoming_status,
            project=project,
        )

    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert json.loads(status.read_text(encoding="utf-8"))["stage"] == "old"


def test_sparse_result_fetch_excludes_raw_artifacts() -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_results")
    command = runner.sparse_results_archive_command()
    assert "results/qwen25_coder_sparse_refusal" in command
    assert "execution_manifest.json" in command
    assert "artifacts/" not in command
    assert "pairs" not in command
    assert "generations" not in command
    assert "*" not in command


def test_sparse_checkpoint_fetch_has_explicit_allowlist() -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_checkpoint")
    command = runner.sparse_checkpoint_archive_command()
    assert "sae_checkpoint.pt" in command
    assert "training_status.json" in command
    assert "*" not in command
    assert "pairs" not in command


def test_sparse_archives_contain_only_explicit_allowlists(tmp_path: Path) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_pack")
    artifacts = (
        tmp_path / "artifacts" / "qwen25_coder_sparse_refusal"
    )
    results = tmp_path / "results" / "qwen25_coder_sparse_refusal"
    artifacts.mkdir(parents=True)
    results.mkdir(parents=True)
    (artifacts / "sae_checkpoint.pt").write_bytes(b"checkpoint")
    (artifacts / "training_status.json").write_text("{}\n", encoding="utf-8")
    (artifacts / "pairs").mkdir()
    (artifacts / "pairs" / "reviewed.jsonl").write_text(
        '{"prompt":"secret"}\n', encoding="utf-8"
    )
    (results / "00_prepare.json").write_text("{}\n", encoding="utf-8")
    (results / "execution_manifest.json").write_text("{}\n", encoding="utf-8")

    checkpoint_archive = tmp_path / "checkpoint.tar.gz"
    results_archive = tmp_path / "results.tar.gz"
    runner.pack_checkpoint_archive(tmp_path, checkpoint_archive)
    runner.pack_results_archive(tmp_path, results_archive)

    with tarfile.open(checkpoint_archive, "r:gz") as handle:
        checkpoint_names = set(handle.getnames())
    with tarfile.open(results_archive, "r:gz") as handle:
        result_names = set(handle.getnames())
    assert checkpoint_names == set(runner.SPARSE_CHECKPOINT_ALLOWLIST)
    assert result_names == {
        "results/qwen25_coder_sparse_refusal/00_prepare.json",
        "results/qwen25_coder_sparse_refusal/execution_manifest.json",
    }
    assert not any(name.startswith("artifacts/") for name in result_names)


def test_sparse_pilot_stops_after_hardware_stage(tmp_path: Path) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_pilot")
    calls: list[dict[str, object]] = []

    def fake_run(root, profile, **kwargs):
        calls.append({"root": root, "profile": profile, **kwargs})
        return {
            "experiment": "qwen_sparse_refusal_hardware_pilot",
            "results": {"status": "pilot_complete"},
        }

    result = runner.run_sparse_action(
        "pilot",
        root=tmp_path,
        run_function=fake_run,
    )

    assert result["results"]["status"] == "pilot_complete"
    assert calls[0]["stop_after"] == "hardware_pilot"


def test_sparse_run_leaves_review_pending_for_later_resume(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_resume")
    responses = iter(
        (
            {
                "experiment": "qwen_sparse_refusal_review_export",
                "results": {"status": "review_pending"},
            },
            {
                "experiment": "qwen_sparse_refusal_finalize",
                "results": {"status": "completed"},
            },
        )
    )

    def fake_run(*_args, **_kwargs):
        return next(responses)

    first = runner.run_sparse_action(
        "run",
        root=tmp_path,
        run_function=fake_run,
    )
    second = runner.run_sparse_action(
        "run",
        root=tmp_path,
        run_function=fake_run,
    )
    assert first["results"]["status"] == "review_pending"
    assert second["results"]["status"] == "completed"


def test_sparse_training_status_records_only_compact_stage_identity(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_status")
    payload = {
        "experiment": "qwen_sparse_refusal_review_export",
        "results": {
            "status": "review_pending",
            "prompt": "must not be copied",
        },
        "provenance": {"run_id": "sparse-run"},
    }

    status = runner.write_training_status(payload, root=tmp_path)
    compact = json.loads(status.read_text(encoding="utf-8"))

    assert compact == {
        "last_completed_stage": "review_export",
        "run_id": "sparse-run",
        "status": "review_pending",
    }
    assert "must not be copied" not in status.read_text(encoding="utf-8")


def test_sparse_execution_manifest_hashes_only_compact_stage_files(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_manifest")
    results = tmp_path / "results" / "qwen25_coder_sparse_refusal"
    results.mkdir(parents=True)
    stage = results / "00_prepare.json"
    stage.write_text(
        json.dumps(
            {
                "experiment": "qwen_sparse_refusal_prepare",
                "results": {"status": "prepared"},
                "provenance": {"run_id": "sparse-run"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = runner.write_execution_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["results"]["files"] == [
        {
            "path": "00_prepare.json",
            "sha256": _sha(stage),
        }
    ]
    assert payload["provenance"] == {"run_id": "sparse-run"}
