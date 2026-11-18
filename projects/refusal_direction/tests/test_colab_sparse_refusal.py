from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
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


class _IdentityExecutor:
    def prepare_environment(self, _root):
        return {
            "run_id": "run",
            "implementation_commit": "implementation",
            "config_sha256": "c" * 64,
            "corpus_sha256": "d" * 64,
            "model_revision": "revision",
        }

    def run_stage(self, *_args, **_kwargs):
        raise AssertionError("fake runtime does not invoke stages")


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
    monkeypatch: pytest.MonkeyPatch,
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
    status = tmp_path / "checkpoint_status.json"
    identity = {
        "run_id": "run",
        "implementation_commit": "implementation",
        "config_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "model_revision": "revision",
    }
    status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "checkpoint_sha256": _sha(checkpoint_source),
            }
        ),
        encoding="utf-8",
    )
    import refusal_direction_replication.qwen_sparse_training as training

    validated: list[tuple[Path, dict[str, str]]] = []
    monkeypatch.setattr(
        training,
        "load_training_checkpoint",
        lambda path, expected: validated.append((Path(path), dict(expected))),
    )

    summary = installer.install_sparse_inputs(
        project=project,
        direction_staging=direction_staging,
        direction_manifest=direction_manifest,
        reviewed_pairs_staging=pairs,
        reviewed_pairs_sha256=_sha(pairs),
        checkpoint_staging=checkpoint_staging,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_status_staging=status,
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
    assert validated == [(checkpoint_staging, identity)]
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
    status = artifacts / "checkpoint_status.json"
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
            and Path(source).name.startswith(".checkpoint_status.json.incoming.")
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
    assert "checkpoint_status.json" in command
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
    checkpoint = artifacts / "sae_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    (artifacts / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "run_id": "run",
                    "implementation_commit": "implementation",
                    "config_sha256": "c" * 64,
                    "corpus_sha256": "d" * 64,
                    "model_revision": "revision",
                },
                "checkpoint_sha256": _sha(checkpoint),
            }
        ),
        encoding="utf-8",
    )
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
        executor=_IdentityExecutor(),
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
        executor=_IdentityExecutor(),
    )
    second = runner.run_sparse_action(
        "run",
        root=tmp_path,
        run_function=fake_run,
        executor=_IdentityExecutor(),
    )
    assert first["results"]["status"] == "review_pending"
    assert second["results"]["status"] == "completed"


def test_sparse_run_status_records_only_compact_stage_identity(
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

    status = runner.write_run_status(payload, root=tmp_path)
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


def test_sparse_installer_requires_lock_and_uses_frozen_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = _load(INSTALL_SCRIPT, "colab_install_sparse_refusal_lock")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="uv 0.12.1 (fixture)\n",
        )

    monkeypatch.setattr(
        installer.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(FileNotFoundError, match="uv.lock"):
        installer.install_locked_project(project)

    (project / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")
    installer.install_locked_project(project)
    assert commands == [
        [sys.executable, "-m", "uv", "--version"],
        [
            sys.executable,
            "-m",
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--project",
            str(project),
        ]
    ]


def test_sparse_runner_activates_frozen_project_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_locked_environment")
    if os.name == "nt":
        site_packages = tmp_path / ".venv/Lib/site-packages"
    else:
        site_packages = (
            tmp_path
            / f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
    site_packages.mkdir(parents=True)
    original = list(sys.path)
    monkeypatch.setattr(sys, "path", original.copy())

    assert runner.activate_locked_environment(tmp_path) == site_packages
    assert sys.path[0] == str(site_packages)


def test_sparse_checkpoint_pack_requires_exact_pair(tmp_path: Path) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_pair_pack")
    artifacts = tmp_path / "artifacts" / "qwen25_coder_sparse_refusal"
    artifacts.mkdir(parents=True)
    (artifacts / "sae_checkpoint.pt").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="checkpoint/status pair"):
        runner.pack_checkpoint_archive(tmp_path, tmp_path / "checkpoint.tar.gz")


def test_sparse_checkpoint_pack_rejects_mismatched_status_hash(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_pair_hash")
    artifacts = tmp_path / "artifacts/qwen25_coder_sparse_refusal"
    artifacts.mkdir(parents=True)
    (artifacts / "sae_checkpoint.pt").write_bytes(b"checkpoint")
    (artifacts / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "run_id": "run",
                    "implementation_commit": "implementation",
                    "config_sha256": "c" * 64,
                    "corpus_sha256": "d" * 64,
                    "model_revision": "revision",
                },
                "checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        runner.pack_checkpoint_archive(
            tmp_path,
            tmp_path / "checkpoint.tar.gz",
        )


def test_sparse_run_status_is_separate_from_training_identity(tmp_path: Path) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_run_status")
    payload = {
        "experiment": "qwen_sparse_refusal_hardware_pilot",
        "results": {"status": "completed"},
        "provenance": {"run_id": "run-identity"},
    }
    status = runner.write_run_status(payload, root=tmp_path)
    assert status.name == "run_status.json"
    assert not (
        tmp_path
        / "artifacts/qwen25_coder_sparse_refusal/training_status.json"
    ).exists()


def test_sparse_restored_identity_and_resume_path_are_explicit(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_resume_identity")
    artifacts = tmp_path / "artifacts/qwen25_coder_sparse_refusal"
    artifacts.mkdir(parents=True)
    checkpoint = artifacts / "sae_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    identity = {
        "run_id": "restored-run",
        "implementation_commit": "implementation",
        "config_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "model_revision": "model-revision",
    }
    (artifacts / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "checkpoint_sha256": _sha(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class Executor:
        def prepare_environment(self, _root):
            return identity

        def run_stage(self, *_args, **_kwargs):
            raise AssertionError("stage execution belongs to the runtime")

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return {"results": {"status": "completed"}}

    runner.run_sparse_action(
        "run",
        root=tmp_path,
        run_function=fake_run,
        executor=Executor(),
    )
    assert captured["run_id"] == "restored-run"
    assert captured["implementation_commit"] == "implementation"
    assert captured["corpus_sha256"] == "d" * 64
    stage_runner = captured["stage_runner"]
    assert stage_runner.resume_from == checkpoint


def test_sparse_restored_identity_rejects_changed_run_identity(
    tmp_path: Path,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_changed_identity")
    artifacts = tmp_path / "artifacts/qwen25_coder_sparse_refusal"
    artifacts.mkdir(parents=True)
    checkpoint = artifacts / "sae_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    identity = {
        "run_id": "restored-run",
        "implementation_commit": "implementation",
        "config_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "model_revision": "model-revision",
    }
    (artifacts / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "checkpoint_sha256": _sha(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    class ChangedExecutor:
        def prepare_environment(self, _root):
            return {**identity, "run_id": "new-run"}

        def run_stage(self, *_args, **_kwargs):
            raise AssertionError("identity rejection must precede stage execution")

    with pytest.raises(ValueError, match="run_id"):
        runner.run_sparse_action(
            "run",
            root=tmp_path,
            run_function=lambda *_args, **_kwargs: {},
            executor=ChangedExecutor(),
        )


def test_sparse_run_builds_production_executor_when_not_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_default_executor")
    executor = _IdentityExecutor()
    created: list[Path] = []
    monkeypatch.setattr(
        runner,
        "create_production_executor",
        lambda root: created.append(Path(root)) or executor,
    )
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return {"results": {"status": "completed"}}

    runner.run_sparse_action(
        "pilot",
        root=tmp_path,
        run_function=fake_run,
    )

    assert created == [tmp_path]
    assert isinstance(captured["stage_runner"], runner.SparseRemoteStageRunner)


def test_real_runtime_stage_runner_pauses_and_resumes_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import refusal_direction_replication.qwen_sparse_runtime as runtime

    runner = _load(RUN_SCRIPT, "colab_run_sparse_refusal_integration")
    (tmp_path / "configs").mkdir()
    config_source = ROOT / "configs/qwen25_coder_sparse_refusal.json"
    config_path = tmp_path / "configs/qwen25_coder_sparse_refusal.json"
    config_path.write_bytes(config_source.read_bytes())
    artifacts = tmp_path / "artifacts/qwen25_coder_sparse_refusal"
    pairs_dir = artifacts / "pairs"
    pairs_dir.mkdir(parents=True)
    reviewed = pairs_dir / "reviewed.jsonl"
    pair_rows = [
        {
            "pair_id": f"pair-{index:04d}",
            "category": f"category-{index % 4}",
            "source_identity": f"source-{index:04d}",
            "offensive_prompt": f"offensive prompt {index}",
            "defensive_prompt": f"defensive prompt {index}",
            "offensive_sha256": hashlib.sha256(
                f"offensive prompt {index}".encode()
            ).hexdigest(),
            "defensive_sha256": hashlib.sha256(
                f"defensive prompt {index}".encode()
            ).hexdigest(),
            "intent_rationale": f"rationale {index}",
            "approved": True,
        }
        for index in range(256)
    ]
    reviewed.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows),
        encoding="utf-8",
    )
    corpus_manifest = artifacts / "corpus_manifest.json"
    corpus_manifest.write_text('{"fixture":"corpus"}\n', encoding="utf-8")
    prepare = tmp_path / "results/qwen25_coder_7b_code_eval/00_prepare.json"
    prepare.parent.mkdir(parents=True)
    prepare.write_text('{"results":{}}\n', encoding="utf-8")
    monkeypatch.setattr(runtime, "FROZEN_PREPARE_SHA256", _sha(prepare))

    class FixtureExecutor:
        def prepare_environment(self, _root):
            return {
                "run_id": "fixture-run",
                "implementation_commit": "fixture-implementation",
                "config_sha256": hashlib.sha256(
                    json.dumps(
                        json.loads(config_path.read_text(encoding="utf-8")),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "corpus_sha256": _sha(corpus_manifest),
                "model_revision": json.loads(
                    config_path.read_text(encoding="utf-8")
                )["model_revision"],
            }

        def run_stage(
            self,
            stage,
            *,
            artifacts_dir,
            store,
            resume_from,
            **_kwargs,
        ):
            assert resume_from is None
            if stage in {"prepare", "hardware_pilot", "sae_fit"}:
                return {"status": "completed"}
            if stage == "sae_quality":
                return {
                    "status": "pass",
                    "explained_variance": 0.8,
                    "mean_cosine": 0.9,
                    "nonfinite_count": 0,
                    "mean_l0": 64.0,
                    "dead_feature_fraction": 0.2,
                    "norm_drift": 0.2,
                }
            if stage == "decompose":
                return {
                    "status": "pass",
                    "cosine": 0.95,
                    "relative_l2": 0.35,
                    "norm": 1.0,
                }
            if stage == "discover":
                return {
                    "status": "completed",
                    "association_count": 1,
                    "association_sha256": "a" * 64,
                }
            if stage == "select":
                rows = []
                aliases = {
                    "sparse_unfiltered": "1",
                    "sparse_filtered": "2",
                    "sparse_random_control": "3",
                }
                for vector, suffix in aliases.items():
                    for intervention in ("addition", "ablation"):
                        for cardinality in (8, 16, 32, 64):
                            for dose in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                                rows.append(
                                    {
                                        "vector": vector,
                                        "intervention": intervention,
                                        "cardinality": cardinality,
                                        "dose": dose,
                                        "dense_desired_effect": 0.8,
                                        "desired_effect": 0.76,
                                        "defensive_false_refusal": cardinality
                                        / 1000,
                                        "defensive_kl": cardinality / 100,
                                        "feature_indices": list(
                                            range(cardinality)
                                        ),
                                        "coefficients": [1.0] * cardinality,
                                        "vector_sha256": suffix * 64,
                                        "vector_norm": 1.0,
                                        "dense_vector_sha256": "d" * 64,
                                        "dense_vector_norm": 1.0,
                                        "dense_dose": 0.5,
                                    }
                                )
                return runtime.freeze_selection(rows)
            if stage == "test":
                response_dir = artifacts_dir / "test_responses"
                response_dir.mkdir(parents=True)
                test_ids = {pair["pair_id"] for pair in store.test_pairs}
                by_id = {
                    row["pair_id"]: row
                    for row in pair_rows
                    if row["pair_id"] in test_ids
                }
                for condition in (
                    "dense_addition",
                    "filtered_addition",
                    "dense_ablation",
                    "filtered_ablation",
                ):
                    path = response_dir / f"{condition}.jsonl"
                    with path.open("w", encoding="utf-8") as handle:
                        for pair_id in sorted(by_id):
                            for side in ("offensive", "defensive"):
                                prompt = by_id[pair_id][f"{side}_prompt"]
                                handle.write(
                                    json.dumps(
                                        {
                                            "pair_id": pair_id,
                                            "side": side,
                                            "condition": condition,
                                            "prompt": prompt,
                                            "prompt_sha256": hashlib.sha256(
                                                prompt.encode()
                                            ).hexdigest(),
                                            "generation": "fixture generation",
                                            "automatic_refusal": False,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                return {
                    "status": "completed",
                    "selection_freeze_sha256": (
                        store.selection_freeze_sha256()
                    ),
                    "test_split_sha256": store.test_split_sha256,
                    "observations": runtime.expected_test_matrix(
                        store.test_pairs
                    ),
                }
            raise AssertionError(stage)

    executor = FixtureExecutor()
    identity = executor.prepare_environment(tmp_path)
    stage_runner = runner.build_remote_stage_runner(tmp_path, executor)
    paused = runtime.run_qwen_sparse_refusal(
        tmp_path,
        run_id=identity["run_id"],
        implementation_commit=identity["implementation_commit"],
        corpus_sha256=identity["corpus_sha256"],
        stage_runner=stage_runner,
    )
    assert paused["results"]["status"] == "review_pending"
    assert not (artifacts / "stage_inputs").exists()
    blind = artifacts / "review/blind_review.jsonl"
    blind_sha = _sha(blind)
    labels = artifacts / "review/responses_reviewed.jsonl"
    labels.write_text(
        "".join(
            json.dumps(
                {
                    "review_id": row["review_id"],
                    "label": "answer",
                    "blind_review_sha256": blind_sha,
                },
                sort_keys=True,
            )
            + "\n"
            for row in (
                json.loads(line)
                for line in blind.read_text(encoding="utf-8").splitlines()
            )
        ),
        encoding="utf-8",
    )
    resumed = runtime.run_qwen_sparse_refusal(
        tmp_path,
        run_id=identity["run_id"],
        implementation_commit=identity["implementation_commit"],
        corpus_sha256=identity["corpus_sha256"],
        stage_runner=stage_runner,
        stop_after="review_merge",
    )
    assert resumed["results"]["status"] == "review_complete"
