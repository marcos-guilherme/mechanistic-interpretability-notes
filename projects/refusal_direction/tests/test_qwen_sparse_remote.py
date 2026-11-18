from pathlib import Path
import json
from types import SimpleNamespace

from refusal_direction_replication.qwen_sparse_remote import (
    HEAVY_STAGES,
    ProductionSparseStageExecutor,
)
from refusal_direction_replication.qwen_sparse_remote_operations import (
    _install_training_checkpoint_pair,
    choose_hardware_pilot,
    configure_generation_identity,
    pack_token_rows,
    prepare_sparse_environment,
)


def test_production_executor_dispatches_every_heavy_stage_without_receipts(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path | None]] = []
    identity = {
        "run_id": "run",
        "implementation_commit": "implementation",
        "config_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "model_revision": "revision",
    }

    def prepare(**_kwargs):
        calls.append(("prepare_environment", None))
        return identity

    operations = {"prepare_environment": prepare}
    for stage in HEAVY_STAGES:
        operations[stage] = (
            lambda *, stage=stage, resume_from=None, **_kwargs: (
                calls.append((stage, resume_from))
                or {"status": "completed"}
            )
        )
    executor = ProductionSparseStageExecutor(
        tmp_path,
        operations=operations,
    )

    assert executor.prepare_environment(tmp_path) == identity
    checkpoint = tmp_path / "sae_checkpoint.pt"
    for stage in HEAVY_STAGES:
        executor.run_stage(
            stage,
            root=tmp_path,
            artifacts_dir=tmp_path / "artifacts",
            store=object(),
            resume_from=checkpoint if stage == "sae_fit" else None,
        )

    assert calls[0] == ("prepare_environment", None)
    assert {stage for stage, _resume in calls[1:]} == set(HEAVY_STAGES)
    assert ("sae_fit", checkpoint) in calls
    assert not (tmp_path / "artifacts/stage_inputs").exists()


def test_production_executor_rejects_review_stages(tmp_path: Path) -> None:
    operations = {"prepare_environment": lambda **_kwargs: {}}
    operations.update(
        {
            stage: (lambda **_kwargs: {"status": "completed"})
            for stage in HEAVY_STAGES
        }
    )
    executor = ProductionSparseStageExecutor(
        tmp_path,
        operations=operations,
    )
    try:
        executor.run_stage(
            "review_export",
            root=tmp_path,
            artifacts_dir=tmp_path / "artifacts",
            store=object(),
            resume_from=None,
        )
    except ValueError as error:
        assert "unsupported heavy stage" in str(error)
    else:
        raise AssertionError("review stage was accepted by heavy executor")


def test_prepare_environment_materializes_manifest_and_private_tokens(
    tmp_path: Path,
) -> None:
    config = {
        "model_revision": "model-revision",
        "artifacts_dir": "artifacts/qwen25_coder_sparse_refusal",
    }
    config_path = tmp_path / "configs/qwen25_coder_sparse_refusal.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    reviewed = (
        tmp_path
        / "artifacts/qwen25_coder_sparse_refusal/pairs/reviewed.jsonl"
    )
    reviewed.parent.mkdir(parents=True)
    reviewed.write_text("", encoding="utf-8")
    sources = (
        {
            "repo_id": "code-search-net/code_search_net",
            "config": "python",
            "split": "train",
            "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
            "target_tokens": 4,
        },
        {
            "repo_id": "HuggingFaceH4/ultrachat_200k",
            "config": "default",
            "split": "train_sft",
            "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
            "target_tokens": 4,
        },
        {
            "repo_id": "rufimelo/DeltaSecommits",
            "config": "default",
            "split": "train",
            "revision": "40de1664860b8e37fb2322ab2b2fbd0b248f3598",
            "target_tokens": 4,
        },
    )
    rows = {
        sources[0]["repo_id"]: [
            {"id": "code-1", "func_code_string": "abcdefgh"}
        ],
        sources[1]["repo_id"]: [
            {
                "id": "chat-1",
                "messages": [{"content": "ijklmnop"}],
            }
        ],
        sources[2]["repo_id"]: [
            {
                "id": "delta-1",
                "prior_version": "qrst",
                "after_version": "uvwx",
            }
        ],
    }

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(character) for character in text]

    identity = prepare_sparse_environment(
        tmp_path,
        config=config,
        tokenizer=Tokenizer(),
        dataset_loader=lambda repo_id, **_kwargs: rows[repo_id],
        source_specs=sources,
        run_id="run-id",
        implementation_commit="implementation",
    )

    manifest_path = (
        tmp_path
        / "artifacts/qwen25_coder_sparse_refusal/corpus_manifest.json"
    )
    token_path = (
        tmp_path
        / "artifacts/qwen25_coder_sparse_refusal/executor/corpus_tokens.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_tokens = json.loads(token_path.read_text(encoding="utf-8"))
    assert manifest["accepted_tokens"] == 12
    assert [row["token_count"] for row in manifest["records"]] == [4, 4, 4]
    assert len(private_tokens["fit"]) == 11
    assert len(private_tokens["holdout"]) == 1
    assert identity["run_id"] == "run-id"
    assert identity["corpus_sha256"]
    assert "abcdefgh" not in manifest_path.read_text(encoding="utf-8")


def test_hardware_pilot_selects_first_safe_frozen_microbatch() -> None:
    observations = [
        {
            "microbatch_tokens": 4096,
            "peak_memory_bytes": 37 * 1024**3,
            "projected_hours": 8.0,
            "finite": True,
        },
        {
            "microbatch_tokens": 2048,
            "peak_memory_bytes": 35 * 1024**3,
            "projected_hours": 10.0,
            "finite": True,
        },
    ]
    result = choose_hardware_pilot(
        observations,
        candidates=(4096, 2048, 1024, 512),
        max_memory_bytes=36 * 1024**3,
        max_projected_hours=12,
    )
    assert result == {
        "status": "completed",
        "microbatch_tokens": 2048,
        "peak_memory_bytes": 35 * 1024**3,
        "projected_hours": 10.0,
    }


def test_hardware_pilot_fails_closed_when_no_candidate_is_safe() -> None:
    result = choose_hardware_pilot(
        [
            {
                "microbatch_tokens": 512,
                "peak_memory_bytes": 35 * 1024**3,
                "projected_hours": 13.0,
                "finite": True,
            }
        ],
        candidates=(4096, 2048, 1024, 512),
        max_memory_bytes=36 * 1024**3,
        max_projected_hours=12,
    )
    assert result["status"] == "resource_projection_failed"


def test_token_packing_preserves_partial_final_sequence() -> None:
    rows, masks = pack_token_rows(
        list(range(300)),
        sequence_length=256,
        pad_token_id=-1,
    )
    assert len(rows) == len(masks) == 2
    assert sum(sum(mask) for mask in masks) == 300
    assert rows[1][44:] == [-1] * (256 - 44)


def test_prepare_environment_reuses_valid_private_corpus_without_loading(
    tmp_path: Path,
) -> None:
    config = {
        "model_revision": "model-revision",
        "artifacts_dir": "artifacts/qwen25_coder_sparse_refusal",
    }
    sources = (
        {
            "repo_id": "code-search-net/code_search_net",
            "config": "python",
            "split": "train",
            "revision": "revision",
            "target_tokens": 4,
        },
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(character) for character in text]

    initial = prepare_sparse_environment(
        tmp_path,
        config=config,
        tokenizer=Tokenizer(),
        dataset_loader=lambda *_args, **_kwargs: [
            {"id": "row", "func_code_string": "abcdefgh"}
        ],
        source_specs=sources,
        run_id="stable-run",
        implementation_commit="stable-implementation",
    )

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("valid prepared corpus must not be rematerialized")

    reused = prepare_sparse_environment(
        tmp_path,
        config=config,
        tokenizer=object(),
        dataset_loader=forbidden_loader,
        source_specs=sources,
        run_id="stable-run",
        implementation_commit="stable-implementation",
    )

    assert reused == initial


def test_existing_checkout_at_frozen_revision_never_fetches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from refusal_direction_replication.qwen_sparse_remote_operations import (
        SparseA100Operations,
    )

    destination = tmp_path / "source"
    (destination / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        assert command[-2:] == ["rev-parse", "HEAD"]
        return SimpleNamespace(stdout="frozen-revision\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    resolved = SparseA100Operations._ensure_checkout(
        destination,
        repository="https://invalid.example/repository.git",
        revision="frozen-revision",
    )

    assert resolved == destination
    assert calls == [
        ["git", "-C", str(destination), "rev-parse", "HEAD"]
    ]


def test_prepare_stage_accepts_versioned_private_corpus_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from refusal_direction_replication.qwen_sparse_remote_operations import (
        SparseA100Operations,
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fit_tokens": 3,
                "holdout_tokens": 1,
            }
        ),
        encoding="utf-8",
    )
    operations = SparseA100Operations()
    monkeypatch.setattr(
        operations,
        "_config",
        lambda _root: {"sae": {"fit_tokens": 3, "holdout_tokens": 1}},
    )
    source_calls: list[Path] = []
    monkeypatch.setattr(
        operations,
        "_source_roots",
        lambda path: source_calls.append(Path(path)) or (Path("a"), Path("b")),
    )

    assert operations.prepare(
        root=tmp_path,
        artifacts_dir=artifacts,
    ) == {"status": "completed"}
    assert source_calls == [artifacts]


def test_generation_identity_uses_frozen_tokenizer_eos() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(
            eos_token_id=999,
            do_sample=True,
        )
    )
    tokenizer = SimpleNamespace(eos_token_id=151645)

    configure_generation_identity(
        model,
        tokenizer,
        {"do_sample": False, "eos": "tokenizer_eos"},
    )

    assert model.generation_config.eos_token_id == 151645
    assert model.generation_config.do_sample is False


def test_remote_checkpoint_pair_rolls_back_on_status_install_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "sae_checkpoint.pt"
    status = artifacts / "checkpoint_status.json"
    checkpoint.write_bytes(b"old-checkpoint")
    status.write_text('{"old":true}\n', encoding="utf-8")
    source = tmp_path / "new-checkpoint.pt"
    source.write_bytes(b"new-checkpoint")
    identity = {
        "run_id": "run",
        "implementation_commit": "implementation",
        "config_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "model_revision": "revision",
    }
    real_replace = __import__("os").replace

    def fail_status_install(source_path, destination_path):
        if (
            Path(destination_path) == status
            and Path(source_path).name == ".checkpoint_status.json.incoming"
        ):
            raise OSError("simulated status install failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr("os.replace", fail_status_install)
    try:
        _install_training_checkpoint_pair(
            source,
            artifacts,
            identity,
        )
    except OSError as error:
        assert "simulated" in str(error)
    else:
        raise AssertionError("checkpoint pair installation did not fail")

    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert status.read_text(encoding="utf-8") == '{"old":true}\n'
