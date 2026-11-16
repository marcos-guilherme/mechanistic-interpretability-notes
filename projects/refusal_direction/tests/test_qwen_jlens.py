from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_jlens import (
    CHECKPOINT_IDENTITY_FIELDS,
    DIM_BATCH_PROBE_ORDER,
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    EXPECTED_SOURCE_LAYERS,
    EXPECTED_TARGET_LAYER,
    FIT_COUNT,
    HOLDOUT_COUNT,
    JLENS_PROFILE,
    JLENS_UPSTREAM_COMMIT,
    JLENS_UPSTREAM_SPEC,
    MAX_PROJECTED_SECONDS,
    SKIP_FIRST,
    SOURCE_DOSE_RUN_ID,
    assert_jlens_contract,
    build_codesearchnet_corpus,
    compact_corpus_records,
    config_hash,
    corpus_hash,
    ensure_hf_text_config,
    installed_jlens_commit,
    is_cuda_oom,
    jlens_metadata_contains_leak,
    load_exclusion_hashes,
    load_fit_checkpoint,
    probe_dim_batch,
    project_fit_duration,
    rank_one_based,
    require_finite_logits,
    select_dim_batch,
    summarize_holdout_metrics,
    valid_token_positions,
    validate_checkpoint_identity,
    validate_checkpoint_payload,
    validate_fitted_lens,
    wrap_hf_model_for_jlens,
    write_fit_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
DATASET_REPO = "code-search-net/code_search_net"
DATASET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"


class CountingTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = list(text)
        if add_special_tokens:
            return ["<s>", *tokens]
        return tokens


def _row(
    index: int,
    *,
    code: str | None = None,
    docstring: str | None = None,
    whole: str | None = None,
) -> dict:
    body = code if code is not None else ("print(1)\n" + "x = 1\n" * 80)
    doc = docstring if docstring is not None else "Return the helper."
    combined = whole if whole is not None else f"def f_{index}():\n    \"\"\"{doc}\"\"\"\n{body}"
    return {
        "repository_name": "org/repo",
        "func_path_in_repository": f"mod_{index}.py",
        "func_name": f"f_{index}",
        "whole_func_string": combined,
        "func_code_string": body,
        "func_documentation_string": doc,
        "language": "python",
        "split_name": "train",
    }


def _long_rows(count: int, *, start: int = 0) -> list[dict]:
    return [_row(index) for index in range(start, start + count)]


def test_jlens_profile_is_isolated_from_pilot_and_dose() -> None:
    original = load_config(ROOT, profile="qwen25_coder_7b.json")
    dose = load_config(ROOT, profile="qwen25_coder_7b_dose.json")
    jlens = load_config(ROOT, profile=JLENS_PROFILE)
    assert jlens["model_name"] == QWEN_MODEL
    assert jlens["model_revision"] == QWEN_REVISION
    assert jlens["model_revision"] == original["model_revision"]
    assert jlens["results_dir"] == "results/qwen25_coder_7b_jlens"
    assert jlens["artifacts_dir"] == "artifacts/qwen25_coder_7b_jlens"
    assert jlens["results_dir"] != original["results_dir"]
    assert jlens["results_dir"] != dose["results_dir"]
    assert jlens["artifacts_dir"] != original["artifacts_dir"]
    assert jlens["artifacts_dir"] != dose["artifacts_dir"]
    assert jlens["n_layers"] == EXPECTED_N_LAYERS == 28
    assert jlens["d_model"] == EXPECTED_D_MODEL == 3584
    assert jlens["source_layers"] == list(range(27)) == EXPECTED_SOURCE_LAYERS
    assert jlens["target_layer"] == EXPECTED_TARGET_LAYER == 27
    assert jlens["max_seq_len"] == 128
    assert jlens["skip_first"] == SKIP_FIRST == 16
    assert jlens["fit_count"] == FIT_COUNT == 100
    assert jlens["holdout_count"] == HOLDOUT_COUNT == 20
    assert jlens["dim_batch_probe_order"] == DIM_BATCH_PROBE_ORDER == [32, 16, 8]
    assert jlens["jlens_upstream_commit"] == JLENS_UPSTREAM_COMMIT
    assert jlens["dataset_repo_id"] == DATASET_REPO
    assert jlens["dataset_config"] == "python"
    assert jlens["dataset_split"] == "train"
    assert jlens["dataset_revision"] == DATASET_REVISION
    from refusal_direction_replication.qwen_jlens import require_dataset_identity

    require_dataset_identity(jlens)
    assert jlens["source_dose_run_id"] == SOURCE_DOSE_RUN_ID
    assert "dose_grid" not in jlens


def test_require_dataset_identity_rejects_drift() -> None:
    from refusal_direction_replication.qwen_jlens import require_dataset_identity

    config = load_config(ROOT, profile=JLENS_PROFILE)
    require_dataset_identity(config)
    drifted = dict(config)
    drifted["dataset_revision"] = "0" * 40
    with pytest.raises(ValueError, match="dataset identity"):
        require_dataset_identity(drifted)
    drifted = dict(config)
    drifted["dataset_config"] = "javascript"
    with pytest.raises(ValueError, match="dataset identity"):
        require_dataset_identity(drifted)
    drifted = dict(config)
    drifted["dataset_split"] = "test"
    with pytest.raises(ValueError, match="dataset identity"):
        require_dataset_identity(drifted)
    drifted = dict(config)
    drifted["dataset_repo_id"] = "other/repo"
    with pytest.raises(ValueError, match="dataset identity"):
        require_dataset_identity(drifted)


def test_corpus_keeps_code_and_docstring_and_deduplicates() -> None:
    rows = [
        _row(0, code="", docstring="docs"),
        _row(1, code="print(1)", docstring=""),
        _row(2),
        _row(3, whole=_row(2)["whole_func_string"]),
        *_long_rows(130, start=10),
    ]
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=CountingTokenizer(),
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=DATASET_REVISION,
    )
    assert corpus.n_selected == 120
    assert len(corpus.fit) == 100
    assert len(corpus.holdout) == 20
    assert all(item.token_count >= 128 for item in (*corpus.fit, *corpus.holdout))
    assert all(item.fit_text for item in (*corpus.fit, *corpus.holdout))
    fit_hashes = {item.content_hash for item in corpus.fit}
    holdout_hashes = {item.content_hash for item in corpus.holdout}
    fit_ids = {item.example_id for item in corpus.fit}
    holdout_ids = {item.example_id for item in corpus.holdout}
    assert fit_hashes.isdisjoint(holdout_hashes)
    assert fit_ids.isdisjoint(holdout_ids)
    assert corpus.n_duplicate_dropped >= 1
    assert corpus.n_missing_code_or_docstring >= 2
    compact = compact_corpus_records(corpus)
    dumped = json.dumps(compact)
    assert "print(1)" not in dumped
    assert "Return the helper" not in dumped
    assert "whole_func_string" not in dumped
    assert compact["dataset_revision"] == DATASET_REVISION
    assert compact["fit_count"] == 100
    assert compact["holdout_count"] == 20
    assert jlens_metadata_contains_leak({"func_code_string": "def f(): pass"})
    assert jlens_metadata_contains_leak({"docstring": "secret"})
    assert not jlens_metadata_contains_leak(compact)


def test_exclusion_hashes_are_recorded_and_never_silently_claimed(
    tmp_path: Path,
) -> None:
    rows = _long_rows(130)
    target = build_codesearchnet_corpus(
        rows[:1],
        tokenizer=CountingTokenizer(),
        fit_count=1,
        holdout_count=0,
        min_tokens=128,
        dataset_revision=DATASET_REVISION,
    ).fit[0]
    missing = load_exclusion_hashes(tmp_path / "missing.txt")
    assert missing.available is False
    assert missing.hashes == frozenset()
    exclusion_path = tmp_path / "exclusions.txt"
    exclusion_path.write_text(target.content_hash + "\n", encoding="utf-8")
    present = load_exclusion_hashes(exclusion_path)
    with_file = build_codesearchnet_corpus(
        rows,
        tokenizer=CountingTokenizer(),
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=DATASET_REVISION,
        exclusion_hashes=present.hashes,
        exclusions_available=present.available,
    )
    without = build_codesearchnet_corpus(
        rows,
        tokenizer=CountingTokenizer(),
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=DATASET_REVISION,
        exclusion_hashes=missing.hashes,
        exclusions_available=missing.available,
    )
    assert with_file.exclusions_available is True
    assert with_file.n_excluded >= 1
    assert target.content_hash not in {
        item.content_hash for item in (*with_file.fit, *with_file.holdout)
    }
    assert without.exclusions_available is False
    assert without.n_excluded == 0
    compact = compact_corpus_records(with_file, exclusion=present)
    assert compact["exclusions_available"] is True
    assert compact["n_excluded"] == with_file.n_excluded
    assert compact["exclusion_file_sha256"] == present.source_sha256
    assert compact["exclusion_file_path"] == present.source_path
    silent = compact_corpus_records(without)
    assert silent["exclusions_available"] is False
    assert silent["benchmark_deduplicated"] is False
    assert silent.get("exclusion_file_sha256") in {None, ""}


def test_probe_falls_back_only_on_cuda_oom_and_aborts_at_eight() -> None:
    calls: list[int] = []

    def oom(_dim_batch: int) -> None:
        calls.append(_dim_batch)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="dim_batch 8 OOMed"):
        probe_dim_batch(oom, order=(32, 16, 8))
    assert calls == [32, 16, 8]

    calls.clear()

    def other(_dim_batch: int) -> None:
        calls.append(_dim_batch)
        raise ValueError("bad graph")

    with pytest.raises(ValueError, match="bad graph"):
        probe_dim_batch(other, order=(32, 16, 8))
    assert calls == [32]

    def ok(dim_batch: int) -> str:
        if dim_batch == 32:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return f"ok-{dim_batch}"

    result = probe_dim_batch(ok, order=(32, 16, 8))
    assert result.selected == 16
    assert result.attempts[0]["dim_batch"] == 32
    assert result.attempts[0]["outcome"] == "cuda_oom"
    assert result.attempts[1]["outcome"] == "ok"
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert not is_cuda_oom(ValueError("CUDA graph invalid"))
    assert select_dim_batch(result) == 16


def test_projected_runtime_aborts_above_twelve_hours() -> None:
    projection = project_fit_duration(probe_seconds=500.0, n_prompts=100)
    assert projection.projected_seconds == 50_000.0
    assert projection.exceeds_limit is True
    assert projection.limit_seconds == MAX_PROJECTED_SECONDS == 12 * 3600
    ok = project_fit_duration(probe_seconds=10.0, n_prompts=100)
    assert ok.exceeds_limit is False
    assert ok.projected_seconds == 1000.0


def test_checkpoint_resumes_only_with_matching_identity(tmp_path: Path) -> None:
    identity = {
        "source_layers": [0, 1, 2],
        "target_layer": 3,
        "skip_first": 16,
        "corpus_hash": "a" * 64,
        "config_hash": "b" * 64,
        "model_revision": QWEN_REVISION,
        "implementation_commit": "c" * 40,
        "run_id": "jlens-run-1",
    }
    running = {
        0: torch.eye(4),
        1: torch.ones(4, 4),
        2: torch.zeros(4, 4),
    }
    path = tmp_path / "jlens_fit_checkpoint.pt"
    write_fit_checkpoint(
        path,
        jacobian_sum=running,
        n_done=4,
        next_idx=4,
        dim_batch=16,
        **identity,
    )
    loaded = load_fit_checkpoint(path)
    validate_checkpoint_identity(loaded, **identity)
    validate_checkpoint_payload(
        loaded,
        source_layers=identity["source_layers"],
        d_model=4,
        dim_batch=16,
        fit_count=100,
    )
    assert loaded["n_done"] == 4
    assert loaded["next_idx"] == 4
    assert loaded["dim_batch"] == 16
    assert torch.equal(loaded["jacobian_sum"][0], torch.eye(4))
    stale = dict(identity)
    stale["run_id"] = "other"
    with pytest.raises(ValueError, match="run_id"):
        validate_checkpoint_identity(loaded, **stale)
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not-a-tensor")
    with pytest.raises(ValueError, match="checkpoint"):
        load_fit_checkpoint(corrupt)
    assert set(CHECKPOINT_IDENTITY_FIELDS) >= {
        "source_layers",
        "target_layer",
        "skip_first",
        "corpus_hash",
        "config_hash",
        "model_revision",
        "implementation_commit",
        "run_id",
    }


def test_holdout_metrics_and_lens_shape_checks() -> None:
    logits = torch.tensor([0.1, 0.9, 0.3, 0.2])
    assert rank_one_based(logits, 1) == 1
    assert rank_one_based(logits, 2) == 2
    assert rank_one_based(logits, 0) == 4
    records = [
        {
            "example_id": "h0",
            "position": 16,
            "layer": 0,
            "jlens_rank": 1,
            "logit_lens_rank": 4,
        },
        {
            "example_id": "h0",
            "position": 17,
            "layer": 0,
            "jlens_rank": 5,
            "logit_lens_rank": 2,
        },
        {
            "example_id": "h1",
            "position": 16,
            "layer": 1,
            "jlens_rank": 10,
            "logit_lens_rank": 1,
        },
        {
            "example_id": "h1",
            "position": 17,
            "layer": 1,
            "jlens_rank": 3,
            "logit_lens_rank": 8,
        },
    ]
    summary = summarize_holdout_metrics(records, source_layers=[0, 1])
    assert summary["n_items"] == 2
    item0 = summary["by_item"]["h0"]["0"]
    assert item0["n_positions"] == 2
    assert item0["jlens_rank_sum"] == 6
    assert item0["jlens_hits@1"] == 1
    assert item0["jlens_hits@5"] == 2
    assert item0["jlens_hits@10"] == 2
    layer0 = summary["by_layer"]["0"]["jlens"]
    assert layer0["n_observations"] == 2
    assert layer0["mean_rank"] == pytest.approx(3.0)
    assert layer0["pass@1"] == pytest.approx(0.5)
    assert layer0["pass@5"] == pytest.approx(1.0)
    assert layer0["pass@10"] == pytest.approx(1.0)
    overall = summary["overall"]["logit_lens"]
    assert overall["n_observations"] == 4
    assert overall["pass@1"] == pytest.approx(0.25)
    assert 16 in valid_token_positions(seq_len=128, skip_first=16)
    assert 127 not in valid_token_positions(seq_len=128, skip_first=16)
    assert 0 not in valid_token_positions(seq_len=128, skip_first=16)

    class FakeLens:
        def __init__(self) -> None:
            self.source_layers = list(range(3))
            self.d_model = 8
            self.n_prompts = 100
            self.jacobians = {
                layer: torch.eye(8) for layer in self.source_layers
            }

    validate_fitted_lens(
        FakeLens(),
        source_layers=list(range(3)),
        d_model=8,
        n_prompts=100,
    )
    broken = FakeLens()
    broken.jacobians[1] = torch.full((8, 8), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_fitted_lens(
            broken,
            source_layers=list(range(3)),
            d_model=8,
            n_prompts=100,
        )
    wrong_shape = FakeLens()
    wrong_shape.jacobians[0] = torch.eye(4)
    with pytest.raises(ValueError, match="shape"):
        validate_fitted_lens(
            wrong_shape,
            source_layers=list(range(3)),
            d_model=8,
            n_prompts=100,
        )


def test_corpus_and_config_hashes_are_stable() -> None:
    rows = _long_rows(120)
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=CountingTokenizer(),
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=DATASET_REVISION,
    )
    first = corpus_hash(corpus)
    second = corpus_hash(
        build_codesearchnet_corpus(
            rows,
            tokenizer=CountingTokenizer(),
            fit_count=100,
            holdout_count=20,
            min_tokens=128,
            dataset_revision=DATASET_REVISION,
        )
    )
    assert first == second
    assert len(first) == 64
    payload = {"a": 1, "b": [2, 3]}
    assert config_hash(payload) == hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_jlens_contract_requires_upstream_adapter_and_no_deps_pin() -> None:
    assert JLENS_UPSTREAM_COMMIT == "581d398613e5602a5af361e1c34d3a92ea82ba8e"
    assert JLENS_UPSTREAM_SPEC.endswith(f"@{JLENS_UPSTREAM_COMMIT}")
    assert "--no-deps" in " ".join(
        ["pip", "install", "--no-deps", JLENS_UPSTREAM_SPEC]
    )
    module = SimpleNamespace(
        from_hf=object(),
        jacobian_for_prompt=object(),
        JacobianLens=type(
            "JacobianLens",
            (),
            {"save": lambda self, path, dtype=None: None, "load": classmethod(lambda cls, path: None)},
        ),
        fitting=SimpleNamespace(SKIP_FIRST_N_POSITIONS=16, jacobian_for_prompt=object()),
        __name__="jlens",
    )
    assert_jlens_contract(module)
    with pytest.raises(ValueError, match="from_hf"):
        assert_jlens_contract(SimpleNamespace(jacobian_for_prompt=object()))


def test_probe_clears_oom_frames_and_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    collected = {"gc": 0, "empty": 0}

    monkeypatch.setattr(
        "refusal_direction_replication.qwen_jlens.gc.collect",
        lambda: collected.__setitem__("gc", collected["gc"] + 1) or 0,
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: collected.__setitem__("empty", collected["empty"] + 1),
    )

    def oom(_dim_batch: int) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="dim_batch 8 OOMed") as caught:
        probe_dim_batch(oom, order=(32, 16, 8))
    assert caught.value.__cause__ is None
    assert collected["gc"] >= 3
    assert collected["empty"] >= 3


def test_checkpoint_rejects_inconsistent_counts_shapes_and_nonfinite(
    tmp_path: Path,
) -> None:
    identity = {
        "source_layers": [0, 1],
        "target_layer": 2,
        "skip_first": 16,
        "corpus_hash": "a" * 64,
        "config_hash": "b" * 64,
        "model_revision": QWEN_REVISION,
        "implementation_commit": "c" * 40,
        "run_id": "jlens-run-1",
    }
    path = tmp_path / "jlens_fit_checkpoint.pt"
    write_fit_checkpoint(
        path,
        jacobian_sum={0: torch.eye(4), 1: torch.eye(4)},
        n_done=3,
        next_idx=4,
        dim_batch=16,
        **identity,
    )
    loaded = load_fit_checkpoint(path)
    with pytest.raises(ValueError, match="n_done"):
        validate_checkpoint_payload(
            loaded,
            source_layers=[0, 1],
            d_model=4,
            dim_batch=16,
            fit_count=100,
        )
    write_fit_checkpoint(
        path,
        jacobian_sum={0: torch.eye(4), 1: torch.full((4, 4), float("nan"))},
        n_done=4,
        next_idx=4,
        dim_batch=16,
        **identity,
    )
    with pytest.raises(ValueError, match="finite"):
        validate_checkpoint_payload(
            load_fit_checkpoint(path),
            source_layers=[0, 1],
            d_model=4,
            dim_batch=16,
            fit_count=100,
        )
    write_fit_checkpoint(
        path,
        jacobian_sum={0: torch.eye(4)},
        n_done=1,
        next_idx=1,
        dim_batch=8,
        **identity,
    )
    with pytest.raises(ValueError, match="layer"):
        validate_checkpoint_payload(
            load_fit_checkpoint(path),
            source_layers=[0, 1],
            d_model=4,
            dim_batch=8,
            fit_count=100,
        )


def test_readout_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_finite_logits(torch.tensor([0.1, float("nan")]), "jlens_logits")
    require_finite_logits(torch.tensor([0.1, 0.2]), "jlens_logits")


def test_exclusion_file_requires_sha256_and_records_provenance(tmp_path: Path) -> None:
    bad = tmp_path / "exclusions.txt"
    bad.write_text("not-a-hash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_exclusion_hashes(bad)
    digest = "a" * 64
    good = tmp_path / "good.txt"
    good.write_text(digest + "\n", encoding="utf-8")
    loaded = load_exclusion_hashes(good)
    assert loaded.available is True
    assert loaded.hashes == frozenset({digest})
    assert loaded.source_sha256 == hashlib.sha256(good.read_bytes()).hexdigest()
    assert loaded.source_path.endswith("good.txt")


def test_installed_jlens_commit_reads_direct_url_metadata() -> None:
    class Dist:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def read_text(self, name: str) -> str | None:
            if name != "direct_url.json":
                return None
            import json as json_lib

            return json_lib.dumps(self.payload)

    matched = Dist(
        {
            "url": "https://github.com/anthropics/jacobian-lens",
            "vcs_info": {"vcs": "git", "commit_id": JLENS_UPSTREAM_COMMIT},
        }
    )
    assert installed_jlens_commit(distribution=matched) == JLENS_UPSTREAM_COMMIT
    with pytest.raises(ValueError, match="commit"):
        installed_jlens_commit(
            distribution=Dist({"vcs_info": {"commit_id": "deadbeef" * 5}})
        )
    with pytest.raises(ValueError, match="direct_url"):
        installed_jlens_commit(distribution=Dist({}))


def test_shim_adds_get_text_config_when_missing() -> None:
    config = SimpleNamespace(num_hidden_layers=2, hidden_size=16)
    model = SimpleNamespace(config=config)
    ensure_hf_text_config(model)
    assert model.config.get_text_config() is config
    already = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=16,
            get_text_config=lambda decoder=False: "kept",
        )
    )
    ensure_hf_text_config(already)
    assert already.config.get_text_config() == "kept"


def test_real_qwen2_upstream_wrap_on_transformers_4_44() -> None:
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from refusal_direction_replication.qwen_jlens_runtime import default_wrap_model

    assert transformers.__version__.startswith(("4.44.", "4.48."))
    config = Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    if transformers.__version__.startswith("4.44."):
        assert not hasattr(model.config, "get_text_config")

    class _Tok:
        bos_token_id = None

        def __call__(
            self,
            text: str,
            return_tensors: str = "pt",
            truncation: bool = True,
            max_length: int = 128,
        ):
            n = min(max_length, 24)
            return SimpleNamespace(input_ids=torch.arange(n).unsqueeze(0) % 64)

    bound = default_wrap_model(model, _Tok())
    assert bound.n_layers == 2
    assert bound.d_model == 16
    from jlens.fitting import jacobian_for_prompt

    jacobians, seq_len, n_valid = jacobian_for_prompt(
        bound,
        "def tiny():\n    return 1\n" * 4,
        [0],
        target_layer=1,
        dim_batch=8,
        max_seq_len=32,
        skip_first=0,
    )
    assert seq_len > 1
    assert n_valid > 0
    assert jacobians[0].shape == (16, 16)
    assert torch.isfinite(jacobians[0]).all()
    wrapped = wrap_hf_model_for_jlens(model, _Tok())
    assert wrapped.n_layers == 2
