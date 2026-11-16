from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refusal_direction_replication.gates import STATUS_PASS
from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_dose import sha256_file
from refusal_direction_replication.qwen_jlens import (
    FINAL_LENS_FILENAME,
    JLENS_PROFILE,
    QWEN_REVISION,
    SOURCE_DOSE_RUN_ID,
)
from refusal_direction_replication.qwen_jlens_runtime import run_qwen_jlens
from refusal_direction_replication.run_progress import EVENTS_FILENAME, STATUS_FILENAME


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "a" * 40
DOSE_RUN = SOURCE_DOSE_RUN_ID
PILOT_RUN = "c0818150-f834-4692-b271-4e10a97eb57e"
PILOT_COMMIT = "2a2eb714246174e4f627fcd0061c61143a4e8276"
DOSE_COMMIT = "10587e1db6ecf53f4e422c1c4a351edfd3647204"


class CountingTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = list(text)
        if add_special_tokens:
            return ["<s>", *tokens]
        return tokens

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 128):
        ids = [1] + [2] * min(max_length - 1, max(20, len(text) // 4))
        return SimpleNamespace(input_ids=torch.tensor([ids[:max_length]]))


class FakeLens:
    def __init__(self, jacobians, *, n_prompts: int, d_model: int) -> None:
        self.jacobians = {int(layer): tensor.float() for layer, tensor in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.saved_path: str | None = None
        self.saved_dtype = None

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        self.saved_path = str(path)
        self.saved_dtype = dtype
        torch.save(
            {
                "J": {layer: tensor.to(dtype) for layer, tensor in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> FakeLens:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(payload["J"], n_prompts=payload["n_prompts"], d_model=payload["d_model"])

    def apply(
        self,
        model,
        prompt: str,
        *,
        layers=None,
        positions=None,
        max_seq_len: int = 128,
        use_jacobian: bool = True,
    ):
        layers = list(self.source_layers if layers is None else layers)
        n_pos = 3 if positions is None else len(list(positions))
        vocab = 8
        lens_logits = {}
        for layer in layers:
            logits = torch.zeros(n_pos, vocab)
            target_index = 1 if use_jacobian else 2
            logits[:, target_index] = 3.0
            logits[:, 0] = 0.5
            lens_logits[layer] = logits
        model_logits = torch.zeros(n_pos, vocab)
        model_logits[:, 1] = 4.0
        input_ids = torch.arange(n_pos + 1).unsqueeze(0)
        return lens_logits, model_logits, input_ids


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row(index: int) -> dict:
    body = "print(1)\n" + "x = 1\n" * 80
    doc = f"Helper {index}"
    return {
        "repository_name": "org/repo",
        "func_path_in_repository": f"mod_{index}.py",
        "func_name": f"f_{index}",
        "whole_func_string": f"def f_{index}():\n    \"\"\"{doc}\"\"\"\n{body}",
        "func_code_string": body,
        "func_documentation_string": doc,
        "language": "python",
        "split_name": "train",
    }


def write_tiny_jlens_config(root: Path) -> dict:
    config = load_config(root, profile=JLENS_PROFILE)
    config = dict(config)
    config["n_layers"] = 4
    config["d_model"] = 8
    config["source_layers"] = [0, 1, 2]
    config["target_layer"] = 3
    (root / "configs" / JLENS_PROFILE).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return config


def write_source_dose(root: Path, *, status: str = STATUS_PASS) -> None:
    shutil.copytree(ROOT / "configs", root / "configs")
    write_tiny_jlens_config(root)
    results = root / "results" / "qwen25_coder_7b_dose"
    results.mkdir(parents=True)
    _write_json(
        results / "dose_followup_evaluation.json",
        {
            "status": status,
            "operational_success": True,
            "run_id": DOSE_RUN,
            "implementation_commit": DOSE_COMMIT,
            "model_revision": QWEN_REVISION,
            "source_pilot_run_id": PILOT_RUN,
            "source_pilot_implementation_commit": PILOT_COMMIT,
        },
    )
    original = root / "results" / "qwen25_coder_7b"
    original.mkdir(parents=True)
    _write_json(
        original / "pilot_evaluation.json",
        {
            "status": "causal_gate_failed",
            "run_id": PILOT_RUN,
            "implementation_commit": PILOT_COMMIT,
        },
    )


def fake_jacobian(model, prompt, source_layers, **kwargs):
    d_model = int(getattr(model, "d_model", 8))
    jacobians = {
        int(layer): torch.eye(d_model) for layer in source_layers
    }
    return jacobians, 24, 7


def install_runtime_fakes(monkeypatch, *, dim_batch_ok: int = 32, probe_seconds: float = 1.0):
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    calls = {"released": 0, "dim_batches": []}

    def release() -> None:
        calls["released"] += 1

    clock_state = {"t": 0.0}

    def clock() -> float:
        clock_state["t"] += probe_seconds / 2
        return clock_state["t"]

    def runner_factory(model):
        def run(dim_batch: int) -> None:
            calls["dim_batches"].append(dim_batch)
            if dim_batch > dim_batch_ok:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            fake_jacobian(model, "probe", [0, 1, 2], dim_batch=dim_batch)

        return run

    return calls, release, clock, runner_factory


def test_runtime_fits_tiny_corpus_and_writes_compact_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    calls, release, clock, _factory = install_runtime_fakes(monkeypatch)
    rows = [_row(index) for index in range(130)]
    model = SimpleNamespace(n_layers=4, d_model=8, config=SimpleNamespace(_commit_hash=QWEN_REVISION))
    tokenizer = CountingTokenizer()

    payload = run_qwen_jlens(
        tmp_path,
        run_id="jlens-run-1",
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (model, tokenizer),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=fake_jacobian,
        lens_cls=FakeLens,
        release_resources=release,
        clock=clock,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    results = tmp_path / "results" / "qwen25_coder_7b_jlens"
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    for name in (
        "00_jlens_prepare.json",
        "01_jlens_probe.json",
        "02_jlens_fit.json",
        "03_jlens_holdout.json",
        "jlens_evaluation.json",
        "execution_manifest.json",
    ):
        assert (results / name).is_file(), name
    prepare = json.loads((results / "00_jlens_prepare.json").read_text(encoding="utf-8"))
    assert prepare["results"]["fit_count"] == 100
    assert prepare["results"]["holdout_count"] == 20
    dumped = json.dumps(prepare)
    assert "Helper " not in dumped
    assert "print(1)" not in dumped
    probe = json.loads((results / "01_jlens_probe.json").read_text(encoding="utf-8"))
    assert probe["results"]["selected_dim_batch"] == 32
    assert probe["results"]["probe_order"] == [32, 16, 8]
    holdout = json.loads((results / "03_jlens_holdout.json").read_text(encoding="utf-8"))
    assert holdout["results"]["n_items"] == 20
    assert "jlens" in holdout["results"]["overall"]
    assert "logit_lens" in holdout["results"]["overall"]
    status = json.loads((artifacts / STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["current_stage"] == "holdout"
    assert status["total_units"] == 20
    assert status["completed_units"] == 20
    assert status["state"] == "completed"
    events = [
        json.loads(line)
        for line in (artifacts / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = {item.get("event") for item in events}
    assert "probe_choice" in names
    assert "checkpoint" in names
    lens_path = artifacts / FINAL_LENS_FILENAME
    assert lens_path.is_file()
    loaded = FakeLens.load(str(lens_path))
    assert loaded.n_prompts == 100
    assert loaded.d_model == 8
    assert calls["released"] >= 1
    original_pilot = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert original_pilot["status"] == "causal_gate_failed"
    dose = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert dose["status"] == "pass"
    assert dose["run_id"] == DOSE_RUN


def test_runtime_falls_back_dim_batch_then_aborts_on_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    rows = [_row(index) for index in range(130)]
    model = SimpleNamespace(n_layers=4, d_model=8, config=SimpleNamespace(_commit_hash=QWEN_REVISION))
    released = {"n": 0}

    def oom_then_ok(model_obj, prompt, source_layers, **kwargs):
        if kwargs.get("dim_batch", 8) > 16:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return fake_jacobian(model_obj, prompt, source_layers, **kwargs)

    payload = run_qwen_jlens(
        tmp_path,
        run_id="jlens-run-2",
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (model, CountingTokenizer()),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=oom_then_ok,
        lens_cls=FakeLens,
        release_resources=lambda: released.__setitem__("n", released["n"] + 1),
        clock=lambda: 0.0,
        probe_clock=iter([0.0, 500.0, 500.0, 1000.0]).__next__,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "aborted"
    assert payload["abort_gate"] == "projected_runtime"
    probe = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b_jlens" / "01_jlens_probe.json").read_text(
            encoding="utf-8"
        )
    )
    assert probe["results"]["selected_dim_batch"] == 16
    assert not (tmp_path / "results" / "qwen25_coder_7b_jlens" / "03_jlens_holdout.json").exists()
    assert released["n"] >= 1
    evaluation = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b_jlens" / "jlens_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["corpus_hash"]
    assert evaluation["config_hash"]
    prepare = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b_jlens" / "00_jlens_prepare.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["corpus_hash"] == prepare["results"]["corpus_hash"]
    assert evaluation["config_hash"] == prepare["results"]["config_hash"]
    from refusal_direction_replication.qwen_jlens_validate import validate_qwen_jlens

    checks = validate_qwen_jlens(tmp_path)
    failed = [name for name, passed in checks.items() if not passed]
    assert failed == ["original_pilot_validator", "original_dose_validator"]


def test_runtime_resumes_matching_checkpoint_and_rejects_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime
    from refusal_direction_replication.qwen_jlens import (
        config_hash,
        corpus_hash,
        write_fit_checkpoint,
        build_codesearchnet_corpus,
    )

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    rows = [_row(index) for index in range(130)]
    config = load_config(tmp_path, profile=JLENS_PROFILE)
    tokenizer = CountingTokenizer()
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=tokenizer,
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=config["dataset_revision"],
    )
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    identity = {
        "source_layers": [0, 1, 2],
        "target_layer": 3,
        "skip_first": 16,
        "corpus_hash": corpus_hash(corpus),
        "config_hash": config_hash(config),
        "model_revision": QWEN_REVISION,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "run_id": "jlens-run-3",
    }
    write_fit_checkpoint(
        artifacts / "jlens_fit_checkpoint.pt",
        jacobian_sum={layer: torch.eye(8) for layer in (0, 1, 2)},
        n_done=40,
        next_idx=40,
        dim_batch=32,
        **identity,
    )
    jacobian_calls: list[int] = []

    def counting_jacobian(model, prompt, source_layers, **kwargs):
        jacobian_calls.append(1)
        return fake_jacobian(model, prompt, source_layers, **kwargs)

    model = SimpleNamespace(n_layers=4, d_model=8, config=SimpleNamespace(_commit_hash=QWEN_REVISION))
    payload = run_qwen_jlens(
        tmp_path,
        run_id="jlens-run-3",
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (model, tokenizer),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=counting_jacobian,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    assert len(jacobian_calls) == 60
    stale = tmp_path / "stale"
    write_source_dose(stale)
    (stale / "configs" / JLENS_PROFILE).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    stale_art = stale / "artifacts" / "qwen25_coder_7b_jlens"
    stale_art.mkdir(parents=True)
    stale_identity = dict(identity)
    stale_identity["run_id"] = "other-run"
    write_fit_checkpoint(
        stale_art / "jlens_fit_checkpoint.pt",
        jacobian_sum={layer: torch.eye(8) for layer in (0, 1, 2)},
        n_done=40,
        next_idx=40,
        dim_batch=32,
        **stale_identity,
    )
    with pytest.raises(ValueError, match="run_id"):
        run_qwen_jlens(
            stale,
            run_id="jlens-run-3",
            load_rows=lambda **_kwargs: rows,
            load_model=lambda **_kwargs: (model, tokenizer),
            wrap_model=lambda hf_model, tok: hf_model,
            jacobian_for_prompt_fn=fake_jacobian,
            lens_cls=FakeLens,
            release_resources=lambda: None,
            clock=lambda: 0.0,
            assert_contract=lambda: None,
        )


def test_runtime_aborts_when_dim_batch_eight_ooms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)

    def always_oom(model_obj, prompt, source_layers, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    payload = run_qwen_jlens(
        tmp_path,
        run_id="jlens-run-oom",
        load_rows=lambda **_kwargs: [_row(index) for index in range(130)],
        load_model=lambda **_kwargs: (
            SimpleNamespace(n_layers=4, d_model=8, config=SimpleNamespace(_commit_hash=QWEN_REVISION)),
            CountingTokenizer(),
        ),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=always_oom,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "aborted"
    assert payload["abort_gate"] == "dim_batch_oom"
    results = tmp_path / "results" / "qwen25_coder_7b_jlens"
    probe = json.loads((results / "01_jlens_probe.json").read_text(encoding="utf-8"))
    assert [item["dim_batch"] for item in probe["results"]["attempts"]] == [32, 16, 8]
    assert all(item["outcome"] == "cuda_oom" for item in probe["results"]["attempts"])
    assert (results / "execution_manifest.json").is_file()
    assert (results / "jlens_evaluation.json").is_file()
    manifest = json.loads((results / "execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"]["status"] == "aborted"


def test_default_load_model_drops_pipeline_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    from refusal_direction_replication import qwen_jlens_runtime as runtime
    from refusal_direction_replication import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "_load_model_for_config",
        lambda config: ("model", "tokenizer", "bfloat16"),
    )
    model, tokenizer = runtime.default_load_model({"dtype": "bfloat16"})
    assert model == "model"
    assert tokenizer == "tokenizer"


def test_runtime_requires_passing_source_dose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path, status="causal_gate_failed")
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    with pytest.raises(ValueError, match="source dose"):
        run_qwen_jlens(
            tmp_path,
            run_id="jlens-run-4",
            load_rows=lambda **_kwargs: [_row(0)],
            load_model=lambda **_kwargs: (SimpleNamespace(), CountingTokenizer()),
            wrap_model=lambda hf_model, tok: hf_model,
            jacobian_for_prompt_fn=fake_jacobian,
            lens_cls=FakeLens,
            assert_contract=lambda: None,
        )


def test_runtime_records_resolved_revision_lens_hash_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    install_runtime_fakes(monkeypatch)
    observed = QWEN_REVISION
    model = SimpleNamespace(
        n_layers=4,
        d_model=8,
        config=SimpleNamespace(_commit_hash=observed),
    )
    payload = run_qwen_jlens(
        tmp_path,
        run_id="jlens-run-rev",
        load_rows=lambda **_kwargs: [_row(index) for index in range(130)],
        load_model=lambda **_kwargs: (model, CountingTokenizer()),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=fake_jacobian,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    results = tmp_path / "results" / "qwen25_coder_7b_jlens"
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    config = load_config(tmp_path, profile=JLENS_PROFILE)
    prepare = json.loads((results / "00_jlens_prepare.json").read_text(encoding="utf-8"))
    fit = json.loads((results / "02_jlens_fit.json").read_text(encoding="utf-8"))
    assert prepare["provenance"]["resolved_model_revision"] == observed
    assert prepare["provenance"]["requested_model_revision"] == QWEN_REVISION
    assert prepare["provenance"]["dataset_repo_id"] == config["dataset_repo_id"]
    assert prepare["provenance"]["dataset_config"] == "python"
    assert prepare["provenance"]["dataset_split"] == "train"
    assert prepare["provenance"]["dataset_revision"] == config["dataset_revision"]
    assert prepare["results"]["config_hash"] == fit["results"]["config_hash"]
    lens_path = artifacts / FINAL_LENS_FILENAME
    assert fit["results"]["lens_sha256"] == sha256_file(lens_path)
    assert fit["results"]["lens_bytes"] == lens_path.stat().st_size
    assert fit["results"]["corpus_hash"]
    assert fit["results"]["config_hash"]
    assert fit["results"]["jlens_upstream_commit"]
    assert fit["results"]["implementation_commit"]
    assert fit["results"]["run_id"] == "jlens-run-rev"
    evaluation = json.loads((results / "jlens_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["lens_sha256"] == fit["results"]["lens_sha256"]
    assert evaluation["lens_bytes"] == lens_path.stat().st_size
    assert evaluation["model_revision"] == observed
    holdout = json.loads((results / "03_jlens_holdout.json").read_text(encoding="utf-8"))
    assert "by_item" in holdout["results"]
    assert len(holdout["results"]["holdout_ids"]) == 20
    events = [
        json.loads(line)
        for line in (artifacts / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [item.get("stage") for item in events if item.get("event") == "initialized"]
    assert stages == ["prepare", "probe", "fit", "holdout"]
    assert any(item.get("event") == "stage_completed" for item in events)
    status = json.loads((artifacts / STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["current_stage"] == "holdout"
    assert status["state"] == "completed"
    assert status["fit_completed_units"] == 100
    assert status["fit_total_units"] == 100


def test_release_compute_runs_if_release_resources_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    install_runtime_fakes(monkeypatch)
    compute = {"n": 0}
    monkeypatch.setattr(
        runtime,
        "release_compute_resources",
        lambda *args, **kwargs: compute.__setitem__("n", compute["n"] + 1),
    )

    def boom() -> None:
        raise RuntimeError("release boom")

    with pytest.raises(RuntimeError, match="release boom"):
        run_qwen_jlens(
            tmp_path,
            run_id="jlens-run-release",
            load_rows=lambda **_kwargs: [_row(index) for index in range(130)],
            load_model=lambda **_kwargs: (
                SimpleNamespace(
                    n_layers=4,
                    d_model=8,
                    config=SimpleNamespace(_commit_hash=QWEN_REVISION),
                ),
                CountingTokenizer(),
            ),
            wrap_model=lambda hf_model, tok: hf_model,
            jacobian_for_prompt_fn=fake_jacobian,
            lens_cls=FakeLens,
            release_resources=boom,
            clock=lambda: 0.0,
            assert_contract=lambda: None,
        )
    assert compute["n"] == 1


def test_readout_rejects_nonfinite_model_logits() -> None:
    from refusal_direction_replication.qwen_jlens import CorpusItem
    from refusal_direction_replication.qwen_jlens_runtime import run_holdout_readout

    class BadLens:
        source_layers = [0]

        def apply(self, model, prompt, *, layers=None, positions=None, **kwargs):
            n_pos = len(list(positions))
            logits = torch.zeros(n_pos, 4)
            model_logits = torch.zeros(n_pos, 4)
            model_logits[0, 0] = float("nan")
            return {0: logits}, model_logits, torch.arange(n_pos + 1).unsqueeze(0)

    item = CorpusItem(
        example_id="h0",
        content_hash="a" * 64,
        token_count=24,
        fit_text="x" * 40,
        source_index=0,
    )
    with pytest.raises(ValueError, match="finite"):
        run_holdout_readout(
            lens=BadLens(),
            model=SimpleNamespace(encode=lambda text, max_length=128: torch.arange(24).unsqueeze(0)),
            tokenizer=CountingTokenizer(),
            items=[item],
            source_layers=[0],
            skip_first=16,
            max_seq_len=128,
        )


def test_runtime_rejects_dataset_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    config_path = tmp_path / "configs" / JLENS_PROFILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_revision"] = "deadbeef" * 5
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    install_runtime_fakes(monkeypatch)
    with pytest.raises(ValueError, match="dataset identity"):
        run_qwen_jlens(
            tmp_path,
            run_id="jlens-run-dataset",
            load_rows=lambda **_kwargs: [_row(index) for index in range(130)],
            load_model=lambda **_kwargs: (
                SimpleNamespace(
                    n_layers=4,
                    d_model=8,
                    config=SimpleNamespace(_commit_hash=QWEN_REVISION),
                ),
                CountingTokenizer(),
            ),
            wrap_model=lambda hf_model, tok: hf_model,
            jacobian_for_prompt_fn=fake_jacobian,
            lens_cls=FakeLens,
            release_resources=lambda: None,
            clock=lambda: 0.0,
            assert_contract=lambda: None,
        )


def test_runtime_rejects_resolved_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    install_runtime_fakes(monkeypatch)
    with pytest.raises(ValueError, match="resolved model revision"):
        run_qwen_jlens(
            tmp_path,
            run_id="jlens-run-mismatch",
            load_rows=lambda **_kwargs: [_row(index) for index in range(130)],
            load_model=lambda **_kwargs: (
                SimpleNamespace(
                    n_layers=4,
                    d_model=8,
                    config=SimpleNamespace(_commit_hash="b" * 40),
                ),
                CountingTokenizer(),
            ),
            wrap_model=lambda hf_model, tok: hf_model,
            jacobian_for_prompt_fn=fake_jacobian,
            lens_cls=FakeLens,
            release_resources=lambda: None,
            clock=lambda: 0.0,
            assert_contract=lambda: None,
        )


def test_runtime_resumes_run_id_from_checkpoint_without_explicit_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime
    from refusal_direction_replication.qwen_jlens import (
        build_codesearchnet_corpus,
        config_hash,
        corpus_hash,
        write_fit_checkpoint,
    )

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    rows = [_row(index) for index in range(130)]
    config = load_config(tmp_path, profile=JLENS_PROFILE)
    tokenizer = CountingTokenizer()
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=tokenizer,
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=config["dataset_revision"],
    )
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    write_fit_checkpoint(
        artifacts / "jlens_fit_checkpoint.pt",
        jacobian_sum={layer: torch.eye(8) for layer in (0, 1, 2)},
        n_done=40,
        next_idx=40,
        dim_batch=32,
        source_layers=[0, 1, 2],
        target_layer=3,
        skip_first=16,
        corpus_hash=corpus_hash(corpus),
        config_hash=config_hash(config),
        model_revision=QWEN_REVISION,
        implementation_commit=IMPLEMENTATION_COMMIT,
        run_id="checkpoint-run-id",
    )
    jacobian_calls: list[int] = []

    def counting_jacobian(model, prompt, source_layers, **kwargs):
        jacobian_calls.append(1)
        return fake_jacobian(model, prompt, source_layers, **kwargs)

    payload = run_qwen_jlens(
        tmp_path,
        run_id=None,
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (
            SimpleNamespace(
                n_layers=4,
                d_model=8,
                config=SimpleNamespace(_commit_hash=QWEN_REVISION),
            ),
            tokenizer,
        ),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=counting_jacobian,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    assert payload["run_id"] == "checkpoint-run-id"
    assert len(jacobian_calls) == 60


def test_fetch_sync_bootstrap_resume_without_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from refusal_direction_replication.qwen_colab_transport import (
        REMOTE_JLENS_CHECKPOINT_STAGING,
        REMOTE_JLENS_EVENTS_STAGING,
        REMOTE_JLENS_STATUS_STAGING,
        atomic_install_jlens_resume,
        extract_qwen_jlens_archive,
        optional_local_jlens_resume_bundle,
        pack_qwen_jlens_results,
    )
    from refusal_direction_replication.qwen_jlens import (
        build_codesearchnet_corpus,
        config_hash,
        corpus_hash,
        write_fit_checkpoint,
    )

    session_a = tmp_path / "session_a"
    write_source_dose(session_a)
    rows = [_row(index) for index in range(130)]
    config = load_config(session_a, profile=JLENS_PROFILE)
    tokenizer = CountingTokenizer()
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=tokenizer,
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=config["dataset_revision"],
    )
    artifacts_a = session_a / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts_a.mkdir(parents=True)
    write_fit_checkpoint(
        artifacts_a / "jlens_fit_checkpoint.pt",
        jacobian_sum={layer: torch.eye(8) for layer in (0, 1, 2)},
        n_done=40,
        next_idx=40,
        dim_batch=32,
        source_layers=[0, 1, 2],
        target_layer=3,
        skip_first=16,
        corpus_hash=corpus_hash(corpus),
        config_hash=config_hash(config),
        model_revision=QWEN_REVISION,
        implementation_commit=IMPLEMENTATION_COMMIT,
        run_id="resume-e2e-run",
    )
    (artifacts_a / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": JLENS_PROFILE,
                "run_id": "resume-e2e-run",
                "current_stage": "fit",
                "state": "running",
                "completed_units": 40,
                "total_units": 100,
                "fit_completed_units": 40,
                "fit_total_units": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = [
        {"event": "initialized", "run_id": "resume-e2e-run", "stage": "prepare"},
        {"event": "initialized", "run_id": "resume-e2e-run", "stage": "probe"},
        {"event": "probe_choice", "run_id": "resume-e2e-run", "dim_batch": 32},
        {"event": "initialized", "run_id": "resume-e2e-run", "stage": "fit"},
    ]
    events.extend(
        {
            "event": "checkpoint",
            "run_id": "resume-e2e-run",
            "n_done": index,
            "stage": "fit",
        }
        for index in range(1, 41)
    )
    (artifacts_a / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )
    archive = tmp_path / "fetched.tar.gz"
    pack_qwen_jlens_results(session_a, archive)
    fetched = tmp_path / "fetched_local"
    fetched.mkdir()
    extract_qwen_jlens_archive(archive, fetched)
    bundle = optional_local_jlens_resume_bundle(fetched)
    remotes = {remote for _path, remote in bundle}
    assert REMOTE_JLENS_CHECKPOINT_STAGING in remotes
    assert REMOTE_JLENS_STATUS_STAGING in remotes
    assert REMOTE_JLENS_EVENTS_STAGING in remotes
    staging = tmp_path / "remote_staging"
    staging.mkdir()
    staged_paths = {}
    for path, remote in bundle:
        dest = staging / Path(remote).name
        dest.write_bytes(path.read_bytes())
        staged_paths[remote] = dest
    session_b = tmp_path / "session_b"
    write_source_dose(session_b)
    atomic_install_jlens_resume(
        checkpoint=staged_paths[REMOTE_JLENS_CHECKPOINT_STAGING],
        status=staged_paths[REMOTE_JLENS_STATUS_STAGING],
        events=staged_paths[REMOTE_JLENS_EVENTS_STAGING],
        project=session_b,
    )
    from refusal_direction_replication import qwen_jlens_runtime as runtime

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    jacobian_calls: list[int] = []

    def counting_jacobian(model, prompt, source_layers, **kwargs):
        jacobian_calls.append(1)
        return fake_jacobian(model, prompt, source_layers, **kwargs)

    payload = run_qwen_jlens(
        session_b,
        run_id=None,
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (
            SimpleNamespace(
                n_layers=4,
                d_model=8,
                config=SimpleNamespace(_commit_hash=QWEN_REVISION),
            ),
            tokenizer,
        ),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=counting_jacobian,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    assert payload["run_id"] == "resume-e2e-run"
    assert len(jacobian_calls) == 60
    events_b = (
        session_b / "artifacts" / "qwen25_coder_7b_jlens" / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert events_b.count('"event": "checkpoint"') >= 100


def test_runtime_reconciles_lagged_status_and_events_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_dose(tmp_path)
    from refusal_direction_replication import qwen_jlens_runtime as runtime
    from refusal_direction_replication.qwen_jlens import (
        build_codesearchnet_corpus,
        config_hash,
        corpus_hash,
        write_fit_checkpoint,
    )

    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)
    rows = [_row(index) for index in range(130)]
    config = load_config(tmp_path, profile=JLENS_PROFILE)
    tokenizer = CountingTokenizer()
    corpus = build_codesearchnet_corpus(
        rows,
        tokenizer=tokenizer,
        fit_count=100,
        holdout_count=20,
        min_tokens=128,
        dataset_revision=config["dataset_revision"],
    )
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    artifacts.mkdir(parents=True)
    write_fit_checkpoint(
        artifacts / "jlens_fit_checkpoint.pt",
        jacobian_sum={layer: torch.eye(8) for layer in (0, 1, 2)},
        n_done=40,
        next_idx=40,
        dim_batch=32,
        source_layers=[0, 1, 2],
        target_layer=3,
        skip_first=16,
        corpus_hash=corpus_hash(corpus),
        config_hash=config_hash(config),
        model_revision=QWEN_REVISION,
        implementation_commit=IMPLEMENTATION_COMMIT,
        run_id="lagged-resume-run",
    )
    (artifacts / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": JLENS_PROFILE,
                "run_id": "lagged-resume-run",
                "current_stage": "fit",
                "state": "running",
                "completed_units": 39,
                "total_units": 100,
                "fit_completed_units": 39,
                "fit_total_units": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lagged_events = [
        {"event": "initialized", "run_id": "lagged-resume-run", "stage": "prepare"},
        {"event": "initialized", "run_id": "lagged-resume-run", "stage": "probe"},
        {"event": "probe_choice", "run_id": "lagged-resume-run", "dim_batch": 32},
        {"event": "initialized", "run_id": "lagged-resume-run", "stage": "fit"},
    ]
    lagged_events.extend(
        {
            "event": "checkpoint",
            "run_id": "lagged-resume-run",
            "n_done": index,
            "stage": "fit",
        }
        for index in range(1, 40)
    )
    (artifacts / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in lagged_events),
        encoding="utf-8",
    )
    jacobian_calls: list[int] = []

    def counting_jacobian(model, prompt, source_layers, **kwargs):
        jacobian_calls.append(1)
        return fake_jacobian(model, prompt, source_layers, **kwargs)

    payload = run_qwen_jlens(
        tmp_path,
        run_id=None,
        load_rows=lambda **_kwargs: rows,
        load_model=lambda **_kwargs: (
            SimpleNamespace(
                n_layers=4,
                d_model=8,
                config=SimpleNamespace(_commit_hash=QWEN_REVISION),
            ),
            tokenizer,
        ),
        wrap_model=lambda hf_model, tok: hf_model,
        jacobian_for_prompt_fn=counting_jacobian,
        lens_cls=FakeLens,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        assert_contract=lambda: None,
    )
    assert payload["status"] == "pass"
    assert payload["run_id"] == "lagged-resume-run"
    assert len(jacobian_calls) == 60
    events = [
        json.loads(line)
        for line in (artifacts / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recoveries = [
        item
        for item in events
        if item.get("event") == "resume_recovered" and item.get("n_done") == 40
    ]
    assert len(recoveries) == 1
    assert any(
        item.get("event") == "checkpoint" and item.get("n_done") == 40 for item in events
    )
    from refusal_direction_replication.run_progress import RunProgressStore

    first = runtime.reconcile_jlens_resume(
        RunProgressStore(artifacts, profile=JLENS_PROFILE, run_id="lagged-resume-run"),
        n_done=40,
        dim_batch=32,
    )
    assert first["recovered"] is False
    recoveries_after = [
        item
        for item in (
            json.loads(line)
            for line in (artifacts / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if item.get("event") == "resume_recovered" and item.get("n_done") == 40
    ]
    assert len(recoveries_after) == 1
