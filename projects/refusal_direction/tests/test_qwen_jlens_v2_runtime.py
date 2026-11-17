from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import weakref

import pytest
import torch

from refusal_direction_replication import qwen_jlens_runtime as runtime
from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_jlens import (
    JLENS_PROFILE,
    JLENS_V2_PROFILE,
    QWEN_REVISION,
    V2_FINAL_LENS_FILENAME,
    load_fit_checkpoint,
    validate_checkpoint_payload,
)
from refusal_direction_replication.qwen_jlens_runtime import run_qwen_jlens
from refusal_direction_replication.run_progress import EVENTS_FILENAME, STATUS_FILENAME
from test_qwen_jlens_runtime import (
    IMPLEMENTATION_COMMIT,
    CountingTokenizer,
    FakeLens,
    _row,
    fake_jacobian,
    write_source_dose,
)

V2_RESULTS = Path("results/v2")
V2_ARTIFACTS = Path("artifacts/v2")


def _rows(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        row = _row(index)
        body = "print(1)\n" + "x = 1\n" * 120
        row["func_code_string"] = body
        row["whole_func_string"] = f"def f_{index}():\n    \"\"\"Helper {index}\"\"\"\n{body}"
        rows.append(row)
    return rows


def _write_v2_fixture(
    root: Path,
    *,
    candidates: list[int],
    holdout_count: int,
    max_seq_len: int,
    max_projected_hours: float,
) -> Path:
    write_source_dose(root)
    config = dict(load_config(root, profile=JLENS_V2_PROFILE))
    config.update(
        {
            "n_layers": 4,
            "d_model": 8,
            "source_layers": [0, 1, 2],
            "target_layer": 3,
            "skip_first": 2,
            "max_seq_len": max_seq_len,
            "min_tokens_before_truncation": 512,
            "holdout_count": holdout_count,
            "fit_count_candidates": candidates,
            "max_fit_count": max(candidates),
            "max_projected_hours": max_projected_hours,
            "results_dir": str(V2_RESULTS),
            "artifacts_dir": str(V2_ARTIFACTS),
            "exclusion_hashes_path": "artifacts/v2/exclusion_hashes.txt",
        }
    )
    path = root / "configs" / JLENS_V2_PROFILE
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _write_frozen_v1_lens(root, config)
    return path


V1_LENS_N_PROMPTS = 7


class V1DistinguishingLens(FakeLens):
    """FakeLens whose Jacobian readout ranks the target 2nd when it is the v1 lens."""

    def apply(self, model, prompt, **kwargs):
        lens_logits, model_logits, ids = super().apply(model, prompt, **kwargs)
        if kwargs.get("use_jacobian", True) and self.n_prompts == V1_LENS_N_PROMPTS:
            for logits in lens_logits.values():
                logits[:, 0] = 5.0  # outranks the target at index 1
        return lens_logits, model_logits, ids


def _write_frozen_v1_lens(root: Path, config: dict[str, Any]) -> Path:
    path = root / str(config["source_v1_lens"])
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(layer) for layer in config["source_layers"]]
    FakeLens(
        {layer: torch.eye(int(config["d_model"])) * (layer + 1) for layer in layers},
        n_prompts=V1_LENS_N_PROMPTS,
        d_model=int(config["d_model"]),
    ).save(str(path))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _tiny_model() -> tuple[Any, Any]:
    model = SimpleNamespace(
        n_layers=4,
        d_model=8,
        config=SimpleNamespace(_commit_hash=QWEN_REVISION),
    )
    return model, CountingTokenizer()


def _tiny_wrapper(hf_model: Any, _tokenizer: Any) -> Any:
    return hf_model


def _clock_for_seconds(seconds: float):
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += seconds
        return state["t"]

    return clock


def _recording_jacobian(calls: list[dict[str, Any]], *, fail_on_call: int | None = None):
    def jacobian(model, prompt, source_layers, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("simulated interruption")
        return fake_jacobian(model, prompt, source_layers, **kwargs)

    return jacobian


def _run(
    root: Path,
    profile: str,
    *,
    rows: list[dict[str, Any]],
    jacobian_fn,
    probe_seconds: float = 1.0,
    run_id: str | None = "v2-run",
    lens_cls: Any = FakeLens,
    expected_v1_lens_sha256: Any = "fixture",
) -> dict[str, Any]:
    if expected_v1_lens_sha256 == "fixture":
        # CPU fixtures pin the hash of the tiny frozen v1 lens they wrote; the
        # v1 profile has no source lens, so nothing is pinned for it.
        relative = _json(root / "configs" / profile).get("source_v1_lens")
        v1_file = root / str(relative) if relative else None
        expected_v1_lens_sha256 = (
            _sha256(v1_file) if v1_file is not None and v1_file.is_file() else None
        )
    return run_qwen_jlens(
        root,
        profile=profile,
        run_id=run_id,
        load_rows=lambda **_kwargs: rows,
        load_model=_tiny_model,
        wrap_model=_tiny_wrapper,
        jacobian_for_prompt_fn=jacobian_fn,
        lens_cls=lens_cls,
        release_resources=lambda: None,
        clock=lambda: 0.0,
        probe_clock=_clock_for_seconds(probe_seconds),
        assert_contract=lambda: None,
        expected_v1_lens_sha256=expected_v1_lens_sha256,
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(root: Path) -> list[dict[str, Any]]:
    path = root / V2_ARTIFACTS / EVENTS_FILENAME
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(autouse=True)
def _commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT)


def test_runtime_selects_fit_count_and_checkpoints_each_item(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    result = _run(
        tmp_path,
        config.name,
        rows=_rows(6),
        jacobian_fn=_recording_jacobian([]),
        probe_seconds=1.0,
    )
    assert result["status"] == "pass"
    assert result["n_fit"] == 4
    checkpoint = load_fit_checkpoint(tmp_path / V2_ARTIFACTS / "jlens_fit_checkpoint.pt")
    assert checkpoint["n_done"] == checkpoint["fit_count"] == 4
    assert (tmp_path / V2_ARTIFACTS / V2_FINAL_LENS_FILENAME).is_file()


def test_runtime_aborts_before_fit_when_every_candidate_exceeds_limit(
    tmp_path: Path,
) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=1,
    )
    calls: list[dict[str, Any]] = []
    result = _run(
        tmp_path,
        config.name,
        rows=_rows(6),
        jacobian_fn=_recording_jacobian(calls),
        probe_seconds=3600.0,
    )
    assert result["status"] == "aborted"
    assert result["abort_gate"] == "projected_runtime"
    assert result["n_fit"] is None
    assert result["n_holdout"] == 2
    assert len(calls) == 1  # the probe only
    assert not (tmp_path / V2_ARTIFACTS / "jlens_fit_checkpoint.pt").exists()
    assert not (tmp_path / V2_RESULTS / "02_jlens_fit.json").exists()
    probe = _json(tmp_path / V2_RESULTS / "01_jlens_probe.json")["results"]
    assert probe["selected_fit_count"] is None
    assert probe["exceeds_limit"] is True
    status = _json(tmp_path / V2_ARTIFACTS / STATUS_FILENAME)
    assert status["state"] == "aborted"
    assert status["abort_gate"] == "projected_runtime"


def test_resume_uses_stored_fit_count_and_finishes_remaining_items(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    rows = _rows(6)
    first_calls: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        # call 1 is the probe, calls 2-3 fit items 0-1, call 4 fails item 2.
        _run(
            tmp_path,
            config.name,
            rows=rows,
            jacobian_fn=_recording_jacobian(first_calls, fail_on_call=4),
            probe_seconds=1.0,
        )
    checkpoint_path = tmp_path / V2_ARTIFACTS / "jlens_fit_checkpoint.pt"
    interrupted = load_fit_checkpoint(checkpoint_path)
    assert interrupted["n_done"] == 2
    assert interrupted["fit_count"] == 4

    # A probe that would now select 2 (or abort) must not be consulted.
    resume_calls: list[dict[str, Any]] = []
    result = _run(
        tmp_path,
        config.name,
        rows=rows,
        jacobian_fn=_recording_jacobian(resume_calls),
        probe_seconds=10_000_000.0,
        run_id=None,
    )
    assert result["status"] == "pass"
    assert result["run_id"] == "v2-run"
    assert result["n_fit"] == 4
    assert len(resume_calls) == 2
    assert len(first_calls) == 4
    assert [item["prompt"] for item in resume_calls] == [
        rows[2]["whole_func_string"],
        rows[3]["whole_func_string"],
    ]
    final = load_fit_checkpoint(checkpoint_path)
    assert final["n_done"] == final["fit_count"] == 4
    probe_events = [item for item in _events(tmp_path) if item.get("event") == "probe_choice"]
    assert len(probe_events) == 1


def test_resume_after_probe_preserves_fit_count_before_first_fit(
    tmp_path: Path,
) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    rows = _rows(6)
    first_calls: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(
            tmp_path,
            config.name,
            rows=rows,
            jacobian_fn=_recording_jacobian(first_calls, fail_on_call=2),
            probe_seconds=1.0,
        )
    assert len(first_calls) == 2
    checkpoint_path = tmp_path / V2_ARTIFACTS / "jlens_fit_checkpoint.pt"
    seeded = load_fit_checkpoint(checkpoint_path)
    assert seeded["n_done"] == seeded["next_idx"] == 0
    assert seeded["fit_count"] == 4
    assert seeded["dim_batch"] == first_calls[0]["dim_batch"]

    resume_calls: list[dict[str, Any]] = []
    result = _run(
        tmp_path,
        config.name,
        rows=rows,
        jacobian_fn=_recording_jacobian(resume_calls),
        probe_seconds=10_000_000.0,
        run_id=None,
    )
    assert result["status"] == "pass"
    assert result["n_fit"] == 4
    assert len(resume_calls) == 4
    probe_events = [item for item in _events(tmp_path) if item.get("event") == "probe_choice"]
    assert len(probe_events) == 1


def test_resume_without_stored_fit_count_is_rejected(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    rows = _rows(6)
    with pytest.raises(RuntimeError):
        _run(
            tmp_path,
            config.name,
            rows=rows,
            jacobian_fn=_recording_jacobian([], fail_on_call=3),
        )
    checkpoint_path = tmp_path / V2_ARTIFACTS / "jlens_fit_checkpoint.pt"
    payload = load_fit_checkpoint(checkpoint_path)
    payload.pop("fit_count")
    torch.save(payload, checkpoint_path)
    calls: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="fit_count"):
        _run(
            tmp_path,
            config.name,
            rows=rows,
            jacobian_fn=_recording_jacobian(calls),
        )
    assert calls == []


def test_pool_is_built_with_max_fit_count_holdout_and_min_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=3,
        max_seq_len=8,
        max_projected_hours=12,
    )
    real = runtime.build_codesearchnet_corpus
    builds: list[dict[str, Any]] = []

    def spy(rows, **kwargs):
        builds.append(kwargs)
        return real(rows, **kwargs)

    monkeypatch.setattr(runtime, "build_codesearchnet_corpus", spy)
    rows = _rows(8)
    calls: list[dict[str, Any]] = []
    result = _run(
        tmp_path,
        config.name,
        rows=rows,
        jacobian_fn=_recording_jacobian(calls),
        probe_seconds=1.0,
    )
    assert result["status"] == "pass"
    assert len(builds) == 1
    assert builds[0]["fit_count"] == 4
    assert builds[0]["holdout_count"] == 3
    assert builds[0]["min_tokens"] == 512
    probe_call = calls[0]
    assert probe_call["prompt"] == rows[0]["whole_func_string"]
    assert probe_call["max_seq_len"] == 8
    assert all(item["max_seq_len"] == 8 for item in calls)


def test_v1_profile_still_reports_and_fits_100_and_20(tmp_path: Path) -> None:
    write_source_dose(tmp_path)
    calls: list[dict[str, Any]] = []
    result = _run(
        tmp_path,
        JLENS_PROFILE,
        rows=[_row(index) for index in range(130)],
        jacobian_fn=_recording_jacobian(calls),
        run_id="v1-run",
    )
    assert result["status"] == "pass"
    assert result["n_fit"] == 100
    assert result["n_holdout"] == 20
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_jlens"
    checkpoint = load_fit_checkpoint(artifacts / "jlens_fit_checkpoint.pt")
    assert checkpoint["n_done"] == checkpoint["fit_count"] == 100
    status = _json(artifacts / STATUS_FILENAME)
    assert status["fit_total_units"] == 100
    assert status["total_units"] == 20
    results = tmp_path / "results" / "qwen25_coder_7b_jlens"
    probe = _json(results / "01_jlens_probe.json")["results"]
    assert "selected_fit_count" not in probe
    assert len(calls) == 101  # one probe + 100 fit prompts


def test_selected_count_is_reported_everywhere_instead_of_constants(
    tmp_path: Path,
) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=3,
        max_seq_len=8,
        max_projected_hours=1,
    )
    result = _run(
        tmp_path,
        config.name,
        rows=_rows(8),
        jacobian_fn=_recording_jacobian([]),
        probe_seconds=1000.0,  # 2 -> 2000s fits in 3600s, 4 -> 4000s does not
    )
    assert result["status"] == "pass"
    assert result["n_fit"] == 2
    assert result["n_holdout"] == 3

    results = tmp_path / V2_RESULTS
    artifacts = tmp_path / V2_ARTIFACTS
    probe = _json(results / "01_jlens_probe.json")["results"]
    assert probe["selected_fit_count"] == 2
    assert probe["fit_count_candidates"] == [2, 4]
    assert probe["projected_seconds"] == 2000.0
    assert probe["projected_limit_seconds"] == 3600
    assert probe["exceeds_limit"] is False
    assert probe["max_seq_len"] == 8

    prepare = _json(results / "00_jlens_prepare.json")["results"]
    assert prepare["fit_count"] == 2
    assert prepare["holdout_count"] == 3

    fit = _json(results / "02_jlens_fit.json")["results"]
    assert fit["n_done"] == fit["n_fit"] == 2
    assert fit["lens"].endswith(V2_FINAL_LENS_FILENAME)

    holdout = _json(results / "03_jlens_holdout.json")["results"]
    assert holdout["n_items"] == 3

    evaluation = _json(results / "jlens_evaluation.json")
    assert evaluation["n_fit"] == 2
    assert evaluation["n_holdout"] == 3
    assert evaluation["corpus_hash"] == prepare["corpus_hash"]

    status = _json(artifacts / STATUS_FILENAME)
    assert status["fit_total_units"] == 2
    assert status["fit_completed_units"] == 2
    assert status["total_units"] == 3
    assert status["completed_units"] == 3

    events = _events(tmp_path)
    checkpoints = [item for item in events if item.get("event") == "checkpoint"]
    assert [item["n_done"] for item in checkpoints] == [1, 2]
    assert {item["total_units"] for item in checkpoints} == {2}
    fit_init = [
        item
        for item in events
        if item.get("event") == "initialized" and item.get("stage") == "fit"
    ]
    assert [item["total_units"] for item in fit_init] == [2]
    holdout_init = [
        item
        for item in events
        if item.get("event") == "initialized" and item.get("stage") == "holdout"
    ]
    assert [item["total_units"] for item in holdout_init] == [3]


def test_validate_checkpoint_payload_accepts_v2_count_above_100() -> None:
    payload = {
        "jacobian_sum": {layer: torch.eye(8) for layer in (0, 1, 2)},
        "n_done": 120,
        "next_idx": 120,
        "dim_batch": 8,
        "fit_count": 150,
    }
    validate_checkpoint_payload(
        payload,
        source_layers=[0, 1, 2],
        d_model=8,
        dim_batch=8,
        fit_count=150,
    )
    with pytest.raises(ValueError, match="fit_count"):
        validate_checkpoint_payload(
            payload,
            source_layers=[0, 1, 2],
            d_model=8,
            dim_batch=8,
        )


def _v1_path(root: Path) -> Path:
    config = load_config(root, profile=JLENS_V2_PROFILE)
    return root / str(config["source_v1_lens"])


def test_runtime_compares_v1_v2_and_logit_lens_on_shared_holdout(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=3,
        max_seq_len=8,
        max_projected_hours=12,
    )
    v1_path = _v1_path(tmp_path)
    v1_before = _sha256(v1_path)
    v1_mtime = v1_path.stat().st_mtime_ns
    siblings = sorted(path.name for path in v1_path.parent.iterdir())

    result = _run(
        tmp_path,
        config.name,
        rows=_rows(8),
        jacobian_fn=_recording_jacobian([]),
        probe_seconds=1.0,
        lens_cls=V1DistinguishingLens,
    )
    assert result["status"] == "pass"

    results = tmp_path / V2_RESULTS
    compare_path = results / "04_jlens_compare.json"
    compare = _json(compare_path)["results"]
    holdout = _json(results / "03_jlens_holdout.json")["results"]
    assert set(compare["methods"]) == {"jlens_v2", "jlens_v1", "logit_lens"}
    assert compare["shared_holdout_ids"] == holdout["holdout_ids"]
    assert len(compare["shared_holdout_ids"]) == 3
    overall = {name: block["overall"] for name, block in compare["methods"].items()}
    assert overall["jlens_v2"]["mean_rank"] == 1.0
    assert overall["jlens_v1"]["mean_rank"] == 2.0
    assert overall["logit_lens"]["mean_rank"] == holdout["overall"]["logit_lens"]["mean_rank"]
    assert overall["jlens_v2"] == holdout["overall"]["jlens"]
    assert compare["n_observations"] == overall["jlens_v2"]["n_observations"]
    assert len(compare["stability"]["halves"]) == 2
    assert compare["v1_lens"]["sha256"] == v1_before
    assert compare["v1_lens"]["n_prompts"] == V1_LENS_N_PROMPTS
    assert compare["v1_lens"]["path"] == _json(config)["source_v1_lens"]

    # Frozen v1 artifact: untouched content, mtime, and directory contents.
    assert _sha256(v1_path) == v1_before
    assert v1_path.stat().st_mtime_ns == v1_mtime
    assert sorted(path.name for path in v1_path.parent.iterdir()) == siblings

    # Ordinary v2 metrics stay in 03 and no raw text reaches the comparison.
    assert "by_item" in holdout
    serialized = compare_path.read_text(encoding="utf-8")
    assert "function-" not in serialized and "print(1)" not in serialized
    assert "fit_text" not in serialized and "decoded" not in serialized
    manifest = _json(results / "execution_manifest.json")["results"]
    assert "04_jlens_compare.json" in manifest["result_files"]


def test_runtime_rejects_missing_v1_lens_before_fitting(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    _v1_path(tmp_path).unlink()
    calls: list[dict[str, Any]] = []
    with pytest.raises(FileNotFoundError, match="v1 lens"):
        _run(
            tmp_path,
            config.name,
            rows=_rows(6),
            jacobian_fn=_recording_jacobian(calls),
        )
    assert calls == []
    assert not (tmp_path / V2_RESULTS / "04_jlens_compare.json").exists()


def test_runtime_rejects_v1_lens_with_other_layers_without_writing_compare(
    tmp_path: Path,
) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )
    v1_path = _v1_path(tmp_path)
    FakeLens({0: torch.eye(8)}, n_prompts=V1_LENS_N_PROMPTS, d_model=8).save(str(v1_path))
    v1_before = _sha256(v1_path)
    calls: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="source_layers"):
        _run(
            tmp_path,
            config.name,
            rows=_rows(6),
            jacobian_fn=_recording_jacobian(calls),
        )
    _assert_failed_before_probe(tmp_path, calls)
    assert _sha256(v1_path) == v1_before


def _assert_failed_before_probe(root: Path, calls: list[dict[str, Any]]) -> None:
    assert calls == []  # neither the probe nor any fit Jacobian ran
    assert not (root / V2_RESULTS / "01_jlens_probe.json").exists()
    assert not (root / V2_RESULTS / "02_jlens_fit.json").exists()
    assert not (root / V2_RESULTS / "04_jlens_compare.json").exists()
    assert not (root / V2_ARTIFACTS / "jlens_fit_checkpoint.pt").exists()
    assert not (root / V2_ARTIFACTS / V2_FINAL_LENS_FILENAME).exists()


def _v2_fixture(tmp_path: Path) -> Path:
    return _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=12,
    )


def test_runtime_rejects_wrong_v1_lens_hash_before_probe(tmp_path: Path) -> None:
    config = _v2_fixture(tmp_path)
    v1_path = _v1_path(tmp_path)
    v1_before = _sha256(v1_path)
    calls: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="sha256"):
        _run(
            tmp_path,
            config.name,
            rows=_rows(6),
            jacobian_fn=_recording_jacobian(calls),
            expected_v1_lens_sha256="0" * 64,
        )
    _assert_failed_before_probe(tmp_path, calls)
    assert _sha256(v1_path) == v1_before


def test_production_default_pins_canonical_frozen_lens_hash(tmp_path: Path) -> None:
    from refusal_direction_replication.qwen_lab_readout import FROZEN_LENS_SHA256

    assert runtime.FROZEN_LENS_SHA256 is FROZEN_LENS_SHA256
    config = _v2_fixture(tmp_path)
    assert _sha256(_v1_path(tmp_path)) != FROZEN_LENS_SHA256  # tiny fixture lens
    calls: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="sha256"):
        _run(
            tmp_path,
            config.name,
            rows=_rows(6),
            jacobian_fn=_recording_jacobian(calls),
            expected_v1_lens_sha256=None,  # no override: canonical hash applies
        )
    _assert_failed_before_probe(tmp_path, calls)


class _TrackingLens(FakeLens):
    loaded: list[Any] = []

    @classmethod
    def load(cls, path: str):
        instance = super().load(path)
        cls.loaded.append(weakref.ref(instance))
        return instance


def test_preflight_v1_lens_is_loaded_then_released_before_probe(tmp_path: Path) -> None:
    config = _v2_fixture(tmp_path)
    _TrackingLens.loaded = []
    seen: list[tuple[int, int]] = []
    inner = _recording_jacobian([])

    def jacobian(model, prompt, source_layers, **kwargs):
        if not seen:  # first Jacobian call is the probe
            gc.collect()
            seen.append(
                (
                    len(_TrackingLens.loaded),
                    sum(ref() is not None for ref in _TrackingLens.loaded),
                )
            )
        return inner(model, prompt, source_layers, **kwargs)

    result = _run(
        tmp_path,
        config.name,
        rows=_rows(6),
        jacobian_fn=jacobian,
        lens_cls=_TrackingLens,
    )
    assert result["status"] == "pass"
    assert seen == [(1, 0)]  # loaded once for preflight, already released


def test_v1_profile_ignores_pinned_hash_and_needs_no_v1_lens(tmp_path: Path) -> None:
    write_source_dose(tmp_path)
    result = _run(
        tmp_path,
        JLENS_PROFILE,
        rows=[_row(index) for index in range(130)],
        jacobian_fn=_recording_jacobian([]),
        run_id="v1-run",
        expected_v1_lens_sha256="0" * 64,
    )
    assert result["status"] == "pass"


def test_runtime_does_not_write_compare_when_v2_run_aborts(tmp_path: Path) -> None:
    config = _write_v2_fixture(
        tmp_path,
        candidates=[2, 4],
        holdout_count=2,
        max_seq_len=8,
        max_projected_hours=1,
    )
    result = _run(
        tmp_path,
        config.name,
        rows=_rows(6),
        jacobian_fn=_recording_jacobian([]),
        probe_seconds=3600.0,
    )
    assert result["status"] == "aborted"
    assert not (tmp_path / V2_RESULTS / "04_jlens_compare.json").exists()


def test_v1_profile_does_not_write_compare(tmp_path: Path) -> None:
    write_source_dose(tmp_path)
    _run(
        tmp_path,
        JLENS_PROFILE,
        rows=[_row(index) for index in range(130)],
        jacobian_fn=_recording_jacobian([]),
        run_id="v1-run",
    )
    results = tmp_path / "results" / "qwen25_coder_7b_jlens"
    assert not (results / "04_jlens_compare.json").exists()
