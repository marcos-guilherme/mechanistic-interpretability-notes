from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refusal_direction_replication.artifacts import ArtifactEnvelope, write_artifact
from refusal_direction_replication.data import PromptExample, prompt_content_hash
from refusal_direction_replication.directions import CandidateDirection, save_direction
from refusal_direction_replication.gates import (
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    evaluate_causal_gate,
)
from refusal_direction_replication.interventions import directional_ablation_hook_specs
from refusal_direction_replication.pipeline import load_config
from refusal_direction_replication.qwen_dose import (
    STATUS_NO_DOSE_QUALIFIED,
    coefficient_checkpoint_name,
    select_dose_coefficients,
    sha256_file,
)
from refusal_direction_replication.qwen_dose_runtime import (
    DOSE_PROFILE,
    DOSE_STAGES,
    run_qwen_dose,
)
from refusal_direction_replication.run_progress import RunProgressStore


ROOT = Path(__file__).resolve().parents[1]
QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
UPSTREAM_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
PILOT_RUN = "c0818150-f834-4692-b271-4e10a97eb57e"
PILOT_COMMIT = "2a2eb714246174e4f627fcd0061c61143a4e8276"
DOSE_RUN = "dose-followup-run-1"
IMPLEMENTATION_COMMIT = "a" * 40
REQUIRED_SPLIT_KEYS = (
    "harmful_train",
    "harmless_train",
    "harmful_val",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _example(example_id: str, text: str, label: str) -> PromptExample:
    return PromptExample(id=example_id, text=text, label=label, source="fixture")  # type: ignore[arg-type]


def _hash(text: str) -> str:
    return prompt_content_hash(text)


def _complete_usage(n: int = 4) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    example_ids = {
        key: [f"{key}-{index:04d}" for index in range(n)]
        for key in REQUIRED_SPLIT_KEYS
    }
    content_hashes = {
        key: [_hash(f"{key} original {index}") for index in range(n)]
        for key in REQUIRED_SPLIT_KEYS
    }
    return example_ids, content_hashes


def _envelope(*, experiment: str, results: dict, config: dict) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        experiment=experiment,
        config=config,
        results=results,
        provenance={
            "model_name": QWEN_MODEL,
            "requested_model_revision": QWEN_REVISION,
            "resolved_model_revision": QWEN_REVISION,
            "upstream_commit": UPSTREAM_COMMIT,
            "run_id": PILOT_RUN,
            "implementation_commit": PILOT_COMMIT,
            "runtime": {"device": "cuda", "dtype": "bfloat16"},
        },
    )


def write_source_pilot(root: Path, *, layer: int = 18, offset: int = -1) -> None:
    shutil.copytree(ROOT / "configs", root / "configs")
    config = load_config(ROOT, profile="qwen25_coder_7b.json")
    results_dir = root / "results" / "qwen25_coder_7b"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b"
    results_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    example_ids, content_hashes = _complete_usage()
    write_artifact(
        results_dir / "00_preflight.json",
        _envelope(experiment="preflight", results={"n_layers": 28}, config=config),
    )
    write_artifact(
        results_dir / "01_baseline.json",
        _envelope(
            experiment="baseline",
            results={
                "example_ids": example_ids,
                "content_hashes": content_hashes,
            },
            config=config,
        ),
    )
    write_artifact(
        results_dir / "02_candidates.json",
        _envelope(
            experiment="extract_candidate_directions",
            results={"candidate_count": 28},
            config=config,
        ),
    )
    write_artifact(
        results_dir / "03_interventions.json",
        _envelope(
            experiment="causal_interventions",
            results={
                "selected": {
                    "layer": layer,
                    "position_offset": offset,
                    "norm": 1.0,
                },
                "example_ids": {
                    "harmful_test": example_ids["harmful_test"],
                    "harmless_test": example_ids["harmless_test"],
                },
                "content_hashes": {
                    "harmful_test": content_hashes["harmful_test"],
                    "harmless_test": content_hashes["harmless_test"],
                },
            },
            config=config,
        ),
    )
    write_artifact(
        results_dir / "final_evaluation.json",
        _envelope(experiment="final_evaluation", results={}, config=config),
    )
    write_artifact(
        results_dir / "execution_manifest.json",
        _envelope(
            experiment="execution_manifest",
            results={"status": STATUS_CAUSAL_GATE_FAILED},
            config=config,
        ),
    )
    _write_json(
        results_dir / "pilot_evaluation.json",
        {
            "status": STATUS_CAUSAL_GATE_FAILED,
            "operational_success": True,
            "run_id": PILOT_RUN,
            "implementation_commit": PILOT_COMMIT,
            "model_name": QWEN_MODEL,
            "model_revision": QWEN_REVISION,
        },
    )
    save_direction(
        artifacts_dir / "selected_direction.pt",
        CandidateDirection(layer, offset, torch.ones(8)),
    )


def write_leftover_datasets(root: Path, *, count: int = 70) -> None:
    dataset = root / "artifacts" / "datasets"
    for label in ("harmful", "harmless"):
        rows = [
            {
                "id": f"{label}_test-{index:04d}",
                "text": f"leftover {label} prompt {index}",
                "label": label,
                "source": "fixture",
            }
            for index in range(10, 10 + count)
        ]
        _write_json(dataset / f"{label}_test.json", rows)


def compact_condition(
    examples: list[PromptExample],
    *,
    refusal_rate: float,
) -> dict:
    n = len(examples)
    n_refusing = int(round(refusal_rate * n))
    flags = [True] * n_refusing + [False] * (n - n_refusing)
    return {
        "refusal_rate": refusal_rate,
        "refusal_flags": flags,
        "n_valid": n,
        "n_invalid": 0,
        "valid_ids": [example.id for example in examples],
        "invalid_ids": [],
        "mean_refusal_log_odds": 1.0 if refusal_rate >= 0.5 else -1.0,
        "per_example_refusal_log_odds": [
            1.0 if flag else -1.0 for flag in flags
        ],
    }


class ScriptedMetrics:
    def __init__(
        self,
        *,
        baseline_harmful: float = 0.90,
        baseline_harmless: float = 0.05,
        ablation_drop_at: float = 0.35,
        addition_increase_at: float = 0.20,
        kl: float = 0.04,
        control_drop: float = 0.0,
        confirmation_pass: bool = True,
    ) -> None:
        self.baseline_harmful = baseline_harmful
        self.baseline_harmless = baseline_harmless
        self.ablation_drop_at = ablation_drop_at
        self.addition_increase_at = addition_increase_at
        self.kl = kl
        self.control_drop = control_drop
        self.confirmation_pass = confirmation_pass
        self.calls: list[tuple[str, float | None]] = []

    def __call__(
        self,
        *,
        condition: str,
        coefficient: float | None,
        examples: list[PromptExample],
        split: str,
        **_kwargs,
    ) -> dict:
        self.calls.append((condition, coefficient, split))
        if condition == "kl":
            return {"harmless_kl": self.kl}
        if condition == "baseline_harmful":
            rate = self.baseline_harmful
        elif condition == "baseline_harmless":
            rate = self.baseline_harmless
        elif condition == "ablation_harmful":
            drop = (
                0.50
                if coefficient is not None and coefficient >= self.ablation_drop_at
                else 0.0
            )
            rate = self.baseline_harmful - drop
        elif condition == "ablation_harmless":
            rate = self.baseline_harmless
        elif condition == "addition_harmless":
            increase = (
                0.50
                if coefficient is not None and coefficient >= self.addition_increase_at
                else 0.0
            )
            rate = self.baseline_harmless + increase
        elif condition == "random_control_harmful":
            if split == "dose_confirmation" and not self.confirmation_pass:
                rate = self.baseline_harmful - 0.50
            else:
                rate = self.baseline_harmful - self.control_drop
        else:
            raise AssertionError(condition)
        if split == "dose_confirmation" and not self.confirmation_pass:
            if condition == "ablation_harmful":
                rate = self.baseline_harmful - 0.01
            if condition == "kl":
                return {"harmless_kl": 0.50}
        return compact_condition(list(examples), refusal_rate=rate)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_coefficient_checkpoint_name_is_deterministic() -> None:
    assert coefficient_checkpoint_name(0.1) == "checkpoint_coeff_0.1.json"
    assert coefficient_checkpoint_name(0.35) == "checkpoint_coeff_0.35.json"
    assert coefficient_checkpoint_name(1.0) == "checkpoint_coeff_1.json"


def test_sha256_file_hashes_bytes(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_dose_runs_prepare_sweep_confirm_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    metrics = ScriptedMetrics()
    releases: list[str] = []

    def release() -> None:
        releases.append("released")

    result = run_qwen_dose(
        tmp_path,
        run_id=DOSE_RUN,
        measure=metrics,
        release_resources=release,
    )
    assert result["status"] == STATUS_PASS
    assert result["run_id"] == DOSE_RUN
    assert result["run_id"] != PILOT_RUN
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    assert (results_dir / "00_dose_prepare.json").is_file()
    assert (results_dir / "01_dose_baseline.json").is_file()
    assert (results_dir / "02_dose_selection.json").is_file()
    assert (results_dir / "03_dose_confirmation.json").is_file()
    assert (results_dir / "dose_followup_evaluation.json").is_file()
    assert (results_dir / "execution_manifest.json").is_file()
    grid = load_config(tmp_path, profile=DOSE_PROFILE)["dose_grid"]
    for coefficient in grid:
        assert (results_dir / coefficient_checkpoint_name(coefficient)).is_file()
    assert releases
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b_dose"
    status = RunProgressStore(
        artifacts, profile=DOSE_PROFILE, run_id=DOSE_RUN
    ).read_status()
    assert status["state"] == "completed"
    assert status["current_stage"] == "confirm"


def test_dose_skips_completed_prepare_and_coefficient_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    first = ScriptedMetrics()
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=first)
    first_prepare_calls = [
        item for item in first.calls if item[2] == "dose_selection"
    ]
    second = ScriptedMetrics()
    result = run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=second)
    assert result["status"] == STATUS_PASS
    assert second.calls == []
    assert first_prepare_calls


def test_stale_checkpoint_with_other_run_id_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id="old-run", measure=ScriptedMetrics())
    with pytest.raises(ValueError, match="stale"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_stale_checkpoint_with_other_source_hash_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    direction = tmp_path / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    save_direction(direction, CandidateDirection(18, -1, torch.ones(8) * 2))
    with pytest.raises(ValueError, match="stale"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_no_qualified_dose_skips_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    metrics = ScriptedMetrics(ablation_drop_at=9.0)
    result = run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=metrics)
    assert result["status"] == STATUS_NO_DOSE_QUALIFIED
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    assert (results_dir / "02_dose_selection.json").is_file()
    assert not (results_dir / "03_dose_confirmation.json").exists()
    assert "dose_confirmation" not in [item[2] for item in metrics.calls]
    status = RunProgressStore(
        tmp_path / "artifacts" / "qwen25_coder_7b_dose",
        profile=DOSE_PROFILE,
        run_id=DOSE_RUN,
    ).read_status()
    assert status["state"] == "aborted"
    again = run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    assert again["status"] == STATUS_NO_DOSE_QUALIFIED


def test_confirmation_failure_is_causal_gate_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    metrics = ScriptedMetrics(confirmation_pass=False)
    result = run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=metrics)
    assert result["status"] == STATUS_CAUSAL_GATE_FAILED
    assert (tmp_path / "results" / "qwen25_coder_7b_dose" / "03_dose_confirmation.json").is_file()
    status = RunProgressStore(
        tmp_path / "artifacts" / "qwen25_coder_7b_dose",
        profile=DOSE_PROFILE,
        run_id=DOSE_RUN,
    ).read_status()
    assert status["state"] == "aborted"
    assert status["abort_gate"] == "causal"


def test_operational_failure_is_retryable_with_same_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    inner = ScriptedMetrics()

    def flaky(**kwargs):
        if kwargs["condition"] == "ablation_harmful" and kwargs["coefficient"] == 0.5:
            raise RuntimeError("simulated oom")
        return inner(**kwargs)

    with pytest.raises(RuntimeError, match="simulated oom"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=flaky)
    status = RunProgressStore(
        tmp_path / "artifacts" / "qwen25_coder_7b_dose",
        profile=DOSE_PROFILE,
        run_id=DOSE_RUN,
    ).read_status()
    assert status["state"] == "failed"
    evaluation = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    )
    assert evaluation["status"] == STATUS_OPERATIONAL_ERROR
    assert "simulated oom" not in json.dumps(evaluation)
    recovered = run_qwen_dose(
        tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics()
    )
    assert recovered["status"] == STATUS_PASS


def test_prepare_records_source_hashes_and_omits_prompt_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    prepare = _read(tmp_path / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json")
    hashes = prepare["provenance"]["source_artifact_hashes"]
    assert hashes["selected_direction"] == sha256_file(
        tmp_path / "artifacts" / "qwen25_coder_7b" / "selected_direction.pt"
    )
    assert hashes["baseline"] == sha256_file(
        tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    )
    assert hashes["interventions"] == sha256_file(
        tmp_path / "results" / "qwen25_coder_7b" / "03_interventions.json"
    )
    assert prepare["provenance"]["run_id"] == DOSE_RUN
    assert prepare["provenance"]["source_pilot_run_id"] == PILOT_RUN
    assert prepare["results"]["selected"] == {"layer": 18, "position_offset": -1}
    dumped = json.dumps(prepare)
    assert "leftover harmful prompt" not in dumped
    assert "prompt" not in prepare["results"]


def test_sweep_progress_counts_completed_coefficients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    events = (
        tmp_path / "artifacts" / "qwen25_coder_7b_dose" / "events.jsonl"
    ).read_text(encoding="utf-8")
    records = [json.loads(line) for line in events.splitlines() if line.strip()]
    sweep_progress = [
        record
        for record in records
        if record.get("event") == "progress" and record.get("stage") == "sweep"
    ]
    grid = load_config(tmp_path, profile=DOSE_PROFILE)["dose_grid"]
    assert sweep_progress[-1]["completed_units"] == len(grid)
    assert sweep_progress[-1]["total_units"] == len(grid)


def test_selection_uses_selector_and_ignores_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    selection = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "02_dose_selection.json"
    )["results"]
    recomputed = select_dose_coefficients(
        ablation_curve=selection["ablation_curve"],
        addition_curve=selection["addition_curve"],
    )
    assert selection["selected_pair"] == recomputed["selected_pair"]
    assert selection["selected_pair"] == {"ablation": 0.35, "addition": 0.2}
    assert "confirmation" not in json.dumps(selection)


def test_random_control_uses_selected_ablation_coefficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    metrics = ScriptedMetrics()
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=metrics)
    control_calls = [
        item
        for item in metrics.calls
        if item[0] == "random_control_harmful"
    ]
    assert control_calls == [("random_control_harmful", 0.35, "dose_confirmation")]
    confirmation = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "03_dose_confirmation.json"
    )
    assert confirmation["results"]["random_control"]["coefficient"] == 0.35
    gate = evaluate_causal_gate(
        baseline_harmful_refusal_rate=confirmation["results"]["baseline"][
            "harmful_refusal_rate"
        ],
        baseline_harmless_refusal_rate=confirmation["results"]["baseline"][
            "harmless_refusal_rate"
        ],
        ablation_harmful_refusal_rate=confirmation["results"]["ablation"][
            "harmful_refusal_rate"
        ],
        addition_harmless_refusal_rate=confirmation["results"]["addition"][
            "harmless_refusal_rate"
        ],
        random_control_harmful_refusal_rate=confirmation["results"]["random_control"][
            "harmful_refusal_rate"
        ],
        ablation_harmless_kl=confirmation["results"]["ablation"]["harmless_kl"],
        thresholds=load_config(tmp_path, profile=DOSE_PROFILE)["causal_gate"],
    )
    evaluation = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "dose_followup_evaluation.json"
    )
    assert evaluation["causal_gate"]["passed"] is True
    assert evaluation["causal_gate"]["passed"] == gate.passed


def test_prepare_rejects_wrong_source_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_source_pilot(tmp_path, layer=7)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    with pytest.raises(ValueError, match="layer"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_compact_results_never_contain_generation_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    leak_keys = {"text", "prompt", "generation", "generations", "outputs", "responses"}
    for path in (tmp_path / "results" / "qwen25_coder_7b_dose").glob("*.json"):
        payload = _read(path)
        dumped = json.dumps(payload)
        assert "leftover harmful prompt" not in dumped
        assert "leftover harmless prompt" not in dumped
        results = payload.get("results") or {}
        assert leak_keys.isdisjoint(results.keys())


def test_dose_stages_tuple() -> None:
    assert DOSE_STAGES == ("prepare", "sweep", "confirm")
    assert DOSE_PROFILE == "qwen25_coder_7b_dose.json"


class _HookModule:
    def register_forward_pre_hook(self, hook):
        return SimpleNamespace(remove=lambda: None)

    def register_forward_hook(self, hook):
        return SimpleNamespace(remove=lambda: None)


def test_fake_model_measure_threads_fractional_ablation_coefficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_qwen_pilot import ChatTokenizer, FakeQwenModel

    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    seen: list[float] = []

    def spy(*args, coefficient: float = 1.0, **kwargs):
        seen.append(float(coefficient))
        return directional_ablation_hook_specs(*args, coefficient=coefficient, **kwargs)

    monkeypatch.setattr(
        "refusal_direction_replication.qwen_dose_runtime.directional_ablation_hook_specs",
        spy,
    )

    class HookedFake(FakeQwenModel):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            for _ in range(28):
                block = _HookModule()
                block.self_attn = _HookModule()
                block.mlp = _HookModule()
                layers.append(block)
            self.model = SimpleNamespace(layers=layers)

    def load_model(_config):
        return HookedFake(), ChatTokenizer(), torch.bfloat16

    result = run_qwen_dose(
        tmp_path,
        run_id=DOSE_RUN,
        load_model=load_model,
    )
    assert result["status"] in {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED, STATUS_NO_DOSE_QUALIFIED}
    grid = load_config(tmp_path, profile=DOSE_PROFILE)["dose_grid"]
    for coefficient in grid:
        assert float(coefficient) in seen
    baseline = _read(
        tmp_path / "results" / "qwen25_coder_7b_dose" / "01_dose_baseline.json"
    )
    dumped = json.dumps(baseline)
    assert "leftover harmful prompt" not in dumped
    assert "generation" not in (baseline.get("results") or {})
    assert "text" not in (baseline.get("results") or {})


def test_confirmation_resumes_exactly_once_from_validated_03(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    evaluation_path = results_dir / "dose_followup_evaluation.json"
    evaluation_path.unlink()
    second = ScriptedMetrics()
    result = run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=second)
    assert result["status"] == STATUS_PASS
    assert evaluation_path.is_file()
    assert not any(item[2] == "dose_confirmation" for item in second.calls)
    assert not any(
        item[0].startswith("ablation") or item[0].startswith("baseline")
        for item in second.calls
    )


def test_skip_prepare_requires_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    (results_dir / "dose_followup_evaluation.json").unlink()
    (results_dir / "00_dose_prepare.json").unlink()
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    assert (results_dir / "00_dose_prepare.json").is_file()


def test_skip_sweep_requires_selection_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    (results_dir / "dose_followup_evaluation.json").unlink()
    (results_dir / "02_dose_selection.json").unlink()
    (results_dir / "03_dose_confirmation.json").unlink()
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    assert (results_dir / "02_dose_selection.json").is_file()


def test_checkpoint_rejects_wrong_implementation_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    path = results_dir / coefficient_checkpoint_name(0.35)
    payload = _read(path)
    payload["provenance"]["implementation_commit"] = "b" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    (results_dir / "dose_followup_evaluation.json").unlink()
    (results_dir / "02_dose_selection.json").unlink()
    (results_dir / "03_dose_confirmation.json").unlink()
    with pytest.raises(ValueError, match="stale"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_checkpoint_rejects_mismatched_coefficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    results_dir = tmp_path / "results" / "qwen25_coder_7b_dose"
    path = results_dir / coefficient_checkpoint_name(0.35)
    payload = _read(path)
    payload["results"]["coefficient"] = 0.75
    path.write_text(json.dumps(payload), encoding="utf-8")
    (results_dir / "dose_followup_evaluation.json").unlink()
    (results_dir / "02_dose_selection.json").unlink()
    (results_dir / "03_dose_confirmation.json").unlink()
    with pytest.raises(ValueError, match="coefficient"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_source_hashes_include_pilot_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())
    prepare = _read(tmp_path / "results" / "qwen25_coder_7b_dose" / "00_dose_prepare.json")
    hashes = prepare["provenance"]["source_artifact_hashes"]
    assert hashes["pilot_evaluation"] == sha256_file(
        tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    )


def test_source_envelopes_must_share_run_id_and_commit(tmp_path: Path) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    config = load_config(tmp_path, profile="qwen25_coder_7b.json")
    write_artifact(
        tmp_path / "results" / "qwen25_coder_7b" / "02_candidates.json",
        ArtifactEnvelope(
            experiment="extract_candidate_directions",
            config=config,
            results={"candidate_count": 28},
            provenance={
                "model_name": QWEN_MODEL,
                "requested_model_revision": QWEN_REVISION,
                "resolved_model_revision": QWEN_REVISION,
                "upstream_commit": UPSTREAM_COMMIT,
                "run_id": "other-run",
                "implementation_commit": PILOT_COMMIT,
                "runtime": {"device": "cuda", "dtype": "bfloat16"},
            },
        ),
    )
    with pytest.raises(ValueError, match="source"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=ScriptedMetrics())


def test_load_model_rejects_mismatched_commit_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_qwen_pilot import ChatTokenizer, FakeQwenModel

    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)

    def load_model(_config):
        return FakeQwenModel(revision="deadbeef" * 5), ChatTokenizer(), torch.bfloat16

    with pytest.raises(ValueError, match="commit_hash|revision"):
        run_qwen_dose(tmp_path, run_id=DOSE_RUN, load_model=load_model)


def test_single_release_boundary_per_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    releases: list[str] = []

    def release() -> None:
        releases.append("released")

    run_qwen_dose(
        tmp_path,
        run_id=DOSE_RUN,
        measure=ScriptedMetrics(),
        release_resources=release,
    )
    assert len(releases) == 3


def test_seed_everything_runs_before_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    order: list[str] = []

    def fake_seed(seed: int) -> None:
        order.append(f"seed:{seed}")

    monkeypatch.setattr(
        "refusal_direction_replication.qwen_dose_runtime.seed_everything",
        fake_seed,
    )
    inner = ScriptedMetrics()

    def measure(**kwargs):
        order.append("measure")
        return inner(**kwargs)

    run_qwen_dose(tmp_path, run_id=DOSE_RUN, measure=measure)
    assert order[0].startswith("seed:")
    assert "measure" in order
    assert order.index(order[0]) < order.index("measure")


def test_atomic_json_write_uses_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from refusal_direction_replication.artifacts import atomic_write_json

    replaced: list[tuple[str, str]] = []
    real_replace = __import__("os").replace

    def tracking_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", tracking_replace)
    path = tmp_path / "compact.json"
    atomic_write_json(path, {"ok": True})
    assert replaced
    assert _read(path) == {"ok": True}


def test_fake_model_actually_executes_registered_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_qwen_pilot import ChatTokenizer, FakeQwenModel

    write_source_pilot(tmp_path)
    write_leftover_datasets(tmp_path)
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT)
    invocations = {"count": 0}

    class CountingModule(_HookModule):
        def register_forward_pre_hook(self, hook):
            original = hook

            def wrapped(module, inputs):
                invocations["count"] += 1
                return original(module, inputs)

            self._pre = getattr(self, "_pre", [])
            self._pre.append(wrapped)

            def remove() -> None:
                if wrapped in self._pre:
                    self._pre.remove(wrapped)

            return SimpleNamespace(remove=remove)

        def register_forward_hook(self, hook):
            original = hook
            self._fwd = getattr(self, "_fwd", [])

            def wrapped(module, inputs, output):
                invocations["count"] += 1
                return original(module, inputs, output)

            self._fwd.append(wrapped)

            def remove() -> None:
                if wrapped in self._fwd:
                    self._fwd.remove(wrapped)

            return SimpleNamespace(remove=remove)

    class ExecutingFake(FakeQwenModel):
        def __init__(self) -> None:
            super().__init__(hidden_size=8)
            layers = []
            for _ in range(28):
                block = CountingModule()
                block.self_attn = CountingModule()
                block.mlp = CountingModule()
                layers.append(block)
            self.model = SimpleNamespace(layers=layers)
            self._param = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            batch, seq = input_ids.shape
            hidden = torch.ones(batch, seq, 8, dtype=torch.float32)
            for block in self.model.layers:
                for hook in getattr(block, "_pre", []):
                    result = hook(block, (hidden,))
                    if result is not None:
                        hidden = result[0] if isinstance(result, tuple) else result
            logits = hidden.sum(-1, keepdim=True).expand(batch, seq, 32)
            return SimpleNamespace(logits=logits)

    def load_model(_config):
        return ExecutingFake(), ChatTokenizer(), torch.bfloat16

    run_qwen_dose(tmp_path, run_id=DOSE_RUN, load_model=load_model)
    assert invocations["count"] > 0
