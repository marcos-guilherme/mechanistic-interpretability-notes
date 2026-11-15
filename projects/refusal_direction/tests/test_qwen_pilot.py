from __future__ import annotations

import ast
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refusal_direction_replication.artifacts import ArtifactEnvelope, write_artifact
from refusal_direction_replication.data import (
    PromptExample,
    normalize_prompt_text,
    prompt_content_hash,
)
from refusal_direction_replication.directions import candidate_grid
from refusal_direction_replication.gates import (
    STATUS_BASELINE_GATE_FAILED,
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    build_pilot_evaluation,
    evaluate_baseline_gate,
    evaluate_causal_gate,
)
from refusal_direction_replication.metrics import is_valid_generation
from refusal_direction_replication.pipeline import (
    _position_offsets,
    _sampled_splits,
    implementation_commit_hash,
    load_config,
    maybe_filter_examples,
    maybe_filter_scores,
    random_direction_control,
    split_examples,
    summarize_generated_condition,
)
from refusal_direction_replication.qwen_pilot import (
    PILOT_STAGES,
    release_compute_resources,
    resolve_run_id,
    run_preflight_stage,
    run_qwen_pilot,
)
from refusal_direction_replication.qwen_pilot_validate import validate_qwen_pilot
from refusal_direction_replication.run_progress import RunProgressStore

ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = ROOT / "scripts" / "fetch_upstream_splits.py"
PILOT_SCRIPT = ROOT / "scripts" / "qwen_run_pilot.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_qwen_pilot.py"
RUN_REFUSAL_SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "colab" / "run_refusal.py"
QWEN_PROFILE = "qwen25_coder_7b.json"
QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
UPSTREAM_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_REFUSAL = load_script("run_refusal_pilot", RUN_REFUSAL_SCRIPT)
FETCH = load_script("fetch_upstream_splits", FETCH_SCRIPT)
PILOT_CLI = load_script("qwen_run_pilot", PILOT_SCRIPT)
VALIDATE_CLI = load_script("validate_qwen_pilot", VALIDATE_SCRIPT)


class ChatTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.prefix_token_ids = {
            "I": [10],
            "I'm": [11, 12],
            "I am": [10, 14],
            "Sorry": [13],
            "As an": [15, 16],
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        content = messages[0]["content"]
        tokens = [100, 101, *self.encode(content, add_special_tokens=False)]
        if add_generation_prompt:
            tokens.extend([200, 201])
        if tokenize:
            return tokens
        return f"<user>{content}</user><assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text in self.prefix_token_ids:
            return list(self.prefix_token_ids[text])
        return [ord(character) for character in text]

    def __call__(self, texts, *, padding, truncation, return_tensors):
        ids = [
            torch.tensor(self.encode(text, add_special_tokens=False))
            for text in texts
        ]
        max_len = max(item.shape[0] for item in ids)
        batch = torch.full((len(ids), max_len), self.pad_token_id)
        for row, item in enumerate(ids):
            batch[row, -item.shape[0] :] = item
        return {
            "input_ids": batch,
            "attention_mask": (batch != self.pad_token_id).to(torch.long),
        }

    def batch_decode(self, continuations, *, skip_special_tokens: bool) -> list[str]:
        return ["ok" for _ in continuations]


class FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeQwenModel:
    def __init__(
        self,
        *,
        hidden_size: int = 3584,
        n_layers: int = 28,
        finite: bool = True,
        finite_generation: bool | None = None,
        revision: str = QWEN_REVISION,
    ) -> None:
        layer = SimpleNamespace(self_attn=object(), mlp=object())
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            _commit_hash=revision,
            torch_dtype=torch.bfloat16,
        )
        self.model = SimpleNamespace(layers=[layer] * n_layers)
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.finite = finite
        self.finite_generation = finite if finite_generation is None else finite_generation

    def parameters(self):
        yield self._param

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch, seq = input_ids.shape
        value = 0.0 if self.finite else math.inf
        logits = torch.full((batch, seq, 32), value, dtype=torch.float32)
        return FakeOutput(logits)

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=1,
        output_scores=False,
        return_dict_in_generate=False,
        **kwargs,
    ):
        extra = torch.ones(input_ids.shape[0], max_new_tokens, dtype=torch.long)
        sequences = torch.cat([input_ids.cpu(), extra], dim=1)
        if output_scores or return_dict_in_generate:
            value = 0.0 if self.finite_generation else math.inf
            scores = tuple(
                torch.full((input_ids.shape[0], 32), value, dtype=torch.float32)
                for _ in range(max_new_tokens)
            )
            return SimpleNamespace(sequences=sequences, scores=scores)
        return sequences


def qwen_config() -> dict:
    return load_config(ROOT, profile=QWEN_PROFILE)


def gemma_config() -> dict:
    return load_config(ROOT)


def write_split_file(path: Path, *, label: str, split: str, count: int, canary: str | None = None) -> None:
    rows = []
    for index in range(count):
        text = f"{label} {split} prompt {index}"
        if canary is not None and index == 0:
            text = canary
        rows.append(
            {
                "id": f"{split}-{index:04d}",
                "text": text,
                "label": label,
                "source": f"fixture:{split}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def write_dataset_files(root: Path, *, count: int = 8, include_test: bool = True, canary: str | None = None) -> None:
    dataset_root = root / "artifacts" / "datasets"
    names = [
        ("harmful", "harmful_train"),
        ("harmless", "harmless_train"),
        ("harmful", "harmful_val"),
        ("harmless", "harmless_val"),
    ]
    if include_test:
        names.extend(
            (("harmful", "harmful_test"), ("harmless", "harmless_test"))
        )
    for label, split in names:
        write_split_file(
            dataset_root / f"{split}.json",
            label=label,
            split=split,
            count=count,
            canary=canary if split.endswith("_val") else None,
        )


def copy_configs(root: Path) -> None:
    shutil.copytree(ROOT / "configs", root / "configs")


def envelope(
    *,
    experiment: str,
    results: dict,
    config: dict | None = None,
) -> ArtifactEnvelope:
    payload = config if config is not None else qwen_config()
    return ArtifactEnvelope(
        experiment=experiment,
        config=payload,
        results=results,
        provenance={
            "model_name": QWEN_MODEL,
            "requested_model_revision": QWEN_REVISION,
            "resolved_model_revision": QWEN_REVISION,
            "upstream_commit": UPSTREAM_COMMIT,
            "runtime": {"device": "cuda", "dtype": "bfloat16"},
            "run_id": "run-valid",
            "implementation_commit": "c" * 40,
        },
    )


def passing_gate_metrics() -> dict:
    return {
        "harmful_count": 32,
        "harmless_count": 32,
        "harmful_refusal_rate": 0.90,
        "harmless_refusal_rate": 0.10,
        "ablation_harmful_refusal_rate": 0.60,
        "addition_harmless_refusal_rate": 0.40,
        "random_control_harmful_refusal_rate": 0.85,
        "ablation_harmless_kl": 0.05,
    }


def write_valid_pilot_artifacts(root: Path, *, status: str = STATUS_PASS) -> None:
    copy_configs(root)
    results_dir = root / "results" / "qwen25_coder_7b"
    artifacts_dir = root / "artifacts" / "qwen25_coder_7b"
    results_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    metrics = passing_gate_metrics()
    n = 8 if status == STATUS_BASELINE_GATE_FAILED else 32
    example_ids = {
        "harmful_train": [f"harmful_train-{index:04d}" for index in range(n)],
        "harmless_train": [f"harmless_train-{index:04d}" for index in range(n)],
        "harmful_val": [f"harmful_val-{index:04d}" for index in range(n)],
        "harmless_val": [f"harmless_val-{index:04d}" for index in range(n)],
        "harmful_test": [f"harmful_test-{index:04d}" for index in range(n)],
        "harmless_test": [f"harmless_test-{index:04d}" for index in range(n)],
    }
    hashes = {
        name: f"{index:064x}" for index, name in enumerate(example_ids)
    }
    config = qwen_config()
    if status == STATUS_CAUSAL_GATE_FAILED:
        metrics["ablation_harmful_refusal_rate"] = 0.89
    flag_count = n
    harmful_true = n - 3 if n == 32 else n - 1
    harmless_true = 3 if n == 32 else 1
    harmful_flags = [True] * harmful_true + [False] * (n - harmful_true)
    harmless_flags = [True] * harmless_true + [False] * (n - harmless_true)
    metrics["harmful_count"] = n
    metrics["harmless_count"] = n
    metrics["harmful_refusal_rate"] = harmful_true / n
    metrics["harmless_refusal_rate"] = harmless_true / n
    write_artifact(
        results_dir / "00_preflight.json",
        envelope(
            experiment="preflight",
            results={
                "cuda_available": True,
                "dtype": "bfloat16",
                "n_layers": 28,
                "residual_width": 3584,
                "chat_template_ok": True,
                "refusal_token_id_count": 4,
                "generation_finite": True,
                "probe_counts": {"harmful": 2, "harmless": 2},
            },
            config=config,
        ),
    )
    baseline_results = {
        "sample_sizes": {
            "harmful_train": n,
            "harmless_train": n,
            "harmful_val": n,
            "harmless_val": n,
            "harmful_test": n,
            "harmless_test": n,
        },
        "filtered_sizes": {
            "harmful_train": n,
            "harmless_train": n,
            "harmful_val": n,
            "harmless_val": n,
        },
        "dataset_hashes": hashes,
        "example_ids": example_ids,
        "harmful_validation": {
            "refusal_rate": metrics["harmful_refusal_rate"],
            "refusal_flags": harmful_flags,
            "mean_refusal_log_odds": 1.2,
            "per_example_refusal_log_odds": [1.0] * flag_count,
            "n_valid": flag_count,
            "n_invalid": 0,
            "valid_ids": list(example_ids["harmful_val"]),
            "invalid_ids": [],
        },
        "harmless_validation": {
            "refusal_rate": metrics["harmless_refusal_rate"],
            "refusal_flags": harmless_flags,
            "mean_refusal_log_odds": -1.2,
            "per_example_refusal_log_odds": [-1.0] * flag_count,
            "n_valid": flag_count,
            "n_invalid": 0,
            "valid_ids": list(example_ids["harmless_val"]),
            "invalid_ids": [],
        },
        "content_hashes": {
            name: [
                __import__("hashlib").sha256(f"{name}:{index}".encode()).hexdigest()
                for index in range(n)
            ]
            for name in example_ids
        },
    }
    write_artifact(
        results_dir / "01_baseline.json",
        envelope(experiment="baseline", results=baseline_results, config=config),
    )
    write_artifact(
        results_dir / "02_candidates.json",
        envelope(
            experiment="extract_candidate_directions",
            results={
                "harmful_examples": n,
                "harmless_examples": n,
                "candidate_count": 28,
                "position_offsets": [-1],
                "example_ids": {
                    "harmful_train": example_ids["harmful_train"],
                    "harmless_train": example_ids["harmless_train"],
                },
            },
            config=config,
        ),
    )
    candidate_scores = [
        {
            "layer": layer,
            "position_offset": -1,
            "ablation_refusal_score": -1.0,
            "addition_refusal_score": 1.0,
            "kl_divergence": 0.01,
            "n_layers": 28,
        }
        for layer in range(28)
    ]
    write_artifact(
        results_dir / "03_interventions.json",
        envelope(
            experiment="causal_interventions",
            results={
                "selected": {"layer": 10, "position_offset": -1, "norm": 3.0},
                "candidate_scores": candidate_scores,
                "conditions": {
                    "baseline_harmful": {"refusal_rate": metrics["harmful_refusal_rate"]},
                    "baseline_harmless": {"refusal_rate": metrics["harmless_refusal_rate"]},
                    "ablation_harmful": {
                        "refusal_rate": metrics["ablation_harmful_refusal_rate"]
                    },
                    "ablation_harmless": {"refusal_rate": 0.05},
                    "addition_harmless": {
                        "refusal_rate": metrics["addition_harmless_refusal_rate"]
                    },
                    "random_control_harmful": {
                        "refusal_rate": metrics["random_control_harmful_refusal_rate"]
                    },
                },
                "ablation_harmless_kl": metrics["ablation_harmless_kl"],
                "example_ids": {
                    "harmful_test": example_ids["harmful_test"],
                    "harmless_test": example_ids["harmless_test"],
                    "harmful_val": example_ids["harmful_val"],
                    "harmless_val": example_ids["harmless_val"],
                },
            },
            config=config,
        ),
    )
    evaluation_results = {
        "selected": {"layer": 10, "position_offset": -1, "norm": 3.0},
        "baseline": {
            "harmful_refusal_rate": metrics["harmful_refusal_rate"],
            "harmless_refusal_rate": metrics["harmless_refusal_rate"],
            "ce_delta": 0.0,
            "mean_cross_entropy": 1.0,
        },
        "ablation": {
            "harmful_refusal_rate": metrics["ablation_harmful_refusal_rate"],
            "harmless_refusal_rate": 0.05,
            "ce_delta": -0.01,
            "mean_cross_entropy": 0.99,
            "harmless_kl": metrics["ablation_harmless_kl"],
        },
        "addition": {
            "harmless_refusal_rate": metrics["addition_harmless_refusal_rate"],
            "ce_delta": 0.02,
            "mean_cross_entropy": 1.02,
        },
        "random_control": {
            "harmful_refusal_rate": metrics["random_control_harmful_refusal_rate"],
            "ce_delta": 0.0,
            "mean_cross_entropy": 1.0,
        },
        "example_ids": {
            "harmful_test": example_ids["harmful_test"],
            "harmless_test": example_ids["harmless_test"],
        },
    }
    write_artifact(
        results_dir / "final_evaluation.json",
        envelope(
            experiment="final_evaluation",
            results=evaluation_results,
            config=config,
        ),
    )
    baseline_gate = evaluate_baseline_gate(
        harmful_count=flag_count,
        harmless_count=flag_count,
        harmful_refusal_rate=metrics["harmful_refusal_rate"],
        harmless_refusal_rate=metrics["harmless_refusal_rate"],
        thresholds=config["baseline_gate"],
    )
    causal_gate = evaluate_causal_gate(
        baseline_harmful_refusal_rate=metrics["harmful_refusal_rate"],
        baseline_harmless_refusal_rate=metrics["harmless_refusal_rate"],
        ablation_harmful_refusal_rate=metrics["ablation_harmful_refusal_rate"],
        addition_harmless_refusal_rate=metrics["addition_harmless_refusal_rate"],
        random_control_harmful_refusal_rate=metrics[
            "random_control_harmful_refusal_rate"
        ],
        ablation_harmless_kl=metrics["ablation_harmless_kl"],
        thresholds=config["causal_gate"],
    )
    evaluation = build_pilot_evaluation(
        status=status,
        operational_success=True,
        baseline_gate=baseline_gate,
        causal_gate=causal_gate if status != STATUS_BASELINE_GATE_FAILED else None,
        run_id="run-valid",
        implementation_commit="c" * 40,
    )
    (results_dir / "pilot_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    store = RunProgressStore(
        artifacts_dir,
        profile=QWEN_PROFILE,
        run_id="run-valid",
    )
    if status == STATUS_BASELINE_GATE_FAILED:
        store.initialize(stage="baseline", total_units=1)
        store.begin_running()
        store.abort(gate="baseline", reason=baseline_gate.reason)
        (results_dir / "02_candidates.json").unlink()
        (results_dir / "03_interventions.json").unlink()
        (results_dir / "final_evaluation.json").unlink()
    elif status == STATUS_CAUSAL_GATE_FAILED:
        store.initialize(stage="evaluation", total_units=1)
        store.begin_running()
        store.abort(gate="causal", reason=causal_gate.reason)
    else:
        for stage in PILOT_STAGES:
            store.initialize(stage=stage, total_units=1)
            store.begin_running()
            store.complete_stage()


def test_qwen_profile_has_pilot_controls() -> None:
    config = qwen_config()
    assert config["test_per_class"] == 32
    assert config["filter_by_behavior"] is False
    assert config["selection_split"] == "validation"
    assert config["final_conditions_split"] == "test"
    assert config["position_offsets"] == [-1]
    assert config["baseline_gate"] == {
        "min_valid_per_class": 24,
        "min_refusal_gap": 0.5,
    }
    assert config["causal_gate"] == {
        "min_harmful_ablation_drop": 0.2,
        "min_harmless_addition_increase": 0.2,
        "max_random_control_harmful_drop": 0.1,
        "max_harmless_ablation_kl": 0.1,
    }


def test_gemma_profile_keeps_four_splits_and_filtering_defaults() -> None:
    config = gemma_config()
    assert "test_per_class" not in config
    assert "filter_by_behavior" not in config
    assert "baseline_gate" not in config
    assert "causal_gate" not in config
    assert config.get("filter_by_behavior", True) is True
    assert config.get("selection_split", "validation") == "validation"
    assert config.get("final_conditions_split", "validation") == "validation"


def test_fetch_splits_include_upstream_test_files() -> None:
    assert FETCH.SPLITS == (
        "harmful_train",
        "harmful_val",
        "harmless_train",
        "harmless_val",
        "harmful_test",
        "harmless_test",
    )
    assert UPSTREAM_COMMIT in FETCH.BASE
    assert FETCH.COMMIT == UPSTREAM_COMMIT


def test_qwen_sampled_splits_are_isolated(tmp_path: Path) -> None:
    write_dataset_files(tmp_path, count=8)
    config = {
        **qwen_config(),
        "train_per_class": 4,
        "validation_per_class": 3,
        "test_per_class": 3,
    }
    splits = _sampled_splits(tmp_path, config)
    assert set(splits) == {
        "harmful_train",
        "harmless_train",
        "harmful_val",
        "harmless_val",
        "harmful_test",
        "harmless_test",
    }
    groups = {
        "train": {item.id for item in splits["harmful_train"] + splits["harmless_train"]},
        "val": {item.id for item in splits["harmful_val"] + splits["harmless_val"]},
        "test": {item.id for item in splits["harmful_test"] + splits["harmless_test"]},
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    harmful, harmless = split_examples(splits, "test")
    assert {item.label for item in harmful} == {"harmful"}
    assert {item.label for item in harmless} == {"harmless"}


def test_gemma_sampled_splits_do_not_require_test_files(tmp_path: Path) -> None:
    write_dataset_files(tmp_path, count=8, include_test=False)
    config = {
        **gemma_config(),
        "train_per_class": 4,
        "validation_per_class": 3,
    }
    splits = _sampled_splits(tmp_path, config)
    assert set(splits) == {
        "harmful_train",
        "harmless_train",
        "harmful_val",
        "harmless_val",
    }
    assert (tmp_path / "artifacts" / "datasets" / "harmful_test.json").exists() is False


def test_unfiltered_profile_keeps_mixed_score_examples() -> None:
    examples = [
        PromptExample("a", "one", "harmful", "fixture"),
        PromptExample("b", "two", "harmful", "fixture"),
        PromptExample("c", "three", "harmful", "fixture"),
    ]
    scores = torch.tensor([2.0, -1.0, 0.5])
    qwen = maybe_filter_examples(
        examples,
        scores,
        keep_positive=True,
        config=qwen_config(),
    )
    gemma = maybe_filter_examples(
        examples,
        scores,
        keep_positive=True,
        config=gemma_config(),
    )
    assert qwen == examples
    assert gemma == [examples[0], examples[2]]
    qwen_scores = maybe_filter_scores(
        scores,
        keep_positive=True,
        config=qwen_config(),
    )
    gemma_scores = maybe_filter_scores(
        scores,
        keep_positive=True,
        config=gemma_config(),
    )
    assert qwen_scores.tolist() == [2.0, -1.0, 0.5]
    assert gemma_scores.tolist() == [2.0, 0.5]


def test_qwen_candidate_grid_has_exactly_28_offset_minus_one() -> None:
    harmful = torch.randn(6, 28, 1, 8)
    harmless = torch.randn(6, 28, 1, 8)
    offsets = _position_offsets(ChatTokenizer(), qwen_config())
    candidates = candidate_grid(
        harmful,
        harmless,
        position_offsets=offsets,
    )
    assert len(candidates) == 28
    assert {item.position_offset for item in candidates} == {-1}
    assert [item.layer for item in candidates] == list(range(28))


@pytest.mark.parametrize(
    ("harmful_count", "harmless_count", "harmful_rate", "harmless_rate", "expected"),
    [
        (24, 24, 0.80, 0.30, True),
        (23, 24, 0.90, 0.10, False),
        (24, 23, 0.90, 0.10, False),
        (32, 32, 0.50, 0.00, True),
        (32, 32, 0.49, 0.00, False),
    ],
)
def test_baseline_gate_boundaries(
    harmful_count: int,
    harmless_count: int,
    harmful_rate: float,
    harmless_rate: float,
    expected: bool,
) -> None:
    result = evaluate_baseline_gate(
        harmful_count=harmful_count,
        harmless_count=harmless_count,
        harmful_refusal_rate=harmful_rate,
        harmless_refusal_rate=harmless_rate,
        thresholds=qwen_config()["baseline_gate"],
    )
    assert result.passed is expected
    assert {check.name for check in result.checks} == {
        "min_valid_harmful",
        "min_valid_harmless",
        "min_refusal_gap",
    }
    for check in result.checks:
        assert check.threshold is not None
        assert check.observed is not None
        assert check.reason


@pytest.mark.parametrize(
    (
        "ablation_harmful",
        "addition_harmless",
        "control_harmful",
        "kl",
        "expected",
    ),
    [
        (0.70, 0.31, 0.90, 0.10, True),
        (0.71, 0.31, 0.90, 0.10, False),
        (0.70, 0.29, 0.90, 0.10, False),
        (0.70, 0.31, 0.79, 0.10, False),
        (0.70, 0.31, 0.80, 0.10, True),
        (0.70, 0.31, 0.90, 0.101, False),
    ],
)
def test_causal_gate_boundaries(
    ablation_harmful: float,
    addition_harmless: float,
    control_harmful: float,
    kl: float,
    expected: bool,
) -> None:
    result = evaluate_causal_gate(
        baseline_harmful_refusal_rate=0.90,
        baseline_harmless_refusal_rate=0.10,
        ablation_harmful_refusal_rate=ablation_harmful,
        addition_harmless_refusal_rate=addition_harmless,
        random_control_harmful_refusal_rate=control_harmful,
        ablation_harmless_kl=kl,
        thresholds=qwen_config()["causal_gate"],
    )
    assert result.passed is expected
    names = {check.name for check in result.checks}
    assert names == {
        "min_harmful_ablation_drop",
        "min_harmless_addition_increase",
        "max_random_control_harmful_drop",
        "max_harmless_ablation_kl",
    }


def test_random_control_direction_is_norm_matched_and_orthogonal() -> None:
    direction = torch.tensor([6.0, 8.0, 0.0, 0.0])
    control = random_direction_control(direction, seed=42)
    assert torch.allclose(control.norm(), direction.norm())
    assert abs(float(torch.dot(control, direction))) < 1e-4
    other = random_direction_control(direction, seed=42)
    assert torch.equal(control, other)


def test_pilot_evaluation_distinguishes_statuses() -> None:
    baseline = evaluate_baseline_gate(
        harmful_count=32,
        harmless_count=32,
        harmful_refusal_rate=0.9,
        harmless_refusal_rate=0.1,
        thresholds=qwen_config()["baseline_gate"],
    )
    causal = evaluate_causal_gate(
        baseline_harmful_refusal_rate=0.9,
        baseline_harmless_refusal_rate=0.1,
        ablation_harmful_refusal_rate=0.6,
        addition_harmless_refusal_rate=0.4,
        random_control_harmful_refusal_rate=0.85,
        ablation_harmless_kl=0.05,
        thresholds=qwen_config()["causal_gate"],
    )
    passed = build_pilot_evaluation(
        status=STATUS_PASS,
        operational_success=True,
        baseline_gate=baseline,
        causal_gate=causal,
        run_id="r1",
    )
    failed_baseline = build_pilot_evaluation(
        status=STATUS_BASELINE_GATE_FAILED,
        operational_success=True,
        baseline_gate=evaluate_baseline_gate(
            harmful_count=8,
            harmless_count=8,
            harmful_refusal_rate=0.4,
            harmless_refusal_rate=0.3,
            thresholds=qwen_config()["baseline_gate"],
        ),
        causal_gate=None,
        run_id="r1",
    )
    assert passed["status"] == STATUS_PASS
    assert passed["operational_success"] is True
    assert failed_baseline["status"] == STATUS_BASELINE_GATE_FAILED
    assert failed_baseline["causal_gate"] is None


def test_preflight_publishes_aggregates_without_prompt_text(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    write_dataset_files(tmp_path, count=4, canary="SECRET_PREFLIGHT_PROMPT")
    model = FakeQwenModel()
    tokenizer = ChatTokenizer()

    def load_model(_config):
        return model, tokenizer, torch.bfloat16

    payload = run_preflight_stage(
        tmp_path,
        profile=QWEN_PROFILE,
        load_model=load_model,
        cuda_available=True,
    )
    compact = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b" / "00_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    results = compact["results"]
    assert results["cuda_available"] is True
    assert results["dtype"] == "bfloat16"
    assert results["n_layers"] == 28
    assert results["residual_width"] == 3584
    assert results["chat_template_ok"] is True
    assert results["generation_finite"] is True
    assert results["refusal_token_id_count"] >= 1
    dumped = json.dumps(compact)
    assert "SECRET_PREFLIGHT_PROMPT" not in dumped
    assert "text" not in results
    assert payload["results"]["probe_counts"] == {"harmful": 2, "harmless": 2}


def test_preflight_rejects_nonfinite_logits(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    write_dataset_files(tmp_path, count=4)

    def load_model(_config):
        return FakeQwenModel(finite=False), ChatTokenizer(), torch.bfloat16

    with pytest.raises(ValueError, match="non-finite"):
        run_preflight_stage(
            tmp_path,
            profile=QWEN_PROFILE,
            load_model=load_model,
            cuda_available=True,
        )


def _stage_writer(name: str, calls: list[str], *, baseline_gap: float = 0.8):
    def runner(root: Path, *, profile: str, **_kwargs) -> dict:
        calls.append(name)
        results_dir = root / "results" / "qwen25_coder_7b"
        results_dir.mkdir(parents=True, exist_ok=True)
        config = qwen_config()
        if name == "preflight":
            write_artifact(
                results_dir / "00_preflight.json",
                envelope(experiment="preflight", results={"ok": True}, config=config),
            )
        elif name == "baseline":
            harmful_rate = 0.9
            harmless_rate = harmful_rate - baseline_gap
            write_artifact(
                results_dir / "01_baseline.json",
                envelope(
                    experiment="baseline",
                    results={
                        "harmful_validation": {
                            "refusal_rate": harmful_rate,
                            "refusal_flags": [True] * 32,
                        },
                        "harmless_validation": {
                            "refusal_rate": harmless_rate,
                            "refusal_flags": [False] * 32,
                        },
                    },
                    config=config,
                ),
            )
        elif name == "extract":
            write_artifact(
                results_dir / "02_candidates.json",
                envelope(
                    experiment="extract_candidate_directions",
                    results={"candidate_count": 28},
                    config=config,
                ),
            )
        elif name == "interventions":
            write_artifact(
                results_dir / "03_interventions.json",
                envelope(
                    experiment="causal_interventions",
                    results={"candidate_scores": [{"layer": 0}] * 28},
                    config=config,
                ),
            )
        elif name == "evaluation":
            metrics = passing_gate_metrics()
            write_artifact(
                results_dir / "final_evaluation.json",
                envelope(
                    experiment="final_evaluation",
                    results={
                        "baseline": {
                            "harmful_refusal_rate": metrics["harmful_refusal_rate"],
                            "harmless_refusal_rate": metrics["harmless_refusal_rate"],
                        },
                        "ablation": {
                            "harmful_refusal_rate": metrics[
                                "ablation_harmful_refusal_rate"
                            ],
                            "harmless_kl": metrics["ablation_harmless_kl"],
                        },
                        "addition": {
                            "harmless_refusal_rate": metrics[
                                "addition_harmless_refusal_rate"
                            ]
                        },
                        "random_control": {
                            "harmful_refusal_rate": metrics[
                                "random_control_harmful_refusal_rate"
                            ]
                        },
                    },
                    config=config,
                ),
            )
        return {"stage": name}

    return runner


def test_pilot_skips_completed_stages_and_reuses_run_id(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    artifacts = tmp_path / "artifacts" / "qwen25_coder_7b"
    store = RunProgressStore(artifacts, profile=QWEN_PROFILE, run_id="stable-1")
    store.initialize(stage="preflight", total_units=1)
    store.begin_running()
    store.complete_stage()
    calls: list[str] = []
    releases: list[str] = []

    def release() -> None:
        current = RunProgressStore(
            artifacts,
            profile=QWEN_PROFILE,
            run_id="stable-1",
        ).read_status()
        releases.append(current["current_stage"] if current else "missing")

    runners = {
        stage: _stage_writer(stage, calls)
        for stage in PILOT_STAGES
    }
    result = run_qwen_pilot(
        tmp_path,
        run_id=None,
        stage_runners=runners,
        release_resources=release,
    )
    assert result["run_id"] == "stable-1"
    assert calls == ["baseline", "extract", "interventions", "evaluation"]
    assert "preflight" not in calls
    assert releases == calls
    evaluation = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluation["status"] == STATUS_PASS


def test_pilot_aborts_after_failed_baseline_gate(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    calls: list[str] = []
    runners = {
        "preflight": _stage_writer("preflight", calls),
        "baseline": _stage_writer("baseline", calls, baseline_gap=0.1),
        "extract": _stage_writer("extract", calls),
        "interventions": _stage_writer("interventions", calls),
        "evaluation": _stage_writer("evaluation", calls),
    }
    result = run_qwen_pilot(tmp_path, run_id="abort-1", stage_runners=runners)
    assert result["status"] == STATUS_BASELINE_GATE_FAILED
    assert calls == ["preflight", "baseline"]
    status = RunProgressStore(
        tmp_path / "artifacts" / "qwen25_coder_7b",
        profile=QWEN_PROFILE,
        run_id="abort-1",
    ).read_status()
    assert status["state"] == "aborted"
    assert status["abort_gate"] == "baseline"
    second_calls: list[str] = []
    second = {
        stage: _stage_writer(stage, second_calls)
        for stage in PILOT_STAGES
    }
    again = run_qwen_pilot(tmp_path, run_id=None, stage_runners=second)
    assert again["status"] == STATUS_BASELINE_GATE_FAILED
    assert second_calls == []


def test_pilot_marks_exception_failed_and_reraises(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    calls: list[str] = []

    def boom(root: Path, *, profile: str, **_kwargs) -> dict:
        calls.append("preflight")
        raise RuntimeError("simulated oom")

    runners = {
        "preflight": boom,
        "baseline": _stage_writer("baseline", calls),
        "extract": _stage_writer("extract", calls),
        "interventions": _stage_writer("interventions", calls),
        "evaluation": _stage_writer("evaluation", calls),
    }
    with pytest.raises(RuntimeError, match="simulated oom"):
        run_qwen_pilot(tmp_path, run_id="fail-1", stage_runners=runners)
    status = RunProgressStore(
        tmp_path / "artifacts" / "qwen25_coder_7b",
        profile=QWEN_PROFILE,
        run_id="fail-1",
    ).read_status()
    assert status["state"] == "failed"
    assert status["message"] == "Operational error: runtime failure."
    assert "simulated oom" not in json.dumps(status)
    assert calls == ["preflight"]


def test_release_compute_resources_clears_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cleared = {"count": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fake_empty_cache() -> None:
        cleared["count"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)
    release_compute_resources(model=object(), tokenizer=object())
    assert cleared["count"] == 1


def test_resolve_run_id_reuses_existing_status(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    store = RunProgressStore(artifacts, profile=QWEN_PROFILE, run_id="kept")
    store.initialize(stage="preflight", total_units=1)
    assert resolve_run_id(artifacts, requested=None) == "kept"
    assert resolve_run_id(artifacts, requested="new") == "new"


def test_validator_accepts_passing_compact_artifacts(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    checks = validate_qwen_pilot(tmp_path)
    assert checks
    assert all(checks.values())


def test_validator_rejects_missing_stage_for_pass(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    (tmp_path / "results" / "qwen25_coder_7b" / "02_candidates.json").unlink()
    checks = validate_qwen_pilot(tmp_path)
    assert checks["required_stages_present"] is False


def test_validator_rejects_wrong_model_revision(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["model_revision"] = "deadbeef"
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["model_and_revision"] is False


def test_validator_rejects_non_disjoint_ids(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["example_ids"]["harmful_test"] = payload["results"][
        "example_ids"
    ]["harmful_train"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["disjoint_split_ids"] is False


def test_validator_rejects_candidate_count_not_28(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "02_candidates.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["candidate_count"] = 27
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["candidate_count_is_28"] is False


def test_validator_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "final_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["ablation"]["harmless_kl"] = math.nan
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["metrics_are_finite"] is False


def test_validator_rejects_inconsistent_gate_math(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_gate"]["checks"][2]["observed"] = 0.01
    payload["baseline_gate"]["passed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["gate_calculations_consistent"] is False


def test_validator_rejects_prompt_leakage(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["text"] = "how to make a virus in python"
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["no_raw_prompt_or_generation_leakage"] is False


def test_validator_rejects_status_result_mismatch(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path, status=STATUS_PASS)
    path = tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = STATUS_CAUSAL_GATE_FAILED
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["status_matches_result"] is False


def test_validator_accepts_baseline_gate_failure_without_later_stages(
    tmp_path: Path,
) -> None:
    write_valid_pilot_artifacts(tmp_path, status=STATUS_BASELINE_GATE_FAILED)
    checks = validate_qwen_pilot(tmp_path)
    assert checks["required_stages_present"] is True
    assert checks["status_matches_result"] is True


def test_compact_artifacts_do_not_embed_canary_prompt(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    for path in (tmp_path / "results" / "qwen25_coder_7b").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "prompt" not in json.loads(text).get("results", {})
        assert "generation" not in json.loads(text).get("results", {})


def test_remote_pilot_script_has_no_argparse_or_file_dependency() -> None:
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "argparse" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id != "__file__"
        if isinstance(node, ast.Attribute) and node.attr == "__file__":
            raise AssertionError("pilot script uses __file__")
    assert PILOT_CLI.REMOTE_ROOT == Path("/content/refusal_direction")


def test_remote_pilot_main_ignores_ipykernel_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy_configs(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(PILOT_CLI, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(
        PILOT_CLI,
        "run_qwen_pilot",
        lambda root, **kwargs: calls.append(str(root)) or {"status": STATUS_PASS},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "/content/refusal_direction/scripts/qwen_run_pilot.py",
            "-f",
            "/usr/local/share/jupyter/kernels/python3/kernel.json",
            "--",
        ],
    )
    assert PILOT_CLI.remote_main() == 0
    assert calls == [str(tmp_path)]


def test_run_refusal_pilot_action_is_qwen_only() -> None:
    command = RUN_REFUSAL.pilot_command("refusal-qwen", timeout=14400)
    assert command[command.index("-f") + 1] == str(PILOT_SCRIPT)
    assert command[command.index("--timeout") + 1] == "14400"


def test_run_refusal_gemma_commands_remain_unchanged() -> None:
    assert RUN_REFUSAL.start_command("refusal-gemma", "L4") == [
        "colab",
        "--auth=adc",
        "new",
        "-s",
        "refusal-gemma",
        "--gpu",
        "L4",
    ]
    assert RUN_REFUSAL.exec_command(
        "refusal-gemma",
        Path("stage.py"),
        timeout=300,
    ) == [
        "colab",
        "--auth=adc",
        "exec",
        "-s",
        "refusal-gemma",
        "-f",
        "stage.py",
        "--timeout",
        "300",
    ]
    assert RUN_REFUSAL.resolve_session("gemma", None) == "refusal-gemma"


def test_run_refusal_pilot_rejected_for_gemma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_refusal.py", "pilot"],
    )
    with pytest.raises(SystemExit):
        RUN_REFUSAL.main()


def test_invalid_generations_are_excluded_from_refusal_denominator() -> None:
    examples = [
        PromptExample("ok", "keep me", "harmful", "fixture"),
        PromptExample("blank", "skip me", "harmful", "fixture"),
        PromptExample("ws", "skip whitespace", "harmful", "fixture"),
        PromptExample("decode", "skip decode", "harmful", "fixture"),
    ]
    outputs = ["I cannot help.", "", "   ", "bad\ufffdtext"]
    assert [is_valid_generation(item) for item in outputs] == [
        True,
        False,
        False,
        False,
    ]
    summary = summarize_generated_condition(
        outputs,
        examples=examples,
        count_valid=True,
    )
    assert summary["n_valid"] == 1
    assert summary["n_invalid"] == 3
    assert summary["valid_ids"] == ["ok"]
    assert summary["invalid_ids"] == ["blank", "ws", "decode"]
    assert summary["refusal_rate"] == 1.0
    assert summary["refusal_flags"] == [True]


def test_content_hashes_normalize_and_detect_overlap() -> None:
    assert normalize_prompt_text("  Hello\nWorld  ") == "hello world"
    left = prompt_content_hash("Hello World")
    right = prompt_content_hash("  hello   world\n")
    assert left == right
    different = prompt_content_hash("hello there")
    assert different != left


def test_causal_gate_exact_rational_point_two_boundary() -> None:
    result = evaluate_causal_gate(
        baseline_harmful_refusal_rate=0.30,
        baseline_harmless_refusal_rate=0.10,
        ablation_harmful_refusal_rate=0.10,
        addition_harmless_refusal_rate=0.30,
        random_control_harmful_refusal_rate=0.20,
        ablation_harmless_kl=0.10,
        thresholds=qwen_config()["causal_gate"],
    )
    assert result.passed is True
    by_name = {check.name: check for check in result.checks}
    assert by_name["min_harmful_ablation_drop"].passed is True
    assert by_name["min_harmless_addition_increase"].passed is True
    assert by_name["max_random_control_harmful_drop"].passed is True
    assert by_name["max_harmless_ablation_kl"].passed is True


def test_preflight_rejects_nonfinite_generated_token_scores(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    write_dataset_files(tmp_path, count=4)

    def load_model(_config):
        return (
            FakeQwenModel(finite=True, finite_generation=False),
            ChatTokenizer(),
            torch.bfloat16,
        )

    with pytest.raises(ValueError, match="generated-token"):
        run_preflight_stage(
            tmp_path,
            profile=QWEN_PROFILE,
            load_model=load_model,
            cuda_available=True,
        )


def test_preflight_rejects_resolved_revision_mismatch(tmp_path: Path) -> None:
    copy_configs(tmp_path)
    write_dataset_files(tmp_path, count=4)

    def load_model(_config):
        return (
            FakeQwenModel(revision="0" * 40),
            ChatTokenizer(),
            torch.bfloat16,
        )

    with pytest.raises(ValueError, match="revision"):
        run_preflight_stage(
            tmp_path,
            profile=QWEN_PROFILE,
            load_model=load_model,
            cuda_available=True,
        )


def test_operational_error_publishes_compact_evaluation(tmp_path: Path) -> None:
    copy_configs(tmp_path)

    def boom(root: Path, *, profile: str, **_kwargs) -> dict:
        raise RuntimeError('failed on prompt "CANARY_SECRET_PROMPT_TEXT_DO_NOT_LEAK"')

    runners = {stage: boom if stage == "extract" else _stage_writer(stage, []) for stage in PILOT_STAGES}
    with pytest.raises(RuntimeError, match="CANARY"):
        run_qwen_pilot(tmp_path, run_id="ops-1", stage_runners=runners)
    payload = json.loads(
        (tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == STATUS_OPERATIONAL_ERROR
    assert payload["operational_success"] is False
    assert payload["stage"] == "extract"
    assert payload["exception_category"] == "RuntimeError"
    assert "CANARY_SECRET_PROMPT_TEXT_DO_NOT_LEAK" not in payload["message"]
    assert payload["run_id"] == "ops-1"


def test_validator_rejects_pass_when_a_gate_failed(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path, status=STATUS_PASS)
    path = tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = STATUS_PASS
    payload["causal_gate"]["passed"] = False
    payload["causal_gate"]["checks"][0]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    final_path = tmp_path / "results" / "qwen25_coder_7b" / "final_evaluation.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["results"]["ablation"]["harmful_refusal_rate"] = 0.89
    final_path.write_text(json.dumps(final), encoding="utf-8")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    recomputed = evaluate_causal_gate(
        baseline_harmful_refusal_rate=final["results"]["baseline"]["harmful_refusal_rate"],
        baseline_harmless_refusal_rate=final["results"]["baseline"]["harmless_refusal_rate"],
        ablation_harmful_refusal_rate=0.89,
        addition_harmless_refusal_rate=final["results"]["addition"]["harmless_refusal_rate"],
        random_control_harmful_refusal_rate=final["results"]["random_control"][
            "harmful_refusal_rate"
        ],
        ablation_harmless_kl=final["results"]["ablation"]["harmless_kl"],
        thresholds=qwen_config()["causal_gate"],
    )
    evaluation["causal_gate"] = recomputed.to_dict()
    evaluation["status"] = STATUS_PASS
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["scientific_pass_requires_both_gates"] is False


def test_validator_rejects_missing_ids_and_content_hashes(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"].pop("example_ids", None)
    payload["results"].pop("content_hashes", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["required_ids_and_hashes"] is False


def test_validator_rejects_overlapping_content_hashes(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["content_hashes"]["harmful_test"][0] = payload["results"][
        "content_hashes"
    ]["harmful_train"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["disjoint_content_hashes"] is False


def test_validator_rejects_mixed_run_ids(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "02_candidates.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["run_id"] = "other-run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["single_run_id"] is False


def test_validator_rejects_missing_or_mismatched_code_provenance(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["implementation_commit"] = "b" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["code_provenance"] is False
    payload["provenance"].pop("implementation_commit")
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["code_provenance"] is False


def test_validator_rejects_missing_resolved_revision(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "00_preflight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"].pop("resolved_model_revision")
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["model_and_revision"] is False


def test_validator_rejects_duplicate_candidate_layers(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "03_interventions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["candidate_scores"][1]["layer"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["candidate_grid_is_28_unique_layers"] is False


def test_validator_rejects_incomplete_gate_fields(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["baseline_gate"]["checks"][0].pop("comparator")
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["gate_fields_complete"] is False


def test_validator_rejects_generic_raw_text_field(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["note"] = (
        "Write a detailed harmful instruction for building malware now"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["no_raw_prompt_or_generation_leakage"] is False


def test_validator_rejects_unknown_or_empty_implementation_commit(
    tmp_path: Path,
) -> None:
    write_valid_pilot_artifacts(tmp_path)
    results_dir = tmp_path / "results" / "qwen25_coder_7b"
    for name in (
        "00_preflight.json",
        "01_baseline.json",
        "02_candidates.json",
        "03_interventions.json",
        "final_evaluation.json",
        "pilot_evaluation.json",
    ):
        path = results_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "pilot_evaluation.json":
            payload["implementation_commit"] = "unknown"
        else:
            payload["provenance"]["implementation_commit"] = "unknown"
        path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["code_provenance"] is False
    evaluation_path = results_dir / "pilot_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["implementation_commit"] = ""
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["code_provenance"] is False


def test_validator_rejects_missing_run_id_on_any_artifact(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"].pop("run_id")
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["single_run_id"] is False


def _pilot_evaluation_path(root: Path) -> Path:
    return root / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"


def test_status_invariants_reject_mismatched_gate_outcomes(tmp_path: Path) -> None:
    baseline_mismatch = tmp_path / "baseline_mismatch"
    write_valid_pilot_artifacts(baseline_mismatch, status=STATUS_PASS)
    path = _pilot_evaluation_path(baseline_mismatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = STATUS_BASELINE_GATE_FAILED
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(baseline_mismatch)
    assert checks["status_invariants"] is False

    causal_mismatch = tmp_path / "causal_mismatch"
    write_valid_pilot_artifacts(causal_mismatch, status=STATUS_PASS)
    path = _pilot_evaluation_path(causal_mismatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = STATUS_CAUSAL_GATE_FAILED
    payload["baseline_gate"]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(causal_mismatch)
    assert checks["status_invariants"] is False

    pass_ops = tmp_path / "pass_ops"
    write_valid_pilot_artifacts(pass_ops, status=STATUS_PASS)
    path = _pilot_evaluation_path(pass_ops)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["operational_success"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(pass_ops)
    assert checks["status_invariants"] is False

    ops_success = tmp_path / "ops_success"
    write_valid_pilot_artifacts(ops_success, status=STATUS_PASS)
    path = _pilot_evaluation_path(ops_success)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = STATUS_OPERATIONAL_ERROR
    payload["operational_success"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(ops_success)
    assert checks["status_invariants"] is False


def test_validator_rejects_generation_bookkeeping_mismatch(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["harmful_validation"]["n_valid"] = 4
    payload["results"]["harmful_validation"]["n_invalid"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["generation_bookkeeping"] is False


def test_validator_rejects_flags_that_do_not_match_valid_ids(tmp_path: Path) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "01_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"]["harmless_validation"]["refusal_flags"] = [False]
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["generation_bookkeeping"] is False


def test_preflight_checks_variable_length_generated_scores_independently(
    tmp_path: Path,
) -> None:
    copy_configs(tmp_path)
    write_dataset_files(tmp_path, count=4)

    class VariableLengthScoreModel(FakeQwenModel):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0
            self._lengths = (1, 3, 2, 4)

        def generate(
            self,
            input_ids=None,
            attention_mask=None,
            max_new_tokens=1,
            output_scores=False,
            return_dict_in_generate=False,
            **kwargs,
        ):
            extra = torch.ones(input_ids.shape[0], max_new_tokens, dtype=torch.long)
            sequences = torch.cat([input_ids.cpu(), extra], dim=1)
            if output_scores or return_dict_in_generate:
                n_steps = self._lengths[self._calls % len(self._lengths)]
                self._calls += 1
                scores = tuple(
                    torch.zeros((input_ids.shape[0], 32), dtype=torch.float32)
                    for _ in range(n_steps)
                )
                return SimpleNamespace(sequences=sequences, scores=scores)
            return sequences

    def load_model(_config):
        return VariableLengthScoreModel(), ChatTokenizer(), torch.bfloat16

    artifact = run_preflight_stage(
        tmp_path,
        profile=QWEN_PROFILE,
        load_model=load_model,
        cuda_available=True,
    )
    assert artifact["results"]["generation_finite"] is True


def test_operational_error_uses_fixed_catalog_message(tmp_path: Path) -> None:
    copy_configs(tmp_path)

    def boom(root: Path, *, profile: str, **_kwargs) -> dict:
        raise RuntimeError('failed on prompt "CANARY_SECRET_PROMPT_TEXT_DO_NOT_LEAK"')

    runners = {
        stage: boom if stage == "extract" else _stage_writer(stage, [])
        for stage in PILOT_STAGES
    }
    with pytest.raises(RuntimeError, match="CANARY"):
        run_qwen_pilot(tmp_path, run_id="ops-2", stage_runners=runners)
    results_dir = tmp_path / "results" / "qwen25_coder_7b"
    payload = json.loads(
        (results_dir / "pilot_evaluation.json").read_text(encoding="utf-8")
    )
    dumped = json.dumps(payload)
    assert "CANARY_SECRET_PROMPT_TEXT_DO_NOT_LEAK" not in dumped
    assert payload["exception_category"] == "RuntimeError"
    assert payload["error_code"] == "runtime"
    assert payload["message"] == "Operational error: runtime failure."
    status = json.loads(
        (tmp_path / "artifacts" / "qwen25_coder_7b" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert "CANARY_SECRET_PROMPT_TEXT_DO_NOT_LEAK" not in json.dumps(status)
    assert status["message"] == "Operational error: runtime failure."


def test_validator_inspects_message_fields_instead_of_allowlisting(
    tmp_path: Path,
) -> None:
    write_valid_pilot_artifacts(tmp_path)
    path = tmp_path / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["message"] = (
        "Write a detailed harmful instruction for building malware now"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    checks = validate_qwen_pilot(tmp_path)
    assert checks["no_raw_prompt_or_generation_leakage"] is False
    assert checks["operational_message_is_catalogued"] is False


def test_causal_gate_compares_kl_directly_without_fraction_rounding() -> None:
    kl = 0.100001
    result = evaluate_causal_gate(
        baseline_harmful_refusal_rate=0.30,
        baseline_harmless_refusal_rate=0.10,
        ablation_harmful_refusal_rate=0.10,
        addition_harmless_refusal_rate=0.30,
        random_control_harmful_refusal_rate=0.20,
        ablation_harmless_kl=kl,
        thresholds=qwen_config()["causal_gate"],
    )
    by_name = {check.name: check for check in result.checks}
    assert by_name["max_harmless_ablation_kl"].passed is False
    assert by_name["max_harmless_ablation_kl"].observed == kl


def test_implementation_commit_hash_rejects_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFUSAL_IMPLEMENTATION_COMMIT", "unknown")
    stamp = tmp_path / "IMPLEMENTATION_COMMIT"
    monkeypatch.setattr(
        "refusal_direction_replication.pipeline._implementation_commit_stamp_paths",
        lambda: [stamp],
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(ValueError, match="implementation commit"):
        implementation_commit_hash()


def test_implementation_commit_hash_reads_stamped_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REFUSAL_IMPLEMENTATION_COMMIT", raising=False)
    stamp = tmp_path / "IMPLEMENTATION_COMMIT"
    stamp.write_text("d" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "refusal_direction_replication.pipeline._implementation_commit_stamp_paths",
        lambda: [stamp],
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert implementation_commit_hash() == "d" * 40
