from __future__ import annotations

import json
from pathlib import Path

import pytest

from refusal_direction_replication.artifacts import atomic_write_json
from refusal_direction_replication.qwen_code_eval import (
    CONDITIONS,
    DATASETS,
    EVALPLUS_REVISION,
    JLENS_REVISION,
    PURPLE_LLAMA_REVISION,
)


def _config(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "qwen25_coder_7b_code_eval.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "model_revision": "c" * 40,
                "dtype": "bfloat16",
                "batch_size": 1,
                "max_new_tokens": 256,
                "format_style": "chat_template",
                "seed": 42,
                "selection_seed": 2026,
                "dataset_size": 32,
                "purplellama_revision": PURPLE_LLAMA_REVISION,
                "evalplus_revision": EVALPLUS_REVISION,
                "jlens_upstream_commit": JLENS_REVISION,
                "ablation_coefficient": 0.2,
                "addition_coefficient": 0.5,
                "random_control_coefficient": 0.2,
                "results_dir": "results/qwen25_coder_7b_code_eval",
                "artifacts_dir": "artifacts/qwen25_coder_7b_code_eval",
                "source_direction": "artifacts/qwen25_coder_7b/selected_direction.pt",
                "source_lens": "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt",
                "source_dose_evaluation": "results/qwen25_coder_7b_dose/dose_followup_evaluation.json",
                "source_jlens_evaluation": "results/qwen25_coder_7b_jlens/jlens_evaluation.json",
                "evalplus_memory_bytes": 1073741824,
                "evalplus_timeout_seconds": 60,
            }
        ),
        encoding="utf-8",
    )


def test_source_context_requires_passing_dose_lens_and_frozen_coefficients(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_runtime import source_context

    _config(tmp_path)
    direction = tmp_path / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        tmp_path
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    dose = tmp_path / "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    jlens = tmp_path / "results/qwen25_coder_7b_jlens/jlens_evaluation.json"
    dose.parent.mkdir(parents=True)
    jlens.parent.mkdir(parents=True)
    atomic_write_json(
        dose,
        {
            "status": "pass",
            "run_id": "dose-run",
            "selected_pair": {"ablation": 0.2, "addition": 0.5},
        },
    )
    atomic_write_json(
        jlens,
        {
            "status": "pass",
            "run_id": "jlens-run",
            "jlens_upstream_commit": JLENS_REVISION,
        },
    )
    config = json.loads(
        (tmp_path / "configs/qwen25_coder_7b_code_eval.json").read_text()
    )
    context = source_context(tmp_path, config, load_direction_payload=False)
    assert context["source_dose_status"] == "pass"
    assert context["source_jlens_status"] == "pass"
    assert context["selected_pair"] == {"ablation": 0.2, "addition": 0.5}
    assert set(context["source_artifact_hashes"]) == {
        "direction",
        "lens",
        "dose_evaluation",
        "jlens_evaluation",
    }

    dose_payload = json.loads(dose.read_text())
    dose_payload["selected_pair"]["addition"] = 0.35
    atomic_write_json(dose, dose_payload)
    with pytest.raises(ValueError, match="coefficient"):
        source_context(tmp_path, config, load_direction_payload=False)


def test_evalplus_subprocess_contract_limits_memory_timeout_and_tempdir(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_runtime import (
        build_evalplus_subprocess,
    )

    spec = build_evalplus_subprocess(
        worker=Path("/content/refusal_direction/scripts/qwen_evalplus_worker.py"),
        problems=tmp_path / "problems.jsonl",
        samples=tmp_path / "samples.jsonl",
        output=tmp_path / "output.json",
        tempdir=tmp_path / "sandbox",
        memory_bytes=1024**3,
        timeout_seconds=60,
        python="python",
    )
    assert spec["command"][0] == "python"
    assert "--memory-bytes" in spec["command"]
    assert spec["env"]["EVALPLUS_MAX_MEMORY_BYTES"] == str(1024**3)
    # The pinned EvalPlus revision passes this environment value directly to
    # min(), so defining it as an environment string causes a TypeError.
    assert "EVALPLUS_TIMEOUT_PER_TASK" not in spec["env"]
    assert spec["cwd"] == tmp_path / "sandbox"
    assert spec["timeout"] == 60 * 32 + 60


def test_orchestrator_resumes_completed_stages_and_finalizes_pending_review(
    tmp_path: Path,
) -> None:
    from refusal_direction_replication.qwen_code_eval_runtime import (
        CODE_EVAL_STAGES,
        run_qwen_code_eval,
    )

    _config(tmp_path)
    direction = tmp_path / "artifacts/qwen25_coder_7b/selected_direction.pt"
    lens = (
        tmp_path
        / "artifacts/qwen25_coder_7b_jlens/qwen25_coder_7b_jacobian_lens.pt"
    )
    direction.parent.mkdir(parents=True)
    lens.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    lens.write_bytes(b"lens")
    dose = tmp_path / "results/qwen25_coder_7b_dose/dose_followup_evaluation.json"
    jlens = tmp_path / "results/qwen25_coder_7b_jlens/jlens_evaluation.json"
    dose.parent.mkdir(parents=True)
    jlens.parent.mkdir(parents=True)
    atomic_write_json(
        dose,
        {
            "status": "pass",
            "run_id": "dose-run",
            "selected_pair": {"ablation": 0.2, "addition": 0.5},
        },
    )
    atomic_write_json(
        jlens,
        {
            "status": "pass",
            "run_id": "jlens-run",
            "jlens_upstream_commit": JLENS_REVISION,
        },
    )
    calls: list[str] = []

    def runner(stage: str, **_kwargs):
        calls.append(stage)
        if stage == "finalize":
            return {
                "status": "pass",
                "human_review_status": "pending",
                "limitation": "pending_real_human_review",
            }
        return {"stage": stage, "completed": True}

    first = run_qwen_code_eval(
        tmp_path,
        run_id="code-eval-run",
        implementation_commit="a" * 40,
        stage_runner=runner,
    )
    assert calls == list(CODE_EVAL_STAGES)
    assert first["status"] == "pass"
    assert first["human_review_status"] == "pending"

    calls.clear()
    second = run_qwen_code_eval(
        tmp_path,
        run_id="code-eval-run",
        implementation_commit="a" * 40,
        stage_runner=runner,
    )
    assert calls == []
    assert second == first
    status = json.loads(
        (
            tmp_path / "artifacts/qwen25_coder_7b_code_eval/status.json"
        ).read_text()
    )
    assert status["state"] == "completed"
    assert status["current_stage"] == "finalize"


def test_stage_dimensions_cover_every_dataset_condition() -> None:
    from refusal_direction_replication.qwen_code_eval_runtime import stage_total_units

    assert stage_total_units("prepare") == 3 * 32
    assert stage_total_units("generate") == len(DATASETS) * len(CONDITIONS) * 32
    assert stage_total_units("evaluate") == len(CONDITIONS) * 32
    assert stage_total_units("readout") == len(DATASETS) * len(CONDITIONS) * 32
    assert stage_total_units("finalize") == 1
