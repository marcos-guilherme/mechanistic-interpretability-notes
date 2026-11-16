"""Resumable five-stage Qwen code evaluation runtime.

The runtime keeps prompts, model generations, executable HumanEval+ samples,
and token-level readouts in the ignored artifact directory. Only aggregate,
text-free JSON envelopes are written to ``results/``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch

from .artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from .data import PromptExample
from .directions import CandidateDirection, load_direction
from .gates import STATUS_OPERATIONAL_ERROR, STATUS_PASS, classify_operational_error
from .interventions import (
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
)
from .metrics import contains_refusal, is_valid_generation, kl_divergence
from .model import (
    format_chat_instruction,
    resolve_refusal_token_ids,
)
from .pipeline import (
    _format_style,
    _load_model_for_config,
    generate_responses,
    implementation_commit_hash,
    last_token_logits,
    load_config,
    project_paths,
    random_direction_control,
    release_compute_resources,
    temporary_hooks,
    transformer_module_lists,
)
from .qwen_code_eval import (
    CODE_EVAL_PROFILE,
    COEFFICIENTS,
    CONDITIONS,
    DATASETS,
    DATASET_SIZE,
    EVALPLUS_REPOSITORY,
    EVALPLUS_REVISION,
    JLENS_REVISION,
    PURPLE_LLAMA_REPOSITORY,
    PURPLE_LLAMA_REVISION,
    REFUSAL_FIRST_TOKEN_LEXICON,
    EvalRecord,
    aggregate_records,
    build_source_identity,
    compact_contains_leak,
    first_response_position,
    rank_token_ids,
    refusal_lexicon_hash,
    select_records,
    summarize_refusal_ranks,
    write_blind_review_sample,
)
from .qwen_dose import EXPECTED_SOURCE_LAYER, EXPECTED_SOURCE_OFFSET, sha256_file
from .qwen_jlens import (
    EXPECTED_SOURCE_LAYERS,
    assert_jlens_contract,
    wrap_hf_model_for_jlens,
)
from .qwen_pilot import resolve_run_id
from .run_progress import RunProgressStore
from .runtime import seed_everything


CODE_EVAL_STAGES = ("prepare", "generate", "evaluate", "readout", "finalize")
STAGE_FILES = {
    "prepare": "00_prepare.json",
    "generate": "01_generate.json",
    "evaluate": "02_evaluate.json",
    "readout": "03_readout.json",
    "finalize": "04_finalize.json",
}
EVALUATION_FILENAME = "code_eval_evaluation.json"
MANIFEST_FILENAME = "execution_manifest.json"
StageRunner = Callable[..., dict[str, Any]]


def apply_lens_with_hooks(
    lens: Any,
    bound_model: Any,
    rendered_text: str,
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    max_seq_len: int,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
    **apply_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Apply the lens inside the same intervention context as generation."""
    with temporary_hooks(pre_hooks=pre_hooks, forward_hooks=forward_hooks):
        return lens.apply(
            bound_model,
            rendered_text,
            layers=list(layers),
            positions=list(positions),
            max_seq_len=max_seq_len,
            **apply_kwargs,
        )


def _default_gpu_probe() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "device_name": torch.cuda.get_device_name(0) if available else "",
        "total_memory_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if available else 0
        ),
    }


def require_a100_preflight(
    probe: Callable[[], Mapping[str, Any]] = _default_gpu_probe,
) -> dict[str, Any]:
    """Fail closed unless code evaluation is running on a 40 GB-class A100."""
    details = dict(probe())
    if not details.get("cuda_available"):
        raise ValueError("code-eval requires CUDA on an A100")
    if "A100" not in str(details.get("device_name", "")).upper():
        raise ValueError("code-eval requires an NVIDIA A100")
    if int(details.get("total_memory_bytes", 0)) < 39_000_000_000:
        raise ValueError("code-eval requires at least approximately 39 GB GPU memory")
    return details


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage_total_units(stage: str, *, dataset_size: int = DATASET_SIZE) -> int:
    if stage == "prepare":
        return len(DATASETS) * dataset_size
    if stage == "generate":
        return len(DATASETS) * len(CONDITIONS) * dataset_size
    if stage == "evaluate":
        return len(CONDITIONS) * dataset_size
    if stage == "readout":
        return len(DATASETS) * len(CONDITIONS) * dataset_size
    if stage == "finalize":
        return 1
    raise ValueError(f"unknown code-eval stage: {stage}")


def _require_config_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "purplellama_revision": PURPLE_LLAMA_REVISION,
        "evalplus_revision": EVALPLUS_REVISION,
        "jlens_upstream_commit": JLENS_REVISION,
        "dataset_size": DATASET_SIZE,
        "batch_size": 1,
        "ablation_coefficient": COEFFICIENTS["ablation"],
        "addition_coefficient": COEFFICIENTS["addition"],
        "random_control_coefficient": COEFFICIENTS["random_control"],
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"code-eval config {field} must be {value!r}")
    if int(config.get("max_new_tokens", 0)) < 256:
        raise ValueError("code-eval max_new_tokens must be at least 256")
    if int(config.get("evalplus_memory_bytes", 0)) != 1024**3:
        raise ValueError("EvalPlus memory limit must be exactly 1 GiB")


def source_context(
    root: Path,
    config: Mapping[str, Any],
    *,
    load_direction_payload: bool = True,
) -> dict[str, Any]:
    _require_config_contract(config)
    paths = {
        "direction": root / str(config["source_direction"]),
        "lens": root / str(config["source_lens"]),
        "dose_evaluation": root / str(config["source_dose_evaluation"]),
        "jlens_evaluation": root / str(config["source_jlens_evaluation"]),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("code-eval source artifacts missing: " + ", ".join(missing))
    dose = _read_json(paths["dose_evaluation"])
    jlens = _read_json(paths["jlens_evaluation"])
    if dose.get("status") != STATUS_PASS:
        raise ValueError("source dose must remain pass")
    if jlens.get("status") != STATUS_PASS:
        raise ValueError("source J-lens must remain pass")
    if jlens.get("jlens_upstream_commit") != JLENS_REVISION:
        raise ValueError("source J-lens upstream commit changed")
    pair = dose.get("selected_pair") or {}
    if (
        float(pair.get("ablation", -1)) != float(config["ablation_coefficient"])
        or float(pair.get("addition", -1)) != float(config["addition_coefficient"])
    ):
        raise ValueError("source dose coefficients do not match frozen code-eval coefficients")
    context: dict[str, Any] = {
        "source_dose_status": dose["status"],
        "source_dose_run_id": dose.get("run_id"),
        "source_jlens_status": jlens["status"],
        "source_jlens_run_id": jlens.get("run_id"),
        "selected_pair": {
            "ablation": float(pair["ablation"]),
            "addition": float(pair["addition"]),
        },
        "source_artifact_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "paths": paths,
    }
    if load_direction_payload:
        direction = load_direction(paths["direction"])
        if direction.layer != EXPECTED_SOURCE_LAYER:
            raise ValueError(
                f"source direction layer must remain {EXPECTED_SOURCE_LAYER}"
            )
        if direction.position_offset != EXPECTED_SOURCE_OFFSET:
            raise ValueError(
                f"source direction offset must remain {EXPECTED_SOURCE_OFFSET}"
            )
        context["direction"] = direction
        context["selected_layer"] = direction.layer
        context["selected_offset"] = direction.position_offset
    return context


def _provenance(
    *,
    config: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "config_sha256": _config_hash(config),
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": config["model_revision"],
        "purplellama_revision": PURPLE_LLAMA_REVISION,
        "evalplus_revision": EVALPLUS_REVISION,
        "jlens_upstream_commit": JLENS_REVISION,
        "source_dose_run_id": source.get("source_dose_run_id"),
        "source_jlens_run_id": source.get("source_jlens_run_id"),
        "source_artifact_hashes": dict(source["source_artifact_hashes"]),
    }


def _write_stage(
    path: Path,
    *,
    stage: str,
    config: Mapping[str, Any],
    results: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if compact_contains_leak(results):
        raise ValueError(f"{stage} compact results contain raw text")
    write_artifact(
        path,
        ArtifactEnvelope(
            experiment=f"code_eval_{stage}",
            config=dict(config),
            results=dict(results),
            provenance=dict(provenance),
        ),
    )
    return _read_json(path)


def _load_stage(
    path: Path,
    *,
    run_id: str,
    implementation_commit: str,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    provenance = payload.get("provenance") or {}
    expected = {
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "config_sha256": _config_hash(config),
        "source_artifact_hashes": dict(source_hashes),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"stale code-eval checkpoint {path.name}: {field}")
    return payload


def build_evalplus_subprocess(
    *,
    worker: Path,
    problems: Path,
    samples: Path,
    output: Path,
    tempdir: Path,
    memory_bytes: int,
    timeout_seconds: int,
    python: str = sys.executable,
) -> dict[str, Any]:
    command = [
        python,
        str(worker),
        "--problems",
        str(problems),
        "--samples",
        str(samples),
        "--output",
        str(output),
        "--memory-bytes",
        str(memory_bytes),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    environment = dict(os.environ)
    environment["EVALPLUS_MAX_MEMORY_BYTES"] = str(memory_bytes)
    # At the pinned EvalPlus revision this value is passed to min() without
    # integer conversion. Leaving it undefined uses the safe numeric default.
    environment.pop("EVALPLUS_TIMEOUT_PER_TASK", None)
    return {
        "command": command,
        "cwd": tempdir,
        "env": environment,
        "timeout": timeout_seconds * DATASET_SIZE + 60,
    }


def _dataset_source_root() -> Path:
    value = os.environ.get("QWEN_CODE_EVAL_SOURCE_ROOT")
    if value:
        return Path(value)
    return Path("/content/qwen_code_eval_sources")


def _selected_dataset_path(artifacts_dir: Path, dataset: str) -> Path:
    return artifacts_dir / "datasets" / f"{dataset}.jsonl"


def _raw_eval_records(artifacts_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for dataset in DATASETS:
            records.extend(
                _read_jsonl(artifacts_dir / "generations" / f"{dataset}__{condition}.jsonl")
            )
    return records


def _serialize_selected(
    path: Path,
    selected: Sequence[EvalRecord],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    problem_field: bool = False,
) -> None:
    records = []
    for item in selected:
        record: dict[str, Any] = {
            "example_id": item.example_id,
            "content_hash": item.content_hash,
            "prompt": item.text,
            "category": item.category,
            "source_index": item.source_index,
            "task_id": item.task_id,
        }
        if problem_field:
            record["problem"] = dict(source_rows[item.source_index])
        records.append(record)
    _write_jsonl(path, records)


def run_prepare_stage(
    *,
    root: Path,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    **_kwargs: Any,
) -> dict[str, Any]:
    source_root = _dataset_source_root()
    purple_root = source_root / "PurpleLlama"
    mitre_path = purple_root / str(config["mitre_relative_path"])
    frr_path = purple_root / str(config["mitre_frr_relative_path"])
    mitre_rows = json.loads(mitre_path.read_text(encoding="utf-8"))
    frr_rows = json.loads(frr_path.read_text(encoding="utf-8"))
    from evalplus.data import get_human_eval_plus
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION

    if HUMANEVAL_PLUS_VERSION != config["evalplus_dataset_version"]:
        raise ValueError("installed HumanEval+ dataset version changed")
    problems = get_human_eval_plus()
    humaneval_rows = [
        {"task_id": task_id, **problem} for task_id, problem in sorted(problems.items())
    ]
    specifications = (
        (
            "mitre",
            mitre_rows,
            "mutated_prompt",
            "mitre_category",
            None,
        ),
        (
            "mitre_frr",
            frr_rows,
            "mutated_prompt",
            "attack_type",
            None,
        ),
        (
            "humaneval_plus",
            humaneval_rows,
            "prompt",
            None,
            "task_id",
        ),
    )
    selections: dict[str, Any] = {}
    completed = 0
    for offset, (dataset, rows, text_field, category_field, task_id_field) in enumerate(
        specifications
    ):
        selected, metadata = select_records(
            rows,
            dataset=dataset,
            text_field=text_field,
            category_field=category_field,
            task_id_field=task_id_field,
            size=int(config["dataset_size"]),
            seed=int(config["selection_seed"]) + offset,
        )
        _serialize_selected(
            _selected_dataset_path(artifacts_dir, dataset),
            selected,
            source_rows=rows,
            problem_field=dataset == "humaneval_plus",
        )
        selections[dataset] = metadata
        completed += len(selected)
        store.update_progress(completed_units=completed)
    selections["source_files"] = {
        "mitre": build_source_identity(
            mitre_path,
            repository=PURPLE_LLAMA_REPOSITORY,
            revision=PURPLE_LLAMA_REVISION,
            relative_path=str(config["mitre_relative_path"]),
        ),
        "mitre_frr": build_source_identity(
            frr_path,
            repository=PURPLE_LLAMA_REPOSITORY,
            revision=PURPLE_LLAMA_REVISION,
            relative_path=str(config["mitre_frr_relative_path"]),
        ),
        "humaneval_plus": {
            "repository": EVALPLUS_REPOSITORY,
            "revision": EVALPLUS_REVISION,
            "dataset_version": config["evalplus_dataset_version"],
            "selected_raw_sha256": sha256_file(
                _selected_dataset_path(artifacts_dir, "humaneval_plus")
            ),
        },
    }
    return selections


def _condition_hooks(
    *,
    condition: str,
    model: Any,
    direction: CandidateDirection,
    control: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[Sequence[tuple[Any, Any]], Sequence[tuple[Any, Any]]]:
    blocks, attentions, mlps = transformer_module_lists(model)
    if condition == "baseline":
        return (), ()
    if condition == "ablation":
        return directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            direction.vector,
            coefficient=float(config["ablation_coefficient"]),
        )
    if condition == "random_control":
        return directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            control,
            coefficient=float(config["random_control_coefficient"]),
        )
    if condition == "addition":
        return (
            [
                (
                    blocks[direction.layer],
                    residual_intervention_pre_hook(
                        direction.vector,
                        mode="add",
                        coefficient=float(config["addition_coefficient"]),
                    ),
                )
            ],
            (),
        )
    raise ValueError(f"unknown condition: {condition}")


def _generation_checkpoint_valid(
    path: Path,
    *,
    run_id: str,
    implementation_commit: str,
    source_hashes: Mapping[str, str],
    expected: int,
) -> bool:
    if not path.is_file():
        return False
    records = _read_jsonl(path)
    if len(records) != expected:
        return False
    return all(
        record.get("run_id") == run_id
        and record.get("implementation_commit") == implementation_commit
        and record.get("source_artifact_hashes") == dict(source_hashes)
        for record in records
    )


def _load_eval_examples(artifacts_dir: Path, dataset: str) -> tuple[list[PromptExample], list[dict[str, Any]]]:
    records = _read_jsonl(_selected_dataset_path(artifacts_dir, dataset))
    label = "harmful" if dataset == "mitre" else "harmless"
    examples = [
        PromptExample(
            id=str(record["example_id"]),
            text=str(record["prompt"]),
            label=label,  # type: ignore[arg-type]
            source=dataset,
        )
        for record in records
    ]
    return examples, records


def run_generate_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    load_model: Callable[..., Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    loader = load_model or _load_model_for_config
    model = tokenizer = None
    completed = 0
    try:
        model, tokenizer, _dtype = loader(config)
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        if resolved != config["model_revision"]:
            raise ValueError("resolved model revision differs from pinned revision")
        direction: CandidateDirection = source["direction"]
        control = random_direction_control(
            direction.vector,
            seed=int(config["random_control_seed"]),
        )
        lexicon_ids = resolve_refusal_token_ids(
            tokenizer, REFUSAL_FIRST_TOKEN_LEXICON
        )
        atomic_write_json(
            artifacts_dir / "refusal_lexicon.json",
            {
                "phrases": list(REFUSAL_FIRST_TOKEN_LEXICON),
                "token_ids": list(lexicon_ids),
                "lexicon_sha256": refusal_lexicon_hash(),
            },
        )
        for dataset in DATASETS:
            examples, _selected = _load_eval_examples(artifacts_dir, dataset)
            baseline_logits_path = artifacts_dir / "logits" / f"{dataset}__baseline.pt"
            for condition in CONDITIONS:
                output_path = (
                    artifacts_dir / "generations" / f"{dataset}__{condition}.jsonl"
                )
                if _generation_checkpoint_valid(
                    output_path,
                    run_id=run_id,
                    implementation_commit=implementation_commit,
                    source_hashes=source["source_artifact_hashes"],
                    expected=len(examples),
                ):
                    completed += len(examples)
                    store.update_progress(completed_units=completed)
                    continue
                pre_hooks, forward_hooks = _condition_hooks(
                    condition=condition,
                    model=model,
                    direction=direction,
                    control=control,
                    config=config,
                )
                logits = last_token_logits(
                    model,
                    tokenizer,
                    examples,
                    batch_size=1,
                    format_style=_format_style(dict(config)),
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                )
                if condition == "baseline":
                    _atomic_torch_save(
                        {
                            "run_id": run_id,
                            "implementation_commit": implementation_commit,
                            "source_artifact_hashes": dict(
                                source["source_artifact_hashes"]
                            ),
                            "logits": logits,
                        },
                        baseline_logits_path,
                    )
                    kl_values: list[float | None] = [None] * len(examples)
                else:
                    baseline_payload = torch.load(
                        baseline_logits_path, map_location="cpu", weights_only=True
                    )
                    if baseline_payload.get("run_id") != run_id:
                        raise ValueError("stale baseline logits run_id")
                    kl_values = [
                        float(item)
                        for item in kl_divergence(
                            baseline_payload["logits"], logits
                        ).tolist()
                    ]
                responses = generate_responses(
                    model,
                    tokenizer,
                    examples,
                    batch_size=1,
                    max_new_tokens=int(config["max_new_tokens"]),
                    format_style=_format_style(dict(config)),
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                    invalid_on_decode_error=True,
                )
                raw_records = []
                for example, response, kl_value in zip(examples, responses, kl_values):
                    token_ids = tokenizer.encode(response, add_special_tokens=False)
                    raw_records.append(
                        {
                            "run_id": run_id,
                            "implementation_commit": implementation_commit,
                            "source_artifact_hashes": dict(
                                source["source_artifact_hashes"]
                            ),
                            "dataset": dataset,
                            "condition": condition,
                            "coefficient": COEFFICIENTS[condition],
                            "example_id": example.id,
                            "prompt": example.text,
                            "response": response,
                            "valid": is_valid_generation(response),
                            "refusal": contains_refusal(response),
                            "kl_from_baseline": kl_value,
                            "first_response_token_id": (
                                int(token_ids[0]) if token_ids else None
                            ),
                        }
                    )
                _write_jsonl(output_path, raw_records)
                completed += len(examples)
                store.update_progress(completed_units=completed)
        return {
            "conditions": {
                condition: {"coefficient": COEFFICIENTS[condition]}
                for condition in CONDITIONS
            },
            "datasets": {dataset: DATASET_SIZE for dataset in DATASETS},
            "selected_layer": direction.layer,
            "selected_offset": direction.position_offset,
            "random_control_seed": int(config["random_control_seed"]),
            "lexicon_token_ids": list(lexicon_ids),
            "lexicon_sha256": refusal_lexicon_hash(),
        }
    finally:
        release_compute_resources(model, tokenizer)


def run_evaluate_stage(
    *,
    root: Path,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    **_kwargs: Any,
) -> dict[str, Any]:
    selected = _read_jsonl(_selected_dataset_path(artifacts_dir, "humaneval_plus"))
    problem_path = artifacts_dir / "evalplus" / "selected_problems.jsonl"
    _write_jsonl(problem_path, [record["problem"] for record in selected])
    worker = root / "scripts" / "qwen_evalplus_worker.py"
    completed = 0
    condition_results: dict[str, Any] = {}
    for condition in CONDITIONS:
        generated = _read_jsonl(
            artifacts_dir / "generations" / f"humaneval_plus__{condition}.jsonl"
        )
        samples_path = artifacts_dir / "evalplus" / f"samples__{condition}.jsonl"
        output_path = artifacts_dir / "evalplus" / f"results__{condition}.json"
        if not output_path.is_file():
            by_id = {record["example_id"]: record for record in selected}
            samples = []
            for record in generated:
                selected_record = by_id[record["example_id"]]
                problem = selected_record["problem"]
                samples.append(
                    {
                        "task_id": problem["task_id"],
                        "solution": str(problem["prompt"]) + str(record["response"]),
                    }
                )
            _write_jsonl(samples_path, samples)
            sandbox = artifacts_dir / "evalplus" / f"temp__{condition}"
            sandbox.mkdir(parents=True, exist_ok=True)
            spec = build_evalplus_subprocess(
                worker=worker,
                problems=problem_path,
                samples=samples_path,
                output=output_path,
                tempdir=sandbox,
                memory_bytes=int(config["evalplus_memory_bytes"]),
                timeout_seconds=int(config["evalplus_timeout_seconds"]),
            )
            subprocess.run(
                spec["command"],
                cwd=spec["cwd"],
                env=spec["env"],
                timeout=spec["timeout"],
                check=True,
            )
        payload = _read_json(output_path)
        records = payload.get("records") or []
        if len(records) != DATASET_SIZE:
            raise ValueError(f"EvalPlus {condition} did not evaluate 32 tasks")
        condition_results[condition] = {
            "n": len(records),
            "pass_count": sum(bool(record["plus_pass"]) for record in records),
            "pass_at_1": sum(bool(record["plus_pass"]) for record in records)
            / len(records),
        }
        completed += len(records)
        store.update_progress(completed_units=completed)
    return {
        "conditions": condition_results,
        "memory_bytes": int(config["evalplus_memory_bytes"]),
        "timeout_seconds_per_task": int(config["evalplus_timeout_seconds"]),
        "parallelism": 1,
        "tempdir_isolated": True,
        "reliability_guard": True,
        "security_limitation": "evalplus_reliability_guard_is_not_a_complete_security_sandbox",
        "executed_datasets": ["humaneval_plus"],
        "never_executed_datasets": ["mitre", "mitre_frr"],
    }


def _readout_checkpoint_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    records = _read_jsonl(path)
    observed = {
        (
            str(record.get("example_id")),
            str(record.get("position_kind")),
            int(record.get("layer", -1)),
        )
        for record in records
    }
    if len(records) != len(observed):
        return False
    example_ids = sorted({item[0] for item in observed})
    if len(example_ids) != DATASET_SIZE:
        return False
    normalized_expected = {
        (example_id, position, layer)
        for example_id in example_ids
        for position in ("instruction_last", "response_first")
        for layer in EXPECTED_SOURCE_LAYERS
    }
    return observed == normalized_expected


def run_readout_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any],
    load_model: Callable[..., Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    assert_jlens_contract()
    loader = load_model or _load_model_for_config
    model = tokenizer = None
    completed = 0
    all_records: list[dict[str, Any]] = []
    try:
        model, tokenizer, _dtype = loader(config)
        bound = wrap_hf_model_for_jlens(model, tokenizer)
        from jlens.lens import JacobianLens

        lens = JacobianLens.load(str(source["paths"]["lens"]))
        source_layers = [int(layer) for layer in lens.source_layers]
        direction: CandidateDirection = source["direction"]
        control = random_direction_control(
            direction.vector,
            seed=int(config["random_control_seed"]),
        )
        lexicon_payload = _read_json(artifacts_dir / "refusal_lexicon.json")
        lexicon_ids = tuple(int(item) for item in lexicon_payload["token_ids"])
        for dataset in DATASETS:
            for condition in CONDITIONS:
                checkpoint = (
                    artifacts_dir / "readout" / f"{dataset}__{condition}.jsonl"
                )
                if _readout_checkpoint_complete(checkpoint):
                    records = _read_jsonl(checkpoint)
                    all_records.extend(records)
                    completed += DATASET_SIZE
                    store.update_progress(completed_units=completed)
                    continue
                generated = _read_jsonl(
                    artifacts_dir / "generations" / f"{dataset}__{condition}.jsonl"
                )
                pre_hooks, forward_hooks = _condition_hooks(
                    condition=condition,
                    model=model,
                    direction=direction,
                    control=control,
                    config=config,
                )
                records = []
                for item in generated:
                    rendered = format_chat_instruction(tokenizer, str(item["prompt"]))
                    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                    if not prompt_ids:
                        raise ValueError("rendered chat prompt encoded to zero tokens")
                    first_id = item.get("first_response_token_id")
                    round_trip = (
                        first_response_position(tokenizer, rendered, int(first_id))
                        if first_id is not None
                        else {"available": False, "reason": "empty_response"}
                    )
                    positions = [len(prompt_ids) - 1]
                    readout_text = rendered
                    kinds = ["instruction_last"]
                    if round_trip["available"]:
                        decoded = tokenizer.decode(
                            [int(first_id)],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        readout_text += decoded
                        positions.append(int(round_trip["position"]))
                        kinds.append("response_first")
                    jlens_logits, _model_logits, _ids = apply_lens_with_hooks(
                        lens,
                        bound,
                        readout_text,
                        layers=source_layers,
                        positions=positions,
                        max_seq_len=int(config["jlens_max_seq_len"]),
                        pre_hooks=pre_hooks,
                        forward_hooks=forward_hooks,
                        use_jacobian=True,
                    )
                    for position_index, kind in enumerate(kinds):
                        for layer in source_layers:
                            records.append(
                                {
                                    "dataset": dataset,
                                    "condition": condition,
                                    "example_id": item["example_id"],
                                    "position_kind": kind,
                                    "layer": layer,
                                    "available": True,
                                    "token_ranks": rank_token_ids(
                                        jlens_logits[layer][position_index],
                                        lexicon_ids,
                                    ),
                                }
                            )
                    if not round_trip["available"]:
                        for layer in source_layers:
                            records.append(
                                {
                                    "dataset": dataset,
                                    "condition": condition,
                                    "example_id": item["example_id"],
                                    "position_kind": "response_first",
                                    "layer": layer,
                                    "available": False,
                                    "unavailable_reason": round_trip["reason"],
                                }
                            )
                _write_jsonl(checkpoint, records)
                all_records.extend(records)
                completed += DATASET_SIZE
                store.update_progress(completed_units=completed)
        return {
            "lexicon_token_ids": list(lexicon_ids),
            "lexicon_sha256": lexicon_payload["lexicon_sha256"],
            "source_layers": source_layers,
            "positions": ["instruction_last", "response_first"],
            "rank_statistic": "minimum_rank_across_preregistered_first_token_lexicon",
            "aggregates": summarize_refusal_ranks(
                all_records, lexicon_token_ids=lexicon_ids
            ),
        }
    finally:
        release_compute_resources(model, tokenizer)


def run_finalize_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    results_dir: Path,
    store: RunProgressStore,
    **_kwargs: Any,
) -> dict[str, Any]:
    records = _raw_eval_records(artifacts_dir)
    selected = _read_jsonl(_selected_dataset_path(artifacts_dir, "humaneval_plus"))
    task_by_example = {
        record["example_id"]: record["problem"]["task_id"] for record in selected
    }
    evalplus_by_condition: dict[str, dict[str, bool]] = {}
    for condition in CONDITIONS:
        payload = _read_json(
            artifacts_dir / "evalplus" / f"results__{condition}.json"
        )
        evalplus_by_condition[condition] = {
            str(item["task_id"]): bool(item["plus_pass"])
            for item in payload["records"]
        }
    compact_records = []
    for record in records:
        evalplus_pass = None
        if record["dataset"] == "humaneval_plus":
            task_id = task_by_example[record["example_id"]]
            evalplus_pass = evalplus_by_condition[record["condition"]][task_id]
        compact_records.append(
            {
                "dataset": record["dataset"],
                "condition": record["condition"],
                "example_id": record["example_id"],
                "valid": record["valid"],
                "refusal": record["refusal"],
                "kl_from_baseline": record["kl_from_baseline"],
                "evalplus_pass": evalplus_pass,
            }
        )
    review = write_blind_review_sample(
        records,
        sample_path=artifacts_dir / "manual_review_blind.jsonl",
        key_path=artifacts_dir / "manual_review_key.json",
        sample_size=int(config["manual_review_sample_size"]),
        seed=int(config["manual_review_seed"]),
    )
    readout = _read_json(results_dir / STAGE_FILES["readout"])["results"]
    store.update_progress(completed_units=1)
    return {
        "status": STATUS_PASS,
        "operational_success": True,
        **aggregate_records(compact_records),
        "jlens": readout["aggregates"],
        **review,
        "scientific_limitations": list(config["scientific_limitations"]),
    }


def _default_stage_runner(stage: str, **kwargs: Any) -> dict[str, Any]:
    functions = {
        "prepare": run_prepare_stage,
        "generate": run_generate_stage,
        "evaluate": run_evaluate_stage,
        "readout": run_readout_stage,
        "finalize": run_finalize_stage,
    }
    return functions[stage](**kwargs)


def _write_terminal_files(
    *,
    results_dir: Path,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": provenance["run_id"],
        "implementation_commit": provenance["implementation_commit"],
        "model_revision": config["model_revision"],
        "source_artifact_hashes": provenance["source_artifact_hashes"],
        **dict(evaluation),
    }
    atomic_write_json(results_dir / EVALUATION_FILENAME, payload)
    _write_stage(
        results_dir / MANIFEST_FILENAME,
        stage="manifest",
        config=config,
        results={
            "status": payload["status"],
            "completed_stages": list(CODE_EVAL_STAGES),
            "result_files": [
                *[STAGE_FILES[stage] for stage in CODE_EVAL_STAGES],
                EVALUATION_FILENAME,
                MANIFEST_FILENAME,
            ],
            "human_review_status": payload.get("human_review_status", "pending"),
        },
        provenance=provenance,
    )


def _reconcile_terminal_status(
    store: RunProgressStore,
    evaluation: Mapping[str, Any],
) -> None:
    if evaluation.get("status") == STATUS_PASS:
        stage = "finalize"
        state = "completed"
        event = "stage_completed"
        completed = total = 1
    else:
        stage = str(evaluation.get("stage") or "prepare")
        state = "failed"
        event = "failed"
        completed = 0
        total = stage_total_units(stage)
    store.write_terminal_snapshot(
        {
            "schema_version": 1,
            "profile": store.profile,
            "run_id": store.run_id,
            "current_stage": stage,
            "state": state,
            "completed_units": completed,
            "total_units": total,
        }
    )
    if not store.has_matching_event(
        event=event,
        state=state,
        stage=stage,
        completed_units=completed,
        total_units=total,
    ):
        store.append_public_event(
            {
                "event": event,
                "stage": stage,
                "state": state,
                "completed_units": completed,
                "total_units": total,
            }
        )


def run_qwen_code_eval(
    root: Path,
    *,
    profile: str = CODE_EVAL_PROFILE,
    run_id: str | None = None,
    implementation_commit: str | None = None,
    stage_runner: StageRunner | None = None,
    load_model: Callable[..., Any] | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] = _default_gpu_probe,
) -> dict[str, Any]:
    require_a100_preflight(gpu_probe)
    root = root.resolve()
    config = load_config(root, profile=profile)
    _require_config_contract(config)
    results_dir, artifacts_dir = project_paths(root, config)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source = source_context(root, config, load_direction_payload=stage_runner is None)
    resolved_run = resolve_run_id(artifacts_dir, requested=run_id)
    commit = implementation_commit or implementation_commit_hash()
    provenance = _provenance(
        config=config,
        run_id=resolved_run,
        implementation_commit=commit,
        source=source,
    )
    store = RunProgressStore(artifacts_dir, profile=profile, run_id=resolved_run)
    terminal_path = results_dir / EVALUATION_FILENAME
    if terminal_path.is_file():
        terminal = _read_json(terminal_path)
        if terminal.get("run_id") != resolved_run:
            raise ValueError("stale code-eval terminal run_id")
        if terminal.get("implementation_commit") != commit:
            raise ValueError("stale code-eval terminal implementation_commit")
        if terminal.get("source_artifact_hashes") != source["source_artifact_hashes"]:
            raise ValueError("stale code-eval terminal source hashes")
        _reconcile_terminal_status(store, terminal)
        return terminal
    runner = stage_runner or _default_stage_runner
    last: dict[str, Any] | None = None
    for stage in CODE_EVAL_STAGES:
        checkpoint_path = results_dir / STAGE_FILES[stage]
        checkpoint = _load_stage(
            checkpoint_path,
            run_id=resolved_run,
            implementation_commit=commit,
            config=config,
            source_hashes=source["source_artifact_hashes"],
        )
        if checkpoint is not None:
            last = checkpoint["results"]
            continue
        try:
            store.initialize(
                stage=stage,
                total_units=stage_total_units(
                    stage, dataset_size=int(config["dataset_size"])
                ),
            )
            store.begin_running()
            results = runner(
                stage,
                root=root,
                config=config,
                results_dir=results_dir,
                artifacts_dir=artifacts_dir,
                store=store,
                source=source,
                run_id=resolved_run,
                implementation_commit=commit,
                load_model=load_model,
            )
            _write_stage(
                checkpoint_path,
                stage=stage,
                config=config,
                results=results,
                provenance=provenance,
            )
            last = results
            if stage == "finalize":
                _write_terminal_files(
                    results_dir=results_dir,
                    config=config,
                    provenance=provenance,
                    evaluation=results,
                )
            store.complete_stage()
        except Exception as exc:
            category, error_code, message = classify_operational_error(exc)
            failure = {
                "status": STATUS_OPERATIONAL_ERROR,
                "operational_success": False,
                "stage": stage,
                "exception_category": category,
                "error_code": error_code,
                "message": message,
                "human_review_status": "pending",
            }
            _write_terminal_files(
                results_dir=results_dir,
                config=config,
                provenance=provenance,
                evaluation=failure,
            )
            try:
                store.fail(message)
            except Exception:
                pass
            raise
    if last is None:
        raise ValueError("code evaluation produced no result")
    return _read_json(terminal_path)
