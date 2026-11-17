"""Synchronized dual-lens readout during one-pass causal generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from .data import prompt_content_hash
from .directions import load_direction
from .gates import (
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    classify_operational_error,
)
from .metrics import contains_refusal, is_valid_generation
from .model import format_chat_instruction
from .pipeline import (
    _load_model_for_config,
    implementation_commit_hash,
    load_config,
    project_paths,
    random_direction_control,
    release_compute_resources,
    temporary_hooks,
    transformer_module_lists,
)
from .qwen_causal_jlens import (
    CAUSAL_JLENS_PROFILE,
    CONDITIONS,
    DATASETS,
    FROZEN_COHORT_DIGEST,
    FROZEN_PREPARE_SHA256,
    FROZEN_EOS_TOKEN_ID,
    FROZEN_SOURCE_DOSE_RUN_ID,
    FROZEN_V1_EVALUATION_SHA256,
    FROZEN_V1_RUN_ID,
    FROZEN_V2_EVALUATION_SHA256,
    FROZEN_V2_RUN_ID,
    LENS_VERSIONS,
    PRIMARY_LAYER,
    PRODUCTION_DATASET_SIZE,
    READOUT_LAYERS,
    READOUT_STEPS,
    TEXT_LEAK_KEYS,
    aggregate_fidelity,
    cluster_bootstrap_difference,
    cohort_digest,
    compact_contains_text_leak,
    observation_key,
    summarize_behavior_changes,
)
from .qwen_code_eval import (
    PURPLE_LLAMA_REPOSITORY,
    PURPLE_LLAMA_REVISION,
    build_source_identity,
)
from .qwen_dose import EXPECTED_SOURCE_LAYER, EXPECTED_SOURCE_OFFSET, sha256_file
from .qwen_jlens import (
    JLENS_UPSTREAM_COMMIT,
    QWEN_MODEL,
    QWEN_REVISION,
    assert_jlens_contract,
    rank_one_based,
    require_finite_logits,
    wrap_hf_model_for_jlens,
)
from .qwen_lab_readout import (
    assert_incremental_readout_parity,
    incremental_logits_from_residual,
    verify_frozen_lens_hash,
)
from .qwen_lab_runtime import condition_hook_specs
from .qwen_pilot import resolve_run_id
from .run_progress import RunProgressStore
from .runtime import seed_everything

StageRunner = Callable[..., dict[str, Any]]
LoadModel = Callable[..., tuple[Any, Any, Any]]
LoadReadout = Callable[..., Any]
ParityCheck = Callable[..., None]


CHECKPOINT_IDENTITY_FIELDS = (
    "run_id",
    "implementation_commit",
    "config_hash",
    "source_artifact_hashes",
    "model_revision",
    "dataset_ids",
    "dataset_hashes",
)
SOURCE_ARTIFACT_KEYS = (
    "prepare",
    "direction",
    "dose_evaluation",
    "v1_lens",
    "v2_lens",
)
CAUSAL_STAGES = ("prepare", "pilot", "panel", "analysis", "finalize")
STAGE_FILES = {
    "prepare": "00_prepare.json",
    "pilot": "01_pilot.json",
    "panel": "02_panel.json",
    "analysis": "03_analysis.json",
}
EVALUATION_FILENAME = "causal_jlens_evaluation.json"
MANIFEST_FILENAME = "execution_manifest.json"
CHECKPOINT_FILENAME = "checkpoint.json"
RESPONSES_FILENAME = "responses.jsonl"
OBSERVATIONS_FILENAME = "observations.jsonl"
PILOT_PROMPTS_PER_DATASET = 2
OBSERVATION_KEYS = (
    "dataset",
    "example_id",
    "condition",
    "generation_step",
    "layer",
    "lens_version",
    "rank",
    "target_token_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        payload[str(key)] = str(value) if isinstance(value, Path) else value
    return payload


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_config_contract(config: Mapping[str, Any]) -> None:
    if str(config.get("model_revision")) != QWEN_REVISION:
        raise ValueError("model revision must remain pinned")
    if str(config.get("jlens_upstream_commit", JLENS_UPSTREAM_COMMIT)) != JLENS_UPSTREAM_COMMIT:
        raise ValueError("jlens upstream commit must remain pinned")
    if str(config.get("purplellama_revision", PURPLE_LLAMA_REVISION)) != PURPLE_LLAMA_REVISION:
        raise ValueError("PurpleLlama revision must remain pinned")
    if int(config.get("batch_size", 0)) != 1:
        raise ValueError("causal J-lens batch_size must be 1")
    if int(config.get("eos_token_id", -1)) != FROZEN_EOS_TOKEN_ID:
        raise ValueError("eos_token_id must remain the frozen Qwen chat EOS id")
    expected_coefficients = {
        "ablation_coefficient": 0.2,
        "addition_coefficient": 0.5,
        "random_control_coefficient": 0.2,
    }
    for field, value in expected_coefficients.items():
        if float(config.get(field, -1)) != float(value):
            raise ValueError(f"config {field} must be {value}")


def _canonical_evaluation_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    missing = [
        key
        for key in ("source_v1_evaluation", "source_v2_evaluation")
        if key not in config
    ]
    if missing:
        raise ValueError(
            "lens evaluation paths must be pinned in the causal profile: "
            + ", ".join(missing)
        )
    return {
        "v1_evaluation": root / str(config["source_v1_evaluation"]),
        "v2_evaluation": root / str(config["source_v2_evaluation"]),
    }


def _required_pin(config: Mapping[str, Any], key: str, frozen: str) -> str:
    if int(config["dataset_size"]) == PRODUCTION_DATASET_SIZE:
        provided = config.get(key)
        if provided is not None and str(provided) != frozen:
            raise ValueError(f"{key} disagrees with frozen pin")
        return frozen
    value = config.get(key)
    if not value:
        raise ValueError(f"{key} missing for fixture contract")
    return str(value)


def _verify_frozen_prepare_pins(
    *,
    prepare_path: Path,
    prepare: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected_sha = _required_pin(config, "expected_prepare_sha256", FROZEN_PREPARE_SHA256)
    expected_digest = _required_pin(
        config, "expected_cohort_digest", FROZEN_COHORT_DIGEST
    )
    actual_sha = sha256_file(prepare_path)
    if actual_sha != expected_sha:
        raise ValueError("frozen prepare sha256 does not match")
    actual_digest = cohort_digest(prepare)
    if actual_digest != expected_digest:
        raise ValueError("frozen cohort digest does not match")


def _verify_one_lens_manifest(
    *,
    evaluation: Mapping[str, Any],
    evaluation_path: Path,
    lens_path: Path,
    dose: Mapping[str, Any],
    config: Mapping[str, Any],
    version: str,
    expected_sha: str,
    expected_run_id: str,
) -> None:
    if evaluation.get("status") != STATUS_PASS:
        raise ValueError(f"{version} lens evaluation status must be pass")
    if str(evaluation.get("model_revision")) != QWEN_REVISION:
        raise ValueError(f"{version} lens evaluation model revision mismatch")
    model_name = evaluation.get("model_name")
    if model_name is not None and str(model_name) != QWEN_MODEL:
        raise ValueError(f"{version} lens evaluation model name mismatch")
    if str(evaluation.get("jlens_upstream_commit")) != JLENS_UPSTREAM_COMMIT:
        raise ValueError(f"{version} lens evaluation jlens commit mismatch")
    dose_status = evaluation.get("source_dose_status")
    if dose_status is not None and dose_status != STATUS_PASS:
        raise ValueError(f"{version} lens evaluation source dose status must be pass")
    expected_dose_run = (
        FROZEN_SOURCE_DOSE_RUN_ID
        if int(config["dataset_size"]) == PRODUCTION_DATASET_SIZE
        else str(dose.get("run_id") or "")
    )
    eval_dose_run = evaluation.get("source_dose_run_id")
    if eval_dose_run is not None and str(eval_dose_run) != expected_dose_run:
        raise ValueError(f"{version} lens evaluation source dose run_id mismatch")
    if str(evaluation.get("run_id")) != expected_run_id:
        raise ValueError(f"frozen {version} run_id does not match")
    if sha256_file(evaluation_path) != expected_sha:
        raise ValueError(f"frozen evaluation sha256 does not match for {version}")
    lens_hash = sha256_file(lens_path)
    if str(evaluation.get("lens_sha256")) != lens_hash:
        raise ValueError(f"{version} lens_sha256 does not match current lens artifact")


def _verify_lens_manifest_provenance(
    *,
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    dose: Mapping[str, Any],
) -> None:
    evaluations = _canonical_evaluation_paths(root, config)
    missing = [name for name, path in evaluations.items() if not path.is_file()]
    if missing:
        raise ValueError("canonical lens evaluation missing: " + ", ".join(missing))
    expected_v1_sha = _required_pin(
        config, "expected_v1_evaluation_sha256", FROZEN_V1_EVALUATION_SHA256
    )
    expected_v2_sha = _required_pin(
        config, "expected_v2_evaluation_sha256", FROZEN_V2_EVALUATION_SHA256
    )
    expected_v1_run = _required_pin(config, "expected_v1_run_id", FROZEN_V1_RUN_ID)
    expected_v2_run = _required_pin(config, "expected_v2_run_id", FROZEN_V2_RUN_ID)
    _verify_one_lens_manifest(
        evaluation=_read_json(evaluations["v1_evaluation"]),
        evaluation_path=evaluations["v1_evaluation"],
        lens_path=paths["v1_lens"],
        dose=dose,
        config=config,
        version="v1",
        expected_sha=expected_v1_sha,
        expected_run_id=expected_v1_run,
    )
    _verify_one_lens_manifest(
        evaluation=_read_json(evaluations["v2_evaluation"]),
        evaluation_path=evaluations["v2_evaluation"],
        lens_path=paths["v2_lens"],
        dose=dose,
        config=config,
        version="v2",
        expected_sha=expected_v2_sha,
        expected_run_id=expected_v2_run,
    )


def _agreed_canonical_hash(*values: Any, name: str) -> str:
    present = [str(value) for value in values if value]
    if not present:
        raise ValueError(f"canonical {name} hash missing")
    unique = set(present)
    if len(unique) != 1:
        raise ValueError(f"{name} hash disagreement across canonical manifests")
    return unique.pop()


def _require_current_hash(actual: str, expected: str, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} hash does not match canonical manifest")


def _verify_prepare_frozen_identity(
    prepare: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    results = prepare.get("results") or {}
    dataset_size = int(config["dataset_size"])
    for dataset in DATASETS:
        block = results.get(dataset) or {}
        ids = list(block.get("example_ids") or [])
        hashes = list(block.get("content_hashes") or [])
        indices = list(block.get("source_indices") or [])
        if not (len(ids) == len(hashes) == len(indices) == dataset_size):
            raise ValueError(f"{dataset} frozen selection size drifted")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{dataset} duplicate example ids")
    source_files = results.get("source_files") or {}
    expected_identity = {
        "mitre": (
            str(config.get("purplellama_repository", PURPLE_LLAMA_REPOSITORY)),
            str(config.get("purplellama_revision", PURPLE_LLAMA_REVISION)),
            str(config["mitre_relative_path"]),
        ),
        "mitre_frr": (
            str(config.get("purplellama_repository", PURPLE_LLAMA_REPOSITORY)),
            str(config.get("purplellama_revision", PURPLE_LLAMA_REVISION)),
            str(config["mitre_frr_relative_path"]),
        ),
    }
    for dataset, (repository, revision, relative_path) in expected_identity.items():
        identity = source_files.get(dataset) or {}
        if not identity:
            raise ValueError(f"{dataset} PurpleLlama source identity missing")
        if (
            str(identity.get("repository")) != repository
            or str(identity.get("revision")) != revision
            or str(identity.get("relative_path")) != relative_path
        ):
            raise ValueError(f"{dataset} PurpleLlama source identity mismatch")


def _verify_canonical_source_hashes(
    *,
    root: Path,
    config: Mapping[str, Any],
    current_hashes: Mapping[str, str],
    prepare: Mapping[str, Any],
    dose: Mapping[str, Any],
) -> None:
    evaluations = _canonical_evaluation_paths(root, config)
    missing = [name for name, path in evaluations.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "canonical lens evaluation missing: " + ", ".join(missing)
        )
    v1_eval = _read_json(evaluations["v1_evaluation"])
    v2_eval = _read_json(evaluations["v2_evaluation"])
    prepare_hashes = (prepare.get("provenance") or {}).get("source_artifact_hashes") or {}
    dose_hashes = dose.get("source_artifact_hashes") or {}
    expected = {
        "direction": _agreed_canonical_hash(
            prepare_hashes.get("direction"),
            dose_hashes.get("selected_direction"),
            name="direction",
        ),
        "dose_evaluation": _agreed_canonical_hash(
            prepare_hashes.get("dose_evaluation"),
            name="dose_evaluation",
        ),
        "v1_lens": _agreed_canonical_hash(
            prepare_hashes.get("lens"),
            v1_eval.get("lens_sha256"),
            name="v1_lens",
        ),
        "v2_lens": _agreed_canonical_hash(
            v2_eval.get("lens_sha256"),
            name="v2_lens",
        ),
    }
    for name, expected_hash in expected.items():
        _require_current_hash(current_hashes[name], expected_hash, name)


def source_context(
    root: Path,
    config: Mapping[str, Any],
    *,
    load_direction_payload: bool = True,
) -> dict[str, Any]:
    _require_config_contract(config)
    paths = {
        "prepare": root / str(config["source_prepare"]),
        "direction": root / str(config["source_direction"]),
        "dose_evaluation": root / str(config["source_dose_evaluation"]),
        "v1_lens": root / str(config["source_v1_lens"]),
        "v2_lens": root / str(config["source_v2_lens"]),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("causal J-lens source artifacts missing: " + ", ".join(missing))
    dose = _read_json(paths["dose_evaluation"])
    if dose.get("status") != STATUS_PASS:
        raise ValueError("source dose must remain pass")
    pair = dose.get("selected_pair") or {}
    if (
        float(pair.get("ablation", -1)) != float(config["ablation_coefficient"])
        or float(pair.get("addition", -1)) != float(config["addition_coefficient"])
    ):
        raise ValueError("source dose coefficients do not match frozen coefficients")
    prepare = _read_json(paths["prepare"])
    _verify_prepare_frozen_identity(prepare, config)
    _verify_frozen_prepare_pins(
        prepare_path=paths["prepare"],
        prepare=prepare,
        config=config,
    )
    _verify_lens_manifest_provenance(
        root=root,
        config=config,
        paths=paths,
        dose=dose,
    )
    current_hashes = {name: sha256_file(path) for name, path in paths.items()}
    _verify_canonical_source_hashes(
        root=root,
        config=config,
        current_hashes=current_hashes,
        prepare=prepare,
        dose=dose,
    )
    context: dict[str, Any] = {
        "source_dose_status": dose["status"],
        "source_dose_run_id": dose.get("run_id"),
        "selected_pair": {
            "ablation": float(pair["ablation"]),
            "addition": float(pair["addition"]),
        },
        "source_artifact_hashes": current_hashes,
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
        context["direction_layer"] = direction.layer
        context["direction_offset"] = direction.position_offset
    return context


def append_jsonl_atomic(
    path: Path,
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    incoming = [records] if isinstance(records, Mapping) else list(records)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = [*existing, *[dict(record) for record in incoming]]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in payload:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class PanelCheckpoint:
    def __init__(
        self,
        identity: Mapping[str, Any],
        completed: Sequence[tuple[str, str, str]] | None = None,
    ) -> None:
        self.identity = {field: identity[field] for field in CHECKPOINT_IDENTITY_FIELDS}
        self._completed: list[tuple[str, str, str]] = [
            (str(dataset), str(example_id), str(condition))
            for dataset, example_id, condition in (completed or ())
        ]

    @classmethod
    def create(cls, identity: Mapping[str, Any]) -> PanelCheckpoint:
        return cls(identity)

    def mark_complete(self, dataset: str, example_id: str, condition: str) -> None:
        key = (str(dataset), str(example_id), str(condition))
        if key not in self._completed:
            self._completed.append(key)

    def pending(
        self, matrix: Sequence[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        done = set(self._completed)
        return [
            (str(dataset), str(example_id), str(condition))
            for dataset, example_id, condition in matrix
            if (str(dataset), str(example_id), str(condition)) not in done
        ]

    def write(self, path: Path) -> None:
        atomic_write_json(
            path,
            {
                "identity": dict(self.identity),
                "completed": [list(item) for item in self._completed],
            },
        )

    @classmethod
    def load(
        cls,
        path: Path,
        expected_identity: Mapping[str, Any],
    ) -> PanelCheckpoint:
        payload = _read_json(path)
        stored = payload.get("identity") or {}
        expected = {
            field: expected_identity[field] for field in CHECKPOINT_IDENTITY_FIELDS
        }
        if stored != expected:
            raise ValueError("checkpoint identity does not match")
        completed = [
            (str(item[0]), str(item[1]), str(item[2]))
            for item in payload.get("completed") or []
        ]
        return cls(stored, completed)


@dataclass(frozen=True)
class GenerationResult:
    generated_token_ids: list[int]
    response_text: str
    observations: list[dict[str, Any]]
    terminal_reason: str


class DualLensReadout:
    def __init__(self, lenses: Mapping[str, Any], bound_model: Any) -> None:
        if tuple(lenses) != LENS_VERSIONS:
            raise ValueError("lenses must be ordered as v1, v2")
        self._lenses = dict(lenses)
        self._bound_model = bound_model

    def rank_target(
        self,
        residual: torch.Tensor,
        *,
        layer: int,
        target_token_id: int,
    ) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for version, lens in self._lenses.items():
            logits = incremental_logits_from_residual(
                lens, self._bound_model, residual, layer=layer
            )
            require_finite_logits(logits, f"{version} readout logits")
            ranks[version] = rank_one_based(logits, target_token_id)
        return ranks


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError, AttributeError):
        return torch.device("cpu")


def _encode_prompt(
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(input_ids)
    else:
        mask = mask.to(device)
    return input_ids, mask


def _tensor_direction(direction: Any) -> torch.Tensor:
    vector = getattr(direction, "vector", direction)
    if not isinstance(vector, torch.Tensor):
        raise TypeError("direction must be a tensor or expose .vector")
    return vector


def _eos_token_id(model: Any, tokenizer: Any) -> int | None:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is not None:
        return int(token_id)
    config = getattr(model, "config", None)
    config_id = getattr(config, "eos_token_id", None)
    if config_id is None:
        return None
    return int(config_id)


def _decode_response(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _capture_hook(captured: dict[int, torch.Tensor], layer: int):
    def capture_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer] = hidden[:, -1, :].detach().clone()

    return capture_hook


def stream_causal_generation(
    *,
    model: Any,
    tokenizer: Any,
    readout: DualLensReadout,
    layers: Sequence[int] = READOUT_LAYERS,
    readout_steps: int = READOUT_STEPS,
    max_new_tokens: int = 64,
    prompt: str,
    condition: str,
    coefficient: float,
    direction: Any,
    random_control: torch.Tensor,
    dataset: str = "mitre",
    example_id: str = "example",
    seed: int | None = None,
) -> GenerationResult:
    if seed is not None:
        seed_everything(int(seed))
    device = _model_device(model)
    blocks, attentions, mlps = transformer_module_lists(model)
    pre_hooks, forward_hooks = condition_hook_specs(
        condition=str(condition),
        coefficient=float(coefficient),
        blocks=blocks,
        attentions=attentions,
        mlps=mlps,
        direction=_tensor_direction(direction),
        random_control=random_control,
    )
    requested_layers = tuple(int(layer) for layer in layers)
    captured: dict[int, torch.Tensor] = {}
    capture_hooks = [
        (blocks[layer], _capture_hook(captured, layer))
        for layer in requested_layers
    ]
    forward_hooks = [*forward_hooks, *capture_hooks]
    rendered = format_chat_instruction(tokenizer, prompt)
    input_ids, attention_mask = _encode_prompt(tokenizer, rendered, device)
    eos_id = _eos_token_id(model, tokenizer)
    past = None
    next_ids = input_ids
    cache_position = torch.arange(input_ids.shape[1], device=device)
    generated_ids: list[int] = []
    observations: list[dict[str, Any]] = []
    terminal_reason = "max_tokens"
    max_steps = int(max_new_tokens)
    readout_limit = int(readout_steps)

    try:
        with temporary_hooks(pre_hooks=pre_hooks, forward_hooks=forward_hooks):
            with torch.inference_mode():
                for step in range(1, max_steps + 1):
                    outputs = model(
                        input_ids=next_ids,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    past = getattr(outputs, "past_key_values", None)
                    logits = outputs.logits[:, -1, :].float()
                    require_finite_logits(logits, "model logits")
                    token_id = int(torch.argmax(logits.reshape(-1)).item())
                    if step <= readout_limit:
                        for layer in requested_layers:
                            residual = captured[layer].reshape(-1)
                            ranks = readout.rank_target(
                                residual,
                                layer=layer,
                                target_token_id=token_id,
                            )
                            for version, rank in ranks.items():
                                observations.append(
                                    {
                                        "dataset": dataset,
                                        "example_id": example_id,
                                        "condition": condition,
                                        "generation_step": step,
                                        "layer": int(layer),
                                        "lens_version": version,
                                        "rank": int(rank),
                                        "target_token_id": int(token_id),
                                    }
                                )
                    del outputs
                    del logits
                    captured.clear()
                    generated_ids.append(int(token_id))
                    if eos_id is not None and token_id == int(eos_id):
                        terminal_reason = "eos"
                        break
                    if step >= max_steps:
                        terminal_reason = "max_tokens"
                        break
                    next_ids = torch.tensor(
                        [[token_id]],
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    ones = torch.ones(
                        (attention_mask.shape[0], 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask, ones], dim=1)
                    cache_position = torch.tensor(
                        [input_ids.shape[1] + step - 1],
                        device=device,
                    )
    finally:
        captured.clear()
        past = None

    if compact_contains_text_leak(observations):
        raise ValueError("observations contain raw text")
    return GenerationResult(
        generated_token_ids=generated_ids,
        response_text=_decode_response(tokenizer, generated_ids),
        observations=observations,
        terminal_reason=terminal_reason,
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
    details = dict(probe())
    if not details.get("cuda_available"):
        raise ValueError("causal J-lens requires CUDA on an A100")
    if "A100" not in str(details.get("device_name", "")).upper():
        raise ValueError("causal J-lens requires an NVIDIA A100")
    if int(details.get("total_memory_bytes", 0)) < 39_000_000_000:
        raise ValueError("causal J-lens requires at least approximately 39 GB GPU memory")
    return details


def _dataset_source_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    value = os.environ.get("QWEN_CAUSAL_JLENS_SOURCE_ROOT") or os.environ.get(
        "QWEN_CODE_EVAL_SOURCE_ROOT"
    )
    if value:
        return Path(value)
    return Path("/content/qwen_code_eval_sources")


def _purplellama_root(source_root: Path, config: Mapping[str, Any]) -> Path:
    relative = str(config["mitre_relative_path"])
    if (source_root / relative).is_file():
        return source_root
    nested = source_root / "PurpleLlama"
    if (nested / relative).is_file():
        return nested
    raise ValueError("PurpleLlama datasets missing")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    if records:
        append_jsonl_atomic(path, records)


def _compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if str(key).casefold() not in TEXT_LEAK_KEYS and key != "generated_token_ids"
    }


def checkpoint_identity(
    *,
    run_id: str,
    implementation_commit: str,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    prepare: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "config_hash": _config_hash(config),
        "source_artifact_hashes": dict(source["source_artifact_hashes"]),
        "model_revision": str(config["model_revision"]),
        "dataset_ids": {
            dataset: list(prepare[dataset]["example_ids"]) for dataset in DATASETS
        },
        "dataset_hashes": {
            dataset: list(prepare[dataset]["content_hashes"]) for dataset in DATASETS
        },
    }


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["example_id"]), str(row["condition"]))


def _terminal_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"eos": 0, "max_tokens": 0}
    for row in rows:
        reason = str(row.get("terminal_reason") or "")
        if reason not in counts:
            raise ValueError("invalid terminal: unknown reason")
        counts[reason] += 1
    return counts


def require_exact_pair_matrix(
    rows: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    keys = [_pair_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate pair keys")
    actual = set(keys)
    wanted = {
        (str(dataset), str(example_id), str(condition))
        for dataset, example_id, condition in expected
    }
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise ValueError(
            f"pair matrix mismatch missing={missing} extra={extra}"
        )
    return keys


def _expected_observation_keys(
    response: Mapping[str, Any],
) -> set[tuple[str, str, str, int, int, str]]:
    dataset, example_id, condition = _pair_key(response)
    actual_steps = min(READOUT_STEPS, int(response["generated_token_count"]))
    return {
        (dataset, example_id, condition, step, int(layer), version)
        for step in range(1, actual_steps + 1)
        for layer in READOUT_LAYERS
        for version in LENS_VERSIONS
    }


def _pair_is_complete(
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    if len(responses) != 1:
        return False
    response = responses[0]
    try:
        generated = int(response["generated_token_count"])
        readout_steps = int(response["readout_steps"])
        terminal = str(response.get("terminal_reason") or "")
    except (KeyError, TypeError, ValueError):
        return False
    actual_steps = min(READOUT_STEPS, generated)
    if readout_steps != actual_steps or not terminal:
        return False
    try:
        keys = [observation_key(row) for row in observations]
    except (KeyError, TypeError, ValueError):
        return False
    if len(keys) != len(set(keys)):
        return False
    return set(keys) == _expected_observation_keys(response)


def require_complete_pair_artifacts(
    responses: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    expected: Sequence[tuple[str, str, str]] | None = None,
) -> None:
    matrix = (
        list(expected)
        if expected is not None
        else [_pair_key(row) for row in responses]
    )
    require_exact_pair_matrix(responses, matrix)
    grouped_obs: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped_obs[_pair_key(row)].append(row)
    grouped_resp: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in responses:
        grouped_resp[_pair_key(row)].append(row)
    if set(grouped_obs) != set(grouped_resp):
        raise ValueError("incomplete observation keys")
    for key, resp_rows in grouped_resp.items():
        if not _pair_is_complete(resp_rows, grouped_obs.get(key, [])):
            raise ValueError(f"incomplete observation keys for pair {key}")


def reconcile_generation_artifacts(
    artifacts_dir: Path,
    *,
    matrix: Sequence[tuple[str, str, str]],
    checkpoint: PanelCheckpoint,
    checkpoint_path: Path,
) -> PanelCheckpoint:
    responses_path = artifacts_dir / RESPONSES_FILENAME
    observations_path = artifacts_dir / OBSERVATIONS_FILENAME
    responses = _read_jsonl(responses_path)
    observations = _read_jsonl(observations_path)
    grouped_resp: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_obs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        grouped_resp[_pair_key(row)].append(row)
    for row in observations:
        grouped_obs[_pair_key(row)].append(row)
    expected = [
        (str(dataset), str(example_id), str(condition))
        for dataset, example_id, condition in matrix
    ]
    keep_responses: list[dict[str, Any]] = []
    keep_observations: list[dict[str, Any]] = []
    completed: list[tuple[str, str, str]] = []
    for key in expected:
        resp_rows = grouped_resp.get(key, [])
        obs_rows = grouped_obs.get(key, [])
        if _pair_is_complete(resp_rows, obs_rows):
            keep_responses.append(resp_rows[0])
            keep_observations.extend(obs_rows)
            completed.append(key)
    _write_jsonl(responses_path, keep_responses)
    _write_jsonl(observations_path, keep_observations)
    repaired = PanelCheckpoint(checkpoint.identity, completed)
    repaired.write(checkpoint_path)
    return repaired


def expected_matrix(
    prepare: Mapping[str, Any],
    *,
    conditions: Sequence[str] = CONDITIONS,
    example_ids: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[str, str, str]]:
    matrix: list[tuple[str, str, str]] = []
    for dataset in DATASETS:
        ids = (
            list(example_ids[dataset])
            if example_ids is not None
            else list(prepare[dataset]["example_ids"])
        )
        for example_id in ids:
            for condition in conditions:
                matrix.append((dataset, str(example_id), str(condition)))
    return matrix


def _condition_coefficient(condition: str, config: Mapping[str, Any]) -> float:
    if condition == "baseline":
        return 0.0
    if condition == "ablation":
        return float(config["ablation_coefficient"])
    if condition == "addition":
        return float(config["addition_coefficient"])
    if condition == "random_control":
        return float(config["random_control_coefficient"])
    raise ValueError(f"unknown condition: {condition}")


def _stable_example_id(dataset: str, source_index: int, digest: str) -> str:
    return f"{dataset}:{int(source_index)}:{digest[:16]}"


def _rehydrate_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    expected: Mapping[str, Any],
    text_field: str,
    category_field: str,
    dataset_size: int,
) -> list[dict[str, Any]]:
    ids = list(expected["example_ids"])
    hashes = list(expected["content_hashes"])
    indices = list(expected["source_indices"])
    if not (len(ids) == len(hashes) == len(indices) == int(dataset_size)):
        raise ValueError(f"{dataset} frozen selection size drifted")
    selected: list[dict[str, Any]] = []
    for example_id, content_hash, source_index in zip(ids, hashes, indices, strict=True):
        try:
            row = rows[int(source_index)]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"{dataset} source index missing") from exc
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{dataset} prompt missing at source index {source_index}")
        digest = prompt_content_hash(text)
        reconstructed = _stable_example_id(dataset, int(source_index), digest)
        if reconstructed != str(example_id):
            raise ValueError(f"{dataset} id mismatch at source index {source_index}")
        if digest != str(content_hash):
            raise ValueError(f"{dataset} hash mismatch at source index {source_index}")
        category_value = row.get(category_field)
        selected.append(
            {
                "example_id": str(example_id),
                "content_hash": str(content_hash),
                "prompt": text,
                "category": None if category_value in {None, ""} else str(category_value),
                "source_index": int(source_index),
            }
        )
    return selected


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
        "jlens_upstream_commit": config.get(
            "jlens_upstream_commit", JLENS_UPSTREAM_COMMIT
        ),
        "source_dose_run_id": source.get("source_dose_run_id"),
        "source_artifact_hashes": dict(source["source_artifact_hashes"]),
        "direction_layer": source.get("direction_layer", EXPECTED_SOURCE_LAYER),
        "direction_offset": source.get("direction_offset", EXPECTED_SOURCE_OFFSET),
        "eos_token_id": int(config["eos_token_id"]),
        "cohort_digest": cohort_digest(_read_json(source["paths"]["prepare"])),
    }


def _write_stage(
    path: Path,
    *,
    stage: str,
    config: Mapping[str, Any],
    results: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if compact_contains_text_leak(results):
        raise ValueError(f"{stage} compact results contain raw text")
    write_artifact(
        path,
        ArtifactEnvelope(
            experiment=f"causal_jlens_{stage}",
            config=_jsonable_config(config),
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
            raise ValueError(f"stale causal J-lens checkpoint {path.name}: {field}")
    return payload


def check_dual_lens_parity(
    *,
    readout: DualLensReadout,
    bound_model: Any,
    prompt: str,
    layer: int = PRIMARY_LAYER,
    **_kwargs: Any,
) -> None:
    for version, lens in readout._lenses.items():
        canonical = assert_incremental_readout_parity(
            lens,
            bound_model,
            prompt,
            layer=int(layer),
            position=-1,
            top_k=5,
            return_canonical_logits=True,
        )
        if canonical is not None:
            require_finite_logits(canonical, f"{version} parity logits")


def _default_load_readout(
    model: Any,
    tokenizer: Any,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
) -> DualLensReadout:
    del config
    from jlens.lens import JacobianLens

    assert_jlens_contract()
    v1 = JacobianLens.load(str(source["paths"]["v1_lens"]))
    v2 = JacobianLens.load(str(source["paths"]["v2_lens"]))
    bound = wrap_hf_model_for_jlens(model, tokenizer)
    return DualLensReadout(lenses={"v1": v1, "v2": v2}, bound_model=bound)


def _verify_observations(observations: Sequence[Mapping[str, Any]]) -> None:
    if compact_contains_text_leak(observations):
        raise ValueError("observations contain raw text")
    for row in observations:
        missing = [key for key in OBSERVATION_KEYS if key not in row]
        if missing:
            raise ValueError("observation missing keys: " + ", ".join(missing))


def _raw_response_record(
    result: GenerationResult,
    *,
    dataset: str,
    example: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "example_id": example["example_id"],
        "content_hash": example["content_hash"],
        "condition": condition,
        "prompt": example["prompt"],
        "response": result.response_text,
        "refusal": contains_refusal(result.response_text),
        "valid": is_valid_generation(result.response_text),
        "generated_token_ids": list(result.generated_token_ids),
        "generated_token_count": len(result.generated_token_ids),
        "readout_steps": min(READOUT_STEPS, len(result.generated_token_ids)),
        "terminal_reason": result.terminal_reason,
    }


def _load_selected_examples(artifacts_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        records = _read_jsonl(artifacts_dir / "datasets" / f"{dataset}.jsonl")
        loaded[dataset] = {str(record["example_id"]): record for record in records}
    return loaded


def _load_or_create_checkpoint(
    path: Path,
    identity: Mapping[str, Any],
) -> PanelCheckpoint:
    if path.is_file():
        return PanelCheckpoint.load(path, expected_identity=identity)
    return PanelCheckpoint.create(identity)


def _run_generation_matrix(
    *,
    matrix: Sequence[tuple[str, str, str]],
    checkpoint: PanelCheckpoint,
    checkpoint_path: Path,
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    examples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    load_model: LoadModel | None,
    load_readout: LoadReadout | None,
    parity_check: ParityCheck | None,
    invoke_parity: bool,
) -> dict[str, Any]:
    responses_path = artifacts_dir / RESPONSES_FILENAME
    observations_path = artifacts_dir / OBSERVATIONS_FILENAME
    checkpoint = reconcile_generation_artifacts(
        artifacts_dir,
        matrix=matrix,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    pending = checkpoint.pending(matrix)
    if not pending:
        responses = _read_jsonl(responses_path)
        observations = _read_jsonl(observations_path)
        require_complete_pair_artifacts(responses, observations, matrix)
        return {
            "response_count": len(responses),
            "observation_count": len(observations),
            "conditions": list(CONDITIONS),
            "terminal_reasons": _terminal_reason_counts(responses),
        }
    loader = load_model or _load_model_for_config
    readout_loader = load_readout or _default_load_readout
    if load_readout is None:
        verify_frozen_lens_hash(
            source["paths"]["v1_lens"],
            source["source_artifact_hashes"]["v1_lens"],
        )
        verify_frozen_lens_hash(
            source["paths"]["v2_lens"],
            source["source_artifact_hashes"]["v2_lens"],
        )
    model = tokenizer = None
    try:
        loaded = loader(dict(config))
        model, tokenizer = loaded[0], loaded[1]
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        if resolved is not None and resolved != config["model_revision"]:
            raise ValueError("resolved model revision differs from pinned revision")
        readout = readout_loader(model, tokenizer, source, config)
        direction = source.get("direction") or load_direction(source["paths"]["direction"])
        control = random_direction_control(
            _tensor_direction(direction),
            seed=int(config["random_control_seed"]),
        )
        if invoke_parity:
            checker = parity_check or check_dual_lens_parity
            first_dataset, first_example_id, _condition = pending[0]
            checker(
                readout=readout,
                bound_model=getattr(readout, "_bound_model", None),
                prompt=examples[first_dataset][first_example_id]["prompt"],
                example_id=first_example_id,
                layer=int(config.get("primary_layer", PRIMARY_LAYER)),
            )
        completed = len(matrix) - len(pending)
        for dataset, example_id, condition in pending:
            example = examples[dataset][example_id]
            result = stream_causal_generation(
                model=model,
                tokenizer=tokenizer,
                readout=readout,
                layers=tuple(int(layer) for layer in config.get("readout_layers", READOUT_LAYERS)),
                readout_steps=int(config.get("readout_steps", READOUT_STEPS)),
                max_new_tokens=int(config.get("max_new_tokens", 64)),
                prompt=str(example["prompt"]),
                condition=str(condition),
                coefficient=_condition_coefficient(str(condition), config),
                direction=direction,
                random_control=control,
                dataset=str(dataset),
                example_id=str(example_id),
                seed=int(config["seed"]),
            )
            if hasattr(model, "active_hook_count") and int(model.active_hook_count) != 0:
                raise ValueError("hooks were not cleaned")
            _verify_observations(result.observations)
            append_jsonl_atomic(
                responses_path,
                _raw_response_record(
                    result,
                    dataset=str(dataset),
                    example=example,
                    condition=str(condition),
                ),
            )
            if result.observations:
                append_jsonl_atomic(observations_path, result.observations)
            checkpoint.mark_complete(dataset, example_id, condition)
            checkpoint.write(checkpoint_path)
            completed += 1
            store.update_progress(completed_units=completed, total_units=len(matrix))
        responses = _read_jsonl(responses_path)
        observations = _read_jsonl(observations_path)
        require_complete_pair_artifacts(responses, observations, matrix)
        return {
            "response_count": len(responses),
            "observation_count": len(observations),
            "conditions": list(CONDITIONS),
            "terminal_reasons": _terminal_reason_counts(responses),
        }
    finally:
        release_compute_resources(model, tokenizer)


def run_prepare_stage(
    *,
    root: Path,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any] | None = None,
    source_root: Path | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    del source
    purple_root = _purplellama_root(_dataset_source_root(source_root), config)
    mitre_path = purple_root / str(config["mitre_relative_path"])
    frr_path = purple_root / str(config["mitre_frr_relative_path"])
    prepare_source = _read_json(root / str(config["source_prepare"]))
    frozen = prepare_source.get("results") or {}
    specifications = (
        ("mitre", mitre_path, "mutated_prompt", "mitre_category"),
        ("mitre_frr", frr_path, "mutated_prompt", "attack_type"),
    )
    selections: dict[str, Any] = {}
    completed = 0
    for dataset, path, text_field, category_field in specifications:
        rows = json.loads(path.read_text(encoding="utf-8"))
        selected = _rehydrate_dataset(
            rows,
            dataset=dataset,
            expected=frozen[dataset],
            text_field=text_field,
            category_field=category_field,
            dataset_size=int(config["dataset_size"]),
        )
        _write_jsonl(artifacts_dir / "datasets" / f"{dataset}.jsonl", selected)
        selections[dataset] = {
            "dataset": dataset,
            "example_ids": [item["example_id"] for item in selected],
            "content_hashes": [item["content_hash"] for item in selected],
            "source_indices": [item["source_index"] for item in selected],
            "selection_size": len(selected),
        }
        completed += len(selected)
        store.update_progress(completed_units=completed)
    selections["source_files"] = {
        "mitre": build_source_identity(
            mitre_path,
            repository=str(config.get("purplellama_repository", PURPLE_LLAMA_REPOSITORY)),
            revision=str(config.get("purplellama_revision", PURPLE_LLAMA_REVISION)),
            relative_path=str(config["mitre_relative_path"]),
        ),
        "mitre_frr": build_source_identity(
            frr_path,
            repository=str(config.get("purplellama_repository", PURPLE_LLAMA_REPOSITORY)),
            revision=str(config.get("purplellama_revision", PURPLE_LLAMA_REVISION)),
            relative_path=str(config["mitre_frr_relative_path"]),
        ),
    }
    frozen_files = frozen.get("source_files") or {}
    for dataset in DATASETS:
        expected = frozen_files.get(dataset) or {}
        actual = selections["source_files"][dataset]
        for field in ("repository", "revision", "relative_path"):
            if expected.get(field) is not None and expected.get(field) != actual.get(field):
                raise ValueError(f"{dataset} PurpleLlama {field} mismatch")
        if expected.get("sha256") and expected["sha256"] != actual["sha256"]:
            raise ValueError(f"{dataset} PurpleLlama source hash mismatch")
    if compact_contains_text_leak(selections):
        raise ValueError("prepare compact results contain raw text")
    return selections


def _pilot_ids(prepare: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        dataset: list(prepare[dataset]["example_ids"][:PILOT_PROMPTS_PER_DATASET])
        for dataset in DATASETS
    }


def run_pilot_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any],
    prepare: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    load_model: LoadModel | None = None,
    load_readout: LoadReadout | None = None,
    parity_check: ParityCheck | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    identity = checkpoint_identity(
        run_id=run_id,
        implementation_commit=implementation_commit,
        config=config,
        source=source,
        prepare=prepare,
    )
    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    checkpoint = _load_or_create_checkpoint(checkpoint_path, identity)
    pilot_ids = _pilot_ids(prepare)
    matrix = expected_matrix(prepare, example_ids=pilot_ids)
    generated = _run_generation_matrix(
        matrix=matrix,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        artifacts_dir=artifacts_dir,
        store=store,
        source=source,
        config=config,
        examples=_load_selected_examples(artifacts_dir),
        load_model=load_model,
        load_readout=load_readout,
        parity_check=parity_check,
        invoke_parity=True,
    )
    responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
    observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
    require_exact_pair_matrix(responses, matrix)
    require_complete_pair_artifacts(responses, observations, matrix)
    payload = {
        "response_count": generated["response_count"],
        "observation_count": generated["observation_count"],
        "pilot_status": "pass",
        "conditions": list(CONDITIONS),
        "pilot_ids": pilot_ids,
        "terminal_reasons": generated.get("terminal_reasons", []),
    }
    if compact_contains_text_leak(payload):
        raise ValueError("pilot compact results contain raw text")
    return payload


def run_panel_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    store: RunProgressStore,
    source: Mapping[str, Any],
    prepare: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    load_model: LoadModel | None = None,
    load_readout: LoadReadout | None = None,
    parity_check: ParityCheck | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    identity = checkpoint_identity(
        run_id=run_id,
        implementation_commit=implementation_commit,
        config=config,
        source=source,
        prepare=prepare,
    )
    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    checkpoint = _load_or_create_checkpoint(checkpoint_path, identity)
    matrix = expected_matrix(prepare)
    generated = _run_generation_matrix(
        matrix=matrix,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        artifacts_dir=artifacts_dir,
        store=store,
        source=source,
        config=config,
        examples=_load_selected_examples(artifacts_dir),
        load_model=load_model,
        load_readout=load_readout,
        parity_check=parity_check,
        invoke_parity=False,
    )
    responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
    observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
    require_exact_pair_matrix(responses, matrix)
    require_complete_pair_artifacts(responses, observations, matrix)
    payload = {
        "response_count": generated["response_count"],
        "observation_count": generated["observation_count"],
        "conditions": list(CONDITIONS),
        "matrix_size": len(matrix),
        "terminal_reasons": generated.get("terminal_reasons", []),
    }
    if compact_contains_text_leak(payload):
        raise ValueError("panel compact results contain raw text")
    return payload


def _refusal_rates(responses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for condition in CONDITIONS:
            items = [
                record
                for record in responses
                if str(record["dataset"]) == dataset
                and str(record["condition"]) == condition
            ]
            refusals = [bool(record.get("refusal")) for record in items]
            output[dataset][condition] = {
                "prompt_count": len(items),
                "refusal_count": sum(refusals),
                "refusal_rate": (
                    sum(refusals) / len(refusals) if refusals else None
                ),
            }
    return output


def run_analysis_stage(
    *,
    config: Mapping[str, Any],
    artifacts_dir: Path,
    prepare: Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    raw_responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
    observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
    expected = expected_matrix(prepare) if prepare is not None else None
    require_complete_pair_artifacts(raw_responses, observations, expected)
    responses = [_compact_record(record) for record in raw_responses]
    _verify_observations(observations)
    fidelity = aggregate_fidelity(observations)
    behavior = summarize_behavior_changes(responses, observations)
    bootstrap: dict[str, Any] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            if condition == "baseline":
                continue
            for lens_version in LENS_VERSIONS:
                key = f"{dataset}/{condition}/{lens_version}"
                bootstrap[key] = cluster_bootstrap_difference(
                    observations,
                    condition=condition,
                    k=int(config.get("primary_k", 5)),
                    layer=int(config.get("primary_layer", PRIMARY_LAYER)),
                    lens_version=lens_version,
                    dataset=dataset,
                    resamples=int(config.get("bootstrap_resamples", 10_000)),
                    seed=int(config.get("bootstrap_seed", 2028)),
                )
    payload = {
        "fidelity": fidelity,
        "bootstrap": bootstrap,
        "behavior_changes": behavior,
        "refusal_rates": _refusal_rates(responses),
        "observation_count": len(observations),
        "response_count": len(responses),
    }
    if compact_contains_text_leak(payload):
        raise ValueError("analysis compact results contain raw text")
    return payload


def _primary_from_bootstrap(bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    primary: dict[str, Any] = {}
    for key, payload in bootstrap.items():
        dataset, condition, lens_version = str(key).split("/")
        primary.setdefault(lens_version, {}).setdefault(dataset, {})[condition] = {
            "estimate": payload["estimate"],
            "ci_lower": payload["ci_lower"],
            "ci_upper": payload["ci_upper"],
            "prompt_count": payload["prompt_count"],
            "equivalent": (
                float(payload["ci_lower"]) >= -0.05
                and float(payload["ci_upper"]) <= 0.05
            ),
        }
    return primary


def run_finalize_stage(
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    prepare: Mapping[str, Any] | None = None,
    pilot: Mapping[str, Any] | None = None,
    panel: Mapping[str, Any] | None = None,
    analysis: Mapping[str, Any] | None = None,
    artifacts_dir: Path | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    if artifacts_dir is not None and prepare is not None:
        responses = _read_jsonl(artifacts_dir / RESPONSES_FILENAME)
        observations = _read_jsonl(artifacts_dir / OBSERVATIONS_FILENAME)
        matrix = expected_matrix(prepare)
        require_complete_pair_artifacts(responses, observations, matrix)
        if (panel or {}).get("response_count") not in {None, len(matrix)}:
            raise ValueError("panel response_count does not match pair matrix")
    del config
    analysis = analysis or {}
    bootstrap = analysis.get("bootstrap") or {}
    evaluation = {
        "status": STATUS_PASS,
        "pilot_status": (pilot or {}).get("pilot_status", "pass"),
        "response_count": (panel or {}).get("response_count"),
        "observation_count": (panel or {}).get("observation_count"),
        "prepare_ids": {
            dataset: (prepare or {}).get(dataset, {}).get("example_ids")
            for dataset in DATASETS
        },
        "results": {
            "primary": _primary_from_bootstrap(bootstrap),
            "refusal_rates": analysis.get("refusal_rates"),
            "behavior_change_analysis": analysis.get("behavior_changes"),
            "fidelity": analysis.get("fidelity"),
        },
    }
    if provenance:
        evaluation.update(
            {
                "run_id": provenance["run_id"],
                "implementation_commit": provenance["implementation_commit"],
                "model_revision": provenance.get(
                    "resolved_model_revision", provenance.get("requested_model_revision")
                ),
                "source_artifact_hashes": provenance["source_artifact_hashes"],
            }
        )
    if compact_contains_text_leak(evaluation):
        raise ValueError("finalize compact results contain raw text")
    return evaluation


def _write_terminal_files(
    *,
    results_dir: Path,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        **dict(evaluation),
        "config": _jsonable_config(config),
        "provenance": dict(provenance),
    }
    atomic_write_json(results_dir / EVALUATION_FILENAME, payload)
    completed_stages = [
        stage
        for stage in CAUSAL_STAGES
        if stage != "finalize"
        and _load_stage(
            results_dir / STAGE_FILES[stage],
            run_id=str(provenance["run_id"]),
            implementation_commit=str(provenance["implementation_commit"]),
            config=config,
            source_hashes=provenance["source_artifact_hashes"],
        )
        is not None
    ]
    _write_stage(
        results_dir / MANIFEST_FILENAME,
        stage="manifest",
        config=config,
        results={
            "status": payload["status"],
            "completed_stages": [*completed_stages, "finalize"],
            "result_files": [
                *[STAGE_FILES[stage] for stage in completed_stages],
                EVALUATION_FILENAME,
                MANIFEST_FILENAME,
            ],
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
        total = 1
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


def _stage_total_units(stage: str, *, dataset_size: int) -> int:
    if stage == "prepare":
        return len(DATASETS) * dataset_size
    if stage == "pilot":
        return len(DATASETS) * PILOT_PROMPTS_PER_DATASET * len(CONDITIONS)
    if stage == "panel":
        return len(DATASETS) * dataset_size * len(CONDITIONS)
    if stage in {"analysis", "finalize"}:
        return 1
    raise ValueError(f"unknown causal J-lens stage: {stage}")


def _stage_completed_units(
    stage: str,
    *,
    artifacts_dir: Path,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    prepare: Mapping[str, Any] | None,
    run_id: str,
    implementation_commit: str,
) -> int:
    if stage in {"prepare", "analysis", "finalize"} or prepare is None:
        return 0
    identity = checkpoint_identity(
        run_id=run_id,
        implementation_commit=implementation_commit,
        config=config,
        source=source,
        prepare=prepare,
    )
    matrix = expected_matrix(
        prepare,
        example_ids=_pilot_ids(prepare) if stage == "pilot" else None,
    )
    checkpoint_path = artifacts_dir / CHECKPOINT_FILENAME
    checkpoint = _load_or_create_checkpoint(checkpoint_path, identity)
    checkpoint = reconcile_generation_artifacts(
        artifacts_dir,
        matrix=matrix,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    return len(matrix) - len(checkpoint.pending(matrix))


def _default_stage_runner(stage: str, **kwargs: Any) -> dict[str, Any]:
    functions = {
        "prepare": run_prepare_stage,
        "pilot": run_pilot_stage,
        "panel": run_panel_stage,
        "analysis": run_analysis_stage,
        "finalize": run_finalize_stage,
    }
    return functions[stage](**kwargs)


def run_qwen_causal_jlens(
    root: Path,
    *,
    profile: str = CAUSAL_JLENS_PROFILE,
    run_id: str | None = None,
    implementation_commit: str | None = None,
    stage_runner: StageRunner | None = None,
    load_model: LoadModel | None = None,
    load_readout: LoadReadout | None = None,
    parity_check: ParityCheck | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] = _default_gpu_probe,
    source_root: Path | None = None,
) -> dict[str, Any]:
    require_a100_preflight(gpu_probe)
    root = root.resolve()
    config = load_config(root, profile=profile)
    _require_config_contract(config)
    results_dir, artifacts_dir = project_paths(root, config)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    source = source_context(
        root,
        config,
        load_direction_payload=stage_runner is None,
    )
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
            raise ValueError("stale causal J-lens terminal run_id")
        if terminal.get("implementation_commit") != commit:
            raise ValueError("stale causal J-lens terminal implementation_commit")
        if terminal.get("source_artifact_hashes") != source["source_artifact_hashes"]:
            raise ValueError("stale causal J-lens terminal source hashes")
        if terminal.get("status") == STATUS_PASS:
            _reconcile_terminal_status(store, terminal)
            return terminal
        terminal_path.unlink()
        (results_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    runner = stage_runner or _default_stage_runner
    accumulated: dict[str, Any] = {}
    last: dict[str, Any] | None = None
    for stage in CAUSAL_STAGES:
        if stage == "finalize":
            checkpoint_path = None
        else:
            checkpoint_path = results_dir / STAGE_FILES[stage]
        checkpoint = (
            None
            if checkpoint_path is None
            else _load_stage(
                checkpoint_path,
                run_id=resolved_run,
                implementation_commit=commit,
                config=config,
                source_hashes=source["source_artifact_hashes"],
            )
        )
        if checkpoint is not None:
            last = checkpoint["results"]
            accumulated[stage] = last
            continue
        try:
            store.initialize(
                stage=stage,
                total_units=_stage_total_units(
                    stage, dataset_size=int(config["dataset_size"])
                ),
                completed_units=_stage_completed_units(
                    stage,
                    artifacts_dir=artifacts_dir,
                    config=config,
                    source=source,
                    prepare=accumulated.get("prepare"),
                    run_id=resolved_run,
                    implementation_commit=commit,
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
                prepare=accumulated.get("prepare"),
                pilot=accumulated.get("pilot"),
                panel=accumulated.get("panel"),
                analysis=accumulated.get("analysis"),
                provenance=provenance,
                run_id=resolved_run,
                implementation_commit=commit,
                load_model=load_model,
                load_readout=load_readout,
                parity_check=parity_check,
                source_root=source_root,
            )
            if stage != "finalize":
                _write_stage(
                    results_dir / STAGE_FILES[stage],
                    stage=stage,
                    config=config,
                    results=results,
                    provenance=provenance,
                )
            accumulated[stage] = results
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
                "pilot_status": (accumulated.get("pilot") or {}).get("pilot_status"),
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
        raise ValueError("causal J-lens run produced no result")
    return _read_json(terminal_path)
