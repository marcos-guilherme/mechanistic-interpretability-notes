from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

import torch
import torch.nn.functional as F

from .artifacts import ArtifactEnvelope, write_artifact
from .data import (
    PromptExample,
    content_hashes_for_examples,
    hash_examples,
    load_local_prompt_corpus,
)
from .directions import (
    CandidateDirection,
    candidate_grid,
    load_direction,
    save_direction,
)
from .interventions import (
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
)
from .metrics import (
    CandidateScore,
    contains_refusal,
    is_valid_generation,
    kl_divergence,
    mean_cross_entropy_delta,
    refusal_log_odds,
    refusal_rate,
)
from .model import (
    FORMAT_STYLE_GEMMA,
    GEMMA_REFUSAL_TOKEN_IDS,
    capture_residual_grid,
    end_of_instruction_offsets,
    format_chat_instruction,
    load_causal_lm,
    resolve_refusal_token_ids,
    tokenize_instructions,
)
from .runtime import runtime_info, seed_everything


def dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def sample_examples(
    examples: Sequence[PromptExample],
    *,
    size: int,
    seed: int,
) -> list[PromptExample]:
    if len(examples) < size:
        raise ValueError(f"need {size} examples, received {len(examples)}")
    return random.Random(seed).sample(list(examples), size)


def save_candidate_artifact(
    path: Path,
    candidates: Sequence[CandidateDirection],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {
                "layer": candidate.layer,
                "position_offset": candidate.position_offset,
                "vector": candidate.vector.cpu(),
            }
            for candidate in candidates
        ],
        path,
    )


def load_candidate_artifact(path: Path) -> list[CandidateDirection]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return [
        CandidateDirection(
            int(item["layer"]),
            int(item["position_offset"]),
            item["vector"],
        )
        for item in payload
    ]


@contextmanager
def temporary_hooks(
    *,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
):
    handles = []
    try:
        for module, hook in pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def filter_examples_by_score(
    examples: Sequence[PromptExample],
    scores: torch.Tensor,
    *,
    keep_positive: bool,
) -> list[PromptExample]:
    if len(examples) != len(scores):
        raise ValueError("examples and scores must align")
    return [
        example
        for example, score in zip(examples, scores.tolist())
        if (score > 0) == keep_positive
    ]


def maybe_filter_examples(
    examples: Sequence[PromptExample],
    scores: torch.Tensor,
    *,
    keep_positive: bool,
    config: dict,
) -> list[PromptExample]:
    if config.get("filter_by_behavior", True):
        return filter_examples_by_score(
            examples,
            scores,
            keep_positive=keep_positive,
        )
    return list(examples)


def maybe_filter_scores(
    scores: torch.Tensor,
    *,
    keep_positive: bool,
    config: dict,
) -> torch.Tensor:
    if config.get("filter_by_behavior", True):
        mask = scores > 0 if keep_positive else scores < 0
        return scores[mask]
    return scores


SPLIT_NAME_TO_SUFFIX = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def split_examples(
    splits: dict[str, list[PromptExample]],
    split_name: str,
) -> tuple[list[PromptExample], list[PromptExample]]:
    try:
        suffix = SPLIT_NAME_TO_SUFFIX[split_name]
    except KeyError as error:
        raise ValueError(f"unknown split name: {split_name}") from error
    return splits[f"harmful_{suffix}"], splits[f"harmless_{suffix}"]


def release_compute_resources(model=None, tokenizer=None) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _record_example_ids(config: dict) -> bool:
    if "record_example_ids" in config:
        return bool(config["record_example_ids"])
    return "test_per_class" in config


def _ids_from_examples(examples: Sequence[PromptExample]) -> list[str]:
    return [example.id for example in examples]


def _ids_from_splits(
    splits: dict[str, Sequence[PromptExample]],
) -> dict[str, list[str]]:
    return {name: _ids_from_examples(items) for name, items in splits.items()}


def summarize_generated_condition(
    outputs: Sequence[str],
    *,
    examples: Sequence[PromptExample] | None = None,
    count_valid: bool = False,
) -> dict:
    if not count_valid:
        return {
            "refusal_rate": refusal_rate(outputs),
            "refusal_flags": [contains_refusal(output) for output in outputs],
        }
    valid_indices = [
        index
        for index, output in enumerate(outputs)
        if is_valid_generation(output)
    ]
    valid_outputs = [outputs[index] for index in valid_indices]
    if valid_outputs:
        flags = [contains_refusal(output) for output in valid_outputs]
        rate = sum(flags) / len(flags)
    else:
        flags = []
        rate = 0.0
    if examples is not None:
        ids = [examples[index].id for index in range(len(outputs))]
        valid_ids = [ids[index] for index in valid_indices]
        invalid_ids = [
            ids[index]
            for index in range(len(outputs))
            if index not in set(valid_indices)
        ]
    else:
        valid_ids = [str(index) for index in valid_indices]
        invalid_ids = [
            str(index)
            for index in range(len(outputs))
            if index not in set(valid_indices)
        ]
    return {
        "refusal_rate": rate,
        "refusal_flags": flags,
        "n_valid": len(valid_indices),
        "n_invalid": len(outputs) - len(valid_indices),
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
    }


def select_candidate(
    candidates: Sequence[CandidateDirection],
    scores: Sequence[CandidateScore],
) -> CandidateDirection:
    if len(candidates) != len(scores) or not candidates:
        raise ValueError("candidates and scores must be non-empty and aligned")
    eligible = [
        index for index, score in enumerate(scores) if score.eligible()
    ]
    if not eligible:
        raise ValueError("all candidate directions were filtered out")
    index = min(
        eligible,
        key=lambda item: scores[item].ablation_refusal_score,
    )
    return candidates[index]


def random_direction_control(
    direction: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(
        direction.shape,
        generator=generator,
        dtype=torch.float32,
    )
    source = direction.float()
    random = random - torch.dot(random, source) / source.square().sum() * source
    return random / random.norm() * source.norm()


def transformer_module_lists(model) -> tuple[list, list, list]:
    blocks = list(model.model.layers)
    attentions = [block.self_attn for block in blocks]
    mlps = [block.mlp for block in blocks]
    return blocks, attentions, mlps


def gemma_module_lists(model) -> tuple[list, list, list]:
    return transformer_module_lists(model)


GEMMA_PROFILE = "gemma_2b.json"
JLENS_PROFILE = "qwen25_coder_7b_jlens.json"
JLENS_V2_PROFILE = "qwen25_coder_7b_jlens_v2.json"
CODE_EVAL_PROFILE = "qwen25_coder_7b_code_eval.json"
LAB_PROFILE = "qwen25_coder_7b_lab.json"
REQUIRED_NON_GEMMA_FIELDS = (
    "results_dir",
    "artifacts_dir",
    "format_style",
    "refusal_prefixes",
    "position_offsets",
    "model_revision",
    "test_per_class",
    "filter_by_behavior",
    "selection_split",
    "final_conditions_split",
    "baseline_gate",
    "causal_gate",
)


def validate_profile_config(config: dict, *, profile: str) -> None:
    if profile in {
        GEMMA_PROFILE,
        JLENS_PROFILE,
        JLENS_V2_PROFILE,
        CODE_EVAL_PROFILE,
        LAB_PROFILE,
    }:
        return
    missing = [
        field
        for field in REQUIRED_NON_GEMMA_FIELDS
        if field not in config
    ]
    if missing:
        raise ValueError(
            "profile "
            f"{profile} missing required profile fields: {missing}"
        )


def load_config(
    root: Path,
    profile: str = "gemma_2b.json",
) -> dict:
    config = json.loads(
        (root / "configs" / profile).read_text(encoding="utf-8")
    )
    validate_profile_config(config, profile=profile)
    return config


def project_paths(root: Path, config: dict) -> tuple[Path, Path]:
    results_name = config.get("results_dir", "results")
    artifacts_name = config.get("artifacts_dir", "artifacts")
    return root / results_name, root / artifacts_name


def _format_style(config: dict) -> str:
    return config.get("format_style", FORMAT_STYLE_GEMMA)


def _position_offsets(tokenizer, config: dict) -> tuple[int, ...]:
    if "position_offsets" in config:
        return tuple(int(item) for item in config["position_offsets"])
    return end_of_instruction_offsets(
        tokenizer,
        format_style=_format_style(config),
    )


def _refusal_token_ids(tokenizer, config: dict) -> tuple[int, ...]:
    prefixes = config.get("refusal_prefixes")
    if prefixes is None:
        return GEMMA_REFUSAL_TOKEN_IDS
    return resolve_refusal_token_ids(tokenizer, prefixes)


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _device_batch(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    device = _model_device(model)
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.inference_mode()
def last_token_logits(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    format_style: str = FORMAT_STYLE_GEMMA,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> torch.Tensor:
    collected = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(
            model,
            tokenize_instructions(
                tokenizer,
                texts,
                format_style=format_style,
            ),
        )
        with temporary_hooks(
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        ):
            output = model(**batch, use_cache=False)
        collected.append(output.logits[:, -1].detach().float().cpu())
    if not collected:
        raise ValueError("examples must not be empty")
    return torch.cat(collected)


def refusal_scores_for_examples(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    format_style: str = FORMAT_STYLE_GEMMA,
    refusal_token_ids: tuple[int, ...] = GEMMA_REFUSAL_TOKEN_IDS,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> torch.Tensor:
    logits = last_token_logits(
        model,
        tokenizer,
        examples,
        batch_size=batch_size,
        format_style=format_style,
        pre_hooks=pre_hooks,
        forward_hooks=forward_hooks,
    )
    return refusal_log_odds(
        logits,
        refusal_token_ids=refusal_token_ids,
    ).float()


@torch.inference_mode()
def capture_examples(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    position_offsets: tuple[int, ...],
    format_style: str = FORMAT_STYLE_GEMMA,
) -> torch.Tensor:
    blocks, _attentions, _mlps = transformer_module_lists(model)
    collected = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(
            model,
            tokenize_instructions(
                tokenizer,
                texts,
                format_style=format_style,
            ),
        )
        collected.append(
            capture_residual_grid(
                model,
                batch,
                block_modules=blocks,
                position_offsets=position_offsets,
                padding_side=tokenizer.padding_side,
            )
        )
    if not collected:
        raise ValueError("examples must not be empty")
    return torch.cat(collected)


def score_candidate_grid(
    model,
    tokenizer,
    candidates: Sequence[CandidateDirection],
    harmful_examples: Sequence[PromptExample],
    harmless_examples: Sequence[PromptExample],
    *,
    batch_size: int,
    format_style: str = FORMAT_STYLE_GEMMA,
    refusal_token_ids: tuple[int, ...] = GEMMA_REFUSAL_TOKEN_IDS,
    progress: bool = True,
) -> list[CandidateScore]:
    blocks, attentions, mlps = transformer_module_lists(model)
    baseline_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless_examples,
        batch_size=batch_size,
        format_style=format_style,
    )
    scores = []
    for index, candidate in enumerate(candidates):
        pre_hooks, forward_hooks = directional_ablation_hook_specs(
            blocks,
            attentions,
            mlps,
            candidate.vector,
        )
        ablated_harmful_logits = last_token_logits(
            model,
            tokenizer,
            harmful_examples,
            batch_size=batch_size,
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        ablated_harmless_logits = last_token_logits(
            model,
            tokenizer,
            harmless_examples,
            batch_size=batch_size,
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        addition_hooks = [
            (
                blocks[candidate.layer],
                residual_intervention_pre_hook(
                    candidate.vector,
                    mode="add",
                    coefficient=1.0,
                ),
            )
        ]
        addition_harmless_logits = last_token_logits(
            model,
            tokenizer,
            harmless_examples,
            batch_size=batch_size,
            format_style=format_style,
            pre_hooks=addition_hooks,
        )
        scores.append(
            CandidateScore(
                ablation_refusal_score=float(
                    refusal_log_odds(
                        ablated_harmful_logits,
                        refusal_token_ids=refusal_token_ids,
                    ).mean()
                ),
                addition_refusal_score=float(
                    refusal_log_odds(
                        addition_harmless_logits,
                        refusal_token_ids=refusal_token_ids,
                    ).mean()
                ),
                kl_divergence=float(
                    kl_divergence(
                        baseline_harmless_logits,
                        ablated_harmless_logits,
                    ).mean()
                ),
                layer=candidate.layer,
                n_layers=len(blocks),
            )
        )
        if progress and (
            (index + 1) % 8 == 0 or index + 1 == len(candidates)
        ):
            print(f"Pontuadas {index + 1}/{len(candidates)} direções.")
    return scores


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    max_new_tokens: int,
    format_style: str = FORMAT_STYLE_GEMMA,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
    invalid_on_decode_error: bool = False,
) -> list[str]:
    responses = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(
            model,
            tokenize_instructions(
                tokenizer,
                texts,
                format_style=format_style,
            ),
        )
        prompt_length = batch["input_ids"].shape[1]
        with temporary_hooks(
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        ):
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuations = generated[:, prompt_length:]
        try:
            decoded = tokenizer.batch_decode(
                continuations,
                skip_special_tokens=True,
            )
        except Exception:
            if not invalid_on_decode_error:
                raise
            decoded = ["\ufffd"] * continuations.shape[0]
        responses.extend(text.strip() for text in decoded)
    if not responses:
        raise ValueError("examples must not be empty")
    return responses


@torch.inference_mode()
def prompt_cross_entropy_losses(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    format_style: str = FORMAT_STYLE_GEMMA,
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> list[float]:
    losses = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(
            model,
            tokenize_instructions(
                tokenizer,
                texts,
                format_style=format_style,
            ),
        )
        with temporary_hooks(
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        ):
            logits = model(**batch, use_cache=False).logits
        token_losses = F.cross_entropy(
            logits[:, :-1].transpose(1, 2).float(),
            batch["input_ids"][:, 1:],
            reduction="none",
        )
        mask = batch["attention_mask"][:, 1:].to(token_losses)
        per_example = (token_losses * mask).sum(-1) / mask.sum(-1)
        losses.extend(per_example.detach().float().cpu().tolist())
    if not losses:
        raise ValueError("examples must not be empty")
    return losses


def _split_specs(config: dict) -> dict[str, tuple[str, int, int]]:
    specs = {
        "harmful_train": ("harmful", config["train_per_class"], 0),
        "harmless_train": ("harmless", config["train_per_class"], 1),
        "harmful_val": ("harmful", config["validation_per_class"], 2),
        "harmless_val": ("harmless", config["validation_per_class"], 3),
    }
    if "test_per_class" in config:
        specs["harmful_test"] = ("harmful", config["test_per_class"], 4)
        specs["harmless_test"] = ("harmless", config["test_per_class"], 5)
    return specs


def _sampled_splits(root: Path, config: dict) -> dict[str, list[PromptExample]]:
    if "dose_grid" in config or "source_direction" in config:
        raise ValueError(
            "dose profiles cannot use pipeline._sampled_splits; "
            "use refusal_direction_replication.qwen_dose.build_dose_splits"
        )
    dataset_root = root / "artifacts" / "datasets"
    sampled = {}
    for name, (label, size, seed_offset) in _split_specs(config).items():
        path = dataset_root / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run scripts/fetch_upstream_splits.py"
            )
        examples = load_local_prompt_corpus(path, label)
        sampled[name] = sample_examples(
            examples,
            size=size,
            seed=config["seed"] + seed_offset,
        )
    return sampled


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return float(sum(values) / len(values))


IMPLEMENTATION_COMMIT_FILENAME = "IMPLEMENTATION_COMMIT"
IMPLEMENTATION_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _implementation_commit_stamp_paths() -> list[Path]:
    return [
        Path("/content/refusal_direction") / IMPLEMENTATION_COMMIT_FILENAME,
        Path(__file__).resolve().parents[2] / IMPLEMENTATION_COMMIT_FILENAME,
    ]


def _usable_commit(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if IMPLEMENTATION_COMMIT_RE.fullmatch(stripped):
        return stripped
    return None


def implementation_commit_hash() -> str:
    env = _usable_commit(os.environ.get("REFUSAL_IMPLEMENTATION_COMMIT"))
    if env:
        return env
    for path in _implementation_commit_stamp_paths():
        if path.is_file():
            stamped = _usable_commit(path.read_text(encoding="utf-8"))
            if stamped:
                return stamped
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    git_commit = _usable_commit(
        completed.stdout if completed.returncode == 0 else None
    )
    if git_commit:
        return git_commit
    raise ValueError("implementation commit is missing")


def _provenance(
    model,
    dtype: torch.dtype,
    config: dict,
    *,
    run_id: str | None = None,
) -> dict:
    model_revision = getattr(model.config, "_commit_hash", None)
    payload = {
        "upstream_commit": config["upstream_commit"],
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": model_revision,
        "runtime": runtime_info(dtype).to_dict(),
    }
    if run_id is not None or _record_example_ids(config):
        payload["run_id"] = run_id or config.get("run_id")
        payload["implementation_commit"] = config.get(
            "implementation_commit"
        ) or implementation_commit_hash()
    return payload


def _write_raw_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _condition_payload(
    outputs: Sequence[str],
    scores: torch.Tensor,
    *,
    examples: Sequence[PromptExample] | None = None,
    config: dict | None = None,
) -> dict:
    count_valid = bool(config is not None and not config.get("filter_by_behavior", True))
    generated = summarize_generated_condition(
        outputs,
        examples=examples,
        count_valid=count_valid,
    )
    return {
        **generated,
        "mean_refusal_log_odds": float(scores.mean()),
        "per_example_refusal_log_odds": scores.tolist(),
    }


def _count_valid(config: dict) -> bool:
    return not config.get("filter_by_behavior", True)


def _load_model_for_config(config: dict):
    dtype = dtype_from_name(config["dtype"])
    model, tokenizer = load_causal_lm(
        config["model_name"],
        dtype=dtype,
        revision=config.get("model_revision"),
    )
    return model, tokenizer, dtype


def run_baseline_stage(
    root: Path,
    *,
    profile: str = "gemma_2b.json",
    run_id: str | None = None,
) -> dict:
    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    format_style = _format_style(config)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    model, tokenizer, dtype = _load_model_for_config(config)
    refusal_token_ids = _refusal_token_ids(tokenizer, config)

    harmful_train_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_train"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmless_train_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_train"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmful_val_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_val"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmless_val_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_val"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )

    harmful_train_filtered = maybe_filter_examples(
        splits["harmful_train"],
        harmful_train_scores,
        keep_positive=True,
        config=config,
    )
    harmless_train_filtered = maybe_filter_examples(
        splits["harmless_train"],
        harmless_train_scores,
        keep_positive=False,
        config=config,
    )
    harmful_val_filtered = maybe_filter_examples(
        splits["harmful_val"],
        harmful_val_scores,
        keep_positive=True,
        config=config,
    )
    harmless_val_filtered = maybe_filter_examples(
        splits["harmless_val"],
        harmless_val_scores,
        keep_positive=False,
        config=config,
    )
    if not all(
        (
            harmful_train_filtered,
            harmless_train_filtered,
            harmful_val_filtered,
            harmless_val_filtered,
        )
    ):
        raise ValueError("baseline filtering removed an entire prompt class")

    harmful_outputs = generate_responses(
        model,
        tokenizer,
        harmful_val_filtered,
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
        format_style=format_style,
        invalid_on_decode_error=_count_valid(config),
    )
    harmless_outputs = generate_responses(
        model,
        tokenizer,
        harmless_val_filtered,
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
        format_style=format_style,
        invalid_on_decode_error=_count_valid(config),
    )
    harmless_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless_val_filtered,
        batch_size=config["batch_size"],
        format_style=format_style,
    )

    results = {
        "sample_sizes": {
            name: len(examples) for name, examples in splits.items()
        },
        "filtered_sizes": {
            "harmful_train": len(harmful_train_filtered),
            "harmless_train": len(harmless_train_filtered),
            "harmful_val": len(harmful_val_filtered),
            "harmless_val": len(harmless_val_filtered),
        },
        "dataset_hashes": {
            name: hash_examples(examples) for name, examples in splits.items()
        },
        "harmful_validation": _condition_payload(
            harmful_outputs,
            maybe_filter_scores(
                harmful_val_scores,
                keep_positive=True,
                config=config,
            ),
            examples=harmful_val_filtered,
            config=config,
        ),
        "harmless_validation": _condition_payload(
            harmless_outputs,
            maybe_filter_scores(
                harmless_val_scores,
                keep_positive=False,
                config=config,
            ),
            examples=harmless_val_filtered,
            config=config,
        ),
        "harmless_prompt_cross_entropy": {
            "mean": _mean(harmless_losses),
            "per_example": harmless_losses,
        },
    }
    if _record_example_ids(config):
        results["example_ids"] = _ids_from_splits(splits)
        results["content_hashes"] = {
            name: content_hashes_for_examples(items)
            for name, items in splits.items()
        }
    artifact = ArtifactEnvelope(
        experiment="baseline",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config, run_id=run_id),
    )
    write_artifact(results_dir / "01_baseline.json", artifact)
    _write_raw_json(
        artifacts_dir / "generations" / "01_baseline.json",
        {
            "harmful": harmful_outputs,
            "harmless": harmless_outputs,
        },
    )
    release_compute_resources(model, tokenizer)
    return asdict(artifact)


def run_extract_stage(
    root: Path,
    *,
    profile: str = "gemma_2b.json",
    run_id: str | None = None,
) -> dict:
    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    format_style = _format_style(config)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    model, tokenizer, dtype = _load_model_for_config(config)
    refusal_token_ids = _refusal_token_ids(tokenizer, config)
    harmful_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_train"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_train"],
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmful = maybe_filter_examples(
        splits["harmful_train"],
        harmful_scores,
        keep_positive=True,
        config=config,
    )
    harmless = maybe_filter_examples(
        splits["harmless_train"],
        harmless_scores,
        keep_positive=False,
        config=config,
    )
    offsets = _position_offsets(tokenizer, config)
    harmful_activations = capture_examples(
        model,
        tokenizer,
        harmful,
        batch_size=config["batch_size"],
        position_offsets=offsets,
        format_style=format_style,
    )
    harmless_activations = capture_examples(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        position_offsets=offsets,
        format_style=format_style,
    )
    candidates = candidate_grid(
        harmful_activations,
        harmless_activations,
        position_offsets=offsets,
    )
    save_candidate_artifact(
        artifacts_dir / "candidates.pt",
        candidates,
    )
    results = {
        "harmful_examples": len(harmful),
        "harmless_examples": len(harmless),
        "activation_shape_harmful": list(harmful_activations.shape),
        "activation_shape_harmless": list(harmless_activations.shape),
        "position_offsets": list(offsets),
        "candidate_count": len(candidates),
        "candidate_norms": [float(item.vector.norm()) for item in candidates],
    }
    if _record_example_ids(config):
        results["example_ids"] = {
            "harmful_train": _ids_from_examples(harmful),
            "harmless_train": _ids_from_examples(harmless),
        }
        results["content_hashes"] = {
            "harmful_train": content_hashes_for_examples(harmful),
            "harmless_train": content_hashes_for_examples(harmless),
        }
    artifact = ArtifactEnvelope(
        experiment="extract_candidate_directions",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config, run_id=run_id),
    )
    write_artifact(results_dir / "02_candidates.json", artifact)
    release_compute_resources(model, tokenizer)
    return asdict(artifact)


def run_intervention_stage(
    root: Path,
    *,
    profile: str = "gemma_2b.json",
    run_id: str | None = None,
) -> dict:
    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    format_style = _format_style(config)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    candidates = load_candidate_artifact(
        artifacts_dir / "candidates.pt"
    )
    model, tokenizer, dtype = _load_model_for_config(config)
    refusal_token_ids = _refusal_token_ids(tokenizer, config)
    selection_split = config.get("selection_split", "validation")
    final_split = config.get("final_conditions_split", "validation")
    selection_harmful_raw, selection_harmless_raw = split_examples(
        splits,
        selection_split,
    )
    selection_harmful_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        selection_harmful_raw,
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    selection_harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        selection_harmless_raw,
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    selection_harmful = maybe_filter_examples(
        selection_harmful_raw,
        selection_harmful_scores,
        keep_positive=True,
        config=config,
    )
    selection_harmless = maybe_filter_examples(
        selection_harmless_raw,
        selection_harmless_scores,
        keep_positive=False,
        config=config,
    )
    scores = score_candidate_grid(
        model,
        tokenizer,
        candidates,
        selection_harmful,
        selection_harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    selected = select_candidate(candidates, scores)
    save_direction(
        artifacts_dir / "selected_direction.pt",
        selected,
    )
    if final_split == selection_split:
        harmful = selection_harmful
        harmless = selection_harmless
    else:
        final_harmful_raw, final_harmless_raw = split_examples(
            splits,
            final_split,
        )
        final_harmful_scores = refusal_scores_for_examples(
            model,
            tokenizer,
            final_harmful_raw,
            batch_size=config["batch_size"],
            format_style=format_style,
            refusal_token_ids=refusal_token_ids,
        )
        final_harmless_scores = refusal_scores_for_examples(
            model,
            tokenizer,
            final_harmless_raw,
            batch_size=config["batch_size"],
            format_style=format_style,
            refusal_token_ids=refusal_token_ids,
        )
        harmful = maybe_filter_examples(
            final_harmful_raw,
            final_harmful_scores,
            keep_positive=True,
            config=config,
        )
        harmless = maybe_filter_examples(
            final_harmless_raw,
            final_harmless_scores,
            keep_positive=False,
            config=config,
        )
    control = random_direction_control(
        selected.vector,
        seed=config["seed"] + 1000,
    )
    blocks, attentions, mlps = transformer_module_lists(model)
    ablation_pre, ablation_forward = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        selected.vector,
    )
    control_pre, control_forward = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        control,
    )
    addition_pre = [
        (
            blocks[selected.layer],
            residual_intervention_pre_hook(
                selected.vector,
                mode="add",
                coefficient=1.0,
            ),
        )
    ]

    conditions = {
        "baseline_harmful": (harmful, (), ()),
        "baseline_harmless": (harmless, (), ()),
        "ablation_harmful": (
            harmful,
            ablation_pre,
            ablation_forward,
        ),
        "ablation_harmless": (
            harmless,
            ablation_pre,
            ablation_forward,
        ),
        "addition_harmless": (harmless, addition_pre, ()),
        "random_control_harmful": (
            harmful,
            control_pre,
            control_forward,
        ),
    }
    aggregate_conditions = {}
    raw_outputs = {}
    for name, (examples, pre_hooks, forward_hooks) in conditions.items():
        logits = last_token_logits(
            model,
            tokenizer,
            examples,
            batch_size=config["batch_size"],
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        outputs = generate_responses(
            model,
            tokenizer,
            examples,
            batch_size=config["batch_size"],
            max_new_tokens=config["max_new_tokens"],
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
            invalid_on_decode_error=_count_valid(config),
        )
        scores_for_condition = refusal_log_odds(
            logits,
            refusal_token_ids=refusal_token_ids,
        ).float()
        aggregate_conditions[name] = _condition_payload(
            outputs,
            scores_for_condition,
            examples=examples,
            config=config,
        )
        raw_outputs[name] = outputs

    baseline_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
    )
    ablated_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
        pre_hooks=ablation_pre,
        forward_hooks=ablation_forward,
    )
    candidate_rows = [
        {
            "layer": candidate.layer,
            "position_offset": candidate.position_offset,
            **asdict(score),
        }
        for candidate, score in zip(candidates, scores)
    ]
    results = {
        "selected": {
            "layer": selected.layer,
            "position_offset": selected.position_offset,
            "norm": float(selected.vector.norm()),
        },
        "candidate_scores": candidate_rows,
        "conditions": aggregate_conditions,
        "ablation_harmless_kl": float(
            kl_divergence(
                baseline_harmless_logits,
                ablated_harmless_logits,
            ).mean()
        ),
    }
    if _record_example_ids(config):
        results["example_ids"] = {
            f"harmful_{SPLIT_NAME_TO_SUFFIX[selection_split]}": _ids_from_examples(
                selection_harmful
            ),
            f"harmless_{SPLIT_NAME_TO_SUFFIX[selection_split]}": _ids_from_examples(
                selection_harmless
            ),
            f"harmful_{SPLIT_NAME_TO_SUFFIX[final_split]}": _ids_from_examples(
                harmful
            ),
            f"harmless_{SPLIT_NAME_TO_SUFFIX[final_split]}": _ids_from_examples(
                harmless
            ),
        }
        results["content_hashes"] = {
            f"harmful_{SPLIT_NAME_TO_SUFFIX[selection_split]}": content_hashes_for_examples(
                selection_harmful
            ),
            f"harmless_{SPLIT_NAME_TO_SUFFIX[selection_split]}": content_hashes_for_examples(
                selection_harmless
            ),
            f"harmful_{SPLIT_NAME_TO_SUFFIX[final_split]}": content_hashes_for_examples(
                harmful
            ),
            f"harmless_{SPLIT_NAME_TO_SUFFIX[final_split]}": content_hashes_for_examples(
                harmless
            ),
        }
    artifact = ArtifactEnvelope(
        experiment="causal_interventions",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config, run_id=run_id),
    )
    write_artifact(results_dir / "03_interventions.json", artifact)
    _write_raw_json(
        artifacts_dir / "generations" / "03_interventions.json",
        raw_outputs,
    )
    release_compute_resources(model, tokenizer)
    return asdict(artifact)


def run_evaluation_stage(
    root: Path,
    *,
    profile: str = "gemma_2b.json",
    run_id: str | None = None,
) -> dict:
    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    format_style = _format_style(config)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    selected = load_direction(
        artifacts_dir / "selected_direction.pt"
    )
    intervention = json.loads(
        (results_dir / "03_interventions.json").read_text(
            encoding="utf-8"
        )
    )
    model, tokenizer, dtype = _load_model_for_config(config)
    refusal_token_ids = _refusal_token_ids(tokenizer, config)
    final_split = config.get("final_conditions_split", "validation")
    _final_harmful_raw, harmless_raw = split_examples(splits, final_split)
    harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        harmless_raw,
        batch_size=config["batch_size"],
        format_style=format_style,
        refusal_token_ids=refusal_token_ids,
    )
    harmless = maybe_filter_examples(
        harmless_raw,
        harmless_scores,
        keep_positive=False,
        config=config,
    )
    blocks, attentions, mlps = transformer_module_lists(model)
    ablation_pre, ablation_forward = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        selected.vector,
    )
    addition_pre = [
        (
            blocks[selected.layer],
            residual_intervention_pre_hook(
                selected.vector,
                mode="add",
                coefficient=1.0,
            ),
        )
    ]
    control = random_direction_control(
        selected.vector,
        seed=config["seed"] + 1000,
    )
    control_pre, control_forward = directional_ablation_hook_specs(
        blocks,
        attentions,
        mlps,
        control,
    )

    baseline_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
    )
    ablation_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
        pre_hooks=ablation_pre,
        forward_hooks=ablation_forward,
    )
    addition_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
        pre_hooks=addition_pre,
    )
    control_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        format_style=format_style,
        pre_hooks=control_pre,
        forward_hooks=control_forward,
    )
    conditions = intervention["results"]["conditions"]
    results = {
        "selected": intervention["results"]["selected"],
        "baseline": {
            "harmful_refusal_rate": conditions[
                "baseline_harmful"
            ]["refusal_rate"],
            "harmless_refusal_rate": conditions[
                "baseline_harmless"
            ]["refusal_rate"],
            "ce_delta": 0.0,
            "mean_cross_entropy": _mean(baseline_losses),
        },
        "ablation": {
            "harmful_refusal_rate": conditions[
                "ablation_harmful"
            ]["refusal_rate"],
            "harmless_refusal_rate": conditions[
                "ablation_harmless"
            ]["refusal_rate"],
            "ce_delta": mean_cross_entropy_delta(
                baseline_losses,
                ablation_losses,
            ),
            "mean_cross_entropy": _mean(ablation_losses),
            "harmless_kl": intervention["results"][
                "ablation_harmless_kl"
            ],
        },
        "addition": {
            "harmless_refusal_rate": conditions[
                "addition_harmless"
            ]["refusal_rate"],
            "ce_delta": mean_cross_entropy_delta(
                baseline_losses,
                addition_losses,
            ),
            "mean_cross_entropy": _mean(addition_losses),
        },
        "random_control": {
            "harmful_refusal_rate": conditions[
                "random_control_harmful"
            ]["refusal_rate"],
            "ce_delta": mean_cross_entropy_delta(
                baseline_losses,
                control_losses,
            ),
            "mean_cross_entropy": _mean(control_losses),
        },
        "per_example_cross_entropy": {
            "baseline": baseline_losses,
            "ablation": ablation_losses,
            "addition": addition_losses,
            "random_control": control_losses,
        },
    }
    if _record_example_ids(config):
        results["example_ids"] = {
            f"harmless_{SPLIT_NAME_TO_SUFFIX[final_split]}": _ids_from_examples(
                harmless
            ),
        }
        results["content_hashes"] = {
            f"harmless_{SPLIT_NAME_TO_SUFFIX[final_split]}": content_hashes_for_examples(
                harmless
            ),
        }
    artifact = ArtifactEnvelope(
        experiment="final_evaluation",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config, run_id=run_id),
    )
    write_artifact(results_dir / "final_evaluation.json", artifact)
    manifest = ArtifactEnvelope(
        experiment="execution_manifest",
        config=config,
        results={
            "completed_stages": [
                "baseline",
                "extract",
                "intervene",
                "evaluate",
            ],
            "result_files": [
                "01_baseline.json",
                "02_candidates.json",
                "03_interventions.json",
                "final_evaluation.json",
            ],
        },
        provenance=_provenance(model, dtype, config, run_id=run_id),
    )
    write_artifact(
        results_dir / "execution_manifest.json",
        manifest,
    )
    release_compute_resources(model, tokenizer)
    return asdict(artifact)


def generated_token_scores(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    max_new_tokens: int,
    format_style: str = FORMAT_STYLE_GEMMA,
) -> None:
    if not examples:
        raise ValueError("examples must not be empty")
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(
            model,
            tokenize_instructions(
                tokenizer,
                texts,
                format_style=format_style,
            ),
        )
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = getattr(generated, "scores", None)
        if not scores:
            raise ValueError("generate did not return generated-token scores")
        for step in scores:
            tensor = step.detach().float().cpu()
            if tensor.numel() == 0:
                raise ValueError(
                    "empty generated-token scores during preflight"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    "non-finite generated-token scores during preflight"
                )


def run_preflight_stage(
    root: Path,
    *,
    profile: str = "qwen25_coder_7b.json",
    load_model=None,
    cuda_available: bool | None = None,
    run_id: str | None = None,
) -> dict:
    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, _artifacts_dir = project_paths(root, config)
    if cuda_available is None:
        cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise ValueError("CUDA is not available")
    loader = load_model or _load_model_for_config
    model = None
    tokenizer = None
    try:
        model, tokenizer, dtype = loader(config)
        param_dtype = next(model.parameters()).dtype
        if param_dtype != torch.bfloat16:
            raise ValueError(
                f"expected bfloat16 load, received {param_dtype}"
            )
        blocks, _attentions, _mlps = transformer_module_lists(model)
        n_layers = len(blocks)
        if n_layers != 28:
            raise ValueError(
                f"expected 28 decoder layers, received {n_layers}"
            )
        residual_width = int(getattr(model.config, "hidden_size"))
        if residual_width != 3584:
            raise ValueError(
                f"expected residual width 3584, received {residual_width}"
            )
        rendered = format_chat_instruction(tokenizer, "probe")
        if not rendered:
            raise ValueError("chat template returned an empty string")
        chat_template_hash = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        refusal_token_ids = _refusal_token_ids(tokenizer, config)
        dataset_root = root / "artifacts" / "datasets"
        harmful = load_local_prompt_corpus(
            dataset_root / "harmful_train.json",
            "harmful",
        )[:2]
        harmless = load_local_prompt_corpus(
            dataset_root / "harmless_train.json",
            "harmless",
        )[:2]
        if len(harmful) < 2 or len(harmless) < 2:
            raise ValueError(
                "preflight requires two harmful and two harmless prompts"
            )
        probes = harmful + harmless
        format_style = _format_style(config)
        resolved = getattr(model.config, "_commit_hash", None)
        requested = config["model_revision"]
        if resolved != requested:
            raise ValueError(
                f"resolved revision {resolved} does not match requested {requested}"
            )
        generated_token_scores(
            model,
            tokenizer,
            probes,
            batch_size=config["batch_size"],
            max_new_tokens=min(8, int(config["max_new_tokens"])),
            format_style=format_style,
        )
        generate_responses(
            model,
            tokenizer,
            probes,
            batch_size=config["batch_size"],
            max_new_tokens=min(8, int(config["max_new_tokens"])),
            format_style=format_style,
            invalid_on_decode_error=True,
        )
        results = {
            "cuda_available": True,
            "dtype": "bfloat16",
            "n_layers": n_layers,
            "residual_width": residual_width,
            "chat_template_ok": True,
            "chat_template_hash": chat_template_hash,
            "refusal_token_id_count": len(refusal_token_ids),
            "refusal_token_ids": list(refusal_token_ids),
            "generation_finite": True,
            "probe_counts": {"harmful": 2, "harmless": 2},
            "probe_hashes": {
                "harmful": hash_examples(harmful),
                "harmless": hash_examples(harmless),
            },
        }
        artifact = ArtifactEnvelope(
            experiment="preflight",
            config=config,
            results=results,
            provenance=_provenance(model, dtype, config, run_id=run_id),
        )
        write_artifact(results_dir / "00_preflight.json", artifact)
        return asdict(artifact)
    finally:
        release_compute_resources(model, tokenizer)
