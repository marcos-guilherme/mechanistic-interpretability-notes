from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .artifacts import ArtifactEnvelope, write_artifact
from .data import PromptExample, hash_examples, load_local_prompt_corpus
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
    kl_divergence,
    mean_cross_entropy_delta,
    refusal_log_odds,
    refusal_rate,
)
from .model import (
    GEMMA_REFUSAL_TOKEN_IDS,
    capture_residual_grid,
    end_of_instruction_offsets,
    load_gemma,
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


def summarize_generated_condition(outputs: Sequence[str]) -> dict:
    return {
        "refusal_rate": refusal_rate(outputs),
        "refusal_flags": [contains_refusal(output) for output in outputs],
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


def gemma_module_lists(model) -> tuple[list, list, list]:
    blocks = list(model.model.layers)
    attentions = [block.self_attn for block in blocks]
    mlps = [block.mlp for block in blocks]
    return blocks, attentions, mlps


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
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> torch.Tensor:
    collected = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(model, tokenize_instructions(tokenizer, texts))
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
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> torch.Tensor:
    logits = last_token_logits(
        model,
        tokenizer,
        examples,
        batch_size=batch_size,
        pre_hooks=pre_hooks,
        forward_hooks=forward_hooks,
    )
    return refusal_log_odds(
        logits,
        refusal_token_ids=GEMMA_REFUSAL_TOKEN_IDS,
    ).float()


@torch.inference_mode()
def capture_examples(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    position_offsets: tuple[int, ...],
) -> torch.Tensor:
    blocks, _attentions, _mlps = gemma_module_lists(model)
    collected = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(model, tokenize_instructions(tokenizer, texts))
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
    progress: bool = True,
) -> list[CandidateScore]:
    blocks, attentions, mlps = gemma_module_lists(model)
    baseline_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless_examples,
        batch_size=batch_size,
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
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        ablated_harmless_logits = last_token_logits(
            model,
            tokenizer,
            harmless_examples,
            batch_size=batch_size,
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
            pre_hooks=addition_hooks,
        )
        scores.append(
            CandidateScore(
                ablation_refusal_score=float(
                    refusal_log_odds(
                        ablated_harmful_logits,
                        refusal_token_ids=GEMMA_REFUSAL_TOKEN_IDS,
                    ).mean()
                ),
                addition_refusal_score=float(
                    refusal_log_odds(
                        addition_harmless_logits,
                        refusal_token_ids=GEMMA_REFUSAL_TOKEN_IDS,
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
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> list[str]:
    responses = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(model, tokenize_instructions(tokenizer, texts))
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
        responses.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                continuations,
                skip_special_tokens=True,
            )
        )
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
    pre_hooks: Sequence[tuple[object, object]] = (),
    forward_hooks: Sequence[tuple[object, object]] = (),
) -> list[float]:
    losses = []
    for start in range(0, len(examples), batch_size):
        texts = [item.text for item in examples[start : start + batch_size]]
        batch = _device_batch(model, tokenize_instructions(tokenizer, texts))
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


def _load_config(root: Path) -> dict:
    return json.loads(
        (root / "configs" / "gemma_2b.json").read_text(encoding="utf-8")
    )


def _sampled_splits(root: Path, config: dict) -> dict[str, list[PromptExample]]:
    dataset_root = root / "artifacts" / "datasets"
    required = {
        "harmful_train": ("harmful", config["train_per_class"], 0),
        "harmless_train": ("harmless", config["train_per_class"], 1),
        "harmful_val": ("harmful", config["validation_per_class"], 2),
        "harmless_val": ("harmless", config["validation_per_class"], 3),
    }
    sampled = {}
    for name, (label, size, seed_offset) in required.items():
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


def _provenance(model, dtype: torch.dtype, config: dict) -> dict:
    model_revision = getattr(model.config, "_commit_hash", None)
    return {
        "upstream_commit": config["upstream_commit"],
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": model_revision,
        "runtime": runtime_info(dtype).to_dict(),
    }


def _write_raw_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _condition_payload(
    outputs: Sequence[str],
    scores: torch.Tensor,
) -> dict:
    generated = summarize_generated_condition(outputs)
    return {
        **generated,
        "mean_refusal_log_odds": float(scores.mean()),
        "per_example_refusal_log_odds": scores.tolist(),
    }


def _load_model_for_config(config: dict):
    dtype = dtype_from_name(config["dtype"])
    model, tokenizer = load_gemma(
        config["model_name"],
        dtype=dtype,
    )
    return model, tokenizer, dtype


def run_baseline_stage(root: Path) -> dict:
    root = root.resolve()
    config = _load_config(root)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    model, tokenizer, dtype = _load_model_for_config(config)

    harmful_train_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_train"],
        batch_size=config["batch_size"],
    )
    harmless_train_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_train"],
        batch_size=config["batch_size"],
    )
    harmful_val_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_val"],
        batch_size=config["batch_size"],
    )
    harmless_val_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_val"],
        batch_size=config["batch_size"],
    )

    harmful_train_filtered = filter_examples_by_score(
        splits["harmful_train"],
        harmful_train_scores,
        keep_positive=True,
    )
    harmless_train_filtered = filter_examples_by_score(
        splits["harmless_train"],
        harmless_train_scores,
        keep_positive=False,
    )
    harmful_val_filtered = filter_examples_by_score(
        splits["harmful_val"],
        harmful_val_scores,
        keep_positive=True,
    )
    harmless_val_filtered = filter_examples_by_score(
        splits["harmless_val"],
        harmless_val_scores,
        keep_positive=False,
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
    )
    harmless_outputs = generate_responses(
        model,
        tokenizer,
        harmless_val_filtered,
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
    )
    harmless_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless_val_filtered,
        batch_size=config["batch_size"],
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
            harmful_val_scores[harmful_val_scores > 0],
        ),
        "harmless_validation": _condition_payload(
            harmless_outputs,
            harmless_val_scores[harmless_val_scores < 0],
        ),
        "harmless_prompt_cross_entropy": {
            "mean": _mean(harmless_losses),
            "per_example": harmless_losses,
        },
    }
    artifact = ArtifactEnvelope(
        experiment="baseline",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config),
    )
    write_artifact(root / "results" / "01_baseline.json", artifact)
    _write_raw_json(
        root / "artifacts" / "generations" / "01_baseline.json",
        {
            "harmful": harmful_outputs,
            "harmless": harmless_outputs,
        },
    )
    return asdict(artifact)


def run_extract_stage(root: Path) -> dict:
    root = root.resolve()
    config = _load_config(root)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    model, tokenizer, dtype = _load_model_for_config(config)
    harmful_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_train"],
        batch_size=config["batch_size"],
    )
    harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_train"],
        batch_size=config["batch_size"],
    )
    harmful = filter_examples_by_score(
        splits["harmful_train"],
        harmful_scores,
        keep_positive=True,
    )
    harmless = filter_examples_by_score(
        splits["harmless_train"],
        harmless_scores,
        keep_positive=False,
    )
    offsets = end_of_instruction_offsets(tokenizer)
    harmful_activations = capture_examples(
        model,
        tokenizer,
        harmful,
        batch_size=config["batch_size"],
        position_offsets=offsets,
    )
    harmless_activations = capture_examples(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        position_offsets=offsets,
    )
    candidates = candidate_grid(
        harmful_activations,
        harmless_activations,
        position_offsets=offsets,
    )
    save_candidate_artifact(
        root / "artifacts" / "candidates.pt",
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
    artifact = ArtifactEnvelope(
        experiment="extract_candidate_directions",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config),
    )
    write_artifact(root / "results" / "02_candidates.json", artifact)
    return asdict(artifact)


def run_intervention_stage(root: Path) -> dict:
    root = root.resolve()
    config = _load_config(root)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    candidates = load_candidate_artifact(
        root / "artifacts" / "candidates.pt"
    )
    model, tokenizer, dtype = _load_model_for_config(config)
    harmful_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmful_val"],
        batch_size=config["batch_size"],
    )
    harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_val"],
        batch_size=config["batch_size"],
    )
    harmful = filter_examples_by_score(
        splits["harmful_val"],
        harmful_scores,
        keep_positive=True,
    )
    harmless = filter_examples_by_score(
        splits["harmless_val"],
        harmless_scores,
        keep_positive=False,
    )
    scores = score_candidate_grid(
        model,
        tokenizer,
        candidates,
        harmful,
        harmless,
        batch_size=config["batch_size"],
    )
    selected = select_candidate(candidates, scores)
    save_direction(
        root / "artifacts" / "selected_direction.pt",
        selected,
    )
    control = random_direction_control(
        selected.vector,
        seed=config["seed"] + 1000,
    )
    blocks, attentions, mlps = gemma_module_lists(model)
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
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        outputs = generate_responses(
            model,
            tokenizer,
            examples,
            batch_size=config["batch_size"],
            max_new_tokens=config["max_new_tokens"],
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        scores_for_condition = refusal_log_odds(
            logits,
            refusal_token_ids=GEMMA_REFUSAL_TOKEN_IDS,
        ).float()
        aggregate_conditions[name] = _condition_payload(
            outputs,
            scores_for_condition,
        )
        raw_outputs[name] = outputs

    baseline_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
    )
    ablated_harmless_logits = last_token_logits(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
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
    artifact = ArtifactEnvelope(
        experiment="causal_interventions",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config),
    )
    write_artifact(root / "results" / "03_interventions.json", artifact)
    _write_raw_json(
        root / "artifacts" / "generations" / "03_interventions.json",
        raw_outputs,
    )
    return asdict(artifact)


def run_evaluation_stage(root: Path) -> dict:
    root = root.resolve()
    config = _load_config(root)
    seed_everything(config["seed"])
    splits = _sampled_splits(root, config)
    selected = load_direction(
        root / "artifacts" / "selected_direction.pt"
    )
    intervention = json.loads(
        (root / "results" / "03_interventions.json").read_text(
            encoding="utf-8"
        )
    )
    model, tokenizer, dtype = _load_model_for_config(config)
    harmless_scores = refusal_scores_for_examples(
        model,
        tokenizer,
        splits["harmless_val"],
        batch_size=config["batch_size"],
    )
    harmless = filter_examples_by_score(
        splits["harmless_val"],
        harmless_scores,
        keep_positive=False,
    )
    blocks, attentions, mlps = gemma_module_lists(model)
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
    )
    ablation_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        pre_hooks=ablation_pre,
        forward_hooks=ablation_forward,
    )
    addition_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
        pre_hooks=addition_pre,
    )
    control_losses = prompt_cross_entropy_losses(
        model,
        tokenizer,
        harmless,
        batch_size=config["batch_size"],
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
    artifact = ArtifactEnvelope(
        experiment="final_evaluation",
        config=config,
        results=results,
        provenance=_provenance(model, dtype, config),
    )
    write_artifact(root / "results" / "final_evaluation.json", artifact)
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
        provenance=_provenance(model, dtype, config),
    )
    write_artifact(
        root / "results" / "execution_manifest.json",
        manifest,
    )
    return asdict(artifact)
