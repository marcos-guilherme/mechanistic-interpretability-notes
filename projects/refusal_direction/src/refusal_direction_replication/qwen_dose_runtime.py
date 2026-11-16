"""Resumable Qwen dose follow-up orchestration.

The original causal pilot stays frozen. This runtime reuses its selected
layer and direction, measures a fractional dose grid on fresh leftover
splits, and confirms one frozen coefficient pair.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import ArtifactEnvelope, atomic_write_json, write_artifact
from .data import PromptExample, content_hashes_for_examples
from .directions import CandidateDirection, load_direction
from .gates import (
    STATUS_CAUSAL_GATE_FAILED,
    STATUS_OPERATIONAL_ERROR,
    STATUS_PASS,
    classify_operational_error,
    evaluate_causal_gate,
)
from .interventions import (
    directional_ablation_hook_specs,
    residual_intervention_pre_hook,
)
from .metrics import kl_divergence, refusal_log_odds
from .pipeline import (
    _condition_payload,
    _format_style,
    _load_model_for_config,
    _refusal_token_ids,
    generate_responses,
    implementation_commit_hash,
    last_token_logits,
    load_config,
    project_paths,
    random_direction_control,
    release_compute_resources,
    transformer_module_lists,
)
from .qwen_dose import (
    EXPECTED_SOURCE_LAYER,
    EXPECTED_SOURCE_OFFSET,
    STATUS_NO_DOSE_QUALIFIED,
    build_dose_evaluation,
    build_dose_splits,
    coefficient_checkpoint_name,
    load_source_envelopes,
    select_dose_coefficients,
    source_artifact_hashes,
)
from .qwen_pilot import resolve_run_id
from .run_progress import RunProgressStore
from .runtime import seed_everything

DOSE_STAGES = ("prepare", "sweep", "confirm")
DOSE_PROFILE = "qwen25_coder_7b_dose.json"
TERMINAL_SCIENCE = frozenset(
    {STATUS_PASS, STATUS_CAUSAL_GATE_FAILED, STATUS_NO_DOSE_QUALIFIED}
)
MeasureFn = Callable[..., Any]
StageRunner = Callable[..., dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _dose_provenance(
    *,
    config: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    source_pilot_run_id: str,
    source_pilot_implementation_commit: str,
    source_hashes: Mapping[str, str],
    resolved_model_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "implementation_commit": implementation_commit,
        "model_name": config["model_name"],
        "requested_model_revision": config["model_revision"],
        "resolved_model_revision": resolved_model_revision or config["model_revision"],
        "upstream_commit": config["upstream_commit"],
        "source_pilot_run_id": source_pilot_run_id,
        "source_pilot_implementation_commit": source_pilot_implementation_commit,
        "source_artifact_hashes": dict(source_hashes),
    }


def _write_compact(
    path: Path,
    *,
    experiment: str,
    config: dict[str, Any],
    results: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    artifact = ArtifactEnvelope(
        experiment=experiment,
        config=config,
        results=results,
        provenance=provenance,
    )
    write_artifact(path, artifact)
    return _read_json(path)


def _grid_values(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def load_reusable_artifact(
    path: Path,
    *,
    run_id: str,
    source_hashes: Mapping[str, str],
    implementation_commit: str | None = None,
    dose_grid: Sequence[float] | None = None,
    coefficient: float | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    provenance = payload.get("provenance") or payload
    stored_run = provenance.get("run_id")
    if stored_run != run_id:
        raise ValueError(f"stale run_id in {path.name}: {stored_run!r}")
    stored_hashes = provenance.get("source_artifact_hashes")
    if stored_hashes != dict(source_hashes):
        raise ValueError(f"stale source hash in {path.name}")
    if implementation_commit is not None:
        stored_commit = provenance.get("implementation_commit")
        if stored_commit != implementation_commit:
            raise ValueError(f"stale implementation_commit in {path.name}")
    if dose_grid is not None:
        stored_grid = _grid_values((payload.get("config") or {}).get("dose_grid"))
        if stored_grid != [float(item) for item in dose_grid]:
            raise ValueError(f"stale dose_grid in {path.name}")
    if coefficient is not None:
        stored_coefficient = (payload.get("results") or {}).get("coefficient")
        if stored_coefficient is None or float(stored_coefficient) != float(
            coefficient
        ):
            raise ValueError(f"coefficient mismatch in {path.name}")
    return payload


def _assert_resolved_revision(model, config: Mapping[str, Any]) -> str:
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    requested = config["model_revision"]
    if resolved != requested:
        raise ValueError(
            f"resolved commit_hash {resolved} does not match requested revision {requested}"
        )
    return str(resolved)


def _source_context(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    envelopes = load_source_envelopes(root)
    evidence = config["source_evidence"]
    baseline_path = root / evidence["baseline"]
    interventions_path = root / evidence["interventions"]
    evaluation_path = root / "results" / "qwen25_coder_7b" / "pilot_evaluation.json"
    direction_path = root / config["source_direction"]
    evaluation = envelopes["pilot_evaluation.json"]
    if evaluation.get("status") != STATUS_CAUSAL_GATE_FAILED:
        raise ValueError("source pilot must remain causal_gate_failed")
    interventions = envelopes["03_interventions.json"]
    selected = (interventions.get("results") or {}).get("selected") or {}
    layer = int(selected.get("layer", -1))
    offset = int(selected.get("position_offset", 0))
    if layer != EXPECTED_SOURCE_LAYER or offset != EXPECTED_SOURCE_OFFSET:
        raise ValueError(
            f"source selected layer must be {EXPECTED_SOURCE_LAYER} "
            f"offset {EXPECTED_SOURCE_OFFSET}, got layer {layer} offset {offset}"
        )
    direction = load_direction(direction_path)
    if (
        direction.layer != EXPECTED_SOURCE_LAYER
        or direction.position_offset != EXPECTED_SOURCE_OFFSET
    ):
        raise ValueError(
            f"source direction layer must be {EXPECTED_SOURCE_LAYER} "
            f"offset {EXPECTED_SOURCE_OFFSET}"
        )
    provenance = interventions.get("provenance") or {}
    source_run = str(
        evaluation.get("run_id") or provenance.get("run_id") or ""
    )
    source_commit = str(
        evaluation.get("implementation_commit")
        or provenance.get("implementation_commit")
        or ""
    )
    if not source_run or not source_commit:
        raise ValueError("source pilot run/commit is missing")
    hashes = source_artifact_hashes(root, config)
    return {
        "direction": direction,
        "source_pilot_run_id": source_run,
        "source_pilot_implementation_commit": source_commit,
        "source_hashes": hashes,
        "baseline_path": baseline_path,
        "interventions_path": interventions_path,
        "evaluation_path": evaluation_path,
    }


def _rebuild_splits(root: Path, config: Mapping[str, Any]):
    dataset_root = root / "artifacts" / "datasets"
    return build_dose_splits(
        original_results_dir=root / "results" / "qwen25_coder_7b",
        harmful_test_path=dataset_root / "harmful_test.json",
        harmless_test_path=dataset_root / "harmless_test.json",
        seed=int(config["dose_split_seed"]),
        selection_size=int(config["dose_selection_per_class"]),
        confirmation_size=int(config["dose_confirmation_per_class"]),
    )


def _make_real_measure(
    model,
    tokenizer,
    config: Mapping[str, Any],
    direction: CandidateDirection,
    control: Any,
) -> MeasureFn:
    format_style = _format_style(dict(config))
    refusal_token_ids = _refusal_token_ids(tokenizer, dict(config))
    blocks, attentions, mlps = transformer_module_lists(model)

    def _hooks(condition: str, coefficient: float | None):
        coeff = 1.0 if coefficient is None else float(coefficient)
        if condition.startswith("baseline"):
            return (), ()
        if condition in {"ablation_harmful", "ablation_harmless", "kl"}:
            return directional_ablation_hook_specs(
                blocks,
                attentions,
                mlps,
                direction.vector,
                coefficient=coeff,
            )
        if condition == "random_control_harmful":
            return directional_ablation_hook_specs(
                blocks,
                attentions,
                mlps,
                control,
                coefficient=coeff,
            )
        if condition == "addition_harmless":
            return (
                [
                    (
                        blocks[direction.layer],
                        residual_intervention_pre_hook(
                            direction.vector,
                            mode="add",
                            coefficient=coeff,
                        ),
                    )
                ],
                (),
            )
        raise ValueError(f"unknown dose condition: {condition}")

    def measure(
        *,
        condition: str,
        coefficient: float | None,
        examples: Sequence[PromptExample],
        split: str,
        **_kwargs,
    ) -> dict[str, Any]:
        del split
        pre_hooks, forward_hooks = _hooks(condition, coefficient)
        if condition == "kl":
            baseline_logits = last_token_logits(
                model,
                tokenizer,
                examples,
                batch_size=int(config["batch_size"]),
                format_style=format_style,
            )
            ablated_logits = last_token_logits(
                model,
                tokenizer,
                examples,
                batch_size=int(config["batch_size"]),
                format_style=format_style,
                pre_hooks=pre_hooks,
                forward_hooks=forward_hooks,
            )
            return {
                "harmless_kl": float(
                    kl_divergence(baseline_logits, ablated_logits).mean()
                )
            }
        logits = last_token_logits(
            model,
            tokenizer,
            examples,
            batch_size=int(config["batch_size"]),
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        outputs = generate_responses(
            model,
            tokenizer,
            examples,
            batch_size=int(config["batch_size"]),
            max_new_tokens=int(config["max_new_tokens"]),
            format_style=format_style,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
            invalid_on_decode_error=True,
        )
        scores = refusal_log_odds(logits, refusal_token_ids=refusal_token_ids)
        return _condition_payload(
            outputs,
            scores,
            examples=examples,
            config=dict(config),
        )

    return measure


def _bind_measure(
    measure: MeasureFn | None,
    *,
    model,
    tokenizer,
    config: Mapping[str, Any],
    direction: CandidateDirection,
    control,
) -> MeasureFn:
    if measure is not None:
        return measure
    return _make_real_measure(model, tokenizer, config, direction, control)


def run_dose_prepare(
    root: Path,
    *,
    profile: str = DOSE_PROFILE,
    run_id: str,
    implementation_commit: str,
) -> dict[str, Any]:
    config = load_config(root, profile=profile)
    results_dir, _artifacts_dir = project_paths(root, config)
    source = _source_context(root, config)
    existing = load_reusable_artifact(
        results_dir / "00_dose_prepare.json",
        run_id=run_id,
        source_hashes=source["source_hashes"],
        implementation_commit=implementation_commit,
        dose_grid=config.get("dose_grid"),
    )
    if existing is not None:
        return existing
    splits, metadata = _rebuild_splits(root, config)
    del splits
    provenance = _dose_provenance(
        config=config,
        run_id=run_id,
        implementation_commit=implementation_commit,
        source_pilot_run_id=source["source_pilot_run_id"],
        source_pilot_implementation_commit=source["source_pilot_implementation_commit"],
        source_hashes=source["source_hashes"],
    )
    return _write_compact(
        results_dir / "00_dose_prepare.json",
        experiment="dose_prepare",
        config=config,
        results={
            "selected": {
                "layer": EXPECTED_SOURCE_LAYER,
                "position_offset": EXPECTED_SOURCE_OFFSET,
            },
            "splits": metadata,
            "source_pilot_status": STATUS_CAUSAL_GATE_FAILED,
            "source_pilot_run_id": source["source_pilot_run_id"],
            "source_pilot_implementation_commit": source[
                "source_pilot_implementation_commit"
            ],
        },
        provenance=provenance,
    )


def _selection_examples(splits) -> tuple[list[PromptExample], list[PromptExample]]:
    return (
        list(splits.harmful_dose_selection),
        list(splits.harmless_dose_selection),
    )


def _confirmation_examples(splits) -> tuple[list[PromptExample], list[PromptExample]]:
    return (
        list(splits.harmful_dose_confirmation),
        list(splits.harmless_dose_confirmation),
    )


def _write_evaluation(
    results_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _write_json(results_dir / "dose_followup_evaluation.json", payload)
    return payload


def _write_manifest(
    results_dir: Path,
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    completed_stages: Sequence[str],
    result_files: Sequence[str],
    status: str,
) -> None:
    _write_compact(
        results_dir / "execution_manifest.json",
        experiment="execution_manifest",
        config=config,
        results={
            "completed_stages": list(completed_stages),
            "result_files": list(result_files),
            "status": status,
        },
        provenance=provenance,
    )


def run_dose_sweep(
    root: Path,
    *,
    profile: str = DOSE_PROFILE,
    run_id: str,
    implementation_commit: str,
    store: RunProgressStore,
    measure: MeasureFn | None = None,
    load_model=None,
    resource_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(root, profile=profile)
    results_dir, _artifacts_dir = project_paths(root, config)
    source = _source_context(root, config)
    provenance = _dose_provenance(
        config=config,
        run_id=run_id,
        implementation_commit=implementation_commit,
        source_pilot_run_id=source["source_pilot_run_id"],
        source_pilot_implementation_commit=source["source_pilot_implementation_commit"],
        source_hashes=source["source_hashes"],
    )
    splits, _metadata = _rebuild_splits(root, config)
    harmful, harmless = _selection_examples(splits)
    grid = [float(item) for item in config["dose_grid"]]
    model = None
    tokenizer = None
    bound: MeasureFn | None = measure
    holder = resource_holder if resource_holder is not None else {}
    try:
        seed_everything(int(config["seed"]))
        if bound is None:
            loader = load_model or _load_model_for_config
            model, tokenizer, _dtype = loader(config)
            resolved = _assert_resolved_revision(model, config)
            holder["model"] = model
            holder["tokenizer"] = tokenizer
            provenance = _dose_provenance(
                config=config,
                run_id=run_id,
                implementation_commit=implementation_commit,
                source_pilot_run_id=source["source_pilot_run_id"],
                source_pilot_implementation_commit=source[
                    "source_pilot_implementation_commit"
                ],
                source_hashes=source["source_hashes"],
                resolved_model_revision=resolved,
            )
            control = random_direction_control(
                source["direction"].vector,
                seed=int(config["seed"]) + 1000,
            )
            bound = _bind_measure(
                None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                direction=source["direction"],
                control=control,
            )
        assert bound is not None
        baseline_path = results_dir / "01_dose_baseline.json"
        baseline = load_reusable_artifact(
            baseline_path,
            run_id=run_id,
            source_hashes=source["source_hashes"],
            implementation_commit=implementation_commit,
            dose_grid=grid,
        )
        if baseline is None:
            harmful_baseline = bound(
                condition="baseline_harmful",
                coefficient=None,
                examples=harmful,
                split="dose_selection",
            )
            harmless_baseline = bound(
                condition="baseline_harmless",
                coefficient=None,
                examples=harmless,
                split="dose_selection",
            )
            baseline = _write_compact(
                baseline_path,
                experiment="dose_baseline",
                config=config,
                results={
                    "harmful": harmful_baseline,
                    "harmless": harmless_baseline,
                    "example_ids": {
                        "harmful_dose_selection": [item.id for item in harmful],
                        "harmless_dose_selection": [item.id for item in harmless],
                    },
                    "content_hashes": {
                        "harmful_dose_selection": content_hashes_for_examples(
                            harmful
                        ),
                        "harmless_dose_selection": content_hashes_for_examples(
                            harmless
                        ),
                    },
                },
                provenance=provenance,
            )
        baseline_harmful_rate = float(baseline["results"]["harmful"]["refusal_rate"])
        baseline_harmless_rate = float(baseline["results"]["harmless"]["refusal_rate"])
        completed = 0
        ablation_curve: list[dict[str, float]] = []
        addition_curve: list[dict[str, float]] = []
        for coefficient in grid:
            checkpoint_path = results_dir / coefficient_checkpoint_name(coefficient)
            checkpoint = load_reusable_artifact(
                checkpoint_path,
                run_id=run_id,
                source_hashes=source["source_hashes"],
                implementation_commit=implementation_commit,
                dose_grid=grid,
                coefficient=coefficient,
            )
            if checkpoint is None:
                ablation_harmful = bound(
                    condition="ablation_harmful",
                    coefficient=coefficient,
                    examples=harmful,
                    split="dose_selection",
                )
                ablation_harmless = bound(
                    condition="ablation_harmless",
                    coefficient=coefficient,
                    examples=harmless,
                    split="dose_selection",
                )
                addition_harmless = bound(
                    condition="addition_harmless",
                    coefficient=coefficient,
                    examples=harmless,
                    split="dose_selection",
                )
                kl_payload = bound(
                    condition="kl",
                    coefficient=coefficient,
                    examples=harmless,
                    split="dose_selection",
                )
                harmful_drop = baseline_harmful_rate - float(
                    ablation_harmful["refusal_rate"]
                )
                harmless_increase = float(
                    addition_harmless["refusal_rate"]
                ) - baseline_harmless_rate
                checkpoint = _write_compact(
                    checkpoint_path,
                    experiment="dose_checkpoint",
                    config=config,
                    results={
                        "coefficient": coefficient,
                        "harmful_refusal_drop": harmful_drop,
                        "harmless_kl": float(kl_payload["harmless_kl"]),
                        "harmless_refusal_increase": harmless_increase,
                        "conditions": {
                            "ablation_harmful": ablation_harmful,
                            "ablation_harmless": ablation_harmless,
                            "addition_harmless": addition_harmless,
                        },
                    },
                    provenance=provenance,
                )
            completed += 1
            store.update_progress(
                completed_units=completed,
                total_units=len(grid),
            )
            results = checkpoint["results"]
            ablation_curve.append(
                {
                    "coefficient": float(results["coefficient"]),
                    "harmful_refusal_drop": float(results["harmful_refusal_drop"]),
                    "harmless_kl": float(results["harmless_kl"]),
                }
            )
            addition_curve.append(
                {
                    "coefficient": float(results["coefficient"]),
                    "harmless_refusal_increase": float(
                        results["harmless_refusal_increase"]
                    ),
                }
            )
        selected = select_dose_coefficients(
            ablation_curve=ablation_curve,
            addition_curve=addition_curve,
            min_harmful_ablation_drop=float(
                config["causal_gate"]["min_harmful_ablation_drop"]
            ),
            max_harmless_ablation_kl=float(
                config["causal_gate"]["max_harmless_ablation_kl"]
            ),
            min_harmless_addition_increase=float(
                config["causal_gate"]["min_harmless_addition_increase"]
            ),
        )
        selection = _write_compact(
            results_dir / "02_dose_selection.json",
            experiment="dose_selection",
            config=config,
            results={
                **selected,
                "baseline_harmful_refusal_rate": baseline_harmful_rate,
                "baseline_harmless_refusal_rate": baseline_harmless_rate,
            },
            provenance=provenance,
        )
        if selected["selected_pair"] is None:
            evaluation = build_dose_evaluation(
                status=STATUS_NO_DOSE_QUALIFIED,
                operational_success=True,
                run_id=run_id,
                implementation_commit=implementation_commit,
                source_pilot_run_id=source["source_pilot_run_id"],
                source_pilot_implementation_commit=source[
                    "source_pilot_implementation_commit"
                ],
                source_artifact_hashes=source["source_hashes"],
                model_revision=str(config["model_revision"]),
                selected_pair=None,
            )
            _write_evaluation(results_dir, evaluation)
            _write_manifest(
                results_dir,
                config=config,
                provenance=provenance,
                completed_stages=["prepare", "sweep"],
                result_files=[
                    "00_dose_prepare.json",
                    "01_dose_baseline.json",
                    "02_dose_selection.json",
                    "dose_followup_evaluation.json",
                    "execution_manifest.json",
                ],
                status=STATUS_NO_DOSE_QUALIFIED,
            )
            store.abort(gate="dose", reason="no qualifying dose pair")
            return evaluation
        return selection
    finally:
        del model
        del tokenizer
        del bound


def _evaluation_from_confirmation(
    confirmation: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run_id: str,
    implementation_commit: str,
    source: Mapping[str, Any],
    provenance: dict[str, Any],
    results_dir: Path,
    store: RunProgressStore,
) -> dict[str, Any]:
    conf_results = confirmation["results"]
    selected_pair = conf_results["selected_pair"]
    gate = evaluate_causal_gate(
        baseline_harmful_refusal_rate=float(
            conf_results["baseline"]["harmful_refusal_rate"]
        ),
        baseline_harmless_refusal_rate=float(
            conf_results["baseline"]["harmless_refusal_rate"]
        ),
        ablation_harmful_refusal_rate=float(
            conf_results["ablation"]["harmful_refusal_rate"]
        ),
        addition_harmless_refusal_rate=float(
            conf_results["addition"]["harmless_refusal_rate"]
        ),
        random_control_harmful_refusal_rate=float(
            conf_results["random_control"]["harmful_refusal_rate"]
        ),
        ablation_harmless_kl=float(conf_results["ablation"]["harmless_kl"]),
        thresholds=config["causal_gate"],
    )
    status = STATUS_PASS if gate.passed else STATUS_CAUSAL_GATE_FAILED
    evaluation = build_dose_evaluation(
        status=status,
        operational_success=True,
        causal_gate=gate,
        run_id=run_id,
        implementation_commit=implementation_commit,
        source_pilot_run_id=source["source_pilot_run_id"],
        source_pilot_implementation_commit=source[
            "source_pilot_implementation_commit"
        ],
        source_artifact_hashes=source["source_hashes"],
        model_revision=str(config["model_revision"]),
        selected_pair=selected_pair,
    )
    _write_evaluation(results_dir, evaluation)
    _write_manifest(
        results_dir,
        config=dict(config),
        provenance=provenance,
        completed_stages=["prepare", "sweep", "confirm"],
        result_files=[
            "00_dose_prepare.json",
            "01_dose_baseline.json",
            "02_dose_selection.json",
            "03_dose_confirmation.json",
            "dose_followup_evaluation.json",
            "execution_manifest.json",
        ],
        status=status,
    )
    if not gate.passed:
        store.abort(gate="causal", reason=gate.reason)
    return evaluation


def run_dose_confirm(
    root: Path,
    *,
    profile: str = DOSE_PROFILE,
    run_id: str,
    implementation_commit: str,
    store: RunProgressStore,
    measure: MeasureFn | None = None,
    load_model=None,
    resource_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(root, profile=profile)
    results_dir, _artifacts_dir = project_paths(root, config)
    source = _source_context(root, config)
    provenance = _dose_provenance(
        config=config,
        run_id=run_id,
        implementation_commit=implementation_commit,
        source_pilot_run_id=source["source_pilot_run_id"],
        source_pilot_implementation_commit=source["source_pilot_implementation_commit"],
        source_hashes=source["source_hashes"],
    )
    grid = [float(item) for item in config["dose_grid"]]
    selection = load_reusable_artifact(
        results_dir / "02_dose_selection.json",
        run_id=run_id,
        source_hashes=source["source_hashes"],
        implementation_commit=implementation_commit,
        dose_grid=grid,
    )
    if selection is None:
        raise ValueError("dose selection artifact is missing")
    selected_pair = selection["results"].get("selected_pair")
    if not selected_pair:
        evaluation = build_dose_evaluation(
            status=STATUS_NO_DOSE_QUALIFIED,
            operational_success=True,
            run_id=run_id,
            implementation_commit=implementation_commit,
            source_pilot_run_id=source["source_pilot_run_id"],
            source_pilot_implementation_commit=source[
                "source_pilot_implementation_commit"
            ],
            source_artifact_hashes=source["source_hashes"],
            model_revision=str(config["model_revision"]),
            selected_pair=None,
        )
        _write_evaluation(results_dir, evaluation)
        _write_manifest(
            results_dir,
            config=config,
            provenance=provenance,
            completed_stages=["prepare", "sweep"],
            result_files=[
                "00_dose_prepare.json",
                "01_dose_baseline.json",
                "02_dose_selection.json",
                "dose_followup_evaluation.json",
                "execution_manifest.json",
            ],
            status=STATUS_NO_DOSE_QUALIFIED,
        )
        store.abort(gate="dose", reason="no qualifying dose pair")
        return evaluation
    existing_confirmation = load_reusable_artifact(
        results_dir / "03_dose_confirmation.json",
        run_id=run_id,
        source_hashes=source["source_hashes"],
        implementation_commit=implementation_commit,
        dose_grid=grid,
    )
    if existing_confirmation is not None:
        stored_pair = (existing_confirmation.get("results") or {}).get("selected_pair")
        if stored_pair != selected_pair:
            raise ValueError("stale selected_pair in 03_dose_confirmation.json")
        return _evaluation_from_confirmation(
            existing_confirmation,
            config=config,
            run_id=run_id,
            implementation_commit=implementation_commit,
            source=source,
            provenance=provenance,
            results_dir=results_dir,
            store=store,
        )
    ablation_coefficient = float(selected_pair["ablation"])
    addition_coefficient = float(selected_pair["addition"])
    splits, _metadata = _rebuild_splits(root, config)
    harmful, harmless = _confirmation_examples(splits)
    model = None
    tokenizer = None
    bound: MeasureFn | None = measure
    holder = resource_holder if resource_holder is not None else {}
    try:
        seed_everything(int(config["seed"]))
        if bound is None:
            loader = load_model or _load_model_for_config
            model, tokenizer, _dtype = loader(config)
            resolved = _assert_resolved_revision(model, config)
            holder["model"] = model
            holder["tokenizer"] = tokenizer
            provenance = _dose_provenance(
                config=config,
                run_id=run_id,
                implementation_commit=implementation_commit,
                source_pilot_run_id=source["source_pilot_run_id"],
                source_pilot_implementation_commit=source[
                    "source_pilot_implementation_commit"
                ],
                source_hashes=source["source_hashes"],
                resolved_model_revision=resolved,
            )
            control = random_direction_control(
                source["direction"].vector,
                seed=int(config["seed"]) + 1000,
            )
            bound = _bind_measure(
                None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                direction=source["direction"],
                control=control,
            )
        assert bound is not None
        harmful_baseline = bound(
            condition="baseline_harmful",
            coefficient=None,
            examples=harmful,
            split="dose_confirmation",
        )
        harmless_baseline = bound(
            condition="baseline_harmless",
            coefficient=None,
            examples=harmless,
            split="dose_confirmation",
        )
        ablation_harmful = bound(
            condition="ablation_harmful",
            coefficient=ablation_coefficient,
            examples=harmful,
            split="dose_confirmation",
        )
        addition_harmless = bound(
            condition="addition_harmless",
            coefficient=addition_coefficient,
            examples=harmless,
            split="dose_confirmation",
        )
        random_control = bound(
            condition="random_control_harmful",
            coefficient=ablation_coefficient,
            examples=harmful,
            split="dose_confirmation",
        )
        kl_payload = bound(
            condition="kl",
            coefficient=ablation_coefficient,
            examples=harmless,
            split="dose_confirmation",
        )
        confirmation = _write_compact(
            results_dir / "03_dose_confirmation.json",
            experiment="dose_confirmation",
            config=config,
            results={
                "selected_pair": dict(selected_pair),
                "baseline": {
                    "harmful_refusal_rate": float(harmful_baseline["refusal_rate"]),
                    "harmless_refusal_rate": float(harmless_baseline["refusal_rate"]),
                    "harmful": harmful_baseline,
                    "harmless": harmless_baseline,
                },
                "ablation": {
                    "harmful_refusal_rate": float(ablation_harmful["refusal_rate"]),
                    "harmless_kl": float(kl_payload["harmless_kl"]),
                    "harmful": ablation_harmful,
                },
                "addition": {
                    "harmless_refusal_rate": float(addition_harmless["refusal_rate"]),
                    "harmless": addition_harmless,
                },
                "random_control": {
                    "harmful_refusal_rate": float(random_control["refusal_rate"]),
                    "coefficient": ablation_coefficient,
                    "harmful": random_control,
                },
                "example_ids": {
                    "harmful_dose_confirmation": [item.id for item in harmful],
                    "harmless_dose_confirmation": [item.id for item in harmless],
                },
            },
            provenance=provenance,
        )
        return _evaluation_from_confirmation(
            confirmation,
            config=config,
            run_id=run_id,
            implementation_commit=implementation_commit,
            source=source,
            provenance=provenance,
            results_dir=results_dir,
            store=store,
        )
    finally:
        del model
        del tokenizer
        del bound


def _existing_terminal_evaluation(
    results_dir: Path,
    *,
    run_id: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    path = results_dir / "dose_followup_evaluation.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("run_id") != run_id:
        raise ValueError("stale run_id in dose_followup_evaluation.json")
    stored_hashes = payload.get("source_artifact_hashes")
    if stored_hashes != dict(source_hashes):
        raise ValueError("stale source hash in dose_followup_evaluation.json")
    if payload.get("status") in TERMINAL_SCIENCE:
        return payload
    return None


def run_qwen_dose(
    root: Path,
    *,
    run_id: str | None = None,
    profile: str = DOSE_PROFILE,
    measure: MeasureFn | None = None,
    release_resources: Callable[[], None] | None = None,
    stage_runners: dict[str, StageRunner] | None = None,
    load_model=None,
) -> dict[str, Any]:
    """Run or resume the isolated Qwen dose follow-up."""

    root = root.resolve()
    config = load_config(root, profile=profile)
    results_dir, artifacts_dir = project_paths(root, config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    source = _source_context(root, config)
    resolved_run_id = resolve_run_id(artifacts_dir, requested=run_id)
    if resolved_run_id == source["source_pilot_run_id"]:
        raise ValueError("dose follow-up run ID must differ from the source pilot")
    store = RunProgressStore(
        artifacts_dir,
        profile=profile,
        run_id=resolved_run_id,
    )
    existing = _existing_terminal_evaluation(
        results_dir,
        run_id=resolved_run_id,
        source_hashes=source["source_hashes"],
    )
    if existing is not None:
        return existing
    current = store.read_status()
    if (
        current is not None
        and current.get("run_id") == resolved_run_id
        and current.get("state") == "aborted"
    ):
        fallback = results_dir / "dose_followup_evaluation.json"
        if fallback.is_file():
            return _read_json(fallback)

    commit = implementation_commit_hash()
    resource_holder: dict[str, Any] = {"model": None, "tokenizer": None}

    def release_loaded() -> None:
        if release_resources is not None:
            release_resources()
        else:
            release_compute_resources(
                resource_holder.get("model"),
                resource_holder.get("tokenizer"),
            )
        resource_holder["model"] = None
        resource_holder["tokenizer"] = None

    last: dict[str, Any] | None = None
    grid = list(config["dose_grid"])

    def default_prepare(root_path: Path, **_kwargs) -> dict[str, Any]:
        return run_dose_prepare(
            root_path,
            profile=profile,
            run_id=resolved_run_id,
            implementation_commit=commit,
        )

    def default_sweep(root_path: Path, **_kwargs) -> dict[str, Any]:
        return run_dose_sweep(
            root_path,
            profile=profile,
            run_id=resolved_run_id,
            implementation_commit=commit,
            store=store,
            measure=measure,
            load_model=load_model,
            resource_holder=resource_holder,
        )

    def default_confirm(root_path: Path, **_kwargs) -> dict[str, Any]:
        return run_dose_confirm(
            root_path,
            profile=profile,
            run_id=resolved_run_id,
            implementation_commit=commit,
            store=store,
            measure=measure,
            load_model=load_model,
            resource_holder=resource_holder,
        )

    runners = stage_runners or {
        "prepare": default_prepare,
        "sweep": default_sweep,
        "confirm": default_confirm,
    }

    for stage in DOSE_STAGES:
        if stage == "prepare":
            existing = load_reusable_artifact(
                results_dir / "00_dose_prepare.json",
                run_id=resolved_run_id,
                source_hashes=source["source_hashes"],
                implementation_commit=commit,
                dose_grid=grid,
            )
            if existing is not None:
                last = existing
                continue
        elif stage == "sweep":
            existing = load_reusable_artifact(
                results_dir / "02_dose_selection.json",
                run_id=resolved_run_id,
                source_hashes=source["source_hashes"],
                implementation_commit=commit,
                dose_grid=grid,
            )
            if existing is not None:
                last = existing
                continue
        try:
            total_units = len(grid) if stage == "sweep" else 1
            store.initialize(stage=stage, total_units=total_units)
            store.begin_running()
            last = runners[stage](root, profile=profile, run_id=resolved_run_id)
            status = last.get("status") if isinstance(last, dict) else None
            if status == STATUS_NO_DOSE_QUALIFIED:
                return last
            if status == STATUS_CAUSAL_GATE_FAILED:
                return last
            store.complete_stage()
            if status == STATUS_PASS:
                return last
        except Exception as exc:
            category, error_code, message = classify_operational_error(exc)
            evaluation = build_dose_evaluation(
                status=STATUS_OPERATIONAL_ERROR,
                operational_success=False,
                run_id=resolved_run_id,
                implementation_commit=commit,
                source_pilot_run_id=source["source_pilot_run_id"],
                source_pilot_implementation_commit=source[
                    "source_pilot_implementation_commit"
                ],
                source_artifact_hashes=source["source_hashes"],
                model_revision=str(config["model_revision"]),
                stage=stage,
                exception_category=category,
                error_code=error_code,
                message=message,
            )
            try:
                _write_evaluation(results_dir, evaluation)
            except Exception:
                pass
            try:
                store.fail(message)
            except Exception:
                pass
            raise
        finally:
            release_loaded()
    if last is None:
        fallback = results_dir / "dose_followup_evaluation.json"
        if fallback.is_file():
            return _read_json(fallback)
        raise ValueError("dose follow-up produced no evaluation")
    return last
