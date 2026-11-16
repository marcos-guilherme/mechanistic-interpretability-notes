from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from refusal_direction_replication.data import (
    PromptExample,
    prompt_content_hash,
)
from refusal_direction_replication.interventions import (
    ablate_direction,
    add_direction,
    directional_ablation_hook_specs,
)
from refusal_direction_replication.pipeline import _sampled_splits, load_config
from refusal_direction_replication.qwen_dose import (
    ADDITION_METRIC_FIELDS,
    ABLATION_METRIC_FIELDS,
    DoseSplits,
    _assert_disjoint_splits,
    build_dose_splits,
    collect_original_usage,
    compact_split_metadata,
    eligible_examples,
    load_source_envelopes,
    metadata_contains_leak,
    sample_dose_halves,
    select_dose_coefficients,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_QWEN_PROFILE = "qwen25_coder_7b.json"
DOSE_PROFILE = "qwen25_coder_7b_dose.json"
ORIGINAL_RESULTS = ROOT / "results" / "qwen25_coder_7b"


def test_dose_profile_matches_original_model_fields_and_isolates_paths() -> None:
    original = load_config(ROOT, profile=ORIGINAL_QWEN_PROFILE)
    dose = load_config(ROOT, profile=DOSE_PROFILE)
    for field in (
        "model_name",
        "model_revision",
        "dtype",
        "upstream_commit",
        "refusal_prefixes",
        "batch_size",
    ):
        assert dose[field] == original[field]
    assert dose["results_dir"] == "results/qwen25_coder_7b_dose"
    assert dose["artifacts_dir"] == "artifacts/qwen25_coder_7b_dose"
    assert dose["results_dir"] != original["results_dir"]
    assert dose["artifacts_dir"] != original["artifacts_dir"]
    assert dose["source_direction"] == (
        "artifacts/qwen25_coder_7b/selected_direction.pt"
    )
    assert dose["source_evidence"] == {
        "baseline": "results/qwen25_coder_7b/01_baseline.json",
        "interventions": "results/qwen25_coder_7b/03_interventions.json",
    }
    assert dose["dose_grid"] == [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    assert 0.0 not in dose["dose_grid"]
    assert dose["dose_selection_per_class"] == 32
    assert dose["dose_confirmation_per_class"] == 32
    assert dose["causal_gate"] == original["causal_gate"]
    assert dose["causal_gate"] == {
        "min_harmful_ablation_drop": 0.2,
        "min_harmless_addition_increase": 0.2,
        "max_random_control_harmful_drop": 0.1,
        "max_harmless_ablation_kl": 0.1,
    }


def test_gemma_and_original_qwen_keep_full_intervention_coefficient() -> None:
    gemma = load_config(ROOT)
    qwen = load_config(ROOT, profile=ORIGINAL_QWEN_PROFILE)
    assert "dose_grid" not in gemma
    assert "dose_grid" not in qwen
    assert inspect.signature(ablate_direction).parameters["coefficient"].default == 1.0
    assert inspect.signature(add_direction).parameters["coefficient"].default == 1.0
    assert (
        inspect.signature(directional_ablation_hook_specs)
        .parameters["coefficient"]
        .default
        == 1.0
    )
    pipeline_source = (
        ROOT / "src" / "refusal_direction_replication" / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert pipeline_source.count("coefficient=1.0") >= 3
    assert "coefficient=0." not in pipeline_source


def test_dose_profile_points_at_unmodified_original_evidence() -> None:
    dose = load_config(ROOT, profile=DOSE_PROFILE)
    baseline = ROOT / dose["source_evidence"]["baseline"]
    interventions = ROOT / dose["source_evidence"]["interventions"]
    assert baseline == ORIGINAL_RESULTS / "01_baseline.json"
    assert interventions == ORIGINAL_RESULTS / "03_interventions.json"
    assert json.loads(baseline.read_text(encoding="utf-8"))["experiment"] == "baseline"
    assert json.loads(interventions.read_text(encoding="utf-8"))["experiment"] == (
        "causal_interventions"
    )


def _example(example_id: str, text: str, label: str = "harmful") -> PromptExample:
    return PromptExample(
        id=example_id,
        text=text,
        label=label,  # type: ignore[arg-type]
        source="fixture",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_corpus(path: Path, examples: list[PromptExample]) -> None:
    rows = [
        {
            "id": example.id,
            "text": example.text,
            "label": example.label,
            "source": example.source,
        }
        for example in examples
    ]
    _write_json(path, rows)


def _original_artifact(
    *,
    experiment: str,
    example_ids: dict[str, list[str]],
    content_hashes: dict[str, list[str]],
    source_revision: str = "upstream-rev",
) -> dict:
    return {
        "experiment": experiment,
        "provenance": {
            "implementation_commit": "impl-rev",
            "upstream_commit": source_revision,
        },
        "results": {
            "example_ids": example_ids,
            "content_hashes": content_hashes,
        },
    }


REQUIRED_SPLIT_KEYS = (
    "harmful_train",
    "harmless_train",
    "harmful_val",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)


def _complete_usage(
    extra_ids: dict[str, list[str]] | None = None,
    extra_hashes: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    example_ids = {
        key: [f"{key}-0001"]
        for key in REQUIRED_SPLIT_KEYS
    }
    content_hashes = {
        key: [prompt_content_hash(f"{key} prompt")]
        for key in REQUIRED_SPLIT_KEYS
    }
    if extra_ids:
        for key, values in extra_ids.items():
            example_ids.setdefault(key, []).extend(values)
    if extra_hashes:
        for key, values in extra_hashes.items():
            content_hashes.setdefault(key, []).extend(values)
    return example_ids, content_hashes


def test_collect_original_usage_unions_ids_and_hashes(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    used_text = "already used prompt"
    baseline_ids, baseline_hashes = _complete_usage()
    baseline_ids["harmless_val"].append("harmless_val-0002")
    baseline_hashes["harmless_val"].append(prompt_content_hash("val prompt"))
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=baseline_ids,
            content_hashes=baseline_hashes,
        ),
    )
    _write_json(
        results_dir / "03_interventions.json",
        _original_artifact(
            experiment="causal_interventions",
            example_ids={"harmful_test": ["harmful_test-0003"]},
            content_hashes={"harmful_test": [prompt_content_hash(used_text)]},
        ),
    )
    used_ids, used_hashes = collect_original_usage(results_dir)
    assert "harmful_train-0001" in used_ids
    assert "harmless_val-0002" in used_ids
    assert "harmful_test-0003" in used_ids
    assert prompt_content_hash("harmful_train prompt") in used_hashes
    assert prompt_content_hash("val prompt") in used_hashes
    assert prompt_content_hash(used_text) in used_hashes


def test_collect_original_usage_fails_on_preflight_only_directory(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "00_preflight.json",
        {
            "experiment": "preflight",
            "provenance": {"upstream_commit": "upstream-rev"},
            "results": {},
        },
    )
    with pytest.raises(ValueError, match="required split"):
        collect_original_usage(results_dir)


def test_collect_original_usage_requires_all_six_split_keys(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    example_ids, content_hashes = _complete_usage()
    example_ids.pop("harmful_val")
    content_hashes.pop("harmful_val")
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=example_ids,
            content_hashes=content_hashes,
        ),
    )
    with pytest.raises(ValueError, match="harmful_val"):
        collect_original_usage(results_dir)


def test_collect_original_usage_rejects_empty_or_non_list_groups(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    example_ids, content_hashes = _complete_usage()
    example_ids["harmful_train"] = []
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=example_ids,
            content_hashes=content_hashes,
        ),
    )
    with pytest.raises(ValueError, match="harmful_train"):
        collect_original_usage(results_dir)

    example_ids, content_hashes = _complete_usage()
    example_ids["harmless_test"] = "harmless_test-0001"
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=example_ids,
            content_hashes=content_hashes,
        ),
    )
    with pytest.raises(ValueError, match="list"):
        collect_original_usage(results_dir)


def test_eligible_examples_exclude_original_id_or_content_hash() -> None:
    used_ids = {"harmful_test-0001"}
    overlap_text = "same wording as a train row"
    used_hashes = {prompt_content_hash(overlap_text)}
    rows = [
        _example("harmful_test-0001", "fresh wording, reused id"),
        _example("harmful_test-0099", overlap_text),
        _example("harmful_test-0100", "truly leftover"),
    ]
    leftover = eligible_examples(
        rows,
        used_ids=used_ids,
        used_hashes=used_hashes,
    )
    assert [item.id for item in leftover] == ["harmful_test-0100"]


def test_dose_sampling_is_deterministic_and_splits_selection_from_confirmation() -> None:
    rows = [_example(f"harmful_test-{index:04d}", f"prompt {index}") for index in range(8)]
    first_selection, first_confirmation = sample_dose_halves(
        rows,
        seed=7,
        selection_size=3,
        confirmation_size=3,
    )
    second_selection, second_confirmation = sample_dose_halves(
        rows,
        seed=7,
        selection_size=3,
        confirmation_size=3,
    )
    other_selection, _other_confirmation = sample_dose_halves(
        rows,
        seed=8,
        selection_size=3,
        confirmation_size=3,
    )
    assert first_selection == second_selection
    assert first_confirmation == second_confirmation
    assert len(first_selection) == 3
    assert len(first_confirmation) == 3
    assert {item.id for item in first_selection}.isdisjoint(
        {item.id for item in first_confirmation}
    )
    assert [item.id for item in first_selection] != [
        item.id for item in other_selection
    ]


def test_sampling_fails_when_fewer_than_required_remain() -> None:
    rows = [_example(f"harmful_test-{index:04d}", f"prompt {index}") for index in range(5)]
    with pytest.raises(ValueError, match="need 64"):
        sample_dose_halves(rows, seed=1, selection_size=32, confirmation_size=32)


def _tiny_splits(
    *,
    harmful_selection: list[PromptExample],
    harmless_selection: list[PromptExample] | None = None,
    harmful_confirmation: list[PromptExample] | None = None,
    harmless_confirmation: list[PromptExample] | None = None,
) -> DoseSplits:
    return DoseSplits(
        harmful_dose_selection=tuple(harmful_selection),
        harmless_dose_selection=tuple(
            harmless_selection
            or [_example("harmless_test-sel", "harmless selection", "harmless")]
        ),
        harmful_dose_confirmation=tuple(
            harmful_confirmation
            or [_example("harmful_test-conf", "harmful confirmation")]
        ),
        harmless_dose_confirmation=tuple(
            harmless_confirmation
            or [_example("harmless_test-conf", "harmless confirmation", "harmless")]
        ),
    )


def test_validate_rejects_original_id_in_a_new_half() -> None:
    splits = _tiny_splits(
        harmful_selection=[_example("harmful_train-0001", "fresh leftover wording")],
    )
    with pytest.raises(ValueError, match="original example ids"):
        _assert_disjoint_splits(
            splits,
            used_ids={"harmful_train-0001"},
            used_hashes=set(),
            selection_size=1,
            confirmation_size=1,
        )


def test_validate_rejects_hash_overlap_between_selection_and_confirmation() -> None:
    shared = "same leftover wording for both halves"
    splits = _tiny_splits(
        harmful_selection=[_example("harmful_test-0100", shared)],
        harmful_confirmation=[_example("harmful_test-0101", shared)],
    )
    with pytest.raises(ValueError, match="hash"):
        _assert_disjoint_splits(
            splits,
            used_ids=set(),
            used_hashes=set(),
            selection_size=1,
            confirmation_size=1,
        )


def test_validate_rejects_pairwise_hash_overlap_across_selection_halves() -> None:
    shared = "same leftover wording across classes"
    splits = _tiny_splits(
        harmful_selection=[_example("harmful_test-0100", shared)],
        harmless_selection=[
            _example("harmless_test-0100", shared, "harmless")
        ],
    )
    with pytest.raises(ValueError, match="hash"):
        _assert_disjoint_splits(
            splits,
            used_ids=set(),
            used_hashes=set(),
            selection_size=1,
            confirmation_size=1,
        )


def test_validate_rejects_duplicate_text_within_the_same_half() -> None:
    splits = _tiny_splits(
        harmful_selection=[
            _example("harmful_test-0100", "duplicated leftover wording"),
            _example("harmful_test-0101", "duplicated leftover wording"),
        ],
        harmless_selection=[
            _example("harmless_test-sel-a", "harmless selection a", "harmless"),
            _example("harmless_test-sel-b", "harmless selection b", "harmless"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        _assert_disjoint_splits(
            splits,
            used_ids=set(),
            used_hashes=set(),
            selection_size=2,
            confirmation_size=1,
        )


def test_build_dose_splits_are_disjoint_and_omit_prompt_text(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    original_text = "original test prompt"
    overlap_text = "hash overlap with train"
    original_test = _example("harmful_test-0001", original_text)
    original_train_hash = _example("harmful_test-0002", overlap_text)
    leftovers = [
        _example(f"harmful_test-{index:04d}", f"leftover harmful {index}")
        for index in range(10, 80)
    ]
    harmless_leftovers = [
        _example(
            f"harmless_test-{index:04d}",
            f"leftover harmless {index}",
            label="harmless",
        )
        for index in range(10, 80)
    ]
    example_ids, content_hashes = _complete_usage(
        extra_ids={
            "harmful_test": [original_test.id],
            "harmless_test": ["harmless_test-0001"],
        },
        extra_hashes={
            "harmful_train": [prompt_content_hash(overlap_text)],
            "harmful_test": [prompt_content_hash(original_text)],
            "harmless_test": [prompt_content_hash("original harmless")],
        },
    )
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=example_ids,
            content_hashes=content_hashes,
            source_revision="9d852fae1a9121c78b29142de733cb1340770cc3",
        ),
    )
    harmful_path = tmp_path / "harmful_test.json"
    harmless_path = tmp_path / "harmless_test.json"
    _write_corpus(harmful_path, [original_test, original_train_hash, *leftovers])
    _write_corpus(
        harmless_path,
        [
            _example("harmless_test-0001", "original harmless", label="harmless"),
            *harmless_leftovers,
        ],
    )
    splits, metadata = build_dose_splits(
        original_results_dir=results_dir,
        harmful_test_path=harmful_path,
        harmless_test_path=harmless_path,
        seed=2026,
        selection_size=32,
        confirmation_size=32,
    )
    original_ids = {
        "harmful_train-0001",
        original_test.id,
        "harmless_test-0001",
    }
    original_hashes = {
        prompt_content_hash(overlap_text),
        prompt_content_hash(original_text),
        prompt_content_hash("original harmless"),
    }
    new_ids = {
        item.id
        for item in (
            *splits.harmful_dose_selection,
            *splits.harmless_dose_selection,
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    new_hashes = {
        prompt_content_hash(item.text)
        for item in (
            *splits.harmful_dose_selection,
            *splits.harmless_dose_selection,
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    assert original_ids.isdisjoint(new_ids)
    assert original_hashes.isdisjoint(new_hashes)
    selection_ids = {
        item.id
        for item in (*splits.harmful_dose_selection, *splits.harmless_dose_selection)
    }
    confirmation_ids = {
        item.id
        for item in (
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    selection_hashes = {
        prompt_content_hash(item.text)
        for item in (*splits.harmful_dose_selection, *splits.harmless_dose_selection)
    }
    confirmation_hashes = {
        prompt_content_hash(item.text)
        for item in (
            *splits.harmful_dose_confirmation,
            *splits.harmless_dose_confirmation,
        )
    }
    assert selection_ids.isdisjoint(confirmation_ids)
    assert selection_hashes.isdisjoint(confirmation_hashes)
    assert len(splits.harmful_dose_selection) == 32
    assert len(splits.harmless_dose_selection) == 32
    assert len(splits.harmful_dose_confirmation) == 32
    assert len(splits.harmless_dose_confirmation) == 32
    assert set(metadata) == {
        "example_ids",
        "content_hashes",
        "counts",
        "seed",
        "source_revision",
    }
    assert metadata["seed"] == {"harmful": 2026, "harmless": 2027}
    assert metadata["source_revision"] == (
        "9d852fae1a9121c78b29142de733cb1340770cc3"
    )
    assert metadata_contains_leak(metadata) is False


def test_build_dose_splits_fails_when_a_class_lacks_64_leftovers(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    example_ids, content_hashes = _complete_usage()
    _write_json(
        results_dir / "01_baseline.json",
        _original_artifact(
            experiment="baseline",
            example_ids=example_ids,
            content_hashes=content_hashes,
        ),
    )
    harmful_rows = [
        _example(f"harmful_test-{index:04d}", f"h {index}") for index in range(10, 80)
    ]
    harmless_rows = [
        _example(f"harmless_test-{index:04d}", f"s {index}", label="harmless")
        for index in range(3)
    ]
    harmful_path = tmp_path / "harmful_test.json"
    harmless_path = tmp_path / "harmless_test.json"
    _write_corpus(harmful_path, harmful_rows)
    _write_corpus(harmless_path, harmless_rows)
    with pytest.raises(ValueError, match="harmless"):
        build_dose_splits(
            original_results_dir=results_dir,
            harmful_test_path=harmful_path,
            harmless_test_path=harmless_path,
            seed=2026,
        )


def test_compact_metadata_helper_omits_prompts() -> None:
    selection = [_example("harmful_test-0010", "secret prompt text")]
    confirmation = [_example("harmful_test-0011", "another secret")]
    metadata = compact_split_metadata(
        harmful_selection=selection,
        harmless_selection=selection,
        harmful_confirmation=confirmation,
        harmless_confirmation=confirmation,
        seed=9,
        source_revision="rev",
    )
    assert metadata["counts"] == {
        "harmful_dose_selection": 1,
        "harmless_dose_selection": 1,
        "harmful_dose_confirmation": 1,
        "harmless_dose_confirmation": 1,
    }
    dumped = json.dumps(metadata)
    assert "secret prompt text" not in dumped
    assert "another secret" not in dumped
    assert metadata["seed"] == {"harmful": 9, "harmless": 10}
    assert metadata_contains_leak(metadata) is False


def test_metadata_leak_check_walks_nested_keys() -> None:
    nested = {
        "outer": {
            "inner": {
                "text": "a nested leaked prompt string here extra",
            }
        }
    }
    assert metadata_contains_leak(nested) is True
    assert metadata_contains_leak({"example_ids": ["harmful_test-0010"]}) is False


def test_metadata_allowlists_aggregated_gate_reasons_without_word_count() -> None:
    concatenated = (
        "min_harmful_ablation_drop: observed 0.01 not >= 0.2; "
        "max_random_control_harmful_drop: observed 0.50 not <= 0.1; "
        "max_harmless_ablation_kl: observed 0.50 not <= 0.1"
    )
    assert len(concatenated.split()) >= 8
    payload = {
        "status": "causal_gate_failed",
        "causal_gate": {
            "name": "causal",
            "passed": False,
            "reason": concatenated,
            "checks": [
                {"name": "min_harmful_ablation_drop", "reason": concatenated[:40]},
            ],
        },
        "abort_reason": concatenated,
    }
    assert metadata_contains_leak(payload) is False
    assert metadata_contains_leak({"prompt": "short"}) is True
    assert metadata_contains_leak({"generation": "also short"}) is True
    assert metadata_contains_leak(
        {"notes": "one two three four five six seven eight nine"}
    ) is True


def test_selects_smallest_qualifying_ablation_and_addition() -> None:
    chosen = select_dose_coefficients(
        ablation_curve=[
            {
                "coefficient": 0.2,
                "harmful_refusal_drop": 0.10,
                "harmless_kl": 0.04,
            },
            {
                "coefficient": 0.35,
                "harmful_refusal_drop": 0.20,
                "harmless_kl": 0.10,
            },
            {
                "coefficient": 0.5,
                "harmful_refusal_drop": 0.40,
                "harmless_kl": 0.09,
            },
        ],
        addition_curve=[
            {"coefficient": 0.1, "harmless_refusal_increase": 0.10},
            {"coefficient": 0.2, "harmless_refusal_increase": 0.20},
            {"coefficient": 0.35, "harmless_refusal_increase": 0.50},
        ],
    )
    assert chosen["ablation_coefficient"] == 0.35
    assert chosen["addition_coefficient"] == 0.2
    assert chosen["selected_pair"] == {"ablation": 0.35, "addition": 0.2}


def test_returns_no_selected_pair_when_either_side_fails() -> None:
    chosen = select_dose_coefficients(
        ablation_curve=[
            {
                "coefficient": 1.0,
                "harmful_refusal_drop": 0.50,
                "harmless_kl": 0.11,
            }
        ],
        addition_curve=[
            {"coefficient": 1.0, "harmless_refusal_increase": 0.50},
        ],
    )
    assert chosen["ablation_coefficient"] is None
    assert chosen["addition_coefficient"] == 1.0
    assert chosen["selected_pair"] is None


def test_dose_selection_boundaries_are_inclusive() -> None:
    chosen = select_dose_coefficients(
        ablation_curve=[
            {
                "coefficient": 0.1,
                "harmful_refusal_drop": 0.199,
                "harmless_kl": 0.10,
            },
            {
                "coefficient": 0.2,
                "harmful_refusal_drop": 0.20,
                "harmless_kl": 0.101,
            },
            {
                "coefficient": 0.35,
                "harmful_refusal_drop": 0.20,
                "harmless_kl": 0.10,
            },
        ],
        addition_curve=[
            {"coefficient": 0.1, "harmless_refusal_increase": 0.199},
            {"coefficient": 0.2, "harmless_refusal_increase": 0.20},
        ],
    )
    assert chosen["ablation_coefficient"] == 0.35
    assert chosen["addition_coefficient"] == 0.2


def test_selection_does_not_read_confirmation_metrics() -> None:
    parameters = inspect.signature(select_dose_coefficients).parameters
    assert "confirmation" not in parameters
    assert "confirmation_curve" not in parameters
    chosen = select_dose_coefficients(
        ablation_curve=[
            {
                "coefficient": 0.75,
                "harmful_refusal_drop": 0.20,
                "harmless_kl": 0.10,
                "confirmation_harmful_refusal_drop": 0.0,
                "confirmation_harmless_kl": 0.9,
            }
        ],
        addition_curve=[
            {
                "coefficient": 0.75,
                "harmless_refusal_increase": 0.20,
                "confirmation_harmless_refusal_increase": 0.0,
            },
        ],
    )
    assert chosen["ablation_coefficient"] == 0.75
    assert chosen["addition_coefficient"] == 0.75
    dumped = json.dumps(chosen)
    assert "confirmation" not in dumped
    assert set(chosen["ablation_curve"][0]) <= set(ABLATION_METRIC_FIELDS)
    assert set(chosen["addition_curve"][0]) <= set(ADDITION_METRIC_FIELDS)


def test_selection_keeps_only_allowlisted_metric_fields() -> None:
    chosen = select_dose_coefficients(
        ablation_curve=[
            {
                "coefficient": 1.0,
                "harmful_refusal_drop": 0.5,
                "harmless_kl": 0.09,
                "prompt": "please leak this prompt text now extra",
                "notes": "drop extra keys",
            }
        ],
        addition_curve=[
            {
                "coefficient": 1.0,
                "harmless_refusal_increase": 0.4,
                "generation": "I cannot help with that request extra",
            }
        ],
    )
    assert chosen["ablation_curve"] == [
        {
            "coefficient": 1.0,
            "harmful_refusal_drop": 0.5,
            "harmless_kl": 0.09,
        }
    ]
    assert chosen["addition_curve"] == [
        {
            "coefficient": 1.0,
            "harmless_refusal_increase": 0.4,
        }
    ]


def test_selection_preserves_full_measured_curves() -> None:
    ablation_curve = [
        {"coefficient": 0.1, "harmful_refusal_drop": 0.0, "harmless_kl": 0.01},
        {"coefficient": 1.0, "harmful_refusal_drop": 0.5, "harmless_kl": 0.09},
    ]
    addition_curve = [
        {"coefficient": 0.1, "harmless_refusal_increase": 0.0},
        {"coefficient": 1.0, "harmless_refusal_increase": 0.4},
    ]
    chosen = select_dose_coefficients(
        ablation_curve=ablation_curve,
        addition_curve=addition_curve,
    )
    assert chosen["ablation_curve"] == ablation_curve
    assert chosen["addition_curve"] == addition_curve
    assert chosen["selected_pair"] == {"ablation": 1.0, "addition": 1.0}


def test_dose_profile_records_the_new_split_seed() -> None:
    dose = load_config(ROOT, profile=DOSE_PROFILE)
    assert dose["dose_split_seed"] == 2026
    assert dose["dose_split_seed"] != dose["seed"]


def test_sampled_splits_rejects_dose_profile_and_points_to_builder(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT, profile=DOSE_PROFILE)
    with pytest.raises(ValueError, match="build_dose_splits"):
        _sampled_splits(tmp_path, config)


def test_sampled_splits_rejects_source_direction_even_without_dose_grid(
    tmp_path: Path,
) -> None:
    config = dict(load_config(ROOT, profile=DOSE_PROFILE))
    config.pop("dose_grid")
    assert "source_direction" in config
    with pytest.raises(ValueError, match="build_dose_splits"):
        _sampled_splits(tmp_path, config)


def test_real_pilot_evaluation_is_validated_without_model_or_revision() -> None:
    evaluation = json.loads(
        (ORIGINAL_RESULTS / "pilot_evaluation.json").read_text(encoding="utf-8")
    )
    assert "model_name" not in evaluation
    assert "model_revision" not in evaluation
    envelopes = load_source_envelopes(ROOT)
    assert envelopes["pilot_evaluation.json"]["run_id"] == evaluation["run_id"]
    assert (
        envelopes["pilot_evaluation.json"]["implementation_commit"]
        == evaluation["implementation_commit"]
    )
    identity = envelopes["00_preflight.json"]["provenance"]
    assert identity["model_name"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert identity["requested_model_revision"] == (
        "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
