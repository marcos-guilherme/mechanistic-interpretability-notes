from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from jspace_llama.jsonio import document_sha256
from jspace_llama.observation_data import DATASET_SHA256, PANEL_NAMES
from jspace_llama.runtime import AssetLoadError


PROJECT = Path(__file__).parents[1]
SCHEMAS = PROJECT / "schemas"
VALIDATOR = PROJECT / "scripts" / "validate_observation.py"


def _provenance() -> dict[str, object]:
    return {
        "source": {
            "commit": "1" * 40,
            "dirty": False,
            "scoped_dirty": False,
            "archive_sha256": "2" * 64,
        },
        "model": {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "resolved_revision": "3" * 40,
            "layer_count": 32,
        },
        "lens": {
            "repo_id": "neuronpedia/jacobian-lens",
            "filename": (
                "llama3.1-8b-it/jlens/Salesforce-wikitext/"
                "Llama-3.1-8B-Instruct_jacobian_lens.pt"
            ),
            "resolved_revision": "4" * 40,
            "sha256": "5" * 64,
            "size_bytes": 123,
            "integrity_verified": True,
        },
        "software": {
            "python": "3.11",
            "pytorch": "2.8",
            "transformers": "4.56",
            "huggingface_hub": "0.34",
            "jspace": "0.1",
            "jspace_commit": "04c822a255fe0a10d84e08ec499e5e09f4b7d37a",
        },
        "hardware": {
            "gpu_name": "NVIDIA L4",
            "cuda_version": "12.6",
            "dtype": "bfloat16",
        },
    }


def _rank_detail(rank: int, layers: list[int]) -> dict[str, object]:
    return {
        "ranks": [{"layer": layer, "rank": rank} for layer in layers],
        "best_rank": rank,
        "best_layer": 12,
        "all_layers_best_rank": rank,
        "all_layers_best_layer": layers[0],
    }


def complete_payload(*, all_pass_at_k_zero: bool = False) -> dict[str, object]:
    layers = list(range(17))
    names = list(PANEL_NAMES) + [f"ordinary-{index}" for index in range(81)]
    rank = 50 if all_pass_at_k_zero else 1
    records = []
    audit = []
    for index, name in enumerate(names):
        intermediate = {
            "text": f"intermediate-{index}",
            "token_id": index,
            "single_token": True,
            "jlens": _rank_detail(rank, layers),
            "logit_lens": _rank_detail(rank, layers),
        }
        record: dict[str, object] = {
            "name": name,
            "prompt": f"prompt {name} ",
            "target": f"target-{name}",
            "intermediates": [intermediate],
        }
        if name in PANEL_NAMES:
            record["top_tokens"] = {
                method: [
                    {
                        "layer": layer,
                        "tokens": [
                            {
                                "token_id": index,
                                "token": f"{method}-{index}",
                                "rank": 1,
                            }
                        ],
                    }
                    for layer in (12, 13)
                ]
                for method in ("jlens", "logit_lens")
            }
        records.append(record)
        token_ids = [index]
        audit.append(
            {
                "name": name,
                "intermediates": [
                    {
                        "text": f"intermediate-{index}",
                        "token_id": index,
                        "token_ids": token_ids,
                        "variant": "leading_space",
                        "single_token": True,
                        "prompt_sha256": hashlib.sha256(
                            f" intermediate-{index}".encode()
                        ).hexdigest(),
                        "token_ids_sha256": hashlib.sha256(
                            f"[{index}]".encode()
                        ).hexdigest(),
                    }
                ],
            }
        )
    pass_value = 0.0 if all_pass_at_k_zero else 1.0
    metrics = {
        method: {
            scope: {
                "n_items": 93,
                "n_intermediates": 93,
                "pass@1": pass_value,
                "pass@5": pass_value,
                "pass@10": pass_value,
            }
            for scope in ("band", "all_layers")
        }
        for method in ("jlens", "logit_lens")
    }
    panel = []
    by_name = {record["name"]: record for record in records}
    for name in PANEL_NAMES:
        source = by_name[name]
        panel.append(
            {
                "name": name,
                "prompt": source["prompt"],
                "target": source["target"],
                "intermediates": copy.deepcopy(source["intermediates"]),
                "top_tokens": copy.deepcopy(source["top_tokens"]),
                "pursuit": [
                    {
                        "layer": layer,
                        "tokens": [
                            {
                                "token_id": 7,
                                "token": "pursuit",
                                "rank": 1,
                                "coefficient": 0.5,
                            }
                        ],
                        "variance_explained": 0.5,
                    }
                    for layer in (12, 13)
                ],
            }
        )
    return {
        "provenance": _provenance(),
        "tokenization_audit": audit,
        "direct": {
            "items": records,
            "metrics": metrics,
            "single_token_metrics": copy.deepcopy(metrics),
        },
        "panel": {"items": panel},
        "layers": {"band": [12, 13], "all_layers": layers},
    }


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_observation", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_success_depends_on_completeness_not_hits(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    outcome = run_observation_pipeline(
        project_root=tmp_path,
        execute=lambda: complete_payload(all_pass_at_k_zero=True),
    )

    assert outcome["status"] == "success"
    assert outcome["metrics"]["jlens"]["band"]["pass@1"] == 0.0
    assert len(outcome["per_item_best_ranks"]) == 93
    assert outcome["panel_names"] == list(PANEL_NAMES)


def test_observation_pipeline_preserves_both_causal_results(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    causal = {
        tmp_path / "results" / "final_evaluation.json": b'{"preserve":true}',
        tmp_path / "results" / "execution_manifest.json": b"causal manifest\n",
    }
    for path, content in causal.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    run_observation_pipeline(
        project_root=tmp_path,
        execute=complete_payload,
    )

    assert {path: path.read_bytes() for path in causal} == causal


def test_observation_hashes_raw_and_validates_strict_schemas(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    outcome = run_observation_pipeline(
        project_root=tmp_path,
        execute=complete_payload,
    )
    manifest = json.loads(
        (tmp_path / "results" / "jspace_observation_manifest.json").read_text()
    )
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"

    assert manifest["raw_artifact_sha256"] == hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    assert outcome["manifest_sha256"] == document_sha256(manifest)
    jsonschema.validate(
        manifest,
        json.loads(
            (SCHEMAS / "jspace_observation_manifest.schema.json").read_text()
        ),
    )
    jsonschema.validate(
        outcome,
        json.loads((SCHEMAS / "jspace_observation.schema.json").read_text()),
    )
    _load_validator().validate_documents(
        manifest,
        outcome,
        schema_dir=SCHEMAS,
        raw_path=raw_path,
    )


def test_incomplete_run_is_inconclusive_and_validator_recomputes_claims(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    payload = complete_payload()
    payload["direct"]["items"].pop()
    outcome = run_observation_pipeline(
        project_root=tmp_path,
        execute=lambda: payload,
    )
    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "incomplete_observation"

    complete = run_observation_pipeline(
        project_root=tmp_path / "complete",
        execute=complete_payload,
    )
    manifest = json.loads(
        (
            tmp_path
            / "complete"
            / "results"
            / "jspace_observation_manifest.json"
        ).read_text()
    )
    tampered = copy.deepcopy(complete)
    tampered["metrics"]["jlens"]["band"]["pass@1"] = 0.25
    validator = _load_validator()
    with pytest.raises(
        validator.ObservationValidationError,
        match="pass@k",
    ):
        validator.validate_documents(
            manifest,
            tampered,
            schema_dir=SCHEMAS,
            raw_path=(
                tmp_path
                / "complete"
                / "artifacts"
                / "jspace_observation_raw.json"
            ),
        )


def test_expected_errors_write_inconclusive_but_programming_errors_reraise(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    outcome = run_observation_pipeline(
        project_root=tmp_path,
        execute=lambda: (_ for _ in ()).throw(
            AssetLoadError("asset unavailable")
        ),
        base_provenance=_provenance(),
    )
    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "asset_load_failed"

    with pytest.raises(TypeError, match="programming bug"):
        run_observation_pipeline(
            project_root=tmp_path / "bug",
            execute=lambda: (_ for _ in ()).throw(
                TypeError("programming bug")
            ),
            base_provenance=_provenance(),
        )
    assert not (
        tmp_path
        / "bug"
        / "results"
        / "jspace_observation.json"
    ).exists()


def test_observation_module_neither_imports_interventions_nor_generates() -> None:
    source = (
        PROJECT / "src" / "jspace_llama" / "observation_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "interventions" not in source
    assert ".generate(" not in source


def test_real_observation_wiring_never_calls_model_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    payload = complete_payload()
    model = SimpleNamespace(
        tokenizer=object(),
        generate=lambda *args, **kwargs: pytest.fail(
            "observational execution must not generate"
        ),
    )
    bundle = SimpleNamespace(
        model=model,
        dictionary=object(),
        lens=SimpleNamespace(source_layers=range(17)),
        layers=(12, 13),
    )
    monkeypatch.setattr(
        pipeline,
        "load_pinned_runtime",
        lambda config, provenance: SimpleNamespace(bundle=bundle),
    )
    monkeypatch.setattr(
        pipeline,
        "load_multihop_items",
        lambda path: tuple(range(93)),
    )
    monkeypatch.setattr(
        pipeline,
        "audit_tokenization",
        lambda items, tokenizer: payload["tokenization_audit"],
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_direct_readout",
        lambda *args, **kwargs: payload["direct"],
    )
    monkeypatch.setattr(
        pipeline,
        "read_qualitative_panel",
        lambda *args, **kwargs: payload["panel"],
    )
    monkeypatch.setattr(
        pipeline,
        "read_source_provenance",
        lambda root: _provenance()["source"],
    )
    monkeypatch.setattr(
        pipeline,
        "collect_software_provenance",
        lambda: _provenance()["software"],
    )

    result = pipeline.execute_real_observation(
        project_root=tmp_path,
        config=__import__(
            "jspace_llama.config",
            fromlist=["ExperimentConfig"],
        ).ExperimentConfig(),
        provenance=_provenance(),
    )

    assert len(result["direct"]["items"]) == 93
    assert result["layers"] == {
        "band": [12, 13],
        "all_layers": list(range(17)),
    }
