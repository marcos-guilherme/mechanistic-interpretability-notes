from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jsonschema
import pytest
import torch

from jspace_llama.jsonio import atomic_write_json, document_sha256
from jspace_llama.observation_data import (
    DATASET_SHA256,
    DatasetValidationError,
    PANEL_NAMES,
    MultihopItem,
)
from jspace_llama.runtime import (
    AssetLoadError,
    CudaRequiredError,
    DimensionMismatchError,
    LayerBandError,
    LensIntegrityError,
    ResourceExhaustionError,
    TokenizationError,
)


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


def production_sized_payload() -> dict[str, object]:
    from jspace_llama.observation_artifacts import recompute_metrics

    payload = complete_payload()
    all_layers = list(range(32))
    band = list(range(12, 29))
    records = payload["direct"]["items"]
    audit = payload["tokenization_audit"]
    for item_index, (record, audit_item) in enumerate(
        zip(records, audit, strict=True)
    ):
        intermediates = []
        audited_intermediates = []
        for hop in range(3):
            text = f"production-intermediate-{item_index:02d}-hop-{hop}"
            token_id = item_index * 3 + hop
            intermediate = {
                "text": text,
                "token_id": token_id,
                "single_token": True,
                "jlens": _rank_detail(item_index + hop + 1, all_layers),
                "logit_lens": _rank_detail(
                    item_index + hop + 2,
                    all_layers,
                ),
            }
            intermediates.append(intermediate)
            audited_intermediates.append(
                {
                    "text": text,
                    "token_id": token_id,
                    "token_ids": [token_id],
                    "variant": "leading_space",
                    "single_token": True,
                    "prompt_sha256": hashlib.sha256(
                        f" {text}".encode()
                    ).hexdigest(),
                    "token_ids_sha256": hashlib.sha256(
                        f"[{token_id}]".encode()
                    ).hexdigest(),
                }
            )
        record["intermediates"] = intermediates
        audit_item["intermediates"] = audited_intermediates
        if record["name"] in PANEL_NAMES:
            record["top_tokens"] = {
                method: [
                    {
                        "layer": layer,
                        "tokens": [
                            {
                                "token_id": (
                                    item_index * 10_000
                                    + layer * 10
                                    + token_index
                                ),
                                "token": (
                                    f"{method}-item-{item_index:02d}-"
                                    f"layer-{layer:02d}-"
                                    f"token-{token_index:02d}"
                                ),
                                "rank": token_index + 1,
                            }
                            for token_index in range(10)
                        ],
                    }
                    for layer in band
                ]
                for method in ("jlens", "logit_lens")
            }
    payload["direct"]["metrics"] = recompute_metrics(records)
    payload["direct"]["single_token_metrics"] = copy.deepcopy(
        payload["direct"]["metrics"]
    )
    by_name = {record["name"]: record for record in records}
    payload["panel"]["items"] = [
        {
            "name": name,
            "prompt": by_name[name]["prompt"],
            "target": by_name[name]["target"],
            "intermediates": copy.deepcopy(by_name[name]["intermediates"]),
            "top_tokens": copy.deepcopy(by_name[name]["top_tokens"]),
            "pursuit": [
                {
                    "layer": layer,
                    "tokens": [
                        {
                            "token_id": (
                                panel_index * 100_000
                                + layer * 100
                                + token_index
                            ),
                            "token": (
                                f"pursuit-item-{panel_index:02d}-"
                                f"layer-{layer:02d}-"
                                f"token-{token_index:02d}"
                            ),
                            "rank": token_index + 1,
                            "coefficient": 1.0 / (token_index + 1),
                        }
                        for token_index in range(25)
                    ],
                    "variance_explained": 0.5,
                }
                for layer in band
            ],
        }
        for panel_index, name in enumerate(PANEL_NAMES)
    ]
    payload["layers"] = {"band": band, "all_layers": all_layers}
    return payload


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_observation", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _published_observation_paths(root: Path) -> list[Path]:
    return [
        root / "artifacts" / "jspace_observation_raw.json",
        root / "results" / "jspace_observation_manifest.json",
        root / "results" / "jspace_observation.json",
        *[
            root / "results" / "figures" / f"{stem}{suffix}"
            for stem in (
                "observation_pass_at_k",
                "observation_rank_heatmap",
                "observation_panel",
            )
            for suffix in (".png", ".svg")
        ],
    ]


def _write_fake_figure_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in (
        "observation_pass_at_k",
        "observation_rank_heatmap",
        "observation_panel",
    ):
        (output_dir / f"{stem}.png").write_bytes(
            b"\x89PNG\r\n\x1a\nfake"
        )
        (output_dir / f"{stem}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
            encoding="utf-8",
        )


def _rewrite_raw_and_hashes(
    raw_path: Path,
    raw: dict[str, object],
    manifest: dict[str, object],
    result: dict[str, object],
) -> None:
    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["raw_artifact_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    result["manifest_sha256"] = document_sha256(manifest)


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


def test_production_sized_compact_and_raw_render_and_validate(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(
        tmp_path,
        execute=production_sized_payload,
    )
    compact_path = tmp_path / "results" / "jspace_observation.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limit = 5 * 1_048_576

    assert 2_000_000 < compact_path.stat().st_size <= limit
    assert 2_000_000 < raw_path.stat().st_size <= limit
    _load_validator().validate_result_files(
        tmp_path / "results",
        SCHEMAS,
        raw_path,
    )
    assert len(list((tmp_path / "results" / "figures").iterdir())) == 6


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


def test_prepublication_failure_preserves_all_prior_observation_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    paths = _published_observation_paths(tmp_path)
    before = {}
    for index, path in enumerate(paths):
        content = f"prior-{index}".encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        before[path] = content
    def fail_render(command: list[str], *, timeout: float):
        return subprocess.CompletedProcess(
            command,
            returncode=1,
            stdout=b"",
            stderr=b"render failed before publication",
        )

    monkeypatch.setattr(pipeline, "_run_bounded_process", fail_render)

    with pytest.raises(RuntimeError, match="render failed"):
        pipeline.run_observation_pipeline(
            project_root=tmp_path,
            execute=complete_payload,
        )

    assert {path: path.read_bytes() for path in paths} == before
    assert list(tmp_path.glob(".observation-run-*")) == []


def test_isolated_renderer_uses_controlled_command_timeout_and_cleans_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    observed: dict[str, object] = {}

    def fake_run(command: list[str], *, timeout: float):
        observed["command"] = command
        observed["timeout"] = timeout
        input_path = Path(command[command.index("--input") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        observed["input_path"] = input_path
        assert input_path.is_file()
        assert json.loads(input_path.read_text(encoding="utf-8"))[
            "status"
        ] == "success"
        _write_fake_figure_outputs(output_dir)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=b"renderer output",
            stderr=b"",
        )

    monkeypatch.setattr(pipeline, "_run_bounded_process", fake_run)
    staging = tmp_path / "staging"
    output_dir = staging / "figures"
    staging.mkdir()

    paths = pipeline._render_observation_figures_isolated(
        {"status": "success"},
        staging=staging,
        output_dir=output_dir,
    )

    command = observed["command"]
    assert command[:3] == [
        sys.executable,
        "-m",
        "jspace_llama.observation_figure_cli",
    ]
    assert observed["timeout"] == pipeline.FIGURE_RENDER_TIMEOUT_SECONDS
    assert Path(command[command.index("--input") + 1]).is_absolute()
    assert Path(command[command.index("--output-dir") + 1]).is_absolute()
    assert not observed["input_path"].exists()
    assert len(paths) == 6


def test_isolated_renderer_timeout_cleans_temporary_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    observed_input: Path | None = None

    def time_out(command: list[str], *, timeout: float):
        nonlocal observed_input
        observed_input = Path(command[command.index("--input") + 1])
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(pipeline, "_run_bounded_process", time_out)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(RuntimeError, match="timed out"):
        pipeline._render_observation_figures_isolated(
            {"status": "success"},
            staging=staging,
            output_dir=staging / "figures",
        )

    assert observed_input is not None
    assert not observed_input.exists()


def test_renderer_failure_diagnostics_are_bounded_and_input_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    observed_input: Path | None = None

    def fail(command: list[str], *, timeout: float):
        nonlocal observed_input
        observed_input = Path(command[command.index("--input") + 1])
        return subprocess.CompletedProcess(
            command,
            returncode=9,
            stdout=b"o" * (pipeline.MAX_RENDER_OUTPUT_BYTES * 2),
            stderr=b"e" * (pipeline.MAX_RENDER_OUTPUT_BYTES * 2),
        )

    monkeypatch.setattr(pipeline, "_run_bounded_process", fail)
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(RuntimeError) as captured:
        pipeline._render_observation_figures_isolated(
            {"status": "success"},
            staging=staging,
            output_dir=staging / "figures",
        )

    assert len(str(captured.value)) < pipeline.MAX_RENDER_OUTPUT_BYTES * 3
    assert "truncated" in str(captured.value)
    assert observed_input is not None
    assert not observed_input.exists()


def test_renderer_subprocess_capture_is_bounded() -> None:
    import jspace_llama.observation_pipeline as pipeline

    size = pipeline.MAX_RENDER_OUTPUT_BYTES * 2
    command = [
        sys.executable,
        "-c",
        (
            "import os; "
            f"os.write(1, b'o' * {size}); "
            f"os.write(2, b'e' * {size})"
        ),
    ]

    completed = pipeline._run_bounded_process(command, timeout=30)

    assert completed.returncode == 0
    assert len(completed.stdout) < pipeline.MAX_RENDER_OUTPUT_BYTES + 32
    assert len(completed.stderr) < pipeline.MAX_RENDER_OUTPUT_BYTES + 32
    assert completed.stdout.endswith(b"...[truncated]")
    assert completed.stderr.endswith(b"...[truncated]")


def test_poisoned_matplotlib_does_not_block_pipeline_import_or_prerender(
    tmp_path: Path,
) -> None:
    script = r"""
import sys
from types import ModuleType

poisoned = ModuleType("matplotlib")
poisoned.__file__ = "poisoned-matplotlib.py"
def fail_if_used(*args, **kwargs):
    raise RuntimeError("poisoned matplotlib accessed")
poisoned.use = fail_if_used
sys.modules["matplotlib"] = poisoned
from jspace_llama.observation_pipeline import run_observation_pipeline
from jspace_llama.runtime import AssetLoadError
assert "jspace_llama.observation_figures" not in sys.modules
provenance = {
    "source": {},
    "model": {},
    "lens": {},
    "software": {},
    "hardware": {},
}
result = run_observation_pipeline(
    sys.argv[1],
    execute=lambda: (_ for _ in ()).throw(AssetLoadError("stop before render")),
    base_provenance=provenance,
)
assert result["reason"]["code"] == "asset_load_failed"
assert "jspace_llama.observation_figures" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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
    incomplete_manifest = json.loads(
        (
            tmp_path
            / "results"
            / "jspace_observation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    incomplete_raw_path = (
        tmp_path / "artifacts" / "jspace_observation_raw.json"
    )
    validator = _load_validator()
    validator.validate_documents(
        incomplete_manifest,
        outcome,
        schema_dir=SCHEMAS,
        raw_path=incomplete_raw_path,
    )
    contradictory_raw = json.loads(
        incomplete_raw_path.read_text(encoding="utf-8")
    )
    contradictory_raw["failure"]["reason_code"] = "dataset_invalid"
    _rewrite_raw_and_hashes(
        incomplete_raw_path,
        contradictory_raw,
        incomplete_manifest,
        outcome,
    )
    with pytest.raises(
        validator.ObservationValidationError,
        match="raw failure differs",
    ):
        validator.validate_documents(
            incomplete_manifest,
            outcome,
            schema_dir=SCHEMAS,
            raw_path=incomplete_raw_path,
        )
    contradictory_raw["failure"]["reason_code"] = "incomplete_observation"
    contradictory_raw["failure"]["error_type"] = "WrongErrorType"
    _rewrite_raw_and_hashes(
        incomplete_raw_path,
        contradictory_raw,
        incomplete_manifest,
        outcome,
    )
    with pytest.raises(
        validator.ObservationValidationError,
        match="error type|failure differs",
    ):
        validator.validate_documents(
            incomplete_manifest,
            outcome,
            schema_dir=SCHEMAS,
            raw_path=incomplete_raw_path,
        )

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


def test_incomplete_provenance_still_reconciles_complete_raw(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    payload = complete_payload()
    payload["provenance"]["source"]["scoped_dirty"] = True
    result = run_observation_pipeline(
        tmp_path,
        execute=lambda: payload,
    )
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validator = _load_validator()

    assert result["status"] == "inconclusive"
    assert result["reason"]["code"] == "incomplete_provenance"
    validator.validate_documents(
        manifest,
        result,
        schema_dir=SCHEMAS,
        raw_path=raw_path,
    )

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    failure_forgery = {
        "provenance": manifest["provenance"],
        "failure": {
            "reason_code": "incomplete_provenance",
            "error_type": "ForgedFailure",
        },
    }
    _rewrite_raw_and_hashes(
        raw_path,
        failure_forgery,
        manifest,
        result,
    )
    with pytest.raises(
        validator.ObservationValidationError,
        match="incomplete provenance.*measurement|raw failure differs",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )

    raw["tokenization_audit"][0]["intermediates"][0][
        "token_ids_sha256"
    ] = "0" * 64
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    with pytest.raises(
        validator.ObservationValidationError,
        match="tokenization.*hash|hashes.*recompute",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


@pytest.mark.parametrize(
    ("error", "reason_code", "error_type"),
    [
        (
            CudaRequiredError("CUDA unavailable"),
            "cuda_required",
            "CudaRequiredError",
        ),
        (
            AssetLoadError("asset unavailable"),
            "asset_load_failed",
            "AssetLoadError",
        ),
        (
            ResourceExhaustionError("resource exhausted"),
            "resource_exhausted",
            "ResourceExhaustionError",
        ),
        (
            torch.OutOfMemoryError("out of memory"),
            "resource_exhausted",
            "OutOfMemoryError",
        ),
        (
            TokenizationError("invalid tokenization"),
            "tokenization_failed",
            "TokenizationError",
        ),
        (
            DimensionMismatchError("dimension mismatch"),
            "dimension_mismatch",
            "DimensionMismatchError",
        ),
        (
            LayerBandError("invalid layer band"),
            "layer_band_invalid",
            "LayerBandError",
        ),
        (
            LensIntegrityError("invalid lens"),
            "lens_integrity_failed",
            "LensIntegrityError",
        ),
        (
            DatasetValidationError("invalid dataset"),
            "dataset_invalid",
            "DatasetValidationError",
        ),
    ],
    ids=[
        "cuda",
        "asset-load",
        "resource-exhausted",
        "torch-oom",
        "tokenization",
        "dimension",
        "layer-band",
        "lens-integrity",
        "dataset",
    ],
)
def test_validator_enforces_pipeline_failure_reason_to_type_mapping(
    tmp_path: Path,
    error: BaseException,
    reason_code: str,
    error_type: str,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(
        tmp_path,
        execute=lambda: (_ for _ in ()).throw(error),
        base_provenance=_provenance(),
    )
    manifest = json.loads(
        (
            tmp_path / "results" / "jspace_observation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validator = _load_validator()

    assert result["reason"]["code"] == reason_code
    assert raw["failure"]["error_type"] == error_type
    validator.validate_documents(
        manifest,
        result,
        schema_dir=SCHEMAS,
        raw_path=raw_path,
    )

    raw["failure"]["error_type"] = "WrongErrorType"
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    with pytest.raises(
        validator.ObservationValidationError,
        match="error type|failure differs",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


def test_validator_rejects_empty_failure_form_incomplete_provenance(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_artifacts import build_observation_artifacts

    manifest, result = build_observation_artifacts(
        run_id="22222222-2222-4222-8222-222222222222",
        started_at="2026-11-11T00:00:00Z",
        ended_at="2026-11-11T00:00:01Z",
        provenance=_provenance(),
        raw_artifact_sha256="0" * 64,
        payload=None,
        reason_code="incomplete_provenance",
    )
    raw = {
        "provenance": manifest["provenance"],
        "failure": {
            "reason_code": "incomplete_provenance",
            "error_type": "ArbitraryFailure",
        },
    }
    raw_path = tmp_path / "jspace_observation_raw.json"
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="incomplete provenance.*measurement|failure.*not allowed",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


def test_validator_requires_raw_artifact(tmp_path: Path) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    run_observation_pipeline(tmp_path, execute=complete_payload)
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    raw_path.unlink()
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="raw artifact|required|regular file",
    ):
        validator.validate_result_files(
            tmp_path / "results",
            SCHEMAS,
            raw_path,
        )


def test_validator_uses_bounded_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(tmp_path, execute=complete_payload)
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    compact_path = tmp_path / "results" / "jspace_observation.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validator = _load_validator()
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: pytest.fail(f"unbounded read used for {path}"),
    )

    assert validator._read_json(compact_path)["run_id"] == result["run_id"]
    assert validator._validate_raw(
        manifest,
        raw_path=raw_path,
    )["provenance"] == manifest["provenance"]


@pytest.mark.parametrize("oversized_artifact", ["compact", "raw"])
def test_validator_rejects_observation_json_above_five_mib(
    tmp_path: Path,
    oversized_artifact: str,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(tmp_path, execute=complete_payload)
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    compact_path = tmp_path / "results" / "jspace_observation.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = compact_path if oversized_artifact == "compact" else raw_path
    target.write_bytes(
        target.read_bytes()
        + b" " * (5 * 1_048_576 - target.stat().st_size + 1)
    )
    if oversized_artifact == "raw":
        manifest["raw_artifact_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        result["manifest_sha256"] = document_sha256(manifest)
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(compact_path, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="too large",
    ):
        validator.validate_result_files(
            tmp_path / "results",
            SCHEMAS,
            raw_path,
        )


def test_validator_rejects_raw_rank_that_contradicts_compact(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(tmp_path, execute=complete_payload)
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["direct"]["items"][0]["intermediates"][0]["jlens"]["ranks"][0][
        "rank"
    ] = 99
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="raw.*compact|direct.*differ",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


def test_validator_rejects_arbitrary_posix_absolute_path(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(tmp_path, execute=complete_payload)
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    forbidden = "/var/lib/private/model.pt"
    result["panel"][0]["top_tokens"]["jlens"][0]["tokens"][0][
        "token"
    ] = forbidden
    raw["panel"]["items"][0]["top_tokens"]["jlens"][0]["tokens"][0][
        "token"
    ] = forbidden
    raw["direct"]["items"][0]["top_tokens"]["jlens"][0]["tokens"][0][
        "token"
    ] = forbidden
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="absolute/local path",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


def test_validator_rejects_forged_raw_for_incomplete_observation(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    payload = complete_payload()
    payload["direct"]["items"].pop()
    result = run_observation_pipeline(
        tmp_path,
        execute=lambda: payload,
    )
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged = {
        "provenance": manifest["provenance"],
        "arbitrary": {"safe_value": "not a real observation"},
    }
    _rewrite_raw_and_hashes(raw_path, forged, manifest, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="raw.*unexpected|raw.*failure|raw.*observation",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entry: entry.update(text="tampered text"),
        lambda entry: entry.update(token_id=999),
        lambda entry: entry.update(token_ids=[999]),
        lambda entry: entry.update(variant="raw"),
        lambda entry: entry.update(single_token=False),
        lambda entry: entry.update(prompt_sha256="0" * 64),
        lambda entry: entry.update(token_ids_sha256="0" * 64),
    ],
    ids=[
        "text",
        "token-id",
        "token-ids",
        "variant",
        "single-token",
        "prompt-hash",
        "token-ids-hash",
    ],
)
def test_validator_recomputes_complete_tokenization_audit_contract(
    tmp_path: Path,
    mutate,
) -> None:
    from jspace_llama.observation_pipeline import run_observation_pipeline

    result = run_observation_pipeline(tmp_path, execute=complete_payload)
    manifest_path = tmp_path / "results" / "jspace_observation_manifest.json"
    raw_path = tmp_path / "artifacts" / "jspace_observation_raw.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    mutate(raw["tokenization_audit"][0]["intermediates"][0])
    _rewrite_raw_and_hashes(raw_path, raw, manifest, result)
    validator = _load_validator()

    with pytest.raises(
        validator.ObservationValidationError,
        match="tokenization|audit|hash",
    ):
        validator.validate_documents(
            manifest,
            result,
            schema_dir=SCHEMAS,
            raw_path=raw_path,
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
    expected_manifest = json.loads(
        (
            tmp_path
            / "results"
            / "jspace_observation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    _load_validator().validate_documents(
        expected_manifest,
        outcome,
        schema_dir=SCHEMAS,
        raw_path=(
            tmp_path
            / "artifacts"
            / "jspace_observation_raw.json"
        ),
    )

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


def test_missing_dataset_is_expected_before_asset_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "require_cuda",
        lambda: SimpleNamespace(
            gpu_name="NVIDIA L4",
            dtype="bfloat16",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_pinned_runtime",
        lambda *args, **kwargs: pytest.fail(
            "dataset must be checked before loading assets"
        ),
    )

    outcome = pipeline.run_observation_pipeline(
        tmp_path,
        base_provenance=_provenance(),
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "dataset_invalid"


def test_cuda_guard_precedes_missing_dataset_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "require_cuda",
        lambda: (_ for _ in ()).throw(
            CudaRequiredError("CUDA unavailable")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_multihop_items",
        lambda path: pytest.fail(
            "dataset validation must follow the CUDA guard"
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "load_pinned_runtime",
        lambda *args, **kwargs: pytest.fail(
            "asset loading must not run without CUDA"
        ),
    )

    outcome = pipeline.run_observation_pipeline(
        tmp_path,
        base_provenance=_provenance(),
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "cuda_required"


def test_empty_tokenization_is_expected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    item = MultihopItem(
        name="item",
        prompt="prompt ",
        target="target",
        intermediates=("intermediate",),
    )
    tokenizer = SimpleNamespace(
        encode=lambda *args, **kwargs: [],
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(tokenizer=tokenizer),
        dictionary=object(),
        lens=SimpleNamespace(source_layers=(12,)),
        layers=(12,),
    )
    monkeypatch.setattr(
        pipeline,
        "load_multihop_items",
        lambda path: (item,),
    )
    monkeypatch.setattr(
        pipeline,
        "require_cuda",
        lambda: object(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_pinned_runtime",
        lambda config, provenance: SimpleNamespace(bundle=bundle),
    )

    outcome = pipeline.run_observation_pipeline(
        tmp_path,
        base_provenance=_provenance(),
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "tokenization_failed"


def test_unrelated_tokenization_programming_error_is_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_pipeline as pipeline

    item = MultihopItem(
        name="item",
        prompt="prompt ",
        target="target",
        intermediates=("intermediate",),
    )
    bundle = SimpleNamespace(
        model=SimpleNamespace(tokenizer=object()),
        dictionary=object(),
        lens=SimpleNamespace(source_layers=(12,)),
        layers=(12,),
    )
    monkeypatch.setattr(
        pipeline,
        "load_multihop_items",
        lambda path: (item,),
    )
    monkeypatch.setattr(
        pipeline,
        "require_cuda",
        lambda: object(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_pinned_runtime",
        lambda config, provenance: SimpleNamespace(bundle=bundle),
    )
    monkeypatch.setattr(
        pipeline,
        "audit_tokenization",
        lambda items, tokenizer: (_ for _ in ()).throw(
            ValueError("programming bug")
        ),
    )

    with pytest.raises(ValueError, match="programming bug"):
        pipeline.run_observation_pipeline(
            tmp_path,
            base_provenance=_provenance(),
        )


def test_observation_module_neither_imports_interventions_nor_generates() -> None:
    source = (
        PROJECT / "src" / "jspace_llama" / "observation_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "interventions" not in source
    assert "from jspace_llama.observation_figures import" not in source
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
        "require_cuda",
        lambda: object(),
        raising=False,
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
