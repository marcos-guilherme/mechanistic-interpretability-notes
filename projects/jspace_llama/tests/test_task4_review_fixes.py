from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import jsonschema
import pytest
import torch
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RevisionNotFoundError,
)

import test_artifact_pipeline as task4
from jspace_llama.config import AssetsConfig
from jspace_llama.interventions import (
    InterventionPreparationError,
    RandomControlError,
    SwapLayer,
    make_random_directions,
)
from jspace_llama.runtime import (
    AssetLoadError,
    JSpaceExperimentError,
    TokenizationError,
    download_lens_file,
    load_components,
)


PROJECT = Path(__file__).parents[1]
ROOT = Path(__file__).parents[3]
VALIDATOR_PATH = PROJECT / "scripts" / "validate_results.py"
PACK_PATH = PROJECT / "scripts" / "colab_pack_results.py"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
SCHEMAS = PROJECT / "schemas"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["model"].update(id="other/model"),
        lambda p: p["model"].update(resolved_revision="a" * 40),
        lambda p: p["model"].update(layer_count=None),
        lambda p: p["lens"].update(repo_id="other/lens"),
        lambda p: p["lens"].update(filename="other.pt"),
        lambda p: p["lens"].update(sha256="d" * 64),
        lambda p: p["lens"].update(size_bytes=True),
        lambda p: p["lens"].update(integrity_verified=False),
        lambda p: p["software"].update(jspace_commit="0" * 40),
        lambda p: p["software"].update(transformers="unknown"),
        lambda p: p["hardware"].update(dtype="float32"),
        lambda p: p["experiment"].update(prompts=[]),
        lambda p: p["experiment"]["prompts"].append(
            copy.deepcopy(p["experiment"]["prompts"][0])
        ),
        lambda p: p["experiment"]["prompts"][0].update(name="wrong"),
        lambda p: p["experiment"]["generation"].update(decoding="sample"),
        lambda p: p["experiment"].update(alphas=[]),
        lambda p: p["experiment"].update(alphas=[math.nan]),
        lambda p: p["experiment"].update(alphas=[math.inf]),
        lambda p: p["experiment"].update(layers=[]),
        lambda p: p["experiment"].update(layers=[12, 12]),
        lambda p: p["experiment"].update(layers=[31, 32]),
        lambda p: p["experiment"]["intervention"].update(
            random_control="unmatched"
        ),
    ],
)
def test_success_completeness_rejects_placeholders_and_invalid_contract(
    mutate,
) -> None:
    from jspace_llama.provenance import provenance_is_complete

    provenance = task4.complete_provenance()
    mutate(provenance)

    assert provenance_is_complete(provenance) is False


def test_success_schema_rejects_wrong_approved_assets() -> None:
    manifest, _ = task4.build_success_artifacts()
    manifest["provenance"]["model"]["id"] = "other/model"
    schema = json.loads(
        (SCHEMAS / "execution_manifest.schema.json").read_text(encoding="utf-8")
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, schema)


def test_success_schema_requires_verified_lens_integrity() -> None:
    _, final = task4.build_success_artifacts()
    final["integrity"]["lens_integrity_verified"] = False
    schema = json.loads(
        (SCHEMAS / "final_evaluation.schema.json").read_text(encoding="utf-8")
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(final, schema)


def test_strict_json_rejects_duplicates_and_nonfinite_values() -> None:
    from jspace_llama.jsonio import StrictJSONError, strict_json_loads

    with pytest.raises(StrictJSONError, match="duplicate"):
        strict_json_loads('{"value": 1, "value": 2}')
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(StrictJSONError, match="non-finite"):
            strict_json_loads(f'{{"value": {constant}}}')


def test_json_writes_reject_nonfinite_without_replacing_destination(
    tmp_path: Path,
) -> None:
    from jspace_llama.jsonio import atomic_write_json, canonical_json_bytes

    destination = tmp_path / "result.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        canonical_json_bytes({"value": math.nan})
    with pytest.raises(ValueError):
        atomic_write_json(destination, {"value": math.inf})

    assert destination.read_text(encoding="utf-8") == '{"old":true}\n'


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"commit":"%s","commit":"%s","dirty":false}'
        % ("a" * 40, "b" * 40),
        '{"schema_version":1,"commit":"%s","dirty":false,"value":NaN}'
        % ("a" * 40),
    ],
)
def test_source_provenance_uses_strict_json(
    tmp_path: Path,
    payload: str,
) -> None:
    from jspace_llama.jsonio import StrictJSONError
    from jspace_llama.provenance import read_source_provenance

    (tmp_path / "source_provenance.json").write_text(
        payload,
        encoding="utf-8",
    )

    with pytest.raises(StrictJSONError):
        read_source_provenance(tmp_path)


@pytest.mark.parametrize("target", ["pack", "runner"])
def test_result_archive_json_checks_are_strict(
    tmp_path: Path,
    target: str,
) -> None:
    payload = b'{"value":1,"value":2}'
    if target == "pack":
        module = load_script(PACK_PATH, "strict_pack_json")
        path = tmp_path / "artifact.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="valid JSON"):
            module._read_json_artifact(path)
    else:
        module = load_script(RUNNER_PATH, "strict_runner_json")
        with pytest.raises(ValueError, match="valid JSON"):
            module._validate_json_bytes("artifact.json", payload)


def test_arbitrary_loader_and_downloader_errors_reraise() -> None:
    class BuggyModel:
        @classmethod
        def load(cls, *args, **kwargs):
            raise TypeError("programming bug")

    with pytest.raises(TypeError, match="programming bug"):
        load_components(
            AssetsConfig(),
            "lens.pt",
            hf_model_cls=BuggyModel,
            lens_cls=object(),
            dictionary_cls=object(),
        )

    def buggy_downloader(**kwargs):
        raise TypeError("programming bug")

    with pytest.raises(TypeError, match="programming bug"):
        download_lens_file(
            AssetsConfig(),
            resolved_revision="a" * 40,
            downloader=buggy_downloader,
        )


def test_revision_lookup_operational_failure_is_explicit() -> None:
    from jspace_llama.provenance import resolve_huggingface_revisions

    class FailingApi:
        def model_info(self, repo_id):
            raise OSError("network unavailable")

    with pytest.raises(AssetLoadError):
        resolve_huggingface_revisions(
            AssetsConfig().model_id,
            AssetsConfig().lens_repo_id,
            api=FailingApi(),
        )

    class InvalidApi:
        def model_info(self, repo_id):
            return SimpleNamespace(sha="moving-branch")

    with pytest.raises(AssetLoadError):
        resolve_huggingface_revisions(
            AssetsConfig().model_id,
            AssetsConfig().lens_repo_id,
            api=InvalidApi(),
        )


def test_pipeline_does_not_catch_base_or_unknown_domain_errors(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    class UnknownDomainError(JSpaceExperimentError):
        pass

    for error in (
        JSpaceExperimentError("base bug"),
        UnknownDomainError("unknown bug"),
    ):
        with pytest.raises(type(error), match="bug"):
            run_pipeline(
                project_root=tmp_path / type(error).__name__,
                execute=lambda error=error: (_ for _ in ()).throw(error),
                base_provenance=task4.complete_provenance(),
            )


def test_control_preparation_errors_are_explicit_domain_errors() -> None:
    assert issubclass(InterventionPreparationError, JSpaceExperimentError)
    assert issubclass(RandomControlError, JSpaceExperimentError)


def test_compact_reasons_use_fixed_allowlist_and_never_exception_text(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    secret = "hf_1234567890secret"
    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: (_ for _ in ()).throw(
            TokenizationError(
                f"ids [101, 202], C:\\cache\\tokenizer, {secret}, raw prompt"
            )
        ),
        base_provenance=task4.complete_provenance(),
    )
    compact = json.dumps(outcome)

    assert outcome["reason"] == {
        "code": "tokenization_failed",
        "message": "intervention concepts are not valid single tokens",
    }
    for forbidden in (
        "ids [101, 202]",
        r"C:\cache\tokenizer",
        secret,
        "raw prompt",
    ):
        assert forbidden not in compact
    raw = json.loads(
        (tmp_path / "artifacts" / "readout_and_causal_swap.json").read_text(
            encoding="utf-8"
        )
    )
    assert "101" in raw["detail"]
    assert "raw prompt" in raw["detail"]
    assert secret not in raw["detail"]


def test_manifest_builder_rejects_noncanonical_reason() -> None:
    from jspace_llama.artifacts import build_execution_manifest

    with pytest.raises(ValueError, match="canonical"):
        build_execution_manifest(
            run_id="run",
            started_at="2026-11-10T20:00:00Z",
            ended_at="2026-11-10T20:01:00Z",
            terminal_status="unexpected_error",
            status="error",
            provenance=task4.complete_provenance(),
            reason={"code": "unexpected_error", "message": "leaked detail"},
        )


def test_known_operational_subclass_uses_parent_reason(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    class SpecificAssetFailure(AssetLoadError):
        pass

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: (_ for _ in ()).throw(
            SpecificAssetFailure("private detail")
        ),
        base_provenance=task4.complete_provenance(),
    )

    assert outcome["reason"]["code"] == "asset_load_failed"


PATH_LIKE_VALUES = (
    "cache at C:\\Users\\name\\model",
    "cache at \\\\server\\share\\model",
    "cache at /home/name/model",
    "cache at ~/models/model",
    "cache at file:///tmp/model",
    "cache at $HOME/models/model",
)


@pytest.mark.parametrize("path_like", PATH_LIKE_VALUES)
def test_validator_rejects_embedded_path_like_strings(path_like: str) -> None:
    from jspace_llama.jsonio import document_sha256

    validator = load_script(VALIDATOR_PATH, "embedded_path_validator")
    manifest, final = task4.build_success_artifacts()
    manifest["provenance"]["hardware"]["gpu_name"] = path_like
    final["manifest_sha256"] = document_sha256(manifest)

    with pytest.raises(validator.ResultsValidationError, match="path"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def write_results(results: Path) -> None:
    manifest, final = task4.build_success_artifacts()
    results.mkdir(parents=True)
    (results / "execution_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (results / "final_evaluation.json").write_text(
        json.dumps(final),
        encoding="utf-8",
    )


def test_validator_rejects_hardlinked_result_file(tmp_path: Path) -> None:
    validator = load_script(VALIDATOR_PATH, "hardlink_validator")
    results = tmp_path / "results"
    write_results(results)
    os.link(
        results / "execution_manifest.json",
        tmp_path / "manifest-link.json",
    )

    with pytest.raises(validator.ResultsValidationError, match="hard-linked"):
        validator.validate_result_files(results, SCHEMAS)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFREG, 0x400, "reparse"),
        (stat.S_IFIFO, 0, "regular"),
    ],
)
def test_validator_rejects_unsafe_file_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    validator = load_script(
        VALIDATOR_PATH,
        f"metadata_validator_{mode}_{attributes}",
    )
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
        st_nlink=1,
        st_size=2,
    )

    with pytest.raises(validator.ResultsValidationError, match=message):
        validator._validate_regular_file(tmp_path / "result.json", metadata)


@pytest.mark.parametrize(
    ("mode", "attributes", "message"),
    [
        (stat.S_IFLNK, 0, "symlink"),
        (stat.S_IFDIR, 0x400, "reparse"),
        (stat.S_IFREG, 0, "directory"),
    ],
)
def test_validator_rejects_unsafe_directory_metadata(
    tmp_path: Path,
    mode: int,
    attributes: int,
    message: str,
) -> None:
    validator = load_script(
        VALIDATOR_PATH,
        f"directory_validator_{mode}_{attributes}",
    )
    metadata = SimpleNamespace(
        st_mode=mode,
        st_file_attributes=attributes,
    )

    with pytest.raises(validator.ResultsValidationError, match=message):
        validator._validate_directory(
            tmp_path / "results",
            metadata=metadata,
        )


def test_validator_rejects_changed_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = load_script(VALIDATOR_PATH, "identity_validator")
    results = tmp_path / "results"
    write_results(results)
    real_fstat = validator.os.fstat

    def changed_fstat(descriptor: int):
        actual = real_fstat(descriptor)
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(validator.os, "fstat", changed_fstat)

    with pytest.raises(validator.ResultsValidationError, match="changed"):
        validator.validate_result_files(results, SCHEMAS)


def test_validator_no_follow_fallback_remains_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = load_script(VALIDATOR_PATH, "fallback_validator")
    results = tmp_path / "results"
    write_results(results)
    monkeypatch.setattr(validator.os, "O_NOFOLLOW", 0, raising=False)

    validator.validate_result_files(results, SCHEMAS)


def test_lens_rehash_detects_post_load_change(tmp_path: Path) -> None:
    from jspace_llama.provenance import (
        LensIntegrityError,
        file_sha256,
        verify_lens_integrity,
    )

    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"downloaded lens")
    digest, size = file_sha256(lens)
    assert verify_lens_integrity(lens, digest, size) is True

    lens.write_bytes(b"changed after load")
    with pytest.raises(LensIntegrityError):
        verify_lens_integrity(lens, digest, size)


def test_final_artifact_agrees_on_verified_lens_integrity() -> None:
    manifest, final = task4.build_success_artifacts()

    assert final["integrity"] == {
        "lens_sha256": manifest["provenance"]["lens"]["sha256"],
        "lens_integrity_verified": True,
    }

    validator = load_script(VALIDATOR_PATH, "integrity_validator")
    final["integrity"]["lens_sha256"] = "0" * 64
    with pytest.raises(validator.ResultsValidationError, match="lens"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def test_unexpected_manifest_write_failure_preserves_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.pipeline as pipeline

    real_write = pipeline.atomic_write_json

    def fail_manifest(path, document):
        if Path(path).name == "execution_manifest.json":
            raise OSError("manifest disk failure")
        return real_write(path, document)

    monkeypatch.setattr(pipeline, "atomic_write_json", fail_manifest)

    with pytest.raises(TypeError, match="original programming error"):
        pipeline.run_pipeline(
            project_root=tmp_path,
            execute=lambda: (_ for _ in ()).throw(
                TypeError("original programming error")
            ),
            base_provenance=task4.complete_provenance(),
        )


def test_validator_gate_equations_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.evaluation as evaluation

    manifest, final = task4.build_success_artifacts()
    monkeypatch.setattr(
        evaluation,
        "evaluate_conditions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("production evaluator must not be called")
        ),
    )
    validator = load_script(VALIDATOR_PATH, "independent_gate_validator")

    validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def test_random_control_linalg_failure_is_expected_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    failure = torch.linalg.LinAlgError("SVD did not converge")

    def fail_svd(*args, **kwargs):
        raise failure

    monkeypatch.setattr(torch.linalg, "svd", fail_svd)
    prepared = {
        12: SwapLayer(
            vectors=torch.eye(2),
            pseudoinverse=torch.eye(2),
            noop=False,
        )
    }

    with pytest.raises(RandomControlError) as caught:
        make_random_directions(prepared, seed=42)
    assert caught.value.__cause__ is failure

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: make_random_directions(prepared, seed=42),
        base_provenance=task4.complete_provenance(),
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "random_control_failed"
    manifest = json.loads(
        (tmp_path / "results" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["terminal_status"] == "expected_failure"


def _hub_operational_error(kind: str) -> BaseException:
    response = SimpleNamespace(
        status_code=401,
        headers={},
        request=SimpleNamespace(url="https://huggingface.co/private/repo"),
    )
    if kind == "auth":
        return HfHubHTTPError("authentication failed", response=response)
    if kind == "gated":
        return GatedRepoError("gated repository", response=response)
    if kind == "revision":
        return RevisionNotFoundError("missing revision", response=response)
    if kind == "offline":
        return OfflineModeIsEnabled("offline mode is enabled")
    if kind == "local_entry":
        return LocalEntryNotFoundError("local cache entry missing")
    if kind == "entry":
        return EntryNotFoundError("repository entry missing")
    raise AssertionError(f"unsupported test exception kind: {kind}")


@pytest.mark.parametrize(
    "kind",
    ["auth", "gated", "revision", "offline", "local_entry", "entry"],
)
def test_huggingface_operational_failures_translate_to_asset_load_error(
    kind: str,
) -> None:
    from jspace_llama.provenance import resolve_huggingface_revisions

    error = _hub_operational_error(kind)

    class FailingApi:
        def model_info(self, repo_id):
            raise error

    with pytest.raises(AssetLoadError) as revision_failure:
        resolve_huggingface_revisions(
            AssetsConfig().model_id,
            AssetsConfig().lens_repo_id,
            api=FailingApi(),
        )
    assert revision_failure.value.__cause__ is error

    def fail_download(**kwargs):
        raise error

    with pytest.raises(AssetLoadError) as download_failure:
        download_lens_file(
            AssetsConfig(),
            resolved_revision="a" * 40,
            downloader=fail_download,
        )
    assert download_failure.value.__cause__ is error


def test_execute_real_experiment_translates_non_os_hub_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    import jspace_llama.pipeline as pipeline

    error = EntryNotFoundError("snapshot entry missing")
    monkeypatch.setattr(
        pipeline,
        "require_cuda",
        lambda: SimpleNamespace(
            gpu_name="NVIDIA L4",
            dtype="bfloat16",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_huggingface_revisions",
        lambda *args: ("a" * 40, "b" * 40),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(AssetLoadError) as caught:
        pipeline.execute_real_experiment(
            project_root=tmp_path,
            config=task4.ExperimentConfig(),
            provenance=task4.complete_provenance(),
        )
    assert caught.value.__cause__ is error


def test_model_component_load_translates_hub_operational_failure() -> None:
    error = EntryNotFoundError("model snapshot entry missing")

    class FailingModel:
        @classmethod
        def load(cls, *args, **kwargs):
            raise error

    with pytest.raises(AssetLoadError) as caught:
        load_components(
            AssetsConfig(),
            "lens.pt",
            hf_model_cls=FailingModel,
            lens_cls=object(),
            dictionary_cls=object(),
        )
    assert caught.value.__cause__ is error


def test_paraphrase_artifact_has_per_prompt_alpha_gate_aggregates() -> None:
    _, final = task4.build_success_artifacts()
    prompt_alphas = final["paraphrases"]["prompt_alphas"]

    assert {
        (item["prompt_name"], item["alpha"])
        for item in prompt_alphas
    } == {
        ("paraphrase_animal", 4.0),
        ("paraphrase_creature", 4.0),
    }
    assert all(item["passed"] is True for item in prompt_alphas)
    assert all(all(item["gates"].values()) for item in prompt_alphas)
    compact = json.dumps(final["paraphrases"])
    assert "raw paraphrase" not in compact
    assert "continuation" not in compact


@pytest.mark.parametrize(
    "tamper",
    [
        lambda p: p.update(prompt_count=1),
        lambda p: p.update(passing_prompt_alpha_count=1),
        lambda p: p["prompt_alphas"][0]["gates"].update(
            baseline_eight=False
        ),
        lambda p: p["prompt_alphas"][0].update(passed=False),
        lambda p: p["prompt_alphas"][0].update(prompt_name="unexpected"),
        lambda p: p["prompt_alphas"][0].update(alpha=99.0),
        lambda p: p["prompt_alphas"].pop(),
    ],
)
def test_validator_recomputes_and_cross_checks_paraphrase_aggregates(
    tamper,
) -> None:
    validator = load_script(
        VALIDATOR_PATH,
        "paraphrase_aggregate_validator",
    )
    manifest, final = task4.build_success_artifacts()
    tamper(final["paraphrases"])

    with pytest.raises(validator.ResultsValidationError, match="paraphrase"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def test_evaluation_failure_merges_into_existing_raw_diagnostics(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    payload = {
        "provenance": task4.complete_provenance(),
        **task4.successful_raw(),
    }
    payload["experiment"]["observations"] = [
        observation
        for observation in payload["experiment"]["observations"]
        if not (
            observation["prompt"]["name"] == "main"
            and observation["condition"] == "target"
        )
    ]

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: payload,
    )

    raw = json.loads(
        (tmp_path / "artifacts" / "readout_and_causal_swap.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["readout"] == payload["readout"]
    assert raw["experiment"] == payload["experiment"]
    assert raw["failure"]["error_type"] == "EvaluationDataError"
    assert raw["failure"]["cause_detail"] is None
    assert outcome["reason"] == {
        "code": "evaluation_data_invalid",
        "message": "required aggregate evaluation data is missing",
    }
    compact = json.dumps(outcome)
    assert "missing primary condition" not in compact
