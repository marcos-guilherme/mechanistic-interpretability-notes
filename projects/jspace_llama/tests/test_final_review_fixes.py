from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import jsonschema
import pytest
import torch
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RevisionNotFoundError,
)

import test_artifact_pipeline as task4
from jspace_llama.config import (
    ExperimentConfig,
    PromptCase,
    PromptConfig,
    SweepConfig,
)
from jspace_llama.experiment import (
    ModelExperimentRunner,
    classify_continuation,
    prepare_prompts,
    run_experiment,
)
from jspace_llama.interventions import SwapLayer, apply_causal_clamp


PROJECT = Path(__file__).parents[1]
ROOT = Path(__file__).parents[3]
SCHEMAS = PROJECT / "schemas"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
VALIDATOR_PATH = PROJECT / "scripts" / "validate_results.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unmeasured_failure_artifact_uses_evaluated_false_and_nulls() -> None:
    from jspace_llama.artifacts import build_artifacts

    provenance = task4.complete_provenance()
    provenance["lens"].update(
        sha256=None,
        size_bytes=None,
        integrity_verified=None,
    )
    manifest, final = build_artifacts(
        run_id="asset-failure",
        started_at="2026-11-10T20:00:00Z",
        ended_at="2026-11-10T20:01:00Z",
        terminal_status="expected_failure",
        provenance=provenance,
        raw_diagnostics=None,
        reason_code="asset_load_failed",
    )

    assert final["integrity"] == {
        "lens_integrity_evaluated": False,
        "lens_sha256": None,
        "lens_integrity_verified": None,
    }
    assert final["primary"]["readout"] == {
        "evaluated": False,
        "spider_present": None,
    }
    jsonschema.validate(
        manifest,
        json.loads(
            (SCHEMAS / "execution_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    jsonschema.validate(
        final,
        json.loads(
            (SCHEMAS / "final_evaluation.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def test_success_requires_evaluated_lens_and_readout() -> None:
    manifest, final = task4.build_success_artifacts()

    assert final["integrity"]["lens_integrity_evaluated"] is True
    assert final["primary"]["readout"] == {
        "evaluated": True,
        "spider_present": True,
    }

    validator = load_script(VALIDATOR_PATH, "final_review_evaluated")
    final["primary"]["readout"]["evaluated"] = False
    from jspace_llama.jsonio import document_sha256

    final["manifest_sha256"] = document_sha256(manifest)
    with pytest.raises(
        validator.ResultsValidationError,
        match="readout|evaluated|status",
    ):
        validator.validate_documents(
            manifest,
            final,
            schema_dir=SCHEMAS,
            project_root=PROJECT,
        )


@pytest.mark.parametrize(
    ("continuation", "status", "value", "incoherent"),
    [
        ("The answer is eight.", "unique", "eight", False),
        ("No, eight.", "negated", None, False),
        ("Six.", "unique", "six", False),
        ("not six", "negated", None, False),
        ("six or eight", "conflicting", None, False),
        ("I cannot determine a number.", "no_answer", None, False),
        ("", "no_answer", None, False),
        ("eight x q z p v", "no_answer", None, True),
    ],
)
def test_constrained_first_answer_parser_is_adversarial(
    continuation: str,
    status: str,
    value: str | None,
    incoherent: bool,
) -> None:
    classified = classify_continuation(continuation)

    assert classified["answer_parse"] == {
        "status": status,
        "value": value,
    }
    assert classified["degradation"]["incoherent"] is incoherent


@pytest.mark.parametrize(
    ("continuation", "value"),
    [
        ("6", "six"),
        ("six", "six"),
        ("Six legs.", "six"),
        ("The answer is six.", "six"),
        ("A spider has eight legs.", "eight"),
        ("8.", "eight"),
        ("Eight legs!", "eight"),
    ],
)
def test_positive_answer_grammar_accepts_only_short_assertions(
    continuation: str,
    value: str,
) -> None:
    assert classify_continuation(continuation)["answer_parse"] == {
        "status": "unique",
        "value": value,
    }


@pytest.mark.parametrize(
    "continuation",
    [
        "Six is not the answer.",
        "not exactly six",
        "Six is incorrect.",
        "Six is wrong.",
        "I cannot confirm six.",
        "I can't confirm six.",
        "I am unable to confirm six.",
        "I am unsure: six.",
        "Maybe six.",
        "Six, but I am uncertain.",
        "Six; however, I am uncertain.",
        "Six, rather tentatively.",
        "Six instead.",
        "Actually, six.",
    ],
)
def test_candidate_is_rejected_when_any_disqualifying_cue_appears(
    continuation: str,
) -> None:
    parsed = classify_continuation(continuation)["answer_parse"]

    assert parsed["status"] != "unique"
    assert parsed["value"] is None


@pytest.mark.parametrize(
    ("condition", "continuation", "failed_gate"),
    [
        ("baseline", "not eight", "baseline_unique_eight"),
        ("baseline", "six and eight", "baseline_unique_eight"),
        ("noop", "eight eight eight eight", "noop_not_degraded"),
        ("noop", "not six", "noop_unique_eight"),
        ("random", "I do not know.", "random_unique_non_six"),
        ("random", "six", "random_unique_non_six"),
        ("target", "six or eight", "target_unique_six"),
        ("target", "six x q z p v", "target_not_degraded"),
    ],
)
def test_answer_gates_reject_invalid_or_degraded_continuations(
    condition: str,
    continuation: str,
    failed_gate: str,
) -> None:
    from jspace_llama.evaluation import evaluate_conditions

    conditions = {
        "baseline": classify_continuation("eight"),
        "noop": classify_continuation("eight"),
        "random": classify_continuation("eight"),
        "target": classify_continuation("six"),
    }
    conditions[condition] = classify_continuation(continuation)

    gates, _ = evaluate_conditions(conditions, spider_present=True)

    assert gates[failed_gate] is False


@pytest.mark.parametrize(
    ("condition", "continuation", "failed_gate"),
    [
        (
            "baseline",
            "Eight, but that may be wrong.",
            "baseline_unique_eight",
        ),
        (
            "baseline",
            "The answer is eight. However, I am unsure.",
            "baseline_unique_eight",
        ),
        (
            "noop",
            "I cannot confirm eight.",
            "noop_unique_eight",
        ),
        (
            "noop",
            "Eight, actually.",
            "noop_unique_eight",
        ),
        (
            "target",
            "Six is not the answer.",
            "target_unique_six",
        ),
        (
            "target",
            "not exactly six",
            "target_unique_six",
        ),
        (
            "target",
            "The answer is six, maybe.",
            "target_unique_six",
        ),
    ],
)
def test_scientific_gates_reject_qualified_candidate_answers(
    condition: str,
    continuation: str,
    failed_gate: str,
) -> None:
    from jspace_llama.evaluation import evaluate_conditions

    conditions = {
        "baseline": classify_continuation("eight"),
        "noop": classify_continuation("eight"),
        "random": classify_continuation("eight"),
        "target": classify_continuation("six"),
    }
    conditions[condition] = classify_continuation(continuation)

    gates, _ = evaluate_conditions(conditions, spider_present=True)

    assert gates[failed_gate] is False


def _write_runtime_fixture(repository: Path) -> Path:
    project = repository / "projects" / "jspace_llama"
    files = {
        "README.md": "user documentation",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "lock",
        "src/jspace_llama/core.py": "VALUE = 1\n",
        "scripts/runtime.py": "RUNTIME = True\n",
        "schemas/result.schema.json": "{}\n",
        "notebooks/01_readout_and_causal_swap.ipynb": (
            '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\n'
        ),
        "tests/test_core.py": "def test_value(): pass\n",
        "results/final_evaluation.json": "{}\n",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    runner = repository / "tools" / "colab" / "run_jspace.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("RUNNER = True\n", encoding="utf-8")
    return project


def test_archive_digest_and_dirty_state_ignore_nonruntime_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_script(RUNNER_PATH, "final_review_archive_digest")
    repository = tmp_path / "repo"
    project = _write_runtime_fixture(repository)
    status_commands: list[list[str]] = []
    committed = {
        path.relative_to(repository).as_posix(): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file()
    }

    def fake_git(command, **kwargs):
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, "9" * 40 + "\n", "")
        if "ls-tree" in command:
            paths = "\n".join(sorted(committed)) + "\n"
            return subprocess.CompletedProcess(
                command,
                0,
                paths.encode(),
                b"",
            )
        if "show" in command:
            label = command[-1].split(":", 1)[1]
            return subprocess.CompletedProcess(
                command,
                0,
                committed[label],
                b"",
            )
        status_commands.append(command)
        dirty = (
            " M projects/jspace_llama/src/jspace_llama/core.py\n"
            if (project / "src" / "jspace_llama" / "core.py").read_text()
            == "VALUE = 2\n"
            else ""
        )
        return subprocess.CompletedProcess(command, 0, dirty, "")

    monkeypatch.setattr(runner.subprocess, "run", fake_git)
    archive = runner.build_archive(
        project=project,
        staging=tmp_path / "staging",
        repository_root=repository,
    )

    def archived_source() -> dict:
        with tarfile.open(archive, "r:gz") as handle:
            source = handle.extractfile("jspace_llama/source_provenance.json")
            assert source is not None
            return json.load(source)

    first = archived_source()
    assert first["scoped_dirty"] is False
    assert first["archive_sha256"]
    assert first["runtime_source_sha256"]
    status_command = status_commands[-1]
    joined = " ".join(status_command)
    assert "--" in status_command
    assert "projects/jspace_llama/src" in joined
    assert "projects/jspace_llama/README.md" not in joined
    assert "projects/jspace_llama/tests" not in joined

    (project / "README.md").write_text("changed docs", encoding="utf-8")
    archive = runner.build_archive(
        project=project,
        staging=tmp_path / "staging",
        repository_root=repository,
    )
    assert archived_source()["archive_sha256"] == first["archive_sha256"]
    assert (
        archived_source()["runtime_source_sha256"]
        == first["runtime_source_sha256"]
    )

    (project / "src" / "jspace_llama" / "core.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    archive = runner.build_archive(
        project=project,
        staging=tmp_path / "staging",
        repository_root=repository,
    )
    final = archived_source()
    assert final["archive_sha256"] != first["archive_sha256"]
    assert final["runtime_source_sha256"] == first["runtime_source_sha256"]
    assert final["scoped_dirty"] is True


def test_validator_independently_recomputes_runtime_source_digest() -> None:
    from jspace_llama.jsonio import document_sha256
    from jspace_llama.provenance import runtime_source_sha256

    validator = load_script(VALIDATOR_PATH, "final_review_source_validator")
    manifest, final = task4.build_success_artifacts()
    manifest["provenance"]["source"].update(
        archive_sha256=runtime_source_sha256(PROJECT, ROOT),
        scoped_dirty=False,
    )
    final["manifest_sha256"] = document_sha256(manifest)
    validator.validate_documents(
        manifest,
        final,
        schema_dir=SCHEMAS,
        project_root=PROJECT,
    )

    manifest["provenance"]["source"]["archive_sha256"] = hashlib.sha256(
        b"tampered source"
    ).hexdigest()
    final["manifest_sha256"] = document_sha256(manifest)
    with pytest.raises(
        validator.ResultsValidationError,
        match="source archive digest",
    ):
        validator.validate_documents(
            manifest,
            final,
            schema_dir=SCHEMAS,
            project_root=PROJECT,
        )


class BosTemplateTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<s>{messages[0]['content']}<assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return [self.bos_token_id, 42]

    def decode(self, token_ids) -> str:
        return "eight"


def test_rendered_template_is_tokenized_once_and_ids_are_reused() -> None:
    import jspace_llama.experiment as experiment

    tokenizer = BosTemplateTokenizer()
    config = ExperimentConfig(
        prompts=PromptConfig(cases=(PromptCase("main", "Question?"),)),
        sweep=SweepConfig(layers=(1,), alphas=(4.0,)),
    )
    prepared = prepare_prompts(config, tokenizer)
    requests = []
    run_experiment(
        config,
        tokenizer,
        layers=[1],
        generate=lambda request: requests.append(request) or "eight",
        prepared_prompts=prepared,
    )

    assert tokenizer.encode_calls == [(prepared[0].rendered, False)]
    assert prepared[0].token_ids.count(tokenizer.bos_token_id) == 1
    assert len(requests) == 4
    assert all(request.token_ids == prepared[0].token_ids for request in requests)

    readout_ids = None

    def capture_readout_ids(model, token_ids, layers, max_seq_len):
        nonlocal readout_ids
        readout_ids = tuple(token_ids)
        return list(token_ids), {}

    from jspace_llama.readout import read_workspace

    read_workspace(
        model=object(),
        dictionary=SimpleNamespace(
            model=SimpleNamespace(tokenizer=tokenizer)
        ),
        token_ids=prepared[0].token_ids,
        layers=[],
        activation_reader=capture_readout_ids,
    )
    assert readout_ids == prepared[0].token_ids

    expected = hashlib.sha256(b"[1,42]").hexdigest()
    provenance = __import__(
        "jspace_llama.provenance",
        fromlist=["build_experiment_provenance"],
    ).build_experiment_provenance(config, prepared, [1])
    assert provenance["prompts"][0]["token_ids_sha256"] == expected

    class Underlying:
        def __init__(self) -> None:
            self.ids: list[list[int]] = []

        def generate(self, ids, **kwargs):
            self.ids.append(ids[0].tolist())
            return torch.cat((ids, torch.tensor([[2]], device=ids.device)), dim=1)

    class Wrapped:
        def __init__(self) -> None:
            self._hf_model = Underlying()
            self.tokenizer = tokenizer
            self.input_device = torch.device("cpu")

    class NullHook(contextlib.nullcontext):
        pass

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(experiment, "MemoryBoundedLensSwap", lambda *a, **k: NullHook())
    monkeypatch.setattr(
        experiment,
        "NormMatchedRandomControl",
        lambda *a, **k: NullHook(),
    )
    try:
        model = Wrapped()
        model_runner = ModelExperimentRunner(model, object())
        for request in requests:
            model_runner(request)
        assert model._hf_model.ids == [[1, 42]] * 4
    finally:
        monkeypatch.undo()


def test_noop_reconstructs_coordinates_and_remains_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.interventions as interventions

    hidden = torch.tensor([[[2.0, -1.0]]], dtype=torch.bfloat16)
    layer = SwapLayer(
        vectors=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        pseudoinverse=torch.linalg.pinv(
            torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        ),
        noop=True,
    )
    calls = 0
    real_reconstruct = interventions._reconstruct_coordinates

    def track_reconstruct(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_reconstruct(*args, **kwargs)

    monkeypatch.setattr(
        interventions,
        "_reconstruct_coordinates",
        track_reconstruct,
    )
    output = apply_causal_clamp(hidden, layer, alpha=4.0)

    assert calls == 1
    assert output.dtype == hidden.dtype
    assert torch.equal(output, hidden)
    assert output.data_ptr() != hidden.data_ptr()


def _gated_error() -> GatedRepoError:
    response = SimpleNamespace(
        status_code=403,
        headers={},
        request=SimpleNamespace(url="https://huggingface.co/private/repo"),
    )
    return GatedRepoError(
        "private request id and repository details",
        response=response,
    )


def test_gated_model_failure_has_only_sanitized_stage_and_category(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline
    from jspace_llama.runtime import translate_huggingface_error

    translated = translate_huggingface_error(
        _gated_error(),
        stage="model_download",
    )
    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: (_ for _ in ()).throw(translated),
        base_provenance=task4.complete_provenance(),
    )

    assert outcome["reason"] == {
        "code": "asset_load_failed",
        "message": "model or lens asset loading failed",
        "stage": "model_download",
        "category": "gated_access_denied",
    }
    compact = json.dumps(outcome)
    assert "private request" not in compact
    assert "private/repo" not in compact


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (
            RevisionNotFoundError(
                "missing revision",
                response=SimpleNamespace(
                    status_code=404,
                    headers={},
                    request=SimpleNamespace(url="https://huggingface.co/repo"),
                ),
            ),
            "revision_not_found",
        ),
        (EntryNotFoundError("missing entry"), "asset_not_found"),
        (
            LocalEntryNotFoundError("missing local cache entry"),
            "local_cache_missing",
        ),
        (OfflineModeIsEnabled("offline"), "offline"),
    ],
)
def test_huggingface_exception_types_have_fixed_categories(
    error: BaseException,
    category: str,
) -> None:
    from jspace_llama.runtime import translate_huggingface_error

    translated = translate_huggingface_error(
        error,
        stage="model_download",
    )

    assert translated.category == category
    assert translated.stage == "model_download"


def test_generic_asset_failure_uses_fixed_fallback_metadata(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline
    from jspace_llama.runtime import AssetLoadError

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: (_ for _ in ()).throw(
            AssetLoadError("sensitive arbitrary failure")
        ),
        base_provenance=task4.complete_provenance(),
    )

    assert outcome["reason"]["stage"] == "asset_load"
    assert outcome["reason"]["category"] == "asset_failure"
    assert "sensitive arbitrary failure" not in json.dumps(outcome)


def test_validator_rejects_operational_metadata_on_non_asset_reason() -> None:
    from jspace_llama.artifacts import build_artifacts
    from jspace_llama.jsonio import document_sha256

    validator = load_script(
        VALIDATOR_PATH,
        "final_review_reason_metadata",
    )
    manifest, final = build_artifacts(
        run_id="failure",
        started_at="2026-11-10T20:00:00Z",
        ended_at="2026-11-10T20:01:00Z",
        terminal_status="expected_failure",
        provenance=task4.complete_provenance(),
        raw_diagnostics=None,
        reason_code="asset_load_failed",
        failure_metadata={
            "stage": "asset_load",
            "category": "asset_failure",
        },
    )
    tampered = {
        "code": "cuda_required",
        "message": "a CUDA runtime is required",
        "stage": "asset_load",
        "category": "asset_failure",
    }
    manifest["reason"] = final["reason"] = tampered
    final["manifest_sha256"] = document_sha256(manifest)

    with pytest.raises(
        validator.ResultsValidationError,
        match="operational metadata|schema validation",
    ):
        validator.validate_documents(
            manifest,
            final,
            schema_dir=SCHEMAS,
            project_root=PROJECT,
        )
