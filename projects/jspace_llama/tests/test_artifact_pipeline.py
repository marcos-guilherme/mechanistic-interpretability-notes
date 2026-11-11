from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import jsonschema
import pytest

from jspace_llama.config import ExperimentConfig
from jspace_llama.runtime import AssetLoadError


ROOT = Path(__file__).parents[3]
PROJECT = ROOT / "projects" / "jspace_llama"
SCHEMAS = PROJECT / "schemas"
VALIDATOR_PATH = PROJECT / "scripts" / "validate_results.py"
RUNNER_PATH = ROOT / "tools" / "colab" / "run_jspace.py"
JSPACE_COMMIT = "04c822a255fe0a10d84e08ec499e5e09f4b7d37a"
SOURCE_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_provenance() -> dict:
    from jspace_llama.provenance import runtime_source_sha256

    prompt_cases = ExperimentConfig().prompts.cases
    return {
        "source": {
            "commit": SOURCE_COMMIT,
            "dirty": False,
            "scoped_dirty": False,
            "archive_sha256": runtime_source_sha256(PROJECT, ROOT),
        },
        "model": {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "resolved_revision": hashlib.sha1(b"model revision").hexdigest(),
            "layer_count": 32,
        },
        "lens": {
            "repo_id": "neuronpedia/jacobian-lens",
            "filename": (
                "llama3.1-8b-it/jlens/Salesforce-wikitext/"
                "Llama-3.1-8B-Instruct_jacobian_lens.pt"
            ),
            "resolved_revision": hashlib.sha1(b"lens revision").hexdigest(),
            "sha256": hashlib.sha256(b"lens bytes").hexdigest(),
            "size_bytes": 1234,
            "integrity_verified": True,
        },
        "software": {
            "python": "3.11.9",
            "pytorch": "2.8.0",
            "transformers": "4.56.2",
            "huggingface_hub": "0.34.4",
            "jspace": "0.1.0",
            "jspace_commit": JSPACE_COMMIT,
        },
        "hardware": {
            "gpu_name": "NVIDIA L4",
            "cuda_version": "12.6",
            "dtype": "bfloat16",
        },
        "experiment": {
            "seed": 42,
            "prompts": [
                {
                    "name": case.name,
                    "text": case.text,
                    "instruction_sha256": hashlib.sha256(
                        case.text.encode()
                    ).hexdigest(),
                    "rendered_prompt_sha256": hashlib.sha256(
                        f"<chat>{case.text}</chat>".encode()
                    ).hexdigest(),
                    "token_ids_sha256": hashlib.sha256(
                        b"[1,2]"
                    ).hexdigest(),
                }
                for case in prompt_cases
            ],
            "generation": {
                "max_new_tokens": 12,
                "do_sample": False,
                "decoding": "greedy",
                "first_line": False,
            },
            "layers": [12, 13],
            "alphas": [4.0],
            "intervention": {
                "source_concept": "spider",
                "target_concept": "ant",
                "prefill_only": True,
                "mode": "source_to_target_clamp",
                "random_control": "norm_matched_orthogonal",
            },
        },
    }


def classification(
    *,
    eight: bool = False,
    six: bool = False,
    ant: bool = False,
    empty: bool = False,
    repetition: bool = False,
) -> dict:
    value = (
        "six"
        if six and not eight
        else "eight" if eight and not six else None
    )
    status = (
        "unique"
        if value is not None
        else "conflicting" if six and eight else "no_answer"
    )
    return {
        "answer_evidence": {"eight": eight, "six": six},
        "answer_parse": {"status": status, "value": value},
        "target_entity_leakage": ant,
        "degradation": {
            "empty": empty,
            "repetition": repetition,
            "incoherent": False,
        },
    }


def successful_raw() -> dict:
    observations = []
    classes = {
        "baseline": classification(eight=True),
        "noop": classification(eight=True),
        "random": classification(eight=True),
        "target": classification(six=True),
    }
    for prompt in (
        "main",
        "paraphrase_animal",
        "paraphrase_creature",
    ):
        for condition, classified in classes.items():
            observations.append(
                {
                    "prompt": {"name": prompt},
                    "parameters": {"alpha": 4.0},
                    "condition": condition,
                    "classification": copy.deepcopy(classified),
                    "continuation": f"raw {prompt} {condition}",
                }
            )
    return {
        "readout": {"spider": {"present": True}, "slots": [{"raw": "slot"}]},
        "experiment": {"observations": observations},
    }


def observation(raw: dict, condition: str, prompt: str = "main") -> dict:
    return next(
        item
        for item in raw["experiment"]["observations"]
        if item["prompt"]["name"] == prompt and item["condition"] == condition
    )


def build_success_artifacts():
    from jspace_llama.artifacts import build_artifacts

    return build_artifacts(
        run_id="run-001",
        started_at="2026-11-10T20:00:00Z",
        ended_at="2026-11-10T20:01:00Z",
        terminal_status="completed",
        provenance=complete_provenance(),
        raw_diagnostics=successful_raw(),
    )


def test_task4_modules_schemas_and_validator_exist() -> None:
    required = [
        PROJECT / "src" / "jspace_llama" / "artifacts.py",
        PROJECT / "src" / "jspace_llama" / "evaluation.py",
        PROJECT / "src" / "jspace_llama" / "provenance.py",
        PROJECT / "src" / "jspace_llama" / "pipeline.py",
        PROJECT / "src" / "jspace_llama" / "jsonio.py",
        SCHEMAS / "execution_manifest.schema.json",
        SCHEMAS / "final_evaluation.schema.json",
        VALIDATOR_PATH,
    ]

    assert all(path.is_file() for path in required)


def test_success_requires_all_primary_gates_and_complete_provenance() -> None:
    manifest, final = build_success_artifacts()

    assert manifest["status"] == "success"
    assert final["status"] == "success"
    assert final["primary"]["passing_alphas"] == [4.0]
    assert final["primary"]["alphas"][0]["passed"] is True
    assert final["primary"]["readout"]["spider_present"] is True
    assert final["paraphrases"]["prompt_count"] == 2
    assert final["paraphrases"]["passing_prompt_alpha_count"] == 2
    assert len(final["paraphrases"]["prompt_alphas"]) == 2


def test_provenance_records_exact_prompts_resolved_revisions_and_file_hash(
    tmp_path: Path,
) -> None:
    from jspace_llama.provenance import (
        build_experiment_provenance,
        file_sha256,
        resolve_huggingface_revisions,
    )

    config = ExperimentConfig()
    prompt = SimpleNamespace(
        name="main",
        instruction="Exact unrendered prompt",
        rendered="<chat>Exact unrendered prompt</chat>",
        token_ids=(1, 2),
    )
    experiment = build_experiment_provenance(config, [prompt], [12, 13])
    assert experiment["prompts"][0]["text"] == "Exact unrendered prompt"
    assert experiment["prompts"][0]["rendered_prompt_sha256"] != (
        experiment["prompts"][0]["instruction_sha256"]
    )

    class FakeApi:
        def model_info(self, repo_id):
            return SimpleNamespace(sha=hashlib.sha1(repo_id.encode()).hexdigest())

    model_revision, lens_revision = resolve_huggingface_revisions(
        config.assets.model_id,
        config.assets.lens_repo_id,
        api=FakeApi(),
    )
    assert model_revision == hashlib.sha1(
        config.assets.model_id.encode()
    ).hexdigest()
    assert lens_revision == hashlib.sha1(
        config.assets.lens_repo_id.encode()
    ).hexdigest()
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"pinned lens")
    digest, size = file_sha256(lens)
    assert digest == (
        "8a79e44f3b87868447e8b0e514e4d9e3974591f3414ed8064a54ca49a7702709"
    )
    assert size == 11


@pytest.mark.parametrize(
    "relative",
    [
        "data/anthropic-jlens/LICENSE",
        "data/anthropic-jlens/README.md",
        "data/anthropic-jlens/evaluations/lens-eval-multihop.json",
    ],
)
def test_runtime_source_digest_covers_observational_data(
    tmp_path: Path,
    relative: str,
) -> None:
    from jspace_llama.provenance import runtime_source_sha256

    repository = tmp_path / "repo"
    project = repository / "projects" / "jspace_llama"
    source = project / "src" / "jspace_llama" / "core.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    for data_relative in (
        "data/anthropic-jlens/LICENSE",
        "data/anthropic-jlens/README.md",
        "data/anthropic-jlens/evaluations/lens-eval-multihop.json",
    ):
        path = project / data_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data_relative, encoding="utf-8")

    before = runtime_source_sha256(project, repository)
    (project / relative).write_text("changed", encoding="utf-8")

    assert runtime_source_sha256(project, repository) != before


def test_historical_causal_manifest_remains_valid() -> None:
    validator = load_script(VALIDATOR_PATH, "validate_historical_causal")
    manifest = json.loads(
        (PROJECT / "results" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    final = json.loads(
        (PROJECT / "results" / "final_evaluation.json").read_text(
            encoding="utf-8"
        )
    )

    validator.validate_documents(
        manifest,
        final,
        schema_dir=SCHEMAS,
        project_root=PROJECT,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", hashlib.sha1(b"tampered historical commit").hexdigest()),
        ("archive_sha256", hashlib.sha256(b"tampered source").hexdigest()),
    ],
)
def test_validator_rejects_tampered_historical_source(
    field: str,
    value: str,
) -> None:
    from jspace_llama.jsonio import document_sha256

    validator = load_script(
        VALIDATOR_PATH,
        f"validate_tampered_historical_{field}",
    )
    manifest = json.loads(
        (PROJECT / "results" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    final = json.loads(
        (PROJECT / "results" / "final_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["provenance"]["source"][field] = value
    final["manifest_sha256"] = document_sha256(manifest)

    with pytest.raises(
        validator.ResultsValidationError,
        match="source (commit|archive digest)",
    ):
        validator.validate_documents(
            manifest,
            final,
            schema_dir=SCHEMAS,
            project_root=PROJECT,
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda raw: raw["readout"]["spider"].update(present=False),
            "readout_spider_missing",
        ),
        (
            lambda raw: observation(raw, "baseline")["classification"][
                "answer_parse"
            ].update(status="no_answer", value=None),
            "baseline_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "baseline")["classification"][
                "degradation"
            ].update(empty=True),
            "baseline_degraded",
        ),
        (
            lambda raw: observation(raw, "noop")["classification"][
                "answer_parse"
            ].update(status="no_answer", value=None),
            "noop_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "noop")["classification"][
                "answer_parse"
            ].update(value="six"),
            "noop_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "random")["classification"][
                "answer_parse"
            ].update(value="six"),
            "random_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "random")["classification"][
                "degradation"
            ].update(repetition=True),
            "random_degraded",
        ),
        (
            lambda raw: observation(raw, "target")["classification"][
                "answer_parse"
            ].update(status="no_answer", value=None),
            "target_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "target")["classification"][
                "answer_parse"
            ].update(value="eight"),
            "target_answer_invalid",
        ),
        (
            lambda raw: observation(raw, "target")["classification"].update(
                target_entity_leakage=True
            ),
            "target_contains_ant",
        ),
        (
            lambda raw: observation(raw, "target")["classification"][
                "degradation"
            ].update(empty=True),
            "target_degraded",
        ),
        (
            lambda raw: observation(raw, "target")["classification"][
                "degradation"
            ].update(repetition=True),
            "target_degraded",
        ),
    ],
)
def test_every_primary_gate_failure_is_inconclusive(mutate, reason: str) -> None:
    from jspace_llama.artifacts import build_artifacts

    raw = successful_raw()
    mutate(raw)

    manifest, final = build_artifacts(
        run_id="run-failure",
        started_at="2026-11-10T20:00:00Z",
        ended_at="2026-11-10T20:01:00Z",
        terminal_status="completed",
        provenance=complete_provenance(),
        raw_diagnostics=raw,
    )

    assert manifest["status"] == "inconclusive"
    assert final["status"] == "inconclusive"
    assert reason in final["primary"]["alphas"][0]["reasons"]


def test_incomplete_provenance_prevents_success() -> None:
    from jspace_llama.artifacts import build_artifacts

    provenance = complete_provenance()
    provenance["model"]["resolved_revision"] = None
    manifest, final = build_artifacts(
        run_id="run-partial",
        started_at="2026-11-10T20:00:00Z",
        ended_at="2026-11-10T20:01:00Z",
        terminal_status="completed",
        provenance=provenance,
        raw_diagnostics=successful_raw(),
    )

    assert manifest["provenance_complete"] is False
    assert manifest["status"] == final["status"] == "inconclusive"
    assert final["reason"]["code"] == "incomplete_provenance"


def test_compact_artifacts_have_no_raw_diagnostic_material() -> None:
    manifest, final = build_success_artifacts()
    compact = json.dumps([manifest, final]).lower()

    for forbidden in (
        "continuation",
        '"rendered_prompt":',
        '"instruction"',
        '"slots"',
        '"token"',
        "raw main",
        "c:\\",
        "/content/",
    ):
        assert forbidden not in compact


def test_atomic_json_write_preserves_old_destination_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jspace_llama import jsonio

    destination = tmp_path / "result.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(jsonio.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        jsonio.atomic_write_json(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == '{"old":true}\n'
    assert list(tmp_path.glob(".*.tmp")) == []


def test_pipeline_writes_expected_failure_artifacts_but_reraises_bugs(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    def expected_failure():
        raise AssetLoadError("secret path C:\\cache\\model")

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=expected_failure,
        base_provenance=complete_provenance(),
    )
    assert outcome["status"] == "inconclusive"
    final = json.loads(
        (tmp_path / "results" / "final_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert final["reason"] == {
        "code": "asset_load_failed",
        "message": "model or lens asset loading failed",
        "stage": "asset_load",
        "category": "asset_failure",
    }
    validator = load_script(VALIDATOR_PATH, "validate_expected_failure")
    manifest = json.loads(
        (tmp_path / "results" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    validator.validate_documents(manifest, final, schema_dir=SCHEMAS)

    bug_root = tmp_path / "bug"
    stale_final = bug_root / "results" / "final_evaluation.json"
    stale_final.parent.mkdir(parents=True)
    stale_final.write_text('{"stale":true}', encoding="utf-8")

    def programming_bug():
        raise TypeError("unexpected bug")

    with pytest.raises(TypeError, match="unexpected bug"):
        run_pipeline(
            project_root=bug_root,
            execute=programming_bug,
            base_provenance=complete_provenance(),
        )
    manifest = json.loads(
        (bug_root / "results" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "error"
    assert manifest["terminal_status"] == "unexpected_error"
    assert not stale_final.exists()


def test_pipeline_separates_raw_and_compact_artifacts(tmp_path: Path) -> None:
    from jspace_llama.pipeline import run_pipeline

    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: {
            "provenance": complete_provenance(),
            **successful_raw(),
        },
    )

    assert outcome["status"] == "success"
    raw_path = tmp_path / "artifacts" / "readout_and_causal_swap.json"
    assert "continuation" in raw_path.read_text(encoding="utf-8")
    assert "continuation" not in (
        tmp_path / "results" / "final_evaluation.json"
    ).read_text(encoding="utf-8")


def test_missing_readout_becomes_expected_inconclusive_artifacts(
    tmp_path: Path,
) -> None:
    from jspace_llama.pipeline import run_pipeline

    raw = successful_raw()
    raw.pop("readout")
    outcome = run_pipeline(
        project_root=tmp_path,
        execute=lambda: {
            "provenance": complete_provenance(),
            **raw,
        },
    )

    assert outcome["status"] == "inconclusive"
    assert outcome["reason"]["code"] == "evaluation_data_invalid"


def test_schemas_accept_valid_artifacts() -> None:
    manifest, final = build_success_artifacts()

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


@pytest.mark.parametrize(
    "tamper",
    [
        lambda manifest, final: final.update(status="inconclusive"),
        lambda manifest, final: final["primary"].update(passing_alpha_count=0),
        lambda manifest, final: final.update(manifest_sha256="0" * 64),
        lambda manifest, final: manifest["provenance"]["lens"].update(
            sha256="0" * 64
        ),
        lambda manifest, final: manifest["provenance"]["model"].update(
            resolved_revision=None
        ),
    ],
)
def test_validator_rejects_tampered_claims_hashes_and_provenance(tamper) -> None:
    validator = load_script(VALIDATOR_PATH, "validate_results_tamper")
    manifest, final = build_success_artifacts()
    tamper(manifest, final)

    with pytest.raises(validator.ResultsValidationError):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("manifest", "cache_path", "/content/.cache/model"),
        ("manifest", "credential", "hf_secret"),
        ("final", "continuation", "six"),
        ("final", "rendered_prompt", "<chat>secret</chat>"),
        ("final", "token_ids", [1, 2, 3]),
        ("final", "model_path", r"C:\models\llama"),
    ],
)
def test_validator_rejects_forbidden_fields_and_local_paths(
    target: str, key: str, value
) -> None:
    validator = load_script(VALIDATOR_PATH, f"validate_results_{key}")
    manifest, final = build_success_artifacts()
    document = manifest if target == "manifest" else final
    document[key] = value

    with pytest.raises(validator.ResultsValidationError, match="forbidden|path"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def test_validator_cli_prints_pass_and_fail(tmp_path: Path) -> None:
    manifest, final = build_success_artifacts()
    results = tmp_path / "results"
    results.mkdir()
    (results / "execution_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (results / "final_evaluation.json").write_text(
        json.dumps(final), encoding="utf-8"
    )

    passed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--results-dir",
            str(results),
            "--schema-dir",
            str(SCHEMAS),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert passed.returncode == 0
    assert passed.stdout.startswith("PASS:")

    final["status"] = "inconclusive"
    (results / "final_evaluation.json").write_text(
        json.dumps(final), encoding="utf-8"
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--results-dir",
            str(results),
            "--schema-dir",
            str(SCHEMAS),
        ],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert failed.stdout.startswith("FAIL:")


def test_validator_recomputes_prompt_hash_and_completed_status() -> None:
    from jspace_llama.jsonio import document_sha256

    validator = load_script(VALIDATOR_PATH, "validate_results_recompute")
    manifest, final = build_success_artifacts()
    manifest["provenance"]["experiment"]["prompts"][0][
        "instruction_sha256"
    ] = hashlib.sha256(b"tampered").hexdigest()
    final["manifest_sha256"] = document_sha256(manifest)
    with pytest.raises(validator.ResultsValidationError, match="prompt hash"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)

    manifest, final = build_success_artifacts()
    reason = {"code": "fake", "message": "tampered scientific status"}
    manifest["status"] = final["status"] = "inconclusive"
    manifest["reason"] = final["reason"] = reason
    final["manifest_sha256"] = document_sha256(manifest)
    with pytest.raises(validator.ResultsValidationError, match="status"):
        validator.validate_documents(manifest, final, schema_dir=SCHEMAS)


def test_archive_embeds_source_commit_and_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_script(RUNNER_PATH, "run_jspace_provenance")
    project = tmp_path / "repo" / "projects" / "jspace_llama"
    for relative in ("README.md", "pyproject.toml", "uv.lock"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    commands = []
    repository = tmp_path / "repo"
    committed = {
        path.relative_to(repository).as_posix(): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }

    def fake_git(command, **kwargs):
        commands.append(command)
        if "rev-parse" in command:
            stdout = "9" * 40 + "\n"
        elif "ls-tree" in command:
            stdout = ("\n".join(sorted(committed)) + "\n").encode()
        elif "show" in command:
            stdout = committed[command[-1].split(":", 1)[1]]
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(runner.subprocess, "run", fake_git)
    archive = runner.build_archive(
        project=project,
        staging=tmp_path / "staging",
        repository_root=tmp_path / "repo",
    )

    with tarfile.open(archive, "r:gz") as handle:
        names = {member.name for member in handle.getmembers()}
        source = json.load(
            handle.extractfile("jspace_llama/source_provenance.json")
        )
    assert source == {
        "schema_version": 1,
        "commit": "9" * 40,
        "dirty": False,
        "scoped_dirty": False,
        "archive_sha256": source["archive_sha256"],
        "runtime_source_sha256": source["runtime_source_sha256"],
    }
    assert "jspace_llama/source_provenance.json" in names
    assert any("status" in command for command in commands)
    assert all(str(tmp_path) not in json.dumps(source) for _ in [0])


def test_notebook_calls_pipeline_and_has_no_duplicate_logic_or_outputs() -> None:
    import nbformat

    notebook = nbformat.read(
        PROJECT / "notebooks" / "01_readout_and_causal_swap.ipynb",
        as_version=4,
    )
    sources = "\n".join(cell.source for cell in notebook.cells)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "run_pipeline" in sources
    assert "run_experiment" not in sources
    assert "read_workspace" not in sources
    assert "download_lens_file" not in sources
    assert "write_text" not in sources
    assert "def " not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
