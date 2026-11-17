"""Tests for the v2-specific Qwen Jacobian-lens validator.

Valid fixtures are produced by the real v2 runtime (CPU fakes for the model, the
Jacobian and the lens), so the validator runs against the real artifact schemas.
CPU fixtures use a tiny layer contract and a ``results/v2`` layout, so tests opt
into ``allow_fixture_contract=True``; the validator and the CLI default to strict
production. Only the frozen v1 lens identity in the comparison artifact is
rewritten to the canonical hash (the fixture cannot reproduce the real file).

The suite is intentionally focused on realistic, high-impact failures. One
structural gate is tested once rather than once per field.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from refusal_direction_replication import qwen_jlens_runtime as runtime
from refusal_direction_replication.qwen_jlens import (
    FIT_COUNT,
    JLENS_V2_PROFILE,
    V2_FINAL_LENS_FILENAME,
    V2_FIT_COUNT_CANDIDATES,
    V2_MAX_FIT_COUNT,
    load_fit_checkpoint,
)
from refusal_direction_replication.qwen_jlens_v2_validate import (
    validate_qwen_jlens_v2,
)
from refusal_direction_replication.qwen_lab_readout import FROZEN_LENS_SHA256
from refusal_direction_replication.run_progress import EVENTS_FILENAME, STATUS_FILENAME
from test_qwen_jlens_runtime import IMPLEMENTATION_COMMIT
from test_qwen_jlens_v2_runtime import (
    _recording_jacobian,
    _rows,
    _run,
    _write_v2_fixture,
)

PROJECT = Path(__file__).resolve().parents[1]
PACKAGED_PROFILE = PROJECT / "configs" / JLENS_V2_PROFILE
SCRIPT = PROJECT / "scripts" / "validate_qwen_jlens_v2.py"
RESULTS = Path("results/v2")
ARTIFACTS = Path("artifacts/v2")
LIMIT_SECONDS = 12 * 3600
N_HOLDOUT = 200

_CACHE_ROOT: list[Path] = []
_CACHE: dict[str, Path] = {}


@pytest.fixture(scope="session", autouse=True)
def _tree_cache_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    _CACHE_ROOT.append(tmp_path_factory.mktemp("v2_validator_trees"))


def _validate(root: Path, **kwargs: Any) -> dict[str, bool]:
    kwargs.setdefault("allow_fixture_contract", True)
    return validate_qwen_jlens_v2(root, **kwargs)


def _oom_jacobian(*_args: Any, **_kwargs: Any) -> Any:
    raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_pristine(root: Path, kind: str) -> None:
    config = _write_v2_fixture(
        root,
        candidates=list(V2_FIT_COUNT_CANDIDATES),
        holdout_count=N_HOLDOUT,
        max_seq_len=8,
        max_projected_hours=12,
    )
    jacobian = _recording_jacobian([])
    probe_seconds = float(LIMIT_SECONDS // 200)  # selects 200 prompts
    if kind == "projected_runtime":
        probe_seconds = float(LIMIT_SECONDS // min(V2_FIT_COUNT_CANDIDATES) + 1)
    elif kind == "dim_batch_oom":
        jacobian = _oom_jacobian
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT
        )
        result = _run(
            root,
            config.name,
            rows=_rows(V2_MAX_FIT_COUNT + N_HOLDOUT),
            jacobian_fn=jacobian,
            probe_seconds=probe_seconds,
        )
    assert result["status"] == ("pass" if kind == "pass" else "aborted")
    if kind == "pass":
        assert result["n_fit"] == 200

        def canonical_v1(payload: dict[str, Any]) -> None:
            payload["results"]["v1_lens"]["sha256"] = FROZEN_LENS_SHA256
            payload["results"]["v1_lens"]["n_prompts"] = FIT_COUNT

        _rewrite_json(root / RESULTS / "04_jlens_compare.json", canonical_v1)


def _write_tree(dest: Path, kind: str = "pass") -> None:
    if kind not in _CACHE:
        pristine = _CACHE_ROOT[0] / f"tree_{kind}"
        pristine.mkdir()
        _build_pristine(pristine, kind)
        _CACHE[kind] = pristine
    shutil.copytree(_CACHE[kind], dest, dirs_exist_ok=True)


def _res(root: Path, name: str) -> Path:
    return root / RESULTS / name


def _art(root: Path, name: str) -> Path:
    return root / ARTIFACTS / name


def _edit_results(root: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    _rewrite_json(_res(root, name), lambda payload: mutate(payload["results"]))


def _edit_profile(root: Path, **overrides: Any) -> None:
    _rewrite_json(
        root / "configs" / JLENS_V2_PROFILE, lambda payload: payload.update(overrides)
    )


def _edit_checkpoint(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = _art(root, "jlens_fit_checkpoint.pt")
    payload = load_fit_checkpoint(path)
    mutate(payload)
    torch.save(payload, path)


def _append_corpus_text(root: Path, text: str) -> None:
    corpus = _art(root, "corpus.jsonl")
    corpus.write_text(
        corpus.read_text(encoding="utf-8")
        + json.dumps({"split": "fit", "fit_text": text}) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Accepted runs
# --------------------------------------------------------------------------


def test_valid_pass_tree_is_accepted_without_the_v1_project_tree(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    shutil.rmtree(tmp_path / "results" / "qwen25_coder_7b")
    shutil.rmtree(tmp_path / "results" / "qwen25_coder_7b_dose")
    shutil.rmtree(tmp_path / "artifacts" / "qwen25_coder_7b_jlens")
    checks = _validate(tmp_path)
    assert checks and all(checks.values())
    assert {
        "selected_fit_count_candidate",
        "holdout_count",
        "v1_lens_canonical_hash",
        "comparison_artifact",
        "no_raw_text_leakage",
        "production_layer_contract",
    } <= set(checks)


def test_valid_aborted_preselection_tree_is_accepted(tmp_path: Path) -> None:
    _write_tree(tmp_path, "projected_runtime")
    prepare = json.loads(_res(tmp_path, "00_jlens_prepare.json").read_text("utf-8"))
    assert prepare["results"]["fit_count"] == V2_MAX_FIT_COUNT  # full 800 + 200 pool
    assert all(_validate(tmp_path).values())


# --------------------------------------------------------------------------
# Candidate count, holdout count and cross-artifact agreement
# --------------------------------------------------------------------------


def test_non_candidate_fit_count_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _rewrite_json(_res(tmp_path, "jlens_evaluation.json"), lambda p: p.update(n_fit=175))
    with pytest.raises(ValueError, match="fit count candidate"):
        _validate(tmp_path)


def test_holdout_count_other_than_200_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"), lambda p: p.update(n_holdout=199)
    )
    with pytest.raises(ValueError, match="holdout count"):
        _validate(tmp_path)


SELECTED_COUNT_TAMPERS: dict[str, Callable[[Path], None]] = {
    "probe": lambda root: _edit_results(
        root, "01_jlens_probe.json", lambda r: r.update(selected_fit_count=150)
    ),
    "checkpoint": lambda root: _edit_checkpoint(
        root, lambda payload: payload.update(fit_count=150)
    ),
}


@pytest.mark.parametrize("source", sorted(SELECTED_COUNT_TAMPERS))
def test_disagreeing_selected_counts_are_rejected(tmp_path: Path, source: str) -> None:
    _write_tree(tmp_path, "pass")
    SELECTED_COUNT_TAMPERS[source](tmp_path)
    with pytest.raises(ValueError, match="selected fit count"):
        _validate(tmp_path)


def test_corpus_hash_that_does_not_match_the_items_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _edit_results(
        tmp_path,
        "00_jlens_prepare.json",
        lambda r: r["fit"][0].update(token_count=r["fit"][0]["token_count"] + 1),
    )
    with pytest.raises(ValueError, match="corpus hash"):
        _validate(tmp_path)


# --------------------------------------------------------------------------
# Missing artifacts, comparison and lens identity
# --------------------------------------------------------------------------


def test_pass_without_comparison_artifact_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _res(tmp_path, "04_jlens_compare.json").unlink()
    with pytest.raises(ValueError, match="comparison artifact"):
        _validate(tmp_path)


def test_pass_without_checkpoint_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _art(tmp_path, "jlens_fit_checkpoint.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        _validate(tmp_path)


def test_pass_without_v2_lens_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _art(tmp_path, V2_FINAL_LENS_FILENAME).unlink()
    with pytest.raises(ValueError, match="v2 lens"):
        _validate(tmp_path)


def test_comparison_v1_hash_must_be_the_canonical_frozen_hash(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    tiny_v1 = tmp_path / "artifacts" / "qwen25_coder_7b_jlens" / (
        "qwen25_coder_7b_jacobian_lens.pt"
    )
    digest = hashlib.sha256(tiny_v1.read_bytes()).hexdigest()
    assert digest != FROZEN_LENS_SHA256
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda r: r["v1_lens"].update(sha256=digest),
    )
    with pytest.raises(ValueError, match="v1 lens sha256"):
        _validate(tmp_path)


def test_comparison_holdout_ids_must_equal_the_holdout_artifact(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")

    def swap(results: dict[str, Any]) -> None:
        ids = results["shared_holdout_ids"]
        ids[0], ids[1] = ids[1], ids[0]

    _edit_results(tmp_path, "04_jlens_compare.json", swap)
    with pytest.raises(ValueError, match="shared_holdout_ids"):
        _validate(tmp_path)


def test_v2_lens_hash_must_match_the_file(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"), lambda p: p.update(lens_sha256="1" * 64)
    )
    with pytest.raises(ValueError, match="v2 lens"):
        _validate(tmp_path)


def test_v2_lens_with_wrong_shape_is_rejected(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    path = _art(tmp_path, V2_FINAL_LENS_FILENAME)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["J"][0] = torch.zeros(3, 3, dtype=torch.float16)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    _edit_results(
        tmp_path, "02_jlens_fit.json", lambda r: r.update(lens_sha256=digest, lens_bytes=size)
    )
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"),
        lambda p: p.update(lens_sha256=digest, lens_bytes=size),
    )
    with pytest.raises(ValueError, match="v2 lens"):
        _validate(tmp_path)


# --------------------------------------------------------------------------
# Raw text: leaks are caught, schema identifiers never false-positive
# --------------------------------------------------------------------------

LEAKS: dict[str, Callable[[Path], None]] = {
    "code_value": lambda root: _edit_results(
        root, "04_jlens_compare.json", lambda r: r.update(note="def f():\n    return 1\n")
    ),
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_raw_text_in_keys_or_values_is_rejected(tmp_path: Path, leak: str) -> None:
    _write_tree(tmp_path, "pass")
    LEAKS[leak](tmp_path)
    with pytest.raises(ValueError, match="raw text"):
        _validate(tmp_path)


REALISTIC_CORPUS_TEXT = '''def collect_status(payload, store):
    """Gather the run status from a payload."""
    error_message = payload.get("error_message")
    schema_version = payload["schema_version"]
    content_hash = store.content_hash(payload)
    current_stage = payload.get("current_stage", "prepare")
    completed_units = payload.get("completed_units", 0)
    status = "completed" if current_stage == "holdout" else "pass"
    gate = "projected_runtime" if error_message else "dim_batch_oom"
    return schema_version, content_hash, completed_units, status, gate, "cuda_oom"
'''


@pytest.mark.parametrize("kind", ["pass", "dim_batch_oom"])
def test_realistic_corpus_with_schema_identifiers_does_not_flag_valid_artifacts(
    tmp_path: Path, kind: str
) -> None:
    """Keys such as ``error_message`` or ``content_hash`` occur in real code."""
    _write_tree(tmp_path, kind)
    _append_corpus_text(tmp_path, REALISTIC_CORPUS_TEXT)
    assert all(_validate(tmp_path).values())


def test_long_fragments_copied_from_the_corpus_are_flagged_but_short_ones_are_not(
    tmp_path: Path,
) -> None:
    _write_tree(tmp_path, "pass")
    sentence = "Compute the total of all supplied values"
    _append_corpus_text(tmp_path, f"x = 1\n# {sentence}\n")
    _edit_results(tmp_path, "02_jlens_fit.json", lambda r: r.update(note="Compute the total"))
    assert all(_validate(tmp_path).values())
    _edit_results(tmp_path, "02_jlens_fit.json", lambda r: r.update(note=sentence))
    with pytest.raises(ValueError, match="raw text"):
        _validate(tmp_path)


# --------------------------------------------------------------------------
# Strict production contract of the root profile
# --------------------------------------------------------------------------


def _root_with_packaged_profile(root: Path, **overrides: Any) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    payload = json.loads(PACKAGED_PROFILE.read_text(encoding="utf-8"))
    payload.update(overrides)
    (root / "configs" / JLENS_V2_PROFILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def test_strict_validation_rejects_a_fixture_profile_by_default(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    with pytest.raises(ValueError, match="production contract"):
        validate_qwen_jlens_v2(tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("d_model", 3585),
        ("results_dir", "results/elsewhere"),
    ],
)
def test_strict_validation_rejects_a_divergent_root_profile(
    tmp_path: Path, key: str, value: Any
) -> None:
    _root_with_packaged_profile(tmp_path, **{key: value})
    with pytest.raises(ValueError, match=f"production contract.*{key}"):
        validate_qwen_jlens_v2(tmp_path)


def test_strict_validation_accepts_the_packaged_profile_contract(tmp_path: Path) -> None:
    _root_with_packaged_profile(tmp_path)
    with pytest.raises(ValueError) as caught:  # no run results exist in this root
        validate_qwen_jlens_v2(tmp_path)
    assert "production contract" not in str(caught.value)
    assert "jlens_evaluation.json is missing" in str(caught.value)


# --------------------------------------------------------------------------
# Path confinement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("results_dir", "ABSOLUTE", "absolute"),
        ("artifacts_dir", "../escape", "traversal"),
        ("source_v1_lens", "ABSOLUTE", "absolute"),
    ],
)
def test_profile_paths_must_stay_inside_the_run_root(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    _write_tree(tmp_path, "pass")
    if value == "ABSOLUTE":
        value = str((tmp_path.parent / "elsewhere").resolve())
    _edit_profile(tmp_path, **{key: value})
    with pytest.raises(ValueError, match=message):
        _validate(tmp_path)


@pytest.mark.parametrize(
    "recorded",
    ["artifacts/other/qwen25_coder_7b_jacobian_lens_v2.pt"],
)
def test_fit_lens_must_be_the_exact_artifacts_dir_path(
    tmp_path: Path, recorded: str
) -> None:
    _write_tree(tmp_path, "pass")
    _edit_results(tmp_path, "02_jlens_fit.json", lambda r: r.update(lens=recorded))
    with pytest.raises(ValueError, match="v2 lens"):
        _validate(tmp_path)


# --------------------------------------------------------------------------
# Malformed input becomes ValueError
# --------------------------------------------------------------------------


def test_malformed_artifact_shapes_raise_value_error_not_key_error(tmp_path: Path) -> None:
    _write_tree(tmp_path, "pass")
    _edit_results(tmp_path, "01_jlens_probe.json", lambda r: r.update(attempts=[7]))
    with pytest.raises(ValueError, match="malformed"):
        _validate(tmp_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("validate_qwen_jlens_v2_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_smoke_is_strict_by_default_and_never_calls_v1_validators(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    good = tmp_path / "good"
    _write_tree(good, "pass")
    assert module.main(good, allow_fixture_contract=True) == 0
    assert "PASS selected_fit_count_candidate" in capsys.readouterr().out
    assert module.main(good) == 1  # strict: the tiny fixture contract is refused
    assert "production contract" in capsys.readouterr().out
    source = SCRIPT.read_text(encoding="utf-8")
    assert "qwen_jlens_validate" not in source
    assert "qwen_pilot_validate" not in source
    assert "qwen_dose_validate" not in source
    assert "allow_fixture_contract" not in source.split("if __name__", 1)[1]
