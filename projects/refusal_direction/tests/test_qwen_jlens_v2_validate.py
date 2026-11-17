"""Tests for the v2-specific Qwen Jacobian-lens validator.

Valid fixtures are produced by the real v2 runtime (CPU fakes for the model, the
Jacobian and the lens), so the validator is exercised against the real artifact
schemas rather than a hand-copied approximation. Only the frozen v1 lens
identity in the comparison artifact is rewritten to the canonical hash, because
the CPU fixture cannot reproduce the real 693 MB frozen file.
"""

from __future__ import annotations

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

RESULTS = Path("results/v2")
ARTIFACTS = Path("artifacts/v2")
LIMIT_SECONDS = 12 * 3600
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_qwen_jlens_v2.py"

_CACHE_ROOT: list[Path] = []
_CACHE: dict[tuple[Any, ...], Path] = {}


@pytest.fixture(scope="session", autouse=True)
def _tree_cache_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    _CACHE_ROOT.append(tmp_path_factory.mktemp("v2_validator_trees"))


def _probe_seconds_for(n_fit: int) -> float:
    """Probe time that makes the runtime select exactly ``n_fit`` prompts."""
    return float(LIMIT_SECONDS // n_fit)


def _oom_jacobian(*_args: Any, **_kwargs: Any) -> Any:
    raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def _rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_pristine(
    root: Path,
    *,
    n_fit: int,
    n_holdout: int,
    abort: str | None,
) -> None:
    candidates = sorted({*V2_FIT_COUNT_CANDIDATES, n_fit})
    config = _write_v2_fixture(
        root,
        candidates=candidates,
        holdout_count=n_holdout,
        max_seq_len=8,
        max_projected_hours=12,
    )
    jacobian = _recording_jacobian([])
    probe_seconds = _probe_seconds_for(n_fit)
    if abort == "projected_runtime":
        probe_seconds = float(LIMIT_SECONDS // min(candidates) + 1)
    elif abort == "dim_batch_oom":
        jacobian = _oom_jacobian
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            runtime, "implementation_commit_hash", lambda: IMPLEMENTATION_COMMIT
        )
        result = _run(
            root,
            config.name,
            rows=_rows(V2_MAX_FIT_COUNT + n_holdout),
            jacobian_fn=jacobian,
            probe_seconds=probe_seconds,
        )
    expected_status = "aborted" if abort else "pass"
    assert result["status"] == expected_status
    if not abort:
        assert result["n_fit"] == n_fit

        def canonical_v1(payload: dict[str, Any]) -> None:
            payload["results"]["v1_lens"]["sha256"] = FROZEN_LENS_SHA256
            payload["results"]["v1_lens"]["n_prompts"] = FIT_COUNT

        _rewrite_json(root / RESULTS / "04_jlens_compare.json", canonical_v1)


def _write_tree(
    dest: Path, *, n_fit: int = 200, n_holdout: int = 200, abort: str | None = None
) -> None:
    key = (n_fit, n_holdout, abort)
    if key not in _CACHE:
        pristine = _CACHE_ROOT[0] / f"tree_{len(_CACHE)}"
        pristine.mkdir()
        _build_pristine(pristine, n_fit=n_fit, n_holdout=n_holdout, abort=abort)
        _CACHE[key] = pristine
    shutil.copytree(_CACHE[key], dest, dirs_exist_ok=True)


def _write_valid_v2_tree(dest: Path, *, n_fit: int, n_holdout: int) -> None:
    _write_tree(dest, n_fit=n_fit, n_holdout=n_holdout)


def _write_aborted_v2_tree(dest: Path, *, gate: str = "projected_runtime") -> None:
    _write_tree(dest, abort=gate)


def _res(root: Path, name: str) -> Path:
    return root / RESULTS / name


def _art(root: Path, name: str) -> Path:
    return root / ARTIFACTS / name


def _edit_results(root: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    _rewrite_json(_res(root, name), lambda payload: mutate(payload["results"]))


def _edit_events(root: Path, mutate: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> None:
    path = _art(root, EVENTS_FILENAME)
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in mutate(events)),
        encoding="utf-8",
    )


def _edit_checkpoint(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = _art(root, "jlens_fit_checkpoint.pt")
    payload = load_fit_checkpoint(path)
    mutate(payload)
    torch.save(payload, path)


# --------------------------------------------------------------------------
# Accepted runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_fit", [100, 200])
def test_v2_validator_accepts_selected_count_and_comparison(
    tmp_path: Path, n_fit: int
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=n_fit, n_holdout=200)
    checks = validate_qwen_jlens_v2(tmp_path)
    assert checks
    assert all(checks.values())
    assert {
        "selected_fit_count_candidate",
        "holdout_count",
        "v1_lens_canonical_hash",
        "comparison_artifact",
        "no_raw_text_leakage",
    } <= set(checks)


def test_v2_validator_does_not_need_the_v1_project_tree(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    shutil.rmtree(tmp_path / "results" / "qwen25_coder_7b")
    shutil.rmtree(tmp_path / "results" / "qwen25_coder_7b_dose")
    shutil.rmtree(tmp_path / "artifacts" / "qwen25_coder_7b_jlens")
    assert all(validate_qwen_jlens_v2(tmp_path).values())


def test_v2_validator_accepts_aborted_pre_selection_run(tmp_path: Path) -> None:
    _write_aborted_v2_tree(tmp_path, gate="projected_runtime")
    results = _res(tmp_path, "00_jlens_prepare.json")
    prepare = json.loads(results.read_text(encoding="utf-8"))["results"]
    assert prepare["fit_count"] == V2_MAX_FIT_COUNT  # the full 800 + 200 pool
    assert all(validate_qwen_jlens_v2(tmp_path).values())


def test_v2_validator_accepts_aborted_dim_batch_oom_run(tmp_path: Path) -> None:
    _write_aborted_v2_tree(tmp_path, gate="dim_batch_oom")
    assert all(validate_qwen_jlens_v2(tmp_path).values())


# --------------------------------------------------------------------------
# Selected count and holdout count
# --------------------------------------------------------------------------


def test_v2_validator_rejects_non_candidate_fit_count(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=175, n_holdout=200)
    with pytest.raises(ValueError, match="fit count candidate"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_holdout_count_other_than_200(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=199)
    with pytest.raises(ValueError, match="holdout count"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_holdout_artifact_with_missing_item(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def drop_last(results: dict[str, Any]) -> None:
        last = results["holdout_ids"].pop()
        results["holdout_hashes"].pop()
        results["by_item"].pop(last)
        results["n_items"] -= 1

    _edit_results(tmp_path, "03_jlens_holdout.json", drop_last)
    with pytest.raises(ValueError, match="holdout"):
        validate_qwen_jlens_v2(tmp_path)


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _tamper_events_probe_choice(root: Path) -> None:
    def mutate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in events:
            if item.get("event") == "probe_choice":
                item["fit_count"] = 150
        return events

    _edit_events(root, mutate)


SELECTED_COUNT_TAMPERS: dict[str, Callable[[Path], None]] = {
    "probe": lambda root: _edit_results(
        root, "01_jlens_probe.json", _set(("selected_fit_count",), 150)
    ),
    "fit": lambda root: _edit_results(
        root, "02_jlens_fit.json", _set(("n_fit",), 150)
    ),
    "prepare": lambda root: _edit_results(
        root, "00_jlens_prepare.json", _set(("fit_count",), 150)
    ),
    "evaluation": lambda root: _rewrite_json(
        _res(root, "jlens_evaluation.json"), _set(("n_fit",), 150)
    ),
    "checkpoint": lambda root: _edit_checkpoint(
        root, lambda payload: payload.update(fit_count=150)
    ),
    "status": lambda root: _rewrite_json(
        _art(root, STATUS_FILENAME), _set(("fit_total_units",), 150)
    ),
    "probe_choice_event": _tamper_events_probe_choice,
}


@pytest.mark.parametrize("source", sorted(SELECTED_COUNT_TAMPERS))
def test_v2_validator_rejects_disagreeing_selected_counts(
    tmp_path: Path, source: str
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    SELECTED_COUNT_TAMPERS[source](tmp_path)
    with pytest.raises(ValueError, match="selected fit count"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_selected_count_that_is_not_the_largest_safe(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=100, n_holdout=200)

    def fast_probe(results: dict[str, Any]) -> None:
        # 1 s per prompt makes 800 the largest safe candidate, not 100.
        results["wall_seconds"] = 1.0
        results["attempts"][-1]["wall_seconds"] = 1.0
        results["projected_seconds"] = 100.0

    _edit_results(tmp_path, "01_jlens_probe.json", fast_probe)
    with pytest.raises(ValueError, match="largest safe"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# Comparison artifact
# --------------------------------------------------------------------------


def test_v2_validator_rejects_pass_without_comparison_artifact(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _res(tmp_path, "04_jlens_compare.json").unlink()
    with pytest.raises(ValueError, match="comparison artifact"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_manifest_that_omits_comparison(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "execution_manifest.json",
        lambda results: results["result_files"].remove("04_jlens_compare.json"),
    )
    with pytest.raises(ValueError, match="manifest"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_comparison_with_different_holdout_ids(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def swap(results: dict[str, Any]) -> None:
        ids = results["shared_holdout_ids"]
        ids[0], ids[1] = ids[1], ids[0]

    _edit_results(tmp_path, "04_jlens_compare.json", swap)
    with pytest.raises(ValueError, match="shared_holdout_ids"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_comparison_with_different_holdout_hashes(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["shared_holdout_hashes"].__setitem__(0, "f" * 64),
    )
    with pytest.raises(ValueError, match="shared_holdout_hashes"):
        validate_qwen_jlens_v2(tmp_path)


@pytest.mark.parametrize("method", ["jlens_v1", "jlens_v2", "logit_lens"])
def test_v2_validator_rejects_comparison_methods_that_are_not_exact(
    tmp_path: Path, method: str
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["methods"].pop(method),
    )
    with pytest.raises(ValueError, match="comparison methods"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_extra_comparison_method(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["methods"].update(jlens_v3={}),
    )
    with pytest.raises(ValueError, match="comparison methods"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_comparison_v2_metrics_that_differ_from_holdout(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["methods"]["jlens_v2"]["overall"].update(
            {"pass@1": 0.0}
        ),
    )
    with pytest.raises(ValueError, match="comparison metric"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_stability_part_that_does_not_recompose(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def tamper(results: dict[str, Any]) -> None:
        for key in ("parts", "halves"):
            results["stability"][key][1]["overall"]["jlens_v1"]["pass@1"] = 0.0

    _edit_results(tmp_path, "04_jlens_compare.json", tamper)
    with pytest.raises(ValueError, match="comparison metric stability"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_stability_parts_that_are_not_contiguous_halves(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def reorder(results: dict[str, Any]) -> None:
        for key in ("parts", "halves"):
            part = results["stability"][key][0]
            part["holdout_ids"][0], part["holdout_ids"][1] = (
                part["holdout_ids"][1],
                part["holdout_ids"][0],
            )

    _edit_results(tmp_path, "04_jlens_compare.json", reorder)
    with pytest.raises(ValueError, match="contiguous halves"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_holdout_metric_that_does_not_recompose(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "03_jlens_holdout.json",
        lambda results: results["by_layer"]["0"]["jlens"].update({"mean_rank": 99.0}),
    )
    with pytest.raises(ValueError, match="holdout metric"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_comparison_v1_hash_that_is_not_canonical(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["v1_lens"].update(sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="v1 lens sha256"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_comparison_v1_hash_of_tiny_fixture_lens(
    tmp_path: Path,
) -> None:
    """Without the canonical rewrite the fixture's own hash must be refused."""
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    fixture_lens = tmp_path / "artifacts" / "qwen25_coder_7b_jlens" / (
        "qwen25_coder_7b_jacobian_lens.pt"
    )
    import hashlib

    digest = hashlib.sha256(fixture_lens.read_bytes()).hexdigest()
    assert digest != FROZEN_LENS_SHA256
    _edit_results(
        tmp_path,
        "04_jlens_compare.json",
        lambda results: results["v1_lens"].update(sha256=digest),
    )
    with pytest.raises(ValueError, match="v1 lens sha256"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# v2 lens file
# --------------------------------------------------------------------------


def test_v2_validator_rejects_missing_v2_lens(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _art(tmp_path, V2_FINAL_LENS_FILENAME).unlink()
    with pytest.raises(ValueError, match="v2 lens"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_malformed_v2_lens_bytes(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _art(tmp_path, V2_FINAL_LENS_FILENAME).write_bytes(b"not a torch file")
    with pytest.raises(ValueError, match="v2 lens"):
        validate_qwen_jlens_v2(tmp_path)


def _rewrite_lens(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = _art(root, V2_FINAL_LENS_FILENAME)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)
    # Keep size and hash artifacts consistent so the lens contents are what fails.
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    for name in ("02_jlens_fit.json",):
        _edit_results(
            root, name, lambda r: r.update(lens_sha256=digest, lens_bytes=size)
        )
    _rewrite_json(
        _res(root, "jlens_evaluation.json"),
        lambda r: r.update(lens_sha256=digest, lens_bytes=size),
    )


LENS_TAMPERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "missing_jacobians": lambda payload: payload.pop("J"),
    "wrong_n_prompts": lambda payload: payload.update(n_prompts=199),
    "wrong_d_model": lambda payload: payload.update(d_model=9),
    "wrong_layers": lambda payload: payload.update(source_layers=[0, 1]),
    "wrong_shape": lambda payload: payload["J"].update(
        {0: torch.zeros(3, 3, dtype=torch.float16)}
    ),
    "non_finite": lambda payload: payload["J"].update(
        {0: torch.full((8, 8), float("nan"), dtype=torch.float16)}
    ),
}


@pytest.mark.parametrize("tamper", sorted(LENS_TAMPERS))
def test_v2_validator_rejects_invalid_v2_lens_contents(
    tmp_path: Path, tamper: str
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_lens(tmp_path, LENS_TAMPERS[tamper])
    with pytest.raises(ValueError, match="v2 lens"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_v2_lens_hash_that_disagrees_with_the_file(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"), _set(("lens_sha256",), "1" * 64)
    )
    with pytest.raises(ValueError, match="v2 lens"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_v2_lens_recorded_under_the_v1_filename(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_results(
        tmp_path,
        "02_jlens_fit.json",
        _set(("lens",), "artifacts/v2/qwen25_coder_7b_jacobian_lens.pt"),
    )
    with pytest.raises(ValueError, match="v2 lens"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# Hashes, provenance, events
# --------------------------------------------------------------------------


def test_v2_validator_rejects_corpus_hash_that_does_not_match_the_items(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def corrupt(results: dict[str, Any]) -> None:
        results["fit"][0]["token_count"] += 1

    _edit_results(tmp_path, "00_jlens_prepare.json", corrupt)
    with pytest.raises(ValueError, match="corpus hash"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_config_hash_that_does_not_match_the_profile(
    tmp_path: Path,
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"), _set(("config_hash",), "2" * 64)
    )
    with pytest.raises(ValueError, match="config hash"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_fit_and_holdout_overlap(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def overlap(results: dict[str, Any]) -> None:
        results["holdout_ids"][0] = results["fit_ids"][0]

    _edit_results(tmp_path, "00_jlens_prepare.json", overlap)
    with pytest.raises(ValueError, match="fit/holdout"):
        validate_qwen_jlens_v2(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_model_revision", "deadbeef"),
        ("dataset_revision", "deadbeef"),
        ("jlens_upstream_commit", "deadbeef"),
        ("model_name", "other/model"),
    ],
)
def test_v2_validator_rejects_wrong_pinned_provenance(
    tmp_path: Path, field: str, value: str
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_json(
        _res(tmp_path, "01_jlens_probe.json"),
        lambda payload: payload["provenance"].update({field: value}),
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_mixed_run_ids(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_json(
        _res(tmp_path, "03_jlens_holdout.json"),
        lambda payload: payload["provenance"].update(run_id="another-run"),
    )
    with pytest.raises(ValueError, match="run_id"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_checkpoint_from_another_run(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_checkpoint(tmp_path, lambda payload: payload.update(run_id="another-run"))
    with pytest.raises(ValueError, match="checkpoint"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_missing_checkpoint_event(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _edit_events(
        tmp_path,
        lambda events: [
            item
            for item in events
            if not (item.get("event") == "checkpoint" and item.get("n_done") == 77)
        ],
    )
    with pytest.raises(ValueError, match="checkpoint events"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_events_of_another_run(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)

    def foreign(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events[0]["run_id"] = "another-run"
        return events

    _edit_events(tmp_path, foreign)
    with pytest.raises(ValueError, match="run_id"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_incomplete_status(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _rewrite_json(_art(tmp_path, STATUS_FILENAME), _set(("state",), "running"))
    with pytest.raises(ValueError, match="status"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# Raw text leakage (keys and values)
# --------------------------------------------------------------------------


def _leak_in_results(name: str, mutate: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda root: _edit_results(root, name, mutate)


def _leak_in_event(root: Path) -> None:
    def mutate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events[2]["detail"] = "import os"
        return events

    _edit_events(root, mutate)


def _leak_corpus_sentence(root: Path) -> None:
    sentence = "Compute the total of all supplied values"
    corpus = _art(root, "corpus.jsonl")
    corpus.write_text(
        corpus.read_text(encoding="utf-8")
        + json.dumps({"split": "fit", "fit_text": sentence}) + "\n",
        encoding="utf-8",
    )
    _edit_results(root, "02_jlens_fit.json", lambda r: r.update(note=sentence))


LEAKS: dict[str, Callable[[Path], None]] = {
    "key_fit_text": _leak_in_results(
        "01_jlens_probe.json", lambda r: r.update(fit_text="anything")
    ),
    "nested_key_code": _leak_in_results(
        "04_jlens_compare.json", lambda r: r["by_item"].update(code={"x": 1})
    ),
    "key_in_status": lambda root: _rewrite_json(
        _art(root, STATUS_FILENAME), lambda p: p.update(docstring="x")
    ),
    "multiline_value": _leak_in_results(
        "04_jlens_compare.json",
        lambda r: r.update(note="def f():\n    return 1\n"),
    ),
    "single_line_code_value": lambda root: _rewrite_json(
        _res(root, "jlens_evaluation.json"),
        lambda p: p.update(message="def f(): return 1"),
    ),
    "value_in_list": _leak_in_results(
        "03_jlens_holdout.json", lambda r: r.update(extra=["ok", 'x = """doc"""'])
    ),
    "value_in_event": _leak_in_event,
    "value_in_status": lambda root: _rewrite_json(
        _art(root, STATUS_FILENAME), lambda p: p.update(note="from os import path")
    ),
    "value_in_key": _leak_in_results(
        "03_jlens_holdout.json",
        lambda r: r["by_item"].update({"def f(x): return x": {}}),
    ),
    "value_copied_from_corpus": _leak_corpus_sentence,
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_v2_validator_rejects_raw_text_in_keys_or_values(
    tmp_path: Path, leak: str
) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    LEAKS[leak](tmp_path)
    with pytest.raises(ValueError, match="raw text"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_scans_the_aborted_artifacts_for_raw_text(tmp_path: Path) -> None:
    _write_aborted_v2_tree(tmp_path)
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"),
        lambda p: p.update(abort_reason="def f():\n    pass"),
    )
    with pytest.raises(ValueError, match="raw text"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# Aborted pre-selection runs
# --------------------------------------------------------------------------


def _touch_checkpoint(root: Path) -> None:
    _art(root, "jlens_fit_checkpoint.pt").write_bytes(b"x")


def _touch_lens(root: Path) -> None:
    _art(root, V2_FINAL_LENS_FILENAME).write_bytes(b"x")


def _touch_result(name: str) -> Callable[[Path], None]:
    def touch(root: Path) -> None:
        _rewrite_json_new(_res(root, name))

    return touch


def _rewrite_json_new(path: Path) -> None:
    path.write_text(json.dumps({"results": {}}) + "\n", encoding="utf-8")


ABORT_EXTRAS: dict[str, Callable[[Path], None]] = {
    "checkpoint": _touch_checkpoint,
    "final_lens": _touch_lens,
    "fit_artifact": _touch_result("02_jlens_fit.json"),
    "holdout_artifact": _touch_result("03_jlens_holdout.json"),
    "compare_artifact": _touch_result("04_jlens_compare.json"),
}


@pytest.mark.parametrize("extra", sorted(ABORT_EXTRAS))
def test_v2_validator_rejects_aborted_run_with_post_selection_artifacts(
    tmp_path: Path, extra: str
) -> None:
    _write_aborted_v2_tree(tmp_path)
    ABORT_EXTRAS[extra](tmp_path)
    with pytest.raises(ValueError, match="pre-selection"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_aborted_run_that_reports_a_fit_count(
    tmp_path: Path,
) -> None:
    _write_aborted_v2_tree(tmp_path)
    _rewrite_json(_res(tmp_path, "jlens_evaluation.json"), _set(("n_fit",), 100))
    with pytest.raises(ValueError, match="n_fit"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_aborted_run_with_a_selected_fit_count(
    tmp_path: Path,
) -> None:
    _write_aborted_v2_tree(tmp_path)
    _edit_results(
        tmp_path, "01_jlens_probe.json", _set(("selected_fit_count",), 100)
    )
    with pytest.raises(ValueError, match="selected_fit_count"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_aborted_prepare_that_is_not_the_full_pool(
    tmp_path: Path,
) -> None:
    _write_aborted_v2_tree(tmp_path)

    def shrink(results: dict[str, Any]) -> None:
        results["fit_count"] = 200
        results["fit_ids"] = results["fit_ids"][:200]
        results["fit_hashes"] = results["fit_hashes"][:200]
        results["fit"] = results["fit"][:200]

    _edit_results(tmp_path, "00_jlens_prepare.json", shrink)
    with pytest.raises(ValueError, match="full candidate pool"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_aborted_run_with_unknown_gate(tmp_path: Path) -> None:
    _write_aborted_v2_tree(tmp_path)
    _rewrite_json(
        _res(tmp_path, "jlens_evaluation.json"), _set(("abort_gate",), "mystery")
    )
    with pytest.raises(ValueError, match="abort gate"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_projected_runtime_abort_that_was_within_limit(
    tmp_path: Path,
) -> None:
    _write_aborted_v2_tree(tmp_path, gate="projected_runtime")

    def within(results: dict[str, Any]) -> None:
        results["exceeds_limit"] = False

    _edit_results(tmp_path, "01_jlens_probe.json", within)
    with pytest.raises(ValueError, match="abort gate"):
        validate_qwen_jlens_v2(tmp_path)


def test_v2_validator_rejects_aborted_status_that_says_completed(
    tmp_path: Path,
) -> None:
    _write_aborted_v2_tree(tmp_path)
    _rewrite_json(_art(tmp_path, STATUS_FILENAME), _set(("state",), "completed"))
    with pytest.raises(ValueError, match="status"):
        validate_qwen_jlens_v2(tmp_path)


# --------------------------------------------------------------------------
# Missing evaluation and CLI
# --------------------------------------------------------------------------


def test_v2_validator_rejects_missing_evaluation(tmp_path: Path) -> None:
    _write_valid_v2_tree(tmp_path, n_fit=200, n_holdout=200)
    _res(tmp_path, "jlens_evaluation.json").unlink()
    with pytest.raises(ValueError, match="jlens_evaluation"):
        validate_qwen_jlens_v2(tmp_path)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("validate_qwen_jlens_v2_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_script_reports_pass_and_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script()
    good = tmp_path / "good"
    _write_valid_v2_tree(good, n_fit=200, n_holdout=200)
    assert module.main(good) == 0
    assert "PASS selected_fit_count_candidate" in capsys.readouterr().out

    bad = tmp_path / "bad"
    _write_valid_v2_tree(bad, n_fit=200, n_holdout=200)
    _res(bad, "04_jlens_compare.json").unlink()
    assert module.main(bad) == 1
    assert "FAIL" in capsys.readouterr().out


def test_v2_script_does_not_use_the_v1_validators() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "validate_qwen_jlens import" not in source
    assert "qwen_pilot_validate" not in source
    assert "qwen_dose_validate" not in source
