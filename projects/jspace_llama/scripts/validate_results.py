from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

import jsonschema


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jspace_llama.jsonio import (
    StrictJSONError,
    document_sha256,
    strict_json_loads,
)
from jspace_llama.provenance import provenance_is_complete, text_sha256


class ResultsValidationError(ValueError):
    pass


FORBIDDEN_KEYS = {
    "continuation",
    "rendered_prompt",
    "instruction",
    "slots",
    "tensor",
    "tensors",
    "token",
    "token_id",
    "token_ids",
    "credential",
    "credentials",
    "secret",
    "password",
    "api_key",
    "cache_path",
    "model_path",
    "lens_path",
}
MAX_JSON_BYTES = 1_048_576
PATH_PATTERNS = (
    re.compile(r"(?i)\bfile://"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]"),
    re.compile(r"\\\\[^\\/\s]+[\\/][^\\/\s]+"),
    re.compile(r"(?<![:/\w])/(?!/)[^\s\"'<>]+"),
    re.compile(
        r"(?i)(?:^|[\s=(\"'])"
        r"(?:~[\\/]|(?:\$HOME|\$\{HOME\}|%USERPROFILE%|%HOME%)[\\/])"
    ),
)
REASON_MESSAGES = {
    "incomplete_provenance": "reproducible provenance is incomplete",
    "primary_gates_failed": "no alpha passed every primary-prompt gate",
    "cuda_required": "a CUDA runtime is required",
    "asset_load_failed": "model or lens asset loading failed",
    "resource_exhausted": "GPU resources were exhausted",
    "tokenization_failed": "intervention concepts are not valid single tokens",
    "dimension_mismatch": "model and lens dimensions do not match",
    "layer_band_invalid": "the selected layer band is invalid",
    "lens_integrity_failed": "lens integrity verification failed",
    "readout_failed": "workspace readout preparation failed",
    "intervention_preparation_failed": "intervention preparation failed",
    "random_control_failed": "random control preparation failed",
    "evaluation_data_invalid": "required aggregate evaluation data is missing",
    "unexpected_error": "unexpected programming error; see notebook traceback",
}


def _is_reparse_point(metadata: object) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & flag)


def _validate_regular_file(path: Path, metadata: object) -> None:
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ResultsValidationError(f"JSON file is a symlink: {path.name}")
    if _is_reparse_point(metadata):
        raise ResultsValidationError(
            f"JSON file is a reparse point: {path.name}"
        )
    if not stat.S_ISREG(mode):
        raise ResultsValidationError(
            f"JSON file is not a regular file: {path.name}"
        )
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ResultsValidationError(f"JSON file is hard-linked: {path.name}")
    if metadata.st_size > MAX_JSON_BYTES:
        raise ResultsValidationError(f"JSON file is too large: {path.name}")


def _validate_directory(
    path: Path,
    *,
    metadata: object | None = None,
) -> object:
    try:
        directory_metadata = (
            metadata if metadata is not None else os.lstat(path)
        )
    except OSError as error:
        raise ResultsValidationError(
            f"cannot access JSON directory: {path}"
        ) from error
    mode = directory_metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ResultsValidationError(f"JSON directory is a symlink: {path}")
    if _is_reparse_point(directory_metadata):
        raise ResultsValidationError(
            f"JSON directory is a reparse point: {path}"
        )
    if not stat.S_ISDIR(mode):
        raise ResultsValidationError(f"JSON path is not a directory: {path}")
    return directory_metadata


def _same_identity(expected: object, actual: object) -> bool:
    return (
        expected.st_dev == actual.st_dev
        and expected.st_ino == actual.st_ino
        and stat.S_IFMT(expected.st_mode) == stat.S_IFMT(actual.st_mode)
        and expected.st_size == actual.st_size
        and getattr(expected, "st_nlink", 1)
        == getattr(actual, "st_nlink", 1)
    )


def _read_json_file(path: Path) -> Any:
    try:
        metadata = os.lstat(path)
        _validate_regular_file(path, metadata)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            actual = os.fstat(descriptor)
            if not _same_identity(metadata, actual):
                raise ResultsValidationError(
                    f"JSON file changed after validation: {path.name}"
                )
            _validate_regular_file(path, actual)
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                payload = handle.read(MAX_JSON_BYTES + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(payload) > MAX_JSON_BYTES:
            raise ResultsValidationError(
                f"JSON file is too large: {path.name}"
            )
        return strict_json_loads(payload)
    except ResultsValidationError:
        raise
    except (OSError, StrictJSONError) as error:
        raise ResultsValidationError(
            f"invalid JSON file {path.name}: {error}"
        ) from error


def _load_schema(path: Path) -> dict[str, Any]:
    schema = _read_json_file(path)
    if not isinstance(schema, dict):
        raise ResultsValidationError(f"schema is not an object: {path.name}")
    return schema


def _runtime_source_sha256(project_root: Path) -> str:
    project = project_root.resolve()
    repository = project.parents[1]
    candidates: list[tuple[str, Path]] = []
    for pattern in (
        "src/**/*.py",
        "notebooks/*.ipynb",
        "schemas/*.json",
        "scripts/*.py",
    ):
        for path in project.glob(pattern):
            if path.is_file() and not path.name.endswith("_output.ipynb"):
                relative = path.relative_to(project).as_posix()
                candidates.append(
                    (f"projects/jspace_llama/{relative}", path)
                )
    for name in ("pyproject.toml", "uv.lock"):
        path = project / name
        if path.is_file():
            candidates.append((f"projects/jspace_llama/{name}", path))
    runner = repository / "tools" / "colab" / "run_jspace.py"
    if runner.is_file():
        candidates.append(("tools/colab/run_jspace.py", runner))
    if not candidates:
        raise ResultsValidationError(
            "no local runtime-critical source files found"
        )

    digest = hashlib.sha256()
    for label, path in sorted(candidates, key=lambda item: item[0]):
        label_bytes = label.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_schema(
    document: dict[str, Any],
    schema_path: Path,
) -> None:
    try:
        jsonschema.Draft202012Validator(
            _load_schema(schema_path),
            format_checker=jsonschema.FormatChecker(),
        ).validate(document)
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        suffix = f" at {location}" if location else ""
        raise ResultsValidationError(
            f"schema validation failed for {schema_path.name}{suffix}: "
            f"{error.message}"
        ) from error


def _reject_forbidden(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_KEYS or normalized.endswith(
                ("_credential", "_secret", "_password", "_api_key")
            ):
                raise ResultsValidationError(
                    f"forbidden compact-artifact field {location}.{key}"
                )
            _reject_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ResultsValidationError(f"non-finite number forbidden at {location}")
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in PATH_PATTERNS):
            raise ResultsValidationError(
                f"absolute/local path forbidden at {location}"
            )


def _degraded(classification: dict[str, Any]) -> bool:
    degradation = classification["degradation"]
    return bool(
        degradation["empty"]
        or degradation["repetition"]
        or degradation["incoherent"]
    )


def _uniquely_parsed_as(
    classification: dict[str, Any],
    expected: str,
) -> bool:
    answer = classification["answer_parse"]
    return answer["status"] == "unique" and answer["value"] == expected


def _evaluate_conditions(
    conditions: dict[str, dict[str, Any]],
    *,
    spider_present: bool,
) -> tuple[dict[str, bool], list[str]]:
    baseline = conditions["baseline"]
    noop = conditions["noop"]
    random = conditions["random"]
    target = conditions["target"]
    gates = {
        "readout_spider_present": bool(spider_present),
        "baseline_unique_eight": _uniquely_parsed_as(baseline, "eight"),
        "baseline_not_degraded": not _degraded(baseline),
        "noop_unique_eight": _uniquely_parsed_as(noop, "eight"),
        "noop_not_degraded": not _degraded(noop),
        "random_unique_non_six": (
            random["answer_parse"]["status"] == "unique"
            and random["answer_parse"]["value"] != "six"
        ),
        "random_not_degraded": not _degraded(random),
        "target_unique_six": _uniquely_parsed_as(target, "six"),
        "target_no_ant": not bool(target["target_entity_leakage"]),
        "target_not_degraded": not _degraded(target),
    }
    reason_by_gate = {
        "readout_spider_present": "readout_spider_missing",
        "baseline_unique_eight": "baseline_answer_invalid",
        "baseline_not_degraded": "baseline_degraded",
        "noop_unique_eight": "noop_answer_invalid",
        "noop_not_degraded": "noop_degraded",
        "random_unique_non_six": "random_answer_invalid",
        "random_not_degraded": "random_degraded",
        "target_unique_six": "target_answer_invalid",
        "target_no_ant": "target_contains_ant",
        "target_not_degraded": "target_degraded",
    }
    reasons = [
        reason_by_gate[name] for name, passed in gates.items() if not passed
    ]
    return gates, reasons


def _recompute_primary(final: dict[str, Any]) -> tuple[list[float], int]:
    primary = final["primary"]
    readout = primary["readout"]
    spider_present = readout["spider_present"]
    if readout["evaluated"] is not True:
        if primary["alphas"] or spider_present is not None:
            raise ResultsValidationError(
                "unevaluated readout cannot carry measurements"
            )
        return [], 0
    if not isinstance(spider_present, bool):
        raise ResultsValidationError(
            "evaluated readout requires spider presence measurement"
        )
    passing: list[float] = []
    for index, alpha in enumerate(primary["alphas"]):
        gates, reasons = _evaluate_conditions(
            alpha["conditions"],
            spider_present=spider_present,
        )
        passed = all(gates.values())
        if alpha["gates"] != gates:
            raise ResultsValidationError(
                f"gate claims do not recompute at primary.alphas[{index}]"
            )
        if alpha["reasons"] != reasons:
            raise ResultsValidationError(
                f"gate reasons do not recompute at primary.alphas[{index}]"
            )
        if alpha["passed"] is not passed:
            raise ResultsValidationError(
                f"alpha pass claim does not recompute at primary.alphas[{index}]"
            )
        if passed:
            passing.append(float(alpha["alpha"]))
    return passing, len(passing)


def _recompute_paraphrases(
    final: dict[str, Any],
    experiment: dict[str, Any],
    *,
    completed: bool,
) -> None:
    paraphrases = final["paraphrases"]
    entries = paraphrases["prompt_alphas"]
    observed_names: set[str] = set()
    observed_pairs: set[tuple[str, float]] = set()
    recomputed_passing = 0
    spider_present = final["primary"]["readout"]["spider_present"]
    for index, entry in enumerate(entries):
        prompt_name = entry["prompt_name"]
        alpha = float(entry["alpha"])
        pair = (prompt_name, alpha)
        if pair in observed_pairs:
            raise ResultsValidationError(
                "paraphrase prompt/alpha pairs must be unique"
            )
        observed_names.add(prompt_name)
        observed_pairs.add(pair)
        gates, _reasons = _evaluate_conditions(
            entry["conditions"],
            spider_present=spider_present,
        )
        if entry["gates"] != gates:
            raise ResultsValidationError(
                f"paraphrase gates do not recompute at prompt_alphas[{index}]"
            )
        passed = all(gates.values())
        if entry["passed"] is not passed:
            raise ResultsValidationError(
                f"paraphrase pass claim does not recompute at "
                f"prompt_alphas[{index}]"
            )
        recomputed_passing += int(passed)

    if paraphrases["prompt_count"] != len(observed_names):
        raise ResultsValidationError(
            "paraphrase prompt_count does not recompute"
        )
    claimed_passing = paraphrases["passing_prompt_alpha_count"]
    if not 0 <= claimed_passing <= len(entries):
        raise ResultsValidationError(
            "paraphrase passing_prompt_alpha_count is out of bounds"
        )
    if claimed_passing != recomputed_passing:
        raise ResultsValidationError(
            "paraphrase passing_prompt_alpha_count does not recompute"
        )

    if not completed:
        if entries or paraphrases["prompt_count"] != 0:
            raise ResultsValidationError(
                "paraphrase aggregates require a completed run"
            )
        return

    prompt_names = [
        prompt["name"] for prompt in experiment["prompts"]
    ]
    if (
        len(set(prompt_names)) != len(prompt_names)
        or prompt_names.count("main") != 1
    ):
        raise ResultsValidationError(
            "paraphrase manifest prompt set is not unique"
        )
    expected_names = {
        prompt_name for prompt_name in prompt_names if prompt_name != "main"
    }
    expected_alphas = {
        float(alpha) for alpha in experiment["alphas"]
    }
    expected_pairs = {
        (prompt_name, alpha)
        for prompt_name in expected_names
        for alpha in expected_alphas
    }
    if observed_names != expected_names:
        raise ResultsValidationError(
            "paraphrase prompt set differs from manifest provenance"
        )
    if observed_pairs != expected_pairs:
        raise ResultsValidationError(
            "paraphrase prompt/alpha pairs differ from manifest provenance"
        )


def validate_documents(
    manifest: dict[str, Any],
    final: dict[str, Any],
    *,
    schema_dir: str | Path,
    project_root: str | Path = PROJECT,
) -> None:
    schemas = Path(schema_dir)
    _validate_directory(schemas)
    _reject_forbidden(manifest)
    _reject_forbidden(final)
    _validate_schema(
        manifest,
        schemas / "execution_manifest.schema.json",
    )
    _validate_schema(
        final,
        schemas / "final_evaluation.schema.json",
    )

    if manifest["run_id"] != final["run_id"]:
        raise ResultsValidationError("run_id differs across artifacts")
    expected_hash = document_sha256(manifest)
    if final["manifest_sha256"] != expected_hash:
        raise ResultsValidationError("manifest_sha256 does not match manifest")
    experiment = manifest["provenance"]["experiment"]
    for prompt in experiment["prompts"]:
        if prompt["instruction_sha256"] != text_sha256(prompt["text"]):
            raise ResultsValidationError(
                f"prompt hash does not match text for {prompt['name']}"
            )
    complete = provenance_is_complete(manifest["provenance"])
    if manifest["provenance_complete"] is not complete:
        raise ResultsValidationError(
            "provenance_complete claim does not recompute"
        )
    source = manifest["provenance"]["source"]
    archive_sha256 = source.get("archive_sha256")
    if archive_sha256 is not None:
        actual_source_sha256 = _runtime_source_sha256(
            Path(project_root),
        )
        if archive_sha256 != actual_source_sha256:
            raise ResultsValidationError(
                "source archive digest does not match local runtime files"
            )
    manifest_lens = manifest["provenance"]["lens"]
    final_integrity = final["integrity"]
    if (
        final_integrity["lens_sha256"] != manifest_lens["sha256"]
        or final_integrity["lens_integrity_verified"]
        is not manifest_lens.get("integrity_verified")
        or final_integrity["lens_integrity_evaluated"]
        is not isinstance(manifest_lens.get("integrity_verified"), bool)
    ):
        raise ResultsValidationError(
            "lens integrity differs across artifacts"
        )
    if manifest["terminal_status"] == "completed" and (
        manifest_lens.get("integrity_verified") is not True
            or final_integrity["lens_integrity_evaluated"] is not True
        or final_integrity["lens_integrity_verified"] is not True
    ):
        raise ResultsValidationError(
            "completed run requires verified lens integrity"
        )

    passing, passing_count = _recompute_primary(final)
    primary = final["primary"]
    if primary["passing_alphas"] != passing:
        raise ResultsValidationError(
            "passing_alphas claim does not recompute"
        )
    if primary["passing_alpha_count"] != passing_count:
        raise ResultsValidationError(
            "passing_alpha_count claim does not recompute"
        )
    completed = manifest["terminal_status"] == "completed"
    _recompute_paraphrases(
        final,
        experiment,
        completed=completed,
    )
    if completed:
        provenance_alphas = [float(value) for value in experiment["alphas"]]
        evaluated_alphas = [
            float(item["alpha"]) for item in primary["alphas"]
        ]
        if evaluated_alphas != provenance_alphas:
            raise ResultsValidationError(
                "evaluated alphas differ from manifest provenance"
            )

    if manifest["terminal_status"] == "unexpected_error":
        raise ResultsValidationError(
            "unexpected errors must surface instead of producing final evaluation"
        )
    expected_status = (
        "success"
        if (
            manifest["terminal_status"] == "completed"
            and complete
            and passing_count > 0
        )
        else "inconclusive"
    )
    if manifest["status"] != final["status"]:
        raise ResultsValidationError("status differs across artifacts")
    if final["status"] != expected_status:
        raise ResultsValidationError("status claim does not recompute")
    if manifest["reason"] != final["reason"]:
        raise ResultsValidationError("reason differs across artifacts")
    if manifest["reason"] is not None:
        code = manifest["reason"]["code"]
        if manifest["reason"]["message"] != REASON_MESSAGES.get(code):
            raise ResultsValidationError(
                "compact reason is not from the fixed allowlist"
            )
        stage = manifest["reason"].get("stage")
        category = manifest["reason"].get("category")
        if (stage is None) != (category is None):
            raise ResultsValidationError(
                "failure stage and category must appear together"
            )
        if stage is not None and code != "asset_load_failed":
            raise ResultsValidationError(
                "operational metadata is only valid for asset failures"
            )
    if final["status"] == "success" and final["reason"] is not None:
        raise ResultsValidationError("success cannot have an inconclusive reason")
    if final["status"] == "inconclusive" and final["reason"] is None:
        raise ResultsValidationError(
            "inconclusive status requires a machine-readable reason"
        )
    if manifest["terminal_status"] == "completed":
        expected_reason_code = (
            None
            if expected_status == "success"
            else (
                "incomplete_provenance"
                if not complete
                else "primary_gates_failed"
            )
        )
        actual_reason_code = (
            None if final["reason"] is None else final["reason"]["code"]
        )
        if actual_reason_code != expected_reason_code:
            raise ResultsValidationError(
                "status reason does not recompute for completed run"
            )


def validate_result_files(results_dir: Path, schema_dir: Path) -> None:
    _validate_directory(results_dir)
    _validate_directory(schema_dir)
    manifest = _read_json_file(results_dir / "execution_manifest.json")
    final = _read_json_file(results_dir / "final_evaluation.json")
    if not isinstance(manifest, dict) or not isinstance(final, dict):
        raise ResultsValidationError("result artifact root must be an object")
    validate_documents(manifest, final, schema_dir=schema_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT / "results",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=PROJECT / "schemas",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_result_files(args.results_dir, args.schema_dir)
    except ResultsValidationError as error:
        print(f"FAIL: {error}")
        return 1
    print("PASS: schemas, hashes, provenance, and claims are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
