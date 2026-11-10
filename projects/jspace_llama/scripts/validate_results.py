from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import jsonschema


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jspace_llama.evaluation import evaluate_conditions
from jspace_llama.jsonio import document_sha256
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
ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    elif isinstance(value, str) and ABSOLUTE_PATH.match(value):
        raise ResultsValidationError(
            f"absolute/local path forbidden at {location}"
        )


def _recompute_primary(final: dict[str, Any]) -> tuple[list[float], int]:
    primary = final["primary"]
    spider_present = primary["readout"]["spider_present"]
    passing: list[float] = []
    for index, alpha in enumerate(primary["alphas"]):
        gates, reasons = evaluate_conditions(
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


def validate_documents(
    manifest: dict[str, Any],
    final: dict[str, Any],
    *,
    schema_dir: str | Path,
) -> None:
    schemas = Path(schema_dir)
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
    complete = provenance_is_complete(manifest["provenance"])
    if manifest["provenance_complete"] is not complete:
        raise ResultsValidationError(
            "provenance_complete claim does not recompute"
        )
    experiment = manifest["provenance"]["experiment"]
    for prompt in experiment["prompts"]:
        if prompt["instruction_sha256"] != text_sha256(prompt["text"]):
            raise ResultsValidationError(
                f"prompt hash does not match text for {prompt['name']}"
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
    if manifest["terminal_status"] == "completed":
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
    try:
        manifest = json.loads(
            (results_dir / "execution_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        final = json.loads(
            (results_dir / "final_evaluation.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsValidationError(str(error)) from error
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
