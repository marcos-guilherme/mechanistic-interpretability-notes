from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
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
from jspace_llama.observation_data import (
    DATASET_SHA256,
    EXPECTED_ITEM_COUNT,
    PANEL_NAMES,
)


class ObservationValidationError(ValueError):
    pass


METHODS = ("jlens", "logit_lens")
SCOPES = ("band", "all_layers")
KS = (1, 5, 10)
FORBIDDEN_KEYS = {
    "activation",
    "activations",
    "logit",
    "logits",
    "continuation",
    "tensor",
    "tensors",
    "credential",
    "credentials",
    "secret",
    "password",
    "api_key",
    "cache_path",
    "model_path",
    "lens_path",
}
ABSOLUTE_PATHS = (
    re.compile(r"(?i)\bfile://"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]"),
    re.compile(r"\\\\[^\\/\s]+[\\/][^\\/\s]+"),
    re.compile(r"(?<![:/\w])/(?:home|content|tmp|users?)/[^\s\"'<>]+", re.I),
)
REASON_MESSAGES = {
    "incomplete_observation": "observational run did not produce every required record",
    "incomplete_provenance": "reproducible provenance is incomplete",
    "cuda_required": "a CUDA runtime is required",
    "asset_load_failed": "model or lens asset loading failed",
    "resource_exhausted": "GPU resources were exhausted",
    "tokenization_failed": "observational tokenization failed",
    "dimension_mismatch": "model and lens dimensions do not match",
    "layer_band_invalid": "the selected layer band is invalid",
    "lens_integrity_failed": "lens integrity verification failed",
    "dataset_invalid": "the pinned observational dataset is invalid",
    "readout_failed": "observational readout failed",
    "unexpected_error": "unexpected programming error; see traceback",
}
MAX_JSON_BYTES = 2_000_000


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ObservationValidationError(
                f"JSON path is not a regular file: {path.name}"
            )
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ObservationValidationError(
                f"JSON file is too large: {path.name}"
            )
        document = strict_json_loads(path.read_bytes())
    except (OSError, StrictJSONError) as error:
        raise ObservationValidationError(
            f"invalid JSON file {path.name}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ObservationValidationError(
            f"JSON root is not an object: {path.name}"
        )
    return document


def _validate_schema(
    document: dict[str, Any],
    schema_path: Path,
) -> None:
    schema = _read_json(schema_path)
    try:
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        ).validate(document)
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        raise ObservationValidationError(
            f"schema validation failed for {schema_path.name}"
            f"{' at ' + location if location else ''}: {error.message}"
        ) from error


def _reject_forbidden(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_KEYS or normalized.endswith(
                ("_credential", "_secret", "_password", "_api_key")
            ):
                raise ObservationValidationError(
                    f"forbidden field at {location}.{key}"
                )
            _reject_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ObservationValidationError(
            f"non-finite value at {location}"
        )
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in ABSOLUTE_PATHS
    ):
        raise ObservationValidationError(
            f"absolute/local path at {location}"
        )


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _provenance_complete(provenance: dict[str, Any]) -> bool:
    try:
        source = provenance["source"]
        model = provenance["model"]
        lens = provenance["lens"]
        software = provenance["software"]
        hardware = provenance["hardware"]
        return all(
            (
                _is_hex(source["commit"], 40),
                isinstance(source["dirty"], bool),
                source["scoped_dirty"] is False,
                _is_hex(source["archive_sha256"], 64),
                model["id"] == "meta-llama/Llama-3.1-8B-Instruct",
                _is_hex(model["resolved_revision"], 40),
                model["layer_count"] == 32,
                lens["repo_id"] == "neuronpedia/jacobian-lens",
                lens["filename"].endswith(
                    "Llama-3.1-8B-Instruct_jacobian_lens.pt"
                ),
                _is_hex(lens["resolved_revision"], 40),
                _is_hex(lens["sha256"], 64),
                isinstance(lens["size_bytes"], int),
                not isinstance(lens["size_bytes"], bool),
                lens["size_bytes"] > 0,
                lens["integrity_verified"] is True,
                software["jspace_commit"]
                == "04c822a255fe0a10d84e08ec499e5e09f4b7d37a",
                all(
                    isinstance(software[key], str) and software[key]
                    for key in (
                        "python",
                        "pytorch",
                        "transformers",
                        "huggingface_hub",
                        "jspace",
                    )
                ),
                isinstance(hardware["gpu_name"], str)
                and bool(hardware["gpu_name"]),
                isinstance(hardware["cuda_version"], str)
                and bool(hardware["cuda_version"]),
                hardware["dtype"] == "bfloat16",
            )
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _metric(
    records: list[dict[str, Any]],
    method: str,
    scope: str,
) -> dict[str, float | int]:
    rank_key = "best_rank" if scope == "band" else "all_layers_best_rank"
    totals = {k: 0.0 for k in KS}
    n_intermediates = 0
    for record in records:
        intermediates = record["intermediates"]
        n_intermediates += len(intermediates)
        for k in KS:
            totals[k] += sum(
                int(intermediate[method][rank_key]) <= k
                for intermediate in intermediates
            ) / len(intermediates)
    return {
        "n_items": len(records),
        "n_intermediates": n_intermediates,
        **{
            f"pass@{k}": totals[k] / len(records) if records else 0.0
            for k in KS
        },
    }


def _validate_counts_and_claims(result: dict[str, Any]) -> bool:
    records = result["per_item_best_ranks"]
    panel = result["panel"]
    complete = (
        len(records) == EXPECTED_ITEM_COUNT
        and len({record["name"] for record in records})
        == EXPECTED_ITEM_COUNT
        and [item["name"] for item in panel] == list(PANEL_NAMES)
        and result["panel_names"] == list(PANEL_NAMES)
    )
    summary = result["tokenization_summary"]
    intermediate_count = sum(
        len(record["intermediates"]) for record in records
    )
    single_count = sum(
        intermediate["single_token"] is True
        for record in records
        for intermediate in record["intermediates"]
    )
    expected_summary = {
        "item_count": len(records),
        "intermediate_count": intermediate_count,
        "single_token_count": single_count,
        "multi_token_count": intermediate_count - single_count,
    }
    if summary != expected_summary:
        raise ObservationValidationError(
            "tokenization counts do not recompute"
        )
    for method in METHODS:
        for scope in SCOPES:
            expected = _metric(records, method, scope)
            claimed = result["metrics"][method][scope]
            if claimed != expected:
                raise ObservationValidationError(
                    f"pass@k claims do not recompute for {method}/{scope}"
                )
    band = result["layers"]["band"]
    all_layers = result["layers"]["all_layers"]
    if complete and (not band or not all_layers or not set(band) <= set(all_layers)):
        raise ObservationValidationError("layer scopes are incomplete")
    for record in records:
        if not record["intermediates"]:
            raise ObservationValidationError("item has no intermediates")
        for intermediate in record["intermediates"]:
            for method in METHODS:
                detail = intermediate[method]
                if [rank["layer"] for rank in detail["ranks"]] != all_layers:
                    raise ObservationValidationError(
                        "rank layers differ from all-layer scope"
                    )
                by_layer = {
                    rank["layer"]: rank["rank"] for rank in detail["ranks"]
                }
                band_best = min(
                    ((by_layer[layer], layer) for layer in band),
                    key=lambda pair: pair[0],
                )
                all_best = min(
                    ((by_layer[layer], layer) for layer in all_layers),
                    key=lambda pair: pair[0],
                )
                if (
                    detail["best_rank"],
                    detail["best_layer"],
                ) != band_best or (
                    detail["all_layers_best_rank"],
                    detail["all_layers_best_layer"],
                ) != all_best:
                    raise ObservationValidationError(
                        "best-rank claims do not recompute"
                    )
    for item in panel:
        for method in METHODS:
            if [
                slot["layer"] for slot in item["top_tokens"][method]
            ] != band:
                raise ObservationValidationError(
                    "top-token detail is not band-only"
                )
        if [slot["layer"] for slot in item["pursuit"]] != band:
            raise ObservationValidationError(
                "pursuit detail differs from workspace band"
            )
    return complete


def _validate_raw(
    manifest: dict[str, Any],
    *,
    raw_path: Path | None,
) -> dict[str, Any] | None:
    claimed = manifest["raw_artifact_sha256"]
    if raw_path is None:
        return None
    payload = raw_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != claimed:
        raise ObservationValidationError(
            "raw artifact SHA-256 does not match manifest"
        )
    try:
        raw = strict_json_loads(payload)
    except StrictJSONError as error:
        raise ObservationValidationError("raw artifact is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ObservationValidationError("raw artifact root is not an object")
    return raw


def validate_documents(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    schema_dir: str | Path,
    raw_path: str | Path | None = None,
) -> None:
    schemas = Path(schema_dir)
    _reject_forbidden(manifest)
    _reject_forbidden(result)
    _validate_schema(
        manifest,
        schemas / "jspace_observation_manifest.schema.json",
    )
    _validate_schema(
        result,
        schemas / "jspace_observation.schema.json",
    )
    if manifest["run_id"] != result["run_id"]:
        raise ObservationValidationError("run_id differs across artifacts")
    if result["manifest_sha256"] != document_sha256(manifest):
        raise ObservationValidationError(
            "manifest_sha256 does not match manifest"
        )
    dataset = manifest["dataset"]
    if (
        dataset["sha256"] != DATASET_SHA256
        or dataset["item_count"] != EXPECTED_ITEM_COUNT
    ):
        raise ObservationValidationError("dataset claim is not pinned")
    raw = _validate_raw(
        manifest,
        raw_path=None if raw_path is None else Path(raw_path),
    )
    complete = _validate_counts_and_claims(result)
    provenance_complete = _provenance_complete(manifest["provenance"])
    expected_status = (
        "success" if complete and provenance_complete else "inconclusive"
    )
    if manifest["status"] == "error":
        raise ObservationValidationError(
            "unexpected errors must surface instead of validating"
        )
    if manifest["status"] != result["status"]:
        raise ObservationValidationError("status differs across artifacts")
    if result["status"] != expected_status:
        raise ObservationValidationError("status claim does not recompute")
    if manifest["reason"] != result["reason"]:
        raise ObservationValidationError("reason differs across artifacts")
    reason = result["reason"]
    if reason is not None and reason["message"] != REASON_MESSAGES[reason["code"]]:
        raise ObservationValidationError("reason is not from the allowlist")
    if result["status"] == "success":
        if reason is not None or manifest["raw_artifact_sha256"] is None:
            raise ObservationValidationError(
                "successful observation needs raw hash and no reason"
            )
        if raw is not None:
            try:
                if (
                    len(raw["direct"]["items"]) != EXPECTED_ITEM_COUNT
                    or len(raw["tokenization_audit"]) != EXPECTED_ITEM_COUNT
                    or [item["name"] for item in raw["panel"]["items"]]
                    != list(PANEL_NAMES)
                ):
                    raise ObservationValidationError(
                        "raw observation is incomplete"
                    )
            except (KeyError, TypeError):
                raise ObservationValidationError(
                    "raw observation is incomplete"
                ) from None
    elif reason is None:
        raise ObservationValidationError(
            "inconclusive observation requires a reason"
        )


def validate_result_files(
    results_dir: Path,
    schema_dir: Path,
    raw_path: Path,
) -> None:
    manifest = _read_json(results_dir / "jspace_observation_manifest.json")
    result = _read_json(results_dir / "jspace_observation.json")
    validate_documents(
        manifest,
        result,
        schema_dir=schema_dir,
        raw_path=raw_path if raw_path.exists() else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    parser.add_argument("--schema-dir", type=Path, default=PROJECT / "schemas")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=PROJECT / "artifacts" / "jspace_observation_raw.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_result_files(args.results_dir, args.schema_dir, args.raw_path)
    except (ObservationValidationError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    print("PASS: observation schemas, hashes, counts, and claims are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
