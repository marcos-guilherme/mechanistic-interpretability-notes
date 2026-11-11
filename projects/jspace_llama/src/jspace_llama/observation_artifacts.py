"""Compact, reproducible artifacts for the observational atlas."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any

from jspace_llama.jsonio import document_sha256
from jspace_llama.observation_data import (
    DATASET_SHA256,
    EXPECTED_ITEM_COUNT,
    PANEL_NAMES,
)


UPSTREAM_REPOSITORY = "anthropics/jacobian-lens"
UPSTREAM_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
KS = (1, 5, 10)
METHODS = ("jlens", "logit_lens")
SCOPES = ("band", "all_layers")
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


class ObservationDataError(ValueError):
    """Raised when an execution payload is structurally incomplete."""


def compact_reason(code: str) -> dict[str, str]:
    try:
        return {"code": code, "message": REASON_MESSAGES[code]}
    except KeyError as error:
        raise ValueError(f"unknown observation reason code: {code}") from error


def observation_provenance_is_complete(provenance: object) -> bool:
    def is_hex(value: object, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and re.fullmatch(r"[0-9a-f]+", value) is not None
        )

    try:
        assert isinstance(provenance, dict)
        source = provenance["source"]
        model = provenance["model"]
        lens = provenance["lens"]
        software = provenance["software"]
        hardware = provenance["hardware"]
        return bool(
            is_hex(source["commit"], 40)
            and isinstance(source["dirty"], bool)
            and source["scoped_dirty"] is False
            and is_hex(source["runtime_source_sha256"], 64)
            and model["id"] == "meta-llama/Llama-3.1-8B-Instruct"
            and is_hex(model["resolved_revision"], 40)
            and model["layer_count"] == 32
            and lens["repo_id"] == "neuronpedia/jacobian-lens"
            and lens["filename"]
            == (
                "llama3.1-8b-it/jlens/Salesforce-wikitext/"
                "Llama-3.1-8B-Instruct_jacobian_lens.pt"
            )
            and is_hex(lens["resolved_revision"], 40)
            and is_hex(lens["sha256"], 64)
            and isinstance(lens["size_bytes"], int)
            and lens["size_bytes"] > 0
            and lens["integrity_verified"] is True
            and software["jspace_commit"]
            == "04c822a255fe0a10d84e08ec499e5e09f4b7d37a"
            and all(
                isinstance(software[key], str) and software[key]
                for key in (
                    "python",
                    "pytorch",
                    "transformers",
                    "huggingface_hub",
                    "jspace",
                )
            )
            and isinstance(hardware["gpu_name"], str)
            and bool(hardware["gpu_name"])
            and isinstance(hardware["cuda_version"], str)
            and bool(hardware["cuda_version"])
            and hardware["dtype"] == "bfloat16"
        )
    except (AssertionError, KeyError, TypeError, AttributeError):
        return False


def _empty_metrics() -> dict[str, object]:
    return {
        method: {
            scope: {
                "n_items": 0,
                "n_intermediates": 0,
                **{f"pass@{k}": 0.0 for k in KS},
            }
            for scope in SCOPES
        }
        for method in METHODS
    }


def _metric_for(
    records: list[dict[str, object]],
    method: str,
    scope: str,
) -> dict[str, float | int]:
    rank_key = "best_rank" if scope == "band" else "all_layers_best_rank"
    totals = {k: 0.0 for k in KS}
    intermediate_count = 0
    for record in records:
        intermediates = record["intermediates"]
        if not isinstance(intermediates, list) or not intermediates:
            raise ObservationDataError("each item must have intermediates")
        intermediate_count += len(intermediates)
        for k in KS:
            totals[k] += sum(
                int(intermediate[method][rank_key]) <= k
                for intermediate in intermediates
            ) / len(intermediates)
    count = len(records)
    return {
        "n_items": count,
        "n_intermediates": intermediate_count,
        **{
            f"pass@{k}": totals[k] / count if count else 0.0
            for k in KS
        },
    }


def recompute_metrics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        method: {
            scope: _metric_for(records, method, scope)
            for scope in SCOPES
        }
        for method in METHODS
    }


def _compact_records(
    records: list[dict[str, object]],
    audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for record, audited in zip(records, audit, strict=True):
        compact.append(
            {
                "name": record["name"],
                "prompt": audited["prompt"],
                "prompt_token_ids": list(audited["prompt_token_ids"]),
                "prompt_token_count": audited["prompt_token_count"],
                "prompt_sha256": audited["prompt_sha256"],
                "prompt_token_ids_sha256": audited[
                    "prompt_token_ids_sha256"
                ],
                "intermediates": [
                    {
                        "text": intermediate["text"],
                        "token_id": intermediate["token_id"],
                        "single_token": intermediate["single_token"],
                        **{
                            method: deepcopy(intermediate[method])
                            for method in METHODS
                        },
                    }
                    for intermediate in record["intermediates"]
                ],
            }
        )
    return compact


def _tokenization_summary(
    audit: list[dict[str, object]],
) -> dict[str, int]:
    intermediates = [
        intermediate
        for item in audit
        for intermediate in item.get("intermediates", [])
    ]
    single = sum(
        intermediate.get("single_token") is True
        for intermediate in intermediates
    )
    return {
        "item_count": len(audit),
        "intermediate_count": len(intermediates),
        "single_token_count": single,
        "multi_token_count": len(intermediates) - single,
    }


def _validate_complete_payload(
    payload: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[int]],
]:
    try:
        direct = payload["direct"]
        records = direct["items"]
        audit = payload["tokenization_audit"]
        panel = payload["panel"]["items"]
        layers = payload["layers"]
    except (KeyError, TypeError) as error:
        raise ObservationDataError("missing observational payload field") from error
    if not all(isinstance(value, list) for value in (records, audit, panel)):
        raise ObservationDataError("observational collections must be lists")
    names = [record.get("name") for record in records]
    audit_names = [entry.get("name") for entry in audit]
    if (
        len(records) != EXPECTED_ITEM_COUNT
        or len(set(names)) != EXPECTED_ITEM_COUNT
        or audit_names != names
    ):
        raise ObservationDataError("observational run must contain 93 aligned items")
    panel_names = [item.get("name") for item in panel]
    if panel_names != list(PANEL_NAMES):
        raise ObservationDataError("observational panel is incomplete or unordered")
    if not isinstance(layers, dict):
        raise ObservationDataError("layer scopes must be an object")
    band = layers.get("band")
    all_layers = layers.get("all_layers")
    if (
        not isinstance(band, list)
        or not band
        or not isinstance(all_layers, list)
        or not all_layers
        or not set(band) <= set(all_layers)
        or len(set(all_layers)) != len(all_layers)
    ):
        raise ObservationDataError("invalid observational layer scopes")
    for record, audit_entry in zip(records, audit, strict=True):
        if not isinstance(record, dict):
            raise ObservationDataError("direct item must be an object")
        intermediates = record.get("intermediates")
        if not isinstance(intermediates, list) or not intermediates:
            raise ObservationDataError("each direct item needs intermediates")
        audited = audit_entry.get("intermediates")
        prompt_ids = audit_entry.get("prompt_token_ids")
        canonical_prompt_ids = (
            "["
            + ",".join(str(token_id) for token_id in prompt_ids)
            + "]"
            if isinstance(prompt_ids, (list, tuple))
            else ""
        )
        if (
            audit_entry.get("prompt") != record.get("prompt")
            or not isinstance(
                audit_entry.get("prompt_token_ids"),
                (list, tuple),
            )
            or not audit_entry["prompt_token_ids"]
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in audit_entry["prompt_token_ids"]
            )
            or audit_entry.get("prompt_token_count")
            != len(audit_entry["prompt_token_ids"])
            or audit_entry.get("prompt_sha256")
            != hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
            or audit_entry.get("prompt_token_ids_sha256")
            != hashlib.sha256(
                canonical_prompt_ids.encode("utf-8")
            ).hexdigest()
            or not isinstance(audited, list)
            or len(audited) != len(intermediates)
            or [
                (
                    entry.get("text"),
                    entry.get("token_id"),
                    entry.get("single_token"),
                )
                for entry in audited
            ]
            != [
                (
                    entry.get("text"),
                    entry.get("token_id"),
                    entry.get("single_token"),
                )
                for entry in intermediates
            ]
        ):
            raise ObservationDataError(
                "tokenization audit differs from direct records"
            )
        for intermediate in intermediates:
            if not isinstance(intermediate, dict):
                raise ObservationDataError("intermediate must be an object")
            for method in METHODS:
                detail = intermediate.get(method)
                if not isinstance(detail, dict):
                    raise ObservationDataError("missing direct rank detail")
                ranks = detail.get("ranks")
                if not isinstance(ranks, list):
                    raise ObservationDataError("rank detail must be a list")
                rank_layers = [
                    entry.get("layer")
                    for entry in ranks
                    if isinstance(entry, dict)
                ]
                if rank_layers != all_layers:
                    raise ObservationDataError(
                        "direct ranks must cover every all-layer slot"
                    )
                by_layer = {
                    int(entry["layer"]): int(entry["rank"])
                    for entry in ranks
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
                    detail.get("best_rank"),
                    detail.get("best_layer"),
                ) != band_best or (
                    detail.get("all_layers_best_rank"),
                    detail.get("all_layers_best_layer"),
                ) != all_best:
                    raise ObservationDataError(
                        "direct best-rank claims are inconsistent"
                    )
        top_tokens = record.get("top_tokens")
        if (record["name"] in PANEL_NAMES) != isinstance(top_tokens, dict):
            raise ObservationDataError(
                "top-token detail must exist only for panel names"
            )
        if top_tokens is not None:
            for method in METHODS:
                if [
                    slot.get("layer") for slot in top_tokens.get(method, [])
                ] != band:
                    raise ObservationDataError(
                        "top-token detail must be workspace-band only"
                    )
    by_name = {record["name"]: record for record in records}
    for panel_item in panel:
        direct_item = by_name[panel_item["name"]]
        if (
            panel_item.get("intermediates") != direct_item["intermediates"]
            or panel_item.get("top_tokens") != direct_item["top_tokens"]
            or [
                slot.get("layer")
                for slot in panel_item.get("pursuit", [])
                if isinstance(slot, dict)
            ]
            != band
        ):
            raise ObservationDataError(
                "panel must preserve direct records and band-only pursuit"
            )
    return records, audit, panel, {
        "band": [int(layer) for layer in band],
        "all_layers": [int(layer) for layer in all_layers],
    }


def build_observation_artifacts(
    *,
    run_id: str,
    started_at: str,
    ended_at: str,
    provenance: dict[str, object],
    raw_artifact_sha256: str | None,
    payload: dict[str, object] | None,
    reason_code: str | None = None,
    unexpected: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the strict manifest and compact result without raw tensors/logits."""
    payload_complete = False
    records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    panel: list[dict[str, object]] = []
    layers: dict[str, list[int]] = {"band": [], "all_layers": []}
    if payload is not None and reason_code is None and not unexpected:
        try:
            records, audit, panel, layers = _validate_complete_payload(payload)
            payload_complete = True
        except (
            ObservationDataError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            reason_code = "incomplete_observation"
    provenance_complete = observation_provenance_is_complete(provenance)
    if payload_complete and not provenance_complete:
        reason_code = "incomplete_provenance"
    status = (
        "error"
        if unexpected
        else "success"
        if payload_complete and provenance_complete
        else "inconclusive"
    )
    if status == "success":
        reason = None
    else:
        reason = compact_reason(
            reason_code or ("unexpected_error" if unexpected else "incomplete_observation")
        )
    manifest = {
        "schema_version": 2,
        "artifact": "jspace_observation_manifest",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "provenance": deepcopy(provenance),
        "dataset": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "sha256": DATASET_SHA256,
            "item_count": EXPECTED_ITEM_COUNT,
        },
        "raw_artifact_sha256": raw_artifact_sha256,
        "reason": reason,
    }
    result = {
        "schema_version": 2,
        "artifact": "jspace_observation",
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "metrics": (
            recompute_metrics(records) if payload_complete else _empty_metrics()
        ),
        "tokenization_summary": (
            _tokenization_summary(audit)
            if payload_complete
            else {
                "item_count": 0,
                "intermediate_count": 0,
                "single_token_count": 0,
                "multi_token_count": 0,
            }
        ),
        "per_item_best_ranks": (
            _compact_records(records, audit) if payload_complete else []
        ),
        "panel": deepcopy(panel) if payload_complete else [],
        "layers": layers,
        "panel_names": list(PANEL_NAMES),
        "manifest_sha256": document_sha256(manifest),
    }
    return manifest, result
