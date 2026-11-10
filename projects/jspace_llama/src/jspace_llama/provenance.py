"""Reproducible, non-secret provenance collection."""

from __future__ import annotations

import hashlib
from importlib import metadata
import math
from pathlib import Path
import platform
import re
from typing import Any, Iterable

from jspace_llama.config import AssetsConfig, ExperimentConfig
from jspace_llama.jsonio import strict_json_loads
from jspace_llama.runtime import (
    HUGGINGFACE_OPERATIONAL_ERRORS,
    AssetLoadError,
    LensIntegrityError,
)


JSPACE_COMMIT = "04c822a255fe0a10d84e08ec499e5e09f4b7d37a"
SOURCE_PROVENANCE_FILENAME = "source_provenance.json"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {"", "unknown", "none", "null", "n/a", "placeholder", "test"}
APPROVED_ASSETS = AssetsConfig()
APPROVED_CONFIG = ExperimentConfig()
APPROVED_PROMPTS = tuple(
    (case.name, case.text) for case in APPROVED_CONFIG.prompts.cases
)
APPROVED_GENERATION = {
    "max_new_tokens": APPROVED_CONFIG.generation.max_new_tokens,
    "do_sample": APPROVED_CONFIG.generation.do_sample,
    "decoding": "greedy",
    "first_line": False,
}
APPROVED_INTERVENTION = {
    "source_concept": APPROVED_CONFIG.source,
    "target_concept": APPROVED_CONFIG.target,
    "prefill_only": True,
    "mode": "source_to_target_clamp",
    "random_control": "norm_matched_orthogonal",
}
APPROVED_MODEL_LAYER_COUNT = 32


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def read_source_provenance(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / SOURCE_PROVENANCE_FILENAME
    payload = strict_json_loads(path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "commit",
        "dirty",
    }:
        raise ValueError("source provenance has unexpected fields")
    if payload["schema_version"] != 1:
        raise ValueError("source provenance schema_version must be 1")
    commit = payload.get("commit")
    dirty = payload.get("dirty")
    if not isinstance(commit, str) or HEX_40.fullmatch(commit) is None:
        raise ValueError("source provenance commit must be a 40-character hash")
    if not isinstance(dirty, bool):
        raise ValueError("source provenance dirty must be boolean")
    return {"commit": commit, "dirty": dirty}


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def collect_software_provenance() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pytorch": package_version("torch"),
        "transformers": package_version("transformers"),
        "huggingface_hub": package_version("huggingface-hub"),
        "jspace": package_version("jspace"),
        "jspace_commit": JSPACE_COMMIT,
    }


def resolve_huggingface_revisions(
    model_id: str,
    lens_repo_id: str,
    *,
    api: Any = None,
) -> tuple[str, str]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    try:
        model_revision = str(api.model_info(model_id).sha)
        lens_revision = str(api.model_info(lens_repo_id).sha)
    except HUGGINGFACE_OPERATIONAL_ERRORS as error:
        raise AssetLoadError(
            "failed to resolve pinned Hugging Face revisions"
        ) from error
    except OSError as error:
        raise AssetLoadError(
            "failed to resolve pinned Hugging Face revisions"
        ) from error
    for label, revision in (
        ("model", model_revision),
        ("lens", lens_revision),
    ):
        if not _is_concrete_hex(revision, HEX_40):
            raise AssetLoadError(
                f"resolved {label} revision is not a commit hash"
            )
    return model_revision, lens_revision


def build_experiment_provenance(
    config: Any,
    prepared_prompts: Iterable[Any],
    layers: Iterable[int],
) -> dict[str, Any]:
    return {
        "seed": int(config.seed),
        "prompts": [
            {
                "name": str(prompt.name),
                "text": str(prompt.instruction),
                "instruction_sha256": text_sha256(prompt.instruction),
                "rendered_prompt_sha256": text_sha256(prompt.rendered),
            }
            for prompt in prepared_prompts
        ],
        "generation": {
            "max_new_tokens": int(config.generation.max_new_tokens),
            "do_sample": bool(config.generation.do_sample),
            "decoding": "greedy",
            "first_line": False,
        },
        "layers": [int(layer) for layer in layers],
        "alphas": [float(alpha) for alpha in config.sweep.alphas],
        "intervention": {
            "source_concept": str(config.source),
            "target_concept": str(config.target),
            "prefill_only": True,
            "mode": "source_to_target_clamp",
            "random_control": "norm_matched_orthogonal",
        },
    }


def verify_lens_integrity(
    path: str | Path,
    expected_sha256: str,
    expected_size: int,
) -> bool:
    actual_sha256, actual_size = file_sha256(path)
    if actual_sha256 != expected_sha256 or actual_size != expected_size:
        raise LensIntegrityError("lens changed after loading")
    return True


def _is_concrete_hex(value: Any, pattern: re.Pattern[str]) -> bool:
    return (
        isinstance(value, str)
        and pattern.fullmatch(value) is not None
        and len(set(value)) > 1
    )


def _is_concrete_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.strip().casefold() not in PLACEHOLDERS
    )


def _valid_prompts(prompts: Any) -> bool:
    if not isinstance(prompts, list) or len(prompts) != len(APPROVED_PROMPTS):
        return False
    names = [prompt.get("name") for prompt in prompts if isinstance(prompt, dict)]
    if len(names) != len(prompts) or len(set(names)) != len(names):
        return False
    for prompt, (expected_name, expected_text) in zip(
        prompts,
        APPROVED_PROMPTS,
        strict=True,
    ):
        if (
            prompt.get("name") != expected_name
            or prompt.get("text") != expected_text
            or prompt.get("instruction_sha256") != text_sha256(expected_text)
            or not _is_concrete_hex(
                prompt.get("rendered_prompt_sha256"),
                HEX_64,
            )
            or prompt.get("rendered_prompt_sha256")
            == prompt.get("instruction_sha256")
        ):
            return False
    return True


def _valid_layers(layers: Any, layer_count: Any) -> bool:
    return (
        isinstance(layer_count, int)
        and not isinstance(layer_count, bool)
        and layer_count == APPROVED_MODEL_LAYER_COUNT
        and isinstance(layers, list)
        and bool(layers)
        and all(
            isinstance(layer, int)
            and not isinstance(layer, bool)
            and 0 <= layer < layer_count
            for layer in layers
        )
        and layers == sorted(set(layers))
    )


def _valid_alphas(alphas: Any) -> bool:
    return (
        isinstance(alphas, list)
        and bool(alphas)
        and all(
            isinstance(alpha, (int, float))
            and not isinstance(alpha, bool)
            and math.isfinite(float(alpha))
            for alpha in alphas
        )
        and len({float(alpha) for alpha in alphas}) == len(alphas)
    )


def provenance_is_complete(provenance: dict[str, Any]) -> bool:
    try:
        source = provenance["source"]
        model = provenance["model"]
        lens = provenance["lens"]
        software = provenance["software"]
        hardware = provenance["hardware"]
        experiment = provenance["experiment"]
        return all(
            (
                _is_concrete_hex(source["commit"], HEX_40),
                isinstance(source["dirty"], bool),
                model["id"] == APPROVED_ASSETS.model_id,
                _is_concrete_hex(model["resolved_revision"], HEX_40),
                model["layer_count"] == APPROVED_MODEL_LAYER_COUNT,
                lens["repo_id"] == APPROVED_ASSETS.lens_repo_id,
                lens["filename"] == APPROVED_ASSETS.lens_filename,
                _is_concrete_hex(lens["resolved_revision"], HEX_40),
                _is_concrete_hex(lens["sha256"], HEX_64),
                isinstance(lens["size_bytes"], int),
                not isinstance(lens["size_bytes"], bool),
                lens["size_bytes"] > 0,
                lens["integrity_verified"] is True,
                software["jspace_commit"] == JSPACE_COMMIT,
                all(
                    _is_concrete_text(software[name])
                    for name in (
                        "python",
                        "pytorch",
                        "transformers",
                        "huggingface_hub",
                        "jspace",
                    )
                ),
                _is_concrete_text(hardware["gpu_name"]),
                _is_concrete_text(hardware["cuda_version"]),
                hardware["dtype"] == APPROVED_ASSETS.dtype,
                isinstance(experiment["seed"], int),
                not isinstance(experiment["seed"], bool),
                _valid_prompts(experiment["prompts"]),
                experiment["generation"] == APPROVED_GENERATION,
                _valid_layers(experiment["layers"], model["layer_count"]),
                _valid_alphas(experiment["alphas"]),
                experiment["intervention"] == APPROVED_INTERVENTION,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
