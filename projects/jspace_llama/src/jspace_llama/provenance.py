"""Reproducible, non-secret provenance collection."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from typing import Any, Iterable


JSPACE_COMMIT = "04c822a255fe0a10d84e08ec499e5e09f4b7d37a"
SOURCE_PROVENANCE_FILENAME = "source_provenance.json"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    model_revision = str(api.model_info(model_id).sha)
    lens_revision = str(api.model_info(lens_repo_id).sha)
    for label, revision in (
        ("model", model_revision),
        ("lens", lens_revision),
    ):
        if HEX_40.fullmatch(revision) is None:
            raise ValueError(f"resolved {label} revision is not a commit hash")
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
        },
        "layers": [int(layer) for layer in layers],
        "alphas": [float(alpha) for alpha in config.sweep.alphas],
        "intervention": {
            "source_concept": str(config.source),
            "target_concept": str(config.target),
            "prefill_only": True,
        },
    }


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
                HEX_40.fullmatch(source["commit"]) is not None,
                isinstance(source["dirty"], bool),
                bool(model["id"]),
                HEX_40.fullmatch(model["resolved_revision"]) is not None,
                bool(lens["repo_id"]),
                bool(lens["filename"]),
                HEX_40.fullmatch(lens["resolved_revision"]) is not None,
                HEX_64.fullmatch(lens["sha256"]) is not None,
                int(lens["size_bytes"]) > 0,
                software["jspace_commit"] == JSPACE_COMMIT,
                all(
                    bool(software[name])
                    and software[name] != "unknown"
                    for name in (
                        "python",
                        "pytorch",
                        "transformers",
                        "huggingface_hub",
                        "jspace",
                    )
                ),
                bool(hardware["gpu_name"]),
                bool(hardware["cuda_version"]),
                bool(hardware["dtype"]),
                isinstance(experiment["seed"], int),
                bool(experiment["prompts"]),
                all(
                    bool(prompt["text"])
                    and (
                        HEX_64.fullmatch(prompt["instruction_sha256"])
                        is not None
                    )
                    and HEX_64.fullmatch(
                        prompt["rendered_prompt_sha256"]
                    )
                    is not None
                    for prompt in experiment["prompts"]
                ),
                bool(experiment["layers"]),
                bool(experiment["alphas"]),
                experiment["intervention"]["prefill_only"] is True,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
