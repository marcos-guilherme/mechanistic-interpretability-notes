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
    translate_huggingface_error,
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


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    canonical = (
        "["
        + ",".join(str(int(token_id)) for token_id in token_ids)
        + "]"
    )
    return text_sha256(canonical)


def _runtime_source_files(
    project_root: Path,
    repository_root: Path,
) -> list[tuple[str, Path]]:
    project = project_root.resolve()
    repository = repository_root.resolve()
    candidates: list[Path] = []
    for pattern in (
        "data/**/*",
        "src/**/*.py",
        "notebooks/*.ipynb",
        "schemas/*.json",
        "scripts/*.py",
    ):
        candidates.extend(project.glob(pattern))
    candidates.extend(
        project / name for name in ("pyproject.toml", "uv.lock")
    )
    runner = repository / "tools" / "colab" / "run_jspace.py"
    if runner.is_file():
        candidates.append(runner)

    labeled: list[tuple[str, Path]] = []
    for path in candidates:
        if not path.is_file() or path.name.endswith("_output.ipynb"):
            continue
        try:
            relative = path.resolve().relative_to(project)
        except ValueError:
            if path.resolve() != runner.resolve():
                continue
            label = "tools/colab/run_jspace.py"
        else:
            label = f"projects/jspace_llama/{relative.as_posix()}"
        labeled.append((label, path))
    return sorted(set(labeled), key=lambda item: item[0])


def runtime_source_sha256(
    project_root: str | Path,
    repository_root: str | Path | None = None,
) -> str:
    project = Path(project_root)
    repository = (
        Path(repository_root)
        if repository_root is not None
        else project.resolve().parents[1]
    )
    digest = hashlib.sha256()
    files = _runtime_source_files(project, repository)
    if not files:
        raise ValueError("no runtime-critical source files found")
    for label, path in files:
        label_bytes = label.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


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
    legacy_fields = {"schema_version", "commit", "dirty"}
    current_fields = legacy_fields | {"scoped_dirty", "archive_sha256"}
    observation_fields = current_fields | {"runtime_source_sha256"}
    if not isinstance(payload, dict) or set(payload) not in {
        frozenset(legacy_fields),
        frozenset(current_fields),
        frozenset(observation_fields),
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
    source = {"commit": commit, "dirty": dirty}
    if "scoped_dirty" in payload:
        scoped_dirty = payload["scoped_dirty"]
        archive_sha256 = payload["archive_sha256"]
        if not isinstance(scoped_dirty, bool):
            raise ValueError("source provenance scoped_dirty must be boolean")
        if not _is_concrete_hex(archive_sha256, HEX_64):
            raise ValueError("source provenance archive_sha256 must be concrete")
        source.update(
            scoped_dirty=scoped_dirty,
            archive_sha256=archive_sha256,
        )
    return source


def read_observation_source_provenance(
    project_root: str | Path,
) -> dict[str, Any]:
    path = Path(project_root) / SOURCE_PROVENANCE_FILENAME
    payload = strict_json_loads(path.read_bytes())
    expected = {
        "schema_version",
        "commit",
        "dirty",
        "scoped_dirty",
        "archive_sha256",
        "runtime_source_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("observation source provenance has unexpected fields")
    if payload["schema_version"] != 1:
        raise ValueError("source provenance schema_version must be 1")
    commit = payload["commit"]
    dirty = payload["dirty"]
    scoped_dirty = payload["scoped_dirty"]
    archive_digest = payload["archive_sha256"]
    runtime_digest = payload["runtime_source_sha256"]
    if not isinstance(commit, str) or HEX_40.fullmatch(commit) is None:
        raise ValueError("source provenance commit must be a 40-character hash")
    if not isinstance(dirty, bool) or not isinstance(scoped_dirty, bool):
        raise ValueError("source provenance dirty fields must be boolean")
    if not _is_concrete_hex(archive_digest, HEX_64):
        raise ValueError("source provenance archive_sha256 must be concrete")
    if not _is_concrete_hex(runtime_digest, HEX_64):
        raise ValueError(
            "source provenance runtime_source_sha256 must be concrete"
        )
    return {
        "commit": commit,
        "dirty": dirty,
        "scoped_dirty": scoped_dirty,
        "runtime_source_sha256": runtime_digest,
    }


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
        raise translate_huggingface_error(
            error,
            stage="revision_resolution",
        ) from error
    except OSError as error:
        raise AssetLoadError(
            "failed to resolve pinned Hugging Face revisions",
            stage="revision_resolution",
            category="io_error",
        ) from error
    for label, revision in (
        ("model", model_revision),
        ("lens", lens_revision),
    ):
        if not _is_concrete_hex(revision, HEX_40):
            raise AssetLoadError(
                f"resolved {label} revision is not a commit hash",
                stage="revision_resolution",
                category="asset_failure",
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
                "token_ids_sha256": token_ids_sha256(prompt.token_ids),
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
            or not _is_concrete_hex(
                prompt.get("token_ids_sha256"),
                HEX_64,
            )
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
                source["scoped_dirty"] is False,
                _is_concrete_hex(source["archive_sha256"], HEX_64),
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
