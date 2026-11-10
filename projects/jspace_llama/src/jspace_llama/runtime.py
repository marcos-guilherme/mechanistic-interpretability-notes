"""Runtime loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from jspace_llama.config import AssetsConfig


class JSpaceExperimentError(RuntimeError):
    """Base error for failures that make an experiment inconclusive."""


class CudaRequiredError(JSpaceExperimentError):
    pass


class AssetLoadError(JSpaceExperimentError):
    pass


class ResourceExhaustionError(JSpaceExperimentError):
    pass


class TokenizationError(JSpaceExperimentError):
    pass


class DimensionMismatchError(JSpaceExperimentError):
    pass


class LayerBandError(JSpaceExperimentError):
    pass


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    gpu_name: str
    dtype: str


@dataclass(frozen=True)
class RuntimeBundle:
    model: Any
    lens: Any
    dictionary: Any
    layers: tuple[int, ...]


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the checkout root from a notebook, Colab cwd, or package path."""
    origins = [
        Path(start).resolve() if start is not None else Path.cwd().resolve(),
        Path(__file__).resolve(),
    ]
    visited: set[Path] = set()
    for origin in origins:
        current = origin.parent if origin.is_file() else origin
        for candidate in (current, *current.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "src" / "jspace_llama").is_dir()
            ):
                return candidate
    raise RuntimeError(
        "could not locate the jspace-llama project root from cwd or package path"
    )


def render_chat_prompt(tokenizer: Any, instruction: str) -> str:
    """Apply the tokenizer's instruction template exactly once."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def require_cuda(torch_module: Any = torch) -> RuntimeInfo:
    if not torch_module.cuda.is_available():
        raise CudaRequiredError("CUDA is required for the real J-space run")
    return RuntimeInfo(
        device="cuda",
        gpu_name=str(torch_module.cuda.get_device_name(0)),
        dtype="bfloat16",
    )


def download_lens_file(
    assets: AssetsConfig,
    *,
    downloader: Callable[..., str] | None = None,
) -> str:
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    try:
        return downloader(
            repo_id=assets.lens_repo_id,
            filename=assets.lens_filename,
        )
    except Exception as exc:
        raise AssetLoadError(
            f"failed to download lens {assets.lens_repo_id}/{assets.lens_filename}"
        ) from exc


def validate_intervention_tokens(
    tokenizer: Any, concepts: Iterable[str]
) -> dict[str, int]:
    """Require one token for each leading-space dictionary concept."""
    token_ids: dict[str, int] = {}
    for concept in concepts:
        candidate = " " + concept.lstrip()
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) != 1:
            raise TokenizationError(
                f"{concept!r} must be exactly one token as {candidate!r}; got {ids!r}"
            )
        token_ids[concept] = int(ids[0])
    return token_ids


def validate_model_lens(model: Any, lens: Any, layers: Iterable[int]) -> None:
    if model.d_model != lens.d_model:
        raise DimensionMismatchError(
            f"model d_model={model.d_model} does not match lens d_model={lens.d_model}"
        )
    chosen = list(layers)
    out_of_range = [layer for layer in chosen if not 0 <= layer < model.n_layers]
    if out_of_range:
        raise LayerBandError(
            f"layers {out_of_range} are outside model range [0, {model.n_layers})"
        )
    missing = [layer for layer in chosen if layer not in set(lens.source_layers)]
    if missing:
        raise LayerBandError(
            f"layers {missing} are not fitted; fitted layers are {lens.source_layers}"
        )
    for layer in chosen:
        shape = tuple(lens.jacobians[layer].shape)
        expected = (model.d_model, model.d_model)
        if shape != expected:
            raise DimensionMismatchError(
                f"lens layer {layer} has shape {shape}, expected {expected}"
            )


def resolve_layer_band(
    model: Any,
    lens: Any,
    requested: Iterable[int] | None,
    *,
    default_band_fn: Callable[..., list[int]] | None = None,
) -> list[int]:
    if default_band_fn is None:
        from jspace import default_band

        default_band_fn = default_band
    requested_list = None if requested is None else list(requested)
    try:
        layers = list(default_band_fn(model, lens, requested_list))
    except ValueError as exc:
        raise LayerBandError(str(exc)) from exc
    if not layers:
        raise LayerBandError("layer band contains no fitted layers")
    validate_model_lens(model, lens, layers)
    return layers


def load_components(
    assets: AssetsConfig,
    lens_path: str | Path,
    *,
    layers: Iterable[int] | None = None,
    hf_model_cls: Any = None,
    lens_cls: Any = None,
    dictionary_cls: Any = None,
) -> RuntimeBundle:
    """Load the pinned model, lens, and on-demand dictionary on CUDA."""
    if hf_model_cls is None or lens_cls is None or dictionary_cls is None:
        from jspace import HFModel, JLensDictionary, JacobianLens

        hf_model_cls = hf_model_cls or HFModel
        lens_cls = lens_cls or JacobianLens
        dictionary_cls = dictionary_cls or JLensDictionary
    try:
        model = hf_model_cls.load(
            assets.model_id,
            dtype=torch.bfloat16,
            device="cuda",
        )
        lens = lens_cls.load(str(lens_path))
        resolved_layers = resolve_layer_band(model, lens, layers)
        dictionary = dictionary_cls(model, lens, device="cuda")
    except torch.OutOfMemoryError as exc:
        raise ResourceExhaustionError(
            f"CUDA out of memory while loading {assets.model_id}: {exc}"
        ) from exc
    except (DimensionMismatchError, LayerBandError):
        raise
    except Exception as exc:
        raise AssetLoadError(
            f"failed to load model/lens assets for {assets.model_id}"
        ) from exc
    return RuntimeBundle(
        model=model,
        lens=lens,
        dictionary=dictionary,
        layers=tuple(resolved_layers),
    )
