from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from .directions import CandidateDirection, load_direction
from .pipeline import dtype_from_name
from .qwen_dose import EXPECTED_SOURCE_LAYER, sha256_file
from .qwen_jlens import QWEN_MODEL, QWEN_REVISION


HF_ABLATED_PROFILE = "qwen25_coder_7b_hf_ablated.json"
EXPECTED_MODEL_NAME = QWEN_MODEL
EXPECTED_MODEL_REVISION = QWEN_REVISION
EXPECTED_ABLATION_COEFFICIENT = 0.2
EXPECTED_DIRECTION_SHA256 = (
    "0b1cff5e7013a80e609de91dde6dfba646c9ab6879c488154008576a24f0a086"
)
EXPECTED_SOURCE_DIRECTION = "artifacts/qwen25_coder_7b/selected_direction.pt"
EXPORT_MANIFEST_FILENAME = "provenance.json"
REMOTE_CODE_FILES = (
    "configuration_qwen2_refusal_ablated.py",
    "modeling_qwen2_refusal_ablated.py",
)
_ALLOWLIST = re.compile(
    r"^(?:"
    r"config\.json|"
    r"generation_config\.json|"
    r"model\.safetensors|"
    r"model\.safetensors\.index\.json|"
    r"model-\d{5}-of-\d{5}\.safetensors|"
    r"tokenizer\.json|"
    r"tokenizer_config\.json|"
    r"special_tokens_map\.json|"
    r"vocab\.json|"
    r"merges\.txt|"
    r"added_tokens\.json|"
    r"chat_template\.jinja|"
    r"configuration_qwen2_refusal_ablated\.py|"
    r"modeling_qwen2_refusal_ablated\.py|"
    r"provenance\.json"
    r")$"
)
_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "configuration_qwen2_refusal_ablated.py",
        "modeling_qwen2_refusal_ablated.py",
        EXPORT_MANIFEST_FILENAME,
    }
)
LoadPretrained = Callable[..., tuple[Any, Any]]


@dataclass(frozen=True)
class HuggingFaceAblatedExport:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def tensor_sha256(vector: torch.Tensor) -> str:
    compact = vector.detach().float().cpu().contiguous()
    return hashlib.sha256(compact.numpy().tobytes()).hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_hf_package_path() -> Path:
    root = _project_root()
    location = str(root)
    if location not in sys.path:
        sys.path.insert(0, location)
    return root


def _remote_code_dir() -> Path:
    return _project_root() / "hf_qwen_ablated"


def _exported_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [path for path in directory.iterdir() if path.is_file()]
    nested = [path for path in directory.iterdir() if path.is_dir()]
    if nested:
        raise ValueError("export directory must not contain nested directories")
    return files


def is_export_allowlisted(name: str) -> bool:
    return _ALLOWLIST.fullmatch(name) is not None


def _has_weight_file(names: set[str]) -> bool:
    if "model.safetensors" in names:
        return True
    return any(
        re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in names
    )


def _is_complete_export(directory: Path) -> bool:
    names = {path.name for path in _exported_files(directory)}
    return _REQUIRED_FILES.issubset(names) and _has_weight_file(names) and all(
        is_export_allowlisted(name) for name in names
    )


def _assert_output_directory(output_dir: Path) -> None:
    existing = _exported_files(output_dir) if output_dir.exists() else []
    if not existing:
        return
    if not _is_complete_export(output_dir):
        raise FileExistsError(
            "refusing to overwrite incomplete export directory"
        )
    raise FileExistsError("refusing to overwrite existing export directory")


def _assert_allowlisted(output_dir: Path) -> None:
    names = {path.name for path in _exported_files(output_dir)}
    illegal = sorted(name for name in names if not is_export_allowlisted(name))
    if illegal:
        raise ValueError(f"export files outside allowlist: {illegal}")
    missing = sorted(_REQUIRED_FILES - names)
    if missing or not _has_weight_file(names):
        raise ValueError(f"export is incomplete: missing {missing}")


def validate_hf_ablated_config(config: Mapping[str, Any]) -> None:
    if config.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"model_name {config.get('model_name')!r} != {EXPECTED_MODEL_NAME!r}"
        )
    revision = str(config.get("model_revision", ""))
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"revision {revision!r} != expected {EXPECTED_MODEL_REVISION!r}"
        )
    coefficient = float(config["ablation_coefficient"])
    if not math.isclose(coefficient, EXPECTED_ABLATION_COEFFICIENT):
        raise ValueError(
            f"coefficient {coefficient!r} != {EXPECTED_ABLATION_COEFFICIENT}"
        )
    layer = int(config["source_layer"])
    if layer != EXPECTED_SOURCE_LAYER:
        raise ValueError(f"source_layer {layer!r} != {EXPECTED_SOURCE_LAYER}")
    if config.get("source_direction") != EXPECTED_SOURCE_DIRECTION:
        raise ValueError("source_direction must remain the frozen Qwen artifact")
    digest = str(config.get("direction_sha256", ""))
    if digest != EXPECTED_DIRECTION_SHA256:
        raise ValueError(
            f"direction_sha256 {digest!r} != expected {EXPECTED_DIRECTION_SHA256!r}"
        )


def load_hf_ablated_config(
    root: Path,
    profile: str = HF_ABLATED_PROFILE,
) -> dict[str, Any]:
    path = Path(root) / "configs" / profile
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("HF ablated profile must contain a JSON object")
    validate_hf_ablated_config(config)
    return config


def _require_export_identity(config: Mapping[str, Any]) -> None:
    if config.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"model_name {config.get('model_name')!r} != {EXPECTED_MODEL_NAME!r}"
        )
    revision = str(config.get("model_revision", ""))
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"revision {revision!r} != expected {EXPECTED_MODEL_REVISION!r}"
        )
    coefficient = float(config["ablation_coefficient"])
    if not math.isclose(coefficient, EXPECTED_ABLATION_COEFFICIENT):
        raise ValueError(
            f"coefficient {coefficient!r} != {EXPECTED_ABLATION_COEFFICIENT}"
        )
    layer = int(config["source_layer"])
    if layer != EXPECTED_SOURCE_LAYER:
        raise ValueError(f"source_layer {layer!r} != {EXPECTED_SOURCE_LAYER}")


def _verify_direction(
    path: Path,
    *,
    expected_sha256: str,
    expected_layer: int,
) -> CandidateDirection:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"direction hash {actual!r} != expected {expected_sha256!r}"
        )
    candidate = load_direction(path)
    if int(candidate.layer) != expected_layer:
        raise ValueError(
            f"direction layer {candidate.layer!r} != {expected_layer}"
        )
    return candidate


def _default_load_pretrained(
    model_name: str,
    *,
    dtype: torch.dtype,
    revision: str,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "revision": revision}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    model.requires_grad_(False)
    return model, tokenizer


def _build_ablated_model(
    base_model: Any,
    candidate: CandidateDirection,
    *,
    coefficient: float,
    direction_sha256: str,
    model_name: str,
    revision: str,
) -> Any:
    _ensure_hf_package_path()
    from hf_qwen_ablated.configuration_qwen2_refusal_ablated import (
        Qwen2RefusalAblatedConfig,
    )
    from hf_qwen_ablated.modeling_qwen2_refusal_ablated import (
        Qwen2RefusalAblatedForCausalLM,
    )

    payload = dict(base_model.config.to_dict())
    for key in ("architectures", "auto_map", "model_type"):
        payload.pop(key, None)
    config = Qwen2RefusalAblatedConfig(
        ablation_coefficient=coefficient,
        direction_sha256=direction_sha256,
        source_layer=int(candidate.layer),
        base_model_name=model_name,
        base_model_revision=revision,
        architectures=["Qwen2RefusalAblatedForCausalLM"],
        **payload,
    )
    model = Qwen2RefusalAblatedForCausalLM(config)
    incompatible = model.load_state_dict(base_model.state_dict(), strict=False)
    leftover = [
        key
        for key in incompatible.missing_keys
        if key != "refusal_direction"
    ]
    if leftover:
        raise ValueError(f"missing base weights: {leftover}")
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected base weights: {list(incompatible.unexpected_keys)}"
        )
    model.refusal_direction.copy_(
        candidate.vector.to(
            device=model.refusal_direction.device,
            dtype=model.refusal_direction.dtype,
        )
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _copy_remote_code(output_dir: Path) -> None:
    source = _remote_code_dir()
    for name in REMOTE_CODE_FILES:
        shutil.copy2(source / name, output_dir / name)


def _file_hashes(output_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(_exported_files(output_dir), key=lambda item: item.name):
        if path.name == EXPORT_MANIFEST_FILENAME:
            continue
        hashes[path.name] = sha256_file(path)
    return hashes


def _write_manifest(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    candidate: CandidateDirection,
    direction_sha256: str,
) -> dict[str, Any]:
    code_paths = [
        Path(__file__),
        *(_remote_code_dir() / name for name in REMOTE_CODE_FILES),
    ]
    digest = hashlib.sha256()
    for path in code_paths:
        digest.update(path.read_bytes())
    manifest = {
        "schema_version": 1,
        "model_name": EXPECTED_MODEL_NAME,
        "base_model_revision": EXPECTED_MODEL_REVISION,
        "ablation_coefficient": EXPECTED_ABLATION_COEFFICIENT,
        "source_layer": EXPECTED_SOURCE_LAYER,
        "source_direction": config["source_direction"],
        "direction_sha256": direction_sha256,
        "direction_vector_sha256": tensor_sha256(candidate.vector),
        "export_code_sha256": digest.hexdigest(),
        "auto_map": {
            "AutoConfig": (
                "configuration_qwen2_refusal_ablated.Qwen2RefusalAblatedConfig"
            ),
            "AutoModelForCausalLM": (
                "modeling_qwen2_refusal_ablated.Qwen2RefusalAblatedForCausalLM"
            ),
        },
        "files": _file_hashes(output_dir),
    }
    path = output_dir / EXPORT_MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_qwen_hf_ablated(
    root: Path,
    *,
    output_dir: Path | None = None,
    config: Mapping[str, Any] | None = None,
    profile: str = HF_ABLATED_PROFILE,
    load_pretrained: LoadPretrained | None = None,
) -> HuggingFaceAblatedExport:
    root = Path(root).resolve()
    resolved = dict(config or load_hf_ablated_config(root, profile=profile))
    _require_export_identity(resolved)
    output = Path(
        output_dir
        if output_dir is not None
        else root / str(resolved["artifacts_dir"])
    ).resolve()
    _assert_output_directory(output)
    direction_path = root / str(resolved["source_direction"])
    candidate = _verify_direction(
        direction_path,
        expected_sha256=str(resolved["direction_sha256"]),
        expected_layer=int(resolved["source_layer"]),
    )
    loader = load_pretrained or _default_load_pretrained
    base_model, tokenizer = loader(
        str(resolved["model_name"]),
        dtype=dtype_from_name(str(resolved["dtype"])),
        revision=str(resolved["model_revision"]),
    )
    model = _build_ablated_model(
        base_model,
        candidate,
        coefficient=float(resolved["ablation_coefficient"]),
        direction_sha256=str(resolved["direction_sha256"]),
        model_name=str(resolved["model_name"]),
        revision=str(resolved["model_revision"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    _copy_remote_code(output)
    manifest = _write_manifest(
        output,
        config=resolved,
        candidate=candidate,
        direction_sha256=str(resolved["direction_sha256"]),
    )
    _assert_allowlisted(output)
    return HuggingFaceAblatedExport(
        output_dir=output,
        manifest_path=output / EXPORT_MANIFEST_FILENAME,
        manifest=manifest,
    )
