"""Qwen-specific Jacobian-lens corpus, probe, checkpoint and holdout helpers.

Jacobian math stays in upstream ``jlens.fitting.jacobian_for_prompt``. This
module owns corpus identity, probe policy, resumable checkpoint identity and
compact holdout arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import gc
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from .data import normalize_prompt_text
from .qwen_dose import METADATA_LEAK_KEYS, sha256_file


JLENS_PROFILE = "qwen25_coder_7b_jlens.json"
JLENS_V2_PROFILE = "qwen25_coder_7b_jlens_v2.json"
QWEN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
JLENS_UPSTREAM = "anthropics/jacobian-lens"
JLENS_UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
JLENS_UPSTREAM_SPEC = (
    f"git+https://github.com/{JLENS_UPSTREAM}.git@{JLENS_UPSTREAM_COMMIT}"
)
RUNTIME_COMPATIBILITY_SMOKE = (
    "2026-11-16-qwen7b-hooks-jlens-transport-unembed"
)
DATASET_REPO_ID = "code-search-net/code_search_net"
DATASET_CONFIG = "python"
DATASET_SPLIT = "train"
DATASET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
DATASET_IDENTITY = {
    "dataset_repo_id": DATASET_REPO_ID,
    "dataset_config": DATASET_CONFIG,
    "dataset_split": DATASET_SPLIT,
    "dataset_revision": DATASET_REVISION,
}
SOURCE_DOSE_RUN_ID = "d76d0e7f-1e86-4953-8eac-de1c352708d2"
EXPECTED_N_LAYERS = 28
EXPECTED_D_MODEL = 3584
EXPECTED_SOURCE_LAYERS = list(range(27))
EXPECTED_TARGET_LAYER = 27
SKIP_FIRST = 16
MAX_SEQ_LEN = 128
MIN_TOKENS_BEFORE_TRUNCATION = 128
FIT_COUNT = 100
HOLDOUT_COUNT = 20
V2_MAX_SEQ_LEN = 512
V2_HOLDOUT_COUNT = 200
V2_FIT_COUNT_CANDIDATES = [100, 150, 200, 250, 300, 400, 600, 800]
V2_MAX_FIT_COUNT = 800
DIM_BATCH_PROBE_ORDER = [32, 16, 8]
MAX_PROJECTED_SECONDS = 12 * 3600
CHECKPOINT_FILENAME = "jlens_fit_checkpoint.pt"
FINAL_LENS_FILENAME = "qwen25_coder_7b_jacobian_lens.pt"
V2_FINAL_LENS_FILENAME = "qwen25_coder_7b_jacobian_lens_v2.pt"
CHECKPOINT_IDENTITY_FIELDS = (
    "source_layers",
    "target_layer",
    "skip_first",
    "corpus_hash",
    "config_hash",
    "model_revision",
    "implementation_commit",
    "run_id",
)
JLENS_LEAK_KEYS = METADATA_LEAK_KEYS | {
    "code",
    "docstring",
    "docstrings",
    "func_code",
    "func_code_string",
    "func_documentation_string",
    "whole_func_string",
    "fit_text",
}
REQUIRED_JLENS_SYMBOLS = (
    "from_hf",
    "jacobian_for_prompt",
    "JacobianLens",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_dataset_identity(config: Mapping[str, Any]) -> None:
    for field_name, expected in DATASET_IDENTITY.items():
        actual = config.get(field_name)
        if actual != expected:
            raise ValueError(
                f"dataset identity {field_name}: {actual!r} != {expected!r}"
            )


class DimBatchOOMError(RuntimeError):
    def __init__(self, attempts: Sequence[Mapping[str, Any]]) -> None:
        super().__init__("dim_batch 8 OOMed")
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class ExclusionHashes:
    available: bool
    hashes: frozenset[str]
    source_path: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class CorpusItem:
    example_id: str
    content_hash: str
    token_count: int
    fit_text: str
    source_index: int


@dataclass(frozen=True)
class JLensCorpus:
    fit: tuple[CorpusItem, ...]
    holdout: tuple[CorpusItem, ...]
    dataset_revision: str
    n_scanned: int
    n_missing_code_or_docstring: int
    n_too_short: int
    n_duplicate_dropped: int
    n_excluded: int
    exclusions_available: bool
    n_selected: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_selected", len(self.fit) + len(self.holdout)
        )


@dataclass(frozen=True)
class ProbeAttempt:
    dim_batch: int
    outcome: str
    wall_seconds: float | None = None


@dataclass(frozen=True)
class ProbeResult:
    selected: int
    attempts: tuple[dict[str, Any], ...]
    wall_seconds: float


@dataclass(frozen=True)
class DurationProjection:
    probe_seconds: float
    n_prompts: int
    projected_seconds: float
    limit_seconds: int
    exceeds_limit: bool


def content_hash_text(text: str) -> str:
    return hashlib.sha256(normalize_prompt_text(text).encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def config_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def corpus_hash(corpus: JLensCorpus) -> str:
    payload = [
        {
            "example_id": item.example_id,
            "content_hash": item.content_hash,
            "token_count": item.token_count,
            "split": split,
        }
        for split, items in (("fit", corpus.fit), ("holdout", corpus.holdout))
        for item in items
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def load_exclusion_hashes(path: Path | None) -> ExclusionHashes:
    if path is None or not path.is_file():
        return ExclusionHashes(available=False, hashes=frozenset())
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not SHA256_RE.fullmatch(stripped):
            raise ValueError(f"exclusion hash is not a valid SHA-256: {stripped}")
        hashes.add(stripped)
    return ExclusionHashes(
        available=True,
        hashes=frozenset(hashes),
        source_path=str(path),
        source_sha256=sha256_file(path),
    )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fit_text(row: Mapping[str, Any]) -> str:
    whole = row.get("whole_func_string")
    if _nonempty(whole):
        return str(whole)
    code = str(row.get("func_code_string") or "")
    docstring = str(row.get("func_documentation_string") or "")
    return f"{code}\n{docstring}".strip()


def _token_count(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=True)
        return len(encoded)
    encoded = tokenizer(text, add_special_tokens=True)
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        return len(ids)
    return len(encoded)


def _row_id(row: Mapping[str, Any], index: int) -> str:
    repo = str(row.get("repository_name") or "unknown")
    path = str(row.get("func_path_in_repository") or "unknown")
    name = str(row.get("func_name") or f"fn_{index}")
    digest = hashlib.sha256(f"{repo}:{path}:{name}:{index}".encode()).hexdigest()[:16]
    return f"csn:{index}:{digest}"


def build_codesearchnet_corpus(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    fit_count: int = FIT_COUNT,
    holdout_count: int = HOLDOUT_COUNT,
    min_tokens: int = MIN_TOKENS_BEFORE_TRUNCATION,
    dataset_revision: str,
    exclusion_hashes: frozenset[str] = frozenset(),
    exclusions_available: bool = False,
) -> JLensCorpus:
    needed = fit_count + holdout_count
    selected: list[CorpusItem] = []
    seen_hashes: set[str] = set()
    n_scanned = 0
    n_missing = 0
    n_too_short = 0
    n_duplicate = 0
    n_excluded = 0
    for index, row in enumerate(rows):
        n_scanned += 1
        code = row.get("func_code_string")
        docstring = row.get("func_documentation_string")
        if not _nonempty(code) or not _nonempty(docstring):
            n_missing += 1
            continue
        text = _fit_text(row)
        digest = content_hash_text(text)
        if digest in seen_hashes:
            n_duplicate += 1
            continue
        if digest in exclusion_hashes:
            n_excluded += 1
            continue
        tokens = _token_count(tokenizer, text)
        if tokens < min_tokens:
            n_too_short += 1
            continue
        seen_hashes.add(digest)
        selected.append(
            CorpusItem(
                example_id=_row_id(row, index),
                content_hash=digest,
                token_count=tokens,
                fit_text=text,
                source_index=index,
            )
        )
        if len(selected) >= needed:
            break
    if len(selected) < needed:
        raise ValueError(
            f"need {needed} unique CodeSearchNet sequences, selected {len(selected)}"
        )
    return JLensCorpus(
        fit=tuple(selected[:fit_count]),
        holdout=tuple(selected[fit_count:needed]),
        dataset_revision=dataset_revision,
        n_scanned=n_scanned,
        n_missing_code_or_docstring=n_missing,
        n_too_short=n_too_short,
        n_duplicate_dropped=n_duplicate,
        n_excluded=n_excluded,
        exclusions_available=exclusions_available,
    )


def compact_corpus_records(
    corpus: JLensCorpus,
    *,
    exclusion: ExclusionHashes | None = None,
) -> dict[str, Any]:
    def _items(items: Sequence[CorpusItem]) -> list[dict[str, Any]]:
        return [
            {
                "example_id": item.example_id,
                "content_hash": item.content_hash,
                "token_count": item.token_count,
                "source_index": item.source_index,
            }
            for item in items
        ]

    payload = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": corpus.dataset_revision,
        "fit_count": len(corpus.fit),
        "holdout_count": len(corpus.holdout),
        "n_selected": corpus.n_selected,
        "n_scanned": corpus.n_scanned,
        "n_missing_code_or_docstring": corpus.n_missing_code_or_docstring,
        "n_too_short": corpus.n_too_short,
        "n_duplicate_dropped": corpus.n_duplicate_dropped,
        "n_excluded": corpus.n_excluded,
        "exclusions_available": corpus.exclusions_available,
        "benchmark_deduplicated": bool(corpus.exclusions_available),
        "fit": _items(corpus.fit),
        "holdout": _items(corpus.holdout),
        "fit_ids": [item.example_id for item in corpus.fit],
        "holdout_ids": [item.example_id for item in corpus.holdout],
        "fit_hashes": [item.content_hash for item in corpus.fit],
        "holdout_hashes": [item.content_hash for item in corpus.holdout],
        "corpus_hash": corpus_hash(corpus),
    }
    if exclusion is not None and exclusion.available:
        payload["exclusion_file_sha256"] = exclusion.source_sha256
        payload["exclusion_file_path"] = exclusion.source_path
    return payload


def jlens_metadata_contains_leak(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).casefold() in JLENS_LEAK_KEYS:
                return True
            if jlens_metadata_contains_leak(value):
                return True
        return False
    if isinstance(obj, list):
        return any(jlens_metadata_contains_leak(item) for item in obj)
    return False


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).casefold()
    return "out of memory" in message and "cuda" in message


def _drop_exception_frames(exc: BaseException) -> None:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _release_cuda_after_oom() -> None:
    gc.collect()
    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def probe_dim_batch(
    runner: Callable[[int], Any],
    *,
    order: Sequence[int] = DIM_BATCH_PROBE_ORDER,
    clock: Callable[[], float] | None = None,
) -> ProbeResult:
    import time

    now = clock or time.perf_counter
    attempts: list[dict[str, Any]] = []
    for dim_batch in order:
        started = now()
        try:
            runner(int(dim_batch))
        except Exception as exc:
            elapsed = now() - started
            if is_cuda_oom(exc):
                attempts.append(
                    {
                        "dim_batch": int(dim_batch),
                        "outcome": "cuda_oom",
                        "wall_seconds": elapsed,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                _drop_exception_frames(exc)
                _release_cuda_after_oom()
                continue
            raise
        elapsed = now() - started
        attempts.append(
            {
                "dim_batch": int(dim_batch),
                "outcome": "ok",
                "wall_seconds": elapsed,
            }
        )
        return ProbeResult(
            selected=int(dim_batch),
            attempts=tuple(attempts),
            wall_seconds=elapsed,
        )
    raise DimBatchOOMError(attempts)


def select_dim_batch(result: ProbeResult) -> int:
    return int(result.selected)


def project_fit_duration(
    *,
    probe_seconds: float,
    n_prompts: int = FIT_COUNT,
    limit_seconds: int = MAX_PROJECTED_SECONDS,
) -> DurationProjection:
    projected = float(probe_seconds) * int(n_prompts)
    return DurationProjection(
        probe_seconds=float(probe_seconds),
        n_prompts=int(n_prompts),
        projected_seconds=projected,
        limit_seconds=int(limit_seconds),
        exceeds_limit=projected > limit_seconds,
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_fit_checkpoint(
    path: Path,
    *,
    jacobian_sum: Mapping[int, torch.Tensor],
    n_done: int,
    next_idx: int,
    source_layers: Sequence[int],
    target_layer: int,
    skip_first: int,
    corpus_hash: str,
    config_hash: str,
    model_revision: str,
    implementation_commit: str,
    run_id: str,
    dim_batch: int,
) -> None:
    payload = {
        "jacobian_sum": {int(layer): tensor for layer, tensor in jacobian_sum.items()},
        "n_done": int(n_done),
        "next_idx": int(next_idx),
        "source_layers": [int(layer) for layer in source_layers],
        "target_layer": int(target_layer),
        "skip_first": int(skip_first),
        "corpus_hash": corpus_hash,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "implementation_commit": implementation_commit,
        "run_id": run_id,
        "dim_batch": int(dim_batch),
    }
    _atomic_torch_save(payload, path)


def load_fit_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"checkpoint is corrupt or unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is corrupt or unreadable: {path}")
    missing = [field for field in CHECKPOINT_IDENTITY_FIELDS if field not in payload]
    if missing or "jacobian_sum" not in payload or "n_done" not in payload or "next_idx" not in payload:
        raise ValueError(f"checkpoint is missing identity fields: {path}")
    return payload


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    target_layer: int,
    skip_first: int,
    corpus_hash: str,
    config_hash: str,
    model_revision: str,
    implementation_commit: str,
    run_id: str,
) -> None:
    expected = {
        "source_layers": [int(layer) for layer in source_layers],
        "target_layer": int(target_layer),
        "skip_first": int(skip_first),
        "corpus_hash": corpus_hash,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "implementation_commit": implementation_commit,
        "run_id": run_id,
    }
    for field_name, expected_value in expected.items():
        actual = payload.get(field_name)
        if field_name == "source_layers":
            actual = [int(layer) for layer in actual]
        if actual != expected_value:
            raise ValueError(
                f"stale checkpoint {field_name}: {actual!r} != {expected_value!r}"
            )


def _checkpoint_tensor(payload: Mapping[str, Any], layer: int) -> torch.Tensor:
    sums = payload["jacobian_sum"]
    if layer in sums:
        return sums[layer]
    if str(layer) in sums:
        return sums[str(layer)]
    raise ValueError(f"checkpoint missing layer {layer}")


def validate_checkpoint_structure(
    payload: Mapping[str, Any],
    *,
    fit_count: int = FIT_COUNT,
) -> None:
    if "n_done" not in payload or "next_idx" not in payload:
        raise ValueError("checkpoint is missing n_done/next_idx")
    n_done = int(payload["n_done"])
    next_idx = int(payload["next_idx"])
    if n_done != next_idx:
        raise ValueError(f"checkpoint n_done {n_done} != next_idx {next_idx}")
    if not 0 <= n_done <= int(fit_count):
        raise ValueError(f"checkpoint n_done {n_done} outside 0..{fit_count}")
    if "dim_batch" not in payload:
        raise ValueError("checkpoint is missing dim_batch")
    if int(payload["dim_batch"]) not in DIM_BATCH_PROBE_ORDER:
        raise ValueError(f"checkpoint dim_batch {payload['dim_batch']!r} is invalid")
    sums = payload.get("jacobian_sum")
    if not isinstance(sums, dict) or not sums:
        raise ValueError("checkpoint jacobian_sum is missing")
    for layer, tensor in sums.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint layer {layer} is not a tensor")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(
                f"checkpoint Jacobian at layer {layer} has shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"checkpoint Jacobian at layer {layer} has dtype {tensor.dtype}"
            )
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError(f"checkpoint Jacobian at layer {layer} is not finite")


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    source_layers: Sequence[int],
    d_model: int,
    dim_batch: int,
    fit_count: int = FIT_COUNT,
) -> None:
    validate_checkpoint_structure(payload, fit_count=fit_count)
    if int(payload["dim_batch"]) != int(dim_batch):
        raise ValueError(
            f"stale checkpoint dim_batch: {payload['dim_batch']!r} != {dim_batch!r}"
        )
    expected_layers = [int(layer) for layer in source_layers]
    sums = payload["jacobian_sum"]
    actual_layers = sorted(int(layer) for layer in sums)
    if actual_layers != sorted(expected_layers):
        raise ValueError(
            f"checkpoint layer keys {actual_layers!r} != {expected_layers!r}"
        )
    for layer in expected_layers:
        tensor = _checkpoint_tensor(payload, layer)
        if tuple(tensor.shape) != (int(d_model), int(d_model)):
            raise ValueError(
                f"checkpoint Jacobian at layer {layer} has shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"checkpoint Jacobian at layer {layer} has dtype {tensor.dtype}"
            )
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError(f"checkpoint Jacobian at layer {layer} is not finite")


def require_finite_logits(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise ValueError(f"{name} must be finite")


def valid_token_positions(seq_len: int, *, skip_first: int = SKIP_FIRST) -> list[int]:
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    return list(range(skip_first, max(skip_first, seq_len - 1)))


def rank_one_based(logits: torch.Tensor, token_id: int) -> int:
    vector = logits.reshape(-1)
    score = vector[int(token_id)]
    return 1 + int((vector > score).sum().item())


def _pass_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(int(rank) <= k for rank in ranks) / len(ranks)


def _metric_block(ranks: Sequence[int]) -> dict[str, float | int]:
    return {
        "n_observations": len(ranks),
        "mean_rank": float(sum(ranks) / len(ranks)) if ranks else 0.0,
        "pass@1": _pass_at_k(ranks, 1),
        "pass@5": _pass_at_k(ranks, 5),
        "pass@10": _pass_at_k(ranks, 10),
    }


def _empty_item_layer_stats() -> dict[str, int]:
    return {
        "n_positions": 0,
        "jlens_rank_sum": 0,
        "jlens_hits@1": 0,
        "jlens_hits@5": 0,
        "jlens_hits@10": 0,
        "logit_lens_rank_sum": 0,
        "logit_lens_hits@1": 0,
        "logit_lens_hits@5": 0,
        "logit_lens_hits@10": 0,
    }


def summarize_holdout_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    source_layers: Sequence[int],
) -> dict[str, Any]:
    by_layer: dict[str, dict[str, dict[str, float | int]]] = {}
    overall_jlens: list[int] = []
    overall_logit: list[int] = []
    by_item: dict[str, dict[str, dict[str, int]]] = {}
    holdout_ids: list[str] = []
    holdout_hashes: list[str] = []
    seen_ids: set[str] = set()
    for layer in source_layers:
        jlens_ranks = [
            int(record["jlens_rank"])
            for record in records
            if int(record["layer"]) == int(layer)
        ]
        logit_ranks = [
            int(record["logit_lens_rank"])
            for record in records
            if int(record["layer"]) == int(layer)
        ]
        by_layer[str(int(layer))] = {
            "jlens": _metric_block(jlens_ranks),
            "logit_lens": _metric_block(logit_ranks),
        }
    for record in records:
        example_id = str(record["example_id"])
        layer_key = str(int(record["layer"]))
        if example_id not in seen_ids:
            seen_ids.add(example_id)
            holdout_ids.append(example_id)
            if "content_hash" in record:
                holdout_hashes.append(str(record["content_hash"]))
        item_layers = by_item.setdefault(example_id, {})
        block = item_layers.setdefault(layer_key, _empty_item_layer_stats())
        jlens_rank = int(record["jlens_rank"])
        logit_rank = int(record["logit_lens_rank"])
        block["n_positions"] += 1
        block["jlens_rank_sum"] += jlens_rank
        block["jlens_hits@1"] += int(jlens_rank <= 1)
        block["jlens_hits@5"] += int(jlens_rank <= 5)
        block["jlens_hits@10"] += int(jlens_rank <= 10)
        block["logit_lens_rank_sum"] += logit_rank
        block["logit_lens_hits@1"] += int(logit_rank <= 1)
        block["logit_lens_hits@5"] += int(logit_rank <= 5)
        block["logit_lens_hits@10"] += int(logit_rank <= 10)
        overall_jlens.append(jlens_rank)
        overall_logit.append(logit_rank)
    return {
        "n_items": len(holdout_ids),
        "n_records": len(records),
        "holdout_ids": holdout_ids,
        "holdout_hashes": holdout_hashes,
        "by_item": by_item,
        "by_layer": by_layer,
        "overall": {
            "jlens": _metric_block(overall_jlens),
            "logit_lens": _metric_block(overall_logit),
        },
    }


def validate_fitted_lens(
    lens: Any,
    *,
    source_layers: Sequence[int],
    d_model: int,
    n_prompts: int,
) -> None:
    expected_layers = [int(layer) for layer in source_layers]
    actual_layers = [int(layer) for layer in getattr(lens, "source_layers")]
    if actual_layers != expected_layers:
        raise ValueError(
            f"lens source_layers {actual_layers!r} != {expected_layers!r}"
        )
    if int(lens.d_model) != int(d_model):
        raise ValueError(f"lens d_model {lens.d_model} != {d_model}")
    if int(lens.n_prompts) != int(n_prompts):
        raise ValueError(f"lens n_prompts {lens.n_prompts} != {n_prompts}")
    jacobians = lens.jacobians
    for layer in expected_layers:
        tensor = jacobians[layer]
        if tuple(tensor.shape) != (int(d_model), int(d_model)):
            raise ValueError(
                f"lens Jacobian at layer {layer} has shape {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError(f"lens Jacobian at layer {layer} is not finite")


def ensure_hf_text_config(hf_model: Any) -> Any:
    config = getattr(hf_model, "config", None)
    if config is None:
        raise ValueError("HuggingFace model is missing config")
    if not hasattr(config, "get_text_config"):
        def get_text_config(_decoder: bool = False, **_kwargs: Any) -> Any:
            return config

        config.get_text_config = get_text_config
    return hf_model


def wrap_hf_model_for_jlens(hf_model: Any, tokenizer: Any, **kwargs: Any) -> Any:
    import jlens

    ensure_hf_text_config(hf_model)
    return jlens.from_hf(hf_model, tokenizer, **kwargs)


def installed_jlens_commit(*, distribution: Any | None = None) -> str:
    if distribution is None:
        import importlib.metadata

        distribution = importlib.metadata.distribution("jlens")
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise ValueError("jlens direct_url metadata is missing")
    payload = json.loads(raw)
    commit = (payload.get("vcs_info") or {}).get("commit_id")
    if not commit:
        raise ValueError("jlens direct_url metadata is missing commit_id")
    if commit != JLENS_UPSTREAM_COMMIT:
        raise ValueError(
            f"installed jlens commit {commit!r} != {JLENS_UPSTREAM_COMMIT}"
        )
    return str(commit)


def assert_jlens_contract(
    module: Any | None = None,
    *,
    require_installed_commit: bool | None = None,
) -> None:
    target = module
    check_install = require_installed_commit
    if target is None:
        import jlens as target  # type: ignore[import-not-found]
        if check_install is None:
            check_install = True
    missing = [
        name for name in REQUIRED_JLENS_SYMBOLS if not hasattr(target, name)
    ]
    if missing:
        raise ValueError(
            "jlens contract missing symbols: " + ", ".join(missing)
        )
    fitting = getattr(target, "fitting", None)
    skip_first = getattr(fitting, "SKIP_FIRST_N_POSITIONS", None)
    if skip_first != SKIP_FIRST:
        raise ValueError(
            f"jlens skip_first is {skip_first!r}, expected {SKIP_FIRST}"
        )
    if not hasattr(fitting, "jacobian_for_prompt"):
        raise ValueError("jlens.fitting.jacobian_for_prompt is missing")
    lens_cls = target.JacobianLens
    if not hasattr(lens_cls, "save") or not hasattr(lens_cls, "load"):
        raise ValueError("JacobianLens.save/load missing")
    if check_install:
        installed_jlens_commit()


def jlens_install_command(python: str = "python") -> list[str]:
    return [python, "-m", "pip", "install", "--no-deps", JLENS_UPSTREAM_SPEC]


def parse_supported_transformers_versions(pyproject: Path) -> frozenset[str]:
    import tomllib

    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    versions: set[str] = set()
    for dependency in payload.get("project", {}).get("dependencies", []):
        if not isinstance(dependency, str) or not dependency.startswith("transformers=="):
            continue
        version = dependency.split("==", 1)[1].split(";", 1)[0].strip()
        if version:
            versions.add(version)
    if not versions:
        raise ValueError("pyproject.toml does not pin Transformers")
    return frozenset(versions)


def assert_pinned_transformers_runtime(
    pyproject: Path,
    *,
    installed: str | None = None,
) -> str:
    import importlib.metadata

    supported = parse_supported_transformers_versions(pyproject)
    version = (
        installed
        if installed is not None
        else importlib.metadata.version("transformers")
    )
    if version not in supported:
        raise ValueError(
            "installed Transformers "
            f"{version!r} is outside project pins {sorted(supported)}; "
            f"runtime_compatibility_smoke={RUNTIME_COMPATIBILITY_SMOKE}"
        )
    return version
