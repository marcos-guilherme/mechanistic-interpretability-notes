"""Concrete private operations for the remote sparse-refusal executor."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .data import prompt_content_hash
from .qwen_sparse_protocol import CORPUS_SOURCES


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _install_training_checkpoint_pair(
    checkpoint_source: Path,
    artifacts_dir: Path,
    identity: Mapping[str, str],
) -> tuple[Path, Path]:
    """Install checkpoint and identity status as one rollback-safe pair."""

    import shutil

    required_identity = {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "corpus_sha256",
        "model_revision",
    }
    if set(identity) != required_identity or any(
        not isinstance(value, str) or not value for value in identity.values()
    ):
        raise ValueError("training checkpoint identity is incomplete")
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts_dir / "sae_checkpoint.pt"
    status = artifacts_dir / "checkpoint_status.json"
    incoming_checkpoint = checkpoint.with_name(
        f".{checkpoint.name}.incoming"
    )
    incoming_status = status.with_name(f".{status.name}.incoming")
    with Path(checkpoint_source).open("rb") as incoming:
        with incoming_checkpoint.open("wb") as output:
            shutil.copyfileobj(incoming, output, length=8 * 1024**2)
            output.flush()
            os.fsync(output.fileno())
    _atomic_json(
        incoming_status,
        {
            "schema_version": 1,
            "identity": dict(identity),
            "checkpoint_sha256": _sha256_file(incoming_checkpoint),
        },
    )
    destinations = (checkpoint, status)
    incoming = (incoming_checkpoint, incoming_status)
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    try:
        for destination in destinations:
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup")
                backup.unlink(missing_ok=True)
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in zip(incoming, destinations, strict=True):
            os.replace(source, destination)
            installed.add(destination)
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for path in (*incoming, *backups.values()):
            path.unlink(missing_ok=True)
    return checkpoint, status


def _row_text(repo_id: str, row: Mapping[str, Any]) -> str:
    if repo_id == "code-search-net/code_search_net":
        for field in ("func_code_string", "whole_func_string", "code"):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value
    elif repo_id == "HuggingFaceH4/ultrachat_200k":
        messages = row.get("messages")
        if isinstance(messages, Sequence) and not isinstance(
            messages, (str, bytes)
        ):
            contents = [
                str(message["content"])
                for message in messages
                if isinstance(message, Mapping)
                and isinstance(message.get("content"), str)
                and str(message["content"]).strip()
            ]
            if contents:
                return "\n".join(contents)
    elif repo_id == "rufimelo/DeltaSecommits":
        prior = row.get("prior_version")
        after = row.get("after_version")
        if (
            isinstance(prior, str)
            and prior.strip()
            and isinstance(after, str)
            and after.strip()
        ):
            return f"{prior}\n{after}"
    raise ValueError(f"dataset row has no approved text fields: {repo_id}")


def _source_identity(
    repo_id: str,
    revision: str,
    row: Mapping[str, Any],
    index: int,
) -> str:
    for field in ("id", "repo", "sha", "commit"):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value):
            return f"{repo_id}@{revision}:{value}"
    return f"{repo_id}@{revision}:row-{index:012d}"


def _exclusion_hashes(
    root: Path,
    artifacts_dir: Path,
) -> tuple[set[str], dict[str, int | str]]:
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    pairs = artifacts_dir / "pairs" / "reviewed.jsonl"
    reviewed_hashes: set[str] = set()
    if pairs.is_file():
        for line in pairs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("offensive_sha256", "defensive_sha256"):
                value = row.get(field)
                if isinstance(value, str):
                    reviewed_hashes.add(value)
    hashes.update(reviewed_hashes)
    counts["reviewed_pair_hashes"] = len(reviewed_hashes)

    prepare = root / "results/qwen25_coder_7b_code_eval/00_prepare.json"
    code_eval_hashes: set[str] = set()
    if prepare.is_file():
        payload = json.loads(prepare.read_text(encoding="utf-8"))

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key == "content_hashes" and isinstance(item, list):
                        code_eval_hashes.update(str(digest) for digest in item)
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
    hashes.update(code_eval_hashes)
    counts["code_eval_hashes"] = len(code_eval_hashes)

    jlens_manifests = {
        "jlens_v1_hashes": (
            root / "results/qwen25_coder_7b_jlens/00_jlens_prepare.json"
        ),
        "jlens_v2_hashes": (
            root / "results/qwen25_coder_7b_jlens_v2/00_jlens_prepare.json"
        ),
    }
    for label, path in jlens_manifests.items():
        source_hashes: set[str] = set()
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, dict):
                raise ValueError(f"{label} manifest results are invalid")
            for field in ("fit_hashes", "holdout_hashes"):
                values = results.get(field)
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or not value for value in values
                ):
                    raise ValueError(
                        f"{label} manifest {field} must be a hash list"
                    )
                source_hashes.update(values)
        hashes.update(source_hashes)
        counts[label] = len(source_hashes)
    return hashes, {
        **counts,
        "effective_hashes": len(hashes),
        "effective_hashes_sha256": _sha256_value(sorted(hashes)),
    }


def _exact_document_holdout(
    records: Sequence[Mapping[str, Any]],
    target_tokens: int,
) -> set[int]:
    """Select a deterministic exact-token subset without splitting documents."""
    if target_tokens <= 0:
        raise ValueError("document holdout target must be positive")
    mask = (1 << (target_tokens + 1)) - 1
    reachable = 1
    predecessor_record = [-1] * (target_tokens + 1)
    predecessor_sum = [-1] * (target_tokens + 1)
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            str(records[index]["content_sha256"]),
            str(records[index]["source_identity"]),
        ),
    )
    for record_index in ranked:
        token_count = int(records[record_index]["token_count"])
        if token_count <= 0 or token_count > target_tokens:
            continue
        shifted = (reachable << token_count) & mask
        newly_reachable = shifted & ~reachable
        pending = newly_reachable
        while pending:
            lowest = pending & -pending
            token_sum = lowest.bit_length() - 1
            predecessor_record[token_sum] = record_index
            predecessor_sum[token_sum] = token_sum - token_count
            pending ^= lowest
        reachable |= shifted
        if reachable & (1 << target_tokens):
            break
    if not reachable & (1 << target_tokens):
        raise ValueError(
            "whole-document split cannot satisfy the frozen 5% token target"
        )
    selected: set[int] = set()
    token_sum = target_tokens
    while token_sum:
        record_index = predecessor_record[token_sum]
        if record_index < 0 or record_index in selected:
            raise RuntimeError("document holdout reconstruction failed")
        selected.add(record_index)
        token_sum = predecessor_sum[token_sum]
    return selected


def _default_dataset_loader(
    repo_id: str,
    *,
    config: str,
    split: str,
    revision: str,
) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        repo_id,
        config,
        split=split,
        revision=revision,
        streaming=True,
    )


def _default_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        revision=str(config["model_revision"]),
    )


def _source_contract(
    source_specs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(source["repo_id"]): {
            "config": source["config"],
            "split": source["split"],
            "revision": str(source["revision"]),
            "target_tokens": int(source["target_tokens"]),
        }
        for source in source_specs
    }


def _resolve_environment_identity(
    artifacts_dir: Path,
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    run_id: str | None,
    implementation_commit: str | None,
) -> dict[str, str]:
    if run_id is None:
        from .qwen_pilot import resolve_run_id

        resolved_run_id = resolve_run_id(artifacts_dir)
    else:
        resolved_run_id = run_id
    if implementation_commit is None:
        from .pipeline import implementation_commit_hash

        resolved_commit = implementation_commit_hash()
    else:
        resolved_commit = implementation_commit
    return {
        "run_id": resolved_run_id,
        "implementation_commit": resolved_commit,
        "config_sha256": _sha256_value(config),
        "corpus_sha256": _sha256_file(manifest_path),
        "model_revision": str(config["model_revision"]),
    }


def _reuse_prepared_corpus(
    root: Path,
    artifacts_dir: Path,
    *,
    config: Mapping[str, Any],
    source_specs: Sequence[Mapping[str, Any]],
    run_id: str | None,
    implementation_commit: str | None,
) -> dict[str, str] | None:
    manifest_path = artifacts_dir / "corpus_manifest.json"
    token_path = artifacts_dir / "executor" / "corpus_tokens.json"
    present = (manifest_path.is_file(), token_path.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise ValueError("prepared corpus manifest/token bundle must travel together")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_payload = json.loads(token_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 4:
        raise ValueError("prepared corpus manifest schema is unsupported")
    if not isinstance(token_payload, dict) or set(token_payload) != {
        "schema_version",
        "fit",
        "holdout",
    }:
        raise ValueError("prepared private token bundle schema is invalid")
    if token_payload["schema_version"] != 1:
        raise ValueError("prepared private token bundle version is unsupported")
    fit = token_payload["fit"]
    holdout = token_payload["holdout"]
    if not isinstance(fit, list) or not isinstance(holdout, list):
        raise ValueError("prepared private token splits must be lists")
    if any(
        isinstance(token, bool) or not isinstance(token, int)
        for split in (fit, holdout)
        for token in split
    ):
        raise ValueError("prepared private token bundle contains invalid tokens")
    if manifest.get("config_sha256") != _sha256_value(config):
        raise ValueError("prepared corpus config identity changed")
    _current_hashes, current_exclusions = _exclusion_hashes(
        Path(root),
        artifacts_dir,
    )
    if manifest.get("exclusions") != current_exclusions:
        raise ValueError("prepared corpus exclusion identity changed")
    if manifest.get("token_bundle_sha256") != _sha256_value(token_payload):
        raise ValueError("prepared private token bundle SHA-256 mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or manifest.get(
        "records_sha256"
    ) != _sha256_value(records):
        raise ValueError("prepared corpus record identity is invalid")
    fit_hashes = {
        record.get("content_sha256")
        for record in records
        if isinstance(record, dict) and record.get("split") == "fit"
    }
    holdout_hashes = {
        record.get("content_sha256")
        for record in records
        if isinstance(record, dict) and record.get("split") == "holdout"
    }
    if (
        len(fit_hashes) + len(holdout_hashes) != len(records)
        or fit_hashes & holdout_hashes
    ):
        raise ValueError("prepared corpus document split is not disjoint")
    split_validation = manifest.get("split_validation")
    expected_validation = {
        "fit_documents": len(fit_hashes),
        "holdout_documents": len(holdout_hashes),
        "overlap_documents": 0,
        "disjoint": True,
        "fit_content_hashes_sha256": _sha256_value(sorted(fit_hashes)),
        "holdout_content_hashes_sha256": _sha256_value(
            sorted(holdout_hashes)
        ),
    }
    if split_validation != expected_validation:
        raise ValueError("prepared corpus split validation is invalid")
    try:
        record_fit_tokens = sum(
            int(record["token_count"])
            for record in records
            if record["split"] == "fit"
        )
        record_holdout_tokens = sum(
            int(record["token_count"])
            for record in records
            if record["split"] == "holdout"
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "prepared corpus record token counts are invalid"
        ) from error
    if (
        record_fit_tokens != len(fit)
        or record_holdout_tokens != len(holdout)
    ):
        raise ValueError("prepared corpus document token counts changed")
    sources = manifest.get("sources")
    contract = _source_contract(source_specs)
    if not isinstance(sources, dict) or set(sources) != set(contract):
        raise ValueError("prepared corpus sources differ from frozen contract")
    for repo_id, expected in contract.items():
        observed = sources[repo_id]
        if not isinstance(observed, dict) or {
            key: observed.get(key) for key in expected
        } != expected:
            raise ValueError(f"prepared corpus source contract changed: {repo_id}")
        if observed.get("accepted_tokens") != expected["target_tokens"]:
            raise ValueError(f"prepared corpus source is incomplete: {repo_id}")
    accepted = sum(source["target_tokens"] for source in contract.values())
    if (
        manifest.get("accepted_tokens") != accepted
        or manifest.get("fit_tokens") != len(fit)
        or manifest.get("holdout_tokens") != len(holdout)
        or len(fit) + len(holdout) != accepted
    ):
        raise ValueError("prepared corpus token counts are inconsistent")
    return _resolve_environment_identity(
        artifacts_dir,
        config=config,
        manifest_path=manifest_path,
        run_id=run_id,
        implementation_commit=implementation_commit,
    )


def _restored_training_identity(
    artifacts_dir: Path,
) -> dict[str, str] | None:
    checkpoint = artifacts_dir / "sae_checkpoint.pt"
    status = artifacts_dir / "checkpoint_status.json"
    present = (checkpoint.is_file(), status.is_file())
    if present == (False, False):
        return None
    if present != (True, True):
        raise ValueError("checkpoint/status pair must both be present")
    payload = json.loads(status.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "identity",
        "checkpoint_sha256",
    }:
        raise ValueError("checkpoint status schema is invalid")
    identity = payload["identity"]
    required = {
        "run_id",
        "implementation_commit",
        "config_sha256",
        "corpus_sha256",
        "model_revision",
    }
    if (
        payload["schema_version"] != 1
        or not isinstance(identity, dict)
        or set(identity) != required
        or payload["checkpoint_sha256"] != _sha256_file(checkpoint)
    ):
        raise ValueError("checkpoint status identity is invalid")
    normalized = {key: str(value) for key, value in identity.items()}
    if any(not value for value in normalized.values()):
        raise ValueError("checkpoint status identity contains an empty value")
    return normalized


def prepare_sparse_environment(
    root: Path,
    *,
    config: Mapping[str, Any],
    tokenizer: Any,
    dataset_loader: Callable[..., Iterable[Mapping[str, Any]]],
    source_specs: Sequence[Mapping[str, Any]] = CORPUS_SOURCES,
    run_id: str | None = None,
    implementation_commit: str | None = None,
) -> dict[str, str]:
    """Materialize a deterministic token-only corpus and compact manifest."""
    root = Path(root).resolve()
    artifacts_dir = root / str(config["artifacts_dir"])
    reused = _reuse_prepared_corpus(
        root,
        artifacts_dir,
        config=config,
        source_specs=source_specs,
        run_id=run_id,
        implementation_commit=implementation_commit,
    )
    if reused is not None:
        return reused
    exclusions, exclusion_manifest = _exclusion_hashes(root, artifacts_dir)
    seen_hashes: set[str] = set()
    records: list[dict[str, Any]] = []
    document_tokens: list[list[int]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    for source in source_specs:
        repo_id = str(source["repo_id"])
        revision = str(source["revision"])
        target_tokens = int(source["target_tokens"])
        accepted_tokens = 0
        accepted_records = 0
        rows = dataset_loader(
            repo_id,
            config=str(source["config"]),
            split=str(source["split"]),
            revision=revision,
        )
        for index, row in enumerate(rows):
            text = _row_text(repo_id, row)
            content_hash = prompt_content_hash(text)
            if content_hash in exclusions or content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            tokens = [
                int(token)
                for token in tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            ]
            remaining = target_tokens - accepted_tokens
            tokens = tokens[:remaining]
            if not tokens:
                continue
            identity = _source_identity(repo_id, revision, row, index)
            records.append(
                {
                    "source": repo_id,
                    "revision": revision,
                    "source_identity": identity,
                    "content_sha256": content_hash,
                    "token_count": len(tokens),
                }
            )
            document_tokens.append(tokens)
            accepted_tokens += len(tokens)
            accepted_records += 1
            if accepted_tokens == target_tokens:
                break
        if accepted_tokens != target_tokens:
            raise ValueError(
                f"{repo_id} exhausted at {accepted_tokens}/{target_tokens} tokens"
            )
        source_stats[repo_id] = {
            "config": source["config"],
            "split": source["split"],
            "revision": revision,
            "target_tokens": target_tokens,
            "accepted_records": accepted_records,
            "accepted_tokens": accepted_tokens,
        }
    total_tokens = sum(len(tokens) for tokens in document_tokens)
    holdout_count = max(1, round(total_tokens * 0.05))
    holdout_documents = _exact_document_holdout(records, holdout_count)
    fit: list[int] = []
    holdout: list[int] = []
    for index, (record, tokens) in enumerate(
        zip(records, document_tokens, strict=True)
    ):
        split = "holdout" if index in holdout_documents else "fit"
        record["split"] = split
        (holdout if split == "holdout" else fit).extend(tokens)
    fit_hashes = {
        str(record["content_sha256"])
        for record in records
        if record["split"] == "fit"
    }
    holdout_hashes = {
        str(record["content_sha256"])
        for record in records
        if record["split"] == "holdout"
    }
    overlap = fit_hashes & holdout_hashes
    if overlap or len(fit) + len(holdout) != total_tokens:
        raise RuntimeError("corpus document split validation failed")
    split_validation = {
        "fit_documents": len(fit_hashes),
        "holdout_documents": len(holdout_hashes),
        "overlap_documents": len(overlap),
        "disjoint": not overlap,
        "fit_content_hashes_sha256": _sha256_value(sorted(fit_hashes)),
        "holdout_content_hashes_sha256": _sha256_value(
            sorted(holdout_hashes)
        ),
    }
    token_payload = {"schema_version": 1, "fit": fit, "holdout": holdout}
    manifest = {
        "schema_version": 4,
        "config_sha256": _sha256_value(config),
        "records": records,
        "records_sha256": _sha256_value(records),
        "token_bundle_sha256": _sha256_value(token_payload),
        "accepted_records": len(records),
        "accepted_tokens": total_tokens,
        "fit_tokens": len(fit),
        "holdout_tokens": len(holdout),
        "split_rule": "content_sha256_ranked_exact_document_subset,5_percent",
        "split_validation": split_validation,
        "exclusions": exclusion_manifest,
        "sources": source_stats,
    }
    manifest_path = _atomic_json(
        artifacts_dir / "corpus_manifest.json",
        manifest,
    )
    _atomic_json(
        artifacts_dir / "executor" / "corpus_tokens.json",
        token_payload,
    )
    return _resolve_environment_identity(
        artifacts_dir,
        config=config,
        manifest_path=manifest_path,
        run_id=run_id,
        implementation_commit=implementation_commit,
    )


def _prepare_environment_operation(*, root: Path, **_kwargs: Any) -> Mapping[str, str]:
    from .qwen_sparse_protocol import load_sparse_config

    config = load_sparse_config(root)
    return prepare_sparse_environment(
        root,
        config=config,
        tokenizer=_default_tokenizer(config),
        dataset_loader=_default_dataset_loader,
    )


def choose_hardware_pilot(
    observations: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[int],
    max_memory_bytes: int,
    max_projected_hours: float,
) -> dict[str, Any]:
    """Choose only the first frozen candidate satisfying both hard limits."""
    indexed: dict[int, Mapping[str, Any]] = {}
    for observation in observations:
        microbatch = int(observation["microbatch_tokens"])
        if microbatch in indexed:
            raise ValueError("duplicate pilot microbatch observation")
        indexed[microbatch] = observation
    for candidate in candidates:
        observation = indexed.get(int(candidate))
        if observation is None:
            continue
        peak = int(observation["peak_memory_bytes"])
        projected = float(observation["projected_hours"])
        if (
            bool(observation["finite"])
            and peak <= max_memory_bytes
            and projected <= max_projected_hours
        ):
            return {
                "status": "completed",
                "microbatch_tokens": int(candidate),
                "peak_memory_bytes": peak,
                "projected_hours": projected,
            }
    return {"status": "resource_projection_failed"}


def pack_token_rows(
    tokens: Sequence[int],
    *,
    sequence_length: int,
    pad_token_id: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Pack all tokens, masking rather than dropping the final partial row."""
    if sequence_length <= 0 or not tokens:
        raise ValueError("token packing requires tokens and sequence_length")
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for start in range(0, len(tokens), sequence_length):
        chunk = [int(token) for token in tokens[start : start + sequence_length]]
        actual = len(chunk)
        rows.append(chunk + [int(pad_token_id)] * (sequence_length - actual))
        masks.append([1] * actual + [0] * (sequence_length - actual))
    return rows, masks


def configure_generation_identity(
    model: Any,
    tokenizer: Any,
    generation: Mapping[str, Any],
) -> None:
    """Bind deterministic generation to the frozen tokenizer EOS identity."""

    if generation.get("do_sample") is not False:
        raise ValueError("sparse generation must be deterministic")
    if generation.get("eos") != "tokenizer_eos":
        raise ValueError("sparse generation EOS source changed")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("tokenizer EOS token is unavailable")
    model.generation_config.eos_token_id = eos_token_id
    model.generation_config.do_sample = False


class SparseA100Operations:
    """Stateful implementation of all heavy stages in one A100 process."""

    def __init__(self) -> None:
        self._models: dict[Path, tuple[Any, Any]] = {}

    @staticmethod
    def _config(root: Path) -> dict[str, Any]:
        from .qwen_sparse_protocol import load_sparse_config

        return load_sparse_config(root)

    @staticmethod
    def _executor_dir(artifacts_dir: Path) -> Path:
        path = artifacts_dir / "executor"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_environment(self, *, root: Path, **_kwargs: Any) -> Mapping[str, str]:
        config = self._config(root)
        artifacts_dir = Path(root) / str(config["artifacts_dir"])
        restored = _restored_training_identity(artifacts_dir)
        run_id = restored["run_id"] if restored else None
        implementation_commit = (
            restored["implementation_commit"] if restored else None
        )
        reused = _reuse_prepared_corpus(
            root,
            artifacts_dir,
            config=config,
            source_specs=CORPUS_SOURCES,
            run_id=run_id,
            implementation_commit=implementation_commit,
        )
        if reused is not None:
            return reused
        return prepare_sparse_environment(
            root,
            config=config,
            tokenizer=_default_tokenizer(config),
            dataset_loader=_default_dataset_loader,
            run_id=run_id,
            implementation_commit=implementation_commit,
        )

    def _model(self, root: Path) -> tuple[Any, Any]:
        root = Path(root).resolve()
        if root not in self._models:
            import torch

            from .model import load_causal_lm

            config = self._config(root)
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[str(config["dtype"])]
            model, tokenizer = load_causal_lm(
                str(config["model_name"]),
                dtype=dtype,
                revision=str(config["model_revision"]),
            )
            configure_generation_identity(
                model,
                tokenizer,
                config["generation"],
            )
            self._models[root] = (model, tokenizer)
        return self._models[root]

    @staticmethod
    def _tokens(artifacts_dir: Path, split: str) -> list[int]:
        payload = json.loads(
            (
                artifacts_dir / "executor/corpus_tokens.json"
            ).read_text(encoding="utf-8")
        )
        if payload.get("schema_version") != 1 or split not in {
            "fit",
            "holdout",
        }:
            raise ValueError("private corpus token bundle is invalid")
        values = payload[split]
        if not isinstance(values, list) or not values:
            raise ValueError(f"private corpus {split} tokens are unavailable")
        return [int(token) for token in values]

    def _activation_batches(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        split: str,
        microbatch_tokens: int,
        activation_scale: float,
        limit_tokens: int | None = None,
    ) -> Iterable[Any]:
        import torch

        from .pipeline import transformer_module_lists
        from .qwen_sparse_training import ActivationBatch

        config = self._config(root)
        model, tokenizer = self._model(root)
        blocks, _attentions, _mlps = transformer_module_lists(model)
        layer = int(config["sae"]["layer"])
        sequence_length = int(config["sae"]["sequence_length"])
        if microbatch_tokens % sequence_length:
            raise ValueError("microbatch_tokens must divide into full sequences")
        sequence_count = microbatch_tokens // sequence_length
        token_values = self._tokens(artifacts_dir, split)
        if limit_tokens is not None:
            token_values = token_values[:limit_tokens]
        device = next(model.parameters()).device
        for start in range(0, len(token_values), microbatch_tokens):
            chunk = token_values[start : start + microbatch_tokens]
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError("tokenizer requires pad or EOS token")
            packed, masks = pack_token_rows(
                chunk,
                sequence_length=sequence_length,
                pad_token_id=int(pad_token_id),
            )
            rows = len(packed)
            if rows > sequence_count:
                raise AssertionError("microbatch row count exceeded")
            input_ids = torch.tensor(
                packed,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.tensor(
                masks,
                dtype=torch.long,
                device=device,
            )
            captured: list[torch.Tensor] = []

            def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
                captured.append(inputs[0])

            handle = blocks[layer].register_forward_pre_hook(capture)
            try:
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("layer 18 activation capture count changed")
            activations = (
                captured[0]
                .detach()
                [attention_mask.bool()]
                .float()
                .mul(float(activation_scale))
                .cpu()
            )
            yield ActivationBatch(activations)

    def prepare(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        config = self._config(root)
        manifest = json.loads(
            (artifacts_dir / "corpus_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("schema_version") != 4
            or manifest.get("fit_tokens") != config["sae"]["fit_tokens"]
            or manifest.get("holdout_tokens")
            != config["sae"]["holdout_tokens"]
        ):
            raise ValueError("corpus manifest differs from frozen SAE split")
        self._source_roots(artifacts_dir)
        return {"status": "completed"}

    def hardware_pilot(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import math
        import time

        import torch

        from .qwen_sparse_sae import TopKSparseAutoencoder
        from .qwen_sparse_training import (
            create_sae_optimizer,
            measure_sae_training_candidate,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("A100 pilot requires CUDA")
        device_name = torch.cuda.get_device_name()
        if "A100" not in device_name.upper():
            raise RuntimeError(f"A100 pilot requires A100, got {device_name}")
        config = self._config(root)
        sae_config = config["sae"]
        calibration_tokens = int(sae_config["calibration_tokens"])
        calibration_batch = min(
            int(min(sae_config["microbatch_candidates"])),
            calibration_tokens,
        )
        unscaled = self._activation_batches(
            root=root,
            artifacts_dir=artifacts_dir,
            split="fit",
            microbatch_tokens=calibration_batch,
            activation_scale=1.0,
            limit_tokens=calibration_tokens,
        )
        norm_sum = 0.0
        token_count = 0
        for batch in unscaled:
            norms = torch.linalg.vector_norm(batch.activations, dim=-1)
            norm_sum += float(norms.sum().item())
            token_count += int(norms.numel())
        if token_count != calibration_tokens or norm_sum <= 0:
            raise RuntimeError("activation calibration token count changed")
        activation_scale = math.sqrt(int(sae_config["d_in"])) / (
            norm_sum / token_count
        )
        observations: list[dict[str, Any]] = []
        effective_batch_tokens = int(sae_config["effective_batch_tokens"])
        for candidate in sae_config["microbatch_candidates"]:
            torch.cuda.empty_cache()
            finite = False
            optimizer_state_allocated = False
            elapsed = float("inf")
            tokens_per_second = 0.0
            sae = None
            optimizer = None
            measurement = None
            try:
                sae = TopKSparseAutoencoder(
                    d_in=int(sae_config["d_in"]),
                    d_sae=int(sae_config["d_sae"]),
                    k=int(sae_config["k"]),
                ).to("cuda")
                optimizer = create_sae_optimizer(sae)
                measurement = measure_sae_training_candidate(
                    sae,
                    optimizer,
                    lambda: self._activation_batches(
                        root=root,
                        artifacts_dir=artifacts_dir,
                        split="fit",
                        microbatch_tokens=int(candidate),
                        activation_scale=activation_scale,
                        limit_tokens=effective_batch_tokens,
                    ),
                    device="cuda",
                    effective_batch_tokens=effective_batch_tokens,
                    learning_rate=float(
                        sae_config["optimizer"]["max_learning_rate"]
                    ),
                    synchronize=torch.cuda.synchronize,
                    reset_peak_memory=torch.cuda.reset_peak_memory_stats,
                    clock=time.perf_counter,
                )
                finite = measurement.finite
                optimizer_state_allocated = (
                    measurement.optimizer_state_allocated
                )
                elapsed = measurement.elapsed_seconds
                tokens_per_second = measurement.tokens_per_second
                peak = int(torch.cuda.max_memory_allocated())
                projected = (
                    elapsed
                    * int(sae_config["fit_tokens"])
                    / effective_batch_tokens
                    / 3600
                )
            except torch.OutOfMemoryError:
                peak = int(torch.cuda.max_memory_allocated())
                projected = float("inf")
            finally:
                del measurement, optimizer, sae
            observations.append(
                {
                    "microbatch_tokens": int(candidate),
                    "peak_memory_bytes": peak,
                    "projected_hours": projected,
                    "finite": finite,
                    "effective_batch_tokens": effective_batch_tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens_per_second,
                    "optimizer_state_allocated": optimizer_state_allocated,
                    "steady_state_warmup_steps": 1,
                }
            )
        decision = choose_hardware_pilot(
            observations,
            candidates=sae_config["microbatch_candidates"],
            max_memory_bytes=36 * 1024**3,
            max_projected_hours=float(sae_config["max_projected_hours"]),
        )
        _atomic_json(
            self._executor_dir(artifacts_dir) / "pilot.json",
            {
                "schema_version": 1,
                "device_name": device_name,
                "activation_scale": activation_scale,
                "observations": observations,
                "decision": decision,
            },
        )
        return {"status": decision["status"]}

    def sae_fit(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        resume_from: Path | None,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import torch

        from .qwen_sparse_sae import TopKSparseAutoencoder
        from .qwen_sparse_training import SparseTrainConfig, train_sae_stream

        config = self._config(root)
        sae_config = config["sae"]
        pilot = json.loads(
            (self._executor_dir(artifacts_dir) / "pilot.json").read_text(
                encoding="utf-8"
            )
        )
        decision = pilot["decision"]
        if decision["status"] != "completed":
            raise ValueError("SAE fit cannot start after failed pilot")
        optimizer = sae_config["optimizer"]
        training = SparseTrainConfig(
            run_id=store.run_id,
            implementation_commit=store.implementation_commit,
            config_sha256=store.config_sha256,
            corpus_sha256=store.corpus_sha256,
            model_revision=str(config["model_revision"]),
            max_learning_rate=float(optimizer["max_learning_rate"]),
            min_learning_rate=float(optimizer["min_learning_rate"]),
            warmup_tokens=int(optimizer["warmup_tokens"]),
            total_fit_tokens=int(sae_config["fit_tokens"]),
            effective_batch_tokens=int(
                sae_config["effective_batch_tokens"]
            ),
            checkpoint_every_tokens=int(
                sae_config["checkpoint_every_tokens"]
            ),
            activation_scale=float(pilot["activation_scale"]),
            dead_feature_window_tokens=int(
                sae_config["dead_feature_window_tokens"]
            ),
            device="cuda",
        )
        sae = TopKSparseAutoencoder(
            d_in=int(sae_config["d_in"]),
            d_sae=int(sae_config["d_sae"]),
            k=int(sae_config["k"]),
        )
        checkpoint = train_sae_stream(
            sae,
            self._activation_batches(
                root=root,
                artifacts_dir=artifacts_dir,
                split="fit",
                microbatch_tokens=int(decision["microbatch_tokens"]),
                activation_scale=float(pilot["activation_scale"]),
                limit_tokens=int(sae_config["fit_tokens"]),
            ),
            training,
            self._executor_dir(artifacts_dir) / "checkpoints",
            resume_from=resume_from,
            on_checkpoint=lambda path: _install_training_checkpoint_pair(
                path,
                artifacts_dir,
                training.identity,
            ),
        )
        if checkpoint.tokens_seen != int(sae_config["fit_tokens"]):
            raise ValueError("SAE fit token count is incomplete")
        torch.cuda.empty_cache()
        return {"status": "completed"}

    def sae_quality(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        from dataclasses import asdict

        from .qwen_sparse_sae import (
            TopKSparseAutoencoder,
            evaluate_sae_quality,
        )
        from .qwen_sparse_training import load_training_checkpoint

        config = self._config(root)
        sae_config = config["sae"]
        identity = {
            "run_id": store.run_id,
            "implementation_commit": store.implementation_commit,
            "config_sha256": store.config_sha256,
            "corpus_sha256": store.corpus_sha256,
            "model_revision": str(config["model_revision"]),
        }
        checkpoint = load_training_checkpoint(
            artifacts_dir / "sae_checkpoint.pt",
            identity,
        )
        sae = TopKSparseAutoencoder(
            d_in=int(sae_config["d_in"]),
            d_sae=int(sae_config["d_sae"]),
            k=int(sae_config["k"]),
        ).to("cuda")
        sae.load_state_dict(checkpoint.model_state)
        pilot = json.loads(
            (self._executor_dir(artifacts_dir) / "pilot.json").read_text(
                encoding="utf-8"
            )
        )
        heldout = (
            batch.activations
            for batch in self._activation_batches(
                root=root,
                artifacts_dir=artifacts_dir,
                split="holdout",
                microbatch_tokens=int(
                    pilot["decision"]["microbatch_tokens"]
                ),
                activation_scale=float(pilot["activation_scale"]),
                limit_tokens=int(sae_config["holdout_tokens"]),
            )
        )
        quality = evaluate_sae_quality(
            sae,
            heldout,
            fit_mean_norm=float(int(sae_config["d_in"]) ** 0.5),
            last_activation_token=checkpoint.last_activation_token,
            fit_tokens_seen=checkpoint.tokens_seen,
            dead_feature_window_tokens=int(
                sae_config["dead_feature_window_tokens"]
            ),
        )
        return {"status": quality.status, **asdict(quality)}

    def decompose(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import torch

        from .directions import load_direction
        from .qwen_sparse_direction import fit_direction_grid
        from .qwen_sparse_sae import TopKSparseAutoencoder
        from .qwen_sparse_training import load_training_checkpoint

        config = self._config(root)
        sae_config = config["sae"]
        identity = {
            "run_id": store.run_id,
            "implementation_commit": store.implementation_commit,
            "config_sha256": store.config_sha256,
            "corpus_sha256": store.corpus_sha256,
            "model_revision": str(config["model_revision"]),
        }
        checkpoint = load_training_checkpoint(
            artifacts_dir / "sae_checkpoint.pt",
            identity,
        )
        sae = TopKSparseAutoencoder(
            int(sae_config["d_in"]),
            int(sae_config["d_sae"]),
            int(sae_config["k"]),
        )
        sae.load_state_dict(checkpoint.model_state)
        direction = load_direction(root / str(config["direction_path"]))
        if direction.layer != int(sae_config["layer"]):
            raise ValueError("dense direction layer differs from SAE layer")
        candidates = fit_direction_grid(
            sae.decoder.weight.detach().float(),
            direction.vector.detach().float(),
            config["cardinality_grid"],
        )
        private = self._executor_dir(artifacts_dir) / "direction_candidates.pt"
        torch.save(candidates, private)
        best = max(
            candidates,
            key=lambda item: (
                item.cosine >= 0.95 and item.relative_l2 <= 0.35,
                item.cosine,
                -item.relative_l2,
            ),
        )
        passed = best.cosine >= 0.95 and best.relative_l2 <= 0.35
        return {
            "status": "pass" if passed else "direction_reconstruction_failed",
            "cosine": best.cosine,
            "relative_l2": best.relative_l2,
            "norm": float(best.reconstruction.norm().item()),
        }

    @staticmethod
    def _reviewed_splits(artifacts_dir: Path) -> Mapping[str, tuple[Any, ...]]:
        from .qwen_sparse_data import ReviewedPair, split_reviewed_pairs

        rows = [
            json.loads(line)
            for line in (
                artifacts_dir / "pairs/reviewed.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reviewed = tuple(
            ReviewedPair(
                pair_id=str(row["pair_id"]),
                category=str(row["category"]),
                source_identity=str(row["source_identity"]),
                offensive_prompt=str(row["offensive_prompt"]),
                defensive_prompt=str(row["defensive_prompt"]),
                intent_rationale=str(row["intent_rationale"]),
                approved=bool(row["approved"]),
            )
            for row in rows
        )
        return split_reviewed_pairs(reviewed)

    def _sae_and_checkpoint(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
    ) -> tuple[Any, Any]:
        from .qwen_sparse_sae import TopKSparseAutoencoder
        from .qwen_sparse_training import load_training_checkpoint

        config = self._config(root)
        sae_config = config["sae"]
        identity = {
            "run_id": store.run_id,
            "implementation_commit": store.implementation_commit,
            "config_sha256": store.config_sha256,
            "corpus_sha256": store.corpus_sha256,
            "model_revision": str(config["model_revision"]),
        }
        checkpoint = load_training_checkpoint(
            artifacts_dir / "sae_checkpoint.pt",
            identity,
        )
        sae = TopKSparseAutoencoder(
            int(sae_config["d_in"]),
            int(sae_config["d_sae"]),
            int(sae_config["k"]),
        ).to("cuda")
        sae.load_state_dict(checkpoint.model_state)
        sae.eval()
        return sae, checkpoint

    def _last_token_latents(
        self,
        *,
        root: Path,
        prompts: Sequence[str],
        sae: Any,
        activation_scale: float,
    ) -> list[Any]:
        import torch

        from .model import format_chat_instruction
        from .pipeline import transformer_module_lists

        config = self._config(root)
        model, tokenizer = self._model(root)
        blocks, _attentions, _mlps = transformer_module_lists(model)
        device = next(model.parameters()).device
        output = []
        for prompt in prompts:
            text = format_chat_instruction(tokenizer, prompt)
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            batch = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            captured: list[torch.Tensor] = []

            def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
                captured.append(inputs[0][:, -1])

            handle = blocks[int(config["sae"]["layer"])].register_forward_pre_hook(
                capture
            )
            try:
                with torch.inference_mode():
                    model(**batch, use_cache=False)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("last-token activation capture changed")
            with torch.inference_mode():
                latent = sae.encode(
                    captured[0].float().mul(activation_scale)
                )
            output.append(latent[0].detach().cpu())
        return output

    @staticmethod
    def _examples(pairs: Sequence[Any], side: str) -> list[Any]:
        from .data import PromptExample

        return [
            PromptExample(
                id=f"{pair.pair_id}:{side}",
                text=str(getattr(pair, f"{side}_prompt")),
                label="harmful" if side == "offensive" else "harmless",
                source="qwen_sparse_reviewed_pairs",
            )
            for pair in pairs
        ]

    def _generate(
        self,
        *,
        root: Path,
        examples: Sequence[Any],
        pre_hooks: Sequence[tuple[Any, Any]] = (),
        forward_hooks: Sequence[tuple[Any, Any]] = (),
    ) -> list[str]:
        from .pipeline import generate_responses

        config = self._config(root)
        generation = config["generation"]
        model, tokenizer = self._model(root)
        return generate_responses(
            model,
            tokenizer,
            examples,
            batch_size=int(generation["batch_size"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            format_style=str(generation["format_style"]),
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
            invalid_on_decode_error=True,
        )

    def discover(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        from dataclasses import asdict
        import math
        import statistics

        import torch
        from torch.nn import functional as F

        from .metrics import contains_refusal
        from .qwen_sparse_direction import (
            PairedFeatureRow,
            associate_features,
            filter_direction,
            random_feature_control,
        )

        config = self._config(root)
        sae, checkpoint = self._sae_and_checkpoint(
            root=root,
            artifacts_dir=artifacts_dir,
            store=store,
        )
        candidates = torch.load(
            self._executor_dir(artifacts_dir) / "direction_candidates.pt",
            map_location="cpu",
            weights_only=False,
        )
        discovery_pairs = self._reviewed_splits(artifacts_dir)["discovery"]
        prompts = [
            str(getattr(pair, f"{side}_prompt"))
            for pair in discovery_pairs
            for side in ("offensive", "defensive")
        ]
        latents = self._last_token_latents(
            root=root,
            prompts=prompts,
            sae=sae,
            activation_scale=checkpoint.activation_scale,
        )
        examples = [
            example
            for side in ("offensive", "defensive")
            for example in self._examples(discovery_pairs, side)
        ]
        baseline_responses = self._generate(root=root, examples=examples)
        refusal_by_key = {
            example.id: float(contains_refusal(response))
            for example, response in zip(
                examples,
                baseline_responses,
                strict=True,
            )
        }
        dense = (
            torch.load(
                root / str(config["direction_path"]),
                map_location="cpu",
                weights_only=True,
            )["vector"]
            .detach()
            .float()
        )
        candidate_by_feature = {
            feature: candidate
            for candidate in sorted(
                candidates,
                key=lambda item: item.cardinality,
                reverse=True,
            )
            for feature in candidate.feature_indices
        }
        rows: list[Any] = []
        feature_values: dict[int, list[tuple[float, float]]] = {
            feature: [] for feature in candidate_by_feature
        }
        cursor = 0
        for pair in discovery_pairs:
            for side in ("offensive", "defensive"):
                latent = latents[cursor]
                cursor += 1
                for feature, candidate in candidate_by_feature.items():
                    position = candidate.feature_indices.index(feature)
                    coefficient = float(candidate.coefficients[position])
                    decoder_column = candidate.decoder_columns[:, position]
                    cosine = float(
                        F.cosine_similarity(
                            decoder_column.reshape(1, -1),
                            dense.reshape(1, -1),
                        ).item()
                    )
                    response_key = f"{pair.pair_id}:{side}"
                    value = float(latent[feature].item())
                    feature_values[feature].append(
                        (value, refusal_by_key[response_key])
                    )
                    rows.append(
                        PairedFeatureRow(
                            pair_id=pair.pair_id,
                            side=side,
                            feature_index=feature,
                            token_activations=(value,),
                            coefficient_sign=(
                                1 if coefficient > 0 else -1
                                if coefficient < 0 else 0
                            ),
                            decoder_cosine=cosine,
                            basal_refusal_association=0.0,
                        )
                    )
        basal: dict[int, float] = {}
        for feature, values in feature_values.items():
            activations = [item[0] for item in values]
            refusals = [item[1] for item in values]
            if (
                len(set(activations)) < 2
                or len(set(refusals)) < 2
            ):
                basal[feature] = 0.0
            else:
                mean_x = statistics.fmean(activations)
                mean_y = statistics.fmean(refusals)
                numerator = sum(
                    (x - mean_x) * (y - mean_y)
                    for x, y in zip(activations, refusals, strict=True)
                )
                denominator = math.sqrt(
                    sum((x - mean_x) ** 2 for x in activations)
                    * sum((y - mean_y) ** 2 for y in refusals)
                )
                basal[feature] = numerator / denominator
        rows = [
            type(row)(
                **{
                    **asdict(row),
                    "basal_refusal_association": basal[row.feature_index],
                }
            )
            for row in rows
        ]
        associations = associate_features(
            rows,
            sorted(candidate_by_feature),
            seed=int(config["bootstrap_seed"]),
        )
        association_map = {
            item.feature_index: item for item in associations
        }
        derived = {"dense": dense, "unfiltered": {}, "filtered": {}, "random": {}}
        for candidate in candidates:
            selected_associations = [
                association_map[index]
                for index in candidate.feature_indices
            ]
            filtered = filter_direction(candidate, selected_associations)
            random_control = random_feature_control(
                filtered,
                range(int(config["sae"]["d_sae"])),
                sae.decoder.weight.detach().cpu(),
                seed=2030 + int(candidate.requested_cardinality or 0),
            )
            cardinality = int(candidate.requested_cardinality or candidate.cardinality)
            derived["unfiltered"][cardinality] = candidate
            derived["filtered"][cardinality] = filtered
            derived["random"][cardinality] = random_control
        torch.save(
            derived,
            self._executor_dir(artifacts_dir) / "derived_vectors.pt",
        )
        association_payload = [asdict(item) for item in associations]
        association_path = _atomic_json(
            self._executor_dir(artifacts_dir) / "feature_associations.json",
            {"associations": association_payload},
        )
        return {
            "status": "completed",
            "association_count": len(associations),
            "association_sha256": _sha256_file(association_path),
        }

    def _hooks(
        self,
        *,
        root: Path,
        vector: Any,
        intervention: str,
        dose: float,
    ) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
        from .interventions import (
            directional_ablation_hook_specs,
            residual_intervention_pre_hook,
        )
        from .pipeline import transformer_module_lists

        config = self._config(root)
        model, _tokenizer = self._model(root)
        blocks, attentions, mlps = transformer_module_lists(model)
        if intervention == "addition":
            return (
                [
                    (
                        blocks[int(config["sae"]["layer"])],
                        residual_intervention_pre_hook(
                            vector,
                            mode="add",
                            coefficient=dose,
                        ),
                    )
                ],
                [],
            )
        if intervention == "ablation":
            return directional_ablation_hook_specs(
                blocks,
                attentions,
                mlps,
                vector,
                coefficient=dose,
            )
        raise ValueError(f"unknown sparse intervention: {intervention}")

    def _evaluate_vector(
        self,
        *,
        root: Path,
        pairs: Sequence[Any],
        vector: Any,
        intervention: str,
        dose: float,
        baseline_defensive_logits: Any,
    ) -> dict[str, float]:
        from .metrics import contains_refusal, kl_divergence
        from .pipeline import last_token_logits

        config = self._config(root)
        model, tokenizer = self._model(root)
        pre_hooks, forward_hooks = self._hooks(
            root=root,
            vector=vector.to(next(model.parameters()).device),
            intervention=intervention,
            dose=dose,
        )
        offensive = self._examples(pairs, "offensive")
        defensive = self._examples(pairs, "defensive")
        offensive_responses = self._generate(
            root=root,
            examples=offensive,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        defensive_responses = self._generate(
            root=root,
            examples=defensive,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        offensive_refusal = sum(
            contains_refusal(response) for response in offensive_responses
        ) / len(offensive_responses)
        defensive_refusal = sum(
            contains_refusal(response) for response in defensive_responses
        ) / len(defensive_responses)
        changed_logits = last_token_logits(
            model,
            tokenizer,
            defensive,
            batch_size=int(config["generation"]["batch_size"]),
            format_style=str(config["generation"]["format_style"]),
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
        )
        defensive_kl = float(
            kl_divergence(
                baseline_defensive_logits,
                changed_logits,
            ).mean().item()
        )
        desired_effect = (
            offensive_refusal
            if intervention == "addition"
            else 1.0 - offensive_refusal
        )
        return {
            "desired_effect": desired_effect,
            "defensive_false_refusal": defensive_refusal,
            "defensive_kl": defensive_kl,
        }

    def select(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import torch

        from .pipeline import last_token_logits
        from .qwen_sparse_runtime import freeze_selection, tensor_sha256

        config = self._config(root)
        derived = torch.load(
            self._executor_dir(artifacts_dir) / "derived_vectors.pt",
            map_location="cpu",
            weights_only=False,
        )
        pairs = self._reviewed_splits(artifacts_dir)["selection"]
        defensive = self._examples(pairs, "defensive")
        model, tokenizer = self._model(root)
        baseline_logits = last_token_logits(
            model,
            tokenizer,
            defensive,
            batch_size=int(config["generation"]["batch_size"]),
            format_style=str(config["generation"]["format_style"]),
        )
        dense_metrics: dict[str, dict[str, float]] = {}
        for intervention in ("addition", "ablation"):
            dense_metrics[intervention] = self._evaluate_vector(
                root=root,
                pairs=pairs,
                vector=derived["dense"],
                intervention=intervention,
                dose=float(config["dense_doses"][intervention]),
                baseline_defensive_logits=baseline_logits,
            )
        rows: list[dict[str, Any]] = []
        family_names = {
            "unfiltered": "sparse_unfiltered",
            "filtered": "sparse_filtered",
            "random": "sparse_random_control",
        }
        for family, frozen_name in family_names.items():
            for intervention in ("addition", "ablation"):
                for cardinality in config["cardinality_grid"]:
                    candidate = derived[family][int(cardinality)]
                    vector = candidate.reconstruction.detach().float()
                    for dose in config["dose_grid"]:
                        metrics = self._evaluate_vector(
                            root=root,
                            pairs=pairs,
                            vector=vector,
                            intervention=intervention,
                            dose=float(dose),
                            baseline_defensive_logits=baseline_logits,
                        )
                        rows.append(
                            {
                                "vector": frozen_name,
                                "intervention": intervention,
                                "cardinality": int(cardinality),
                                "dose": float(dose),
                                "dense_desired_effect": dense_metrics[
                                    intervention
                                ]["desired_effect"],
                                **metrics,
                                "feature_indices": list(
                                    candidate.feature_indices
                                ),
                                "coefficients": [
                                    float(value)
                                    for value in candidate.coefficients.tolist()
                                ],
                                "vector_sha256": tensor_sha256(vector),
                                "vector_norm": float(vector.norm().item()),
                                "dense_vector_sha256": tensor_sha256(
                                    derived["dense"]
                                ),
                                "dense_vector_norm": float(
                                    derived["dense"].norm().item()
                                ),
                                "dense_dose": float(
                                    config["dense_doses"][intervention]
                                ),
                            }
                        )
        freeze = freeze_selection(rows)
        vectors: dict[str, Any] = {"dense": derived["dense"]}
        aliases = {
            "sparse_unfiltered": "unfiltered",
            "sparse_filtered": "filtered",
            "sparse_random_control": "random",
        }
        for frozen_name, family in aliases.items():
            for intervention in ("addition", "ablation"):
                spec = freeze["selections"][frozen_name][intervention]
                cardinality = int(spec["cardinality"])
                vector = (
                    derived[family][cardinality].reconstruction.detach().float()
                )
                if tensor_sha256(vector) != spec["vector_sha256"]:
                    raise ValueError("selected vector differs from freeze")
                vectors[f"{family}_{intervention}"] = vector
        torch.save(
            vectors,
            self._executor_dir(artifacts_dir) / "selection_vectors.pt",
        )
        return freeze

    def test(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import torch

        from .metrics import contains_refusal
        from .qwen_sparse_runtime import (
            CONDITIONS,
            expected_test_matrix,
            sparse_condition_hook_specs,
        )

        config = self._config(root)
        freeze = json.loads(
            store.selection_freeze_path.read_text(encoding="utf-8")
        )
        vectors = torch.load(
            self._executor_dir(artifacts_dir) / "selection_vectors.pt",
            map_location="cpu",
            weights_only=True,
        )
        model, _tokenizer = self._model(root)
        from .pipeline import transformer_module_lists

        blocks, attentions, mlps = transformer_module_lists(model)
        pairs = self._reviewed_splits(artifacts_dir)["test"]
        primary = {
            "dense_addition",
            "filtered_addition",
            "dense_ablation",
            "filtered_ablation",
        }
        review_dir = artifacts_dir / "test_responses"
        private_dir = artifacts_dir / "test_responses_private"
        review_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        for condition in CONDITIONS:
            pre_hooks, forward_hooks = sparse_condition_hook_specs(
                condition=condition,
                blocks=blocks,
                attentions=attentions,
                mlps=mlps,
                vectors=vectors,
                freeze=freeze,
            )
            output_path = (
                review_dir if condition in primary else private_dir
            ) / f"{condition}.jsonl"
            rows: list[dict[str, Any]] = []
            for side in ("offensive", "defensive"):
                examples = self._examples(pairs, side)
                responses = self._generate(
                    root=root,
                    examples=examples,
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                )
                for pair, response in zip(pairs, responses, strict=True):
                    prompt = str(getattr(pair, f"{side}_prompt"))
                    rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "side": side,
                            "condition": condition,
                            "prompt": prompt,
                            "prompt_sha256": prompt_content_hash(prompt),
                            "generation": response,
                            "automatic_refusal": contains_refusal(response),
                        }
                    )
            output_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
        return {
            "status": "completed",
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "test_split_sha256": store.test_split_sha256,
            "observations": expected_test_matrix(store.test_pairs),
        }

    @staticmethod
    def _ensure_checkout(
        destination: Path,
        *,
        repository: str,
        revision: str,
    ) -> Path:
        import subprocess

        destination = Path(destination)
        git_dir = destination / ".git"
        if not git_dir.is_dir():
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(
                    f"source checkout destination is nonempty: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "-C", str(destination), "init"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "remote",
                    "add",
                    "origin",
                    repository,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            completed = subprocess.run(
                ["git", "-C", str(destination), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip() == revision:
                return destination
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", revision],
            check=True,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip() != revision:
            raise ValueError(f"source checkout did not resolve {revision}")
        return destination

    def _source_roots(self, artifacts_dir: Path) -> tuple[Path, Path]:
        from .qwen_sparse_protocol import PURPLELLAMA_REVISION
        from .qwen_sparse_runtime import EVALPLUS_REVISION

        root = self._executor_dir(artifacts_dir) / "sources"
        purple = self._ensure_checkout(
            root / "PurpleLlama",
            repository="https://github.com/meta-llama/PurpleLlama.git",
            revision=PURPLELLAMA_REVISION,
        )
        evalplus = self._ensure_checkout(
            root / "evalplus",
            repository="https://github.com/evalplus/evalplus.git",
            revision=EVALPLUS_REVISION,
        )
        return purple, evalplus

    def _external_prompts(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
    ) -> Mapping[str, list[dict[str, Any]]]:
        from .qwen_causal_jlens_runtime import _rehydrate_dataset

        prepare_path = root / "results/qwen25_coder_7b_code_eval/00_prepare.json"
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
        purple, _evalplus = self._source_roots(artifacts_dir)
        base_config = prepare["config"]
        mitre_rows = json.loads(
            (
                purple / str(base_config["mitre_relative_path"])
            ).read_text(encoding="utf-8")
        )
        frr_rows = json.loads(
            (
                purple / str(base_config["mitre_frr_relative_path"])
            ).read_text(encoding="utf-8")
        )
        return {
            "mitre": _rehydrate_dataset(
                mitre_rows,
                dataset="mitre",
                expected=prepare["results"]["mitre"],
                text_field="mutated_prompt",
                category_field="mitre_category",
                dataset_size=32,
            ),
            "mitre_frr": _rehydrate_dataset(
                frr_rows,
                dataset="mitre_frr",
                expected=prepare["results"]["mitre_frr"],
                text_field="mutated_prompt",
                category_field="attack_type",
                dataset_size=32,
            ),
        }

    @staticmethod
    def _compact_metrics(rows: Sequence[Mapping[str, Any]], *, code: bool = False) -> dict[str, Any]:
        count = len(rows)
        payload = {
            "n": count,
            "refusal_rate": sum(bool(row["refused"]) for row in rows) / count,
            "mean_kl": sum(float(row["kl_from_baseline"]) for row in rows) / count,
            "invalid_generation_rate": sum(
                bool(row["invalid_generation"]) for row in rows
            )
            / count,
        }
        if code:
            payload["pass_at_1"] = (
                sum(bool(row["plus_pass"]) for row in rows) / count
            )
        return payload

    def external_transfer(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import torch

        from .data import PromptExample
        from .metrics import contains_refusal, is_valid_generation, kl_divergence
        from .pipeline import last_token_logits, transformer_module_lists
        from .qwen_sparse_runtime import (
            EXTERNAL_CONDITIONS,
            condition_dose,
            frozen_external_transfer_identity,
            sparse_condition_hook_specs,
        )

        config = self._config(root)
        prepare_path = root / "results/qwen25_coder_7b_code_eval/00_prepare.json"
        identity = frozen_external_transfer_identity(prepare_path)
        datasets = self._external_prompts(root=root, artifacts_dir=artifacts_dir)
        freeze = json.loads(
            store.selection_freeze_path.read_text(encoding="utf-8")
        )
        vectors = torch.load(
            self._executor_dir(artifacts_dir) / "selection_vectors.pt",
            map_location="cpu",
            weights_only=True,
        )
        model, tokenizer = self._model(root)
        blocks, attentions, mlps = transformer_module_lists(model)
        observations: list[dict[str, Any]] = []
        metrics: dict[str, dict[str, Any]] = {}
        for dataset, items in datasets.items():
            examples = [
                PromptExample(
                    id=item["example_id"],
                    text=item["prompt"],
                    label="harmful" if dataset == "mitre" else "harmless",
                    source=dataset,
                )
                for item in items
            ]
            baseline_logits = last_token_logits(
                model,
                tokenizer,
                examples,
                batch_size=int(config["generation"]["batch_size"]),
                format_style=str(config["generation"]["format_style"]),
            )
            metrics[dataset] = {}
            for condition in EXTERNAL_CONDITIONS:
                pre_hooks, forward_hooks = sparse_condition_hook_specs(
                    condition=condition,
                    blocks=blocks,
                    attentions=attentions,
                    mlps=mlps,
                    vectors=vectors,
                    freeze=freeze,
                )
                responses = self._generate(
                    root=root,
                    examples=examples,
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                )
                changed_logits = last_token_logits(
                    model,
                    tokenizer,
                    examples,
                    batch_size=int(config["generation"]["batch_size"]),
                    format_style=str(config["generation"]["format_style"]),
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                )
                kl_values = kl_divergence(
                    baseline_logits,
                    changed_logits,
                ).tolist()
                condition_rows = []
                for item, response, kl_value in zip(
                    items,
                    responses,
                    kl_values,
                    strict=True,
                ):
                    row = {
                        "dataset": dataset,
                        "example_id": item["example_id"],
                        "content_sha256": item["content_hash"],
                        "condition": condition,
                        "dose": condition_dose(condition, freeze),
                        "refused": contains_refusal(response),
                        "kl_from_baseline": float(kl_value),
                        "invalid_generation": not is_valid_generation(response),
                    }
                    condition_rows.append(row)
                    observations.append(row)
                metrics[dataset][condition] = self._compact_metrics(
                    condition_rows
                )
        return {
            "status": "completed",
            "prepare_sha256": identity["prepare_sha256"],
            "selection_freeze_sha256": store.selection_freeze_sha256(),
            "observations": observations,
            "metrics": metrics,
        }

    def code_preservation(
        self,
        *,
        root: Path,
        artifacts_dir: Path,
        store: Any,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        import subprocess
        import sys

        import torch

        from .data import PromptExample
        from .metrics import contains_refusal, is_valid_generation, kl_divergence
        from .pipeline import last_token_logits, transformer_module_lists
        from .qwen_code_eval_runtime import evalplus_output_matches_identity
        from .qwen_sparse_runtime import (
            CODE_PRESERVATION_CONDITIONS,
            EVALPLUS_MEMORY_BYTES,
            EVALPLUS_REVISION,
            EVALPLUS_TIMEOUT_SECONDS,
            build_code_preservation_jobs,
            frozen_humaneval_identity,
            sparse_condition_hook_specs,
        )

        config = self._config(root)
        prepare_path = root / "results/qwen25_coder_7b_code_eval/00_prepare.json"
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
        identity = frozen_humaneval_identity(prepare_path)
        _purple, evalplus_root = self._source_roots(artifacts_dir)
        sys.path.insert(0, str(evalplus_root))
        from evalplus.data import get_human_eval_plus

        problems_map = get_human_eval_plus()
        source_rows = [
            {"task_id": task_id, **problem}
            for task_id, problem in sorted(problems_map.items())
        ]
        expected = prepare["results"]["humaneval_plus"]
        items = []
        for example_id, content_hash, source_index in zip(
            expected["example_ids"],
            expected["content_hashes"],
            expected["source_indices"],
            strict=True,
        ):
            problem = source_rows[int(source_index)]
            if prompt_content_hash(problem["prompt"]) != content_hash:
                raise ValueError("HumanEval+ frozen prompt hash changed")
            items.append(
                {
                    "example_id": str(example_id),
                    "content_sha256": str(content_hash),
                    "problem": problem,
                }
            )
        freeze = json.loads(
            store.selection_freeze_path.read_text(encoding="utf-8")
        )
        vectors = torch.load(
            self._executor_dir(artifacts_dir) / "selection_vectors.pt",
            map_location="cpu",
            weights_only=True,
        )
        model, tokenizer = self._model(root)
        blocks, attentions, mlps = transformer_module_lists(model)
        examples = [
            PromptExample(
                id=item["example_id"],
                text=item["problem"]["prompt"],
                label="harmless",
                source="humaneval_plus",
            )
            for item in items
        ]
        baseline_logits = last_token_logits(
            model,
            tokenizer,
            examples,
            batch_size=int(config["generation"]["batch_size"]),
            format_style=str(config["generation"]["format_style"]),
        )
        condition_map = {
            "baseline": "baseline",
            "dense": "dense_addition",
            "filtered": "filtered_addition",
        }
        private = self._executor_dir(artifacts_dir) / "evalplus"
        problems_path = private / "problems.jsonl"
        problems_path.parent.mkdir(parents=True, exist_ok=True)
        problems_path.write_text(
            "".join(
                json.dumps(item["problem"], sort_keys=True) + "\n"
                for item in items
            ),
            encoding="utf-8",
        )
        sample_paths: dict[str, Path] = {}
        generated: dict[str, list[dict[str, Any]]] = {}
        for output_condition, hook_condition in condition_map.items():
            pre_hooks, forward_hooks = sparse_condition_hook_specs(
                condition=hook_condition,
                blocks=blocks,
                attentions=attentions,
                mlps=mlps,
                vectors=vectors,
                freeze=freeze,
            )
            responses = self._generate(
                root=root,
                examples=examples,
                pre_hooks=pre_hooks,
                forward_hooks=forward_hooks,
            )
            changed_logits = last_token_logits(
                model,
                tokenizer,
                examples,
                batch_size=int(config["generation"]["batch_size"]),
                format_style=str(config["generation"]["format_style"]),
                pre_hooks=pre_hooks,
                forward_hooks=forward_hooks,
            )
            kl_values = kl_divergence(
                baseline_logits,
                changed_logits,
            ).tolist()
            samples = private / f"{output_condition}_samples.jsonl"
            samples.write_text(
                "".join(
                    json.dumps(
                        {
                            "task_id": item["problem"]["task_id"],
                            "solution": response,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for item, response in zip(items, responses, strict=True)
                ),
                encoding="utf-8",
            )
            sample_paths[output_condition] = samples
            generated[output_condition] = [
                {
                    "item": item,
                    "response": response,
                    "kl": float(kl_value),
                }
                for item, response, kl_value in zip(
                    items,
                    responses,
                    kl_values,
                    strict=True,
                )
            ]
        identity_dir = private / "identities"
        output_dir = private / "outputs"
        temp_root = private / "tmp"
        worker = root / "scripts/qwen_evalplus_worker.py"
        jobs = build_code_preservation_jobs(
            worker=worker,
            problems=problems_path,
            sample_paths=sample_paths,
            output_dir=output_dir,
            identity_dir=identity_dir,
            temp_root=temp_root,
            timeout_seconds=EVALPLUS_TIMEOUT_SECONDS,
            remote_vm=True,
            memory_bytes=EVALPLUS_MEMORY_BYTES,
            identity=identity,
            freeze=freeze,
        )
        worker_outputs: dict[str, list[dict[str, Any]]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        for condition in CODE_PRESERVATION_CONDITIONS:
            identity_dir.mkdir(parents=True, exist_ok=True)
            expected_task_ids = [
                item["problem"]["task_id"] for item in items
            ]
            worker_identity = {
                "run_id": store.run_id,
                "implementation_commit": store.implementation_commit,
                "config_sha256": store.config_sha256,
                "source_artifact_hashes": {
                    "prepare": identity["prepare_sha256"],
                    "selection_freeze": store.selection_freeze_sha256(),
                },
                "condition": condition,
                "expected_task_ids": expected_task_ids,
                "expected_task_ids_sha256": _sha256_value(
                    sorted(expected_task_ids)
                ),
            }
            identity_path = identity_dir / f"{condition}.json"
            _atomic_json(identity_path, worker_identity)
            job = jobs["jobs"][condition]
            job["cwd"].mkdir(parents=True, exist_ok=True)
            environment = dict(job["env"])
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    [
                        str(evalplus_root),
                        environment.get("PYTHONPATH", ""),
                    ],
                )
            )
            subprocess.run(
                job["command"],
                cwd=job["cwd"],
                env=environment,
                timeout=job["timeout"],
                check=True,
                capture_output=True,
                text=True,
            )
            output = json.loads(
                (output_dir / f"{condition}.json").read_text(
                    encoding="utf-8"
                )
            )
            if not evalplus_output_matches_identity(output, worker_identity):
                raise ValueError("EvalPlus output identity changed")
            pass_by_task = {
                row["task_id"]: bool(row["plus_pass"])
                for row in output["records"]
            }
            rows = []
            for generated_row in generated[condition]:
                item = generated_row["item"]
                response = generated_row["response"]
                rows.append(
                    {
                        "example_id": item["example_id"],
                        "content_sha256": item["content_sha256"],
                        "plus_pass": pass_by_task[item["problem"]["task_id"]],
                        "refused": contains_refusal(response),
                        "kl_from_baseline": generated_row["kl"],
                        "invalid_generation": not is_valid_generation(response),
                    }
                )
            worker_outputs[condition] = rows
            metrics[condition] = self._compact_metrics(rows, code=True)
        return {
            "status": "completed",
            "prepare_sha256": jobs["prepare_sha256"],
            "selection_freeze_sha256": jobs[
                "selection_freeze_sha256"
            ],
            "conditions": list(jobs["conditions"]),
            "doses": jobs["doses"],
            "memory_bytes": jobs["memory_bytes"],
            "timeout_seconds": EVALPLUS_TIMEOUT_SECONDS,
            "parallelism": jobs["parallelism"],
            "remote_vm": jobs["remote_vm"],
            "evalplus_revision": EVALPLUS_REVISION,
            "job_manifest_sha256": jobs["job_manifest_sha256"],
            "worker_outputs": worker_outputs,
            "metrics": metrics,
        }


def production_operations() -> Mapping[str, Callable[..., Mapping[str, Any]]]:
    """Return one stateful operation set for a complete remote process."""
    operations = SparseA100Operations()
    return {
        "prepare_environment": operations.prepare_environment,
        "prepare": operations.prepare,
        "hardware_pilot": operations.hardware_pilot,
        "sae_fit": operations.sae_fit,
        "sae_quality": operations.sae_quality,
        "decompose": operations.decompose,
        "discover": operations.discover,
        "select": operations.select,
        "test": operations.test,
        "external_transfer": operations.external_transfer,
        "code_preservation": operations.code_preservation,
    }
