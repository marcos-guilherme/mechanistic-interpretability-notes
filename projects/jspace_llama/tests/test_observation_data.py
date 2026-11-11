from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jspace_llama import observation_data
from jspace_llama.observation_data import (
    DatasetValidationError,
    MultihopItem,
    PANEL_NAMES,
    audit_tokenization,
    canonical_first_token,
    load_multihop_items,
)


PROJECT = Path(__file__).parents[1]
DATASET = (
    PROJECT
    / "data"
    / "anthropic-jlens"
    / "evaluations"
    / "lens-eval-multihop.json"
)


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {
            " Brazil": [11],
            "Brazil": [12, 13],
            " multiword": [20, 21],
            "multiword": [22, 23],
        }[text]


class FallbackTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {" Brazil": [], "Brazil": [12, 13]}[text]


class EmptyTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return []


def _valid_document() -> dict[str, object]:
    return {
        "items": [
            {
                "name": f"item-{index}",
                "prompt": f"Fact {index} is ",
                "target": f"target-{index}",
                "intermediates": [f"middle-{index}"],
            }
            for index in range(93)
        ]
    }


def _write_with_matching_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: object,
) -> Path:
    path = tmp_path / "lens-eval-multihop.json"
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(
        observation_data,
        "DATASET_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return path


def test_canonical_token_prefers_leading_space_and_records_full_sequence() -> None:
    assert canonical_first_token(FakeTokenizer(), "Brazil") == (
        11,
        (11,),
        "leading_space",
    )


def test_canonical_token_falls_back_to_raw_variant() -> None:
    assert canonical_first_token(FallbackTokenizer(), "Brazil") == (
        12,
        (12, 13),
        "raw",
    )


def test_canonical_token_rejects_variants_without_ids() -> None:
    with pytest.raises(ValueError, match="token IDs"):
        canonical_first_token(EmptyTokenizer(), "Brazil")


def test_audit_marks_only_the_selected_variant_as_single_token() -> None:
    items = (
        type(
            "Item",
            (),
            {
                "name": "one",
                "intermediates": ("Brazil", "multiword"),
            },
        )(),
    )

    audit = audit_tokenization(items, FakeTokenizer())

    assert audit[0]["name"] == "one"
    assert audit[0]["intermediates"][0] == {
        "text": "Brazil",
        "token_id": 11,
        "token_ids": (11,),
        "variant": "leading_space",
        "single_token": True,
        "prompt_sha256": hashlib.sha256(b" Brazil").hexdigest(),
        "token_ids_sha256": hashlib.sha256(b"[11]").hexdigest(),
    }
    assert audit[0]["intermediates"][1]["single_token"] is False
    assert audit[0]["intermediates"][1]["token_ids"] == (20, 21)
    assert "path" not in json.dumps(audit).lower()


def test_loader_accepts_exact_pinned_dataset() -> None:
    items = load_multihop_items(DATASET)

    assert len(items) == 93
    assert isinstance(items[0], MultihopItem)
    assert items[0] == MultihopItem(
        name="carnival-ocean",
        prompt=(
            "Fact: The ocean on the coast of the country where Carnival is "
            "most famously celebrated is the "
        ),
        target="Atlantic",
        intermediates=("Brazil",),
    )
    assert all(item.prompt.endswith((" ", ' "')) for item in items)
    assert next(
        item for item in items if item.name == "inv-balloon-opposite"
    ).prompt.endswith(' "')
    assert set(PANEL_NAMES) <= {item.name for item in items}


def test_loader_rejects_wrong_digest(tmp_path: Path) -> None:
    path = tmp_path / "lens-eval-multihop.json"
    path.write_text('{"items":[]}', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_multihop_items(path)


def test_loader_rejects_root_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["extra"] = True
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="root"):
        load_multihop_items(path)


def test_loader_rejects_wrong_item_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"].pop()
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="93"):
        load_multihop_items(path)


def test_loader_rejects_unexpected_item_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"][0]["extra"] = True
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="unexpected fields"):
        load_multihop_items(path)


def test_loader_rejects_duplicate_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"][1]["name"] = document["items"][0]["name"]
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="duplicated"):
        load_multihop_items(path)


@pytest.mark.parametrize("field", ["name", "prompt", "target"])
def test_loader_rejects_non_string_scalar_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    document = _valid_document()
    document["items"][0][field] = 1
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="non-empty strings"):
        load_multihop_items(path)


def test_loader_rejects_empty_intermediates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"][0]["intermediates"] = []
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="intermediates"):
        load_multihop_items(path)


def test_loader_rejects_empty_intermediate_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"][0]["intermediates"] = [""]
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="intermediates"):
        load_multihop_items(path)


def test_loader_rejects_prompt_without_final_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _valid_document()
    document["items"][0]["prompt"] = "missing final space"
    path = _write_with_matching_digest(tmp_path, monkeypatch, document)

    with pytest.raises(DatasetValidationError, match="final space"):
        load_multihop_items(path)
