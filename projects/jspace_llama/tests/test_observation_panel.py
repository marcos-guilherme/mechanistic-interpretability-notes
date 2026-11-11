from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from jspace_llama.observation_data import MultihopItem, PANEL_NAMES
from jspace_llama.observation_panel import (
    read_final_activations,
    read_qualitative_panel,
    select_panel,
)


def _item(name: str) -> MultihopItem:
    return MultihopItem(
        name=name,
        prompt=f"prompt for {name} ",
        target=f"target-{name}",
        intermediates=(f"intermediate-{name}",),
    )


def fake_93_items() -> tuple[MultihopItem, ...]:
    ordinary = [_item(f"ordinary-{index}") for index in range(81)]
    panel = [_item(name) for name in reversed(PANEL_NAMES)]
    return tuple(ordinary[:40] + panel + ordinary[40:])


def _direct_records(
    items: tuple[MultihopItem, ...],
) -> tuple[dict[str, object], ...]:
    records = []
    for item in items:
        methods = {}
        for offset, method in enumerate(("jlens", "logit_lens")):
            methods[method] = [
                {
                    "layer": layer,
                    "tokens": [
                        {
                            "token_id": token_id,
                            "token": f"{method}-{layer}-{token_id}",
                            "rank": rank,
                        }
                        for rank, token_id in enumerate(
                            range(offset, offset + 12),
                            start=1,
                        )
                    ],
                }
                for layer in (11, 12, 13)
            ]
        records.append(
            {
                "name": item.name,
                "prompt": item.prompt,
                "target": item.target,
                "intermediates": [
                    {
                        "text": item.intermediates[0],
                        "token_id": 99,
                        "single_token": True,
                        "jlens": {
                            "ranks": [{"layer": 12, "rank": 4}],
                            "best_rank": 4,
                            "best_layer": 12,
                            "all_layers_best_rank": 2,
                            "all_layers_best_layer": 11,
                        },
                        "logit_lens": {
                            "ranks": [{"layer": 12, "rank": 6}],
                            "best_rank": 6,
                            "best_layer": 12,
                            "all_layers_best_rank": 3,
                            "all_layers_best_layer": 11,
                        },
                    }
                ],
                "top_tokens": methods,
            }
        )
    return tuple(records)


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"decoded-{token_ids[0]}"


class FakeDictionary:
    def __init__(self) -> None:
        self.model = SimpleNamespace(tokenizer=FakeTokenizer())
        self.lens = SimpleNamespace(
            apply=lambda *args, **kwargs: pytest.fail(
                "panel must consume direct records instead of recomputing logits"
            )
        )
        self.unit_calls: list[int] = []
        self.evicted: list[int] = []

    def unit(self, layer: int) -> torch.Tensor:
        self.unit_calls.append(layer)
        return torch.eye(4)

    def evict(self, layer: int) -> None:
        self.evicted.append(layer)


def _activation_reader(model, items, layers):
    assert tuple(item.name for item in items) == PANEL_NAMES
    assert tuple(layers) == (12, 13)
    return {
        item.name: {
            layer: torch.tensor([float(layer), 1.0, 2.0, 3.0])
            for layer in layers
        }
        for item in items
    }


def _pursuit(hidden, vectors, *, k):
    assert hidden.device.type == vectors.device.type == "cpu"
    assert k == 25
    return SimpleNamespace(
        indices=torch.tensor([13, 12]),
        coeffs=torch.tensor([0.25, 0.75]),
        var_explained=torch.tensor(0.625),
    )


def test_panel_is_fixed_and_complete() -> None:
    selected = select_panel(fake_93_items())

    assert tuple(item.name for item in selected) == PANEL_NAMES


def test_panel_rejects_a_missing_registered_item() -> None:
    items = tuple(
        item for item in fake_93_items() if item.name != PANEL_NAMES[-1]
    )

    with pytest.raises(ValueError, match=PANEL_NAMES[-1]):
        select_panel(items)


def test_dictionary_is_materialized_once_and_evicted_per_layer() -> None:
    panel = tuple(_item(name) for name in PANEL_NAMES)
    dictionary = FakeDictionary()
    forbidden_model = SimpleNamespace(
        generate=lambda *args, **kwargs: pytest.fail(
            "qualitative panel must not generate"
        )
    )

    result = read_qualitative_panel(
        model=forbidden_model,
        dictionary=dictionary,
        items=fake_93_items(),
        direct_records=_direct_records(panel),
        layers=(12, 13),
        k=25,
        top=10,
        activation_reader=_activation_reader,
        pursuit_fn=_pursuit,
    )

    assert dictionary.unit_calls == [12, 13]
    assert dictionary.evicted == [12, 13]
    assert len(result["items"]) == 12
    assert [item["name"] for item in result["items"]] == list(PANEL_NAMES)
    first = result["items"][0]
    assert set(first) == {
        "name",
        "prompt",
        "target",
        "intermediates",
        "top_tokens",
        "pursuit",
    }
    jlens = first["intermediates"][0]["jlens"]
    assert jlens["best_rank"] == 4
    assert jlens["best_layer"] == 12
    assert jlens["all_layers_best_rank"] == 2
    assert jlens["all_layers_best_layer"] == 11
    assert set(first["top_tokens"]) == {"jlens", "logit_lens"}
    assert [slot["layer"] for slot in first["top_tokens"]["jlens"]] == [12, 13]
    assert len(first["top_tokens"]["jlens"][0]["tokens"]) == 10
    assert first["pursuit"][0] == {
        "layer": 12,
        "tokens": [
            {
                "token_id": 12,
                "token": "decoded-12",
                "rank": 1,
                "coefficient": 0.75,
            },
            {
                "token_id": 13,
                "token": "decoded-13",
                "rank": 2,
                "coefficient": 0.25,
            },
        ],
        "variance_explained": 0.625,
    }
    json.dumps(result)


def test_pursuit_failure_still_evicts_current_dictionary_layer() -> None:
    panel = tuple(_item(name) for name in PANEL_NAMES)
    dictionary = FakeDictionary()

    def failing_pursuit(hidden, vectors, *, k):
        raise RuntimeError("pursuit failed")

    with pytest.raises(RuntimeError, match="pursuit failed"):
        read_qualitative_panel(
            model=object(),
            dictionary=dictionary,
            items=panel,
            direct_records=_direct_records(panel),
            layers=(12, 13),
            activation_reader=_activation_reader,
            pursuit_fn=failing_pursuit,
        )

    assert dictionary.unit_calls == [12]
    assert dictionary.evicted == [12]


class RecordingModel:
    def __init__(self) -> None:
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )
        self.input_device = torch.device("cpu")
        self.encoded: list[tuple[str, int]] = []

    def encode(self, prompt: str, *, max_length: int) -> torch.Tensor:
        self.encoded.append((prompt, max_length))
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]],
            requires_grad=True,
        )
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def generate(self, *args, **kwargs):
        raise AssertionError("activation capture must not generate")


def test_final_activations_capture_only_last_position_on_cpu() -> None:
    model = RecordingModel()
    items = (_item(PANEL_NAMES[0]), _item(PANEL_NAMES[1]))

    activations = read_final_activations(model, items, layers=(0, 1))

    assert model.encoded == [
        (items[0].prompt, 512),
        (items[1].prompt, 512),
    ]
    assert set(activations) == {items[0].name, items[1].name}
    for by_layer in activations.values():
        assert set(by_layer) == {0, 1}
        assert all(tensor.device.type == "cpu" for tensor in by_layer.values())
        assert all(not tensor.requires_grad for tensor in by_layer.values())
        assert torch.equal(by_layer[0], torch.tensor([3.0, 30.0]))
        assert torch.equal(by_layer[1], torch.tensor([3.0, 30.0]))
