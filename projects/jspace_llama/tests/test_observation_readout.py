from __future__ import annotations

import torch
import pytest

from jspace_llama.observation_data import MultihopItem
from jspace_llama.observation_readout import (
    aggregate_pass_at_k,
    evaluate_direct_readout,
    rank_one_based,
    ObservationReadoutError,
)
from jspace_llama.runtime import TokenizationError


def test_rank_is_one_based_and_uses_competition_ranking() -> None:
    logits = torch.tensor([0.0, 3.0, 3.0, 2.0])

    assert rank_one_based(logits, 1) == 1
    assert rank_one_based(logits, 3) == 3


def test_pass_at_k_averages_each_items_intermediate_fraction() -> None:
    records = [
        {"intermediates": [{"best_rank": 1}]},
        {"intermediates": [{"best_rank": 2}, {"best_rank": 9}]},
    ]

    assert aggregate_pass_at_k(records) == {
        "n_items": 2,
        "n_intermediates": 3,
        "pass@1": 0.5,
        "pass@5": 0.75,
        "pass@10": 1.0,
    }


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class GenerateForbiddenModel:
    tokenizer = FakeTokenizer()

    def encode(self, prompt: str, *, max_length: int) -> torch.Tensor:
        assert max_length == 512
        return torch.tensor(
            [[1, 2] if prompt.endswith(" ") else [1, 3]],
            dtype=torch.long,
        )

    def generate(self, *args, **kwargs):
        raise AssertionError("direct readout must not generate")


class FakeLens:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply(
        self,
        model,
        prompt,
        *,
        layers,
        positions,
        max_seq_len,
        use_jacobian,
    ):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "layers": layers,
                "positions": positions,
                "max_seq_len": max_seq_len,
                "use_jacobian": use_jacobian,
            }
        )
        by_method = {
            True: {
                1: [0.0, 1.0, 5.0, 4.0],
                2: [0.0, 5.0, 1.0, 4.0],
                3: [0.0, 4.0, 3.0, 5.0],
            },
            False: {
                1: [0.0, 5.0, 4.0, 3.0],
                2: [0.0, 1.0, 5.0, 4.0],
                3: [0.0, 3.0, 4.0, 5.0],
            },
        }
        logits = {
            layer: torch.tensor([by_method[use_jacobian][layer]])
            for layer in layers
        }
        return logits, torch.empty(0), torch.empty(0, dtype=torch.long)


class FakeDictionary:
    def __init__(self) -> None:
        self.model = GenerateForbiddenModel()
        self.lens = FakeLens()


def test_direct_readout_keeps_prompts_methods_and_scopes_distinct() -> None:
    model = GenerateForbiddenModel()
    dictionary = FakeDictionary()
    items = (
        MultihopItem(
            name="carnival-ocean",
            prompt="raw final space ",
            target="unused",
            intermediates=("first",),
        ),
        MultihopItem(
            name="ordinary",
            prompt='preserved delimiter "',
            target="unused",
            intermediates=("second",),
        ),
    )
    audit = (
        {
            "name": "carnival-ocean",
            "prompt": "raw final space ",
            "prompt_token_ids": [1, 2],
            "intermediates": [
                {
                    "text": "first",
                    "token_id": 1,
                    "single_token": True,
                }
            ],
        },
        {
            "name": "ordinary",
            "prompt": 'preserved delimiter "',
            "prompt_token_ids": [1, 3],
            "intermediates": [
                {
                    "text": "second",
                    "token_id": 0,
                    "single_token": False,
                }
            ],
        },
    )

    result = evaluate_direct_readout(
        model,
        dictionary,
        items,
        audit,
        band_layers=(2, 3),
        all_layers=(1, 2, 3),
        detail_names=frozenset({"carnival-ocean", "ordinary"}),
        top=2,
    )

    assert len(dictionary.lens.calls) == 4
    assert [call["prompt"] for call in dictionary.lens.calls] == [
        "raw final space ",
        "raw final space ",
        'preserved delimiter "',
        'preserved delimiter "',
    ]
    assert {call["use_jacobian"] for call in dictionary.lens.calls} == {
        True,
        False,
    }
    assert all(call["positions"] == [-1] for call in dictionary.lens.calls)
    assert all(call["layers"] == [1, 2, 3] for call in dictionary.lens.calls)
    assert all(call["max_seq_len"] == 512 for call in dictionary.lens.calls)

    panel, ordinary = result["items"]
    first = panel["intermediates"][0]
    assert first["jlens"]["ranks"] == [
        {"layer": 1, "rank": 3},
        {"layer": 2, "rank": 1},
        {"layer": 3, "rank": 2},
    ]
    assert first["jlens"]["best_rank"] == 1
    assert first["jlens"]["best_layer"] == 2
    assert first["jlens"]["all_layers_best_rank"] == 1
    assert first["jlens"]["all_layers_best_layer"] == 2
    assert "band_best_rank" not in first["jlens"]
    assert first["logit_lens"]["best_rank"] == 3
    assert first["logit_lens"]["best_layer"] == 2
    assert first["logit_lens"]["all_layers_best_rank"] == 1
    assert first["logit_lens"]["all_layers_best_layer"] == 1
    assert [slot["layer"] for slot in panel["top_tokens"]["jlens"]] == [2, 3]
    assert [slot["layer"] for slot in panel["top_tokens"]["logit_lens"]] == [
        2,
        3,
    ]
    assert panel["top_tokens"]["jlens"][0] == {
        "layer": 2,
        "tokens": [
            {"token_id": 1, "token": "token-1", "rank": 1},
            {"token_id": 3, "token": "token-3", "rank": 2},
        ],
    }
    assert "top_tokens" not in ordinary
    assert result["metrics"]["jlens"]["band"]["pass@1"] == 0.5
    assert result["metrics"]["logit_lens"]["band"]["pass@1"] == 0.0
    assert result["single_token_metrics"]["jlens"]["band"]["n_items"] == 1


@pytest.mark.parametrize(
    ("logits", "token_id"),
    [
        (torch.tensor([[0.0, float("nan")]]), 1),
        (torch.tensor([[[0.0, 1.0]]]), 1),
        (torch.tensor([[0.0, 1.0]]), 9),
    ],
)
def test_direct_readout_rejects_malformed_numerical_outputs(
    logits: torch.Tensor,
    token_id: int,
) -> None:
    model = GenerateForbiddenModel()
    dictionary = FakeDictionary()
    dictionary.lens.apply = lambda *args, **kwargs: (
        {1: logits},
        torch.empty(0),
        torch.empty(0),
    )
    item = MultihopItem("one", "raw final space ", "target", ("middle",))
    audit = (
        {
            "name": "one",
            "prompt": item.prompt,
            "prompt_token_ids": [1, 2],
            "intermediates": [
                {
                    "text": "middle",
                    "token_id": token_id,
                    "single_token": True,
                }
            ],
        },
    )

    with pytest.raises(ObservationReadoutError):
        evaluate_direct_readout(
            model,
            dictionary,
            (item,),
            audit,
            band_layers=(1,),
            all_layers=(1,),
            detail_names=frozenset(),
        )


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        (),
        ({"unused": torch.tensor([0.0])},),
        (
            {1: torch.tensor([[0.0, 1.0]])},
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        ),
        [
            {1: torch.tensor([[0.0, 1.0]])},
            torch.empty(0),
            torch.empty(0),
        ],
    ],
    ids=["none", "empty-tuple", "one-tuple", "four-tuple", "list"],
)
def test_direct_readout_maps_malformed_lens_envelopes(
    envelope: object,
) -> None:
    model = GenerateForbiddenModel()
    dictionary = FakeDictionary()
    dictionary.lens.apply = lambda *args, **kwargs: envelope
    item = MultihopItem("one", "raw final space ", "target", ("middle",))
    audit = (
        {
            "name": "one",
            "prompt": item.prompt,
            "prompt_token_ids": [1, 2],
            "intermediates": [
                {
                    "text": "middle",
                    "token_id": 1,
                    "single_token": True,
                }
            ],
        },
    )

    with pytest.raises(
        ObservationReadoutError,
        match="lens output envelope",
    ):
        evaluate_direct_readout(
            model,
            dictionary,
            (item,),
            audit,
            band_layers=(1,),
            all_layers=(1,),
            detail_names=frozenset(),
        )


def test_direct_readout_does_not_wrap_lens_apply_exceptions() -> None:
    model = GenerateForbiddenModel()
    dictionary = FakeDictionary()

    def fail(*args, **kwargs):
        raise RuntimeError("lens programming failure")

    dictionary.lens.apply = fail
    item = MultihopItem("one", "raw final space ", "target", ("middle",))
    audit = (
        {
            "name": "one",
            "prompt": item.prompt,
            "prompt_token_ids": [1, 2],
            "intermediates": [
                {
                    "text": "middle",
                    "token_id": 1,
                    "single_token": True,
                }
            ],
        },
    )

    with pytest.raises(RuntimeError, match="lens programming failure"):
        evaluate_direct_readout(
            model,
            dictionary,
            (item,),
            audit,
            band_layers=(1,),
            all_layers=(1,),
            detail_names=frozenset(),
        )


def test_direct_readout_rejects_lens_prompt_tokenization_drift() -> None:
    model = GenerateForbiddenModel()
    dictionary = FakeDictionary()
    item = MultihopItem("one", "raw final space ", "target", ("middle",))
    audit = (
        {
            "name": "one",
            "prompt": item.prompt,
            "prompt_token_ids": [999],
            "intermediates": [
                {
                    "text": "middle",
                    "token_id": 1,
                    "single_token": True,
                }
            ],
        },
    )

    with pytest.raises(TokenizationError, match="tokens differ from audit"):
        evaluate_direct_readout(
            model,
            dictionary,
            (item,),
            audit,
            band_layers=(1,),
            all_layers=(1,),
        )

    assert dictionary.lens.calls == []
