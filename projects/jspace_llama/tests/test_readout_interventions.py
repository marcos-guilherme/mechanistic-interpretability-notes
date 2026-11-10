import contextlib
import weakref
from types import SimpleNamespace

import pytest
import torch
from jspace import LensSwap

from jspace_llama.experiment import GenerationRequest, ModelExperimentRunner

from jspace_llama.interventions import (
    MemoryBoundedLensSwap,
    NormMatchedRandomControl,
    InterventionPreparationError,
    SwapLayer,
    apply_causal_clamp,
    apply_norm_matched_random,
    make_random_directions,
    prepare_swap_layers,
)
from jspace_llama.readout import ReadoutError, read_workspace


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return {7: " spider", 8: " animal", 9: " legs", 99: ":"}[token_ids[0]]


class ReadoutDictionary:
    def __init__(self, *, fail_layer: int | None = None) -> None:
        self.lens = SimpleNamespace(source_layers=[1, 2])
        self.model = SimpleNamespace(tokenizer=FakeTokenizer())
        self.evicted: list[int] = []
        self.fail_layer = fail_layer

    def unit(self, layer: int) -> torch.Tensor:
        if layer == self.fail_layer:
            raise RuntimeError("dictionary materialization failed")
        return torch.eye(3)

    def evict(self, layer: int) -> None:
        self.evicted.append(layer)


def fake_activations(model, token_ids, layers, max_seq_len):
    assert tuple(token_ids) == (99,)
    assert max_seq_len == 32
    return [99], {
        1: torch.tensor([[[1.0, 0.0, 0.0]]]),
        2: torch.tensor([[[0.0, 1.0, 0.0]]]),
    }


def fake_pursuit(hidden, dictionary, *, k):
    assert k == 3
    if hidden[0] == 1:
        return SimpleNamespace(
            indices=torch.tensor([7, 8]),
            coeffs=torch.tensor([2.0, 1.0]),
            var_explained=0.75,
        )
    return SimpleNamespace(
        indices=torch.tensor([9, 7]),
        coeffs=torch.tensor([3.0, 0.5]),
        var_explained=0.5,
    )


def test_workspace_readout_is_json_ready_and_evicts_each_layer() -> None:
    dictionary = ReadoutDictionary()

    result = read_workspace(
        model=object(),
        dictionary=dictionary,
        token_ids=(99,),
        layers=[1, 2],
        positions=[-1],
        k=3,
        max_seq_len=32,
        activation_reader=fake_activations,
        pursuit_fn=fake_pursuit,
        focus_concept="spider",
    )

    assert dictionary.evicted == [1, 2]
    assert result["spider"] == {
        "concept": "spider",
        "present": True,
        "rank": 2,
        "peak_coefficient": 2.0,
    }
    assert result["slots"][0] == {
        "layer": 1,
        "position": -1,
        "position_token": ":",
        "entries": [
            {"token_id": 7, "token": " spider", "coefficient": 2.0},
            {"token_id": 8, "token": " animal", "coefficient": 1.0},
        ],
        "variance_explained": 0.75,
    }


def test_workspace_readout_evicts_failing_layer() -> None:
    dictionary = ReadoutDictionary(fail_layer=2)

    with pytest.raises(RuntimeError, match="materialization failed"):
        read_workspace(
            model=object(),
            dictionary=dictionary,
            token_ids=(99,),
            layers=[1, 2],
            k=3,
            max_seq_len=32,
            activation_reader=fake_activations,
            pursuit_fn=fake_pursuit,
        )

    assert dictionary.evicted == [1, 2]


def test_workspace_releases_layer_tensor_before_materializing_next() -> None:
    class LifetimeDictionary(ReadoutDictionary):
        def __init__(self) -> None:
            super().__init__()
            self.previous: weakref.ReferenceType[torch.Tensor] | None = None

        def unit(self, layer: int) -> torch.Tensor:
            if layer == 2:
                assert self.previous is not None
                assert self.previous() is None, "previous layer tensor is retained"
            tensor = torch.eye(3)
            self.previous = weakref.ref(tensor)
            return tensor

    dictionary = LifetimeDictionary()

    read_workspace(
        model=object(),
        dictionary=dictionary,
        token_ids=(99,),
        layers=[1, 2],
        positions=[-1],
        k=3,
        max_seq_len=32,
        activation_reader=fake_activations,
        pursuit_fn=fake_pursuit,
    )

    assert dictionary.evicted == [1, 2]


class SwapDictionary:
    def __init__(
        self,
        *,
        fail_layer: int | None = None,
        missing_layer: int | None = None,
    ) -> None:
        self.evicted: list[int] = []
        self.fail_layer = fail_layer
        self.missing_layer = missing_layer

    def token_vector(self, layer: int, token: str):
        if layer == self.fail_layer and token == "ant":
            raise RuntimeError("target vector failed")
        if layer == self.missing_layer and token == "ant":
            raise KeyError(token)
        vectors = {
            "spider": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "ant": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        }
        return {"spider": 7, "ant": 8}[token], vectors[token]

    def evict(self, layer: int) -> None:
        self.evicted.append(layer)


def test_swap_preparation_evicts_successful_and_failing_layers() -> None:
    dictionary = SwapDictionary()

    prepared = prepare_swap_layers(
        dictionary, layers=[1, 2], source="spider", target="ant"
    )

    assert list(prepared) == [1, 2]
    assert dictionary.evicted == [1, 2]
    assert prepared[1].vectors.device.type == "cpu"

    failing = SwapDictionary(fail_layer=2)
    with pytest.raises(RuntimeError, match="target vector failed"):
        prepare_swap_layers(
            failing, layers=[1, 2], source="spider", target="ant"
        )
    assert failing.evicted == [1, 2]

    missing = SwapDictionary(missing_layer=2)
    with pytest.raises(InterventionPreparationError):
        prepare_swap_layers(
            missing, layers=[1, 2], source="spider", target="ant"
        )


def test_causal_clamp_matches_upstream_math_and_noop_is_exact() -> None:
    vectors = torch.eye(2)
    layer = SwapLayer(
        vectors=vectors,
        pseudoinverse=torch.linalg.pinv(vectors.T),
        noop=False,
    )
    hidden = torch.tensor([[[2.0, -1.0], [2.0, 1.0]]])

    patched = apply_causal_clamp(hidden, layer, alpha=1.0)

    assert torch.equal(patched, torch.tensor([[[0.0, 2.0], [0.0, 3.0]]]))

    noop = SwapLayer(
        vectors=torch.ones(2, 2),
        pseudoinverse=torch.ones(2, 2),
        noop=True,
    )
    assert torch.equal(apply_causal_clamp(hidden, noop, alpha=4.0), hidden)


def test_random_control_is_deterministic_orthogonal_and_norm_matched() -> None:
    vectors = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    prepared = {
        3: SwapLayer(
            vectors=vectors,
            pseudoinverse=torch.linalg.pinv(vectors.T),
            noop=False,
        )
    }
    hidden = torch.tensor([[[2.0, 1.0, 5.0, -2.0], [1.0, 0.0, 0.0, 4.0]]])

    directions_a = make_random_directions(prepared, seed=42)
    directions_b = make_random_directions(prepared, seed=42)
    directions_c = make_random_directions(prepared, seed=43)

    assert torch.equal(directions_a[3], directions_b[3])
    assert not torch.equal(directions_a[3], directions_c[3])
    assert torch.allclose(vectors @ directions_a[3], torch.zeros(2), atol=1e-6)

    causal = apply_causal_clamp(hidden, prepared[3], alpha=4.0) - hidden
    random_patched = apply_norm_matched_random(
        hidden,
        prepared[3],
        directions_a[3],
        alpha=4.0,
    )
    random_delta = random_patched - hidden

    assert torch.allclose(
        random_delta.norm(dim=-1), causal.norm(dim=-1), atol=1e-6
    )


def test_patch_hook_context_patches_prefill_but_not_decode() -> None:
    model = SimpleNamespace(layers=[torch.nn.Identity() for _ in range(4)])
    dictionary = SwapDictionary()

    hook = MemoryBoundedLensSwap(
        model,
        dictionary,
        "spider",
        "ant",
        layers=[1],
        alpha=1.0,
    )
    hidden = torch.tensor([[[2.0, 1.0, 5.0, -2.0]]])

    with hook:
        prefill = model.layers[1](hidden.clone())
        decode = model.layers[1](hidden.clone())

    assert torch.equal(prefill, torch.tensor([[[0.0, 3.0, 5.0, -2.0]]]))
    assert torch.equal(decode, hidden)


def test_custom_clamp_matches_upstream_lens_swap() -> None:
    model = SimpleNamespace(layers=[torch.nn.Identity() for _ in range(3)])
    hidden = torch.tensor([[[2.0, -1.0, 5.0, -2.0], [2.0, 1.0, 0.0, 4.0]]])
    upstream = LensSwap(
        model,
        SwapDictionary(),
        "spider",
        "ant",
        layers=[1],
        alpha=4.0,
        patch_generation=False,
        mode="clamp",
    )
    custom = MemoryBoundedLensSwap(
        model,
        SwapDictionary(),
        "spider",
        "ant",
        layers=[1],
        alpha=4.0,
    )

    with upstream:
        upstream_prefill = model.layers[1](hidden.clone())
    with custom:
        custom_prefill = model.layers[1](hidden.clone())

    assert torch.equal(custom_prefill, upstream_prefill)


class TinyGeneratingModel:
    def __init__(self) -> None:
        self.layers = [torch.nn.Identity() for _ in range(3)]
        self.hidden = torch.tensor([[[2.0, 1.0, 5.0, -2.0]]])
        self.calls: list[dict[str, object]] = []
        self._hf_model = self
        self.input_device = torch.device("cpu")
        self.tokenizer = SimpleNamespace(
            eos_token_id=2,
            decode=lambda token_ids: "continuation",
        )

    def generate(
        self,
        input_ids,
        *,
        max_new_tokens,
        do_sample,
        pad_token_id,
    ):
        prefill = self.layers[1](self.hidden.clone())
        decode = self.layers[1](self.hidden.clone())
        self.calls.append(
            {
                "input_ids": input_ids.tolist(),
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": pad_token_id,
                "prefill": prefill,
                "decode": decode,
            }
        )
        return torch.cat((input_ids, torch.tensor([[2]])), dim=1)


def _request(condition: str) -> GenerationRequest:
    return GenerationRequest(
        prompt_name="main",
        instruction="instruction",
        rendered_prompt="<chat>instruction</chat>",
        token_ids=(1, 42),
        condition=condition,
        layers=(1,),
        alpha=1.0,
        seed=42,
        source="spider",
        target="ant",
        max_new_tokens=7,
    )


def test_model_runner_exercises_real_hook_contexts() -> None:
    model = TinyGeneratingModel()
    runner = ModelExperimentRunner(model, SwapDictionary())

    for condition in ("baseline", "noop", "random", "target"):
        assert runner(_request(condition)) == "continuation"

    baseline, noop, random, target = model.calls
    assert all(call["input_ids"] == [[1, 42]] for call in model.calls)
    assert torch.equal(noop["prefill"], model.hidden)
    assert not torch.equal(random["prefill"], model.hidden)
    assert not torch.equal(target["prefill"], model.hidden)
    for call in model.calls:
        assert torch.equal(call["decode"], model.hidden)
