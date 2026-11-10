from types import SimpleNamespace

import pytest
import torch

from jspace_llama.config import (
    AssetsConfig,
    ExperimentConfig,
    GenerationConfig,
    PromptConfig,
    SweepConfig,
)
from jspace_llama.runtime import (
    DimensionMismatchError,
    LayerBandError,
    ResourceExhaustionError,
    TokenizationError,
    download_lens_file,
    find_project_root,
    load_components,
    render_chat_prompt,
    require_cuda,
    resolve_layer_band,
    validate_intervention_tokens,
    validate_model_lens,
)


def test_defaults_pin_approved_assets_and_experiment() -> None:
    assets = AssetsConfig()
    generation = GenerationConfig()
    prompts = PromptConfig()
    sweep = SweepConfig()
    experiment = ExperimentConfig()

    assert assets.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert assets.lens_repo_id == "neuronpedia/jacobian-lens"
    assert assets.lens_filename == (
        "llama3.1-8b-it/jlens/Salesforce-wikitext/"
        "Llama-3.1-8B-Instruct_jacobian_lens.pt"
    )
    assert assets.dtype == "bfloat16"
    assert generation.do_sample is False
    assert 1 <= generation.max_new_tokens <= 32
    assert prompts.cases[0].text == (
        "The animal that spins webs has this many legs:"
    )
    assert len(prompts.cases) >= 3
    assert all("web" in case.text.lower() for case in prompts.cases)
    assert sweep.layers is None
    assert sweep.alphas == (4.0,)
    assert experiment.seed == 42
    assert experiment.source == "spider"
    assert experiment.target == "ant"


class RecordingTokenizer:
    def __init__(self) -> None:
        self.chat_calls: list[tuple[list[dict[str, str]], bool, bool]] = []
        self.encodings = {" spider": [101], " ant": [202], " split": [3, 4]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return f"<chat>{messages[0]['content']}</chat>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]


def test_chat_template_is_applied_once_to_instruction() -> None:
    tokenizer = RecordingTokenizer()

    rendered = render_chat_prompt(tokenizer, "Count spider legs.")

    assert rendered == "<chat>Count spider legs.</chat>"
    assert tokenizer.chat_calls == [
        (
            [{"role": "user", "content": "Count spider legs."}],
            False,
            True,
        )
    ]


def test_intervention_tokens_require_one_leading_space_token() -> None:
    tokenizer = RecordingTokenizer()

    assert validate_intervention_tokens(tokenizer, ("spider", "ant")) == {
        "spider": 101,
        "ant": 202,
    }

    with pytest.raises(TokenizationError, match="exactly one token"):
        validate_intervention_tokens(tokenizer, ("split",))


def test_model_lens_dimensions_and_fitted_band_are_validated() -> None:
    model = SimpleNamespace(d_model=4, n_layers=5)
    lens = SimpleNamespace(
        d_model=4,
        source_layers=[1, 2, 3],
        jacobians={layer: torch.eye(4) for layer in (1, 2, 3)},
    )

    validate_model_lens(model, lens, [1, 3])

    with pytest.raises(DimensionMismatchError, match="model d_model=4"):
        validate_model_lens(
            model,
            SimpleNamespace(
                d_model=5, source_layers=[1], jacobians={1: torch.eye(5)}
            ),
            [1],
        )
    with pytest.raises(LayerBandError, match="not fitted"):
        validate_model_lens(model, lens, [0, 1])


def test_layer_band_uses_upstream_default_and_rejects_empty_result() -> None:
    model = SimpleNamespace(d_model=2, n_layers=4)
    lens = SimpleNamespace(
        d_model=2,
        source_layers=[1, 2],
        jacobians={1: torch.eye(2), 2: torch.eye(2)},
    )
    calls: list[tuple[object, object, object]] = []

    def default_band(model_arg, lens_arg, band):
        calls.append((model_arg, lens_arg, band))
        return [1, 2] if band is None else [layer for layer in band if layer in (1, 2)]

    assert resolve_layer_band(model, lens, None, default_band_fn=default_band) == [1, 2]
    assert calls == [(model, lens, None)]

    with pytest.raises(LayerBandError, match="no fitted layers"):
        resolve_layer_band(model, lens, [0], default_band_fn=default_band)


def test_cuda_requirement_reports_gpu_and_bfloat16() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA L4",
        )
    )

    info = require_cuda(fake_torch)

    assert info.device == "cuda"
    assert info.gpu_name == "NVIDIA L4"
    assert info.dtype == "bfloat16"

    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        )


def test_download_and_component_loading_use_exact_pinned_assets() -> None:
    assets = AssetsConfig()
    download_calls: list[dict[str, str]] = []

    def downloader(**kwargs):
        download_calls.append(kwargs)
        return "C:/cache/lens.pt"

    revision = "a" * 40
    assert (
        download_lens_file(
            assets,
            resolved_revision=revision,
            downloader=downloader,
        )
        == "C:/cache/lens.pt"
    )
    assert download_calls == [
        {
            "repo_id": assets.lens_repo_id,
            "filename": assets.lens_filename,
            "revision": revision,
        }
    ]

    events: list[tuple] = []

    class FakeHFModel:
        @classmethod
        def load(cls, model_id, **kwargs):
            events.append(("model", model_id, kwargs))
            return SimpleNamespace(
                d_model=2,
                n_layers=3,
                tokenizer=object(),
                lm_head=SimpleNamespace(weight=torch.zeros(5, 2)),
            )

    class FakeLens:
        @classmethod
        def load(cls, path):
            events.append(("lens", path))
            return SimpleNamespace(
                d_model=2,
                source_layers=[1],
                jacobians={1: torch.eye(2)},
            )

    class FakeDictionary:
        def __init__(self, model, lens, *, device):
            events.append(("dictionary", model, lens, device))

    bundle = load_components(
        assets,
        "C:/cache/lens.pt",
        layers=[1],
        hf_model_cls=FakeHFModel,
        lens_cls=FakeLens,
        dictionary_cls=FakeDictionary,
    )

    assert bundle.model is not None
    assert bundle.lens is not None
    assert bundle.dictionary is not None
    assert events[0] == (
        "model",
        assets.model_id,
        {"dtype": torch.bfloat16, "device": "cuda"},
    )
    assert events[1] == ("lens", "C:/cache/lens.pt")
    assert events[2][3] == "cuda"


def test_component_loading_preserves_oom_as_specific_inconclusive_error() -> None:
    class OomHFModel:
        @classmethod
        def load(cls, model_id, **kwargs):
            raise torch.OutOfMemoryError("allocation failed")

    with pytest.raises(ResourceExhaustionError, match="out of memory"):
        load_components(
            AssetsConfig(),
            "C:/cache/lens.pt",
            layers=[1],
            hf_model_cls=OomHFModel,
            lens_cls=object(),
            dictionary_cls=object(),
        )


def test_project_root_is_found_from_nested_local_or_colab_directory(
    tmp_path,
) -> None:
    project = tmp_path / "checkout" / "projects" / "jspace_llama"
    nested = project / "notebooks" / "work"
    (project / "src" / "jspace_llama").mkdir(parents=True)
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "jspace-llama"\n', encoding="utf-8"
    )

    assert find_project_root(nested) == project.resolve()
