from pathlib import Path

import nbformat

from jspace_llama.config import (
    ExperimentConfig,
    GenerationConfig,
    PromptCase,
    PromptConfig,
    SweepConfig,
)
from jspace_llama.experiment import (
    CONDITIONS,
    classify_continuation,
    prepare_prompts,
    run_experiment,
)


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_readout_and_causal_swap.ipynb"


class ChatTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        instruction = messages[0]["content"]
        self.calls.append(instruction)
        return f"<chat>{instruction}</chat>"


def test_classification_records_answer_leakage_and_degradation() -> None:
    classified = classify_continuation(
        "Eight (8) is mentioned; six is not. An ant appears."
    )

    assert classified == {
        "answer_evidence": {"eight": True, "six": True},
        "target_entity_leakage": True,
        "degradation": {"empty": False, "repetition": False},
    }
    assert classify_continuation("   ")["degradation"]["empty"] is True
    assert (
        classify_continuation("legs legs legs legs")["degradation"]["repetition"]
        is True
    )


def test_orchestration_reuses_chat_prompt_and_preserves_condition_order() -> None:
    config = ExperimentConfig(
        generation=GenerationConfig(max_new_tokens=7),
        prompts=PromptConfig(
            cases=(
                PromptCase("first", "The web spinner has how many legs?"),
                PromptCase("second", "How many legs has an animal spinning webs?"),
            )
        ),
        sweep=SweepConfig(layers=(3, 4), alphas=(1.0, 4.0)),
    )
    tokenizer = ChatTokenizer()
    requests = []

    def generate(request):
        requests.append(request)
        return {
            "baseline": "eight",
            "noop": "8",
            "random": "six",
            "target": "six ant",
        }[request.condition]

    prepared = prepare_prompts(config, tokenizer)
    result = run_experiment(
        config,
        tokenizer,
        layers=[3, 4],
        generate=generate,
        prepared_prompts=prepared,
    )

    assert tokenizer.calls == [case.text for case in config.prompts.cases]
    assert [request.condition for request in requests] == list(CONDITIONS) * 4
    assert all(request.rendered_prompt.startswith("<chat>") for request in requests)
    assert all(request.max_new_tokens == 7 for request in requests)
    assert len(result["observations"]) == 16
    first = result["observations"][0]
    assert first["condition"] == "baseline"
    assert first["continuation"] == "eight"
    assert first["parameters"] == {
        "layers": [3, 4],
        "alpha": 1.0,
        "seed": 42,
    }
    target = result["observations"][3]
    assert target["classification"]["answer_evidence"]["six"] is True
    assert target["classification"]["target_entity_leakage"] is True


def test_notebook_is_thin_ordered_and_contains_no_outputs_or_secrets() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    sources = "\n".join(cell.source for cell in notebook.cells)
    code_sources = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert len(notebook.cells) >= 6
    assert sources.index("ExperimentConfig") < sources.index("require_cuda")
    assert sources.index("require_cuda") < sources.index("read_workspace")
    assert sources.index("read_workspace") < sources.index("run_experiment")
    assert sources.index("run_experiment") < sources.index("write_text")
    assert "def " not in code_sources
    assert "HF_TOKEN" not in sources
    assert "hf_" + "token" not in sources.lower()
    assert "api_key" not in sources.lower()
    assert "find_project_root" in sources
    assert 'project_root / "results" / "readout_and_causal_swap.json"' in sources
    assert "output_path.parent.mkdir(parents=True, exist_ok=True)" in sources
    assert 'Path("../results' not in sources
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
