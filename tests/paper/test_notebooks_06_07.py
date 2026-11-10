import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _notebook_source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_06_uses_published_circuit_and_semantic_activation_patching() -> None:
    source = _notebook_source("06_circuit_discovery.ipynb")

    assert "PUBLISHED_CIRCUIT" in source
    assert "activation_patch_matrix" in source
    assert "SemanticPosition" in source
    assert "não é uma descoberta do zero" in source
    assert "results/paper/06_circuit_discovery.json" in source.replace("\\", "/")


def test_notebook_07_compares_historical_and_modern_path_patching() -> None:
    source = _notebook_source("07_path_patching.ipynb")

    assert "paper_path_patch" in source
    assert "get_act_patch_direct_path" in source
    assert "Duplicate/Induction" in source
    assert "S-Inhibition" in source
    assert "Name Movers" in source
    assert "results/paper/07_path_patching.json" in source.replace("\\", "/")
