from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT = Path(__file__).parents[1]
NOTEBOOK = PROJECT / "notebooks" / "02_observe_multihop_jspace.ipynb"


def test_observation_notebook_is_thin_and_output_free() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    ).lower()

    assert "run_observation_pipeline" in code
    assert "run_observation_pipeline(project_root=project_root)" in code
    assert "gradient_pursuit" not in code
    assert "lens.apply" not in code
    assert "generate(" not in code
    assert "LensSwap" not in code
    assert "HF_TOKEN" not in code
    assert "does not generate text" in markdown
    assert "does not modify activations" in markdown
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
