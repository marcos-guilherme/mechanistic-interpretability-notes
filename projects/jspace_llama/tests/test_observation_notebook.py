from __future__ import annotations

import ast
import builtins
from pathlib import Path

import nbformat
import pytest


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
    assert "from jspace_llama.runtime import find_project_root" in code
    assert "project_root = find_project_root()" in code
    assert "Path.cwd()" not in code
    assert 'project_root / "pyproject.toml"' not in code
    assert "project_root.parent" not in code
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


def test_observation_notebook_guards_every_post_import_cell() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
    import_tree = ast.parse(code_cells[0].source)

    first = import_tree.body[0]
    assert isinstance(first, ast.Assign)
    assert [target.id for target in first.targets if isinstance(target, ast.Name)] == [
        "observation_imports_ready"
    ]
    assert isinstance(first.value, ast.Constant) and first.value.value is False

    last = import_tree.body[-1]
    assert isinstance(last, ast.Assign)
    assert [target.id for target in last.targets if isinstance(target, ast.Name)] == [
        "observation_imports_ready"
    ]
    assert isinstance(last.value, ast.Constant) and last.value.value is True
    assert all(
        cell.source.startswith(
            'if not globals().get("observation_imports_ready", False):'
        )
        for cell in code_cells[1:]
    )


def test_failed_import_cannot_run_stale_notebook_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
    calls: list[str] = []
    namespace = {
        "observation_imports_ready": True,
        "find_project_root": lambda: calls.append("root"),
        "run_observation_pipeline": lambda **kwargs: calls.append("pipeline"),
    }
    real_import = builtins.__import__

    def fail_observation_import(name, *args, **kwargs):
        if name == "jspace_llama.observation_pipeline":
            raise ImportError("simulated fresh import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_observation_import)

    with pytest.raises(ImportError, match="simulated fresh import failure"):
        exec(code_cells[0].source, namespace)

    assert namespace["observation_imports_ready"] is False
    for cell in code_cells[1:]:
        with pytest.raises(RuntimeError, match="imports did not complete"):
            exec(cell.source, namespace)
    assert calls == []
