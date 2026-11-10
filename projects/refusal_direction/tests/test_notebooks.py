import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NAMES = (
    "01_dataset_and_baseline.ipynb",
    "02_extract_candidate_directions.ipynb",
    "03_causal_interventions.ipynb",
    "04_evaluation_and_validation.ipynb",
)


def test_notebooks_are_thin_and_explanatory() -> None:
    for name in NAMES:
        notebook = json.loads(
            (ROOT / "notebooks" / name).read_text(encoding="utf-8")
        )
        code = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        assert code
        assert all(
            any(line.lstrip().startswith("#") for line in cell["source"])
            for cell in code
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell["source"]
        )
        assert "refusal_direction_replication" in source
        assert "run_gemma_pipeline" not in source
