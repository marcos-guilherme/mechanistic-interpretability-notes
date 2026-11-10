import json
from pathlib import Path
import subprocess


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
    assert "N_DEV = 64" in source
    assert (
        "apply_ln_to_stack" in source
        or "ln_final.hook_scale" in source
    )
    assert "was_use_attn_result" in source
    assert "results/paper/06_circuit_discovery.json" in source.replace("\\", "/")


def test_notebook_07_compares_historical_and_modern_path_patching() -> None:
    source = _notebook_source("07_path_patching.ipynb")

    assert "paper_path_patch" in source
    assert "get_act_patch_direct_path" in source
    assert "N_PATH = 64" in source
    assert "published_path_grid_edges" in source
    assert "published_path_grid" in source
    assert "freeze_mlps=False" in source
    assert "freeze_mlps=True" in source
    assert "allow_internal_interactions" in source
    assert "multi_" in source
    assert "outros leitores" in source
    assert "results/paper/07_path_patching.json" in source.replace("\\", "/")


def test_notebook_06_artifact_separates_end_dla_from_local_projections() -> None:
    artifact = json.loads(
        (
            ROOT
            / "results"
            / "paper"
            / "06_circuit_discovery.json"
        ).read_text(encoding="utf-8")
    )

    dla_end = artifact["direct_logit_attribution_end"]
    projections = artifact["ln_final_scaled_logit_projection"]
    assert artifact["config"]["n_examples"] >= 64
    assert artifact["classification"] == "diagnostic-validation"
    assert len(dla_end) == len(dla_end[0]) == 12
    assert len(projections) == len(projections[0]) == 12
    assert len(projections[0][0]) == 5
    assert "Only END is direct attribution" in artifact["projection_note"]
    _assert_portable_traceability(
        artifact,
        {
            "patching": "src/ioi_replication/paper/patching.py",
            "circuit": "src/ioi_replication/paper/circuit.py",
            "runtime": "src/ioi_replication/runtime.py",
            "notebook": "notebooks/06_circuit_discovery.ipynb",
        },
    )


def test_notebook_07_artifact_keeps_freeze_modes_and_methods_separate() -> None:
    artifact = json.loads(
        (
            ROOT
            / "results"
            / "paper"
            / "07_path_patching.json"
        ).read_text(encoding="utf-8")
    )

    historical = artifact["historical_four_pass_semantic"]
    assert artifact["config"]["n_examples"] >= 64
    assert artifact["classification"] == "path-grid"
    assert set(historical) == {"freeze_mlps_false", "freeze_mlps_true"}
    assert set(historical["freeze_mlps_false"]) == set(
        historical["freeze_mlps_true"]
    )
    assert any(name.startswith("multi_") for name in historical["freeze_mlps_false"])
    assert "transformerlens_direct_path_all_positions" in artifact
    grid = artifact["published_path_grid"]
    assert grid["n_edges"] == 88
    assert len(grid["rows"]) == 88
    assert len(grid["sender_labels"]) == len(grid["effect_matrix"])
    assert all(
        len(row) == len(grid["receiver_labels"])
        for row in grid["effect_matrix"]
    )
    _assert_portable_traceability(
        artifact,
        {
            "patching": "src/ioi_replication/paper/patching.py",
            "circuit": "src/ioi_replication/paper/circuit.py",
            "runtime": "src/ioi_replication/runtime.py",
            "notebook": "notebooks/07_path_patching.ipynb",
        },
    )


def _assert_portable_traceability(
    artifact: dict,
    source_paths: dict[str, str],
) -> None:
    traceability = artifact["traceability"]
    assert traceability["hash_method"] == (
        "git rev-parse <source_commit>:<path> (normalized Git blob IDs)"
    )
    assert set(traceability["source_blobs"]) == set(source_paths)
    for name, relative_path in source_paths.items():
        expected = subprocess.check_output(
            [
                "git",
                "rev-parse",
                f"{traceability['source_commit']}:{relative_path}",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        assert traceability["source_blobs"][name] == expected
