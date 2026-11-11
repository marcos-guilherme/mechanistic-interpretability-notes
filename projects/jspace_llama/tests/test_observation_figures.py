from __future__ import annotations

import copy
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

from jspace_llama.observation_data import PANEL_NAMES
from jspace_llama.jsonio import atomic_write_json


def compact_result() -> dict[str, object]:
    layers = list(range(17))
    records = []
    panel = []
    for index in range(93):
        name = PANEL_NAMES[index] if index < len(PANEL_NAMES) else f"item-{index}"
        records.append(
            {
                "name": name,
                "intermediates": [
                    {
                        "text": f"intermediate-{index}",
                        "single_token": True,
                        "jlens": {
                            "ranks": [
                                {"layer": layer, "rank": index + layer + 1}
                                for layer in layers
                            ],
                            "best_rank": index + 1,
                            "best_layer": 12,
                            "all_layers_best_rank": index + 1,
                            "all_layers_best_layer": 0,
                        },
                        "logit_lens": {
                            "ranks": [
                                {"layer": layer, "rank": index + layer + 2}
                                for layer in layers
                            ],
                            "best_rank": index + 2,
                            "best_layer": 12,
                            "all_layers_best_rank": index + 2,
                            "all_layers_best_layer": 0,
                        },
                    }
                ],
            }
        )
        if index < len(PANEL_NAMES):
            panel.append(
                {
                    "name": name,
                    "prompt": f"prompt {name} ",
                    "target": f"target-{name}",
                    "intermediates": records[-1]["intermediates"],
                    "top_tokens": {
                        method: [
                            {
                                "layer": 12,
                                "tokens": [
                                    {
                                        "token_id": index,
                                        "token": f"{method}-{index}",
                                        "rank": 1,
                                    }
                                ],
                            }
                        ]
                        for method in ("jlens", "logit_lens")
                    },
                    "pursuit": [
                        {
                            "layer": 12,
                            "tokens": [
                                {
                                    "token_id": index + 100,
                                    "token": f"pursuit-{index}",
                                    "rank": 1,
                                    "coefficient": 0.75,
                                }
                            ],
                            "variance_explained": 0.5,
                        }
                    ],
                }
            )
    metrics = {
        "jlens": {
            "band": {
                "n_items": 93,
                "n_intermediates": 93,
                "pass@1": 0.2,
                "pass@5": 0.4,
                "pass@10": 0.6,
            },
            "all_layers": {
                "n_items": 93,
                "n_intermediates": 93,
                "pass@1": 0.3,
                "pass@5": 0.5,
                "pass@10": 0.7,
            },
        },
        "logit_lens": {
            "band": {
                "n_items": 93,
                "n_intermediates": 93,
                "pass@1": 0.1,
                "pass@5": 0.2,
                "pass@10": 0.3,
            },
            "all_layers": {
                "n_items": 93,
                "n_intermediates": 93,
                "pass@1": 0.15,
                "pass@5": 0.25,
                "pass@10": 0.35,
            },
        },
    }
    return {
        "schema_version": 1,
        "artifact": "jspace_observation",
        "run_id": "fixture",
        "status": "success",
        "reason": None,
        "metrics": metrics,
        "tokenization_summary": {
            "item_count": 93,
            "intermediate_count": 93,
            "single_token_count": 93,
            "multi_token_count": 0,
        },
        "per_item_best_ranks": records,
        "panel": panel,
        "layers": {"band": [12, 13], "all_layers": layers},
        "panel_names": list(PANEL_NAMES),
        "manifest_sha256": "0" * 64,
    }


def test_figure_matrices_are_deterministic() -> None:
    from jspace_llama.observation_figures import build_figure_matrices

    first = build_figure_matrices(compact_result())
    second = build_figure_matrices(compact_result())

    assert first == second
    assert first["pass_at_k"]["jlens_band"] == [0.2, 0.4, 0.6]
    assert first["rank_heatmap"]["shape"] == [93, 17]


def test_panel_rows_render_every_expected_intermediate_variant() -> None:
    from jspace_llama.observation_figures import (
        _panel_rows,
        build_figure_matrices,
    )

    result = compact_result()
    item = next(
        entry
        for entry in result["panel"]
        if entry["name"] == "nhop-primary-planet"
    )
    second = copy.deepcopy(item["intermediates"][0])
    second["text"] = "second expected hop"
    second["jlens"]["best_layer"] = 13
    second["jlens"]["best_rank"] = 2
    item["intermediates"].append(second)

    row = next(
        entry
        for entry in _panel_rows(build_figure_matrices(result))
        if entry[0] == "nhop-primary-planet"
    )

    assert row[1] == "intermediate-10\nsecond expected hop"
    assert row[2] == "L12 / 11\nL13 / 2"


def test_all_figure_families_write_png_and_svg(tmp_path: Path) -> None:
    from jspace_llama.observation_figures import write_observation_figures

    paths = write_observation_figures(compact_result(), tmp_path)

    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert len(paths) == 6
    assert all(path.stat().st_size > 0 for path in paths)
    assert all("<!DOCTYPE" not in path.read_text() for path in paths if path.suffix == ".svg")
    assert list(tmp_path.glob(".*.tmp")) == []


def test_figure_cli_renders_in_current_interpreter_child(
    tmp_path: Path,
) -> None:
    from jspace_llama.observation_figures import write_observation_figures

    input_path = tmp_path / "compact.json"
    output_dir = tmp_path / "figures"
    direct_dir = tmp_path / "direct"
    result = compact_result()
    atomic_write_json(input_path, result)
    direct_paths = write_observation_figures(result, direct_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jspace_llama.observation_figure_cli",
            "--input",
            str(input_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    paths = sorted(output_dir.iterdir())
    assert len(paths) == 6
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.stat().st_size > 0 for path in paths)
    assert {
        path.name: path.read_bytes() for path in paths
    } == {
        path.name: path.read_bytes() for path in direct_paths
    }


def test_figure_cli_rejects_non_strict_json_before_render(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "duplicate.json"
    output_dir = tmp_path / "figures"
    input_path.write_text(
        '{"status":"success","status":"success"}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jspace_llama.observation_figure_cli",
            "--input",
            str(input_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode != 0
    assert "duplicate JSON key" in completed.stderr
    assert not output_dir.exists()


def test_figure_cli_rejects_input_above_five_mib(tmp_path: Path) -> None:
    from jspace_llama import observation_figure_cli as cli

    input_path = tmp_path / "oversized.json"
    input_path.write_bytes(
        b'{"artifact":"jspace_observation","status":"success","padding":"'
        + b"x" * (5 * 1_048_576)
        + b'"}'
    )

    with pytest.raises(ValueError, match="too large"):
        cli._load_compact_result(input_path)


def test_figure_cli_uses_bounded_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jspace_llama import observation_figure_cli as cli

    input_path = tmp_path / "compact.json"
    atomic_write_json(input_path, compact_result())
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: pytest.fail(f"unbounded read used for {path}"),
    )

    assert cli._load_compact_result(input_path)["status"] == "success"


def test_figure_cli_rejects_file_changed_during_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jspace_llama import jsonio, observation_figure_cli as cli

    input_path = tmp_path / "compact.json"
    atomic_write_json(input_path, compact_result())
    real_fstat = jsonio.os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        actual = real_fstat(descriptor)
        if calls == 1:
            return actual
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(jsonio.os, "fstat", changed_fstat)

    with pytest.raises(ValueError, match="changed during validation"):
        cli._load_compact_result(input_path)


def test_figure_cli_rejects_symlink_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jspace_llama import jsonio, observation_figure_cli as cli

    input_path = tmp_path / "compact.json"
    atomic_write_json(input_path, compact_result())
    monkeypatch.setattr(
        jsonio.os,
        "lstat",
        lambda path: SimpleNamespace(
            st_mode=stat.S_IFLNK,
            st_size=input_path.stat().st_size,
        ),
    )

    with pytest.raises(ValueError, match="regular file"):
        cli._load_compact_result(input_path)


def test_figure_generation_failure_preserves_all_existing_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jspace_llama.observation_figures as figures

    destinations = [
        tmp_path / f"{stem}{suffix}"
        for stem in figures.FIGURE_STEMS
        for suffix in (".png", ".svg")
    ]
    for path in destinations:
        path.write_bytes(b"previous")
    monkeypatch.setattr(
        figures,
        "_panel_figure",
        lambda matrices: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        figures.write_observation_figures(compact_result(), tmp_path)

    assert all(path.read_bytes() == b"previous" for path in destinations)
    assert list(tmp_path.glob(".*.tmp")) == []
