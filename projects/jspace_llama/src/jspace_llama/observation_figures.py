"""Deterministic static figures for compact observational results."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


METHODS = ("jlens", "logit_lens")
SCOPES = ("band", "all_layers")
KS = (1, 5, 10)
VOCABULARY_SIZE = 128_256
FIGURE_STEMS = (
    "observation_pass_at_k",
    "observation_rank_heatmap",
    "observation_panel",
)


def _method_key(method: str, scope: str) -> str:
    return f"{method}_{scope}"


def build_figure_matrices(
    result: dict[str, object],
) -> dict[str, object]:
    """Project compact JSON into stable, serialization-friendly matrices."""
    metrics = result["metrics"]
    pass_at_k = {
        _method_key(method, scope): [
            float(metrics[method][scope][f"pass@{k}"]) for k in KS
        ]
        for method in METHODS
        for scope in SCOPES
    }
    layers = [int(layer) for layer in result["layers"]["all_layers"]]
    heatmap: list[list[int]] = []
    names: list[str] = []
    for record in result["per_item_best_ranks"]:
        names.append(str(record["name"]))
        row: list[int] = []
        for layer in layers:
            ranks = [
                int(entry["rank"])
                for intermediate in record["intermediates"]
                for entry in intermediate["jlens"]["ranks"]
                if int(entry["layer"]) == layer
            ]
            row.append(min(ranks, default=VOCABULARY_SIZE))
        heatmap.append(row)
    return {
        "pass_at_k": pass_at_k,
        "rank_heatmap": {
            "values": heatmap,
            "shape": [len(heatmap), len(layers)],
            "item_names": names,
            "layers": layers,
        },
        "panel": result["panel"],
    }


def _pass_at_k_figure(matrices: dict[str, object]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=120)
    width = 0.18
    x = list(range(len(KS)))
    styles = (
        ("jlens_band", "J-lens · band", "#1f77b4"),
        ("logit_lens_band", "Logit lens · band", "#ff7f0e"),
        ("jlens_all_layers", "J-lens · all layers", "#8fbce6"),
        ("logit_lens_all_layers", "Logit lens · all layers", "#ffc078"),
    )
    for offset, (key, label, color) in enumerate(styles):
        positions = [value + (offset - 1.5) * width for value in x]
        axis.bar(
            positions,
            matrices["pass_at_k"][key],
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, [f"pass@{k}" for k in KS])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of intermediates")
    axis.set_title("Direct readout recovery")
    axis.legend(loc="upper left", fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def _rank_heatmap_figure(matrices: dict[str, object]) -> plt.Figure:
    heatmap = matrices["rank_heatmap"]
    values = [
        [max(1, min(VOCABULARY_SIZE, int(value))) for value in row]
        for row in heatmap["values"]
    ]
    figure, axis = plt.subplots(figsize=(9, 7), dpi=120)
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis_r",
        norm=LogNorm(vmin=1, vmax=VOCABULARY_SIZE),
    )
    axis.set_xticks(
        range(len(heatmap["layers"])),
        heatmap["layers"],
        rotation=90,
        fontsize=7,
    )
    axis.set_xlabel("Layer")
    axis.set_ylabel("Dataset item (fixed source order)")
    axis.set_title("J-lens intermediate rank by item and layer")
    figure.colorbar(image, ax=axis, label="One-based rank (log scale)")
    figure.tight_layout()
    return figure


def _first_tokens(slots: object, *, limit: int = 3) -> str:
    if not isinstance(slots, list):
        return "—"
    tokens: list[str] = []
    for slot in slots:
        for token in slot.get("tokens", []):
            text = str(token.get("token", "")).replace("\n", "\\n")
            if text and text not in tokens:
                tokens.append(text)
            if len(tokens) == limit:
                return ", ".join(tokens)
    return ", ".join(tokens) if tokens else "—"


def _panel_rows(matrices: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in matrices["panel"]:
        intermediates = item["intermediates"]
        rows.append(
            [
                str(item["name"]),
                "\n".join(
                    str(intermediate["text"])
                    for intermediate in intermediates
                ),
                "\n".join(
                    (
                        f"L{intermediate['jlens']['best_layer']} / "
                        f"{intermediate['jlens']['best_rank']}"
                    )
                    for intermediate in intermediates
                ),
                _first_tokens(item["top_tokens"]["jlens"]),
                _first_tokens(item["top_tokens"]["logit_lens"]),
                _first_tokens(item["pursuit"]),
            ]
        )
    return rows


def _panel_figure(matrices: dict[str, object]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13, 9), dpi=120)
    axis.axis("off")
    table = axis.table(
        cellText=_panel_rows(matrices),
        colLabels=(
            "Item",
            "Expected intermediate",
            "Best J-lens\nlayer / rank",
            "J-lens direct",
            "Logit-lens direct",
            "Pursuit",
        ),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.16, 0.17, 0.11, 0.19, 0.19, 0.18),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.2)
    axis.set_title(
        "Registered qualitative panel · direct and pursuit outputs remain separate",
        pad=18,
    )
    figure.tight_layout()
    return figure


def _strip_and_validate_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"<!DOCTYPE.*?>\s*",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    root = ET.parse(path).getroot()
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("Matplotlib output root is not svg")


def write_observation_figures(
    result: dict[str, object],
    output_dir: str | Path,
) -> list[Path]:
    """Generate all three figure families before replacing any destination."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "jspace-observation"
    matrices = build_figure_matrices(result)
    builders = (
        _pass_at_k_figure,
        _rank_heatmap_figure,
        _panel_figure,
    )
    temporary: list[tuple[Path, Path]] = []
    outputs: list[Path] = []
    try:
        for stem, builder in zip(FIGURE_STEMS, builders, strict=True):
            figure = builder(matrices)
            try:
                for suffix in (".png", ".svg"):
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".{stem}.",
                        suffix=".tmp",
                        dir=destination,
                    )
                    os.close(descriptor)
                    temp = Path(name)
                    target = destination / f"{stem}{suffix}"
                    figure.savefig(
                        temp,
                        format=suffix.removeprefix("."),
                        dpi=120,
                        metadata={"Date": None},
                    )
                    if suffix == ".svg":
                        _strip_and_validate_svg(temp)
                    temporary.append((temp, target))
                    outputs.append(target)
            finally:
                plt.close(figure)
        for temp, target in temporary:
            os.replace(temp, target)
        return outputs
    except BaseException:
        for temp, _target in temporary:
            temp.unlink(missing_ok=True)
        raise
