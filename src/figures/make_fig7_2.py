#!/usr/bin/env python3
"""Combine Fig7 (Harmfulness Projection) and Fig8 (Refusal Projection) into a single 2-row figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

SCRIPT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = SCRIPT_DIR / "out_pt" / "Figure"

FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name() if FONT_PATH.exists() else "Times New Roman"
plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

DATASETS = [
    ("advbench",             "AdvBench",              "#c43b3b"),
    ("alpaca",               "Alpaca",                "#2f7d4f"),
    ("misrepresentation",    "Misrepresentation",     "#3b66c4"),
    ("authority_endorsement","Authority endorsement",  "#8b5fbf"),
    ("expert_endorsement",   "Expert endorsement",    "#d28c2d"),
]

MODELS = [
    ("llama", "llama-3.1-8b-instruct",  "LLaMA 3.1 8B"),
    ("qwen",  "qwen2.5-7b-instruct",    "Qwen 2.5 7B"),
]

MODEL_DIRS = {
    "llama": "llama-3.1-8b-instruct",
    "qwen": "qwen2.5-7b-instruct",
}


def latest_clean_projection(model_key: str, measure: str) -> Path:
    model_dir = SCRIPT_DIR / "out_pt" / MODEL_DIRS[model_key]
    if measure == "harmfulness_tinst":
        pattern = "clean-tinst-harmfulness-projection-*.json"
    elif measure == "refusal_tpost":
        pattern = "clean-tpost-refusal-projection-*.json"
    else:
        raise ValueError(measure)
    candidates = [
        p for p in model_dir.rglob(pattern)
        if f"/{measure}/clean/" in str(p)
        and not p.name.endswith("-metadata.json")
    ]
    if not candidates:
        suite_alias = "llama31" if model_key == "llama" else "qwen25"
        suite_root = SCRIPT_DIR / "out_pt" / "clean_projection_suite"
        candidates = [
            p for p in suite_root.rglob(pattern)
            if f"/models/{suite_alias}/{measure}/clean/" in str(p)
        ]
    if not candidates:
        raise FileNotFoundError(f"No clean projection JSON for {model_key} {measure}")
    resolved = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[resolve] {model_key} {measure} -> {resolved}", flush=True)
    return resolved


DATA_FILES = {
    "harm": {
        "llama": latest_clean_projection("llama", "harmfulness_tinst"),
        "qwen":  latest_clean_projection("qwen", "harmfulness_tinst"),
    },
    "refusal": {
        "llama": latest_clean_projection("llama", "refusal_tpost"),
        "qwen":  latest_clean_projection("qwen", "refusal_tpost"),
    },
}

TICK_FS  = 20
AXIS_FS  = 32
TITLE_FS = 35
LABEL_FS = 24
LEGEND_FS = 20


def load_summaries(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["summaries"]


def figure_limits(summaries: dict) -> tuple[float, float]:
    values = []
    for key, _, _ in DATASETS:
        if key not in summaries:
            continue
        means = np.array(summaries[key]["all_layer_mean"], dtype=np.float64)
        stds  = np.array(summaries[key]["all_layer_std"],  dtype=np.float64)
        values.extend((means - stds)[np.isfinite(means - stds)])
        values.extend((means + stds)[np.isfinite(means + stds)])
    if not values:
        return -1.0, 1.0
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max((hi - lo) * 0.08, 0.05)
    return lo - margin, hi + margin


def layer_ticks(n: int) -> list[int]:
    step = 4
    ticks = list(range(0, n, step))
    last = n - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


def draw_panel(ax, summaries: dict):
    ymin, ymax = figure_limits(summaries)
    max_layers = 0
    for key, label, color in DATASETS:
        if key not in summaries:
            continue
        means = np.array(summaries[key]["all_layer_mean"], dtype=np.float64)
        stds  = np.array(summaries[key]["all_layer_std"],  dtype=np.float64)
        layers = np.arange(len(means))
        max_layers = max(max_layers, len(means))
        ax.plot(layers, means, label=label, color=color, linewidth=2.4)
        ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.13, linewidth=0)

    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.75)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, max_layers - 0.5)
    ax.set_xticks(layer_ticks(max_layers))
    ax.tick_params(axis="both", labelsize=TICK_FS, length=4.0, width=1.0, pad=3.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)


def main():
    n_models = len(MODELS)
    # Width: 6.6 per model column, extra 1.5 for row labels on the left
    fig_w = 6.6 * n_models + 1.5
    fig_h = 5.4 * 2 + 0.8   # two rows + space for legend + model names

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(fig_w, fig_h),
        sharey=False,
        squeeze=False,
    )

    row_configs = [
        ("harm",    "Harmfulness Projection"),
        ("refusal", "Refusal Projection"),
    ]

    for row_idx, (measure_key, row_label) in enumerate(row_configs):
        for col_idx, (model_key, model_dir, model_name) in enumerate(MODELS):
            ax = axes[row_idx][col_idx]
            summaries = load_summaries(DATA_FILES[measure_key][model_key])
            draw_panel(ax, summaries)

            # x-axis label only on bottom row
            if row_idx == len(row_configs) - 1:
                ax.set_xlabel("Layer", fontsize=AXIS_FS, labelpad=10)

        # row label as y-axis label on the leftmost panel
        axes[row_idx][0].set_ylabel(row_label, fontsize=AXIS_FS, labelpad=12)

    # model names at the very bottom, below the bottom row
    letter_labels = [chr(ord("a") + i) for i in range(n_models)]
    for col_idx, (_, _, model_name) in enumerate(MODELS):
        ax = axes[-1][col_idx]
        letter = letter_labels[col_idx]
        ax.text(
            0.5, -0.30,
            f"({letter}) {model_name}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=TITLE_FS, fontweight="bold",
        )

    # single shared legend at the top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        fontsize=LEGEND_FS,
        frameon=True,
        fancybox=False,
        edgecolor="lightgray",
        facecolor="white",
        framealpha=1.0,
        handlelength=2.2,
        columnspacing=1.2,
        borderpad=0.45,
    )

    fig.subplots_adjust(
        left=0.09, right=0.995,
        bottom=0.14, top=0.88,
        hspace=0.42, wspace=0.14,
    )

    out_path = OUT_DIR / "Fig9.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
