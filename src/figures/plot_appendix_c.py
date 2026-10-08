#!/usr/bin/env python3
"""
Plot steering figures for LLaMA 3.1 8B and Qwen 2.5 7B.

Changes:
- Exclude alpha = 0
- Plot alpha = 1, 2, 3, 4 only
- Use 2 x 2 subplot layout
- Remove bridge shading
- Remove Bridge from legend
- Remove peak annotations such as L3, L15
- Put model name at the top
- Put legend below model name
- Put alpha labels inside each subplot
- Show Refusal Rate (%) only on the left column
- Keep y-axis tick labels visible on the right column
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# ── Font ──────────────────────────────────────────────────────────────────────
FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")

if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "Times New Roman"

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parents[1]
OUT_PT = SCRIPT_DIR / "out_pt"

LLAMA_AGG = (
    OUT_PT / "llama-3.1-8b-instruct/steering/"
    "steering-hidden-harmfulness_refusal-L0-31-alpha_0_to_4-20260515-033345-wildguard-judge/"
    "aggregate_result.json"
)

QWEN_AGG = (
    OUT_PT / "qwen2.5-7b-instruct/steering/"
    "steering-hidden-harmfulness_refusal-L0-27-alpha_0_to_4-20260515-030918-wildguard-judge/"
    "aggregate_result.json"
)


# ── Visual constants ───────────────────────────────────────────────────────────
STEER_COLORS = {
    "harmfulness": "#d73027",
    "refusal": "#4575b4",
}

STEER_LABELS = {
    "harmfulness": "Harmfulness Steering",
    "refusal": "Refusal Steering",
}

TICK_FS = 15
AXIS_FS = 18
LEG_FS = 14
TITLE_FS = 26


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_figure(fig, output_path: Path, *, dpi: int, **kwargs) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, **kwargs)
    fig.savefig(output_path.with_suffix(".pdf"), **kwargs)


def _layer_sort_key(name: str):
    nums = []

    for part in name.replace("L", "").split("_"):
        try:
            nums.append(int(part))
        except ValueError:
            pass

    return (nums[0] if nums else 10 ** 9, nums, name)


def _sorted_layer_names(names) -> list[str]:
    return sorted(names, key=_layer_sort_key)


def _load_steering_data(
    agg_path: Path,
) -> tuple[float, list[float], list[str], dict, dict]:
    """
    Parse aggregate_result.json.

    Returns:
        baseline_refusal:
            Baseline refusal percentage from alpha = 0.
            This value is loaded but not plotted.

        alphas:
            Sorted alpha values.

        layers:
            Sorted layer name list.

        harm_idx:
            Dictionary indexed by (layer_name, alpha) for harmfulness steering rows.

        ref_idx:
            Dictionary indexed by (layer_name, alpha) for refusal steering rows.
    """
    agg = _load(agg_path)
    rows = agg["by_target_layer_alpha"]

    baseline_rows = [r for r in rows if r["target"] == "baseline"]
    baseline_refusal = baseline_rows[0]["refusal"]["percentage"] if baseline_rows else 0.0

    harm_rows = [r for r in rows if r["target"] == "harmfulness"]
    ref_rows = [r for r in rows if r["target"] == "refusal"]

    alphas = sorted({r["alpha"] for r in harm_rows})
    layers = _sorted_layer_names({r["layer_group_name"] for r in harm_rows})

    harm_idx = {
        (r["layer_group_name"], r["alpha"]): r
        for r in harm_rows
    }

    ref_idx = {
        (r["layer_group_name"], r["alpha"]): r
        for r in ref_rows
    }

    return baseline_refusal, alphas, layers, harm_idx, ref_idx


def _draw_alpha_subplot(
    ax,
    harm_idx: dict,
    ref_idx: dict,
    layers: list[str],
    alpha: float,
    y_max: float,
    show_xlabel: bool,
) -> None:
    """
    Draw one alpha subplot.

    Only harmfulness and refusal steering curves are plotted.
    Bridge shading and peak annotations are intentionally removed.
    """
    x = np.arange(len(layers))

    _markers = {"harmfulness": "o", "refusal": "^"}
    for target, idx in [("harmfulness", harm_idx), ("refusal", ref_idx)]:
        y = np.array([
            idx[(lg, alpha)]["refusal"]["percentage"]
            if (lg, alpha) in idx
            else np.nan
            for lg in layers
        ], dtype=float)

        ax.plot(
            x,
            y,
            marker=_markers[target],
            linewidth=2.0,
            markersize=4,
            color=STEER_COLORS[target],
            zorder=3,
        )

    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(layers) - 0.5)

    step = 4
    ticks = list(range(0, len(layers), step))

    if ticks and ticks[-1] != len(layers) - 1:
        if len(layers) - 1 - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(len(layers) - 1)

    ax.set_xticks(ticks)

    if show_xlabel:
        ax.set_xticklabels(
            [layers[i].replace("L", "") for i in ticks],
            rotation=0,
            ha="center",
            fontsize=TICK_FS,
        )
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def render_single_model(
    agg_path: Path,
    model_label: str,
    output_path: Path,
    dpi: int = 160,
) -> None:
    baseline_refusal, alphas, layers, harm_idx, ref_idx = _load_steering_data(agg_path)

    # Exclude alpha = 0
    all_alphas = [a for a in alphas if float(a) != 0.0]
    n_alphas = len(all_alphas)

    # alpha = 1, 2, 3, 4
    nrows, ncols = 2, 2

    # Compute y-axis range from all non-baseline values
    all_vals = [
        idx[(lg, a)]["refusal"]["percentage"]
        for a in all_alphas
        for idx in [harm_idx, ref_idx]
        for lg in layers
        if (lg, a) in idx
    ]

    all_vals = [v for v in all_vals if not math.isnan(v)]

    if all_vals:
        top = max(all_vals)

        if top <= 30:
            y_max = float(math.ceil(top / 5) * 5)
        else:
            y_max = float(math.ceil(top / 10) * 10)

        y_max = max(y_max, 10.0)
    else:
        y_max = 100.0

    fig_w = 11.5
    fig_h = 7.8

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ai, alpha in enumerate(all_alphas):
        ax = axes[ai]

        row = ai // ncols
        col = ai % ncols

        show_xlabel = row == nrows - 1

        _draw_alpha_subplot(
            ax=ax,
            harm_idx=harm_idx,
            ref_idx=ref_idx,
            layers=layers,
            alpha=alpha,
            y_max=y_max,
            show_xlabel=show_xlabel,
        )

        # Show y-axis label only on the left column
        if col == 0:
            ax.set_ylabel("Refusal Rate (%)", fontsize=AXIS_FS)
        else:
            ax.set_ylabel("")

        # Keep y-axis tick labels visible on both columns
        ax.tick_params(axis="y", labelleft=True)
        ax.yaxis.set_tick_params(labelleft=True)

        # Alpha label inside each subplot
        if float(alpha).is_integer():
            alpha_label = f"α = {int(alpha)}"
        else:
            alpha_label = f"α = {alpha}"

        ax.text(
            0.03,
            0.92,
            alpha_label,
            transform=ax.transAxes,
            fontsize=AXIS_FS,
            fontweight="bold",
            ha="left",
            va="top",
        )

        # Show x-axis label only on the bottom row
        if show_xlabel:
            ax.set_xlabel("Steered Layer", fontsize=AXIS_FS)
        else:
            ax.set_xlabel("")

    # Remove unused axes if alpha count is smaller than 4
    for j in range(n_alphas, len(axes)):
        fig.delaxes(axes[j])

    # Legend without Bridge
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=STEER_COLORS["harmfulness"],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=STEER_LABELS["harmfulness"],
        ),
        plt.Line2D(
            [0],
            [0],
            color=STEER_COLORS["refusal"],
            linewidth=2.0,
            marker="^",
            markersize=5,
            label=STEER_LABELS["refusal"],
        ),
    ]

    # Model title at the top
    fig.suptitle(
        model_label,
        fontsize=TITLE_FS,
        fontweight="bold",
        y=0.985,
    )

    # Legend below the title, above subplots
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        fontsize=LEG_FS,
        frameon=True,
        fancybox=False,
        edgecolor="lightgray",
        facecolor="white",
        framealpha=1.0,
        handlelength=2.0,
        columnspacing=1.8,
        borderpad=0.45,
    )

    # Layout adjustment
    fig.subplots_adjust(
        top=0.81,
        bottom=0.10,
        left=0.08,
        right=0.98,
        wspace=0.16,
        hspace=0.32,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Appendix C steering figures.")
    parser.add_argument("--output-dir", default=str(OUT_PT / "Figure"))
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    render_single_model(
        agg_path=LLAMA_AGG,
        model_label="LLaMA 3.1 8B",
        output_path=out_dir / "Fig3_alpha1to4_llama31_2x2.png",
        dpi=args.dpi,
    )

    render_single_model(
        agg_path=QWEN_AGG,
        model_label="Qwen 2.5 7B",
        output_path=out_dir / "Fig3_alpha1to4_qwen25_2x2.png",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()