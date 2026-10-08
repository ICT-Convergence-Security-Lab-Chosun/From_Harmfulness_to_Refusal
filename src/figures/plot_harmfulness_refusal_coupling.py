#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "Times New Roman"

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.serif": [FONT_NAME, "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Harmfulness-to-Refusal Coupling Test outputs.")
    parser.add_argument("result_json", help="Path to harmfulness_refusal_coupling.py JSON output.")
    parser.add_argument(
        "--variant",
        choices=["full", "direct", "indirect", "direct_ratio"],
        default=None,
        help="For path_patching_coupling.py outputs, select which map to plot.",
    )
    parser.add_argument("--raw", action="store_true", help="Plot raw DeltaR instead of std-normalized DeltaR.")
    parser.add_argument("--vmin", type=float, default=None, help="Manual colorbar minimum.")
    parser.add_argument("--vmax", type=float, default=None, help="Manual colorbar maximum.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def finite_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 98))
    if limit <= 0:
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    if limit <= 0:
        limit = 1.0
    return -limit, limit


def plot_heatmap(
    payload: dict,
    values: np.ndarray,
    out_path: Path,
    raw: bool,
    dpi: int,
    variant: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    metadata = payload["metadata"]
    # The experiment stores C[L, k] as row=injection layer and column=measurement layer.
    # Plot it directly so y=L is the source injection layer and x=k is the downstream readout layer.
    plot_values = values
    masked = np.ma.masked_invalid(plot_values)
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = finite_color_limits(plot_values)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax)
    if variant == "direct_ratio":
        cbar_label = "Direct / full normalized coupling"
    else:
        cbar_label = "Delta refusal projection" if raw else "Delta refusal projection / baseline std"
    cbar.set_label(cbar_label)

    ax.set_xlabel("downstream layer k where refusal projection is read at t_post")
    ax.set_ylabel("source layer L where harmfulness direction is added at t_inst")
    title_kind = "raw" if raw else "std-normalized"
    variant_label = f" {variant}" if variant else ""
    ax.set_title(
        f"Harmfulness Injection -> Downstream Refusal Readout{variant_label} ({title_kind})\n"
        f"{metadata['model']} alpha={metadata['alpha']}"
    )
    ax.set_xticks(range(plot_values.shape[1]))
    ax.set_yticks(range(plot_values.shape[0]))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_behavior(payload: dict, out_path: Path, dpi: int) -> bool:
    behavioral = payload.get("behavioral_coupling") or payload.get("behavioral_validation")
    if not behavioral:
        return False
    experiments = behavioral.get("experiments") or []
    if not experiments:
        return False

    layers = [item["layer"] for item in experiments]
    if "delta_refusal_rate" in experiments[0]:
        series = [("Full", [item["delta_refusal_rate"] for item in experiments], "o", "#d73027")]
    else:
        series = [
            ("Full", [item.get("delta_from_baseline_full", np.nan) for item in experiments], "o", "#d73027"),
            ("Indirect", [item.get("indirect_delta", np.nan) for item in experiments], "^", "#4575b4"),
        ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.85)
    for label, values, marker, color in series:
        ax.plot(layers, values, marker=marker, markersize=6.0, linewidth=2.0, color=color, label=label)

    ticks = list(range(0, max(layers) + 1, 4))
    if ticks and ticks[-1] != max(layers):
        ticks.append(max(layers))
    ax.set_xticks(ticks)
    ax.set_xlabel("Steered Layer", fontsize=13)
    ax.set_ylabel(r"$\Delta$ Refusal Rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_json)
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "path_patching_coupling" in payload:
        coupling = payload["path_patching_coupling"]
        variant = args.variant or "full"
        if variant == "direct_ratio":
            values = np.array(coupling["direct_ratio"]["normalized"], dtype=np.float64)
            suffix = "direct_ratio-normalized"
        else:
            key = "coupling_map" if args.raw else "normalized_coupling_map"
            values = np.array(coupling[variant][key], dtype=np.float64)
            suffix = f"{variant}-{'raw' if args.raw else 'normalized'}"
    else:
        key = "coupling_map" if args.raw else "normalized_coupling_map"
        internal = payload["internal_coupling"]
        if args.variant is not None:
            raise ValueError("--variant is only valid for path_patching_coupling.py outputs.")
        if "readouts" in internal and "refusal" in internal["readouts"]:
            values = np.array(internal["readouts"]["refusal"][key], dtype=np.float64)
        else:
            values = np.array(internal[key], dtype=np.float64)
        variant = None
        suffix = "raw" if args.raw else "normalized"

    output_dir = Path(args.output_dir) if args.output_dir else result_path.parent
    stem = result_path.stem
    heatmap_path = output_dir / f"{stem}-{suffix}-heatmap.png"
    plot_heatmap(
        payload,
        values,
        heatmap_path,
        args.raw,
        args.dpi,
        variant=variant,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(f"saved {heatmap_path}")

    behavior_path = output_dir / f"{stem}-behavioral-coupling.png"
    if plot_behavior(payload, behavior_path, args.dpi):
        print(f"saved {behavior_path}")


if __name__ == "__main__":
    main()
