#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = SCRIPT_DIR / "out_pt"
DEFAULT_FIGURE_DIR = DEFAULT_BASE_DIR / "Figure"

FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name() if FONT_PATH.exists() else "Times New Roman"
plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)

MODELS = [
    ("llama31", "llama-3.1-8b-instruct", "(a) LLaMA 3.1"),
    ("qwen25", "qwen2.5-7b-instruct", "(b) Qwen 2.5"),
]
MODEL_DISPLAY_NAMES = {
    "aya-23-8b": "Aya 23 8B",
    "internlm2_5-7b-chat": "InternLM 2.5 7B",
    "olmo-2-1124-7b-instruct": "OLMo 2 7B",
    "falcon3-7b-instruct": "Falcon3 7B",
    "gemma-2-9b-it": "Gemma 2 9B",
    "granite-3.1-8b-instruct": "Granite 3.1 8B",
    "llama-2-7b-chat-hf": "LLaMA 2 7B",
    "llama-2-13b-chat-hf": "LLaMA 2 13B",
    "llama-2-70b-chat-hf": "LLaMA 2 70B",
    "llama-3.1-8b-instruct": "LLaMA 3.1 8B",
    "llama-3.1-70b-instruct": "LLaMA 3.1 70B",
    "llama-3.1-405b-instruct": "LLaMA 3.1 405B",
    "mistral-7b-instruct-v0.3": "Mistral 7B",
    "qwen2.5-7b-instruct": "Qwen 2.5 7B",
    "qwen3.5-9b": "Qwen 3.5 9B",
    "qwen2.5-14b-instruct": "Qwen 2.5 14B",
    "qwen2.5-32b-instruct": "Qwen 2.5 32B",
    "qwen2.5-72b-instruct": "Qwen 2.5 72B",
    "yi-1.5-9b-chat": "Yi 1.5 9B",
}
DATASETS = [
    ("advbench", "AdvBench", "#c43b3b"),
    ("alpaca", "Alpaca", "#2f7d4f"),
    ("misrepresentation", "Misrepresentation", "#3b66c4"),
    ("authority_endorsement", "Authority endorsement", "#8b5fbf"),
    ("expert_endorsement", "Expert endorsement", "#d28c2d"),
]
FIG_CONFIG = {
    "fig7": {
        "measure": "harmfulness_tinst",
        "pattern": "clean-tinst-harmfulness-projection-*.json",
        "ylabel": "Harmfulness Projection",
        "output": "Fig7.png",
    },
    "fig8": {
        "measure": "refusal_tpost",
        "pattern": "clean-tpost-refusal-projection-*.json",
        "ylabel": "Refusal Projection",
        "output": "Fig8.png",
    },
}

TICK_FS = 16
AXIS_FS = 22
TITLE_FS = 28
LEGEND_FS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper Fig7/Fig8 from clean projection JSON outputs.")
    parser.add_argument("--fig", choices=["fig7", "fig8", "all"], default="all")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--llama-fig7", default=None)
    parser.add_argument("--qwen-fig7", default=None)
    parser.add_argument("--llama-fig8", default=None)
    parser.add_argument("--qwen-fig8", default=None)
    parser.add_argument("--model", default=None, help="Render a single model by output directory name.")
    parser.add_argument("--fig7-input", default=None, help="Explicit single-model Fig7 JSON.")
    parser.add_argument("--fig8-input", default=None, help="Explicit single-model Fig8 JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_clean_json(base_dir: Path, model_dir: str, measure: str, pattern: str) -> Path:
    lowered = model_dir.lower()
    if "llama" in lowered:
        model_key = "llama31"
    elif "qwen" in lowered:
        model_key = "qwen25"
    elif "aya" in lowered:
        model_key = "aya23"
    elif "internlm" in lowered:
        model_key = "internlm25"
    elif "olmo" in lowered:
        model_key = "olmo2"
    elif "gemma" in lowered:
        model_key = "gemma"
    elif "granite" in lowered:
        model_key = "granite"
    elif "falcon" in lowered:
        model_key = "falcon3"
    elif "mistral" in lowered:
        model_key = "mistral"
    elif "yi" in lowered:
        model_key = "yi"
    else:
        model_key = model_dir
    search_dirs = [
        base_dir / measure / "clean",
        base_dir / model_dir / "clean_projection" / measure / "clean",
        base_dir / model_key / measure / "clean",
    ]
    candidates: list[Path] = []
    for search_dir in search_dirs:
        candidates.extend(search_dir.glob(pattern))
    candidates = [
        path for path in candidates
        if not path.name.endswith("-layerwise-projection.json")
    ]
    candidates = [path for path in candidates if "injection" not in path.name]
    candidates.sort(key=lambda path: path.stat().st_mtime)
    if not candidates:
        tried = "\n".join(str(path) for path in search_dirs)
        raise FileNotFoundError(f"No clean projection JSON found matching {pattern}. Tried:\n{tried}")
    return candidates[-1]


def explicit_or_latest(args: argparse.Namespace, fig_name: str, model_key: str, base_dir: Path) -> Path:
    config = FIG_CONFIG[fig_name]
    attr = f"{'llama' if model_key == 'llama31' else 'qwen'}_{fig_name}".replace("_", "-")
    explicit = getattr(args, attr.replace("-", "_"), None)
    if explicit:
        return Path(explicit)
    model_dir = "llama-3.1-8b-instruct" if model_key == "llama31" else "qwen2.5-7b-instruct"
    return latest_clean_json(base_dir, model_dir, config["measure"], config["pattern"])


def single_explicit_or_latest(args: argparse.Namespace, fig_name: str, base_dir: Path) -> Path:
    explicit = getattr(args, f"{fig_name}_input", None)
    if explicit:
        return Path(explicit)
    config = FIG_CONFIG[fig_name]
    return latest_clean_json(base_dir, args.model, config["measure"], config["pattern"])


def model_label(model_dir_name: str, panel_index: int = 0) -> str:
    letter = chr(ord("a") + panel_index)
    display = MODEL_DISPLAY_NAMES.get(model_dir_name, model_dir_name)
    return f"({letter}) {display}"


def layer_ticks(size: int) -> list[int]:
    step = 4
    ticks = list(range(0, size, step))
    last = size - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


def summaries_for(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    summaries = payload.get("summaries")
    if not summaries:
        raise KeyError(f"{path} does not contain summaries")
    return summaries


def figure_limits(summaries: dict[str, dict[str, Any]]) -> tuple[float, float]:
    values = []
    for dataset_key, _, _ in DATASETS:
        if dataset_key not in summaries:
            continue
        means = np.array(summaries[dataset_key]["all_layer_mean"], dtype=np.float64)
        stds = np.array(summaries[dataset_key]["all_layer_std"], dtype=np.float64)
        values.extend((means - stds)[np.isfinite(means - stds)])
        values.extend((means + stds)[np.isfinite(means + stds)])
    if not values:
        return -1.0, 1.0
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    margin = max((ymax - ymin) * 0.08, 0.05)
    return ymin - margin, ymax + margin


def render(
    fig_name: str,
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]] = MODELS,
) -> None:
    config = FIG_CONFIG[fig_name]
    summaries_by_model = {model_key: summaries_for(path) for model_key, path in paths.items()}

    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 5.4), sharey=False, squeeze=False)
    axes = axes[0]

    for ax, (model_key, _, row_label) in zip(axes, models):
        summaries = summaries_by_model[model_key]
        ymin, ymax = figure_limits(summaries)
        max_layers = 0
        for dataset_key, dataset_label, color in DATASETS:
            if dataset_key not in summaries:
                continue
            means = np.array(summaries[dataset_key]["all_layer_mean"], dtype=np.float64)
            stds = np.array(summaries[dataset_key]["all_layer_std"], dtype=np.float64)
            layers = np.arange(len(means))
            max_layers = max(max_layers, len(means))
            ax.plot(layers, means, label=dataset_label, color=color, linewidth=2.4)
            ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.13, linewidth=0)

        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.75)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.5, max_layers - 0.5)
        ax.set_xticks(layer_ticks(max_layers))
        ax.set_xlabel("Layer", fontsize=AXIS_FS, labelpad=10)
        ax.text(
            0.5,
            -0.27,
            row_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=TITLE_FS,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=TICK_FS, length=4.0, width=1.0, pad=3.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)

    axes[0].set_ylabel(config["ylabel"], fontsize=AXIS_FS, labelpad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
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
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.84, wspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")
    for model_key, path in paths.items():
        print(f"  {model_key}: {path}")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    fig_names = ["fig7", "fig8"] if args.fig == "all" else [args.fig]
    for fig_name in fig_names:
        if args.model:
            models = [("model", args.model, model_label(args.model))]
            paths = {"model": single_explicit_or_latest(args, fig_name, base_dir)}
        else:
            models = MODELS
            paths = {
                model_key: explicit_or_latest(args, fig_name, model_key, base_dir)
                for model_key, _, _ in MODELS
            }
        render(fig_name, paths, output_dir / FIG_CONFIG[fig_name]["output"], args.dpi, models=models)


if __name__ == "__main__":
    main()
