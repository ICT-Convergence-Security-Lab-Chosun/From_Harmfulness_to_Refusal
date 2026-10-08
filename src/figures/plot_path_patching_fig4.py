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
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "Times New Roman"

plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.serif": [FONT_NAME, "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)

MODELS = [
    ("llama", "llama-3.1-8b-instruct", "(a) LLaMA 3.1"),
    ("qwen", "qwen2.5-7b-instruct", "(b) Qwen 2.5"),
]
VARIANTS = [("full", "Full"), ("indirect", "Indirect")]
MODEL_LABEL_FS = 28
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
    "qwen2.5-14b-instruct": "Qwen 2.5 14B",
    "qwen2.5-32b-instruct": "Qwen 2.5 32B",
    "qwen2.5-72b-instruct": "Qwen 2.5 72B",
    "yi-1.5-9b-chat": "Yi 1.5 9B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Fig4 and Fig4_2 from activation_patching_coupling.py outputs."
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--alpha", type=float, default=3.0, help="Alpha label used for auto-discovery patterns.")
    parser.add_argument("--fig4", default="Fig4.png")
    parser.add_argument("--fig4-2", default="Fig4_2.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--llama", default=None, help="Explicit LLaMA activation-patching JSON for Fig4.")
    parser.add_argument("--qwen", default=None, help="Explicit Qwen activation-patching JSON for Fig4.")
    parser.add_argument("--llama-behavior", default=None, help="Explicit LLaMA JSON with behavioral_validation.")
    parser.add_argument("--qwen-behavior", default=None, help="Explicit Qwen JSON with behavioral_validation.")
    parser.add_argument("--model", default=None, help="Render a single model by output directory name.")
    parser.add_argument("--input", default=None, help="Explicit single-model activation-patching JSON for Fig4.")
    parser.add_argument("--behavior-input", default=None, help="Explicit single-model JSON with behavioral_validation.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def alpha_label(alpha: float) -> str:
    text = f"{alpha:g}"
    return text.replace("-", "m").replace(".", "p")


def candidate_result_jsons(model_dir: Path, alpha_tag: str) -> list[Path]:
    paths = []
    search_specs = [
        ("activation_patching", f"activation-patching-hidden-alpha_{alpha_tag}-*.json"),
        ("path_patching", f"path-patching-hidden-alpha_{alpha_tag}-*.json"),
    ]
    for subdir, pattern in search_specs:
        for path in (model_dir / subdir).rglob(pattern):
            if "generation-for-judge" in path.name:
                continue
            paths.append(path)
    return sorted(paths, key=lambda path: path.stat().st_mtime)


def latest_result_json(base_dir: Path, model_dir_name: str, alpha_tag: str) -> Path:
    matches = candidate_result_jsons(base_dir / model_dir_name, alpha_tag)
    if not matches:
        raise FileNotFoundError(f"No activation-patching result JSON found under {base_dir / model_dir_name}")
    return matches[-1]


def latest_behavior_json(base_dir: Path, model_dir_name: str, alpha_tag: str) -> Path:
    matches = candidate_result_jsons(base_dir / model_dir_name, alpha_tag)
    for path in reversed(matches):
        payload = load_json(path)
        behavioral = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
        if behavioral and behavioral.get("experiments"):
            return path
    raise FileNotFoundError(f"No behavioral_validation result JSON found under {base_dir / model_dir_name}")


def model_label(model_dir_name: str, panel_index: int = 0) -> str:
    letter = chr(ord("a") + panel_index)
    display = MODEL_DISPLAY_NAMES.get(model_dir_name, model_dir_name)
    return f"({letter}) {display}"


def coupling_values(payload: dict[str, Any], variant: str) -> np.ndarray:
    coupling = payload.get("activation_patching_coupling") or payload["path_patching_coupling"]
    return np.array(coupling[variant]["normalized_coupling_map"], dtype=np.float64)


def symmetric_limits(values_list: list[np.ndarray]) -> tuple[float, float]:
    finite_chunks = [values[np.isfinite(values)] for values in values_list]
    finite = np.concatenate([chunk for chunk in finite_chunks if chunk.size])
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 98))
    if limit <= 0:
        limit = float(np.nanmax(np.abs(finite)))
    if limit <= 0:
        limit = 1.0
    return -limit, limit


def layer_ticks(size: int) -> list[int]:
    step = 4
    ticks = list(range(0, size, step))
    last = size - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


def mask_non_downstream(values: np.ndarray) -> np.ndarray:
    injection_layer = np.arange(values.shape[0])[:, None]
    downstream_layer = np.arange(values.shape[1])[None, :]
    masked = values.copy()
    masked[downstream_layer <= injection_layer] = np.nan
    return masked


def render_heatmap_grid(
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]] = MODELS,
) -> None:
    payloads = {key: load_json(path) for key, path in paths.items()}
    maps = {
        model_key: {variant: coupling_values(payloads[model_key], variant) for variant, _ in VARIANTS}
        for model_key, _, _ in models
    }
    row_limits = {
        model_key: symmetric_limits([maps[model_key][variant] for variant, _ in VARIANTS])
        for model_key, _, _ in models
    }

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(len(models), 2, figsize=(9.8, 4.35 * len(models) + 0.4), squeeze=False)
    row_images = {}

    for row_idx, (model_key, _, row_label) in enumerate(models):
        for col_idx, (variant, variant_label) in enumerate(VARIANTS):
            ax = axes[row_idx, col_idx]
            values = mask_non_downstream(maps[model_key][variant])
            masked = np.ma.masked_invalid(values)
            vmin, vmax = row_limits[model_key]
            image = ax.imshow(masked, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            row_images[row_idx] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])

            xticks = layer_ticks(values.shape[1])
            yticks = layer_ticks(values.shape[0])
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", labelsize=18, length=4, width=1.0, pad=3)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            if row_idx == 0:
                ax.set_title(variant_label, fontsize=22, pad=10)
            if row_idx == len(models) - 1:
                ax.set_xlabel("Downstream Layer", fontsize=24, labelpad=12)
            if col_idx == 0:
                ax.set_ylabel("Injection Layer", fontsize=24, labelpad=12)

        axes[row_idx, 0].text(
            -0.36,
            0.5,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=MODEL_LABEL_FS,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.21, right=0.90, bottom=0.10, top=0.93, wspace=0.20, hspace=0.18)
    for row_idx, image in row_images.items():
        anchor = axes[row_idx, -1].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.018, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=18, length=4, width=1.0)
        cbar.outline.set_linewidth(1.0)
        cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=24, labelpad=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def behavioral_series(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    behavioral = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
    if not behavioral or not behavioral.get("experiments"):
        raise ValueError("JSON does not contain behavioral_validation experiments.")

    experiments = sorted(behavioral["experiments"], key=lambda item: int(item["layer"]))
    layers = np.array([int(item["layer"]) for item in experiments], dtype=int)
    if "delta_refusal_rate" in experiments[0]:
        full = np.array([item.get("delta_refusal_rate", np.nan) for item in experiments], dtype=np.float64)
        indirect = np.full_like(full, np.nan)
    else:
        full = np.array([item.get("delta_from_baseline_full", np.nan) for item in experiments], dtype=np.float64)
        indirect = np.array([item.get("indirect_delta", np.nan) for item in experiments], dtype=np.float64)
    return layers, full, indirect


def render_behavior_grid(
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]] = MODELS,
) -> None:
    payloads = {key: load_json(path) for key, path in paths.items()}
    series = {model_key: behavioral_series(payloads[model_key]) for model_key, _, _ in models}

    finite = []
    for _, full, indirect in series.values():
        finite.extend(full[np.isfinite(full)])
        finite.extend(indirect[np.isfinite(indirect)])
    if finite:
        ymin = min(0.0, float(np.min(finite))) - 0.03
        ymax = max(0.0, float(np.max(finite))) + 0.03
    else:
        ymin, ymax = -0.05, 0.05

    fig, axes = plt.subplots(len(models), 1, figsize=(7.2, 3.1 * len(models)), sharex=False)
    axes = np.array(axes).reshape(len(models))

    for row_idx, (model_key, _, row_label) in enumerate(models):
        ax = axes[row_idx]
        layers, full, indirect = series[model_key]
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.85)
        ax.plot(layers, full, marker="o", markersize=5.5, linewidth=2.0, color="#d73027", label="Full")
        if np.isfinite(indirect).any():
            ax.plot(
                layers,
                indirect,
                marker="^",
                markersize=6.0,
                linewidth=2.0,
                color="#4575b4",
                label="Indirect",
            )

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(float(np.min(layers)) - 0.5, float(np.max(layers)) + 0.5)
        ax.set_xticks(layer_ticks(int(np.max(layers)) + 1))
        ax.set_ylabel(r"$\Delta$ Refusal Rate", fontsize=22, labelpad=9)
        ax.set_xlabel("Steered Layer", fontsize=22, labelpad=8)
        ax.tick_params(axis="both", labelsize=16, length=4, width=1.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)
        ax.text(
            -0.22,
            0.5,
            row_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=MODEL_LABEL_FS,
            fontweight="bold",
        )
        if row_idx == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=16, ncol=2, handlelength=1.9)

    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.10, top=0.98, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    alpha_tag = alpha_label(args.alpha)

    if args.model:
        models = [("model", args.model, model_label(args.model))]
        heatmap_paths = {
            "model": Path(args.input) if args.input else latest_result_json(base_dir, args.model, alpha_tag),
        }
        behavior_paths = {
            "model": Path(args.behavior_input)
            if args.behavior_input
            else latest_behavior_json(base_dir, args.model, alpha_tag),
        }
    else:
        models = MODELS
        heatmap_paths = {
            "llama": Path(args.llama)
            if args.llama
            else latest_result_json(base_dir, "llama-3.1-8b-instruct", alpha_tag),
            "qwen": Path(args.qwen)
            if args.qwen
            else latest_result_json(base_dir, "qwen2.5-7b-instruct", alpha_tag),
        }
        behavior_paths = {
            "llama": Path(args.llama_behavior)
            if args.llama_behavior
            else latest_behavior_json(base_dir, "llama-3.1-8b-instruct", alpha_tag),
            "qwen": Path(args.qwen_behavior)
            if args.qwen_behavior
            else latest_behavior_json(base_dir, "qwen2.5-7b-instruct", alpha_tag),
        }

    print("Fig4 inputs:")
    for model_key, path in heatmap_paths.items():
        print(f"  {model_key}: {path}")
    render_heatmap_grid(heatmap_paths, output_dir / args.fig4, args.dpi, models=models)

    print("Fig4_2 inputs:")
    for model_key, path in behavior_paths.items():
        print(f"  {model_key}: {path}")
    render_behavior_grid(behavior_paths, output_dir / args.fig4_2, args.dpi, models=models)


if __name__ == "__main__":
    main()
