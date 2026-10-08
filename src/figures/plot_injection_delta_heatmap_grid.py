#!/usr/bin/env python3
"""
Plot injection-delta heatmaps without averaging over readout layers.

Reads injection result JSONs produced by clean_tinst_harmfulness_projection.py.
For each model and dataset, it plots the full matrix:

    rows: injection layer L
    cols: readout layer K
    value: mean injected-clean projection delta at readout layer K
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import font_manager

from model_utils import get_output_root, normalize_model_name, resolve_model_spec
from activation_patching_coupling import resolve_base_dir


FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name() if FONT_PATH.exists() else "Times New Roman"

DEFAULT_BASE_DIR = "out_pt"
DEFAULT_MODELS = "llama31,qwen25"
DEFAULT_MEASURE = "refusal_tpost"
DEFAULT_DATASETS = "alpaca,misrepresentation,authority_endorsement,expert_endorsement"
DATASET_LABELS = {
    "alpaca": "Alpaca",
    "misrepresentation": "Misrepresentation",
    "authority_endorsement": "Authority endorsement",
    "expert_endorsement": "Expert endorsement",
}
MODEL_CAPTIONS = {
    "llama31": "(a) LLaMA 3.1",
    "qwen25": "(b) Qwen 2.5",
    "qwen35": "(a) Qwen 3.5 9B",
    "aya23": "(a) Aya 23 8B",
    "aya-23-8b": "(a) Aya 23 8B",
    "internlm25": "(a) InternLM 2.5 7B",
    "internlm2_5-7b-chat": "(a) InternLM 2.5 7B",
    "olmo2": "(a) OLMo 2 7B",
    "olmo-2-1124-7b-instruct": "(a) OLMo 2 7B",
    "falcon3-7b-instruct": "(a) Falcon3 7B",
    "gemma-2-9b-it": "(a) Gemma 2 9B",
    "granite-3.1-8b-instruct": "(a) Granite 3.1 8B",
    "llama-2-7b-chat-hf": "(a) LLaMA 2 7B",
    "llama-2-13b-chat-hf": "(a) LLaMA 2 13B",
    "llama-2-70b-chat-hf": "(a) LLaMA 2 70B",
    "llama-3.1-8b-instruct": "(a) LLaMA 3.1 8B",
    "llama-3.1-70b-instruct": "(a) LLaMA 3.1 70B",
    "llama-3.1-405b-instruct": "(a) LLaMA 3.1 405B",
    "mistral-7b-instruct-v0.3": "(a) Mistral 7B",
    "qwen2.5-7b-instruct": "(a) Qwen 2.5 7B",
    "qwen3.5-9b": "(a) Qwen 3.5 9B",
    "qwen2.5-14b-instruct": "(a) Qwen 2.5 14B",
    "qwen2.5-32b-instruct": "(a) Qwen 2.5 32B",
    "qwen2.5-72b-instruct": "(a) Qwen 2.5 72B",
    "yi-1.5-9b-chat": "(a) Yi 1.5 9B",
}
MODEL_ROW_LABEL_FS = 34
TICK_FS = 16
AXIS_FS = 22
COLORBAR_TICK_FS = 16
COLORBAR_LABEL_FS = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full injection-layer x readout-layer delta heatmaps."
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--inputs",
        default=None,
        help="Optional comma-separated JSON paths matching --models order.",
    )
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--measure", default=DEFAULT_MEASURE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--separate-scales",
        action="store_true",
        help="Use an independent color scale per subplot instead of one per model row.",
    )
    parser.add_argument(
        "--global-scale",
        action="store_true",
        help="Use one shared color scale across all models and datasets.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all cells. By default readout_layer <= injection_layer is masked white.",
    )
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_model_list(raw: str) -> list[str]:
    if raw.strip().lower() in {"both", "all"}:
        raw = DEFAULT_MODELS
    models = [normalize_model_name(item.strip()) or item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model alias.")
    return list(dict.fromkeys(models))


def parse_input_list(raw: str | None, models: list[str]) -> dict[str, Path] | None:
    if not raw:
        return None
    paths = [Path(item.strip()).expanduser() for item in raw.split(",") if item.strip()]
    if len(paths) != len(models):
        raise ValueError("--inputs must contain exactly one JSON path per --models entry.")
    return dict(zip(models, paths))


def latest_json_for_model(base_dir: str | Path, model: str, measure: str) -> Path:
    base = resolve_base_dir(base_dir)
    search_dirs = [
        base / measure / "injection",
        base / model / "clean_projection" / measure / "injection",
        base / model / measure / "injection",
    ]
    try:
        spec = resolve_model_spec(model=model, model_id=None)
    except ValueError:
        spec = None
    if spec is not None:
        search_dirs.extend(
            [
                get_output_root(base, model=model, model_id=spec.model_id) / "clean_projection" / measure / "injection",
                base / spec.output_name / measure / "injection",
            ]
        )
    candidates: list[Path] = []
    for search_dir in search_dirs:
        candidates.extend(search_dir.glob("*.json"))
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        tried = "\n".join(str(path) for path in search_dirs)
        raise FileNotFoundError(f"No injection JSON found. Tried:\n{tried}")
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_delta_map(payload: dict[str, Any], dataset_name: str) -> np.ndarray:
    summaries = payload.get("injection_delta_summaries") or {}
    if dataset_name not in summaries:
        raise KeyError(f"Missing injection_delta_summaries['{dataset_name}']")
    values = summaries[dataset_name].get("delta_mean_map")
    if values is None:
        raise KeyError(f"Missing delta_mean_map for dataset '{dataset_name}'")
    # Stored and plotted as [injection_layer, readout_layer].
    return np.array(values, dtype=np.float64)


def display_name(model: str) -> str:
    return MODEL_CAPTIONS.get(model, model)


def default_output_dir(base_dir: str | Path, measure: str) -> Path:
    return resolve_base_dir(base_dir) / "clean_projection" / measure / "heatmap_grid_plots"


def downstream_mask(matrix: np.ndarray) -> np.ndarray:
    injection_layer = np.arange(matrix.shape[0])[:, None]
    readout_layer = np.arange(matrix.shape[1])[None, :]
    return readout_layer > injection_layer


def apply_downstream_mask(matrix: np.ndarray, show_all: bool) -> np.ndarray:
    if show_all:
        return matrix
    masked = matrix.copy()
    masked[~downstream_mask(matrix)] = np.nan
    return masked


def finite_symmetric_limit(maps: list[np.ndarray], show_all: bool) -> float:
    masked_maps = [apply_downstream_mask(m, show_all) for m in maps]
    values = np.concatenate([m[np.isfinite(m)].reshape(-1) for m in masked_maps if np.isfinite(m).any()])
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(values), 98))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def layer_ticks(size: int) -> list[int]:
    step = 4
    ticks = list(range(0, size, step))
    last = size - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


def save_plot(
    maps: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    measure: str,
    dpi: int,
    separate_scales: bool,
    global_scale: bool,
    show_all: bool,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": FONT_NAME,
            "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    models = list(maps)
    datasets = list(next(iter(maps.values())))
    fig, axes = plt.subplots(
        len(models),
        len(datasets),
        figsize=(4.35 * len(datasets) + 1.35, 4.35 * len(models) + 0.25),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    global_vmax = finite_symmetric_limit(
        [maps[model][dataset] for model in models for dataset in datasets],
        show_all=show_all,
    )
    model_vmax = {
        model: finite_symmetric_limit([maps[model][dataset] for dataset in datasets], show_all=show_all)
        for model in models
    }
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    row_images = {}
    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            matrix = apply_downstream_mask(maps[model][dataset], show_all=show_all)
            if separate_scales:
                vmax = finite_symmetric_limit([matrix], show_all=True)
            elif global_scale:
                vmax = global_vmax
            else:
                vmax = model_vmax[model]
            image = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                aspect="equal",
                origin="lower",
            )
            row_images[model] = image
            ax.set_box_aspect(matrix.shape[0] / matrix.shape[1])
            ax.set_xticks(layer_ticks(matrix.shape[1]))
            ax.set_yticks(layer_ticks(matrix.shape[0]))
            ax.tick_params(axis="both", labelsize=TICK_FS, length=4.0, width=1.0, pad=3.0)
            ax.grid(False)
            if row_idx == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=AXIS_FS, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel("Injection Layer", fontsize=AXIS_FS, labelpad=12)
            if row_idx == len(models) - 1:
                ax.set_xlabel("Readout Layer", fontsize=AXIS_FS, labelpad=12)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            if separate_scales:
                cbar = fig.colorbar(image, ax=ax)
                cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=4, width=1.0)
                cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=COLORBAR_LABEL_FS, labelpad=14)

        axes[row_idx][0].text(
            -0.36,
            0.5,
            display_name(model),
            transform=axes[row_idx][0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=MODEL_ROW_LABEL_FS,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.205, right=0.91, bottom=0.08, top=0.92, wspace=0.20, hspace=0.02)

    if not separate_scales:
        if global_scale:
            image = next(iter(row_images.values()))
            anchor = axes[-1][-1].get_position()
            cax = fig.add_axes([anchor.x1 + 0.012, axes[-1][0].get_position().y0, 0.016, axes[0][0].get_position().y1 - axes[-1][0].get_position().y0])
            cbar = fig.colorbar(image, cax=cax)
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=4, width=1.0)
            cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=COLORBAR_LABEL_FS, labelpad=14)
        else:
            for row_idx, model in enumerate(models):
                anchor = axes[row_idx][-1].get_position()
                cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.016, anchor.height])
                cbar = fig.colorbar(row_images[model], cax=cax)
                cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=4, width=1.0)
                cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=COLORBAR_LABEL_FS, labelpad=14)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models = parse_model_list(args.models)
    datasets = parse_csv(args.datasets)
    explicit_inputs = parse_input_list(args.inputs, models)
    input_paths = explicit_inputs or {
        model: latest_json_for_model(args.base_dir, model, args.measure)
        for model in models
    }

    maps: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {}
    for model, path in input_paths.items():
        payload = load_json(path)
        maps[model] = {dataset: get_delta_map(payload, dataset) for dataset in datasets}
        metadata[model] = {"input_path": str(path.resolve())}
        print(f"[heatmap-grid] {model}: input={path}", flush=True)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(args.base_dir, args.measure)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = args.output_name or f"full-injection-delta-heatmap-grid-{stamp}"
    output_base = output_dir / output_name

    png_path = output_base.with_suffix(".png")
    save_plot(
        maps=maps,
        output_path=png_path,
        measure=args.measure,
        dpi=int(args.dpi),
        separate_scales=bool(args.separate_scales),
        global_scale=bool(args.global_scale),
        show_all=bool(args.show_all),
    )
    print(f"[heatmap-grid] saved plot to {png_path}", flush=True)

    json_path = output_base.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "script": "plot_injection_delta_heatmap_grid.py",
                    "measure": args.measure,
                    "models": models,
                    "datasets": datasets,
                    "per_model": metadata,
                    "axis_note": (
                        "heatmap x=readout layer, y=injection layer; y=0 starts at bottom"
                    ),
                    "mask_note": (
                        "default masks readout_layer <= injection_layer as white; "
                        "--show-all disables this mask"
                    ),
                    "scale_note": (
                        "default color scale is shared within each model row across datasets; "
                        "--global-scale shares across all panels; --separate-scales uses each subplot independently"
                    ),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[heatmap-grid] saved metadata to {json_path}", flush=True)


if __name__ == "__main__":
    main()
