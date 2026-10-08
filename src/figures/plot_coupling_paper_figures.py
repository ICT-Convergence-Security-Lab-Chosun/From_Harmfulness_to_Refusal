#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name() if FONT_PATH.exists() else "Times New Roman"
plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
    }
)


MODEL_CAPTIONS = {
    "llama31": "(a) LLaMA 3.1",
    "qwen25": "(b) Qwen 2.5",
}
MODEL_DISPLAY_NAMES = {
    "aya23": "Aya 23 8B",
    "internlm25": "InternLM 2.5 7B",
    "olmo2": "OLMo 2 7B",
    "llama2_7b": "LLaMA 2 7B",
    "llama2_13b": "LLaMA 2 13B",
    "llama2_70b": "LLaMA 2 70B",
    "llama31": "LLaMA 3.1",
    "qwen25": "Qwen 2.5",
    "mistral": "Mistral 7B",
    "gemma2": "Gemma 2 9B",
    "falcon3": "Falcon3 7B",
    "granite": "Granite 3.1 8B",
    "yi": "Yi 1.5 9B",
}
MODEL_ROW_LABEL_FS = 34
MODEL_PANEL_LABEL_FS = 26
TICK_FS = 16
AXIS_FS = 22
COLORBAR_TICK_FS = 16
COLORBAR_LABEL_FS = 22

DATASET_LABELS = {
    "alpaca_common": "Alpaca",
    "sorry_misrepresentation": "Misrepresentation",
    "sorry_authority_endorsement": "Authority endorsement",
    "sorry_expert_endorsement": "Expert endorsement",
}

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = SCRIPT_DIR / "out_pt"
DEFAULT_RUN_DIR = DEFAULT_BASE_DIR / "fig_coupling_runs"
DEFAULT_FIGURE_DIR = DEFAULT_BASE_DIR / "Figure"
FIG5_DATASETS = [
    "sorry_misrepresentation",
    "sorry_authority_endorsement",
    "sorry_expert_endorsement",
]
DATASET_ALIASES = {
    "alpaca": "alpaca_common",
    "alpaca_common": "alpaca_common",
    "misrepresentation": "sorry_misrepresentation",
    "sorry_misrepresentation": "sorry_misrepresentation",
    "authority_endorsement": "sorry_authority_endorsement",
    "sorry_authority_endorsement": "sorry_authority_endorsement",
    "expert_endorsement": "sorry_expert_endorsement",
    "sorry_expert_endorsement": "sorry_expert_endorsement",
}


def load_values(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    values = np.array(payload["internal_coupling"]["normalized_coupling_map"], dtype=np.float64)
    return values


def symmetric_limits(maps: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([values[np.isfinite(values)] for values in maps])
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
    if ticks and ticks[-1] != last:
        if last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


def plot_grid(
    paths: list[list[Path]],
    output: Path,
    dpi: int,
    model_by_row: bool,
    show_panel_captions: bool,
    column_labels: list[str] | None = None,
    row_captions: list[str] | None = None,
    panel_captions: list[str] | None = None,
) -> None:
    maps = [[load_values(path) for path in row] for row in paths]
    n_rows = len(paths)
    n_cols = len(paths[0])
    if model_by_row:
        model_limits = [symmetric_limits(row) for row in maps]
        limit_for = lambda row_idx, col_idx: model_limits[row_idx]
    else:
        model_limits = [
            symmetric_limits([maps[row_idx][col_idx] for row_idx in range(len(maps))])
            for col_idx in range(n_cols)
        ]
        limit_for = lambda row_idx, col_idx: model_limits[col_idx]

    panel_size = 4.35
    fig_width = panel_size * n_cols + (1.35 if model_by_row else 1.55)
    fig_height = panel_size * n_rows + (0.55 if n_rows == 1 else 0.25)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    colorbar_images = [None for _ in range(n_rows if model_by_row else n_cols)]
    for row_idx, row in enumerate(maps):
        for col_idx, values in enumerate(row):
            ax = axes[row_idx][col_idx]
            masked = np.ma.masked_invalid(values)
            vmin, vmax = limit_for(row_idx, col_idx)
            image = ax.imshow(masked, origin="lower", aspect="equal", cmap="coolwarm", vmin=vmin, vmax=vmax)
            colorbar_images[row_idx if model_by_row else col_idx] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            xticks = layer_ticks(values.shape[1])
            yticks = layer_ticks(values.shape[0])
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(axis="both", labelsize=TICK_FS, length=4.0, width=1.0, pad=3.0)
            ax.grid(False)
            if col_idx == 0:
                ax.set_ylabel("Injection Layer", fontsize=AXIS_FS, labelpad=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Downstream Layer", fontsize=AXIS_FS, labelpad=10)
            if column_labels and row_idx == 0:
                ax.set_title(column_labels[col_idx], fontsize=AXIS_FS, fontweight="bold", pad=14)

        if model_by_row:
            axes[row_idx][0].text(
                -0.36,
                0.5,
                row_captions[row_idx]
                if row_captions
                else MODEL_CAPTIONS["llama31" if row_idx == 0 else "qwen25"],
                transform=axes[row_idx][0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=MODEL_ROW_LABEL_FS,
                fontweight="bold",
            )
        elif show_panel_captions:
            for col_idx in range(n_cols):
                axes[-1][col_idx].text(
                    0.5,
                    -0.24,
                    panel_captions[col_idx]
                    if panel_captions
                    else MODEL_CAPTIONS["llama31" if col_idx == 0 else "qwen25"],
                    transform=axes[-1][col_idx].transAxes,
                    ha="center",
                    va="top",
                    fontsize=MODEL_PANEL_LABEL_FS,
                    fontweight="bold",
                )

    fig.subplots_adjust(
        left=0.205 if model_by_row else 0.115,
        right=0.905,
        bottom=0.16 if n_rows == 1 else 0.08,
        top=0.92 if column_labels else 0.985,
        wspace=0.32 if not model_by_row else 0.20,
        hspace=0.02 if model_by_row else 0.16,
    )
    for idx, image in enumerate(colorbar_images):
        if image is None:
            continue
        if model_by_row:
            anchor = axes[idx][n_cols - 1].get_position()
        else:
            anchor = axes[0][idx].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.016, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=4, width=1.0)
        # if idx == len(colorbar_images) - 1:
        # cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=15)
        if model_by_row:
            # Fig5 uses row-wise colorbars, so label both upper and lower colorbars.
            cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=COLORBAR_LABEL_FS, labelpad=14)
        elif idx == len(colorbar_images) - 1:
            # Fig2 labels only the final, rightmost colorbar.
            cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=COLORBAR_LABEL_FS, labelpad=14) 

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def latest_matching(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files match {directory / pattern}")
    return matches[-1]


def normalize_model_key(raw: str) -> str:
    value = raw.lower()
    if value in {"llama2", "llama-2"}:
        return "llama2"
    if "llama-2-7b" in value:
        return "llama2_7b"
    if "llama-2-13b" in value:
        return "llama2_13b"
    if "llama-2-70b" in value:
        return "llama2_70b"
    if "llama" in value:
        return "llama31"
    if "qwen" in value:
        return "qwen25"
    if "aya" in value:
        return "aya23"
    if "internlm" in value:
        return "internlm25"
    if "olmo" in value:
        return "olmo2"
    if "mistral" in value:
        return "mistral"
    if "gemma" in value:
        return "gemma2"
    if "falcon" in value:
        return "falcon3"
    if "granite" in value:
        return "granite"
    if "yi" in value:
        return "yi"
    return value


def model_keys_match(payload_model: str, requested_model: str) -> bool:
    normalized_payload_model = normalize_model_key(payload_model)
    normalized_requested_model = normalize_model_key(requested_model)
    if normalized_payload_model == normalized_requested_model:
        return True
    return normalized_payload_model == "llama2" and normalized_requested_model.startswith("llama2_")


def model_display_name(model_key: str) -> str:
    normalized = normalize_model_key(model_key)
    return MODEL_DISPLAY_NAMES.get(normalized, model_key)


def model_caption(model_key: str, panel_index: int = 0) -> str:
    letter = chr(ord("a") + panel_index)
    return f"({letter}) {model_display_name(model_key)}"


def dataset_key_from_payload(payload: dict) -> str:
    metadata = payload.get("metadata", {})
    input_path = metadata.get("input_path")
    if input_path:
        key = Path(input_path).stem.replace("-", "_")
        return DATASET_ALIASES.get(key, key)
    label = metadata.get("label") or ""
    key = str(label).replace("-", "_")
    for alias, canonical in DATASET_ALIASES.items():
        if key == alias or key.endswith(f"_{alias}"):
            return canonical
    return key


def latest_coupling_json(search_root: Path, model_key: str, dataset_key: str) -> Path:
    normalized_dataset_key = DATASET_ALIASES.get(dataset_key, dataset_key)
    candidates: list[Path] = []
    for path in search_root.rglob("harm-to-refusal-coupling-hidden-alpha_3-*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if "internal_coupling" not in payload:
            continue
        metadata = payload.get("metadata", {})
        if not model_keys_match(str(metadata.get("model", "")), model_key):
            continue
        if dataset_key_from_payload(payload) != normalized_dataset_key:
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No latest coupling JSON found for model={model_key} dataset={dataset_key} under {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def default_fig2_paths(run_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    llama = latest_coupling_json(run_dir, "llama31", "alpaca_common")
    qwen = latest_coupling_json(run_dir, "qwen25", "alpaca_common")
    return [llama], [qwen], ["alpaca_common"]


def default_fig5_paths(run_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    llama_paths = [
        latest_coupling_json(run_dir, "llama31", dataset)
        for dataset in FIG5_DATASETS
    ]
    qwen_paths = [
        latest_coupling_json(run_dir, "qwen25", dataset)
        for dataset in FIG5_DATASETS
    ]
    return llama_paths, qwen_paths, FIG5_DATASETS


def default_single_model_paths(run_dir: Path, model_key: str) -> tuple[list[Path], list[Path], list[str]]:
    fig2_paths = [latest_coupling_json(run_dir, model_key, "alpaca_common")]
    fig5_paths = [latest_coupling_json(run_dir, model_key, dataset) for dataset in FIG5_DATASETS]
    return fig2_paths, fig5_paths, ["alpaca_common", *FIG5_DATASETS]


def render_figure(
    fig_name: str,
    llama_paths: list[Path],
    qwen_paths: list[Path],
    rows: list[str],
    output: Path,
    dpi: int,
) -> None:
    if len(llama_paths) != len(qwen_paths) or len(llama_paths) != len(rows):
        raise ValueError("--llama, --qwen, and --rows must have the same length.")

    if fig_name == "fig5":
        paths = [llama_paths, qwen_paths]
    else:
        paths = [[llama_path, qwen_path] for llama_path, qwen_path in zip(llama_paths, qwen_paths)]

    plot_grid(
        paths=paths,
        output=output,
        dpi=dpi,
        model_by_row=fig_name == "fig5",
        show_panel_captions=fig_name == "fig2",
        column_labels=[DATASET_LABELS.get(row, row) for row in rows] if fig_name == "fig5" else None,
    )


def render_single_model_figure(
    fig_name: str,
    model_key: str,
    paths: list[Path],
    rows: list[str],
    output: Path,
    dpi: int,
) -> None:
    if len(paths) != len(rows):
        raise ValueError("--paths and --rows must have the same length.")

    if fig_name == "fig5":
        plot_grid(
            paths=[paths],
            output=output,
            dpi=dpi,
            model_by_row=True,
            show_panel_captions=False,
            column_labels=[DATASET_LABELS.get(row, row) for row in rows],
            row_captions=[model_caption(model_key)],
        )
    else:
        plot_grid(
            paths=[paths],
            output=output,
            dpi=dpi,
            model_by_row=False,
            show_panel_captions=True,
            panel_captions=[model_caption(model_key)],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", choices=["fig2", "fig5", "all"], default="all")
    parser.add_argument("--llama", nargs="+")
    parser.add_argument("--qwen", nargs="+")
    parser.add_argument("--model", help="Render a single-model paper-style figure using only this model's JSON files.")
    parser.add_argument("--paths", nargs="+", help="Explicit single-model coupling JSON paths.")
    parser.add_argument("--rows", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--run-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    if args.model or args.paths:
        if not args.model:
            raise ValueError("--model is required for single-model rendering.")
        if args.fig == "all" and args.paths:
            raise ValueError("--fig must be fig2 or fig5 when explicit --paths are supplied.")
        if args.paths:
            if not args.rows:
                raise ValueError("--rows is required with explicit --paths.")
            output = Path(args.output) if args.output else output_dir / ("Fig2.png" if args.fig == "fig2" else "Fig5.png")
            render_single_model_figure(
                fig_name=args.fig,
                model_key=args.model,
                paths=[Path(path) for path in args.paths],
                rows=args.rows,
                output=output,
                dpi=args.dpi,
            )
            return

        fig2_paths, fig5_paths, rows = default_single_model_paths(run_dir, args.model)
        if args.fig in ("fig2", "all"):
            output = Path(args.output) if args.output and args.fig == "fig2" else output_dir / "Fig2.png"
            render_single_model_figure("fig2", args.model, fig2_paths, rows[:1], output, args.dpi)
        if args.fig in ("fig5", "all"):
            output = Path(args.output) if args.output and args.fig == "fig5" else output_dir / "Fig5.png"
            render_single_model_figure("fig5", args.model, fig5_paths, rows[1:], output, args.dpi)
        return

    if args.llama or args.qwen or args.rows:
        if not (args.llama and args.qwen and args.rows):
            raise ValueError("--llama, --qwen, and --rows must be supplied together.")
        if args.fig == "all":
            raise ValueError("--fig must be fig2 or fig5 when explicit paths are supplied.")
        output = Path(args.output) if args.output else output_dir / ("Fig2.png" if args.fig == "fig2" else "Fig5.png")
        render_figure(
            fig_name=args.fig,
            llama_paths=[Path(path) for path in args.llama],
            qwen_paths=[Path(path) for path in args.qwen],
            rows=args.rows,
            output=output,
            dpi=args.dpi,
        )
        return

    if args.fig in ("fig2", "all"):
        llama_paths, qwen_paths, rows = default_fig2_paths(run_dir)
        output = Path(args.output) if args.output and args.fig == "fig2" else output_dir / "Fig2.png"
        render_figure("fig2", llama_paths, qwen_paths, rows, output, args.dpi)

    if args.fig in ("fig5", "all"):
        llama_paths, qwen_paths, rows = default_fig5_paths(run_dir)
        output = Path(args.output) if args.output and args.fig == "fig5" else output_dir / "Fig5.png"
        render_figure("fig5", llama_paths, qwen_paths, rows, output, args.dpi)


if __name__ == "__main__":
    main()
