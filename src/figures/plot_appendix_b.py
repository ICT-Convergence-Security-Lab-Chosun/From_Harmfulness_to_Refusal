#!/usr/bin/env python3
"""
Appendix B figure generator — fully self-contained.

Generates Fig2, Fig3, Fig4, Fig4_2, Fig5, Fig6, Fig7, Fig8
for LLaMA 3.1 70B, Qwen 2.5 14B, Qwen 2.5 32B, Qwen 2.5 72B
→ out_pt/Figure_Appendix_B/

Default steering coefficient (alpha) = 6.  Pass --steering-alpha N to override.

Layout:
  - Fig2       : horizontal (1 row × 4 cols, one model per col)
  - Fig3       : vertical (4 rows, one per model)
  - Fig4       : vertical (4 rows × 2 cols, full / indirect)
  - Fig4_2     : vertical (4 rows, behavioral series)
  - Fig5       : vertical (4 rows × 3 cols, multi-dataset coupling)
  - Fig6       : vertical (4 rows × 3 cols, adversarial steering)
  - Fig7 / Fig8: vertical (4 rows, clean projection)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------
FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = (
    font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    if FONT_PATH.exists()
    else "Times New Roman"
)
plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.serif": [FONT_NAME, "Times New Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR   = SCRIPT_DIR / "out_pt"
DEFAULT_FIGURE_DIR = DEFAULT_BASE_DIR / "Figure_Appendix_B"

# ---------------------------------------------------------------------------
# Model / figure config
# ---------------------------------------------------------------------------
# (alias, model_dir_name, coupling_key, display_name, panel_label)
APPENDIX_MODELS = [
    ("llama31_70b", "llama-3.1-70b-instruct",  "llama31", "LLaMA 3.1 70B", "(a) LLaMA 3.1 70B"),
    ("qwen25_14b",  "qwen2.5-14b-instruct",    "qwen25",  "Qwen 2.5 14B",  "(b) Qwen 2.5 14B"),
    ("qwen25_32b",  "qwen2.5-32b-instruct",    "qwen25",  "Qwen 2.5 32B",  "(c) Qwen 2.5 32B"),
    ("qwen25_72b",  "qwen2.5-72b-instruct",    "qwen25",  "Qwen 2.5 72B",  "(d) Qwen 2.5 72B"),
]

# Bridge layer ranges (inclusive, 0-based) for shading in Fig3.
# Set to None to disable shading for a model.
BRIDGE_LAYERS_FIG3: dict[str, tuple[int, int] | None] = {
    # Bridge layer ranges are inclusive and 0-based.
    "llama-3.1-70b-instruct": (16, 26),
    "qwen2.5-14b-instruct":   (24, 25),
    "qwen2.5-32b-instruct":   (33, 38),
    "qwen2.5-72b-instruct":   (48, 51),
}

FIG5_DATASETS = [
    ("misrepresentation",     "Misrepresentation"),
    ("authority_endorsement", "Authority Endorsement"),
    ("expert_endorsement",    "Expert Endorsement"),
]

# ===========================================================================
# Shared utility
# ===========================================================================

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_figure(fig, output: Path, *, dpi: int, **kwargs) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, **kwargs)
    fig.savefig(output.with_suffix(".pdf"), **kwargs)


def _layer_ticks(size: int, max_ticks: int = 9) -> list[int]:
    """
    Return reasonably dense layer ticks while always showing the last layer.
    Avoid placing the last two ticks too close to each other.
    """
    if size <= 1:
        return [0]

    last = size - 1

    # target density: a bit denser than before
    raw_step = math.ceil(last / (max_ticks - 1))

    # choose a visually nice step
    nice_steps = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    step = next((s for s in nice_steps if s >= raw_step), raw_step)

    ticks = list(range(0, size, step))

    if ticks[-1] != last:
        ticks.append(last)

    # If the last tick is too close to the previous one, replace previous with last
    if len(ticks) >= 2 and (ticks[-1] - ticks[-2]) < max(2, step * 0.5):
        ticks[-2] = last
        ticks = ticks[:-1]

    # remove duplicates and sort
    ticks = sorted(set(ticks))

    return ticks

def _sym_limits(maps: list[np.ndarray]) -> tuple[float, float]:
    chunks = [v[np.isfinite(v)] for v in maps]
    finite = np.concatenate([c for c in chunks if c.size > 0])
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 98))
    if limit <= 0:
        limit = float(np.nanmax(np.abs(finite)))
    if limit <= 0:
        limit = 1.0
    return -limit, limit


# ===========================================================================
# Coupling helpers
# ===========================================================================

DATASET_ALIASES: dict[str, str] = {
    "alpaca":                       "alpaca_common",
    "alpaca_common":                "alpaca_common",
    "misrepresentation":            "sorry_misrepresentation",
    "sorry_misrepresentation":      "sorry_misrepresentation",
    "authority_endorsement":        "sorry_authority_endorsement",
    "sorry_authority_endorsement":  "sorry_authority_endorsement",
    "expert_endorsement":           "sorry_expert_endorsement",
    "sorry_expert_endorsement":     "sorry_expert_endorsement",
}

def _load_coupling_values(path: Path) -> np.ndarray:
    payload = _load_json(path)
    return np.array(payload["internal_coupling"]["normalized_coupling_map"], dtype=np.float64)


def _normalize_model_key(raw: str) -> str:
    v = raw.lower()
    if "llama-2-7b"  in v: return "llama2_7b"
    if "llama-2-13b" in v: return "llama2_13b"
    if "llama-2-70b" in v: return "llama2_70b"
    if v in {"llama2", "llama-2"}: return "llama2"
    if "llama"   in v: return "llama31"
    if "qwen"    in v: return "qwen25"
    if "aya"     in v: return "aya23"
    if "internlm" in v: return "internlm25"
    if "olmo"    in v: return "olmo2"
    if "mistral" in v: return "mistral"
    if "gemma"   in v: return "gemma2"
    if "falcon"  in v: return "falcon3"
    if "granite" in v: return "granite"
    if "yi"      in v: return "yi"
    return v


def _model_keys_match(payload_model: str, requested_model: str) -> bool:
    np_ = _normalize_model_key(payload_model)
    nr_ = _normalize_model_key(requested_model)
    if np_ == nr_:
        return True
    return np_ == "llama2" and nr_.startswith("llama2_")


def _dataset_key_from_payload(payload: dict) -> str:
    meta = payload.get("metadata", {})
    inp  = meta.get("input_path")
    if inp:
        key = Path(inp).stem.replace("-", "_")
        return DATASET_ALIASES.get(key, key)
    label = str(meta.get("label") or "").replace("-", "_")
    for alias, canonical in DATASET_ALIASES.items():
        if label == alias or label.endswith(f"_{alias}"):
            return canonical
    return label


def _alpha_int_str(alpha: float) -> str:
    """Return '6' for 6.0, '3' for 3.0, etc."""
    return str(int(alpha)) if alpha == int(alpha) else str(alpha)


def latest_coupling_json(
    search_root: Path, model_key: str, dataset_key: str, alpha: float = 6.0
) -> Path:
    normalized = DATASET_ALIASES.get(dataset_key, dataset_key)
    alpha_str  = _alpha_int_str(alpha)
    pattern    = f"harm-to-refusal-coupling-hidden-alpha_{alpha_str}-*.json"
    candidates: list[Path] = []
    for path in search_root.rglob(pattern):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if "internal_coupling" not in payload:
            continue
        meta = payload.get("metadata", {})
        if not _model_keys_match(str(meta.get("model", "")), model_key):
            continue
        if _dataset_key_from_payload(payload) != normalized:
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No coupling JSON (alpha={alpha}) for model={model_key} "
            f"dataset={dataset_key} under {search_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_coupling_grid(
    paths: list[list[Path]],
    output: Path,
    dpi: int,
    model_by_row: bool,
    show_panel_captions: bool,
    column_labels: list[str] | None = None,
    row_captions: list[str] | None = None,
    panel_captions: list[list[str]] | None = None,
    # --- font / layout knobs (each render_fig* sets these independently) ---
    tick_fs: int = 13,
    axis_fs: int = 21,
    cbar_tick_fs: int = 13,
    cbar_label_fs: int = 18,
    row_label_fs: int = 23,
    panel_label_fs: int = 21,
    col_header_fs: int = 22,
    per_row_scale: bool = False,
) -> None:
    maps   = [[_load_coupling_values(p) for p in row] for row in paths]
    n_rows = len(paths)
    n_cols = len(paths[0])

    # Pre-compute per-row color scales if requested (used by Fig5)
    row_scales: dict[int, tuple[float, float]] = {}
    if per_row_scale:
        for ri, row_maps in enumerate(maps):
            row_scales[ri] = _sym_limits(row_maps)

    # more compact figure size
    panel_size = 3.95
    fig_width  = panel_size * n_cols + 1.45
    fig_height = panel_size * n_rows + 1.10

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    panel_images = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for ri, row in enumerate(maps):
        for ci, values in enumerate(row):
            ax = axes[ri][ci]

            # per-row scale when requested, otherwise per-panel
            if per_row_scale:
                vmin, vmax = row_scales[ri]
            else:
                vmin, vmax = _sym_limits([values])

            image = ax.imshow(
                np.ma.masked_invalid(values),
                origin="lower",
                aspect="equal",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            panel_images[ri][ci] = image

            ax.set_box_aspect(values.shape[0] / values.shape[1])

            xticks = _layer_ticks(values.shape[1], max_ticks=10)
            yticks = _layer_ticks(values.shape[0], max_ticks=8)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.tick_params(
                axis="both",
                labelsize=tick_fs,
                length=4.0,
                width=1.0,
                pad=2.0,
            )
            ax.grid(False)

            # y label only on left column
            if ci == 0:
                ax.set_ylabel("Injection Layer", fontsize=axis_fs, labelpad=6)
            else:
                ax.set_ylabel("")

            # x label on every subplot
            ax.set_xlabel("Downstream Layer", fontsize=axis_fs, labelpad=5)

            # model caption below each subplot
            if show_panel_captions and panel_captions is not None:
                ax.text(
                    0.5,
                    -0.22,
                    panel_captions[ri][ci],
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=panel_label_fs,
                    fontweight="bold",
                    clip_on=False,
                )

            # for Fig5-like layouts
            if column_labels and ri == 0:
                ax.text(
                    0.5,
                    1.08,
                    column_labels[ci],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=col_header_fs,
                    fontweight="bold",
                )

        if model_by_row:
            axes[ri][0].text(
                -0.28,
                0.5,
                row_captions[ri] if row_captions else f"({chr(ord('a') + ri)})",
                transform=axes[ri][0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=row_label_fs,
                fontweight="bold",
            )

    # compact spacing
    fig.subplots_adjust(
        left   = 0.10 if not model_by_row else 0.17,
        right  = 0.95,
        bottom = 0.09,
        top    = 0.97,
        wspace = 0.36 if not model_by_row else 0.20,
        hspace = 0.34 if not model_by_row else 0.14,
    )

    # per-panel colorbar; when per_row_scale is used, only show rightmost column
    for ri in range(n_rows):
        for ci in range(n_cols):
            if per_row_scale and ci != n_cols - 1:
                continue  # hide non-rightmost colorbars for clean per-row layout

            image = panel_images[ri][ci]
            ax = axes[ri][ci]
            anchor = ax.get_position()

            cax = fig.add_axes([
                anchor.x1 + 0.006,
                anchor.y0,
                0.013,
                anchor.height,
            ])

            cbar = fig.colorbar(image, cax=cax)
            cbar.ax.tick_params(labelsize=cbar_tick_fs, length=4, width=1.0)

            # label the rightmost colorbar
            if ci == n_cols - 1:
                cbar.set_label(
                    r"$\Delta$ Refusal Projection",
                    fontsize=cbar_label_fs,
                    labelpad=7,
                )

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {output}")


# ===========================================================================
# Activation-patching helpers
# ===========================================================================

def _patching_coupling_values(payload: dict, variant: str) -> np.ndarray:
    coupling = payload.get("activation_patching_coupling") or payload["path_patching_coupling"]
    return np.array(coupling[variant]["normalized_coupling_map"], dtype=np.float64)


def _patching_behavioral_series(
    payload: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bv = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
    if not bv or not bv.get("experiments"):
        raise ValueError("JSON does not contain behavioral_validation experiments.")
    exps   = sorted(bv["experiments"], key=lambda x: int(x["layer"]))
    layers = np.array([int(x["layer"]) for x in exps], dtype=int)
    if "delta_refusal_rate" in exps[0]:
        full     = np.array([x.get("delta_refusal_rate", np.nan) for x in exps], dtype=np.float64)
        indirect = np.full_like(full, np.nan)
    else:
        full     = np.array([x.get("delta_from_baseline_full", np.nan) for x in exps], dtype=np.float64)
        indirect = np.array([x.get("indirect_delta", np.nan) for x in exps], dtype=np.float64)
    return layers, full, indirect


def _candidate_patching_jsons(model_dir: Path, alpha: float = 6.0) -> list[Path]:
    alpha_str = _alpha_int_str(alpha)
    paths: list[Path] = []
    for subdir, pattern in [
        ("activation_patching", f"activation-patching-hidden-alpha_{alpha_str}-*.json"),
        ("path_patching",       f"path-patching-hidden-alpha_{alpha_str}-*.json"),
    ]:
        for p in (model_dir / subdir).rglob(pattern):
            if "generation-for-judge" not in p.name:
                paths.append(p)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def latest_patching_json(base_dir: Path, model_dir_name: str, alpha: float = 6.0) -> Path:
    matches = _candidate_patching_jsons(base_dir / model_dir_name, alpha)
    if not matches:
        raise FileNotFoundError(
            f"No activation-patching JSON (alpha={alpha}) under {base_dir / model_dir_name}"
        )
    return matches[-1]


def latest_patching_behavior_json(
    base_dir: Path, model_dir_name: str, alpha: float = 6.0
) -> Path:
    for p in reversed(_candidate_patching_jsons(base_dir / model_dir_name, alpha)):
        payload = _load_json(p)
        bv = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
        if bv and bv.get("experiments"):
            return p
    raise FileNotFoundError(
        f"No behavioral_validation JSON (alpha={alpha}) under {base_dir / model_dir_name}"
    )


def _mask_non_downstream(values: np.ndarray) -> np.ndarray:
    inj_layer = np.arange(values.shape[0])[:, None]
    dn_layer  = np.arange(values.shape[1])[None, :]
    masked    = values.copy()
    masked[dn_layer <= inj_layer] = np.nan
    return masked


def render_patching_heatmap_grid(
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]],
    # --- font / layout knobs (each render_fig* sets these independently) ---
    tick_fs: int = 18,
    title_fs: int = 22,
    axis_fs: int = 24,
    row_label_fs: int = 28,
    cbar_tick_fs: int = 18,
    cbar_label_fs: int = 24,
) -> None:
    VARIANTS = [("full", "Full"), ("indirect", "Indirect")]
    payloads  = {key: _load_json(p) for key, p in paths.items()}
    maps = {
        mk: {v: _patching_coupling_values(payloads[mk], v) for v, _ in VARIANTS}
        for mk, _, _ in models
    }
    row_limits = {mk: _sym_limits([maps[mk][v] for v, _ in VARIANTS]) for mk, _, _ in models}

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        len(models), 2,
        figsize=(9.8, 4.35 * len(models) + 0.4),
        squeeze=False,
    )
    row_images: dict[int, object] = {}

    for ri, (mk, _, row_label) in enumerate(models):
        for ci, (variant, variant_label) in enumerate(VARIANTS):
            ax = axes[ri, ci]
            values = _mask_non_downstream(maps[mk][variant])
            vmin, vmax = row_limits[mk]
            image = ax.imshow(
                np.ma.masked_invalid(values),
                origin="lower", aspect="equal",
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            row_images[ri] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            ax.set_xticks(_layer_ticks(values.shape[1]))
            ax.set_yticks(_layer_ticks(values.shape[0]))
            ax.tick_params(axis="both", labelsize=tick_fs, length=4, width=1.0, pad=3)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(1.2)
            if ri == 0:
                ax.set_title(variant_label, fontsize=title_fs, pad=10)
            if ri == len(models) - 1:
                ax.set_xlabel("Downstream Layer", fontsize=axis_fs, labelpad=12)
            if ci == 0:
                ax.set_ylabel("Injection Layer",  fontsize=axis_fs, labelpad=12)

        axes[ri, 0].text(
            -0.36, 0.5, row_label,
            transform=axes[ri, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=row_label_fs, fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.21, right=0.90, bottom=0.10, top=0.93, wspace=0.20, hspace=0.18,
    )
    for ri, image in row_images.items():
        anchor = axes[ri, -1].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.018, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=cbar_tick_fs, length=4, width=1.0)
        cbar.outline.set_linewidth(1.0)
        cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=cbar_label_fs, labelpad=14)

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def render_patching_behavior_grid(
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]],
    # --- font / layout knobs (each render_fig* sets these independently) ---
    axis_fs: int = 22,
    ylabel_fs: int | None = None,
    tick_fs: int = 16,
    row_label_fs: int = 28,
    legend_fs: int = 16,
) -> None:
    payloads = {key: _load_json(p) for key, p in paths.items()}
    series   = {mk: _patching_behavioral_series(payloads[mk]) for mk, _, _ in models}

    finite: list[float] = []
    for _, full, indirect in series.values():
        finite.extend(full[np.isfinite(full)].tolist())
        finite.extend(indirect[np.isfinite(indirect)].tolist())
    ymin = min(0.0, float(np.min(finite))) - 0.03 if finite else -0.05
    ymax = max(0.0, float(np.max(finite))) + 0.03 if finite else  0.05

    fig, axes = plt.subplots(len(models), 1, figsize=(7.2, 3.1 * len(models)), sharex=False)
    axes = np.array(axes).reshape(len(models))

    for ri, (mk, _, row_label) in enumerate(models):
        ax = axes[ri]
        layers, full, indirect = series[mk]
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.85)
        ax.plot(layers, full, marker="o", markersize=5.5, linewidth=2.0,
                color="#d73027", label="Full")
        if np.isfinite(indirect).any():
            ax.plot(layers, indirect, marker="^", markersize=6.0, linewidth=2.0,
                    color="#4575b4", label="Indirect")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(float(np.min(layers)) - 0.5, float(np.max(layers)) + 0.5)
        ax.set_xticks(_layer_ticks(int(np.max(layers)) + 1))
        ax.set_ylabel(r"$\Delta$ Refusal Rate", fontsize=ylabel_fs if ylabel_fs is not None else axis_fs, labelpad=9)
        if ri == len(models) - 1:
            ax.set_xlabel("Steered Layer", fontsize=axis_fs, labelpad=8)
        ax.tick_params(axis="both", labelsize=tick_fs, length=4, width=1.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)
        ax.text(
            -0.22, 0.5, row_label,
            transform=ax.transAxes,
            rotation=90, ha="center", va="center",
            fontsize=row_label_fs, fontweight="bold",
        )
        if ri == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=legend_fs, ncol=2, handlelength=1.9)

    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.10, top=0.98, hspace=0.34)
    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ===========================================================================
# Steering helpers
# ===========================================================================

STEER_COLORS = {"harmfulness": "#d73027", "refusal": "#4575b4"}
STEER_LABELS = {"harmfulness": "Harmfulness Steering", "refusal": "Refusal Steering"}
BRIDGE_COLOR = "#DCEAF7"
BRIDGE_ALPHA = 0.42
LABEL_COLORS = {"refusal": "#d73027", "accept": "#1a9850", "other": "#cccccc"}
COMPOSE_LABEL_ORDER = ["refusal"]
PLOT_LABEL_ORDER    = ["refusal", "other"]


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


def _normalize_dataset_key(raw: str) -> str:
    key = Path(raw).stem.replace("-", "_")
    aliases = {
        "alpaca_common":                    "alpaca",
        "steering_alpaca_data_instruction": "alpaca",
        "sorry_misrepresentation":          "misrepresentation",
        "sorry_authority_endorsement":      "authority_endorsement",
        "sorry_expert_endorsement":         "expert_endorsement",
    }
    return aliases.get(key, key)


def _compose_source_path(aggregate_path: Path) -> Path:
    judge_dir   = aggregate_path.parent
    source_stem = judge_dir.name.removesuffix("-wildguard-judge")
    return judge_dir.parent / f"{source_stem}.json"


def _aggregate_matches(aggregate_path: Path, target: str, dataset: str, alpha: float | None = None) -> tuple[bool, float]:
    source_path = _compose_source_path(aggregate_path)
    if not source_path.exists():
        return False, aggregate_path.stat().st_mtime
    try:
        source    = _load_json(source_path)
        aggregate = _load_json(aggregate_path)
    except Exception:
        return False, aggregate_path.stat().st_mtime
    meta = source.get("metadata", {})
    if _normalize_dataset_key(str(meta.get("input_path", ""))) != dataset:
        return False, source_path.stat().st_mtime
    rows = aggregate.get("by_target_layer_alpha", [])
    if target not in {row.get("target") for row in rows}:
        return False, source_path.stat().st_mtime
    if alpha is not None:
        available = {row.get("alpha") for row in rows if row.get("target") == target}
        if alpha not in available:
            return False, source_path.stat().st_mtime
    return True, source_path.stat().st_mtime


def _compose_agg_path(base_dir: Path, model: str, target: str, dataset: str, alpha: float | None = None) -> Path:
    steering_dir = base_dir / model / "steering"
    candidates: list[tuple[float, Path]] = []
    for agg_path in steering_dir.glob(
        "steering-hidden-*-wildguard-judge/aggregate_result.json"
    ):
        ok, mtime = _aggregate_matches(agg_path, target, dataset, alpha)
        if ok:
            candidates.append((mtime, agg_path))
    if not candidates:
        raise FileNotFoundError(
            f"No aggregate_result.json for model={model} target={target} "
            f"dataset={dataset} alpha={alpha} under {steering_dir}"
        )
    _, path = max(candidates, key=lambda x: x[0])
    return path


def _load_cell_info(
    base_dir: Path, model: str, target: str, dataset: str,
    steering_alpha: float | None = None,
) -> tuple:
    """Return (alphas, layers, baseline, idx) from aggregate JSON.

    If steering_alpha is given, only that alpha is kept (plus baseline).
    """
    with _compose_agg_path(base_dir, model, target, dataset, steering_alpha).open() as f:
        agg = json.load(f)
    steered  = [r for r in agg["by_target_layer_alpha"] if r["target"] == target]
    baseline = next(
        (r for r in agg["by_target_layer_alpha"] if r["target"] == "baseline"), None
    )
    if steering_alpha is not None:
        steered = [r for r in steered if r["alpha"] == steering_alpha]
    alphas = sorted({r["alpha"] for r in steered})
    layers = _sorted_layer_names({r["layer_group_name"] for r in steered})
    idx    = {(r["layer_group_name"], r["alpha"]): r for r in steered}
    return alphas, layers, baseline, idx


def _draw_subplot(
    ax,
    rows_for_cell: list[dict],
    layers: list[str],
    baseline,
    show_legend: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    tick_fs: int = 9,
    axis_fs: int = 10,
    plot_labels: list[str] | None = None,
) -> None:
    lookup = {r["layer_group_name"]: r for r in rows_for_cell}
    x = np.arange(len(layers))
    for label in (plot_labels or PLOT_LABEL_ORDER):
        y = [lookup[lg][label]["percentage"] if lg in lookup else np.nan for lg in layers]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4,
                color=LABEL_COLORS[label], zorder=3)

    ax.set_xlim(-0.5, len(layers) - 0.5)
    step  = 4
    ticks = list(range(0, len(layers), step))
    if ticks and ticks[-1] != len(layers) - 1:
        if len(layers) - 1 - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(len(layers) - 1)
    ax.set_xticks(ticks)
    if show_xlabel:
        ax.set_xticklabels(
            [layers[i].replace("L", "") for i in ticks],
            rotation=0, ha="center", fontsize=tick_fs,
        )
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=tick_fs)
    if show_ylabel:
        ax.set_ylabel("Refusal Rate (%)", fontsize=axis_fs, labelpad=8)
        ax.yaxis.set_label_coords(-0.18, 0.5)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_merged_subplot(
    ax,
    harm_idx: dict,
    ref_idx: dict,
    layers: list[str],
    alpha: float,
    bridge_layers: tuple[int, int] | None = None,
    y_max: float = 100.0,
    show_xlabel: bool = False,
    show_ylabel: bool = True,
    tick_fs: int = 9,
    axis_fs: int = 10,
) -> None:
    x = np.arange(len(layers))

    if bridge_layers is not None:
        b_start, b_end = bridge_layers
        bridge_names   = {f"L{i}" for i in range(b_start, b_end + 1)}
        xs_bridge      = [xi for xi, lg in enumerate(layers) if lg in bridge_names]
        if xs_bridge:
            ax.axvspan(
                xs_bridge[0] - 0.5,
                xs_bridge[-1] + 0.5,
                facecolor=BRIDGE_COLOR,
                edgecolor="none",
                alpha=BRIDGE_ALPHA,
                zorder=0,
            )

    y_data: dict[str, np.ndarray] = {}
    for target, idx in [("harmfulness", harm_idx), ("refusal", ref_idx)]:
        y_data[target] = np.array(
            [
                idx[(lg, alpha)]["refusal"]["percentage"]
                if (lg, alpha) in idx else np.nan
                for lg in layers
            ],
            dtype=float,
        )

    _markers = {"harmfulness": "o", "refusal": "^"}
    for target in ["harmfulness", "refusal"]:
        ax.plot(
            x,
            y_data[target],
            marker=_markers[target],
            linewidth=2.4,
            markersize=5.2,
            color=STEER_COLORS[target],
            zorder=3,
        )

    ax.set_ylim(0, y_max)

    # Decide horizontal push direction to separate overlapping annotations.
    # When the two peaks are far apart (>20 layers) they don't need lateral
    # separation — center each annotation directly above/below its peak.
    harm_arr = y_data["harmfulness"]
    ref_arr  = y_data["refusal"]
    harm_peak_x = int(np.nanargmax(harm_arr)) if not np.isnan(harm_arr).all() else 0
    ref_peak_x  = int(np.nanargmax(ref_arr))  if not np.isnan(ref_arr).all()  else 0
    peak_distance = abs(ref_peak_x - harm_peak_x)
    if peak_distance > 20:
        # harmfulness: directly above; refusal: push LEFT into clear space before the spike
        _x_offsets = {"harmfulness": 0, "refusal": -28}
    elif harm_peak_x <= ref_peak_x:
        _x_offsets = {"harmfulness": -20, "refusal": 20}
    else:
        _x_offsets = {"harmfulness": 20, "refusal": -20}

    for target in ["harmfulness", "refusal"]:
        y_arr = y_data[target]
        if not np.isnan(y_arr).all():
            peak_x    = int(np.nanargmax(y_arr))
            peak_y    = float(y_arr[peak_x])
            layer_num = layers[peak_x].replace("L", "")
            color     = STEER_COLORS[target]
            # 0.92 threshold: only peaks very close to y_max go below; others go above
            offset_y, va = (-14, "top") if peak_y > y_max * 0.92 else (13, "bottom")
            ax.annotate(
                f"L{layer_num}",
                xy=(peak_x, peak_y),
                xytext=(_x_offsets[target], offset_y),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=tick_fs, color=color, fontweight="bold",
                fontfamily=FONT_NAME,
                clip_on=False, zorder=6,
            )

    ax.set_xlim(-0.5, len(layers) - 0.5)
    step  = 4
    ticks = list(range(0, len(layers), step))
    if ticks and ticks[-1] != len(layers) - 1:
        if len(layers) - 1 - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(len(layers) - 1)
    ax.set_xticks(ticks)
    if show_xlabel:
        ax.set_xticklabels(
            [layers[i].replace("L", "") for i in ticks],
            rotation=0, ha="center", fontsize=tick_fs,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", labelsize=tick_fs, length=4.0, width=1.0, pad=1.5)
    if show_ylabel:
        ax.set_ylabel("Refusal Rate (%)", fontsize=axis_fs, labelpad=3)
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_composed_grid(
    fig,
    outer_gs,
    models,
    col_keys,
    cell_info,
    TICK_FS: int = 11,
    AXIS_FS: int = 13,
) -> None:
    n_macro_r = len(models)
    n_alphas  = len(cell_info[(models[0][0], col_keys[0][0])][0])

    for ri, (m, _) in enumerate(models):
        for ci, (col_key, _) in enumerate(col_keys):
            alphas, layers, baseline, idx = cell_info[(m, col_key)]
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                n_alphas, 1, subplot_spec=outer_gs[ri, ci], hspace=0.12,
            )
            for ai, alpha in enumerate(alphas):
                ax            = fig.add_subplot(inner_gs[ai])
                cell_rows     = [idx[(lg, alpha)] for lg in layers if (lg, alpha) in idx]
                is_last_alpha = (ai == n_alphas - 1)
                is_first_col  = (ci == 0)
                is_bottom     = (ri == n_macro_r - 1) and is_last_alpha
                _draw_subplot(
                    ax, cell_rows, layers, baseline,
                    show_legend=False,
                    show_xlabel=is_last_alpha,
                    show_ylabel=is_first_col,
                    tick_fs=TICK_FS, axis_fs=AXIS_FS,
                    plot_labels=COMPOSE_LABEL_ORDER,
                )
                if is_bottom:
                    ax.set_xlabel("Steered Layer", fontsize=AXIS_FS)


def _add_col_headers(fig, outer_gs, labels: list[str], y_frac: float, fs: int = 12) -> None:
    for ci, lbl in enumerate(labels):
        pos      = outer_gs[0, ci].get_position(fig)
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, y_frac, lbl, fontsize=fs, fontweight="bold",
                 ha="center", va="center")


def _add_row_labels(fig, outer_gs, models, fs: int = 12) -> None:
    for ri, (_, m_label) in enumerate(models):
        pos      = outer_gs[ri, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        offset   = 0.105 + min(0.035, max(0, len(m_label) - 14) * 0.004)
        fig.text(
            max(0.045, pos.x0 - offset), y_center, m_label,
            fontsize=fs, fontweight="bold",
            rotation=90, ha="center", va="center",
        )


# ===========================================================================
# Clean-projection helpers
# ===========================================================================

DATASETS_PROJ = [
    ("advbench",              "AdvBench",             "#c43b3b"),
    ("alpaca",                "Alpaca",               "#2f7d4f"),
    ("misrepresentation",     "Misrepresentation",    "#3b66c4"),
    ("authority_endorsement", "Authority endorsement","#8b5fbf"),
    ("expert_endorsement",    "Expert endorsement",   "#d28c2d"),
]

FIG_CONFIG_PROJ = {
    "fig7": {"measure": "harmfulness_tinst", "ylabel": "Harmfulness Projection"},
    "fig8": {"measure": "refusal_tpost",     "ylabel": "Refusal Projection"},
}

def _find_clean_projection_json(model_dir: Path, measure: str) -> Path:
    candidates = [
        p for p in model_dir.rglob(f"{measure}/clean/clean-*.json")
        if "injection" not in p.name and "layerwise" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No clean projection JSON for measure={measure} under {model_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _figure_limits(summaries: dict) -> tuple[float, float]:
    values: list[float] = []
    for ds_key, _, _ in DATASETS_PROJ:
        if ds_key not in summaries:
            continue
        means = np.array(summaries[ds_key]["all_layer_mean"], dtype=np.float64)
        stds  = np.array(summaries[ds_key]["all_layer_std"],  dtype=np.float64)
        valid = np.isfinite(means - stds) & np.isfinite(means + stds)
        values.extend((means - stds)[valid].tolist())
        values.extend((means + stds)[valid].tolist())
    if not values:
        return -1.0, 1.0
    ymin   = float(np.nanmin(values))
    ymax   = float(np.nanmax(values))
    margin = max((ymax - ymin) * 0.08, 0.05)
    return ymin - margin, ymax + margin


# ===========================================================================
# Render functions
# ===========================================================================

# ---------------------------------------------------------------------------
# Fig2 — Coupling heatmap, 2×2
# ---------------------------------------------------------------------------

def render_fig2(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    # ---- font sizes for Fig2 (change here without touching Fig5) ----
    TICK_FS        = 16
    AXIS_FS        = 22
    CBAR_TICK_FS   = 16
    CBAR_LABEL_FS  = 22
    ROW_LABEL_FS   = 28
    PANEL_LABEL_FS = 26
    COL_HEADER_FS  = 22

    paths: list[Path] = []
    panel_captions: list[str] = []

    for alias, dir_name, coupling_key, _, panel_label in APPENDIX_MODELS:
        paths.append(
            latest_coupling_json(
                base_dir / dir_name,
                coupling_key,
                "alpaca_common",
                alpha,
            )
        )
        panel_captions.append(panel_label)

    paths_2x2 = [paths[:2], paths[2:]]
    panel_captions_2x2 = [panel_captions[:2], panel_captions[2:]]

    plot_coupling_grid(
        paths=paths_2x2,
        output=output,
        dpi=dpi,
        model_by_row=False,
        show_panel_captions=True,
        panel_captions=panel_captions_2x2,
        tick_fs=TICK_FS, axis_fs=AXIS_FS,
        cbar_tick_fs=CBAR_TICK_FS, cbar_label_fs=CBAR_LABEL_FS,
        row_label_fs=ROW_LABEL_FS, panel_label_fs=PANEL_LABEL_FS,
        col_header_fs=COL_HEADER_FS,
    )


# ---------------------------------------------------------------------------
# Fig3 — Steering (Alpaca), vertical 4×1
# ---------------------------------------------------------------------------

def render_fig3(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    models = [(dir_name, panel_label) for _, dir_name, _, _, panel_label in APPENDIX_MODELS]

    cell_info = {
        (m, t): _load_cell_info(base_dir, m, t, "alpaca", steering_alpha=alpha)
        for m, _ in models
        for t in ["harmfulness", "refusal"]
    }

    n_alphas   = len(cell_info[(models[0][0], "harmfulness")][0])
    max_layers = max(
        len(cell_info[(m, t)][1])
        for m, _ in models for t in ["harmfulness", "refusal"]
    )
    TICK_FS, AXIS_FS, LEG_FS = 20, 22, 14

    fig_w    = max(10.5, max_layers * 0.20)
    fig_h    = 3.55 * n_alphas * len(models) + 0.7
    fig      = plt.figure(figsize=(fig_w, fig_h))
    leg_frac = 0.62 / fig_h

    # Keep physical left label area ~2.5 inches regardless of figure width
    left = min(0.240, 2.5 / fig_w)

    outer_gs = gridspec.GridSpec(
        len(models), 1, figure=fig,
        top=1.0 - leg_frac, bottom=0.09,
        left=left, right=0.975, hspace=0.48,
    )

    n_macro_r = len(models)
    for ri, (m, _) in enumerate(models):
        harm_alphas, harm_layers, _, harm_idx = cell_info[(m, "harmfulness")]
        _ref_alphas, ref_layers,  _, ref_idx  = cell_info[(m, "refusal")]
        layers = _sorted_layer_names(set(harm_layers) | set(ref_layers))
        alphas_list = harm_alphas
        bridge = BRIDGE_LAYERS_FIG3.get(m)

        all_vals = [
            idx[(lg, a)]["refusal"]["percentage"]
            for a in alphas_list
            for target, idx in [("harmfulness", harm_idx), ("refusal", ref_idx)]
            for lg in layers
            if (lg, a) in idx
        ]
        all_vals = [v for v in all_vals if not math.isnan(v)]
        if all_vals:
            top   = max(all_vals)
            y_max = float(math.ceil(top / 5) * 5 if top <= 30 else math.ceil(top / 10) * 10)
            y_max = max(y_max, 10.0)
        else:
            y_max = 100.0

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_alphas, 1, subplot_spec=outer_gs[ri, 0], hspace=0.12,
        )
        for ai, a in enumerate(alphas_list):
            ax            = fig.add_subplot(inner_gs[ai])
            is_last_alpha = (ai == n_alphas - 1)
            is_bottom     = (ri == n_macro_r - 1) and is_last_alpha
            _draw_merged_subplot(
                ax, harm_idx, ref_idx, layers, a,
                bridge_layers=bridge, y_max=y_max,
                show_xlabel=is_last_alpha, show_ylabel=False,
                tick_fs=TICK_FS, axis_fs=AXIS_FS,
            )
            if is_bottom:
                ax.set_xlabel("Steered Layer", fontsize=AXIS_FS, labelpad=4)

    handles = [
        plt.Line2D([0], [0], color=STEER_COLORS["harmfulness"], linewidth=2.4,
                   marker="o", markersize=6, label=STEER_LABELS["harmfulness"]),
        plt.Line2D([0], [0], color=STEER_COLORS["refusal"], linewidth=2.4,
                   marker="^", markersize=6, label=STEER_LABELS["refusal"]),
    ]
    if any(v is not None for v in BRIDGE_LAYERS_FIG3.values()):
        handles.append(
            mpatches.Patch(facecolor=BRIDGE_COLOR, edgecolor="none", alpha=BRIDGE_ALPHA, label="Bridge")
        )
    fig.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.545, 0.995),
        ncol=len(handles), fontsize=LEG_FS,
        frameon=True, fancybox=False,
        edgecolor="lightgray", facecolor="white", framealpha=1.0,
        handlelength=2.0, columnspacing=1.4, borderpad=0.45, borderaxespad=0.0,
    )
    # Place row labels and the shared y-axis title at equal intervals.
    # x_tick_offset: estimated physical distance (inches) from the axes left
    # edge to the tick-label center (pad + half of "100" glyph width at 15pt).
    x_model_offset = 1.55
    x_tick_offset  = 0.17
    x_rate_offset  = (x_model_offset + x_tick_offset) / 2  # ≈ 0.86 in

    for ri, (_, m_label) in enumerate(models):
        pos      = outer_gs[ri, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            max(0.025, pos.x0 - x_model_offset / fig_w),
            y_center,
            m_label,
            fontsize=26,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
        fig.text(
            max(0.040, pos.x0 - x_rate_offset / fig_w),
            y_center,
            "Refusal Rate (%)",
            fontsize=AXIS_FS,
            rotation=90,
            ha="center",
            va="center",
        )

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ---------------------------------------------------------------------------
# Fig4 — Activation patching heatmap, vertical 4×2
# ---------------------------------------------------------------------------

def render_fig4(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    # ---- font sizes for Fig4 (change here without touching Fig4_2) ----
    TICK_FS       = 16
    TITLE_FS      = 22
    AXIS_FS       = 22
    ROW_LABEL_FS  = 28
    CBAR_TICK_FS  = 16
    CBAR_LABEL_FS = 22

    ap_models = [
        (alias, dir_name, panel_label)
        for alias, dir_name, _, _, panel_label in APPENDIX_MODELS
    ]
    paths = {
        alias: latest_patching_json(base_dir, dir_name, alpha)
        for alias, dir_name, _, _, _ in APPENDIX_MODELS
    }
    render_patching_heatmap_grid(
        paths, output, dpi, models=ap_models,
        tick_fs=TICK_FS, title_fs=TITLE_FS, axis_fs=AXIS_FS,
        row_label_fs=ROW_LABEL_FS, cbar_tick_fs=CBAR_TICK_FS, cbar_label_fs=CBAR_LABEL_FS,
    )


# ---------------------------------------------------------------------------
# Fig4_2 — Behavioral grid, vertical 4×1
# ---------------------------------------------------------------------------

def render_fig4_2(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    # ---- font sizes for Fig4_2 (change here without touching Fig4) ----
    AXIS_FS      = 18
    YLABEL_FS    = 18
    TICK_FS      = 15
    ROW_LABEL_FS = 26
    LEGEND_FS    = 14

    ap_models = [
        (alias, dir_name, panel_label)
        for alias, dir_name, _, _, panel_label in APPENDIX_MODELS
    ]
    paths = {
        alias: latest_patching_behavior_json(base_dir, dir_name, alpha)
        for alias, dir_name, _, _, _ in APPENDIX_MODELS
    }
    render_patching_behavior_grid(
        paths, output, dpi, models=ap_models,
        axis_fs=AXIS_FS, ylabel_fs=YLABEL_FS, tick_fs=TICK_FS, row_label_fs=ROW_LABEL_FS, legend_fs=LEGEND_FS,
    )


# ---------------------------------------------------------------------------
# Fig5 — Coupling heatmap multi-dataset, vertical 4×3
# ---------------------------------------------------------------------------

def render_fig5(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    # ---- font sizes for Fig5 (change here without touching Fig2) ----
    TICK_FS        = 16
    AXIS_FS        = 22
    CBAR_TICK_FS   = 16
    CBAR_LABEL_FS  = 22
    ROW_LABEL_FS   = 28
    PANEL_LABEL_FS = 26
    COL_HEADER_FS  = 22

    row_paths: list[list[Path]] = []
    row_captions: list[str]     = []
    for alias, dir_name, coupling_key, _, panel_label in APPENDIX_MODELS:
        model_dir = base_dir / dir_name
        ds_paths  = [
            latest_coupling_json(model_dir, coupling_key, ds_key, alpha)
            for ds_key, _ in FIG5_DATASETS
        ]
        row_paths.append(ds_paths)
        row_captions.append(panel_label)

    plot_coupling_grid(
        paths=row_paths,
        output=output,
        dpi=dpi,
        model_by_row=True,
        show_panel_captions=False,
        column_labels=[ds_label for _, ds_label in FIG5_DATASETS],
        row_captions=row_captions,
        tick_fs=TICK_FS, axis_fs=AXIS_FS,
        cbar_tick_fs=CBAR_TICK_FS, cbar_label_fs=CBAR_LABEL_FS,
        row_label_fs=ROW_LABEL_FS, panel_label_fs=PANEL_LABEL_FS,
        col_header_fs=COL_HEADER_FS,
        per_row_scale=True,
    )


# ---------------------------------------------------------------------------
# Fig6 — Adversarial steering, vertical 4×3
# ---------------------------------------------------------------------------

def render_fig6(base_dir: Path, output: Path, dpi: int, alpha: float) -> None:
    models   = [(dir_name, panel_label) for _, dir_name, _, _, panel_label in APPENDIX_MODELS]
    datasets = FIG5_DATASETS

    cell_info = {
        (m, ds_key): _load_cell_info(base_dir, m, "harmfulness", ds_key, steering_alpha=alpha)
        for m, _ in models
        for ds_key, _ in datasets
    }

    TICK_FS      = 13
    AXIS_FS      = 20
    HEADER_FS    = 20
    ROW_LABEL_FS = 22

    n_rows = len(models)
    n_cols = len(datasets)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.2, n_rows * 3.5),
        squeeze=False,
    )

    for ri, (m, m_label) in enumerate(models):
        for ci, (ds_key, ds_label) in enumerate(datasets):
            ax        = axes[ri][ci]
            is_bottom = (ri == n_rows - 1)
            is_left   = (ci == 0)

            alphas_list, layers, _, idx = cell_info[(m, ds_key)]
            n_layers = len(layers)

            if alphas_list:
                a = alphas_list[0]
                y_vals = np.array(
                    [idx[(lg, a)]["refusal"]["percentage"] if (lg, a) in idx else np.nan
                     for lg in layers],
                    dtype=float,
                )
                ax.plot(
                    np.arange(n_layers), y_vals,
                    marker="o", linewidth=2.0, markersize=4,
                    color=LABEL_COLORS["refusal"], zorder=3,
                )

            # Y axis: fixed 0–100 with clean ticks
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])

            # X axis: adaptive step so we get ~8–11 ticks regardless of model size
            step = 4 if n_layers <= 40 else 8
            ticks = list(range(0, n_layers, step))
            last  = n_layers - 1
            if ticks[-1] != last:
                if last - ticks[-1] <= step // 2:
                    ticks[-1] = last   # replace last tick to avoid crowding
                else:
                    ticks.append(last)
            ax.set_xticks(ticks)
            ax.set_xlim(-0.5, last + 0.5)

            # Show tick labels on every row
            ax.set_xticklabels(
                [layers[i].replace("L", "") for i in ticks],
                fontsize=TICK_FS,
            )
            # "Steered Layer" axis label only on the bottom row
            if is_bottom:
                ax.set_xlabel("Steered Layer", fontsize=AXIS_FS, labelpad=6)

            ax.tick_params(axis="both", labelsize=TICK_FS, length=4, width=1.0, pad=2)

            if is_left:
                ax.set_ylabel("Refusal Rate (%)", fontsize=AXIS_FS, labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            ax.grid(axis="y", alpha=0.22, linewidth=0.8, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(1.0)

            if ri == 0:
                ax.set_title(ds_label, fontsize=HEADER_FS, fontweight="bold", pad=10)

        # Row label – bold, rotated; placed close to the y-axis label
        axes[ri][0].text(
            -0.29, 0.5, m_label,
            transform=axes[ri][0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=ROW_LABEL_FS, fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.14, right=0.99,
        bottom=0.08, top=0.94,
        wspace=0.30, hspace=0.40,
    )

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ---------------------------------------------------------------------------
# Fig7 / Fig8 — Clean projection, vertical 4×1
# ---------------------------------------------------------------------------

def _render_clean_projection(
    measure: str,
    ylabel: str,
    base_dir: Path,
    output: Path,
    dpi: int,
    # --- font / layout knobs ---
    tick_fs: int,
    axis_fs: int,
    title_fs: int,
    legend_fs: int,
) -> None:
    paths:  list[Path] = []
    labels: list[str]  = []
    for alias, dir_name, _, _, panel_label in APPENDIX_MODELS:
        paths.append(_find_clean_projection_json(base_dir / dir_name, measure))
        labels.append(panel_label)

    summaries_list = [_load_json(p).get("summaries", {}) for p in paths]
    n = len(APPENDIX_MODELS)

    fig, axes = plt.subplots(
        n, 1, figsize=(16.5, 5.2 * n), sharey=False, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, summaries in zip(axes, summaries_list):
        ymin, ymax = _figure_limits(summaries)
        max_layers = 0

        for ds_key, ds_label, color in DATASETS_PROJ:
            if ds_key not in summaries:
                continue
            means = np.array(summaries[ds_key]["all_layer_mean"], dtype=np.float64)
            stds  = np.array(summaries[ds_key]["all_layer_std"],  dtype=np.float64)
            layers = np.arange(len(means))
            max_layers = max(max_layers, len(means))
            ax.plot(layers, means, label=ds_label, color=color, linewidth=2.4)
            ax.fill_between(layers, means - stds, means + stds,
                            color=color, alpha=0.13, linewidth=0)

        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.75)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.5, max_layers - 0.5)
        ax.set_xticks(_layer_ticks(max_layers))
        ax.set_xlabel("Layer", fontsize=axis_fs, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=axis_fs, labelpad=8)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.tick_params(axis="both", labelsize=tick_fs, length=4.0, width=1.0, pad=3.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)

    LEFT, RIGHT = 0.14, 0.985
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.955),
        bbox_transform=fig.transFigure,
        ncol=5, fontsize=legend_fs,
        frameon=True, fancybox=False,
        edgecolor="lightgray", facecolor="white", framealpha=1.0,
        handlelength=1.6, columnspacing=1.4,
        handletextpad=0.45, borderpad=0.5, borderaxespad=0.0,
    )
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=0.06, top=0.91, hspace=0.40)

    for ax, panel_label in zip(axes, labels):
        pos      = ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.04, y_center, panel_label,
                 fontsize=title_fs, fontweight="bold",
                 rotation=90, ha="center", va="center")

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def render_fig7(base_dir: Path, output: Path, dpi: int) -> None:
    # ---- font sizes for Fig7 (change here without touching Fig8) ----
    TICK_FS   = 20
    AXIS_FS   = 28
    TITLE_FS  = 32
    LEGEND_FS = 23
    _render_clean_projection(
        measure="harmfulness_tinst", ylabel="Harmfulness Projection",
        base_dir=base_dir, output=output, dpi=dpi,
        tick_fs=TICK_FS, axis_fs=AXIS_FS, title_fs=TITLE_FS, legend_fs=LEGEND_FS,
    )


def render_fig8(base_dir: Path, output: Path, dpi: int) -> None:
    # ---- font sizes for Fig8 (change here without touching Fig7) ----
    TICK_FS   = 20
    AXIS_FS   = 28
    TITLE_FS  = 32
    LEGEND_FS = 23
    _render_clean_projection(
        measure="refusal_tpost", ylabel="Refusal Projection",
        base_dir=base_dir, output=output, dpi=dpi,
        tick_fs=TICK_FS, axis_fs=AXIS_FS, title_fs=TITLE_FS, legend_fs=LEGEND_FS,
    )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Appendix B figures for "
            "LLaMA 3.1 70B, Qwen 2.5 14B, Qwen 2.5 32B, Qwen 2.5 72B."
        )
    )
    parser.add_argument("--base-dir",   default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--dpi",        type=int,   default=200)
    parser.add_argument(
        "--steering-alpha", type=float, default=6.0,
        help="Steering coefficient to use when loading steering / coupling / patching data "
             "(default: 6.0).  Coupling JSON filenames embed this as 'alpha_N'.",
    )
    parser.add_argument(
        "--figs", default="all",
        help=(
            "Comma-separated list of figures to render "
            "(1, 2, 4, 5, 6, 7, 8, 9) or 'all'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    alpha      = args.steering_alpha
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    requested = (
        {"1", "2", "4", "5", "6", "7", "8", "9"}
        if args.figs.strip().lower() == "all"
        else {f.strip() for f in args.figs.split(",")}
    )

    print(f"Steering alpha = {alpha}")

    if "1" in requested:
        print("\n=== Fig1 (coupling heatmap, horizontal 1x4) ===")
        try:    render_fig2(base_dir, output_dir / "Fig1.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig1 failed: {e}")

    if "2" in requested:
        print("\n=== Fig2 (steering Alpaca, vertical 4x1) ===")
        try:    render_fig3(base_dir, output_dir / "Fig2.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig2 failed: {e}")

    if "4" in requested:
        print("\n=== Fig4 (behavioral grid, vertical 4x1) ===")
        try:    render_fig4_2(base_dir, output_dir / "Fig4.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig4 failed: {e}")

    if "5" in requested:
        print("\n=== Fig5 (activation patching heatmap, vertical 4x2) ===")
        try:    render_fig4(base_dir, output_dir / "Fig5.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig5 failed: {e}")

    if "6" in requested:
        print("\n=== Fig6 (coupling heatmap multi-dataset, vertical 4x3) ===")
        try:    render_fig5(base_dir, output_dir / "Fig6.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig6 failed: {e}")

    if "7" in requested:
        print("\n=== Fig7 (adversarial steering, vertical 4x3) ===")
        try:    render_fig6(base_dir, output_dir / "Fig7.png", dpi, alpha)
        except Exception as e: print(f"[WARN] Fig7 failed: {e}")

    if "8" in requested:
        print("\n=== Fig8 (harmfulness projection, vertical 4x1) ===")
        try:    render_fig7(base_dir, output_dir / "Fig8.png", dpi)
        except Exception as e: print(f"[WARN] Fig8 failed: {e}")

    if "9" in requested:
        print("\n=== Fig9 (refusal projection, vertical 4x1) ===")
        try:    render_fig8(base_dir, output_dir / "Fig9.png", dpi)
        except Exception as e: print(f"[WARN] Fig9 failed: {e}")

    print(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
