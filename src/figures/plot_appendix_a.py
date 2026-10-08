#!/usr/bin/env python3
"""
Appendix A figure generator — fully self-contained.

Generates Fig2, Fig3, Fig3_2, Fig4, Fig4_2, Fig5, Fig6, Fig7, Fig8
for Gemma 2 9B, Falcon3 7B, OLMo 2 7B, Qwen 3.5 9B → out_pt/Figure_Appendix_A/.

Layout:
  - Fig2 : 2 × 2 grid (one model per panel)
  - Fig3_2 : horizontal (1 row × N cols, one model per col)
  - Fig3, Fig4, Fig4_2, Fig5, Fig6, Fig7, Fig8 : vertical (N rows, one per model)
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
DEFAULT_BASE_DIR = SCRIPT_DIR / "out_pt"
DEFAULT_FIGURE_DIR = DEFAULT_BASE_DIR / "Figure_Appendix_A"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "steered_tinst_restore_refusal_projection"

# ---------------------------------------------------------------------------
# Model / figure config
# ---------------------------------------------------------------------------
# (alias, model_dir_name, coupling_key, display_name, panel_label)
APPENDIX_MODELS = [
    ("gemma",   "gemma-2-9b-it",          "gemma2",  "Gemma 2 9B", "(a) Gemma 2 9B"),
    ("falcon3", "falcon3-7b-instruct",     "falcon3", "Falcon3 7B", "(b) Falcon3 7B"),
    ("olmo2",   "olmo-2-1124-7b-instruct", "olmo2",   "OLMo 2 7B",  "(c) OLMo 2 7B"),
    ("qwen35",  "qwen3.5-9b",              "qwen35",  "Qwen 3.5 9B", "(d) Qwen 3.5 9B"),
]

# Bridge layer ranges (inclusive, 0-based) for Fig3 shading
BRIDGE_LAYERS_FIG3: dict[str, tuple[int, int]] = {
    "gemma-2-9b-it":           (17, 20),
    "falcon3-7b-instruct":     (12, 17),
    "olmo-2-1124-7b-instruct": (13, 16),
    "qwen3.5-9b":              (11, 16),
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


def _layer_ticks(size: int) -> list[int]:
    step = 4
    ticks = list(range(0, size, step))
    last = size - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
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
# Coupling helpers  (from plot_coupling_paper_figures.py)
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

_COUPLING_TICK_FS        = 16
_COUPLING_AXIS_FS        = 22
_COUPLING_CBAR_TICK_FS   = 16
_COUPLING_CBAR_LABEL_FS  = 22
_COUPLING_ROW_LABEL_FS   = 28
_COUPLING_PANEL_LABEL_FS = 26


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
    inp = meta.get("input_path")
    if inp:
        key = Path(inp).stem.replace("-", "_")
        return DATASET_ALIASES.get(key, key)
    label = str(meta.get("label") or "").replace("-", "_")
    for alias, canonical in DATASET_ALIASES.items():
        if label == alias or label.endswith(f"_{alias}"):
            return canonical
    return label


def latest_coupling_json(search_root: Path, model_key: str, dataset_key: str) -> Path:
    normalized = DATASET_ALIASES.get(dataset_key, dataset_key)
    candidates: list[Path] = []
    for path in search_root.rglob("harm-to-refusal-coupling-hidden-alpha_3-*.json"):
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
            f"No coupling JSON for model={model_key} dataset={dataset_key} under {search_root}"
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
    panel_captions: list[str] | None = None,
    independent_panels: bool = False,
) -> None:
    maps   = [[_load_coupling_values(p) for p in row] for row in paths]
    n_rows = len(paths)
    n_cols = len(paths[0])

    if independent_panels:
        limits    = [[_sym_limits([maps[ri][ci]]) for ci in range(n_cols)] for ri in range(n_rows)]
        limit_for = lambda ri, ci: limits[ri][ci]
    elif model_by_row:
        limits    = [_sym_limits(row) for row in maps]
        limit_for = lambda ri, ci: limits[ri]
    else:
        limits    = [_sym_limits([maps[ri][ci] for ri in range(n_rows)]) for ci in range(n_cols)]
        limit_for = lambda ri, ci: limits[ci]

    if independent_panels:
        fig_width, fig_height = 3.95 * n_cols + 1.45, 3.95 * n_rows + 1.10
    else:
        panel_size = 4.35
        fig_width  = panel_size * n_cols + (1.35 if model_by_row else 1.55)
        fig_height = panel_size * n_rows + (0.55 if n_rows == 1 else 0.25)
    fig, axes  = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    cbar_images: list = [None] * (n_rows if model_by_row else n_cols)
    panel_images: dict[tuple[int, int], object] = {}
    for ri, row in enumerate(maps):
        for ci, values in enumerate(row):
            ax = axes[ri][ci]
            vmin, vmax = limit_for(ri, ci)
            image = ax.imshow(np.ma.masked_invalid(values),
                              origin="lower", aspect="equal",
                              cmap="coolwarm", vmin=vmin, vmax=vmax)
            if independent_panels:
                panel_images[(ri, ci)] = image
            else:
                cbar_images[ri if model_by_row else ci] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            ax.set_xticks(_layer_ticks(values.shape[1]))
            ax.set_yticks(_layer_ticks(values.shape[0]))
            ax.tick_params(axis="both", labelsize=_COUPLING_TICK_FS,
                           length=4.0, width=1.0, pad=3.0)
            ax.grid(False)
            if ci == 0:
                ax.set_ylabel("Injection Layer",   fontsize=_COUPLING_AXIS_FS, labelpad=10)
            if ri == n_rows - 1 or independent_panels:
                ax.set_xlabel("Downstream Layer",  fontsize=_COUPLING_AXIS_FS, labelpad=10)
            if column_labels and ri == 0:
                ax.set_title(column_labels[ci], fontsize=_COUPLING_AXIS_FS,
                             fontweight="bold", pad=14)

        if model_by_row:
            axes[ri][0].text(
                -0.36, 0.5,
                row_captions[ri] if row_captions else f"({chr(ord('a') + ri)})",
                transform=axes[ri][0].transAxes,
                rotation=90, ha="center", va="center",
                fontsize=_COUPLING_ROW_LABEL_FS, fontweight="bold",
            )
        elif show_panel_captions and not independent_panels:
            for ci in range(n_cols):
                axes[-1][ci].text(
                    0.5, -0.27,
                    panel_captions[ci] if panel_captions else f"({chr(ord('a') + ci)})",
                    transform=axes[-1][ci].transAxes,
                    ha="center", va="top",
                    fontsize=_COUPLING_PANEL_LABEL_FS, fontweight="bold",
                )

    if independent_panels and show_panel_captions:
        idx = 0
        for ri in range(n_rows):
            for ci in range(n_cols):
                axes[ri][ci].text(
                    0.5, -0.22,
                    panel_captions[idx] if panel_captions else f"({chr(ord('a') + idx)})",
                    transform=axes[ri][ci].transAxes,
                    ha="center", va="top",
                    fontsize=_COUPLING_PANEL_LABEL_FS, fontweight="bold",
                    clip_on=False,
                )
                idx += 1

    if independent_panels:
        fig.subplots_adjust(left=0.10, right=0.95, bottom=0.09, top=0.97, wspace=0.36, hspace=0.34)
    else:
        fig.subplots_adjust(
            left   = 0.205 if model_by_row else 0.115,
            right  = 0.905,
            bottom = 0.16 if n_rows == 1 else 0.08,
            top    = 0.92 if column_labels else 0.985,
            wspace = 0.45 if not model_by_row else 0.20,
            hspace = 0.02 if model_by_row else 0.16,
        )
    if independent_panels:
        for (ri, ci), image in panel_images.items():
            anchor = axes[ri][ci].get_position()
            cax = fig.add_axes([anchor.x1 + 0.006, anchor.y0, 0.013, anchor.height])
            cbar = fig.colorbar(image, cax=cax)
            cbar.ax.tick_params(labelsize=_COUPLING_CBAR_TICK_FS, length=4, width=1.0)
            if ci == n_cols - 1:
                cbar.set_label(r"$\Delta$ Refusal Projection",
                               fontsize=_COUPLING_CBAR_LABEL_FS, labelpad=7)
    else:
        for idx, image in enumerate(cbar_images):
            if image is None:
                continue
            anchor = (axes[idx][n_cols - 1] if model_by_row else axes[0][idx]).get_position()
            cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.016, anchor.height])
            cbar = fig.colorbar(image, cax=cax)
            cbar.ax.tick_params(labelsize=_COUPLING_CBAR_TICK_FS, length=4, width=1.0)
            if model_by_row or idx == len(cbar_images) - 1:
                cbar.set_label(r"$\Delta$ Refusal Projection",
                               fontsize=_COUPLING_CBAR_LABEL_FS, labelpad=14)

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ===========================================================================
# Activation-patching helpers  (from plot_path_patching_fig4.py)
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


def _candidate_patching_jsons(model_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for subdir, pattern in [
        ("activation_patching", "activation-patching-hidden-alpha_3-*.json"),
        ("path_patching",       "path-patching-hidden-alpha_3-*.json"),
    ]:
        for p in (model_dir / subdir).rglob(pattern):
            if "generation-for-judge" not in p.name:
                paths.append(p)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def latest_patching_json(base_dir: Path, model_dir_name: str) -> Path:
    matches = _candidate_patching_jsons(base_dir / model_dir_name)
    if not matches:
        raise FileNotFoundError(
            f"No activation-patching JSON under {base_dir / model_dir_name}"
        )
    return matches[-1]


def latest_patching_behavior_json(base_dir: Path, model_dir_name: str) -> Path:
    for p in reversed(_candidate_patching_jsons(base_dir / model_dir_name)):
        payload = _load_json(p)
        bv = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
        if bv and bv.get("experiments"):
            return p
    raise FileNotFoundError(
        f"No behavioral_validation JSON under {base_dir / model_dir_name}"
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
            image = ax.imshow(np.ma.masked_invalid(values),
                              origin="lower", aspect="equal",
                              cmap=cmap, vmin=vmin, vmax=vmax)
            row_images[ri] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            ax.set_xticks(_layer_ticks(values.shape[1]))
            ax.set_yticks(_layer_ticks(values.shape[0]))
            ax.tick_params(axis="both", labelsize=16, length=4, width=1.0, pad=3)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(1.2)
            if ri == 0:
                ax.set_title(variant_label, fontsize=22, pad=10)
            if ri == len(models) - 1:
                ax.set_xlabel("Downstream Layer", fontsize=22, labelpad=12)
            if ci == 0:
                ax.set_ylabel("Injection Layer",  fontsize=22, labelpad=12)

        axes[ri, 0].text(
            -0.36, 0.5, row_label,
            transform=axes[ri, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=28, fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.21, right=0.90, bottom=0.10, top=0.93, wspace=0.20, hspace=0.18,
    )
    for ri, image in row_images.items():
        anchor = axes[ri, -1].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.018, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=16, length=4, width=1.0)
        cbar.outline.set_linewidth(1.0)
        cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=22, labelpad=14)

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def render_patching_behavior_grid(
    paths: dict[str, Path],
    output: Path,
    dpi: int,
    models: list[tuple[str, str, str]],
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
        ax.set_ylabel(r"$\Delta$ Refusal Rate", fontsize=18, labelpad=9)
        if ri == len(models) - 1:
            ax.set_xlabel("Steered Layer", fontsize=18, labelpad=8)
        ax.tick_params(axis="both", labelsize=15, length=4, width=1.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)
        ax.text(
            -0.22, 0.5, row_label,
            transform=ax.transAxes,
            rotation=90, ha="center", va="center",
            fontsize=26, fontweight="bold",
        )
        if ri == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=14, ncol=2, handlelength=1.9)

    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.10, top=0.98, hspace=0.34)
    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ===========================================================================
# Steering helpers  (from judge_steering_wildguard.py)
# ===========================================================================

STEER_COLORS = {"harmfulness": "#d73027", "refusal": "#4575b4"}
STEER_LABELS = {"harmfulness": "Harmfulness Steering", "refusal": "Refusal Steering"}
BRIDGE_COLOR = "#DCEAF7"
LABEL_COLORS = {"refusal": "#d73027", "accept": "#1a9850", "other": "#cccccc"}
LABEL_TITLES = {"refusal": "Refusal", "accept": "Accept", "other": "Other"}
COMPOSE_LABEL_ORDER = ["refusal"]
PLOT_LABEL_ORDER    = ["refusal", "other"]

_COMPOSE_MODEL_DIR = {
    "llama": "llama-3.1-8b-instruct",
    "qwen":  "qwen2.5-7b-instruct",
}


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


def _aggregate_matches(aggregate_path: Path, target: str, dataset: str) -> tuple[bool, float]:
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
    if target not in {row.get("target") for row in aggregate.get("by_target_layer_alpha", [])}:
        return False, source_path.stat().st_mtime
    return True, source_path.stat().st_mtime


def _compose_agg_path(base_dir: Path, model: str, target: str, dataset: str) -> Path:
    steering_dir = base_dir / _COMPOSE_MODEL_DIR.get(model, model) / "steering"
    candidates: list[tuple[float, Path]] = []
    for agg_path in steering_dir.glob(
        "steering-hidden-*-wildguard-judge/aggregate_result.json"
    ):
        ok, mtime = _aggregate_matches(agg_path, target, dataset)
        if ok:
            candidates.append((mtime, agg_path))
    if not candidates:
        raise FileNotFoundError(
            f"No aggregate_result.json for model={model} target={target} "
            f"dataset={dataset} under {steering_dir}"
        )
    _, path = max(candidates, key=lambda x: x[0])
    return path


def _load_cell_info(
    base_dir: Path, model: str, target: str, dataset: str
) -> tuple:
    """Return (alphas, layers, baseline, idx) from aggregate JSON."""
    with _compose_agg_path(base_dir, model, target, dataset).open() as f:
        agg = json.load(f)
    steered  = [r for r in agg["by_target_layer_alpha"] if r["target"] == target]
    baseline = next(
        (r for r in agg["by_target_layer_alpha"] if r["target"] == "baseline"), None
    )
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
            ax.axvspan(xs_bridge[0] - 0.5, xs_bridge[-1] + 0.5,
                       facecolor=BRIDGE_COLOR, edgecolor="none", alpha=0.6, zorder=0)

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
        ax.plot(x, y_data[target], marker=_markers[target], linewidth=2.0, markersize=4,
                color=STEER_COLORS[target], zorder=3)

    ax.set_ylim(0, y_max)

    for target in ["harmfulness", "refusal"]:
        y_arr = y_data[target]
        if not np.isnan(y_arr).all():
            peak_x    = int(np.nanargmax(y_arr))
            peak_y    = float(y_arr[peak_x])
            layer_num = layers[peak_x].replace("L", "")
            color     = STEER_COLORS[target]
            offset_y, va = (-9, "top") if peak_y > y_max * 0.85 else (8, "bottom")
            ax.annotate(
                f"L{layer_num}",
                xy=(peak_x, peak_y),
                xytext=(0, offset_y),
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
    ax.tick_params(axis="y", labelsize=tick_fs)
    if show_ylabel:
        ax.set_ylabel("Refusal Rate (%)", fontsize=axis_fs)
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
# Tinst-restore helpers  (from steered_tinst_restore_refusal_projection_appendix.py)
# ===========================================================================

TINST_MODEL_CONFIGS: dict[str, dict] = {
    "gemma": {
        "display":    "Gemma 2 9B",
        "steer_layer": 12,
        "patch_spans": {
            "patch_bridge":  list(range(17, 21)),
            "patch_readout": list(range(25, 42)),
        },
    },
    "falcon3": {
        "display":    "Falcon 3 7B",
        "steer_layer": 3,
        "patch_spans": {
            "patch_bridge":  list(range(12, 18)),
            "patch_readout": list(range(18, 28)),
        },
    },
    "olmo2": {
        "display":    "OLMo 2 7B",
        "steer_layer": 12,
        "patch_spans": {
            "patch_bridge":  list(range(13, 17)),
            "patch_readout": list(range(17, 32)),
        },
    },
    "qwen35": {
        "display":    "Qwen 3.5 9B",
        "steer_layer": 7,
        "patch_spans": {
            "patch_bridge":  list(range(11, 17)),
            "patch_readout": list(range(17, 31)),
        },
    },
}


def _format_layer_span_label(model_name: str, condition: str, name: str) -> str:
    layers = TINST_MODEL_CONFIGS[model_name]["patch_spans"][condition]
    return f"{name}\n($l={layers[0]}$-${layers[-1]}$)"


def _plot_delta_drop_bars(
    ax,
    drops: np.ndarray,
    drop_stderrs: np.ndarray,
    title: str,
    colors: list[str],
    show_ylabel: bool,
    model_name: str,
) -> None:
    x            = np.arange(len(drops))
    delta_values = -drops
    bars = ax.bar(x, delta_values, color=colors, edgecolor="#20242b", linewidth=1.0, width=0.62)
    ax.axhline(0.0, color="#20242b", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(
        x,
        [
            _format_layer_span_label(model_name, "patch_bridge",  "Bridge"),
            _format_layer_span_label(model_name, "patch_readout", "Readout"),
        ],
    )
    ax.set_title(title, fontsize=26, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=18, length=4.0, width=1.0)
    if show_ylabel:
        ax.set_ylabel(r"$\Delta$ Refusal Projection", fontsize=22, labelpad=12)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.grid(axis="y", alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    val_range = max(float(np.abs(delta_values).max()), 0.1)
    text_offset = val_range * 0.07
    for bar, val in zip(bars, delta_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(val) - text_offset,
            f"{val:.2f}",
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="black",
        )


def _latest_tinst_json(results_dir: Path, model_alias: str) -> Path:
    candidates = sorted(
        (results_dir / model_alias).glob(
            f"{model_alias}-tinst-restore-final-tpost-refusal-*.json"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No tinst-restore JSON for {model_alias} under {results_dir / model_alias}"
        )
    return candidates[-1]


# ===========================================================================
# Clean-projection helpers  (from plot_clean_projection_paper_figures.py)
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

PROJ_TICK_FS   = 16
PROJ_AXIS_FS   = 22
PROJ_TITLE_FS  = 28
PROJ_LEGEND_FS = 16


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
# Fig2 — Coupling heatmap, horizontal 1×3
# ---------------------------------------------------------------------------

def render_fig2(base_dir: Path, output: Path, dpi: int) -> None:
    paths: list[Path] = []
    panel_captions: list[str] = []
    for alias, dir_name, coupling_key, _, panel_label in APPENDIX_MODELS:
        paths.append(latest_coupling_json(base_dir / dir_name, coupling_key, "alpaca_common"))
        panel_captions.append(panel_label)

    n_cols = 2
    paths_grid = [paths[i:i + n_cols] for i in range(0, len(paths), n_cols)]
    plot_coupling_grid(
        paths=paths_grid,
        output=output,
        dpi=dpi,
        model_by_row=False,
        show_panel_captions=True,
        panel_captions=panel_captions,
        independent_panels=True,
    )


# ---------------------------------------------------------------------------
# Fig3 — Steering (Alpaca), vertical 3×1
# ---------------------------------------------------------------------------

def render_fig3(base_dir: Path, output: Path, dpi: int) -> None:
    models = [(dir_name, panel_label) for _, dir_name, _, _, panel_label in APPENDIX_MODELS]

    cell_info = {
        (m, t): _load_cell_info(base_dir, m, t, "alpaca")
        for m, _ in models
        for t in ["harmfulness", "refusal"]
    }

    n_alphas   = len(cell_info[(models[0][0], "harmfulness")][0])
    max_layers = max(
        len(cell_info[(m, t)][1])
        for m, _ in models for t in ["harmfulness", "refusal"]
    )
    TICK_FS, AXIS_FS, LEG_FS = 15, 18, 14

    fig_w    = max(9, max_layers * 0.18)
    fig_h    = 3.2 * n_alphas * len(models) + 0.6
    fig      = plt.figure(figsize=(fig_w, fig_h))
    leg_frac = 0.62 / fig_h

    outer_gs = gridspec.GridSpec(
        len(models), 1, figure=fig,
        top=1.0 - leg_frac, bottom=0.09,
        left=0.245, right=0.96, hspace=0.55,
    )

    n_macro_r = len(models)
    for ri, (m, _) in enumerate(models):
        harm_alphas, harm_layers, _, harm_idx = cell_info[(m, "harmfulness")]
        _ref_alphas, ref_layers,  _, ref_idx  = cell_info[(m, "refusal")]
        layers = _sorted_layer_names(set(harm_layers) | set(ref_layers))
        alphas = harm_alphas
        bridge = BRIDGE_LAYERS_FIG3.get(m)

        all_vals = [
            idx[(lg, a)]["refusal"]["percentage"]
            for a in alphas
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
        for ai, alpha in enumerate(alphas):
            ax            = fig.add_subplot(inner_gs[ai])
            is_last_alpha = (ai == n_alphas - 1)
            is_bottom     = (ri == n_macro_r - 1) and is_last_alpha
            _draw_merged_subplot(
                ax, harm_idx, ref_idx, layers, alpha,
                bridge_layers=bridge, y_max=y_max,
                show_xlabel=is_last_alpha, show_ylabel=False,
                tick_fs=TICK_FS, axis_fs=AXIS_FS,
            )
            if is_bottom:
                ax.set_xlabel("Steered Layer", fontsize=AXIS_FS)

    handles = [
        plt.Line2D([0], [0], color=STEER_COLORS["harmfulness"], linewidth=2.0,
                   marker="o", markersize=5, label=STEER_LABELS["harmfulness"]),
        plt.Line2D([0], [0], color=STEER_COLORS["refusal"], linewidth=2.0,
                   marker="^", markersize=5, label=STEER_LABELS["refusal"]),
        mpatches.Patch(facecolor=BRIDGE_COLOR, edgecolor="none", alpha=0.7, label="Bridge"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.545, 0.995),
        ncol=len(handles), fontsize=LEG_FS,
        frameon=True, fancybox=False,
        edgecolor="lightgray", facecolor="white", framealpha=1.0,
        handlelength=2.0, columnspacing=1.4, borderpad=0.45, borderaxespad=0.0,
    )
    _add_row_labels(fig, outer_gs, models, fs=26)
    for ri, _ in enumerate(models):
        pos      = outer_gs[ri, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.185, y_center, "Refusal Rate (%)",
                 fontsize=AXIS_FS, rotation=90, ha="center", va="center")

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ---------------------------------------------------------------------------
# Fig3_2 — Steered t_inst restore bar chart, horizontal 1×3
# ---------------------------------------------------------------------------

def render_fig3_2(results_dir: Path, output: Path, dpi: int) -> None:
    model_names = [alias for alias, _, _, _, _ in APPENDIX_MODELS]
    paths    = {name: _latest_tinst_json(results_dir, name) for name in model_names}
    payloads = {name: _load_json(paths[name])                for name in model_names}

    labels = ["patch_bridge", "patch_readout"]
    drops_by_model = {
        name: np.array(
            [payloads[name]["summaries"][lbl]["drop_vs_no_patch_mean"] for lbl in labels],
            dtype=np.float64,
        )
        for name in model_names
    }
    stderrs_by_model = {
        name: np.array(
            [payloads[name]["summaries"][lbl]["drop_vs_no_patch_stderr"] for lbl in labels],
            dtype=np.float64,
        )
        for name in model_names
    }

    colors = ["#3f7f6b", "#b45b55"]
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(13.6 * n / 3, 4.8), sharey=False, squeeze=False)
    axes = axes[0]
    panel_labels = {alias: panel_label for alias, _, _, _, panel_label in APPENDIX_MODELS}
    for idx, name in enumerate(model_names):
        delta_values = -drops_by_model[name]
        lows         = delta_values - stderrs_by_model[name]
        low          = min(float(lows.min()), 0.0)
        pad          = max((0.0 - low) * 0.18, 0.05)
        _plot_delta_drop_bars(
            ax=axes[idx],
            drops=drops_by_model[name],
            drop_stderrs=stderrs_by_model[name],
            title="",
            colors=colors,
            show_ylabel=(idx == 0),
            model_name=name,
        )
        val_range = max(abs(low), 0.1)
        axes[idx].set_ylim(low - pad - val_range * 0.10, pad * 0.3)
        axes[idx].tick_params(axis="y", labelleft=True)
        axes[idx].text(
            0.5, -0.31, panel_labels[name],
            transform=axes[idx].transAxes,
            ha="center", va="top", fontsize=26, fontweight="bold",
        )

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.31, top=0.94, wspace=0.22)
    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {output}")


# ---------------------------------------------------------------------------
# Fig4 — Activation patching heatmap, vertical 3×2
# ---------------------------------------------------------------------------

def render_fig4(base_dir: Path, output: Path, dpi: int) -> None:
    ap_models = [
        (alias, dir_name, panel_label)
        for alias, dir_name, _, _, panel_label in APPENDIX_MODELS
    ]
    paths = {
        alias: latest_patching_json(base_dir, dir_name)
        for alias, dir_name, _, _, _ in APPENDIX_MODELS
    }
    render_patching_heatmap_grid(paths, output, dpi, models=ap_models)


# ---------------------------------------------------------------------------
# Fig4_2 — Behavioral grid, vertical 3×1
# ---------------------------------------------------------------------------

def render_fig4_2(base_dir: Path, output: Path, dpi: int) -> None:
    ap_models = [
        (alias, dir_name, panel_label)
        for alias, dir_name, _, _, panel_label in APPENDIX_MODELS
    ]
    paths = {
        alias: latest_patching_behavior_json(base_dir, dir_name)
        for alias, dir_name, _, _, _ in APPENDIX_MODELS
    }
    render_patching_behavior_grid(paths, output, dpi, models=ap_models)


# ---------------------------------------------------------------------------
# Fig5 — Coupling heatmap multi-dataset, vertical 3×3
# ---------------------------------------------------------------------------

def render_fig5(base_dir: Path, output: Path, dpi: int) -> None:
    row_paths: list[list[Path]] = []
    row_captions: list[str]     = []
    for alias, dir_name, coupling_key, _, panel_label in APPENDIX_MODELS:
        model_dir = base_dir / dir_name
        ds_paths  = [
            latest_coupling_json(model_dir, coupling_key, ds_key)
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
    )


# ---------------------------------------------------------------------------
# Fig6 — Adversarial steering, vertical 3×3
# ---------------------------------------------------------------------------

def render_fig6(base_dir: Path, output: Path, dpi: int) -> None:
    models   = [(dir_name, panel_label) for _, dir_name, _, _, panel_label in APPENDIX_MODELS]
    datasets = FIG5_DATASETS

    cell_info = {
        (m, ds_key): _load_cell_info(base_dir, m, "harmfulness", ds_key)
        for m, _ in models
        for ds_key, _ in datasets
    }

    TICK_FS, AXIS_FS = 15, 18
    n_alphas   = len(cell_info[(models[0][0], datasets[0][0])][0])
    max_layers = max(
        len(cell_info[(m, ds_key)][1])
        for m, _ in models for ds_key, _ in datasets
    )

    n_cols   = len(datasets)
    fig_w    = max(12, max_layers * 0.09 * n_cols)
    fig_h    = 2.4 * n_alphas * len(models) + 0.4
    fig      = plt.figure(figsize=(fig_w, fig_h))
    hdr_frac = 0.50 / fig_h

    # left=0.32: gives enough room for row-label / ylabel / tick numbers
    outer_gs = gridspec.GridSpec(
        len(models), n_cols, figure=fig,
        top=1.0 - hdr_frac, bottom=0.09,
        left=0.32, right=0.94,
        hspace=0.30, wspace=0.14,
    )

    _draw_composed_grid(
        fig, outer_gs,
        models=models, col_keys=datasets, cell_info=cell_info,
        TICK_FS=TICK_FS, AXIS_FS=AXIS_FS,
    )

    # Increase labelpad so ylabel doesn't overlap tick numbers
    for ax in fig.get_axes():
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_FS, labelpad=34)

    _add_col_headers(
        fig, outer_gs,
        labels=[ds_label for _, ds_label in datasets],
        y_frac=1.0 - hdr_frac / 2,
        fs=16,
    )
    # Manual row labels (left=0.32 means default _add_row_labels offset would be off)
    for ri, (_, m_label) in enumerate(models):
        pos      = outer_gs[ri, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        x_pos    = max(0.03, pos.x0 - 0.08)
        fig.text(x_pos, y_center, m_label,
                 fontsize=18, fontweight="bold",
                 rotation=90, ha="center", va="center")

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ---------------------------------------------------------------------------
# Fig7 / Fig8 — Clean projection, vertical 3×1
# ---------------------------------------------------------------------------

def render_fig7_or_fig8(fig_name: str, base_dir: Path, output: Path, dpi: int) -> None:
    config = FIG_CONFIG_PROJ[fig_name]
    measure = config["measure"]

    paths: list[Path] = []
    labels: list[str] = []

    for alias, dir_name, _, _, panel_label in APPENDIX_MODELS:
        paths.append(_find_clean_projection_json(base_dir / dir_name, measure))
        labels.append(panel_label)

    summaries_list = [_load_json(p).get("summaries", {}) for p in paths]
    n = len(APPENDIX_MODELS)

    # Keep the figure wide enough so the single-line legend does not overlap.
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(16.5, 5.2 * n),
        sharey=False,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, summaries in zip(axes, summaries_list):
        ymin, ymax = _figure_limits(summaries)
        max_layers = 0

        for ds_key, ds_label, color in DATASETS_PROJ:
            if ds_key not in summaries:
                continue

            means = np.array(
                summaries[ds_key]["all_layer_mean"],
                dtype=np.float64,
            )
            stds = np.array(
                summaries[ds_key]["all_layer_std"],
                dtype=np.float64,
            )
            layers = np.arange(len(means))
            max_layers = max(max_layers, len(means))

            ax.plot(
                layers,
                means,
                label=ds_label,
                color=color,
                linewidth=2.4,
            )
            ax.fill_between(
                layers,
                means - stds,
                means + stds,
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.75)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.5, max_layers - 0.5)
        ax.set_xticks(_layer_ticks(max_layers))

        ax.set_xlabel(
            "Layer",
            fontsize=PROJ_AXIS_FS,
            labelpad=10,
        )

        # Show a y-axis label for each row and align all labels to the same coordinate.
        ax.set_ylabel(
            config["ylabel"],
            fontsize=PROJ_AXIS_FS,
            labelpad=8,
        )
        ax.yaxis.set_label_coords(-0.05, 0.5)

        ax.tick_params(
            axis="both",
            labelsize=PROJ_TICK_FS,
            length=4.0,
            width=1.0,
            pad=3.0,
        )
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.0)

    # Align the legend and lower plot area to the same left/right bounds.
    LEFT, RIGHT = 0.14, 0.985

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        bbox_transform=fig.transFigure,
        ncol=5,
        fontsize=PROJ_LEGEND_FS,
        frameon=True,
        fancybox=False,
        edgecolor="lightgray",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.35,
        columnspacing=0.9,
        handletextpad=0.35,
        borderpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        bottom=0.06,
        top=0.87,
        hspace=0.40,
    )

    # Left-side model name.
    for ax, panel_label in zip(axes, labels):
        pos = ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.04,
            y_center,
            panel_label,
            fontsize=PROJ_TITLE_FS,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Appendix A figures for Gemma 2, Falcon3, OLMo 2, Qwen 3.5."
    )
    parser.add_argument("--base-dir",    default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir",  default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory with steered t_inst restore result JSONs (for Fig3_2).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--figs", default="all",
        help="Comma-separated list of figures to render (1,2,3,4,5,6,7,8,9) or 'all'.",
    )
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
    base_dir    = Path(args.base_dir)
    output_dir  = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    requested = (
        {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        if args.figs.strip().lower() == "all"
        else {f.strip() for f in args.figs.split(",")}
    )

    if "1" in requested:
        print("\n=== Fig1 (coupling heatmap, 2x2 grid) ===")
        try:    render_fig2(base_dir, output_dir / "Fig1.png", dpi)
        except Exception as e: print(f"[WARN] Fig1 failed: {e}")

    if "2" in requested:
        print("\n=== Fig2 (steering Alpaca, vertical 3x1) ===")
        try:    render_fig3(base_dir, output_dir / "Fig2.png", dpi)
        except Exception as e: print(f"[WARN] Fig2 failed: {e}")

    if "3" in requested:
        print("\n=== Fig3 (tinst-restore bars, horizontal 1x3) ===")
        try:    render_fig3_2(results_dir, output_dir / "Fig3.png", dpi)
        except Exception as e: print(f"[WARN] Fig3 failed: {e}")

    if "4" in requested:
        print("\n=== Fig4 (behavioral grid, vertical 3x1) ===")
        try:    render_fig4_2(base_dir, output_dir / "Fig4.png", dpi)
        except Exception as e: print(f"[WARN] Fig4 failed: {e}")

    if "5" in requested:
        print("\n=== Fig5 (activation patching heatmap, vertical 3x2) ===")
        try:    render_fig4(base_dir, output_dir / "Fig5.png", dpi)
        except Exception as e: print(f"[WARN] Fig5 failed: {e}")

    if "6" in requested:
        print("\n=== Fig6 (coupling heatmap multi-dataset, vertical 3x3) ===")
        try:    render_fig5(base_dir, output_dir / "Fig6.png", dpi)
        except Exception as e: print(f"[WARN] Fig6 failed: {e}")

    if "7" in requested:
        print("\n=== Fig7 (adversarial steering, vertical 3x3) ===")
        try:    render_fig6(base_dir, output_dir / "Fig7.png", dpi)
        except Exception as e: print(f"[WARN] Fig7 failed: {e}")

    if "8" in requested:
        print("\n=== Fig8 (harmfulness projection, vertical 3x1) ===")
        try:    render_fig7_or_fig8("fig7", base_dir, output_dir / "Fig8.png", dpi)
        except Exception as e: print(f"[WARN] Fig8 failed: {e}")

    if "9" in requested:
        print("\n=== Fig9 (refusal projection, vertical 3x1) ===")
        try:    render_fig7_or_fig8("fig8", base_dir, output_dir / "Fig9.png", dpi)
        except Exception as e: print(f"[WARN] Fig9 failed: {e}")

    print(f"\nDone. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
