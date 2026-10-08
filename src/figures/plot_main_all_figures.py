#!/usr/bin/env python3
"""
Standalone figure generator — all main paper figures (Fig2–Fig9).

Reads directly from hardcoded result-file paths. No other source files needed.
Run:
    python plot_all_figures.py                  # all figures
    python plot_all_figures.py --fig fig3       # single figure
    python plot_all_figures.py --output-dir /tmp/figs
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

# ─── Font ─────────────────────────────────────────────────────────────────────
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

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parents[1]
OUT_PT     = SCRIPT_DIR / "out_pt"
RESULTS    = SCRIPT_DIR / "results"

# Hardcoded result file paths (alpha = 3 throughout)
DATA = {
    # Fig2 — coupling heatmap, Alpaca dataset
    "fig2_llama": OUT_PT / "fig_coupling_runs/llama31/harm-to-refusal-coupling-hidden-alpha_3-fig2_alpaca-20260506-174411.json",
    "fig2_qwen":  OUT_PT / "fig_coupling_runs/qwen25/harm-to-refusal-coupling-hidden-alpha_3-fig2_alpaca-20260506-174411.json",
    # Fig3 — steering Alpaca (harmfulness + refusal, alpha=3)
    "fig3_llama_agg": OUT_PT / "llama-3.1-8b-instruct/steering/steering-hidden-harmfulness_refusal-L0-31-alpha_0_to_3-alpaca-20260509-222119-wildguard-judge/aggregate_result.json",
    "fig3_qwen_agg":  OUT_PT / "qwen2.5-7b-instruct/steering/steering-hidden-harmfulness_refusal-L0-27-alpha_0_to_3-alpaca-20260509-221730-wildguard-judge/aggregate_result.json",
    # Fig3_2 — tinst-restore bridge/readout bar chart
    "fig3_2_llama": RESULTS / "steered_tinst_restore_refusal_projection/llama31/llama31-tinst-restore-final-tpost-refusal-alpha_3-20260513-173347.json",
    "fig3_2_qwen":  RESULTS / "steered_tinst_restore_refusal_projection/qwen25/qwen25-tinst-restore-final-tpost-refusal-alpha_3-20260513-173411.json",
    # Fig4 / Fig4_2 — activation patching heatmap + behavioral validation
    "fig4_llama": OUT_PT / "llama-3.1-8b-instruct/activation_patching/activation-patching-hidden-alpha_3-20260509-133640/activation-patching-hidden-alpha_3-20260509-133640.json",
    "fig4_qwen":  OUT_PT / "qwen2.5-7b-instruct/activation_patching/activation-patching-hidden-alpha_3-20260509-134824/activation-patching-hidden-alpha_3-20260509-134824.json",
    # Fig5 — coupling heatmap, 3 adversarial datasets
    "fig5_llama_misrep":    OUT_PT / "fig_coupling_runs/llama31/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_misrepresentation-20260506-174941.json",
    "fig5_llama_authority": OUT_PT / "fig_coupling_runs/llama31/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_authority_endorsement-20260506-174941.json",
    "fig5_llama_expert":    OUT_PT / "fig_coupling_runs/llama31/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_expert_endorsement-20260506-174941.json",
    "fig5_qwen_misrep":    OUT_PT / "fig_coupling_runs/qwen25/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_misrepresentation-20260506-174941.json",
    "fig5_qwen_authority": OUT_PT / "fig_coupling_runs/qwen25/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_authority_endorsement-20260506-174941.json",
    "fig5_qwen_expert":    OUT_PT / "fig_coupling_runs/qwen25/harm-to-refusal-coupling-hidden-alpha_3-fig5-sorry_expert_endorsement-20260506-174941.json",
    # Fig6 — steering adversarial datasets (harmfulness only, alpha=3)
    "fig6_llama_misrep":    OUT_PT / "llama-3.1-8b-instruct/steering/steering-hidden-harmfulness_refusal-L0-31-alpha_0_to_3-misrepresentation-20260509-221736-wildguard-judge/aggregate_result.json",
    "fig6_llama_authority": OUT_PT / "llama-3.1-8b-instruct/steering/steering-hidden-harmfulness_refusal-L0-31-alpha_0_to_3-authority_endorsement-20260509-221713-wildguard-judge/aggregate_result.json",
    "fig6_llama_expert":    OUT_PT / "llama-3.1-8b-instruct/steering/steering-hidden-harmfulness_refusal-L0-31-alpha_0_to_3-expert_endorsement-20260509-225520-wildguard-judge/aggregate_result.json",
    "fig6_qwen_misrep":    OUT_PT / "qwen2.5-7b-instruct/steering/steering-hidden-harmfulness_refusal-L0-27-alpha_0_to_3-misrepresentation-20260509-221917-wildguard-judge/aggregate_result.json",
    "fig6_qwen_authority": OUT_PT / "qwen2.5-7b-instruct/steering/steering-hidden-harmfulness_refusal-L0-27-alpha_0_to_3-authority_endorsement-20260509-221750-wildguard-judge/aggregate_result.json",
    "fig6_qwen_expert":    OUT_PT / "qwen2.5-7b-instruct/steering/steering-hidden-harmfulness_refusal-L0-27-alpha_0_to_3-expert_endorsement-20260509-225553-wildguard-judge/aggregate_result.json",
    # Fig7 — harmfulness projection (clean, all datasets)
    "fig7_llama": OUT_PT / "llama-3.1-8b-instruct/clean_projection/harmfulness_tinst/clean/clean-tinst-harmfulness-projection-20260502-160301.json",
    "fig7_qwen":  OUT_PT / "qwen2.5-7b-instruct/clean_projection/harmfulness_tinst/clean/clean-tinst-harmfulness-projection-20260502-160314.json",
    # Fig8 — refusal projection (clean, all datasets)
    "fig8_llama": OUT_PT / "llama-3.1-8b-instruct/clean_projection/refusal_tpost/clean/clean-tpost-refusal-projection-20260502-160332.json",
    "fig8_qwen":  OUT_PT / "qwen2.5-7b-instruct/clean_projection/refusal_tpost/clean/clean-tpost-refusal-projection-20260502-160346.json",
    # Fig9 — injection-delta heatmap grid (refusal_tpost, alpha=3)
    "fig9_llama": OUT_PT / "clean_projection_suite/20260509-155405/models/llama31/refusal_tpost/injection/clean-tpost-refusal-projection-alpha_3-20260509-155617.json",
    "fig9_qwen":  OUT_PT / "clean_projection_suite/20260509-155405/models/qwen25/refusal_tpost/injection/clean-tpost-refusal-projection-alpha_3-20260509-155712.json",
}

# ─── Common constants ─────────────────────────────────────────────────────────
MODELS_LLAMA_QWEN = [
    ("llama", "llama-3.1-8b-instruct", "(a) LLaMA 3.1"),
    ("qwen",  "qwen2.5-7b-instruct",   "(b) Qwen 2.5"),
]
DATASET_LABELS = {
    "alpaca_common":             "Alpaca",
    "alpaca":                    "Alpaca",
    "sorry_misrepresentation":   "Misrepresentation",
    "misrepresentation":         "Misrepresentation",
    "sorry_authority_endorsement": "Authority endorsement",
    "authority_endorsement":     "Authority endorsement",
    "sorry_expert_endorsement":  "Expert endorsement",
    "expert_endorsement":        "Expert endorsement",
}
PROJECTION_DATASETS = [
    ("advbench",             "AdvBench",             "#c43b3b"),
    ("alpaca",               "Alpaca",               "#2f7d4f"),
    ("misrepresentation",    "Misrepresentation",    "#3b66c4"),
    ("authority_endorsement","Authority endorsement","#8b5fbf"),
    ("expert_endorsement",   "Expert endorsement",   "#d28c2d"),
]
STEER_COLORS  = {"harmfulness": "#d73027", "refusal": "#4575b4"}
STEER_LABELS  = {"harmfulness": "Harmfulness Steering", "refusal": "Refusal Steering"}
BRIDGE_COLOR  = "#DCEAF7"
LABEL_COLORS  = {"refusal": "#d73027", "accept": "#1a9850", "other": "#cccccc"}
LABEL_TITLES  = {"refusal": "Refusal", "accept": "Accept", "other": "Other"}


def _load(path: Path) -> dict:
    path = _resolve_existing_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_figure(fig, output: Path, *, dpi: int, **kwargs) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, **kwargs)
    fig.savefig(output.with_suffix(".pdf"), **kwargs)


DATASET_ALIASES = {
    "alpaca": "alpaca_common",
    "alpaca_common": "alpaca_common",
    "steering_alpaca_data_instruction": "alpaca_common",
    "misrepresentation": "sorry_misrepresentation",
    "sorry_misrepresentation": "sorry_misrepresentation",
    "authority_endorsement": "sorry_authority_endorsement",
    "sorry_authority_endorsement": "sorry_authority_endorsement",
    "expert_endorsement": "sorry_expert_endorsement",
    "sorry_expert_endorsement": "sorry_expert_endorsement",
}

MODEL_DIRS = {
    "llama31": "llama-3.1-8b-instruct",
    "qwen25": "qwen2.5-7b-instruct",
}


def _latest(paths: list[Path], description: str) -> Path:
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(description)
    return max(existing, key=lambda p: p.stat().st_mtime)


def _normalize_model_key(raw: str) -> str:
    v = raw.lower()
    if "llama" in v:
        return "llama31"
    if "qwen" in v:
        return "qwen25"
    return v


def _model_key_from_path(path: Path) -> str:
    text = str(path).lower()
    if "llama" in text:
        return "llama31"
    if "qwen" in text:
        return "qwen25"
    raise ValueError(f"Could not infer model key from {path}")


def _dataset_key_from_path(path: Path) -> str:
    text = str(path).replace("-", "_")
    for key in (
        "alpaca_common",
        "alpaca",
        "misrepresentation",
        "authority_endorsement",
        "expert_endorsement",
    ):
        if key in text:
            return DATASET_ALIASES[key]
    raise ValueError(f"Could not infer dataset key from {path}")


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
    return DATASET_ALIASES.get(label, label)


def _payload_model_matches(payload: dict, model_key: str) -> bool:
    meta = payload.get("metadata", {})
    return _normalize_model_key(str(meta.get("model", ""))) == model_key


def _resolve_coupling_path(path: Path) -> Path:
    model_key = _model_key_from_path(path)
    dataset_key = _dataset_key_from_path(path)
    candidates = []
    for candidate in OUT_PT.rglob("harm-to-refusal-coupling-hidden-alpha_3-*.json"):
        try:
            payload = _load(candidate)
        except Exception:
            continue
        if "internal_coupling" not in payload:
            continue
        if not _payload_model_matches(payload, model_key):
            continue
        if _dataset_key_from_payload(payload) != dataset_key:
            continue
        candidates.append(candidate)
    return _latest(
        candidates,
        f"No coupling JSON for model={model_key} dataset={dataset_key} under {OUT_PT}",
    )


def _compose_source_path(aggregate_path: Path) -> Path:
    judge_dir = aggregate_path.parent
    source_stem = judge_dir.name.removesuffix("-wildguard-judge")
    return judge_dir.parent / f"{source_stem}.json"


def _aggregate_matches(aggregate_path: Path, target: str, dataset_key: str) -> tuple[bool, float]:
    source_path = _compose_source_path(aggregate_path)
    try:
        aggregate = _load(aggregate_path)
    except Exception:
        return False, aggregate_path.stat().st_mtime
    targets = {row.get("target") for row in aggregate.get("by_target_layer_alpha", [])}
    if target not in targets:
        return False, aggregate_path.stat().st_mtime
    if source_path.exists():
        try:
            source = _load(source_path)
        except Exception:
            return False, aggregate_path.stat().st_mtime
        meta = source.get("metadata", {})
        source_dataset = DATASET_ALIASES.get(
            Path(str(meta.get("input_path", ""))).stem.replace("-", "_"),
            "",
        )
        if source_dataset != dataset_key:
            return False, source_path.stat().st_mtime
        return True, source_path.stat().st_mtime
    parent_name = aggregate_path.parent.name.replace("-", "_")
    return dataset_key.replace("sorry_", "") in parent_name, aggregate_path.stat().st_mtime


def _resolve_steering_path(path: Path, target: str) -> Path:
    if path.exists():
        try:
            payload = _load(path)
            if target in {row.get("target") for row in payload.get("by_target_layer_alpha", [])}:
                return path
        except Exception:
            pass
    model_key = _model_key_from_path(path)
    dataset_key = _dataset_key_from_path(path)
    steering_dir = OUT_PT / MODEL_DIRS[model_key] / "steering"
    candidates: list[tuple[float, Path]] = []
    for aggregate_path in steering_dir.glob("steering-hidden-*-wildguard-judge/aggregate_result.json"):
        ok, mtime = _aggregate_matches(aggregate_path, target, dataset_key)
        if ok:
            candidates.append((mtime, aggregate_path))
    if not candidates:
        raise FileNotFoundError(
            f"No aggregate_result.json for model={model_key} target={target} "
            f"dataset={dataset_key} under {steering_dir}"
        )
    _, resolved = max(candidates, key=lambda x: x[0])
    print(f"[resolve] {path} -> {resolved}", flush=True)
    return resolved


def _resolve_tinst_path(path: Path) -> Path:
    model_key = _model_key_from_path(path)
    model_alias = "llama31" if model_key == "llama31" else "qwen25"
    candidates = list(
        (RESULTS / "steered_tinst_restore_refusal_projection" / model_alias).glob(
            f"{model_alias}-tinst-restore-final-tpost-refusal-alpha_3-*.json"
        )
    )
    return _latest(candidates, f"No tinst-restore JSON for {model_alias} under {RESULTS}")


def _resolve_patching_path(path: Path) -> Path:
    model_key = _model_key_from_path(path)
    model_dir = OUT_PT / MODEL_DIRS[model_key] / "activation_patching"
    candidates = []
    for candidate in model_dir.glob("activation-patching-hidden-alpha_3-*/*.json"):
        try:
            payload = _load(candidate)
        except Exception:
            continue
        if payload.get("activation_patching_coupling") or payload.get("path_patching_coupling"):
            candidates.append(candidate)
    return _latest(candidates, f"No activation patching JSON for {model_key} under {model_dir}")


def _resolve_projection_path(path: Path) -> Path:
    model_key = _model_key_from_path(path)
    model_dir = OUT_PT / MODEL_DIRS[model_key]
    name = path.name
    if "clean-tinst-harmfulness-projection" in name:
        measure = "harmfulness_tinst"
        pattern = "clean-tinst-harmfulness-projection-*.json"
    elif "clean-tpost-refusal-projection" in name:
        measure = "refusal_tpost"
        pattern = "clean-tpost-refusal-projection-*.json"
    else:
        raise FileNotFoundError(path)
    mode = "injection" if "alpha_3" in name or "injection" in path.parts else "clean"
    candidates = [
        p
        for p in model_dir.rglob(pattern)
        if f"/{measure}/{mode}/" in str(p)
        and not p.name.endswith("-metadata.json")
    ]
    if not candidates:
        suite_alias = "llama31" if model_key == "llama31" else "qwen25"
        suite_root = OUT_PT / "clean_projection_suite"
        candidates = [
            p
            for p in suite_root.rglob(pattern)
            if f"/models/{suite_alias}/{measure}/{mode}/" in str(p)
        ]
    return _latest(candidates, f"No {measure}/{mode} projection JSON for {model_key} under {OUT_PT}")


def _resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    if "harm-to-refusal-coupling" in path.name:
        resolved = _resolve_coupling_path(path)
    elif path.name == "aggregate_result.json":
        raise FileNotFoundError(path)
    elif "tinst-restore-final-tpost-refusal" in path.name:
        resolved = _resolve_tinst_path(path)
    elif "activation-patching-hidden-alpha_3" in path.name:
        resolved = _resolve_patching_path(path)
    elif "clean-tinst-harmfulness-projection" in path.name or "clean-tpost-refusal-projection" in path.name:
        resolved = _resolve_projection_path(path)
    else:
        raise FileNotFoundError(path)
    print(f"[resolve] {path} -> {resolved}", flush=True)
    return resolved


def layer_ticks(size: int) -> list[int]:
    step = 4
    ticks = list(range(0, size, step))
    last = size - 1
    if not ticks or ticks[-1] != last:
        if ticks and last - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(last)
    return ticks


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


# ═══════════════════════════════════════════════════════════════════════════════
# Fig2 & Fig5 — coupling heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

COUPLING_TICK_FS       = 16
COUPLING_AXIS_FS       = 22
COUPLING_ROW_LABEL_FS  = 28
COUPLING_PANEL_FS      = 26
COUPLING_CBAR_TICK_FS  = 16
COUPLING_CBAR_LABEL_FS = 22


def _coupling_values(path: Path) -> np.ndarray:
    payload = _load(path)
    return np.array(payload["internal_coupling"]["normalized_coupling_map"], dtype=np.float64)


def _symmetric_limits(maps: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([v[np.isfinite(v)] for v in maps])
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), 98))
    limit = limit if limit > 0 else float(np.nanmax(np.abs(finite)))
    limit = limit if limit > 0 else 1.0
    return -limit, limit


def _plot_coupling_grid(
    paths: list[list[Path]],
    output: Path,
    dpi: int,
    model_by_row: bool,
    show_panel_captions: bool,
    column_labels: list[str] | None = None,
    row_captions: list[str] | None = None,
    panel_captions: list[str] | None = None,
) -> None:
    maps = [[_coupling_values(p) for p in row] for row in paths]
    n_rows, n_cols = len(paths), len(paths[0])

    if model_by_row:
        limits = [_symmetric_limits(row) for row in maps]
        limit_for = lambda r, c: limits[r]
    else:
        limits = [_symmetric_limits([maps[r][c] for r in range(n_rows)]) for c in range(n_cols)]
        limit_for = lambda r, c: limits[c]

    panel_size = 4.35
    fig_w = panel_size * n_cols + (1.35 if model_by_row else 1.55)
    fig_h = panel_size * n_rows + (0.55 if n_rows == 1 else 0.25)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    cbar_images = [None] * (n_rows if model_by_row else n_cols)
    for ri, row in enumerate(maps):
        for ci, values in enumerate(row):
            ax = axes[ri][ci]
            masked = np.ma.masked_invalid(values)
            vmin, vmax = limit_for(ri, ci)
            image = ax.imshow(masked, origin="lower", aspect="equal", cmap="coolwarm", vmin=vmin, vmax=vmax)
            cbar_images[ri if model_by_row else ci] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            ax.set_xticks(layer_ticks(values.shape[1]))
            ax.set_yticks(layer_ticks(values.shape[0]))
            ax.tick_params(axis="both", labelsize=COUPLING_TICK_FS, length=4, width=1, pad=3)
            ax.grid(False)
            if ci == 0:
                ax.set_ylabel("Injection Layer", fontsize=COUPLING_AXIS_FS, labelpad=10)
            if ri == n_rows - 1:
                ax.set_xlabel("Downstream Layer", fontsize=COUPLING_AXIS_FS, labelpad=10)
            if column_labels and ri == 0:
                ax.set_title(column_labels[ci], fontsize=COUPLING_AXIS_FS, fontweight="bold", pad=14)

        if model_by_row:
            caption = row_captions[ri] if row_captions else ("(a) LLaMA 3.1" if ri == 0 else "(b) Qwen 2.5")
            axes[ri][0].text(-0.36, 0.5, caption,
                             transform=axes[ri][0].transAxes, rotation=90,
                             ha="center", va="center",
                             fontsize=COUPLING_ROW_LABEL_FS, fontweight="bold")
        elif show_panel_captions:
            for ci in range(n_cols):
                caption = panel_captions[ci] if panel_captions else ("(a) LLaMA 3.1" if ci == 0 else "(b) Qwen 2.5")
                axes[-1][ci].text(0.5, -0.24, caption,
                                  transform=axes[-1][ci].transAxes,
                                  ha="center", va="top",
                                  fontsize=COUPLING_PANEL_FS, fontweight="bold")

    fig.subplots_adjust(
        left=0.205 if model_by_row else 0.115,
        right=0.905,
        bottom=0.16 if n_rows == 1 else 0.08,
        top=0.92 if column_labels else 0.985,
        wspace=0.32 if not model_by_row else 0.20,
        hspace=0.02 if model_by_row else 0.16,
    )
    for idx, image in enumerate(cbar_images):
        if image is None:
            continue
        anchor = (axes[idx][n_cols - 1] if model_by_row else axes[0][idx]).get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.016, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=COUPLING_CBAR_TICK_FS, length=4, width=1)
        if model_by_row or idx == len(cbar_images) - 1:
            cbar.set_label(r"$\Delta$ Refusal Projection",
                           fontsize=COUPLING_CBAR_LABEL_FS, labelpad=14)

    _save_figure(fig, output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output}")


def render_fig2(output_dir: Path, dpi: int = 300) -> None:
    """Fig2: coupling heatmap, Alpaca, LLaMA × Qwen (side by side)."""
    paths = [[DATA["fig2_llama"], DATA["fig2_qwen"]]]
    _plot_coupling_grid(paths, output_dir / "Fig2.png", dpi,
                        model_by_row=False, show_panel_captions=True)


def render_fig5(output_dir: Path, dpi: int = 300) -> None:
    """Fig5: coupling heatmap, 3 adversarial datasets, LLaMA (row0) × Qwen (row1)."""
    llama_paths = [DATA["fig5_llama_misrep"], DATA["fig5_llama_authority"], DATA["fig5_llama_expert"]]
    qwen_paths  = [DATA["fig5_qwen_misrep"],  DATA["fig5_qwen_authority"],  DATA["fig5_qwen_expert"]]
    col_labels  = ["Misrepresentation", "Authority endorsement", "Expert endorsement"]
    _plot_coupling_grid([llama_paths, qwen_paths], output_dir / "Fig7.png", dpi,
                        model_by_row=True, show_panel_captions=False, column_labels=col_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig3 — steering Alpaca (harmfulness + refusal on same axes)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_steering_cell(aggregate_path: Path, target: str) -> tuple:
    """Return (alphas, layers, idx) for a given target from an aggregate_result.json."""
    aggregate_path = _resolve_steering_path(aggregate_path, target)
    agg = _load(aggregate_path)
    steered = [r for r in agg["by_target_layer_alpha"] if r["target"] == target]
    alphas  = sorted({r["alpha"] for r in steered})
    layers  = _sorted_layer_names({r["layer_group_name"] for r in steered})
    idx     = {(r["layer_group_name"], r["alpha"]): r for r in steered}
    return alphas, layers, idx


def _draw_merged_steering(ax, harm_idx, ref_idx, layers, alpha,
                          bridge_layers=None, y_max=100.0,
                          show_xlabel=False, show_ylabel=True,
                          tick_fs=9, axis_fs=10):
    x = np.arange(len(layers))

    if bridge_layers is not None:
        b_start, b_end = bridge_layers
        bridge_names = {f"L{i}" for i in range(b_start, b_end + 1)}
        xs_bridge = [xi for xi, lg in enumerate(layers) if lg in bridge_names]
        if xs_bridge:
            ax.axvspan(xs_bridge[0] - 0.5, xs_bridge[-1] + 0.5,
                       facecolor=BRIDGE_COLOR, edgecolor="none", alpha=0.6, zorder=0)

    y_data = {}
    for target, idx in [("harmfulness", harm_idx), ("refusal", ref_idx)]:
        y_data[target] = np.array([
            idx[(lg, alpha)]["refusal"]["percentage"] if (lg, alpha) in idx else np.nan
            for lg in layers
        ], dtype=float)

    _markers = {"harmfulness": "o", "refusal": "^"}
    for target in ["harmfulness", "refusal"]:
        ax.plot(x, y_data[target], marker=_markers[target], linewidth=2.0, markersize=4,
                color=STEER_COLORS[target], zorder=3)

    ax.set_ylim(0, y_max)

    for target in ["harmfulness", "refusal"]:
        y_arr = y_data[target]
        if not np.isnan(y_arr).all():
            peak_x = int(np.nanargmax(y_arr))
            peak_y = float(y_arr[peak_x])
            layer_num = layers[peak_x].replace("L", "")
            color = STEER_COLORS[target]
            offset_y, va = (-9, "top") if peak_y > y_max * 0.85 else (8, "bottom")
            ax.annotate(f"L{layer_num}", xy=(peak_x, peak_y), xytext=(0, offset_y),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=tick_fs, color=color, fontweight="bold",
                        fontfamily=FONT_NAME, clip_on=False, zorder=6)

    ax.set_xlim(-0.5, len(layers) - 0.5)
    step = 4
    ticks = list(range(0, len(layers), step))
    if ticks and ticks[-1] != len(layers) - 1:
        if len(layers) - 1 - ticks[-1] <= step // 2:
            ticks.pop()
        ticks.append(len(layers) - 1)
    ax.set_xticks(ticks)
    if show_xlabel:
        ax.set_xticklabels([layers[i].replace("L", "") for i in ticks], rotation=0,
                            ha="center", fontsize=tick_fs)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=tick_fs)
    if show_ylabel:
        ax.set_ylabel("Refusal Rate (%)", fontsize=axis_fs)
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)


def render_fig3(output_dir: Path, dpi: int = 160) -> None:
    """Fig3: Alpaca steering — harmfulness + refusal lines per model."""
    TICK_FS, AXIS_FS, LEG_FS = 15, 18, 14

    models_cfg = [
        ("llama", "(a) LLaMA 3.1", DATA["fig3_llama_agg"], (8, 11),  False),
        ("qwen",  "(b) Qwen 2.5",  DATA["fig3_qwen_agg"],  (13, 15), True),
    ]

    # Load cell info per (model, target)
    cells = {}
    for m, _, agg_path, _, _ in models_cfg:
        cells[(m, "harmfulness")] = _load_steering_cell(agg_path, "harmfulness")
        cells[(m, "refusal")]     = _load_steering_cell(agg_path, "refusal")

    n_alphas   = len(cells[(models_cfg[0][0], "harmfulness")][0])
    max_layers = max(len(cells[(m, t)][1]) for m, _, _, _, _ in models_cfg for t in ("harmfulness", "refusal"))

    fig_w = max(9, max_layers * 0.18)
    fig_h = 3.2 * n_alphas * len(models_cfg) + 0.6
    fig   = plt.figure(figsize=(fig_w, fig_h))
    leg_frac = 0.62 / fig_h

    outer_gs = gridspec.GridSpec(len(models_cfg), 1, figure=fig,
                                 top=1.0 - leg_frac, bottom=0.09,
                                 left=0.245, right=0.96, hspace=0.55)

    n_macro_r = len(models_cfg)
    for ri, (m, m_label, _, bridge_layers, autoscale) in enumerate(models_cfg):
        harm_alphas, harm_layers, harm_idx = cells[(m, "harmfulness")]
        _,           ref_layers,  ref_idx  = cells[(m, "refusal")]

        layers = _sorted_layer_names(set(harm_layers) | set(ref_layers))
        alphas = harm_alphas

        if autoscale:
            all_vals = [
                idx[(lg, a)]["refusal"]["percentage"]
                for a in alphas
                for _, idx in [("harmfulness", harm_idx), ("refusal", ref_idx)]
                for lg in layers if (lg, a) in idx
            ]
            all_vals = [v for v in all_vals if not math.isnan(v)]
            if all_vals:
                top = max(all_vals)
                y_max = float(math.ceil(top / 5) * 5 if top <= 30 else math.ceil(top / 10) * 10)
                y_max = max(y_max, 10.0)
            else:
                y_max = 100.0
        else:
            y_max = 100.0

        inner_gs = gridspec.GridSpecFromSubplotSpec(n_alphas, 1, subplot_spec=outer_gs[ri, 0],
                                                    hspace=0.12)
        for ai, alpha in enumerate(alphas):
            ax = fig.add_subplot(inner_gs[ai])
            is_last_alpha = (ai == n_alphas - 1)
            is_bottom     = (ri == n_macro_r - 1) and is_last_alpha
            _draw_merged_steering(ax, harm_idx, ref_idx, layers, alpha,
                                  bridge_layers=bridge_layers, y_max=y_max,
                                  show_xlabel=is_last_alpha, show_ylabel=False,
                                  tick_fs=TICK_FS, axis_fs=AXIS_FS)
            if is_bottom:
                ax.set_xlabel("Steered Layer", fontsize=AXIS_FS)

    # Legend
    has_bridge = any(cfg[3] is not None for cfg in models_cfg)
    handles = [
        plt.Line2D([0], [0], color=STEER_COLORS["harmfulness"], linewidth=2.0,
                   marker="o", markersize=5, label=STEER_LABELS["harmfulness"]),
        plt.Line2D([0], [0], color=STEER_COLORS["refusal"], linewidth=2.0,
                   marker="^", markersize=5, label=STEER_LABELS["refusal"]),
    ]
    if has_bridge:
        handles.append(mpatches.Patch(facecolor=BRIDGE_COLOR, edgecolor="none", alpha=0.7, label="Bridge"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.545, 0.995),
               ncol=len(handles), fontsize=LEG_FS, frameon=True, fancybox=False,
               edgecolor="lightgray", facecolor="white", framealpha=1.0,
               handlelength=2.0, columnspacing=1.4, borderpad=0.45)

    # Row labels (model names)
    for ri, (_, m_label, _, _, _) in enumerate(models_cfg):
        pos = outer_gs[ri, 0].get_position(fig)
        y_center = (pos.y0 + pos.y1) / 2
        offset = 0.105 + min(0.035, max(0, len(m_label) - 14) * 0.004)
        fig.text(max(0.045, pos.x0 - offset), y_center, m_label,
                 fontsize=26, fontweight="bold", rotation=90, ha="center", va="center")
        fig.text(0.185, y_center, "Refusal Rate (%)", fontsize=AXIS_FS,
                 rotation=90, ha="center", va="center")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig3.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig3_2 — tinst-restore bridge/readout bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_delta_drop_bars(ax, drops, drop_stderrs, colors, show_ylabel, xtick_labels=None):
    x = np.arange(len(drops))
    delta_values = -drops
    bars = ax.bar(x, delta_values, color=colors, edgecolor="#20242b", linewidth=1.0, width=0.62)
    ax.axhline(0.0, color="#20242b", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(x, xtick_labels or ["Bridge", "Readout"])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.tick_params(axis="both", labelsize=18, length=4, width=1)
    if show_ylabel:
        ax.set_ylabel(r"$\Delta$ Refusal Projection", fontsize=22, labelpad=12)
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
            ha="center", va="top",
            fontsize=15, fontweight="bold", color="black",
        )


def render_fig3_2(output_dir: Path, dpi: int = 200) -> None:
    """Fig3_2: Bridge/Readout restoration bar charts for LLaMA and Qwen."""
    models_cfg = [
        ("llama31", DATA["fig3_2_llama"],
         ["Bridge\n($l=8$-$11$)",  "Readout\n($l=12$-$20$)"], "(a) LLaMA 3.1"),
        ("qwen25",  DATA["fig3_2_qwen"],
         ["Bridge\n($l=13$-$15$)", "Readout\n($l=18$-$22$)"], "(b) Qwen 2.5"),
    ]
    labels = ["patch_bridge", "patch_readout"]
    colors = ["#3f7f6b", "#b45b55"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharey=False)
    for idx, (_, path, xtick_labels, caption) in enumerate(models_cfg):
        payload = _load(path)
        drops       = np.array([payload["summaries"][lbl]["drop_vs_no_patch_mean"]   for lbl in labels])
        drop_stderrs = np.array([payload["summaries"][lbl]["drop_vs_no_patch_stderr"] for lbl in labels])
        delta = -drops
        low   = min(float((delta - drop_stderrs).min()), 0.0)
        pad   = max((0.0 - low) * 0.18, 0.05)
        _plot_delta_drop_bars(axes[idx], drops, drop_stderrs, colors,
                              show_ylabel=(idx == 0), xtick_labels=xtick_labels)
        val_range = max(abs(low), 0.1)
        axes[idx].set_ylim(low - pad - val_range * 0.10, 0.0 + pad * 0.3)
        axes[idx].tick_params(axis="y", labelleft=True)
        axes[idx].text(0.5, -0.31, caption,
                       transform=axes[idx].transAxes, ha="center", va="top",
                       fontsize=30, fontweight="bold")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.31, top=0.94, wspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig4.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig4 — activation patching heatmap
# Fig4_2 — behavioral validation line chart
# ═══════════════════════════════════════════════════════════════════════════════

PATCHING_MODEL_LABEL_FS = 28
VARIANTS = [("full", "Full"), ("indirect", "Indirect")]


def _mask_non_downstream(values: np.ndarray) -> np.ndarray:
    inj = np.arange(values.shape[0])[:, None]
    dwn = np.arange(values.shape[1])[None, :]
    masked = values.copy()
    masked[dwn <= inj] = np.nan
    return masked


def render_fig4(output_dir: Path, dpi: int = 300) -> None:
    """Fig4: activation patching heatmap (Full / Indirect) for LLaMA and Qwen."""
    models_cfg = [
        ("llama", DATA["fig4_llama"], "(a) LLaMA 3.1"),
        ("qwen",  DATA["fig4_qwen"],  "(b) Qwen 2.5"),
    ]
    payloads = {m: _load(p) for m, p, _ in models_cfg}

    def _coupling(payload, variant):
        coupling = payload.get("activation_patching_coupling") or payload["path_patching_coupling"]
        return np.array(coupling[variant]["normalized_coupling_map"], dtype=np.float64)

    maps = {m: {v: _coupling(payloads[m], v) for v, _ in VARIANTS} for m, _, _ in models_cfg}
    row_limits = {m: _symmetric_limits([maps[m][v] for v, _ in VARIANTS]) for m, _, _ in models_cfg}

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(len(models_cfg), 2,
                             figsize=(9.8, 4.35 * len(models_cfg) + 0.4), squeeze=False)
    row_images = {}
    for ri, (m, _, row_label) in enumerate(models_cfg):
        for ci, (variant, variant_label) in enumerate(VARIANTS):
            ax = axes[ri, ci]
            values = _mask_non_downstream(maps[m][variant])
            masked = np.ma.masked_invalid(values)
            vmin, vmax = row_limits[m]
            image = ax.imshow(masked, origin="lower", aspect="equal",
                              cmap=cmap, vmin=vmin, vmax=vmax)
            row_images[ri] = image
            ax.set_box_aspect(values.shape[0] / values.shape[1])
            ax.set_xticks(layer_ticks(values.shape[1]))
            ax.set_yticks(layer_ticks(values.shape[0]))
            ax.tick_params(axis="both", labelsize=16, length=4, width=1, pad=3)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            if ri == 0:
                ax.set_title(variant_label, fontsize=22, pad=10)
            if ri == len(models_cfg) - 1:
                ax.set_xlabel("Downstream Layer", fontsize=22, labelpad=12)
            if ci == 0:
                ax.set_ylabel("Injection Layer", fontsize=22, labelpad=12)
        axes[ri, 0].text(-0.36, 0.5, row_label,
                         transform=axes[ri, 0].transAxes, rotation=90,
                         ha="center", va="center",
                         fontsize=PATCHING_MODEL_LABEL_FS, fontweight="bold")

    fig.subplots_adjust(left=0.21, right=0.90, bottom=0.10, top=0.93, wspace=0.20, hspace=0.18)
    for ri, image in row_images.items():
        anchor = axes[ri, -1].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.018, anchor.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.tick_params(labelsize=16, length=4, width=1)
        cbar.outline.set_linewidth(1.0)
        cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=22, labelpad=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig6.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


def render_fig4_2(output_dir: Path, dpi: int = 300) -> None:
    """Fig4_2: behavioral validation (delta refusal rate by steered layer)."""
    models_cfg = [
        ("llama", DATA["fig4_llama"], "(a) LLaMA 3.1"),
        ("qwen",  DATA["fig4_qwen"],  "(b) Qwen 2.5"),
    ]

    def _behavioral(payload):
        behavioral = payload.get("behavioral_validation") or payload.get("behavioral_coupling")
        if not behavioral or not behavioral.get("experiments"):
            raise ValueError("No behavioral_validation in JSON")
        exps = sorted(behavioral["experiments"], key=lambda e: int(e["layer"]))
        layers = np.array([int(e["layer"]) for e in exps], dtype=int)
        if "delta_refusal_rate" in exps[0]:
            full     = np.array([e.get("delta_refusal_rate", np.nan) for e in exps])
            indirect = np.full_like(full, np.nan)
        else:
            full     = np.array([e.get("delta_from_baseline_full", np.nan) for e in exps])
            indirect = np.array([e.get("indirect_delta", np.nan) for e in exps])
        return layers, full, indirect

    payloads = {m: _load(p) for m, p, _ in models_cfg}
    series   = {m: _behavioral(payloads[m]) for m, _, _ in models_cfg}

    finite = []
    for _, full, indirect in series.values():
        finite.extend(full[np.isfinite(full)])
        finite.extend(indirect[np.isfinite(indirect)])
    ymin = min(0.0, float(np.min(finite))) - 0.03 if finite else -0.05
    ymax = max(0.0, float(np.max(finite))) + 0.03 if finite else 0.05

    fig, axes = plt.subplots(len(models_cfg), 1, figsize=(7.2, 3.1 * len(models_cfg)), sharex=False)
    axes = np.array(axes).reshape(len(models_cfg))
    for ri, (m, _, row_label) in enumerate(models_cfg):
        ax = axes[ri]
        layers, full, indirect = series[m]
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.85)
        ax.plot(layers, full, marker="o", markersize=5.5, linewidth=2.0, color="#d73027", label="Full")
        if np.isfinite(indirect).any():
            ax.plot(layers, indirect, marker="^", markersize=6.0, linewidth=2.0,
                    color="#4575b4", label="Indirect")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(float(np.min(layers)) - 0.5, float(np.max(layers)) + 0.5)
        ax.set_xticks(layer_ticks(int(np.max(layers)) + 1))
        ax.set_ylabel(r"$\Delta$ Refusal Rate", fontsize=18, labelpad=9)
        if ri == len(models_cfg) - 1:
            ax.set_xlabel("Steered Layer", fontsize=18, labelpad=8)
        ax.tick_params(axis="both", labelsize=15, length=4, width=1)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.22, 0.5, row_label, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=26, fontweight="bold")
        if ri == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=14, ncol=2, handlelength=1.9)

    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.10, top=0.98, hspace=0.34)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig5.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig6 — steering adversarial datasets (harmfulness only)
# ═══════════════════════════════════════════════════════════════════════════════

def render_fig6(output_dir: Path, dpi: int = 160) -> None:
    """Fig6: adversarial dataset steering — LLaMA and Qwen merged as two lines per subplot."""
    TICK_FS, AXIS_FS, LEG_FS = 15, 18, 14

    MODEL_COLORS = {"llama": STEER_COLORS["harmfulness"], "qwen": STEER_COLORS["refusal"]}
    MODEL_LABELS_TEXT = {"llama": "LLaMA 3.1", "qwen": "Qwen 2.5"}

    datasets_cfg = [
        ("misrepresentation",     "Misrepresentation"),
        ("authority_endorsement", "Authority Endorsement"),
        ("expert_endorsement",    "Expert Endorsement"),
    ]
    models_cfg = [
        ("llama", {
            "misrepresentation":     DATA["fig6_llama_misrep"],
            "authority_endorsement": DATA["fig6_llama_authority"],
            "expert_endorsement":    DATA["fig6_llama_expert"],
        }),
        ("qwen", {
            "misrepresentation":     DATA["fig6_qwen_misrep"],
            "authority_endorsement": DATA["fig6_qwen_authority"],
            "expert_endorsement":    DATA["fig6_qwen_expert"],
        }),
    ]

    # Load cell info
    cell_info = {}
    for m, agg_paths in models_cfg:
        for ds, _ in datasets_cfg:
            alphas, layers, idx = _load_steering_cell(agg_paths[ds], "harmfulness")
            cell_info[(m, ds)] = (alphas, layers, idx)

    ref_alphas = cell_info[(models_cfg[0][0], datasets_cfg[0][0])][0]
    n_alphas = len(ref_alphas)
    n_cols = len(datasets_cfg)

    max_layer_num = max(
        max(int(lg.replace("L", "")) for lg in cell_info[(m, ds)][1])
        for m, _ in models_cfg for ds, _ in datasets_cfg
    )

    fig_w = max(10, (max_layer_num + 1) * 0.09 * n_cols)
    fig_h = 2.8 * n_alphas + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    leg_frac = 0.55 / fig_h
    hdr_frac = 0.45 / fig_h
    top_frac = leg_frac + hdr_frac   # legend above, column headers below it

    outer_gs = gridspec.GridSpec(n_alphas, n_cols, figure=fig,
                                 top=1.0 - top_frac,
                                 bottom=0.08,
                                 left=0.12, right=0.98,
                                 hspace=0.40, wspace=0.14)

    step = 4
    x_ticks = list(range(0, max_layer_num + 1, step))
    if x_ticks and x_ticks[-1] != max_layer_num:
        if max_layer_num - x_ticks[-1] <= step // 2:
            x_ticks.pop()
        x_ticks.append(max_layer_num)

    for ai, alpha in enumerate(ref_alphas):
        for ci, (ds, _) in enumerate(datasets_cfg):
            ax = fig.add_subplot(outer_gs[ai, ci])
            is_last = (ai == n_alphas - 1)

            for m, _ in models_cfg:
                _, layers_m, idx_m = cell_info[(m, ds)]
                x_vals = [int(lg.replace("L", "")) for lg in layers_m]
                y_vals = [
                    idx_m[(lg, alpha)]["refusal"]["percentage"] if (lg, alpha) in idx_m else np.nan
                    for lg in layers_m
                ]
                ax.plot(x_vals, np.array(y_vals, dtype=float),
                        marker="o", linewidth=2.0, markersize=4,
                        color=MODEL_COLORS[m], zorder=3)

            ax.set_xlim(-0.5, max_layer_num + 0.5)
            ax.set_xticks(x_ticks)
            if is_last:
                ax.set_xticklabels([str(t) for t in x_ticks], rotation=0,
                                    ha="center", fontsize=TICK_FS)
                ax.set_xlabel("Steered Layer", fontsize=AXIS_FS)
            else:
                ax.set_xticklabels([])
            ax.set_ylim(0, 100)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            if ci == 0:
                ax.set_ylabel("Refusal Rate (%)", fontsize=AXIS_FS, labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.grid(axis="y", alpha=0.2, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)

    # Column headers (sit between legend and grid)
    hdr_y = 1.0 - leg_frac - hdr_frac / 2
    for ci, (_, ds_label) in enumerate(datasets_cfg):
        pos = outer_gs[0, ci].get_position(fig)
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, hdr_y, ds_label,
                 fontsize=16, fontweight="bold", ha="center", va="center")

    # Legend at top, centered over full figure width
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.0,
                   marker="o", markersize=5, label=MODEL_LABELS_TEXT[m])
        for m, _ in models_cfg
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles), fontsize=LEG_FS, frameon=True, fancybox=False,
               edgecolor="lightgray", facecolor="white", framealpha=1.0,
               handlelength=2.0, columnspacing=1.4, borderpad=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig8.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig7 & Fig8 — clean projection line charts
# ═══════════════════════════════════════════════════════════════════════════════

PROJ_TICK_FS   = 16
PROJ_AXIS_FS   = 22
PROJ_TITLE_FS  = 28
PROJ_LEGEND_FS = 16

FIG_PROJ_CONFIG = {
    "fig7": {"ylabel": "Harmfulness Projection", "llama_key": "fig7_llama", "qwen_key": "fig7_qwen"},
    "fig8": {"ylabel": "Refusal Projection",     "llama_key": "fig8_llama", "qwen_key": "fig8_qwen"},
}


def _projection_limits(summaries):
    values = []
    for ds_key, _, _ in PROJECTION_DATASETS:
        if ds_key not in summaries:
            continue
        means = np.array(summaries[ds_key]["all_layer_mean"], dtype=np.float64)
        stds  = np.array(summaries[ds_key]["all_layer_std"],  dtype=np.float64)
        values.extend((means - stds)[np.isfinite(means - stds)])
        values.extend((means + stds)[np.isfinite(means + stds)])
    if not values:
        return -1.0, 1.0
    ymin, ymax = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max((ymax - ymin) * 0.08, 0.05)
    return ymin - margin, ymax + margin


def _render_projection(fig_name: str, output_dir: Path, dpi: int = 300) -> None:
    cfg = FIG_PROJ_CONFIG[fig_name]
    models = [
        ("llama31", DATA[cfg["llama_key"]], "(a) LLaMA 3.1"),
        ("qwen25",  DATA[cfg["qwen_key"]],  "(b) Qwen 2.5"),
    ]
    summaries_by_model = {}
    for m, path, _ in models:
        payload = _load(path)
        summaries_by_model[m] = payload["summaries"]

    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 5.4),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, (m, _, row_label) in zip(axes, models):
        summaries = summaries_by_model[m]
        ymin, ymax = _projection_limits(summaries)
        max_layers = 0
        for ds_key, ds_label, color in PROJECTION_DATASETS:
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
        ax.set_xticks(layer_ticks(max_layers))
        ax.set_xlabel("Layer", fontsize=PROJ_AXIS_FS, labelpad=10)
        ax.text(0.5, -0.27, row_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=PROJ_TITLE_FS, fontweight="bold")
        ax.tick_params(axis="both", labelsize=PROJ_TICK_FS, length=4, width=1, pad=3)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(cfg["ylabel"], fontsize=PROJ_AXIS_FS, labelpad=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025),
               ncol=5, fontsize=PROJ_LEGEND_FS, frameon=True, fancybox=False,
               edgecolor="lightgray", facecolor="white", framealpha=1.0,
               handlelength=2.2, columnspacing=1.2, borderpad=0.45)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.24, top=0.84, wspace=0.14)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / ("Fig9_1.png" if fig_name == "fig7" else "Fig9_2.png")
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


def render_fig7(output_dir: Path, dpi: int = 300) -> None:
    _render_projection("fig7", output_dir, dpi)


def render_fig8(output_dir: Path, dpi: int = 300) -> None:
    _render_projection("fig8", output_dir, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig9 — injection-delta heatmap grid
# ═══════════════════════════════════════════════════════════════════════════════

FIG9_DATASETS = ["alpaca", "misrepresentation", "authority_endorsement", "expert_endorsement"]
FIG9_TICK_FS       = 16
FIG9_AXIS_FS       = 22
FIG9_CBAR_TICK_FS  = 16
FIG9_CBAR_LABEL_FS = 22
FIG9_ROW_LABEL_FS  = 28


def _get_delta_map(payload, dataset_name) -> np.ndarray:
    summaries = payload.get("injection_delta_summaries") or {}
    if dataset_name not in summaries:
        raise KeyError(f"Missing injection_delta_summaries['{dataset_name}']")
    values = summaries[dataset_name]["delta_mean_map"]
    return np.array(values, dtype=np.float64)


def _apply_downstream_mask(matrix: np.ndarray) -> np.ndarray:
    inj = np.arange(matrix.shape[0])[:, None]
    rdo = np.arange(matrix.shape[1])[None, :]
    masked = matrix.copy()
    masked[rdo <= inj] = np.nan
    return masked


def _finite_symmetric_vmax(maps: list[np.ndarray]) -> float:
    masked = [_apply_downstream_mask(m) for m in maps]
    values = np.concatenate([m[np.isfinite(m)].reshape(-1) for m in masked if np.isfinite(m).any()])
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(values), 98))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def render_fig9(output_dir: Path, dpi: int = 200) -> None:
    """Fig9: injection-delta heatmap grid — refusal_tpost, alpha=3."""
    models_order = ["llama31", "qwen25"]
    model_labels = {"llama31": "(a) LLaMA 3.1", "qwen25": "(b) Qwen 2.5"}
    payloads = {
        "llama31": _load(DATA["fig9_llama"]),
        "qwen25":  _load(DATA["fig9_qwen"]),
    }

    maps: dict[str, dict[str, np.ndarray]] = {
        m: {ds: _get_delta_map(payloads[m], ds) for ds in FIG9_DATASETS}
        for m in models_order
    }

    model_vmax = {
        m: _finite_symmetric_vmax([maps[m][ds] for ds in FIG9_DATASETS])
        for m in models_order
    }

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")
    n_rows, n_cols = len(models_order), len(FIG9_DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.35 * n_cols + 1.35, 4.35 * n_rows + 0.25),
                             squeeze=False)

    row_images = {}
    for ri, m in enumerate(models_order):
        for ci, ds in enumerate(FIG9_DATASETS):
            ax = axes[ri][ci]
            matrix = _apply_downstream_mask(maps[m][ds])
            vmax = model_vmax[m]
            image = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax,
                              aspect="equal", origin="lower")
            row_images[m] = image
            ax.set_box_aspect(matrix.shape[0] / matrix.shape[1])
            ax.set_xticks(layer_ticks(matrix.shape[1]))
            ax.set_yticks(layer_ticks(matrix.shape[0]))
            ax.tick_params(axis="both", labelsize=FIG9_TICK_FS, length=4, width=1, pad=3)
            ax.grid(False)
            if ri == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=FIG9_AXIS_FS,
                             fontweight="bold", pad=12)
            if ci == 0:
                ax.set_ylabel("Injection Layer", fontsize=FIG9_AXIS_FS, labelpad=12)
            if ri == n_rows - 1:
                ax.set_xlabel("Readout Layer", fontsize=FIG9_AXIS_FS, labelpad=12)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
        axes[ri][0].text(-0.36, 0.5, model_labels[m],
                         transform=axes[ri][0].transAxes, rotation=90,
                         ha="center", va="center",
                         fontsize=FIG9_ROW_LABEL_FS, fontweight="bold")

    fig.subplots_adjust(left=0.205, right=0.91, bottom=0.08, top=0.92, wspace=0.20, hspace=0.02)
    for ri, m in enumerate(models_order):
        anchor = axes[ri][-1].get_position()
        cax = fig.add_axes([anchor.x1 + 0.012, anchor.y0, 0.016, anchor.height])
        cbar = fig.colorbar(row_images[m], cax=cax)
        cbar.ax.tick_params(labelsize=FIG9_CBAR_TICK_FS, length=4, width=1)
        cbar.set_label(r"$\Delta$ Refusal Projection", fontsize=FIG9_CBAR_LABEL_FS, labelpad=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Fig10.png"
    _save_figure(fig, out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FIGS = {
    "fig2":   render_fig2,
    "fig3":   render_fig3,
    "fig3_2": render_fig3_2,
    "fig4":   render_fig4,
    "fig4_2": render_fig4_2,
    "fig5":   render_fig5,
    "fig6":   render_fig6,
    "fig7":   render_fig7,
    "fig8":   render_fig8,
    "fig9":   render_fig9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all paper figures from saved result files.")
    parser.add_argument("--fig", choices=list(ALL_FIGS) + ["all"], default="all",
                        help="Which figure to render (default: all)")
    parser.add_argument("--output-dir", default=str(OUT_PT / "Figure"),
                        help="Output directory (default: out_pt/Figure/)")
    parser.add_argument("--dpi", type=int, default=None,
                        help="Override DPI for all figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figs = list(ALL_FIGS.items()) if args.fig == "all" else [(args.fig, ALL_FIGS[args.fig])]

    for name, fn in figs:
        print(f"\n── {name} ──────────────────────────────")
        import inspect
        sig = inspect.signature(fn)
        kwargs = {"output_dir": output_dir}
        if args.dpi is not None and "dpi" in sig.parameters:
            kwargs["dpi"] = args.dpi
        fn(**kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
