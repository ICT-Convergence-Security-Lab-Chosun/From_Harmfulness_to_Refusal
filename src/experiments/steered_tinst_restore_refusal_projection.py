#!/usr/bin/env python3
"""
Steered t_inst Restoration -> Final t_post Refusal Projection
=============================================================

Inject the harmfulness vector at t_inst, restore clean t_inst hidden states over
preconfigured layer spans, and measure the final-layer t_post refusal projection.

Default experiment:
  - LLaMA 3.1: steer L3, Bridge L8-L11, Readout L12-L20
  - Qwen 2.5:  steer L7, Bridge L13-L15, Readout L18-L22
  - Alpaca common 100 samples
  - alpha = 3.0

Layer labels are zero-based by default, matching the rest of this repo's L{idx}
plots and filenames.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from tqdm import tqdm

from activation_patching_coupling import (
    build_direction_candidates,
    direction_for_layer,
    gather_token_rows,
    get_restore_t_inst_hook,
    resolve_base_dir,
    resolve_existing_path,
    validate_direction_matrix,
)
from activation_steering import (
    build_component_modules,
    build_prompts,
    get_input_device,
    get_prefill_only_steering_hook,
    load_direction_matrices,
    normalize_direction_matrix,
    resolve_target_positions_for_batch,
)
from extract_single_activations import load_model_and_tokenizer
from model_utils import normalize_model_name, resolve_model_spec
from utils import read_row


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "alpaca_common.json"
DEFAULT_BASE_DIR = SCRIPT_DIR / "out_pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "steered_tinst_restore_refusal_projection"
DEFAULT_FIGURE_PATH = DEFAULT_BASE_DIR / "Figure" / "Fig4.png"

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

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama31": {
        "display": "LLaMA 3.1 8B",
        "steer_layer": 3, #3
        "patch_spans": {
            "patch_bridge": list(range(8, 12)),
            "patch_readout": list(range(12, 21)),
        },
    },
    "qwen25": {
        "display": "Qwen 2.5 7B",
        "steer_layer": 7,
        "patch_spans": {
            "patch_bridge": list(range(13, 16)),
            "patch_readout": list(range(18, 23)),
        },
    },
}

CONDITIONS = ["no_patch", "patch_bridge", "patch_readout"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject harmfulness at t_inst, restore clean t_inst hidden states over "
            "Bridge/Readout spans, and measure final-layer t_post refusal projection."
        )
    )
    parser.add_argument("--model", default=None, help="Single model alias, e.g. llama31 or qwen25.")
    parser.add_argument("--model-id", default=None, help="HF model id. Preferred over --model.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model aliases. Use 'both' or omit model args for llama31,qwen25.",
    )
    parser.add_argument("--parallel-models", action="store_true")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--component", default="hidden", choices=["hidden"])
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmfulness-vector", default=None)
    parser.add_argument("--refusal-vector", default=None)
    parser.add_argument("--normalize-injection-direction", action="store_true")
    parser.add_argument(
        "--normalize-measurement-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--direction-norm-eps", type=float, default=1e-8)
    parser.add_argument("--use-persuade", type=int, default=0)
    parser.add_argument("--use-sys", type=int, default=0)
    parser.add_argument("--use-template", type=int, default=1)
    parser.add_argument("--do-not-use-last-inst-tok", type=int, default=0)
    parser.add_argument("--use-inversion", type=int, default=0)
    parser.add_argument("--inversion-prompt-idx", type=int, default=0)
    parser.add_argument("--save-per-example", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--auto-plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Render the combined LLaMA/Qwen figure from the latest saved JSON files without rerunning models.",
    )
    parser.add_argument("--figure-output", default=str(DEFAULT_FIGURE_PATH))
    parser.add_argument("--plot-dpi", type=int, default=200)
    args = parser.parse_args()

    if args.models is None and args.model is None and args.model_id is None:
        args.models = "llama31,qwen25"

    if args.models:
        return args

    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    args.model = normalize_model_name(spec.model_name)
    args.model_id = spec.model_id
    return args


def parse_model_list(raw: str) -> list[str]:
    if raw.strip().lower() in {"both", "all"}:
        raw = "llama31,qwen25"
    models = [normalize_model_name(item.strip()) for item in raw.split(",") if item.strip()]
    models = [model for model in models if model]
    if not models:
        raise ValueError("--models must contain at least one model alias.")
    return list(dict.fromkeys(models))


def child_argv_for_model(args: argparse.Namespace, model_name: str) -> list[str]:
    child_args: list[str] = []
    skip_next = False
    names_with_values = {"--models", "--model", "--model-id", "--output-dir"}
    flag_names = {"--parallel-models"}
    original = sys.argv[1:]
    for idx, arg in enumerate(original):
        if skip_next:
            skip_next = False
            continue
        if arg in flag_names:
            continue
        if arg in names_with_values:
            skip_next = idx + 1 < len(original)
            continue
        if any(arg.startswith(f"{name}=") for name in names_with_values | flag_names):
            continue
        child_args.append(arg)

    child_args.extend(["--model", model_name])
    child_args.extend(["--output-dir", str(Path(args.output_dir) / model_name)])
    return [sys.executable, str(Path(__file__).resolve()), *child_args]


def run_model_jobs(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    commands = [child_argv_for_model(args, model_name) for model_name in models]
    print(
        f"[tinst-restore] launching {len(commands)} model job(s): {', '.join(models)} "
        f"({'parallel' if args.parallel_models else 'sequential'})",
        flush=True,
    )

    if args.parallel_models:
        procs = [(model_name, subprocess.Popen(cmd)) for model_name, cmd in zip(models, commands)]
        failures: list[tuple[str, int]] = []
        for model_name, proc in procs:
            code = proc.wait()
            if code != 0:
                failures.append((model_name, code))
        if failures:
            failed = ", ".join(f"{model_name}={code}" for model_name, code in failures)
            raise subprocess.CalledProcessError(failures[0][1], f"parallel model jobs failed: {failed}")
        return

    for model_name, cmd in zip(models, commands):
        print(f"\n[tinst-restore] starting {model_name}", flush=True)
        subprocess.run(cmd, check=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path, start: int, count: int) -> list[Any]:
    rows = read_row(str(path))[start : start + count]
    if not rows:
        raise ValueError(f"No rows selected from {path}")
    return rows


def resolve_direction_paths(args: argparse.Namespace) -> dict[str, Path]:
    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    base_dir = resolve_base_dir(args.base_dir)

    def _resolve(override: str | None, mode_dir: str, suffix: str) -> Path:
        if override:
            return Path(override).expanduser()
        return resolve_existing_path(
            build_direction_candidates(
                base_dir=base_dir,
                output_name=spec.output_name,
                model_alias=args.model,
                mode_dir=mode_dir,
                component=args.component,
                suffix=suffix,
            )
        )

    return {
        "harmfulness": _resolve(args.harmfulness_vector, "hf", "harmful-minus-harmless"),
        "refusal": _resolve(args.refusal_vector, "refuse", "refuse-minus-accept"),
    }


def project_final_refusal(
    hidden_states: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
    refusal_directions: torch.Tensor,
    final_layer: int,
) -> torch.Tensor:
    activation = hidden_states[final_layer + 1]
    selected = gather_token_rows(activation, positions.to(activation.device))
    direction = direction_for_layer(refusal_directions, final_layer).to(
        device=selected.device,
        dtype=torch.float32,
    )
    return selected.float().matmul(direction).detach().cpu()


def condition_patch_layers(model_name: str, condition: str) -> list[int]:
    if condition == "no_patch":
        return []
    return list(MODEL_CONFIGS[model_name]["patch_spans"][condition])


def summarize_values(values: torch.Tensor) -> dict[str, float]:
    values = values.to(dtype=torch.float64)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "stderr": float(values.std(unbiased=False).item() / max(values.numel(), 1) ** 0.5),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def compute_experiment(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    harmfulness_directions: torch.Tensor,
    refusal_directions: torch.Tensor,
) -> dict[str, Any]:
    modules = build_component_modules(model, args.component)
    num_layers = len(modules)
    config = MODEL_CONFIGS[args.model]
    steer_layer = int(config["steer_layer"])
    final_layer = num_layers - 1
    input_device = get_input_device(model)

    if steer_layer < 0 or steer_layer >= num_layers:
        raise IndexError(f"Steer layer {steer_layer} out of range for {num_layers}-layer model.")
    for condition in CONDITIONS:
        for layer in condition_patch_layers(args.model, condition):
            if layer < 0 or layer >= num_layers:
                raise IndexError(f"{condition} layer {layer} out of range for {num_layers}-layer model.")

    condition_batches: dict[str, list[torch.Tensor]] = {condition: [] for condition in CONDITIONS}
    clean_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc=f"{args.model} restore conditions"):
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(input_device)
            attention_mask = inputs.attention_mask.to(input_device)
            prefill_len = int(input_ids.shape[1])

            shared_pos_kwargs = dict(
                tokenizer=tokenizer,
                batch_prompts=batch_prompts,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                model_name=args.model,
                model_id=args.model_id,
                use_template=bool(args.use_template),
                do_not_use_last_inst_tok=bool(args.do_not_use_last_inst_tok),
            )
            t_inst = resolve_target_positions_for_batch(target="harmfulness", **shared_pos_kwargs)
            t_post = resolve_target_positions_for_batch(target="refusal", **shared_pos_kwargs)

            clean_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            clean_batches.append(
                project_final_refusal(clean_out.hidden_states, t_post, refusal_directions, final_layer)
            )

            needed_patch_layers = sorted(
                {
                    layer
                    for condition in CONDITIONS
                    for layer in condition_patch_layers(args.model, condition)
                }
            )
            clean_tinst_acts = {
                layer: gather_token_rows(
                    clean_out.hidden_states[layer + 1].detach().cpu(),
                    t_inst.cpu(),
                )
                for layer in needed_patch_layers
            }

            inject_direction = direction_for_layer(harmfulness_directions, steer_layer)
            inject_hook = get_prefill_only_steering_hook(
                direction=inject_direction,
                alpha=float(args.alpha),
                positions=t_inst,
                attention_mask=inputs.attention_mask,
            )

            for condition in CONDITIONS:
                with ExitStack() as stack:
                    stack.callback(modules[steer_layer].register_forward_hook(inject_hook).remove)
                    for patch_layer in condition_patch_layers(args.model, condition):
                        restore_hook = get_restore_t_inst_hook(
                            clean_activations=clean_tinst_acts[patch_layer],
                            positions=t_inst,
                            prefill_seq_len=prefill_len,
                        )
                        stack.callback(modules[patch_layer].register_forward_hook(restore_hook).remove)

                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                condition_batches[condition].append(
                    project_final_refusal(out.hidden_states, t_post, refusal_directions, final_layer)
                )
                del out

            del clean_out, clean_tinst_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    condition_values = {
        condition: torch.cat(batches, dim=0).to(dtype=torch.float64)
        for condition, batches in condition_batches.items()
    }
    clean_values = torch.cat(clean_batches, dim=0).to(dtype=torch.float64)
    no_patch = condition_values["no_patch"]

    summaries = {}
    for condition, values in condition_values.items():
        summary = summarize_values(values)
        drop = no_patch - values
        summary.update(
            {
                "drop_vs_no_patch_mean": float(drop.mean().item()),
                "drop_vs_no_patch_std": float(drop.std(unbiased=False).item()),
                "drop_vs_no_patch_stderr": float(
                    drop.std(unbiased=False).item() / max(drop.numel(), 1) ** 0.5
                ),
            }
        )
        summaries[condition] = summary

    return {
        "clean_projection": clean_values,
        "condition_values": condition_values,
        "summaries": summaries,
        "config": {
            "steer_layer": steer_layer,
            "final_measure_layer": final_layer,
            "patch_spans": {
                key: value for key, value in config["patch_spans"].items()
            },
        },
    }


def make_output_base(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_label = f"alpha_{float(args.alpha):g}".replace("-", "m").replace(".", "p")
    custom = f"-{args.label}" if args.label else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir / f"{args.model}-tinst-restore-final-tpost-refusal-{alpha_label}{custom}-{stamp}"


def save_bar_chart(
    output_base: Path,
    model_name: str,
    display_name: str,
    summaries: dict[str, dict[str, float]],
    dpi: int,
) -> Path:
    labels = ["patch_bridge", "patch_readout"]
    drops = np.array([summaries[label]["drop_vs_no_patch_mean"] for label in labels], dtype=np.float64)
    drop_stderrs = np.array(
        [summaries[label]["drop_vs_no_patch_stderr"] for label in labels],
        dtype=np.float64,
    )

    colors = ["#3f7f6b", "#b45b55"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _plot_delta_drop_bars(
        ax=ax,
        drops=drops,
        drop_stderrs=drop_stderrs,
        title=display_name,
        colors=colors,
        show_ylabel=True,
    )
    fig.tight_layout()

    path = Path(f"{output_base}-bar.png")
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _plot_delta_drop_bars(
    ax: plt.Axes,
    drops: np.ndarray,
    drop_stderrs: np.ndarray,
    title: str,
    colors: list[str],
    show_ylabel: bool,
    xtick_labels: list[str] | None = None,
) -> None:
    x = np.arange(len(drops))
    delta_values = -drops
    bars = ax.bar(
        x,
        delta_values,
        color=colors,
        edgecolor="#20242b",
        linewidth=1.0,
        width=0.62,
    )
    ax.axhline(0.0, color="#20242b", linestyle="--", linewidth=1.2, alpha=0.8)

    if xtick_labels is None:
        xtick_labels = ["Bridge", "Readout"]

    ax.set_xticks(x, xtick_labels)
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


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_result_json(output_dir: Path, model_name: str) -> Path:
    candidates = sorted(
        (output_dir / model_name).glob(f"{model_name}-tinst-restore-final-tpost-refusal-*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No saved result JSON found for {model_name} under {output_dir / model_name}")
    return candidates[-1]


def save_combined_figure(
    output_dir: Path,
    figure_output: Path,
    dpi: int,
    result_paths: dict[str, Path] | None = None,
) -> Path:
    model_names = ["llama31", "qwen25"]
    paths = result_paths or {
        model_name: latest_result_json(output_dir, model_name)
        for model_name in model_names
    }
    payloads = {model_name: load_payload(paths[model_name]) for model_name in model_names}

    labels = ["patch_bridge", "patch_readout"]
    drops_by_model = {
        model_name: np.array(
            [payloads[model_name]["summaries"][label]["drop_vs_no_patch_mean"] for label in labels],
            dtype=np.float64,
        )
        for model_name in model_names
    }
    drop_stderrs_by_model = {
        model_name: np.array(
            [payloads[model_name]["summaries"][label]["drop_vs_no_patch_stderr"] for label in labels],
            dtype=np.float64,
        )
        for model_name in model_names
    }

    colors = ["#3f7f6b", "#b45b55"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharey=False)
    for idx, model_name in enumerate(model_names):
        delta_values = -drops_by_model[model_name]
        model_lows = delta_values - drop_stderrs_by_model[model_name]
        low = min(float(model_lows.min()), 0.0)
        high = 0.0
        pad = max((high - low) * 0.18, 0.05)
        if model_name == "llama31":
            xtick_labels = [
                "Bridge\n($l=8$-$11$)",
                "Readout\n($l=12$-$20$)"
            ]
        else:
            xtick_labels = [
                "Bridge\n($l=13$-$15$)",
                "Readout\n($l=18$-$22$)"
            ]

        _plot_delta_drop_bars(
            ax=axes[idx],
            drops=drops_by_model[model_name],
            drop_stderrs=drop_stderrs_by_model[model_name],
            title="",
            colors=colors,
            show_ylabel=idx == 0,
            xtick_labels=xtick_labels,
        )
        val_range = max(abs(low), 0.1)
        axes[idx].set_ylim(low - pad - val_range * 0.10, high + pad * 0.3)
        axes[idx].tick_params(axis="y", labelleft=True)
        axes[idx].text(
            0.5,
            -0.31,
            "(a) LLaMA 3.1" if model_name == "llama31" else "(b) Qwen 2.5",
            transform=axes[idx].transAxes,
            ha="center",
            va="top",
            fontsize=30,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.31, top=0.94, wspace=0.18)
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(figure_output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return figure_output


def main() -> None:
    args = parse_args()
    if args.plot_existing:
        figure_path = save_combined_figure(
            output_dir=Path(args.output_dir).expanduser(),
            figure_output=Path(args.figure_output).expanduser(),
            dpi=args.plot_dpi,
        )
        print(f"[tinst-restore] saved combined figure to {figure_path}", flush=True)
        return

    if args.models:
        run_model_jobs(args)
        models = parse_model_list(args.models)
        if args.auto_plot and {"llama31", "qwen25"}.issubset(set(models)):
            figure_path = save_combined_figure(
                output_dir=Path(args.output_dir).expanduser(),
                figure_output=Path(args.figure_output).expanduser(),
                dpi=args.plot_dpi,
            )
            print(f"[tinst-restore] saved combined figure to {figure_path}", flush=True)
        return

    set_seed(args.seed)
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported configured model: {args.model}. Use one of {sorted(MODEL_CONFIGS)}.")

    rows = load_rows(Path(args.input).expanduser(), args.start, args.count)
    prompts = build_prompts(rows, args)

    print(
        f"[tinst-restore] model={args.model} count={len(prompts)} alpha={args.alpha} "
        f"conditions={','.join(CONDITIONS)}",
        flush=True,
    )
    print("[tinst-restore] loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, args.model_id)
    model.eval()

    modules = build_component_modules(model, args.component)
    num_layers = len(modules)

    direction_paths = resolve_direction_paths(args)
    print(f"[tinst-restore] harmfulness vector: {direction_paths['harmfulness']}", flush=True)
    print(f"[tinst-restore] refusal vector: {direction_paths['refusal']}", flush=True)
    direction_matrices = load_direction_matrices(direction_paths, normalize=False, eps=args.direction_norm_eps)
    harmfulness_directions = normalize_direction_matrix(
        direction_matrices["harmfulness"],
        normalize=bool(args.normalize_injection_direction),
        eps=float(args.direction_norm_eps),
    )
    refusal_directions = normalize_direction_matrix(
        direction_matrices["refusal"],
        normalize=bool(args.normalize_measurement_direction),
        eps=float(args.direction_norm_eps),
    )
    validate_direction_matrix("harmfulness", harmfulness_directions, num_layers)
    validate_direction_matrix("refusal", refusal_directions, num_layers)

    experiment = compute_experiment(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        args=args,
        harmfulness_directions=harmfulness_directions,
        refusal_directions=refusal_directions,
    )

    output_base = make_output_base(args)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "model": args.model,
        "model_id": args.model_id,
        "display": MODEL_CONFIGS[args.model]["display"],
        "input": str(Path(args.input).expanduser().resolve()),
        "start": args.start,
        "count": len(prompts),
        "batch_size": args.batch_size,
        "alpha": float(args.alpha),
        "component": args.component,
        "conditions": CONDITIONS,
        "layer_indexing": "zero_based",
        "normalize_injection_direction": bool(args.normalize_injection_direction),
        "normalize_measurement_direction": bool(args.normalize_measurement_direction),
        "harmfulness_vector_path": str(direction_paths["harmfulness"].resolve()),
        "refusal_vector_path": str(direction_paths["refusal"].resolve()),
        **experiment["config"],
    }
    payload: dict[str, Any] = {
        "metadata": metadata,
        "summaries": experiment["summaries"],
        "clean_projection_summary": summarize_values(experiment["clean_projection"]),
    }

    if args.save_per_example:
        payload["per_example"] = {
            "clean": experiment["clean_projection"].tolist(),
            **{
                condition: values.tolist()
                for condition, values in experiment["condition_values"].items()
            },
        }

    output_path = output_base.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[tinst-restore] saved results to {output_path}", flush=True)

    if args.save_npy:
        np.save(f"{output_base}-clean.npy", experiment["clean_projection"].numpy())
        for condition, values in experiment["condition_values"].items():
            np.save(f"{output_base}-{condition}.npy", values.numpy())
        print(f"[tinst-restore] saved arrays to {output_base}-*.npy", flush=True)

    if args.auto_plot:
        plot_path = save_bar_chart(
            output_base=output_base,
            model_name=args.model,
            display_name=MODEL_CONFIGS[args.model]["display"],
            summaries=experiment["summaries"],
            dpi=args.plot_dpi,
        )
        print(f"[tinst-restore] saved plot to {plot_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
