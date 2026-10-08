#!/usr/bin/env python3
"""
Clean t_inst Harmfulness Projection
===================================

Measure how much harmfulness/refusal signal is already present at a target
token position without any activation injection.

Default comparison:
    advbench clean                -> expected high
    alpaca clean                  -> expected low
    sorry-misrepresentation clean -> measured

Default metric is a signed scalar projection:
    hidden_state(layer, t_inst) dot harmfulness_direction(layer)

Alternative:
    hidden_state(layer, t_post) dot refusal_direction(layer)
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from activation_steering import (
    build_component_modules,
    build_prompts,
    get_input_device,
    get_prefill_only_steering_hook,
    load_direction_matrices,
    resolve_target_positions_for_batch,
)
from extract_single_activations import load_model_and_tokenizer
from model_utils import get_output_root, normalize_model_name, resolve_model_spec
from activation_patching_coupling import (
    build_direction_candidates,
    direction_for_layer,
    gather_token_rows,
    normalize_direction_matrix,
    parse_layers,
    resolve_base_dir,
    resolve_existing_path,
    validate_direction_matrix,
)
from utils import read_row


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"

DEFAULT_ADV_INPUT = DATA_DIR / "advbench_common.json"
DEFAULT_ALPACA_INPUT = DATA_DIR / "alpaca_common.json"
DEFAULT_MISREP_INPUT = DATA_DIR / "sorry-misrepresentation.json"
DEFAULT_AUTHORITY_INPUT = DATA_DIR / "sorry-authority-endorsement.json"
DEFAULT_EXPERT_INPUT = DATA_DIR / "sorry-expert-endorsement.json"
DEFAULT_BASE_DIR = "out_pt"
DEFAULT_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean forwards on configured datasets and measure "
            "layer-wise projections at t_inst or t_post, optionally after harmfulness injection."
        )
    )
    parser.add_argument("--model", default=None, help="Model alias, e.g. llama31 or qwen25.")
    parser.add_argument("--model-id", default=None, help="HF model id; preferred over --model.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model aliases to run as child jobs, e.g. llama31,qwen25. Use 'both' for llama31,qwen25.",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="When --models is set, launch model jobs concurrently instead of sequentially.",
    )
    parser.add_argument("--advbench-input", default=str(DEFAULT_ADV_INPUT))
    parser.add_argument("--alpaca-input", default=str(DEFAULT_ALPACA_INPUT))
    parser.add_argument("--misrep-input", default=str(DEFAULT_MISREP_INPUT))
    parser.add_argument("--authority-input", default=str(DEFAULT_AUTHORITY_INPUT))
    parser.add_argument("--expert-input", default=str(DEFAULT_EXPERT_INPUT))
    parser.add_argument(
        "--datasets",
        default="advbench,alpaca,misrepresentation",
        help=(
            "Comma-separated subset/order from: advbench,alpaca,misrepresentation,"
            "authority_endorsement,expert_endorsement."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to summarize. Use 'all', '0-31', or comma-separated layers/ranges.",
    )
    parser.add_argument(
        "--measure",
        default="harmfulness_tinst",
        choices=["harmfulness_tinst", "refusal_tpost"],
        help=(
            "Projection to measure. harmfulness_tinst = t_inst onto harmfulness direction; "
            "refusal_tpost = t_post onto refusal direction."
        ),
    )
    parser.add_argument(
        "--run-injection",
        action="store_true",
        help=(
            "Also inject harmfulness at t_inst and measure injected-clean projection "
            "changes for each injection layer."
        ),
    )
    parser.add_argument("--alpha", type=float, default=3.0, help="Injection coefficient for --run-injection.")
    parser.add_argument(
        "--injection-layers",
        default=None,
        help="Injection layers for --run-injection. Defaults to --layers.",
    )
    parser.add_argument("--one-based-layers", action="store_true")
    parser.add_argument("--component", default="hidden", choices=["hidden"])
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmfulness-vector", default=None)
    parser.add_argument("--refusal-vector", default=None)
    parser.add_argument(
        "--normalize-injection-direction",
        action="store_true",
        help="L2-normalize harmfulness directions before injection.",
    )
    parser.add_argument(
        "--normalize-measurement-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unit directions for signed scalar projection. Enabled by default.",
    )
    parser.add_argument("--direction-norm-eps", type=float, default=1e-8)
    parser.add_argument("--use-persuade", type=int, default=0)
    parser.add_argument("--use-sys", type=int, default=0)
    parser.add_argument("--use-template", type=int, default=1)
    parser.add_argument("--do-not-use-last-inst-tok", type=int, default=0)
    parser.add_argument("--use-inversion", type=int, default=0)
    parser.add_argument("--inversion-prompt-idx", type=int, default=0)
    parser.add_argument(
        "--save-per-example",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store per-example projection arrays in the JSON. Disabled by default.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Optionally save per-dataset projection matrices as .npy files. Disabled by default.",
    )
    parser.add_argument(
        "--auto-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save layer-wise summary plots after the JSON output. Enabled by default.",
    )
    parser.add_argument("--plot-dpi", type=int, default=200)
    args = parser.parse_args()

    if not args.models:
        spec = resolve_model_spec(model=args.model, model_id=args.model_id)
        args.model = normalize_model_name(spec.model_name)
        args.model_id = spec.model_id
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dataset_names(raw: str) -> list[str]:
    aliases = {
        "advbench": "advbench",
        "alpaca": "alpaca",
        "misrepresentation": "misrepresentation",
        "misrep": "misrepresentation",
        "authority": "authority_endorsement",
        "authority_endorsement": "authority_endorsement",
        "expert": "expert_endorsement",
        "expert_endorsement": "expert_endorsement",
    }
    names = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        name = aliases.get(token)
        if name is None:
            raise ValueError(f"Unsupported dataset '{token}'. Choose from {sorted(set(aliases.values()))}.")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("--datasets must contain at least one dataset.")
    return names


def parse_model_list(raw: str) -> list[str]:
    if raw.strip().lower() in {"both", "all"}:
        raw = "llama31,qwen25"
    models = [normalize_model_name(item.strip()) for item in raw.split(",") if item.strip()]
    models = [model for model in models if model]
    if not models:
        raise ValueError("--models must contain at least one model alias.")
    return list(dict.fromkeys(models))


def _skip_cli_option(
    argv: list[str],
    index: int,
    names_with_values: set[str],
    flag_names: set[str],
) -> tuple[bool, int]:
    arg = argv[index]
    if arg in flag_names or any(arg.startswith(f"{name}=") for name in flag_names):
        return True, index + 1
    for name in names_with_values:
        if arg == name:
            return True, min(index + 2, len(argv))
        if arg.startswith(f"{name}="):
            return True, index + 1
    return False, index + 1


def child_argv_for_model(args: argparse.Namespace, model_name: str) -> list[str]:
    names_with_values = {"--models", "--model", "--model-id", "--output-dir"}
    flag_names = {"--parallel-models"}
    child_args: list[str] = []
    index = 0
    original = sys.argv[1:]
    while index < len(original):
        should_skip, next_index = _skip_cli_option(original, index, names_with_values, flag_names)
        if not should_skip:
            child_args.append(original[index])
        index = next_index

    child_args.extend(["--model", model_name])
    if args.output_dir:
        child_args.extend(["--output-dir", str(Path(args.output_dir) / model_name)])
    return [sys.executable, str(Path(__file__).resolve()), *child_args]


def run_model_jobs(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    commands = [child_argv_for_model(args, model_name) for model_name in models]
    print(
        f"[clean-projection] launching {len(commands)} model job(s): {', '.join(models)} "
        f"({'parallel' if args.parallel_models else 'sequential'})",
        flush=True,
    )
    for model_name, cmd in zip(models, commands):
        print(f"[clean-projection] job {model_name}: {shlex.join(cmd)}", flush=True)

    if not args.parallel_models:
        for model_name, cmd in zip(models, commands):
            print(f"\n[clean-projection] starting {model_name}", flush=True)
            subprocess.run(cmd, check=True)
        return

    procs = [(model_name, subprocess.Popen(cmd)) for model_name, cmd in zip(models, commands)]
    failures: list[tuple[str, int]] = []
    for model_name, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append((model_name, code))
    if failures:
        failed = ", ".join(f"{model_name}={code}" for model_name, code in failures)
        raise subprocess.CalledProcessError(failures[0][1], f"parallel model jobs failed: {failed}")


def dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "advbench": Path(args.advbench_input).expanduser(),
        "alpaca": Path(args.alpaca_input).expanduser(),
        "misrepresentation": Path(args.misrep_input).expanduser(),
        "authority_endorsement": Path(args.authority_input).expanduser(),
        "expert_endorsement": Path(args.expert_input).expanduser(),
    }


def measure_config(measure: str) -> dict[str, str]:
    if measure == "harmfulness_tinst":
        return {
            "direction_key": "harmfulness",
            "position_target": "harmfulness",
            "position_name": "t_inst",
            "direction_mode_dir": "hf",
            "direction_suffix": "harmful-minus-harmless",
            "override_attr": "harmfulness_vector",
            "plot_title": "Clean t_inst Harmfulness Projection",
            "plot_ylabel": "Projection on harmfulness direction",
            "filename_label": "clean-tinst-harmfulness-projection",
            "position_description": "last user token before the assistant suffix",
        }
    if measure == "refusal_tpost":
        return {
            "direction_key": "refusal",
            "position_target": "refusal",
            "position_name": "t_post",
            "direction_mode_dir": "refuse",
            "direction_suffix": "refuse-minus-accept",
            "override_attr": "refusal_vector",
            "plot_title": "Clean t_post Refusal Projection",
            "plot_ylabel": "Projection on refusal direction",
            "filename_label": "clean-tpost-refusal-projection",
            "position_description": "assistant-prefix token",
        }
    raise ValueError(f"Unsupported measure: {measure}")


def direction_config(direction_key: str) -> dict[str, str]:
    if direction_key == "harmfulness":
        return {
            "direction_mode_dir": "hf",
            "direction_suffix": "harmful-minus-harmless",
            "override_attr": "harmfulness_vector",
        }
    if direction_key == "refusal":
        return {
            "direction_mode_dir": "refuse",
            "direction_suffix": "refuse-minus-accept",
            "override_attr": "refusal_vector",
        }
    raise ValueError(f"Unsupported direction key: {direction_key}")


def resolve_direction_path(args: argparse.Namespace, direction_key: str) -> Path:
    config = direction_config(direction_key)
    override = getattr(args, config["override_attr"])
    if override:
        return Path(override).expanduser()

    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    base_dir = resolve_base_dir(args.base_dir)
    return resolve_existing_path(
        build_direction_candidates(
            base_dir=base_dir,
            output_name=spec.output_name,
            model_alias=args.model,
            mode_dir=config["direction_mode_dir"],
            component=args.component,
            suffix=config["direction_suffix"],
        )
    )


def resolve_measurement_direction_path(args: argparse.Namespace) -> Path:
    return resolve_direction_path(args, measure_config(args.measure)["direction_key"])


def load_rows(path: Path, start: int, count: int) -> list[Any]:
    rows = read_row(str(path))[start : start + count]
    if not rows:
        raise ValueError(f"No rows selected from {path}")
    return rows


def compute_clean_projections(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    directions: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Return projection matrix with shape [num_examples, num_layers]."""
    config = measure_config(args.measure)
    input_device = get_input_device(model)
    batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc="clean projection"):
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(input_device)
            attention_mask = inputs.attention_mask.to(input_device)

            positions = resolve_target_positions_for_batch(
                tokenizer=tokenizer,
                batch_prompts=batch_prompts,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                target=config["position_target"],
                model_name=args.model,
                model_id=args.model_id,
                use_template=bool(args.use_template),
                do_not_use_last_inst_tok=bool(args.do_not_use_last_inst_tok),
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            layer_values = []
            for layer in range(num_layers):
                activation = outputs.hidden_states[layer + 1]
                selected = gather_token_rows(activation, positions.to(activation.device))
                direction = direction_for_layer(directions, layer)
                direction = direction.to(device=selected.device, dtype=torch.float32)
                layer_values.append(selected.float().matmul(direction))

            batch_matrix = torch.stack(layer_values, dim=1).detach().cpu()
            batches.append(batch_matrix)

            del outputs
            torch.cuda.empty_cache()

    return torch.cat(batches, dim=0).to(dtype=torch.float64)


def compute_projection_experiment(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    measurement_directions: torch.Tensor,
    injection_directions: torch.Tensor | None,
    injection_layers: list[int],
    num_layers: int,
) -> dict[str, Any]:
    """Return clean projections and optional injected-clean deltas."""
    config = measure_config(args.measure)
    modules = build_component_modules(model, args.component)
    input_device = get_input_device(model)
    clean_batches: list[torch.Tensor] = []
    delta_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in injection_layers}

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc="projection experiment"):
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(input_device)
            attention_mask = inputs.attention_mask.to(input_device)

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
            measure_positions = resolve_target_positions_for_batch(
                target=config["position_target"],
                **shared_pos_kwargs,
            )

            clean_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            clean_projection = project_hidden_states(
                hidden_states=clean_outputs.hidden_states,
                positions=measure_positions,
                directions=measurement_directions,
                num_layers=num_layers,
            )
            clean_batches.append(clean_projection.detach().cpu())

            if injection_directions is not None and injection_layers:
                injection_positions = resolve_target_positions_for_batch(
                    target="harmfulness",
                    **shared_pos_kwargs,
                )
                for inject_layer in injection_layers:
                    direction = direction_for_layer(injection_directions, inject_layer)
                    inject_hook = get_prefill_only_steering_hook(
                        direction=direction,
                        alpha=float(args.alpha),
                        positions=injection_positions,
                        attention_mask=inputs.attention_mask,
                    )
                    with ExitStack() as stack:
                        handle = modules[inject_layer].register_forward_hook(inject_hook)
                        stack.callback(handle.remove)
                        injected_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )

                    injected_projection = project_hidden_states(
                        hidden_states=injected_outputs.hidden_states,
                        positions=measure_positions,
                        directions=measurement_directions,
                        num_layers=num_layers,
                    )
                    delta_batches[inject_layer].append(
                        (injected_projection - clean_projection).detach().cpu()
                    )
                    del injected_outputs

            del clean_outputs
            torch.cuda.empty_cache()

    clean_matrix = torch.cat(clean_batches, dim=0).to(dtype=torch.float64)
    delta_matrices = {
        layer: torch.cat(batches, dim=0).to(dtype=torch.float64)
        for layer, batches in delta_batches.items()
        if batches
    }
    return {
        "clean": clean_matrix,
        "injection_deltas": delta_matrices,
    }


def project_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
    directions: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    layer_values = []
    for layer in range(num_layers):
        activation = hidden_states[layer + 1]
        selected = gather_token_rows(activation, positions.to(activation.device))
        direction = direction_for_layer(directions, layer)
        direction = direction.to(device=selected.device, dtype=torch.float32)
        layer_values.append(selected.float().matmul(direction))
    return torch.stack(layer_values, dim=1)


def summarize_matrix(matrix: torch.Tensor, selected_layers: list[int]) -> dict[str, Any]:
    selected = matrix[:, selected_layers]
    per_layer = []
    for layer in selected_layers:
        values = matrix[:, layer]
        per_layer.append(
            {
                "layer": int(layer),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "median": float(values.median().item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
            }
        )

    flat = selected.flatten()
    return {
        "n": int(matrix.shape[0]),
        "selected_layers": [int(layer) for layer in selected_layers],
        "selected_layer_mean": float(selected.mean().item()),
        "selected_layer_std": float(selected.std(unbiased=False).item()),
        "selected_layer_median": float(flat.median().item()),
        "per_layer": per_layer,
        "all_layer_mean": [float(x) for x in matrix.mean(dim=0).tolist()],
        "all_layer_std": [float(x) for x in matrix.std(dim=0, unbiased=False).tolist()],
    }


def summarize_injection_deltas(
    delta_matrices: dict[int, torch.Tensor],
    selected_layers: list[int],
    num_layers: int,
) -> dict[str, Any]:
    delta_mean = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    delta_std = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    delta_selected_mean = torch.full((num_layers,), float("nan"), dtype=torch.float64)

    per_injection_layer = []
    for inject_layer, matrix in sorted(delta_matrices.items()):
        delta_mean[inject_layer, :] = matrix.mean(dim=0)
        delta_std[inject_layer, :] = matrix.std(dim=0, unbiased=False)
        selected = matrix[:, selected_layers]
        delta_selected_mean[inject_layer] = selected.mean()
        per_injection_layer.append(
            {
                "injection_layer": int(inject_layer),
                "selected_layer_mean_delta": float(selected.mean().item()),
                "selected_layer_std_delta": float(selected.std(unbiased=False).item()),
                "per_readout_layer": [
                    {
                        "layer": int(layer),
                        "mean_delta": float(matrix[:, layer].mean().item()),
                        "std_delta": float(matrix[:, layer].std(unbiased=False).item()),
                    }
                    for layer in selected_layers
                ],
            }
        )

    return {
        "delta_mean_map": delta_mean.tolist(),
        "delta_std_map": delta_std.tolist(),
        "selected_layer_mean_delta_by_injection_layer": delta_selected_mean.tolist(),
        "per_injection_layer": per_injection_layer,
    }


def build_comparison(
    summaries: dict[str, dict[str, Any]],
    selected_layers: list[int],
    eps: float,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    if not {"advbench", "alpaca", "misrepresentation"}.issubset(summaries):
        return comparison

    adv_means = np.array(summaries["advbench"]["all_layer_mean"], dtype=np.float64)
    alpaca_means = np.array(summaries["alpaca"]["all_layer_mean"], dtype=np.float64)
    misrep_means = np.array(summaries["misrepresentation"]["all_layer_mean"], dtype=np.float64)
    denom = adv_means - alpaca_means
    relative = np.full_like(misrep_means, np.nan)
    valid = np.abs(denom) > eps
    relative[valid] = (misrep_means[valid] - alpaca_means[valid]) / denom[valid]

    selected_relative = relative[selected_layers]
    selected_relative = selected_relative[np.isfinite(selected_relative)]
    comparison["misrep_relative_to_alpaca_advbench"] = {
        "description": (
            "Layer-wise (misrepresentation_mean - alpaca_mean) / "
            "(advbench_mean - alpaca_mean). alpaca=0, advbench=1."
        ),
        "selected_layer_mean": (
            float(selected_relative.mean()) if selected_relative.size else None
        ),
        "per_layer": [
            {"layer": int(layer), "relative_position": float(relative[layer])}
            if np.isfinite(relative[layer])
            else {"layer": int(layer), "relative_position": None}
            for layer in selected_layers
        ],
        "all_layers": [
            float(x) if np.isfinite(x) else None
            for x in relative.tolist()
        ],
    }

    comparison["selected_layer_mean_differences"] = {
        "advbench_minus_alpaca": float(
            summaries["advbench"]["selected_layer_mean"]
            - summaries["alpaca"]["selected_layer_mean"]
        ),
        "misrepresentation_minus_alpaca": float(
            summaries["misrepresentation"]["selected_layer_mean"]
            - summaries["alpaca"]["selected_layer_mean"]
        ),
        "advbench_minus_misrepresentation": float(
            summaries["advbench"]["selected_layer_mean"]
            - summaries["misrepresentation"]["selected_layer_mean"]
        ),
    }
    return comparison


def make_output_base(args: argparse.Namespace) -> Path:
    config = measure_config(args.measure)
    mode_dir = "injection" if args.run_injection else "clean"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root(args.base_dir, model=args.model, model_id=args.model_id)
        / "clean_projection"
        / args.measure
        / mode_dir
    )
    if args.output_dir:
        output_dir = output_dir / args.measure / mode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    custom = f"-{args.label}" if args.label else ""
    alpha_value = f"{float(args.alpha):g}".replace("-", "m").replace(".", "p")
    alpha_label = f"-alpha_{alpha_value}" if args.run_injection else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir / f"{config['filename_label']}{alpha_label}{custom}-{stamp}"


def save_plots(
    output_base: Path,
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    selected_layers: list[int],
    measure: str,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    config = measure_config(measure)
    saved_paths: list[Path] = []
    colors = {
        "advbench": "#c43b3b",
        "alpaca": "#2f7d4f",
        "misrepresentation": "#3b66c4",
        "authority_endorsement": "#8b5fbf",
        "expert_endorsement": "#d28c2d",
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for name in summaries:
        if name not in summaries:
            continue
        means = np.array(summaries[name]["all_layer_mean"], dtype=np.float64)
        stds = np.array(summaries[name]["all_layer_std"], dtype=np.float64)
        layers = np.arange(len(means))
        color = colors.get(name)
        ax.plot(layers, means, label=name, color=color, linewidth=2.0)
        ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.14)

    ax.set_title(config["plot_title"])
    ax.set_xlabel("Layer")
    ax.set_ylabel(config["plot_ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if selected_layers:
        ax.axvspan(min(selected_layers), max(selected_layers), color="black", alpha=0.04)
    fig.tight_layout()
    projection_path = Path(f"{output_base}-layerwise-projection.png")
    fig.savefig(projection_path, dpi=dpi)
    plt.close(fig)
    saved_paths.append(projection_path)

    relative = comparison.get("misrep_relative_to_alpaca_advbench")
    if relative:
        values = np.array(
            [np.nan if value is None else float(value) for value in relative["all_layers"]],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(10, 4.8))
        layers = np.arange(len(values))
        ax.plot(layers, values, color=colors["misrepresentation"], linewidth=2.0)
        ax.axhline(0.0, color=colors["alpaca"], linestyle="--", linewidth=1.2, label="alpaca mean")
        ax.axhline(1.0, color=colors["advbench"], linestyle="--", linewidth=1.2, label="advbench mean")
        ax.set_title(f"Misrepresentation Relative {config['direction_key'].title()} Position")
        ax.set_xlabel("Layer")
        ax.set_ylabel("(misrep - alpaca) / (advbench - alpaca)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        if selected_layers:
            ax.axvspan(min(selected_layers), max(selected_layers), color="black", alpha=0.04)
        fig.tight_layout()
        relative_path = Path(f"{output_base}-misrep-relative-position.png")
        fig.savefig(relative_path, dpi=dpi)
        plt.close(fig)
        saved_paths.append(relative_path)

    return saved_paths


def save_injection_plots(
    output_base: Path,
    injection_summaries: dict[str, dict[str, Any]],
    measure: str,
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    config = measure_config(measure)
    saved_paths: list[Path] = []
    for dataset_name, summary in injection_summaries.items():
        delta_map = np.array(summary["delta_mean_map"], dtype=np.float64)
        if not np.isfinite(delta_map).any():
            continue

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        vmax = np.nanpercentile(np.abs(delta_map), 98)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        image = ax.imshow(delta_map, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{dataset_name}: Injection Delta, {config['position_name']} {config['direction_key']}")
        ax.set_xlabel("Readout layer")
        ax.set_ylabel("Injection layer")
        fig.colorbar(image, ax=ax, label="Injected - clean projection")
        fig.tight_layout()
        heatmap_path = Path(f"{output_base}-{dataset_name}-injection-delta-heatmap.png")
        fig.savefig(heatmap_path, dpi=dpi)
        plt.close(fig)
        saved_paths.append(heatmap_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    if args.models:
        run_model_jobs(args)
        return

    set_seed(args.seed)

    dataset_names = parse_dataset_names(args.datasets)
    paths = dataset_paths(args)

    print(
        f"[clean-projection] model={args.model} count={args.count} "
        f"layers={args.layers} measure={args.measure} "
        f"injection={args.run_injection} datasets={','.join(dataset_names)}",
        flush=True,
    )
    print("[clean-projection] loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, args.model_id)
    model.eval()
    print("[clean-projection] model loaded", flush=True)

    modules = build_component_modules(model, args.component)
    num_layers = len(modules)
    selected_layers = parse_layers(args.layers, num_layers, args.one_based_layers)
    injection_layers = (
        parse_layers(args.injection_layers or args.layers, num_layers, args.one_based_layers)
        if args.run_injection
        else []
    )

    config = measure_config(args.measure)
    direction_path = resolve_measurement_direction_path(args)
    print(f"[clean-projection] {config['direction_key']} vector: {direction_path}", flush=True)
    direction_paths = {config["direction_key"]: direction_path}
    if args.run_injection:
        harmfulness_path = resolve_direction_path(args, "harmfulness")
        direction_paths["harmfulness"] = harmfulness_path
        print(f"[clean-projection] injection harmfulness vector: {harmfulness_path}", flush=True)
        print(
            f"[clean-projection] {len(injection_layers)} injection layers: "
            f"{injection_layers[0]}..{injection_layers[-1]} alpha={args.alpha}",
            flush=True,
        )

    direction_matrices = load_direction_matrices(direction_paths, normalize=False, eps=float(args.direction_norm_eps))
    directions = normalize_direction_matrix(
        direction_matrices[config["direction_key"]],
        normalize=bool(args.normalize_measurement_direction),
        eps=float(args.direction_norm_eps),
    )
    validate_direction_matrix(config["direction_key"], directions, num_layers)
    injection_directions = None
    if args.run_injection:
        injection_directions = normalize_direction_matrix(
            direction_matrices["harmfulness"],
            normalize=bool(args.normalize_injection_direction),
            eps=float(args.direction_norm_eps),
        )
        validate_direction_matrix("harmfulness", injection_directions, num_layers)

    projections: dict[str, torch.Tensor] = {}
    summaries: dict[str, dict[str, Any]] = {}
    injection_delta_matrices: dict[str, dict[int, torch.Tensor]] = {}
    injection_summaries: dict[str, dict[str, Any]] = {}

    for dataset_name in dataset_names:
        rows = load_rows(paths[dataset_name], args.start, args.count)
        prompts = build_prompts(rows, args)
        print(
            f"[clean-projection] {dataset_name}: loaded {len(rows)} rows from {paths[dataset_name]}",
            flush=True,
        )
        experiment = compute_projection_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            args=args,
            measurement_directions=directions,
            injection_directions=injection_directions,
            injection_layers=injection_layers,
            num_layers=num_layers,
        )
        matrix = experiment["clean"]
        projections[dataset_name] = matrix
        summaries[dataset_name] = summarize_matrix(matrix, selected_layers)
        if args.run_injection:
            deltas = experiment["injection_deltas"]
            injection_delta_matrices[dataset_name] = deltas
            injection_summaries[dataset_name] = summarize_injection_deltas(
                delta_matrices=deltas,
                selected_layers=selected_layers,
                num_layers=num_layers,
            )

    comparison = build_comparison(summaries, selected_layers, float(args.direction_norm_eps))

    output_base = make_output_base(args)
    payload: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "script": "clean_tinst_harmfulness_projection.py",
            "model": args.model,
            "model_id": getattr(tokenizer, "name_or_path", None),
            "start": args.start,
            "count": args.count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "component": args.component,
            "measure": args.measure,
            "run_injection": bool(args.run_injection),
            "alpha": float(args.alpha) if args.run_injection else None,
            "injection_layers": injection_layers,
            "selected_layers": selected_layers,
            "normalize_measurement_direction": bool(args.normalize_measurement_direction),
            "normalize_injection_direction": bool(args.normalize_injection_direction),
            f"{config['direction_key']}_vector_path": str(direction_path.resolve()),
            "dataset_paths": {name: str(paths[name].resolve()) for name in dataset_names},
            "position": {
                "target": config["position_name"],
                "description": config["position_description"],
            },
            "method_note": (
                "No activation injection is applied. The script runs clean forward passes "
                f"and projects hidden states at {config['position_name']} onto the "
                f"{config['direction_key']} direction."
            ),
        },
        "summaries": summaries,
        "comparison": comparison,
        "injection_delta_summaries": injection_summaries if args.run_injection else None,
    }

    if args.save_per_example:
        payload["per_example_projections"] = {
            name: matrix.tolist()
            for name, matrix in projections.items()
        }

    output_path = output_base.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[clean-projection] saved results to {output_path}", flush=True)

    if args.save_npy:
        for name, matrix in projections.items():
            np.save(f"{output_base}-{name}.npy", matrix.numpy())
            if args.run_injection:
                for inject_layer, delta_matrix in injection_delta_matrices[name].items():
                    np.save(
                        f"{output_base}-{name}-injection_L{inject_layer:02d}-delta.npy",
                        delta_matrix.numpy(),
                    )
        print(f"[clean-projection] saved projection matrices to {output_base}-*.npy", flush=True)

    if args.auto_plot:
        plot_paths = save_plots(
            output_base=output_base,
            summaries=summaries,
            comparison=comparison,
            selected_layers=selected_layers,
            measure=args.measure,
            dpi=args.plot_dpi,
        )
        for plot_path in plot_paths:
            print(f"[clean-projection] saved plot to {plot_path}", flush=True)
        if args.run_injection:
            injection_plot_paths = save_injection_plots(
                output_base=output_base,
                injection_summaries=injection_summaries,
                measure=args.measure,
                dpi=args.plot_dpi,
            )
            for plot_path in injection_plot_paths:
                print(f"[clean-projection] saved plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
