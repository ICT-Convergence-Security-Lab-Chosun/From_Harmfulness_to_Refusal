#!/usr/bin/env python3

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
from utils import read_row


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATHS = [
    REPO_ROOT / "data" / "alpaca_common.json",
    REPO_ROOT / "data" / "sorry-misrepresentation.json",
    REPO_ROOT / "data" / "sorry-authority-endorsement.json",
    REPO_ROOT / "data" / "sorry-expert-endorsement.json",
]
DEFAULT_MODELS = "llama31,qwen25"
DEFAULT_BASE_DIR = "out_pt"
DEFAULT_COUNT = 100
DEFAULT_LAYERS = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure harmfulness-to-refusal coupling by injecting only the harmfulness "
            "direction at an instruction token and projecting downstream activations on "
            "the refusal direction."
        )
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"Comma-separated model aliases used when --model/--model-id is omitted. Default: {DEFAULT_MODELS}.",
    )
    parser.add_argument("--model", default=None, help="Optional single model alias, e.g. llama31 or qwen25.")
    parser.add_argument("--model-id", default=None, help="Optional single HF model id; preferred over --model/--models.")
    parser.add_argument(
        "--input",
        default=",".join(str(path) for path in DEFAULT_INPUT_PATHS),
        help="Dataset path or comma-separated dataset paths. Default: alpaca_common and three sorry datasets.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"Number of examples. Default: {DEFAULT_COUNT}.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help=f"Injection layers. Default: {DEFAULT_LAYERS}. Use '0-31' or comma-separated layers/ranges to override.",
    )
    parser.add_argument("--one-based-layers", action="store_true")
    parser.add_argument(
        "--component",
        default="hidden",
        choices=["hidden"],
        help="Only hidden-state coupling is supported for this circuit test.",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Direction/output base directory. Default: out_pt next to this script.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmfulness-vector", default=None)
    parser.add_argument("--refusal-vector", default=None)
    parser.add_argument(
        "--normalize-injection-direction",
        action="store_true",
        help="L2-normalize harmfulness directions before injecting.",
    )
    parser.add_argument(
        "--normalize-measurement-direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unit refusal directions for signed scalar projection. Enabled by default.",
    )
    parser.add_argument("--direction-norm-eps", type=float, default=1e-8)
    parser.add_argument("--use-persuade", type=int, default=0)
    parser.add_argument("--use-sys", type=int, default=0)
    parser.add_argument("--use-template", type=int, default=1)
    parser.add_argument("--do-not-use-last-inst-tok", type=int, default=0)
    parser.add_argument("--use-inversion", type=int, default=0)
    parser.add_argument("--inversion-prompt-idx", type=int, default=0)
    parser.add_argument(
        "--save-npy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save raw and std-normalized coupling maps as .npy files. Disabled by default.",
    )
    parser.add_argument(
        "--auto-plot",
        dest="auto_plot",
        action="store_true",
        default=True,
        help="Run plot_harmfulness_refusal_coupling.py after saving the JSON output. Enabled by default.",
    )
    parser.add_argument(
        "--no-auto-plot",
        dest="auto_plot",
        action="store_false",
        help="Skip the automatic plot step after saving the JSON output.",
    )
    parser.add_argument("--plot-output-dir", default=None, help="Optional output directory passed to the plot script.")
    parser.add_argument("--plot-dpi", type=int, default=200, help="DPI passed to the plot script.")
    parser.add_argument(
        "--auto-paper-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After all coupling jobs finish, try to compose paper Fig2/Fig5. Enabled by default.",
    )
    return parser.parse_args()


def parse_csv_paths(raw: str) -> list[str]:
    paths = [item.strip() for item in raw.split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one input dataset must be selected.")
    return paths


def parse_model_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.model_id:
        spec = resolve_model_spec(model=args.model, model_id=args.model_id)
        return [(normalize_model_name(spec.model_name), spec.model_id)]
    if args.model:
        spec = resolve_model_spec(model=args.model, model_id=None)
        return [(normalize_model_name(spec.model_name), spec.model_id)]

    model_tokens = [item.strip() for item in args.models.split(",") if item.strip()]
    if not model_tokens:
        raise ValueError("At least one model must be selected.")
    specs = []
    for token in model_tokens:
        spec = resolve_model_spec(model=token, model_id=None)
        specs.append((normalize_model_name(spec.model_name), spec.model_id))
    return specs


def default_label_for_input(input_path: str) -> str:
    return Path(input_path).stem.replace("-", "_")


def make_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layer_token(token: str, one_based: bool) -> list[int]:
    token = token.strip()
    if not token:
        return []
    if "-" in token:
        left, right = token.split("-", maxsplit=1)
        start = int(left.strip())
        end = int(right.strip())
        step = 1 if end >= start else -1
        layers = list(range(start, end + step, step))
    else:
        layers = [int(token)]
    if one_based:
        layers = [layer - 1 for layer in layers]
    return layers


def parse_layers(raw: str, num_layers: int, one_based: bool) -> list[int]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(num_layers))

    layers: list[int] = []
    for token in raw.split(","):
        layers.extend(parse_layer_token(token, one_based))
    layers = sorted(dict.fromkeys(layers))
    if not layers:
        raise ValueError("At least one layer must be selected.")
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise IndexError(f"Layer {layer} is out of range for a model with {num_layers} layers.")
    return layers


def normalize_direction_matrix(matrix: torch.Tensor, normalize: bool, eps: float) -> torch.Tensor:
    matrix = matrix.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not normalize:
        return matrix
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(eps)


def resolve_base_dir(base_dir: str | Path) -> Path:
    raw = Path(base_dir).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([REPO_ROOT / raw, SCRIPT_DIR / raw])
        if raw.parts and raw.parts[0] == "src":
            candidates.append(REPO_ROOT / Path(*raw.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_direction_candidates(
    base_dir: Path,
    output_name: str,
    model_alias: str,
    mode_dir: str,
    component: str,
    suffix: str,
) -> list[Path]:
    model_dirs = [
        base_dir / output_name,
        base_dir / model_alias,
    ]
    if model_alias == "llama31":
        model_dirs.append(base_dir / "llama3")
    if model_alias == "qwen25":
        model_dirs.append(base_dir / "qwen")

    candidates: list[Path] = []
    for model_dir in dict.fromkeys(model_dirs):
        candidates.extend(
            [
                model_dir / f"{mode_dir}-{component}-{suffix}.pt",
                model_dir / f"{model_alias}-{mode_dir}-{component}-{suffix}.pt",
            ]
        )
    candidates.extend(
        [
            base_dir / f"{mode_dir}-{component}-{suffix}.pt",
            base_dir / f"{model_alias}-{mode_dir}-{component}-{suffix}.pt",
        ]
    )
    return candidates


def resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Missing required direction file. Tried:\n" + "\n".join(str(path) for path in candidates))


def resolve_coupling_direction_paths(args: argparse.Namespace) -> dict[str, Path]:
    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    base_dir = resolve_base_dir(args.base_dir)
    harmfulness_path = (
        Path(args.harmfulness_vector).expanduser()
        if args.harmfulness_vector
        else resolve_existing_path(
            build_direction_candidates(
                base_dir=base_dir,
                output_name=spec.output_name,
                model_alias=args.model,
                mode_dir="hf",
                component=args.component,
                suffix="harmful-minus-harmless",
            )
        )
    )
    refusal_path = (
        Path(args.refusal_vector).expanduser()
        if args.refusal_vector
        else resolve_existing_path(
            build_direction_candidates(
                base_dir=base_dir,
                output_name=spec.output_name,
                model_alias=args.model,
                mode_dir="refuse",
                component=args.component,
                suffix="refuse-minus-accept",
            )
        )
    )
    return {
        "harmfulness": harmfulness_path,
        "refusal": refusal_path,
    }


def validate_direction_matrix(name: str, matrix: torch.Tensor, num_layers: int) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{name} direction must have shape [layers, hidden] or [1, hidden], got {tuple(matrix.shape)}")
    if matrix.shape[0] not in {1, num_layers}:
        raise ValueError(
            f"{name} direction layer axis must be 1 or {num_layers}, got shape={tuple(matrix.shape)}"
        )


def direction_for_layer(matrix: torch.Tensor, layer: int) -> torch.Tensor:
    return matrix[0] if matrix.shape[0] == 1 else matrix[layer]


def gather_token_rows(activation: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = []
    for sample_idx, pos in enumerate(positions.detach().to(dtype=torch.long, device="cpu").tolist()):
        rows.append(activation[sample_idx, int(pos), :])
    return torch.stack(rows, dim=0)


def project_layer_at_positions(
    hidden_states: tuple[torch.Tensor, ...],
    layer: int,
    positions: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    activation = hidden_states[layer + 1]
    selected = gather_token_rows(activation, positions.to(activation.device))
    return selected.float().matmul(direction.to(device=selected.device, dtype=torch.float32))


def project_all_layers(
    hidden_states: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
    directions: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    projections = []
    for layer in range(num_layers):
        projections.append(
            project_layer_at_positions(
                hidden_states=hidden_states,
                layer=layer,
                positions=positions,
                direction=direction_for_layer(directions, layer),
            )
        )
    return torch.stack(projections, dim=1)


def new_readout_stats(num_layers: int) -> dict[str, Any]:
    return {
        "delta_sum": torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "delta_sumsq": torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "delta_count": torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "base_projection_batches": [],
    }


def update_readout_stats(stats: dict[str, Any], deltas: torch.Tensor, inject_layer: int, num_layers: int) -> None:
    for measure_layer in range(inject_layer + 1, num_layers):
        values = deltas[:, measure_layer]
        stats["delta_sum"][inject_layer, measure_layer] += values.sum()
        stats["delta_sumsq"][inject_layer, measure_layer] += (values * values).sum()
        stats["delta_count"][inject_layer, measure_layer] += values.numel()


def finalize_readout_stats(
    stats: dict[str, Any],
    injection_layers: list[int],
    num_layers: int,
    eps: float,
) -> dict[str, Any]:
    delta_sum = stats["delta_sum"]
    delta_sumsq = stats["delta_sumsq"]
    delta_count = stats["delta_count"]

    mean = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    std = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    valid = delta_count > 0
    mean[valid] = delta_sum[valid] / delta_count[valid]
    variance = delta_sumsq[valid] / delta_count[valid] - mean[valid] * mean[valid]
    std[valid] = variance.clamp_min(0.0).sqrt()

    base_projection_matrix = torch.cat(stats["base_projection_batches"], dim=0).to(dtype=torch.float64)
    base_projection_std = base_projection_matrix.std(dim=0, unbiased=False).clamp_min(eps)
    normalized = mean / base_projection_std.view(1, -1)

    downstream_summary = torch.full((num_layers,), float("nan"), dtype=torch.float64)
    for layer in injection_layers:
        downstream = normalized[layer, layer + 1 :]
        downstream = downstream[torch.isfinite(downstream)]
        if downstream.numel() > 0:
            downstream_summary[layer] = downstream.mean()

    return {
        "coupling_map": mean.tolist(),
        "coupling_std": std.tolist(),
        "coupling_count": delta_count.to(dtype=torch.int64).tolist(),
        "normalized_coupling_map": normalized.tolist(),
        "base_projection_mean": base_projection_matrix.mean(dim=0).tolist(),
        "base_projection_std": base_projection_std.tolist(),
        "downstream_summary": downstream_summary.tolist(),
    }


def make_default_output_root(args: argparse.Namespace) -> Path:
    return get_output_root(resolve_base_dir(args.base_dir), model=args.model, model_id=args.model_id) / "coupling"


def make_output_base(args: argparse.Namespace) -> Path:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else make_default_output_root(args)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    custom = f"-{args.label}" if args.label else ""
    alpha_label = f"{args.alpha:g}".replace("-", "m").replace(".", "p")
    stamp = getattr(args, "result_stamp", None) or make_run_stamp()
    return output_dir / f"harm-to-refusal-coupling-{args.component}-alpha_{alpha_label}{custom}-{stamp}"


def compute_internal_coupling(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    harmfulness_directions: torch.Tensor,
    refusal_directions: torch.Tensor,
    injection_layers: list[int],
) -> dict[str, Any]:
    modules = build_component_modules(model, args.component)
    num_layers = len(modules)
    input_device = get_input_device(model)

    readout_stats = new_readout_stats(num_layers)

    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="internal coupling"):
            batch_prompts = prompts[start : start + args.batch_size]
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(input_device)
            attention_mask = inputs.attention_mask.to(input_device)

            t_inst = resolve_target_positions_for_batch(
                tokenizer=tokenizer,
                batch_prompts=batch_prompts,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                target="harmfulness",
                model_name=args.model,
                model_id=args.model_id,
                use_template=bool(args.use_template),
                do_not_use_last_inst_tok=bool(args.do_not_use_last_inst_tok),
            )
            t_post = resolve_target_positions_for_batch(
                tokenizer=tokenizer,
                batch_prompts=batch_prompts,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                target="refusal",
                model_name=args.model,
                model_id=args.model_id,
                use_template=bool(args.use_template),
                do_not_use_last_inst_tok=bool(args.do_not_use_last_inst_tok),
            )

            base_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            base_refusal_projections = project_all_layers(
                hidden_states=base_outputs.hidden_states,
                positions=t_post,
                directions=refusal_directions,
                num_layers=num_layers,
            )
            readout_stats["base_projection_batches"].append(base_refusal_projections.detach().cpu())

            for inject_layer in injection_layers:
                direction = direction_for_layer(harmfulness_directions, inject_layer)
                with ExitStack() as stack:
                    handle = modules[inject_layer].register_forward_hook(
                        get_prefill_only_steering_hook(
                            direction=direction,
                            alpha=float(args.alpha),
                            positions=t_inst,
                            attention_mask=inputs.attention_mask,
                        )
                    )
                    stack.callback(handle.remove)
                    injected_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                injected_refusal_projections = project_all_layers(
                    hidden_states=injected_outputs.hidden_states,
                    positions=t_post,
                    directions=refusal_directions,
                    num_layers=num_layers,
                )
                refusal_deltas = (
                    injected_refusal_projections - base_refusal_projections
                ).detach().cpu().to(dtype=torch.float64)
                update_readout_stats(readout_stats, refusal_deltas, inject_layer, num_layers)

            del base_outputs
            torch.cuda.empty_cache()

    refusal = finalize_readout_stats(readout_stats, injection_layers, num_layers, args.direction_norm_eps)

    return {
        "coupling_map": refusal["coupling_map"],
        "coupling_std": refusal["coupling_std"],
        "coupling_count": refusal["coupling_count"],
        "normalized_coupling_map": refusal["normalized_coupling_map"],
        "base_refusal_projection_mean": refusal["base_projection_mean"],
        "base_refusal_projection_std": refusal["base_projection_std"],
        "downstream_summary": refusal["downstream_summary"],
    }


def save_numpy_maps(base_path: Path, internal: dict[str, Any]) -> None:
    np.save(f"{base_path}-raw.npy", np.array(internal["coupling_map"], dtype=np.float64))
    np.save(f"{base_path}-normalized.npy", np.array(internal["normalized_coupling_map"], dtype=np.float64))


def run_auto_plot(output_path: Path, args: argparse.Namespace) -> None:
    plot_script = SCRIPT_DIR / "figures" / "plot_harmfulness_refusal_coupling.py"
    cmd = [
        sys.executable,
        str(plot_script),
        str(output_path),
        "--dpi",
        str(args.plot_dpi),
    ]
    if args.plot_output_dir:
        cmd.extend(["--output-dir", args.plot_output_dir])

    print("\nrunning auto plot:")
    print("  " + shlex.join(cmd))
    subprocess.run(cmd, check=True)


def run_single_dataset(
    model,
    tokenizer,
    args: argparse.Namespace,
    harmfulness_directions: torch.Tensor,
    refusal_directions: torch.Tensor,
    direction_paths: dict[str, Path],
    injection_layers: list[int],
) -> Path:
    set_seed(args.seed)
    rows = read_row(args.input)
    rows = rows[args.start : args.start + args.count]
    if not rows:
        raise ValueError("No rows selected from the input dataset.")
    print(f"[coupling] loaded {len(rows)} rows from {args.input}", flush=True)

    prompts = build_prompts(rows, args)
    print("[coupling] starting internal coupling forward passes...", flush=True)
    internal = compute_internal_coupling(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        args=args,
        harmfulness_directions=harmfulness_directions,
        refusal_directions=refusal_directions,
        injection_layers=injection_layers,
    )

    output_base = make_output_base(args)
    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "model_id": getattr(tokenizer, "name_or_path", None) or getattr(model.config, "_name_or_path", None),
            "component": args.component,
            "input_path": str(Path(args.input).resolve()),
            "start": args.start,
            "count": len(rows),
            "batch_size": args.batch_size,
            "alpha": float(args.alpha),
            "seed": args.seed,
            "injection_layers": injection_layers,
            "normalize_injection_direction": bool(args.normalize_injection_direction),
            "normalize_measurement_direction": bool(args.normalize_measurement_direction),
            "direction_norm_eps": float(args.direction_norm_eps),
            "harmfulness_vector_path": str(direction_paths["harmfulness"].resolve()),
            "refusal_vector_path": str(direction_paths["refusal"].resolve()),
            "harmfulness_direction_shape": list(harmfulness_directions.shape),
            "refusal_direction_shape": list(refusal_directions.shape),
            "prompt_settings": {
                "use_persuade": bool(args.use_persuade),
                "use_sys": bool(args.use_sys),
                "use_template": bool(args.use_template),
                "do_not_use_last_inst_tok": bool(args.do_not_use_last_inst_tok),
                "use_inversion": bool(args.use_inversion),
                "inversion_prompt_idx": args.inversion_prompt_idx,
            },
            "positions": {
                "intervention": "t_inst, last user instruction token before assistant suffix",
                "measurement": "t_post, last assistant-prefix token before generation",
            },
        },
        "internal_coupling": internal,
    }

    output_path = output_base.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.save_npy:
        save_numpy_maps(output_base, internal)

    print(f"saved harmfulness-to-refusal coupling results to {output_path}")
    if args.save_npy:
        print(f"saved numpy maps to {output_base}-raw.npy and {output_base}-normalized.npy")
    if args.auto_plot:
        run_auto_plot(output_path, args)
    return output_path


def run_model(
    args: argparse.Namespace,
    model_alias: str,
    model_id: str,
    input_paths: list[str],
    run_output_root: Path,
    run_stamp: str,
) -> list[Path]:
    model_args = argparse.Namespace(**vars(args))
    model_args.model = model_alias
    model_args.model_id = model_id
    model_args.output_dir = str(run_output_root / model_alias)
    model_args.result_stamp = run_stamp
    set_seed(model_args.seed)

    print(
        "[coupling] config: "
        f"model={model_args.model} model_id={model_args.model_id} "
        f"inputs={len(input_paths)} count={model_args.count} layers={model_args.layers} "
        f"alpha={model_args.alpha} component={model_args.component} base_dir={model_args.base_dir}",
        flush=True,
    )

    print("[coupling] loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_args.model, model_args.model_id)
    model.eval()
    print("[coupling] model loaded", flush=True)

    modules = build_component_modules(model, model_args.component)
    num_layers = len(modules)
    injection_layers = parse_layers(model_args.layers, num_layers, model_args.one_based_layers)
    print(
        f"[coupling] selected {len(injection_layers)} injection layers out of {num_layers}: "
        f"{injection_layers[0]}..{injection_layers[-1]}" if injection_layers else "[coupling] no layers selected",
        flush=True,
    )

    direction_paths = resolve_coupling_direction_paths(model_args)
    print(f"[coupling] harmfulness vector: {direction_paths['harmfulness']}", flush=True)
    print(f"[coupling] refusal vector: {direction_paths['refusal']}", flush=True)
    direction_matrices = load_direction_matrices(
        direction_paths,
        normalize=False,
        eps=float(model_args.direction_norm_eps),
    )
    harmfulness_directions = normalize_direction_matrix(
        direction_matrices["harmfulness"],
        normalize=bool(model_args.normalize_injection_direction),
        eps=float(model_args.direction_norm_eps),
    )
    refusal_directions = normalize_direction_matrix(
        direction_matrices["refusal"],
        normalize=bool(model_args.normalize_measurement_direction),
        eps=float(model_args.direction_norm_eps),
    )
    validate_direction_matrix("harmfulness", harmfulness_directions, num_layers)
    validate_direction_matrix("refusal", refusal_directions, num_layers)

    output_paths = []
    for input_path in input_paths:
        run_args = argparse.Namespace(**vars(model_args))
        run_args.input = input_path
        input_label = default_label_for_input(input_path)
        if model_args.label and len(input_paths) > 1:
            run_args.label = f"{model_args.label}-{input_label}"
        elif model_args.label:
            run_args.label = model_args.label
        else:
            run_args.label = input_label

        print(f"\n[coupling] dataset: {run_args.input} label={run_args.label}", flush=True)
        output_paths.append(
            run_single_dataset(
                model=model,
                tokenizer=tokenizer,
                args=run_args,
                harmfulness_directions=harmfulness_directions,
                refusal_directions=refusal_directions,
                direction_paths=direction_paths,
                injection_layers=injection_layers,
            )
        )

    del model, tokenizer
    torch.cuda.empty_cache()
    return output_paths


def _index_outputs_for_paper_figures(output_paths: list[Path]) -> dict[tuple[str, str], Path]:
    indexed: dict[tuple[str, str], Path] = {}
    for output_path in output_paths:
        try:
            with output_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f).get("metadata", {})
        except Exception as exc:
            print(f"[coupling] could not inspect {output_path}: {exc}", flush=True)
            continue
        model = normalize_model_name(str(metadata.get("model", "")))
        dataset = Path(str(metadata.get("input_path", ""))).stem.replace("-", "_")
        if model and dataset:
            indexed[(model, dataset)] = output_path
    return indexed


def _run_paper_figure_command(cmd: list[str], label: str) -> None:
    print(f"\n[coupling] composing paper {label}:", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"[coupling] {label} composition did not complete. "
            "This usually means the expected llama/qwen coupling JSONs are not all present yet.",
            flush=True,
        )


def run_paper_figure_composer(output_paths: list[Path], args: argparse.Namespace) -> None:
    plot_script = SCRIPT_DIR / "figures" / "plot_coupling_paper_figures.py"
    if not plot_script.exists():
        print(f"[coupling] paper figure script not found at {plot_script}, skipping.", flush=True)
        return
    indexed = _index_outputs_for_paper_figures(output_paths)

    fig2_dataset = "alpaca_common"
    if ("llama31", fig2_dataset) in indexed and ("qwen25", fig2_dataset) in indexed:
        _run_paper_figure_command(
            [
                sys.executable,
                str(plot_script),
                "--fig",
                "fig2",
                "--llama",
                str(indexed[("llama31", fig2_dataset)]),
                "--qwen",
                str(indexed[("qwen25", fig2_dataset)]),
                "--rows",
                fig2_dataset,
                "--dpi",
                str(args.plot_dpi),
            ],
            "Fig2",
        )
    else:
        print("[coupling] skipping Fig2: llama31/qwen25 alpaca_common outputs are not both present.", flush=True)

    fig5_datasets = [
        "sorry_misrepresentation",
        "sorry_authority_endorsement",
        "sorry_expert_endorsement",
    ]
    if all(("llama31", dataset) in indexed and ("qwen25", dataset) in indexed for dataset in fig5_datasets):
        _run_paper_figure_command(
            [
                sys.executable,
                str(plot_script),
                "--fig",
                "fig5",
                "--llama",
                *[str(indexed[("llama31", dataset)]) for dataset in fig5_datasets],
                "--qwen",
                *[str(indexed[("qwen25", dataset)]) for dataset in fig5_datasets],
                "--rows",
                *fig5_datasets,
                "--dpi",
                str(args.plot_dpi),
            ],
            "Fig5",
        )
    else:
        print("[coupling] skipping Fig5: expected llama31/qwen25 sorry-dataset outputs are not all present.", flush=True)


def main() -> None:
    args = parse_args()
    input_paths = parse_csv_paths(args.input)
    model_specs = parse_model_specs(args)
    run_stamp = make_run_stamp()

    if args.output_dir:
        run_output_root = Path(args.output_dir)
    else:
        root_args = argparse.Namespace(**vars(args))
        root_args.model, root_args.model_id = model_specs[0]
        run_output_root = make_default_output_root(root_args) / f"run-{run_stamp}"

    output_paths = []
    for model_alias, model_id in model_specs:
        output_paths.extend(run_model(args, model_alias, model_id, input_paths, run_output_root, run_stamp))

    print("\n[coupling] completed outputs:")
    for output_path in output_paths:
        print(f"  {output_path}")

    if args.auto_paper_figures:
        run_paper_figure_composer(output_paths, args)


if __name__ == "__main__":
    main()
