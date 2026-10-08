#!/usr/bin/env python3
"""
Activation Patching Coupling Experiment
=======================================
Separates DIRECT vs INDIRECT causal pathways from harmfulness (t_inst)
to refusal representation (t_post-inst).

    Full pathway:   inject at L → freely propagate → measure at K
    Direct pathway: inject at L → restore t_inst at L+1..K-1 → measure at K
    Indirect:       Full - Direct  (computed from outputs)

Primary metric:    refusal direction projection (continuous, no generation needed)
Validation metric: actual refusal rate via WildGuard (optional, subset only)

Usage
-----
    python activation_patching_coupling.py --model llama31 --input data/advbench.json
    python activation_patching_coupling.py --model qwen25  --input data/misrepresentation.json \\
        --run-generation --generation-layers 10,15,20
    python activation_patching_coupling.py --models llama31,qwen25 --parallel-models \\
        --input data/advbench.json --layers all --run-generation
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from collections import Counter
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
    build_response_records,
    classify_response,
    get_input_device,
    get_prefill_only_steering_hook,
    load_direction_matrices,
    resolve_target_positions_for_batch,
)
from extract_single_activations import load_model_and_tokenizer
from model_utils import get_output_root, normalize_model_name, resolve_model_spec
from utils import read_row


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "alpaca_common.json"
DEFAULT_BASE_DIR   = "out_pt"
DEFAULT_COUNT      = 100
DEFAULT_MODELS     = "llama31,qwen25"


# Argument parsing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Path patching: separate direct vs indirect harmfulness→refusal pathways."
    )
    parser.add_argument("--model",    default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--models", default=None,
                        help=(
                            "Comma-separated model aliases to run as child jobs. "
                            f"Default when no single model is given: {DEFAULT_MODELS}."
                        ))
    parser.add_argument("--parallel-models", action="store_true",
                        help="When --models is set, launch model jobs concurrently instead of sequentially.")
    parser.add_argument("--input",    default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--start",    type=int, default=0)
    parser.add_argument("--count",    type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha",    type=float, default=3.0)
    parser.add_argument("--layers",   default="all",
                        help="Injection layers. 'all', '0-31', or comma-separated.")
    parser.add_argument("--one-based-layers", action="store_true")
    parser.add_argument("--component", default="hidden", choices=["hidden"])
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label",    default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--harmfulness-vector", default=None)
    parser.add_argument("--refusal-vector",     default=None)
    parser.add_argument("--normalize-injection-direction", action="store_true")
    parser.add_argument("--normalize-measurement-direction",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direction-norm-eps", type=float, default=1e-8)
    parser.add_argument("--use-persuade", type=int, default=0)
    parser.add_argument("--use-sys", type=int, default=0)
    parser.add_argument("--use-template",  type=int, default=1)
    parser.add_argument("--do-not-use-last-inst-tok", type=int, default=0)
    parser.add_argument("--use-inversion", type=int, default=0)
    parser.add_argument("--inversion-prompt-idx", type=int, default=0)
    # Behavioral validation
    parser.add_argument("--run-generation",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Run actual generation to measure refusal rates.")
    parser.add_argument("--generation-layers", default=None,
                        help="Layer subset for generation (default: same as --layers).")
    parser.add_argument("--generation-count", type=int, default=30,
                        help="Number of samples for behavioral validation. Default: 30.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--save-generation-records",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Store baseline/full/direct generated responses in the result JSON.")
    parser.add_argument("--auto-judge-generation",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="After --run-generation, run judge_steering_wildguard.py on generated responses.")
    parser.add_argument("--judge-backend", choices=["wildguard", "openrouter"], default="wildguard")
    parser.add_argument("--judge-mode", choices=["judge", "plot", "both"], default="both")
    parser.add_argument("--wildguard-model", default="allenai/wildguard")
    parser.add_argument("--guard-batch-size", type=int, default=8)
    parser.add_argument("--judge-openrouter-model", default="openai/gpt-4o-mini")
    parser.add_argument("--openrouter-api-key", default=None)
    parser.add_argument("--openrouter-workers", type=int, default=16)
    parser.add_argument("--judge-output-dir", default=None)
    parser.add_argument("--save-npy",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-plot", dest="auto_plot", action="store_true", default=True)
    parser.add_argument("--no-auto-plot", dest="auto_plot", action="store_false")
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument("--plot-dpi",  type=int, default=200)
    parser.add_argument(
        "--auto-paper-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After activation-patching jobs finish, try to compose paper Fig4/Fig4_2.",
    )

    args = parser.parse_args()
    if args.models is None and args.model is None and args.model_id is None:
        args.models = DEFAULT_MODELS

    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    args.model    = normalize_model_name(spec.model_name)
    args.model_id = spec.model_id
    return args


# Utilities shared with the coupling scripts

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
        start, end = int(left.strip()), int(right.strip())
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
    for tok in raw.split(","):
        layers.extend(parse_layer_token(tok, one_based))
    layers = sorted(dict.fromkeys(layers))
    if not layers:
        raise ValueError("At least one layer must be selected.")
    for l in layers:
        if l < 0 or l >= num_layers:
            raise IndexError(f"Layer {l} out of range for {num_layers}-layer model.")
    return layers


def parse_model_list(raw: str) -> list[str]:
    models = [normalize_model_name(item.strip()) for item in raw.split(",") if item.strip()]
    models = [model for model in models if model]
    if not models:
        raise ValueError("--models must contain at least one model alias.")
    return list(dict.fromkeys(models))


def _skip_cli_option(argv: list[str], index: int, names_with_values: set[str], flag_names: set[str]) -> tuple[bool, int]:
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
    names_with_values = {"--models", "--model", "--model-id", "--output-dir", "--judge-output-dir"}
    flag_names = {"--parallel-models", "--auto-paper-figures", "--no-auto-paper-figures"}
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
    if args.judge_output_dir:
        child_args.extend(["--judge-output-dir", str(Path(args.judge_output_dir) / model_name)])
    child_args.append("--no-auto-paper-figures")
    return [sys.executable, str(Path(__file__).resolve()), *child_args]


def run_model_jobs(args: argparse.Namespace) -> None:
    models = parse_model_list(args.models)
    commands = [child_argv_for_model(args, model_name) for model_name in models]
    print(
        f"[activation-patching] launching {len(commands)} model job(s): {', '.join(models)} "
        f"({'parallel' if args.parallel_models else 'sequential'})",
        flush=True,
    )
    for model_name, cmd in zip(models, commands):
        print(f"[activation-patching] job {model_name}: {shlex.join(cmd)}", flush=True)

    if not args.parallel_models:
        for model_name, cmd in zip(models, commands):
            print(f"\n[activation-patching] starting {model_name}", flush=True)
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


def direction_for_layer(matrix: torch.Tensor, layer: int) -> torch.Tensor:
    return matrix[0] if matrix.shape[0] == 1 else matrix[layer]


def gather_token_rows(activation: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Extract hidden state at t_inst/t_post for each sample. Returns [batch, hidden]."""
    return torch.stack([
        activation[i, int(pos), :]
        for i, pos in enumerate(positions.detach().cpu().tolist())
    ], dim=0)


def project_all_layers(
    hidden_states: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
    directions: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Project t_post hidden state onto refusal direction at every layer. [batch, layers]"""
    projections = []
    for layer in range(num_layers):
        activation = hidden_states[layer + 1]          # +1 for embedding layer offset
        selected   = gather_token_rows(activation, positions.to(activation.device))
        d = direction_for_layer(directions, layer).to(device=selected.device, dtype=torch.float32)
        projections.append(selected.float().matmul(d))
    return torch.stack(projections, dim=1)             # [batch, num_layers]


def normalize_direction_matrix(mat: torch.Tensor, normalize: bool, eps: float) -> torch.Tensor:
    mat = mat.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return mat / mat.norm(dim=-1, keepdim=True).clamp_min(eps) if normalize else mat


def new_readout_stats(num_layers: int) -> dict[str, Any]:
    return {
        "delta_sum":   torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "delta_sumsq": torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "delta_count": torch.zeros((num_layers, num_layers), dtype=torch.float64),
        "base_projection_batches": [],
    }


def update_readout_stats(
    stats: dict[str, Any], deltas: torch.Tensor, inject_layer: int, num_layers: int
) -> None:
    for measure_layer in range(inject_layer + 1, num_layers):
        v = deltas[:, measure_layer]
        stats["delta_sum"]  [inject_layer, measure_layer] += v.sum()
        stats["delta_sumsq"][inject_layer, measure_layer] += (v * v).sum()
        stats["delta_count"][inject_layer, measure_layer] += v.numel()


def finalize_readout_stats(
    stats: dict[str, Any],
    injection_layers: list[int],
    num_layers: int,
    eps: float,
) -> dict[str, Any]:
    delta_sum   = stats["delta_sum"]
    delta_sumsq = stats["delta_sumsq"]
    delta_count = stats["delta_count"]

    mean  = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    std   = torch.full((num_layers, num_layers), float("nan"), dtype=torch.float64)
    valid = delta_count > 0
    mean[valid]    = delta_sum[valid] / delta_count[valid]
    variance       = delta_sumsq[valid] / delta_count[valid] - mean[valid] ** 2
    std[valid]     = variance.clamp_min(0.0).sqrt()

    base_mat  = torch.cat(stats["base_projection_batches"], dim=0).to(torch.float64)
    base_std  = base_mat.std(dim=0, unbiased=False).clamp_min(eps)
    normalized = mean / base_std.view(1, -1)

    downstream_summary = torch.full((num_layers,), float("nan"), dtype=torch.float64)
    for layer in injection_layers:
        ds = normalized[layer, layer + 1:]
        ds = ds[torch.isfinite(ds)]
        if ds.numel() > 0:
            downstream_summary[layer] = ds.mean()

    return {
        "coupling_map":            mean.tolist(),
        "coupling_std":            std.tolist(),
        "normalized_coupling_map": normalized.tolist(),
        "base_projection_mean":    base_mat.mean(dim=0).tolist(),
        "base_projection_std":     base_std.tolist(),
        "downstream_summary":      downstream_summary.tolist(),
    }


# Direction path resolution

def resolve_base_dir(base_dir: str | Path) -> Path:
    raw = Path(base_dir).expanduser()
    for cand in [raw, REPO_ROOT / raw, SCRIPT_DIR / raw]:
        if cand.exists():
            return cand
    return raw


def build_direction_candidates(
    base_dir: Path, output_name: str, model_alias: str, mode_dir: str,
    component: str, suffix: str,
) -> list[Path]:
    model_dirs = [base_dir / output_name, base_dir / model_alias]
    if model_alias == "llama31":
        model_dirs.append(base_dir / "llama3")
    if model_alias == "qwen25":
        model_dirs.append(base_dir / "qwen")

    candidates: list[Path] = []
    for md in dict.fromkeys(model_dirs):
        candidates += [
            md / f"{mode_dir}-{component}-{suffix}.pt",
            md / f"{model_alias}-{mode_dir}-{component}-{suffix}.pt",
        ]
    candidates += [
        base_dir / f"{mode_dir}-{component}-{suffix}.pt",
        base_dir / f"{model_alias}-{mode_dir}-{component}-{suffix}.pt",
    ]
    return candidates


def resolve_existing_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Missing direction file. Tried:\n" + "\n".join(str(p) for p in candidates))


def resolve_coupling_direction_paths(args: argparse.Namespace) -> dict[str, Path]:
    spec     = resolve_model_spec(model=args.model, model_id=args.model_id)
    base_dir = resolve_base_dir(args.base_dir)

    def _resolve(override, mode_dir, suffix):
        if override:
            return Path(override).expanduser()
        return resolve_existing_path(build_direction_candidates(
            base_dir=base_dir, output_name=spec.output_name,
            model_alias=args.model, mode_dir=mode_dir,
            component=args.component, suffix=suffix,
        ))

    return {
        "harmfulness": _resolve(args.harmfulness_vector, "hf",     "harmful-minus-harmless"),
        "refusal":     _resolve(args.refusal_vector,     "refuse",  "refuse-minus-accept"),
    }


def validate_direction_matrix(name: str, mat: torch.Tensor, num_layers: int) -> None:
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(mat.shape)}")
    if mat.shape[0] not in {1, num_layers}:
        raise ValueError(f"{name} layer axis must be 1 or {num_layers}, got {tuple(mat.shape)}")


# Restoration hook

def get_restore_t_inst_hook(
    clean_activations: torch.Tensor,   # [batch, hidden] on CPU
    positions: torch.Tensor,           # [batch] t_inst positions
    prefill_seq_len: int,              # for prefill-only guard
) -> Any:
    """
    Forward hook that overwrites t_inst positions with stored CLEAN activations.

    Applied to intermediate layers (inject_layer+1 .. num_layers-1) during the
    DIRECT-pathway run.  This blocks the injected harmfulness signal from flowing
    forward through the t_inst residual stream, so only the L→K direct path
    (if any) contributes to the readout at K.

    Prefill-only guard: during autoregressive generation (seq_len == 1) the hook
    is a no-op, matching the behaviour of get_prefill_only_steering_hook.
    """
    def hook(module, input, output):
        # Unpack tuple outputs (hidden, past_kv, ...)
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None

        # Skip generation steps
        if hidden.shape[1] != prefill_seq_len:
            return output

        hidden = hidden.clone()
        for sample_idx, pos in enumerate(positions.detach().cpu().tolist()):
            pos = int(pos)
            if 0 <= pos < hidden.shape[1]:
                hidden[sample_idx, pos, :] = clean_activations[sample_idx].to(
                    device=hidden.device, dtype=hidden.dtype
                )

        return (hidden,) + rest if rest is not None else hidden

    return hook


# Projection-based patching metric

def compute_path_patching(
    model,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    harmfulness_directions: torch.Tensor,
    refusal_directions: torch.Tensor,
    injection_layers: list[int],
) -> dict[str, Any]:
    """
    Computes three coupling heatmaps (source layer × readout layer):

        full     — inject at L, propagate freely       → standard coupling map
        direct   — inject at L, restore t_inst at L+1..K-1 → direct-path only
        indirect — full minus direct                   → all other paths combined

    The direct map isolates the contribution of the t_inst residual stream channel.
    If direct ≈ full the pathway is a single-hop; if indirect dominates, multi-hop
    routing (e.g. via other token positions) is the primary mechanism.
    """
    modules    = build_component_modules(model, args.component)
    num_layers = len(modules)
    input_dev  = get_input_device(model)

    full_stats   = new_readout_stats(num_layers)
    direct_stats = new_readout_stats(num_layers)

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(prompts), args.batch_size), desc="activation patching"):
            batch_prompts = prompts[batch_start: batch_start + args.batch_size]
            inputs   = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids      = inputs.input_ids.to(input_dev)
            attention_mask = inputs.attention_mask.to(input_dev)
            prefill_len    = input_ids.shape[1]

            # Resolve t_inst at the last user token and t_post at the assistant prefix.
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
            t_post = resolve_target_positions_for_batch(target="refusal",     **shared_pos_kwargs)

            # Clean pass: cache t_inst activations and baseline t_post projections.
            base_t_inst_acts: dict[int, torch.Tensor] = {}  # layer → [batch, hidden]
            save_handles = []

            def make_save_hook(layer_idx: int):
                def _hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    base_t_inst_acts[layer_idx] = gather_token_rows(
                        h.detach().cpu(), t_inst.cpu()
                    )
                return _hook

            for li in range(num_layers):
                save_handles.append(
                    modules[li].register_forward_hook(make_save_hook(li))
                )

            base_out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False,
            )
            for h in save_handles:
                h.remove()

            base_proj = project_all_layers(
                hidden_states=base_out.hidden_states,
                positions=t_post,
                directions=refusal_directions,
                num_layers=num_layers,
            )  # [batch, num_layers]

            full_stats  ["base_projection_batches"].append(base_proj.detach().cpu())
            direct_stats["base_projection_batches"].append(base_proj.detach().cpu())

            # Compare full propagation against the restored direct path for each layer.
            for inject_layer in injection_layers:
                direction   = direction_for_layer(harmfulness_directions, inject_layer)
                inject_hook = get_prefill_only_steering_hook(
                    direction=direction,
                    alpha=float(args.alpha),
                    positions=t_inst,
                    attention_mask=inputs.attention_mask,
                )

                # Full pathway: inject once and let the residual stream evolve normally.
                with ExitStack() as stack:
                    stack.callback(
                        modules[inject_layer].register_forward_hook(inject_hook).remove
                    )
                    full_out = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False,
                    )

                full_proj   = project_all_layers(full_out.hidden_states, t_post, refusal_directions, num_layers)
                full_deltas = (full_proj - base_proj).detach().cpu().to(torch.float64)
                update_readout_stats(full_stats, full_deltas, inject_layer, num_layers)

                # Direct pathway: restore t_inst after injection so downstream readouts
                # mostly capture paths that do not keep flowing through t_inst.
                with ExitStack() as stack:
                    stack.callback(
                        modules[inject_layer].register_forward_hook(inject_hook).remove
                    )
                    for mid_layer in range(inject_layer + 1, num_layers):
                        restore_hook = get_restore_t_inst_hook(
                            clean_activations=base_t_inst_acts[mid_layer],
                            positions=t_inst,
                            prefill_seq_len=prefill_len,
                        )
                        stack.callback(
                            modules[mid_layer].register_forward_hook(restore_hook).remove
                        )
                    direct_out = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False,
                    )

                direct_proj   = project_all_layers(direct_out.hidden_states, t_post, refusal_directions, num_layers)
                direct_deltas = (direct_proj - base_proj).detach().cpu().to(torch.float64)
                update_readout_stats(direct_stats, direct_deltas, inject_layer, num_layers)

                del full_out, direct_out

            del base_out, base_t_inst_acts
            torch.cuda.empty_cache()

    # Finalize aggregate maps.
    full_r   = finalize_readout_stats(full_stats,   injection_layers, num_layers, args.direction_norm_eps)
    direct_r = finalize_readout_stats(direct_stats, injection_layers, num_layers, args.direction_norm_eps)

    full_mean   = torch.tensor(full_r  ["coupling_map"])
    direct_mean = torch.tensor(direct_r["coupling_map"])
    full_norm   = torch.tensor(full_r  ["normalized_coupling_map"])
    direct_norm = torch.tensor(direct_r["normalized_coupling_map"])

    indirect_mean = full_mean   - direct_mean
    indirect_norm = full_norm   - direct_norm

    # Direct ratio: what fraction of full-pathway signal travels via direct path?
    # NaN where full is zero (undefined ratio).
    ratio = torch.full_like(full_norm, float("nan"))
    nonzero = full_norm.abs() > 1e-12
    ratio[nonzero] = direct_norm[nonzero] / full_norm[nonzero]

    return {
        "full": {
            "coupling_map":            full_r["coupling_map"],
            "normalized_coupling_map": full_r["normalized_coupling_map"],
            "coupling_std":            full_r["coupling_std"],
            "downstream_summary":      full_r["downstream_summary"],
        },
        "direct": {
            "coupling_map":            direct_r["coupling_map"],
            "normalized_coupling_map": direct_r["normalized_coupling_map"],
            "coupling_std":            direct_r["coupling_std"],
            "downstream_summary":      direct_r["downstream_summary"],
        },
        "indirect": {
            "coupling_map":            indirect_mean.tolist(),
            "normalized_coupling_map": indirect_norm.tolist(),
        },
        "direct_ratio": {
            # 1.0 means mostly direct; 0.0 means mostly indirect.
            "normalized": ratio.tolist(),
            "description": (
                "Fraction of full-pathway signal attributable to the direct "
                "t_inst residual stream channel (direct / full). "
                "NaN where full ≈ 0."
            ),
        },
        "base_refusal_projection_mean": full_r["base_projection_mean"],
        "base_refusal_projection_std":  full_r["base_projection_std"],
    }


# Generation-based validation metric

def _generate_with_hooks(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hooks: list[tuple],          # list of (module, hook_fn)
    max_new_tokens: int,
    tokenizer,
) -> list[str]:
    """Run model.generate with arbitrary pre-registered hooks."""
    handles = []
    for module, hook_fn in hooks:
        handles.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    for h in handles:
        h.remove()

    generated = out[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def run_behavioral_validation(
    model,
    tokenizer,
    rows: list[Any],
    prompts: list[str],
    args: argparse.Namespace,
    harmfulness_directions: torch.Tensor,
    generation_layers: list[int],
) -> dict[str, Any]:
    """
    For each selected layer, compare refusal rates under three conditions:
        baseline  — no injection
        full      — inject harmfulness, propagate freely
        direct    — inject harmfulness, restore t_inst at intermediate layers

    Uses WildGuard (via classify_response) as the judge.
    Runs on a subset (--generation-count) to keep cost manageable.
    """
    modules    = build_component_modules(model, args.component)
    num_layers = len(modules)
    input_dev  = get_input_device(model)

    # Subset for behavioral validation
    subset_rows    = rows  [:args.generation_count]
    subset_prompts = prompts[:args.generation_count]

    # Baseline generation without injection.
    baseline_texts: list[str] = []
    for b_start in range(0, len(subset_prompts), args.batch_size):
        batch = subset_prompts[b_start: b_start + args.batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        texts  = _generate_with_hooks(
            model, inputs.input_ids.to(input_dev), inputs.attention_mask.to(input_dev),
            hooks=[], max_new_tokens=args.max_new_tokens, tokenizer=tokenizer,
        )
        baseline_texts.extend(texts)

    baseline_records = build_response_records(subset_rows, baseline_texts, args.start)
    baseline_labels  = [classify_response(r["response"]) for r in baseline_records]
    baseline_rate    = baseline_labels.count("refusal") / max(1, len(baseline_labels))

    # Per-layer generation for the full and direct paths.
    experiments = []

    for layer_idx, inject_layer in enumerate(
        tqdm(generation_layers, desc="behavioral validation")
    ):
        set_seed(args.seed + 2000 + layer_idx)
        direction = direction_for_layer(harmfulness_directions, inject_layer)

        full_texts:   list[str] = []
        direct_texts: list[str] = []

        for b_start in range(0, len(subset_prompts), args.batch_size):
            batch  = subset_prompts[b_start: b_start + args.batch_size]
            inputs = tokenizer(batch, padding=True, return_tensors="pt")
            input_ids      = inputs.input_ids.to(input_dev)
            attention_mask = inputs.attention_mask.to(input_dev)
            prefill_len    = input_ids.shape[1]

            shared_pos_kwargs = dict(
                tokenizer=tokenizer,
                batch_prompts=batch,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                model_name=args.model,
                model_id=args.model_id,
                use_template=bool(args.use_template),
                do_not_use_last_inst_tok=bool(args.do_not_use_last_inst_tok),
            )
            t_inst = resolve_target_positions_for_batch(target="harmfulness", **shared_pos_kwargs)

            inject_hook = get_prefill_only_steering_hook(
                direction=direction,
                alpha=float(args.alpha),
                positions=t_inst,
                attention_mask=inputs.attention_mask,
            )

            # Full pathway
            full_texts.extend(_generate_with_hooks(
                model, input_ids, attention_mask,
                hooks=[(modules[inject_layer], inject_hook)],
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
            ))

            # Direct pathway: need clean t_inst acts first
            base_t_inst_acts: dict[int, torch.Tensor] = {}
            save_handles = []
            def make_save_hook(li):
                def _h(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    base_t_inst_acts[li] = gather_token_rows(h.detach().cpu(), t_inst.cpu())
                return _h
            for li in range(num_layers):
                save_handles.append(modules[li].register_forward_hook(make_save_hook(li)))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      output_hidden_states=False, use_cache=False)
            for h in save_handles:
                h.remove()

            restore_hooks = [
                (modules[mid_layer], get_restore_t_inst_hook(
                    clean_activations=base_t_inst_acts[mid_layer],
                    positions=t_inst,
                    prefill_seq_len=prefill_len,
                ))
                for mid_layer in range(inject_layer + 1, num_layers)
            ]
            direct_texts.extend(_generate_with_hooks(
                model, input_ids, attention_mask,
                hooks=[(modules[inject_layer], inject_hook)] + restore_hooks,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
            ))

        full_records   = build_response_records(subset_rows, full_texts,   args.start)
        direct_records = build_response_records(subset_rows, direct_texts, args.start)

        full_labels   = [classify_response(r["response"]) for r in full_records]
        direct_labels = [classify_response(r["response"]) for r in direct_records]

        full_rate   = full_labels.count("refusal")   / max(1, len(full_labels))
        direct_rate = direct_labels.count("refusal") / max(1, len(direct_labels))

        experiments.append({
            "layer":             inject_layer,
            "full_refusal_rate":    full_rate,
            "direct_refusal_rate":  direct_rate,
            "indirect_delta":       full_rate - direct_rate,   # indirect path contribution
            "delta_from_baseline_full":   full_rate   - baseline_rate,
            "delta_from_baseline_direct": direct_rate - baseline_rate,
            "full_records":          full_records if args.save_generation_records else [],
            "direct_records":        direct_records if args.save_generation_records else [],
        })

    return {
        "baseline_refusal_rate": baseline_rate,
        "n_samples":             len(subset_prompts),
        "baseline_records":      baseline_records if args.save_generation_records else [],
        "experiments":           experiments,
    }


# Output helpers

def make_output_base(args: argparse.Namespace) -> Path:
    parent_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root(args.base_dir, model=args.model, model_id=args.model_id)
             / "activation_patching"
    )
    custom      = f"-{args.label}" if args.label else ""
    alpha_label = f"{args.alpha:g}".replace("-", "m").replace(".", "p")
    stamp       = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name    = f"activation-patching-{args.component}-alpha_{alpha_label}{custom}-{stamp}"
    run_dir     = parent_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / run_name


def save_numpy_maps(base_path: Path, coupling: dict[str, Any]) -> None:
    for key in ("full", "direct", "indirect"):
        np.save(
            f"{base_path}-{key}-raw.npy",
            np.array(coupling[key]["coupling_map"] if key != "indirect"
                     else coupling[key]["coupling_map"], dtype=np.float64),
        )
        np.save(
            f"{base_path}-{key}-normalized.npy",
            np.array(coupling[key]["normalized_coupling_map"], dtype=np.float64),
        )
    np.save(
        f"{base_path}-direct_ratio-normalized.npy",
        np.array(coupling["direct_ratio"]["normalized"], dtype=np.float64),
    )


def run_auto_plot(output_path: Path, args: argparse.Namespace) -> None:
    # Reuse existing plot script for full/direct/indirect heatmaps
    plot_script = SCRIPT_DIR / "figures" / "plot_harmfulness_refusal_coupling.py"
    if not plot_script.exists():
        print(f"[activation-patching] plot script not found at {plot_script}, skipping.")
        return
    for variant in ("full", "direct", "indirect"):
        cmd = [
            sys.executable, str(plot_script),
            str(output_path),
            "--variant", variant,
            "--dpi", str(args.plot_dpi),
        ]
        if args.plot_output_dir:
            cmd += ["--output-dir", args.plot_output_dir]
        print(f"\n[activation-patching] plotting {variant} map:")
        print("  " + shlex.join(cmd))
        subprocess.run(cmd, check=True)


def build_generation_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    behavioral = payload.get("behavioral_validation") or {}
    baseline_records = behavioral.get("baseline_records") or []
    experiments = []
    for item in behavioral.get("experiments") or []:
        layer = int(item["layer"])
        for target, response_key, rate_key in [
            ("path_full", "full_records", "full_refusal_rate"),
            ("path_direct", "direct_records", "direct_refusal_rate"),
        ]:
            response_rows = item.get(response_key) or []
            if not response_rows:
                continue
            experiments.append({
                "target": target,
                "layer_group_name": f"L{layer:02d}",
                "layer_group": [layer],
                "position_description": (
                    "inject harmfulness at t_inst; "
                    + ("freely propagate" if target == "path_full" else "restore t_inst at intermediate layers")
                ),
                "steer_positions": ["t_inst"],
                "results": [{
                    "alpha": float(metadata.get("alpha", 0.0)),
                    "heuristic_refusal_rate": item.get(rate_key),
                    "responses": response_rows,
                }],
            })

    return {
        "metadata": {
            **metadata,
            "script": "activation_patching_coupling.py",
            "judge_source": "behavioral_validation",
            "format_note": "activation_steering-compatible response JSON for judge_steering_wildguard.py",
        },
        "shared_baseline": {
            "alpha": 0.0,
            "heuristic_refusal_rate": behavioral.get("baseline_refusal_rate"),
            "responses": baseline_records,
        },
        "experiments": experiments,
    }


def save_generation_judge_payload(output_base: Path, payload: dict[str, Any]) -> Path | None:
    behavioral = payload.get("behavioral_validation")
    if not behavioral:
        return None
    judge_payload = build_generation_judge_payload(payload)
    if not judge_payload["shared_baseline"]["responses"] or not judge_payload["experiments"]:
        print("[activation-patching] no saved generation records available for judge, skipping.", flush=True)
        return None
    judge_input_path = Path(f"{output_base}-generation-for-judge.json")
    with judge_input_path.open("w", encoding="utf-8") as f:
        json.dump(judge_payload, f, ensure_ascii=False, indent=2)
    print(f"[activation-patching] saved generation judge input to {judge_input_path}", flush=True)
    return judge_input_path


def run_auto_generation_judge(judge_input_path: Path, args: argparse.Namespace) -> None:
    judge_script = SCRIPT_DIR / "experiments" / "judge_steering_wildguard.py"
    if not judge_script.exists():
        print(f"[activation-patching] judge script not found at {judge_script}, skipping.", flush=True)
        return
    cmd = [
        sys.executable,
        str(judge_script),
        "--input",
        str(judge_input_path),
        "--mode",
        args.judge_mode,
        "--judge-backend",
        args.judge_backend,
    ]
    if args.judge_backend == "wildguard":
        cmd.extend([
            "--wildguard-model",
            args.wildguard_model,
            "--guard-batch-size",
            str(args.guard_batch_size),
        ])
    else:
        cmd.extend([
            "--openrouter-model",
            args.judge_openrouter_model,
            "--openrouter-workers",
            str(args.openrouter_workers),
        ])
        if args.openrouter_api_key:
            cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
    if args.judge_output_dir:
        cmd.extend(["--output-dir", args.judge_output_dir])
    else:
        cmd.extend(["--output-dir", str(judge_input_path.parent / "wildguard_judge")])

    print("\n[activation-patching] judging generated responses:", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def run_paper_figure_composer(args: argparse.Namespace) -> None:
    plot_script = SCRIPT_DIR / "figures" / "plot_path_patching_fig4.py"
    if not plot_script.exists():
        print(f"[activation-patching] paper figure script not found at {plot_script}, skipping.", flush=True)
        return
    cmd = [
        sys.executable,
        str(plot_script),
        "--base-dir",
        str(resolve_base_dir(args.base_dir)),
        "--dpi",
        str(args.plot_dpi),
    ]
    print("\n[activation-patching] composing paper Fig4/Fig4_2:", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            "[activation-patching] Fig4/Fig4_2 composition did not complete. "
            "This usually means the expected llama/qwen activation-patching JSONs are not all present yet.",
            flush=True,
        )


# Main

def main() -> None:
    args = parse_args()
    if args.models:
        run_model_jobs(args)
        if args.auto_paper_figures:
            run_paper_figure_composer(args)
        return

    set_seed(args.seed)

    print(
        f"[activation-patching] model={args.model}  input={args.input}  "
        f"count={args.count}  alpha={args.alpha}  layers={args.layers}",
        flush=True,
    )

    rows = read_row(args.input)[args.start: args.start + args.count]
    if not rows:
        raise ValueError("No rows selected from the input dataset.")
    print(f"[activation-patching] loaded {len(rows)} rows", flush=True)

    print("[activation-patching] loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model, args.model_id)
    model.eval()
    print("[activation-patching] model loaded", flush=True)

    prompts    = build_prompts(rows, args)
    modules    = build_component_modules(model, args.component)
    num_layers = len(modules)
    injection_layers = parse_layers(args.layers, num_layers, args.one_based_layers)
    print(
        f"[activation-patching] {len(injection_layers)} injection layers: "
        f"{injection_layers[0]}..{injection_layers[-1]}",
        flush=True,
    )

    # Load and validate direction matrices
    direction_paths = resolve_coupling_direction_paths(args)
    print(f"[activation-patching] harmfulness vector : {direction_paths['harmfulness']}", flush=True)
    print(f"[activation-patching] refusal vector     : {direction_paths['refusal']}",     flush=True)

    direction_matrices = load_direction_matrices(
        direction_paths, normalize=False, eps=float(args.direction_norm_eps)
    )
    harmfulness_dirs = normalize_direction_matrix(
        direction_matrices["harmfulness"],
        normalize=bool(args.normalize_injection_direction),
        eps=float(args.direction_norm_eps),
    )
    refusal_dirs = normalize_direction_matrix(
        direction_matrices["refusal"],
        normalize=bool(args.normalize_measurement_direction),
        eps=float(args.direction_norm_eps),
    )
    validate_direction_matrix("harmfulness", harmfulness_dirs, num_layers)
    validate_direction_matrix("refusal",     refusal_dirs,     num_layers)

    # Projection heatmaps.
    print("[activation-patching] running projection-based activation patching...", flush=True)
    coupling = compute_path_patching(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        args=args,
        harmfulness_directions=harmfulness_dirs,
        refusal_directions=refusal_dirs,
        injection_layers=injection_layers,
    )

    # Generation/refusal-rate validation.
    behavioral = None
    if args.run_generation:
        gen_layers = parse_layers(
            args.generation_layers or args.layers, num_layers, args.one_based_layers
        )
        print(
            f"[activation-patching] running behavioral validation on {args.generation_count} samples "
            f"across {len(gen_layers)} layers...",
            flush=True,
        )
        behavioral = run_behavioral_validation(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            prompts=prompts,
            args=args,
            harmfulness_directions=harmfulness_dirs,
            generation_layers=gen_layers,
        )

    # Save outputs.
    output_base = make_output_base(args)
    payload = {
        "metadata": {
            "created_at":  datetime.now().isoformat(timespec="seconds"),
            "script":      "activation_patching_coupling.py",
            "model":       args.model,
            "model_id":    getattr(tokenizer, "name_or_path", None),
            "input_path":  str(Path(args.input).resolve()),
            "start":       args.start,
            "count":       len(rows),
            "batch_size":  args.batch_size,
            "alpha":       float(args.alpha),
            "seed":        args.seed,
            "injection_layers": injection_layers,
            "normalize_injection_direction":   bool(args.normalize_injection_direction),
            "normalize_measurement_direction": bool(args.normalize_measurement_direction),
            "harmfulness_vector_path": str(direction_paths["harmfulness"].resolve()),
            "refusal_vector_path":     str(direction_paths["refusal"].resolve()),
            "positions": {
                "intervention": "t_inst — last user token before assistant suffix",
                "measurement":  "t_post — assistant-prefix token",
                "restoration":  "t_inst — same position, restored to clean activation "
                                "at intermediate layers in direct-pathway runs",
            },
            "method_note": (
                "Direction-based activation patching (Heimersheim & Nanda 2024; Zou et al. 2023). "
                "Harmfulness direction = DiM(AdvBench) - DiM(Alpaca), injected as a continuous "
                "vector rather than swapping paired-instance activations. "
                "This avoids semantic incoherence from non-parallel corpora while "
                "preserving causal intervention structure (Wang et al. 2022)."
            ),
        },
        "path_patching_coupling": coupling,
        "behavioral_validation":  behavioral,
    }

    output_path = output_base.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[activation-patching] saved results to {output_path}", flush=True)

    if args.save_npy:
        save_numpy_maps(output_base, coupling)
        print(f"[activation-patching] saved .npy maps to {output_base}-*.npy", flush=True)

    if args.auto_plot:
        run_auto_plot(output_path, args)

    judge_input_path = save_generation_judge_payload(output_base, payload)
    if args.run_generation and args.auto_judge_generation and judge_input_path is not None:
        del model
        torch.cuda.empty_cache()
        run_auto_generation_judge(judge_input_path, args)

    if args.auto_paper_figures:
        run_paper_figure_composer(args)


if __name__ == "__main__":
    main()
