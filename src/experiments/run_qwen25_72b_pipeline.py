#!/usr/bin/env python3
"""Thin orchestration wrapper for the Qwen2.5 72B Instruct pipeline.

This script intentionally does not reimplement experiment logic. It only calls
the existing scripts with explicit paths, so token-position logic stays in the
original experiment files.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
MODEL_ALIAS = "qwen25_72b"
MODEL_OUTPUT_NAME = "qwen2.5-72b-instruct"
PROMPT_TEMPLATE_NOTE = (
    "direct generation: tokenizer.apply_chat_template() when available; "
    "steering/coupling/projection: prefix='<|im_start|>user\\n', "
    "suffix='<|im_end|>\\n<|im_start|>assistant'"
)

DATASETS = {
    "alpaca": REPO_ROOT / "data" / "alpaca_common.json",
    "misrepresentation": REPO_ROOT / "data" / "sorry-misrepresentation.json",
    "authority_endorsement": REPO_ROOT / "data" / "sorry-authority-endorsement.json",
    "expert_endorsement": REPO_ROOT / "data" / "sorry-expert-endorsement.json",
}

DIRECT_DATASETS = (
    "advbench-common,alpaca-common,"
    "sorry-misrepresentation,sorry-authority-endorsement,sorry-expert-endorsement"
)

DEFAULT_STAGES = "direct,directions,coupling,steering,patching,clean_projection"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected Qwen2.5 72B Instruct experiment stages without editing experiment logic."
    )
    parser.add_argument("--stages", default=DEFAULT_STAGES, help=f"Comma-separated stages. Default: {DEFAULT_STAGES}")
    parser.add_argument("--run-name", default=None, help="Stable suffix for outputs. Default: timestamp.")
    parser.add_argument("--base-dir", default=str(SCRIPT_DIR / "out_pt"))
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--direction-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--guard-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--steering-layer-groups", default="all")
    parser.add_argument("--steering-alphas", default=None, help="Default uses --alpha.")
    parser.add_argument(
        "--steering-datasets",
        default="alpaca,misrepresentation,authority_endorsement,expert_endorsement",
    )
    parser.add_argument("--steering-targets", default="harmfulness,refusal")
    parser.add_argument("--patching-datasets", default="alpaca")
    parser.add_argument("--direct-output-dir", default=None)
    parser.add_argument("--judge-backend", choices=["wildguard", "openrouter"], default="wildguard")
    parser.add_argument("--wildguard-model", default="allenai/wildguard")
    parser.add_argument("--openrouter-model", default="openai/gpt-4o-mini")
    parser.add_argument("--openrouter-api-key", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def run_command(label: str, cmd: list[str], dry_run: bool) -> None:
    print(f"\n[pipeline] {label}", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def model_output_root(args: argparse.Namespace) -> Path:
    return Path(args.base_dir).expanduser() / MODEL_OUTPUT_NAME


def isolated_figure_root(args: argparse.Namespace) -> Path:
    return model_output_root(args) / "figures"


def direct_output_dir(args: argparse.Namespace, run_name: str) -> Path:
    if args.direct_output_dir:
        return Path(args.direct_output_dir).expanduser()
    return SCRIPT_DIR / "out_responses" / f"direct_refusal_{MODEL_ALIAS}_{run_name}"


def build_direct_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    out_dir = direct_output_dir(args, run_name)
    eval_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "eval_direct_refusal.py"),
        "--models",
        f"{MODEL_ALIAS}={MODEL_ID}",
        "--datasets",
        DIRECT_DATASETS,
        "--count",
        str(args.count),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(out_dir),
    ]
    judge_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "judge_direct_refusal.py"),
        "--input",
        str(out_dir / "responses.json"),
        "--judge-backend",
        args.judge_backend,
    ]
    if args.judge_backend == "wildguard":
        judge_cmd.extend(["--guard-batch-size", str(args.guard_batch_size)])
    else:
        judge_cmd.extend(["--judge-model", args.openrouter_model])
    return [("direct response generation", eval_cmd), ("direct refusal judge/plot", judge_cmd)]


def build_direction_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "extract_used_directions.py"),
        "--model-id",
        MODEL_ID,
        "--modes",
        "hf,refuse",
        "--output-base-dir",
        args.base_dir,
        "--batch-size",
        str(args.batch_size),
        "--left",
        "0",
        "--right",
        str(args.direction_count),
    ]


def build_coupling_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    commands = []
    for dataset_key, input_path in DATASETS.items():
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "experiments" / "harmfulness_refusal_coupling.py"),
            "--model-id",
            MODEL_ID,
            "--input",
            str(input_path),
            "--count",
            str(args.count),
            "--batch-size",
            str(args.batch_size),
            "--alpha",
            str(args.alpha),
            "--layers",
            args.layers,
            "--base-dir",
            args.base_dir,
            "--label",
            f"{run_name}_{dataset_key}",
            "--plot-output-dir",
            str(isolated_figure_root(args) / "coupling" / dataset_key),
            "--no-auto-paper-figures",
        ]
        commands.append((f"coupling {dataset_key}", cmd))
    return commands


def build_coupling_figure_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return [
        (
            "coupling paper-style figures",
            [
                sys.executable,
                str(SCRIPT_DIR / "figures" / "plot_coupling_paper_figures.py"),
                "--model",
                MODEL_OUTPUT_NAME,
                "--run-dir",
                str(model_output_root(args) / "coupling"),
                "--output-dir",
                str(model_output_root(args) / "Figure"),
                "--dpi",
                "300",
            ],
        )
    ]


def build_steering_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    alphas = args.steering_alphas or str(args.alpha)
    commands = []
    for dataset_key in parse_csv(args.steering_datasets):
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown steering dataset: {dataset_key}")
        for target in parse_csv(args.steering_targets):
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "activation_steering.py"),
                "--model-id",
                MODEL_ID,
                "--base-dir",
                args.base_dir,
                "--input",
                str(DATASETS[dataset_key]),
                "--targets",
                target,
                "--component",
                "hidden",
                "--layer-groups",
                args.steering_layer_groups,
                "--alphas",
                alphas,
                "--count",
                str(args.count),
                "--batch-size",
                str(args.batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--label",
                f"{run_name}_{dataset_key}_{target}",
                "--judge-backend",
                args.judge_backend,
                "--no-auto-paper-figures",
            ]
            if args.judge_backend == "wildguard":
                cmd.extend(["--wildguard-model", args.wildguard_model, "--guard-batch-size", str(args.guard_batch_size)])
            else:
                cmd.extend(["--openrouter-model", args.openrouter_model])
                if args.openrouter_api_key:
                    cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
            commands.append((f"steering {dataset_key}/{target}", cmd))
    return commands


def build_patching_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    commands = []
    for dataset_key in parse_csv(args.patching_datasets):
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown patching dataset: {dataset_key}")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "activation_patching_coupling.py"),
            "--model-id",
            MODEL_ID,
            "--input",
            str(DATASETS[dataset_key]),
            "--count",
            str(args.count),
            "--batch-size",
            str(args.batch_size),
            "--alpha",
            str(args.alpha),
            "--layers",
            args.layers,
            "--base-dir",
            args.base_dir,
            "--label",
            f"{run_name}_{dataset_key}",
            "--judge-backend",
            args.judge_backend,
            "--no-auto-paper-figures",
        ]
        if args.judge_backend == "wildguard":
            cmd.extend(["--wildguard-model", args.wildguard_model, "--guard-batch-size", str(args.guard_batch_size)])
        else:
            cmd.extend(["--judge-openrouter-model", args.openrouter_model])
            if args.openrouter_api_key:
                cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
        commands.append((f"patching {dataset_key}", cmd))
    return commands


def build_clean_projection_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    output_root = model_output_root(args) / "clean_projection_suite" / run_name / "models"
    clean_datasets = "advbench,alpaca,misrepresentation,authority_endorsement,expert_endorsement"
    injection_datasets = "alpaca,misrepresentation,authority_endorsement,expert_endorsement"
    specs = [
        ("clean harmfulness_tinst", "harmfulness_tinst", clean_datasets, False),
        ("clean refusal_tpost", "refusal_tpost", clean_datasets, False),
        ("injection refusal_tpost", "refusal_tpost", injection_datasets, True),
        ("injection harmfulness_tinst", "harmfulness_tinst", injection_datasets, True),
    ]
    commands = []
    for label, measure, datasets, run_injection in specs:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "experiments" / "clean_tinst_harmfulness_projection.py"),
            "--model-id",
            MODEL_ID,
            "--count",
            str(args.count),
            "--batch-size",
            str(args.batch_size),
            "--layers",
            args.layers,
            "--base-dir",
            args.base_dir,
            "--seed",
            "42",
            "--plot-dpi",
            "200",
            "--datasets",
            datasets,
            "--measure",
            measure,
            "--output-dir",
            str(output_root),
            "--label",
            run_name,
        ]
        if run_injection:
            cmd.extend(["--run-injection", "--alpha", str(args.alpha)])
        commands.append((f"clean projection {label}", cmd))
    return commands


def build_steering_figure_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return [
        (
            "steering paper-style figures",
            [
                sys.executable,
                str(SCRIPT_DIR / "experiments" / "judge_steering_wildguard.py"),
                "--compose-model",
                MODEL_OUTPUT_NAME,
                "--compose-output-dir",
                str(model_output_root(args) / "Figure"),
            ],
        )
    ]


def build_patching_figure_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return [
        (
            "patching paper-style figures",
            [
                sys.executable,
                str(SCRIPT_DIR / "figures" / "plot_path_patching_fig4.py"),
                "--base-dir",
                args.base_dir,
                "--output-dir",
                str(model_output_root(args) / "Figure"),
                "--model",
                MODEL_OUTPUT_NAME,
                "--dpi",
                "300",
            ],
        )
    ]


def build_clean_projection_figure_commands(args: argparse.Namespace, run_name: str) -> list[tuple[str, list[str]]]:
    suite_models_dir = model_output_root(args) / "clean_projection_suite" / run_name / "models"
    figure_dir = model_output_root(args) / "Figure"
    return [
        (
            "clean projection Fig7/Fig8",
            [
                sys.executable,
                str(SCRIPT_DIR / "figures" / "plot_clean_projection_paper_figures.py"),
                "--fig",
                "all",
                "--base-dir",
                str(suite_models_dir),
                "--output-dir",
                str(figure_dir),
                "--model",
                MODEL_OUTPUT_NAME,
                "--dpi",
                "300",
            ],
        ),
        (
            "clean projection Fig9",
            [
                sys.executable,
                str(SCRIPT_DIR / "figures" / "plot_injection_delta_heatmap_grid.py"),
                "--models",
                MODEL_OUTPUT_NAME,
                "--measure",
                "refusal_tpost",
                "--base-dir",
                str(suite_models_dir),
                "--datasets",
                "alpaca,misrepresentation,authority_endorsement,expert_endorsement",
                "--output-dir",
                str(figure_dir),
                "--output-name",
                "Fig9",
                "--dpi",
                "300",
            ],
        ),
    ]


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    stages = parse_csv(args.stages)
    valid_stages = set(parse_csv(DEFAULT_STAGES))
    unknown = [stage for stage in stages if stage not in valid_stages]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")

    commands: list[tuple[str, list[str]]] = []
    if "direct" in stages:
        commands.extend(build_direct_commands(args, run_name))
    if "directions" in stages:
        commands.append(("direction extraction", build_direction_command(args)))
    if "coupling" in stages:
        commands.extend(build_coupling_commands(args, run_name))
        commands.extend(build_coupling_figure_commands(args))
    if "steering" in stages:
        commands.extend(build_steering_commands(args, run_name))
        commands.extend(build_steering_figure_commands(args))
    if "patching" in stages:
        commands.extend(build_patching_commands(args, run_name))
        commands.extend(build_patching_figure_commands(args))
    if "clean_projection" in stages:
        commands.extend(build_clean_projection_commands(args, run_name))
        commands.extend(build_clean_projection_figure_commands(args, run_name))

    print(f"[pipeline] model: {MODEL_ID}", flush=True)
    print(f"[pipeline] alias: {MODEL_ALIAS}", flush=True)
    print(f"[pipeline] chat template: {PROMPT_TEMPLATE_NOTE}", flush=True)
    print(f"[pipeline] stages: {', '.join(stages)}", flush=True)
    print(f"[pipeline] planned commands: {len(commands)}", flush=True)
    for label, cmd in commands:
        run_command(label, cmd, args.dry_run)


if __name__ == "__main__":
    main()
