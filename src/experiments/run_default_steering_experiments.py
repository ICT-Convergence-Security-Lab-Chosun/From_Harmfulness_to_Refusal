#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_DIR = SCRIPT_DIR / "out_pt"

MODEL_IDS = {
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
}

DATASETS = {
    "alpaca": REPO_ROOT / "data" / "alpaca_common.json",
    "misrepresentation": REPO_ROOT / "data" / "sorry-misrepresentation.json",
    "authority_endorsement": REPO_ROOT / "data" / "sorry-authority-endorsement.json",
    "expert_endorsement": REPO_ROOT / "data" / "sorry-expert-endorsement.json",
}


def parse_csv(raw: str, choices: dict[str, object], what: str) -> list[str]:
    values = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in choices:
            raise ValueError(f"Unknown {what}: {key}. Choices: {', '.join(choices)}")
        values.append(key)
    if not values:
        raise ValueError(f"At least one {what} is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the default activation-steering matrix: LLaMA 3.1 and Qwen2.5, "
            "Alpaca/Sorry prompt styles, one steering vector per output file, "
            "all layers separately, alpha=3.0, WildGuard judge+plot."
        )
    )
    parser.add_argument("--models", default="llama31,qwen25", help=f"Comma-separated models: {', '.join(MODEL_IDS)}")
    parser.add_argument("--datasets", default="alpaca,misrepresentation,authority_endorsement,expert_endorsement",
                        help=f"Comma-separated datasets: {', '.join(DATASETS)}")
    parser.add_argument("--targets", default="harmfulness,refusal", help="Comma-separated steering vectors to run.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--layer-groups", default="all")
    parser.add_argument("--component", default="hidden", choices=["hidden", "attn", "mlp"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--guard-batch-size", type=int, default=8)
    parser.add_argument("--judge-backend", choices=["wildguard", "openrouter"], default="wildguard")
    parser.add_argument("--wildguard-model", default="allenai/wildguard")
    parser.add_argument("--openrouter-model", default="openai/gpt-4o-mini")
    parser.add_argument("--openrouter-workers", type=int, default=16)
    parser.add_argument("--openrouter-api-key", default=None)
    parser.add_argument("--no-auto-judge", action="store_true", help="Only generate steering responses.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, model_key: str, dataset_key: str, target: str) -> list[str]:
    label = f"default_{model_key}_{dataset_key}_{target}_alpha{args.alpha:g}"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "activation_steering.py"),
        "--model-id",
        MODEL_IDS[model_key],
        "--base-dir",
        args.base_dir,
        "--input",
        str(DATASETS[dataset_key]),
        "--targets",
        target,
        "--component",
        args.component,
        "--layer-groups",
        args.layer_groups,
        "--alphas",
        str(args.alpha),
        "--start",
        str(args.start),
        "--count",
        str(args.count),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--label",
        label,
        "--judge-backend",
        args.judge_backend,
    ]
    if args.no_auto_judge:
        cmd.append("--no-auto-judge")
    elif args.judge_backend == "wildguard":
        cmd.extend(["--wildguard-model", args.wildguard_model, "--guard-batch-size", str(args.guard_batch_size)])
    else:
        cmd.extend(["--openrouter-model", args.openrouter_model, "--openrouter-workers", str(args.openrouter_workers)])
        if args.openrouter_api_key:
            cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
    return cmd


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models, MODEL_IDS, "model")
    datasets = parse_csv(args.datasets, DATASETS, "dataset")
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    for target in targets:
        if target not in {"harmfulness", "refusal"}:
            raise ValueError(f"Unknown target: {target}")

    commands = [
        build_command(args, model_key, dataset_key, target)
        for model_key in models
        for dataset_key in datasets
        for target in targets
    ]

    print(f"planned runs: {len(commands)}")
    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {shlex.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
