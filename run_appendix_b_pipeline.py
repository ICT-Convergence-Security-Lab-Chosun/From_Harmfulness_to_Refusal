#!/usr/bin/env python3
"""Run Appendix B experiment pipelines and render Appendix B figures."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
EXPERIMENTS = SRC / "experiments"


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run 70B/14B/32B/72B experiments, judging, and Appendix B rendering."
    )
    parser.add_argument("--run-name", default="appendix_b")
    parser.add_argument("--stages", default="direct,directions,coupling,steering,patching,clean_projection")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--direction-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--guard-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--judge-backend", choices=["wildguard", "openrouter"], default="wildguard")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def model_pipeline_args(args: argparse.Namespace, model_run_name: str) -> list[str]:
    return [
        "--run-name", model_run_name,
        "--stages", args.stages,
        "--count", str(args.count),
        "--direction-count", str(args.direction_count),
        "--batch-size", str(args.batch_size),
        "--guard-batch-size", str(args.guard_batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--judge-backend", args.judge_backend,
    ]


def main() -> None:
    args, extra = parse_args()
    pipelines = [
        ("llama31_70b", EXPERIMENTS / "run_llama31_70b_pipeline.py"),
        ("qwen25_14b", EXPERIMENTS / "run_qwen25_14b_pipeline.py"),
        ("qwen25_32b", EXPERIMENTS / "run_qwen25_32b_pipeline.py"),
        ("qwen25_72b", EXPERIMENTS / "run_qwen25_72b_pipeline.py"),
    ]

    for label, script in pipelines:
        cmd = [sys.executable, str(script)] + model_pipeline_args(args, f"{args.run_name}_{label}") + extra
        if args.dry_run:
            cmd.append("--dry-run")
        run(cmd, args.dry_run)

    tinst_cmd = [
        sys.executable,
        str(EXPERIMENTS / "steered_tinst_restore_refusal_projection_appendix_B.py"),
        "--models", "llama31_70b,qwen14b,qwen32b,qwen72b",
        "--count", str(args.count),
        "--batch-size", str(args.batch_size),
        "--output-dir", str(SRC / "results" / "steered_tinst_restore_refusal_projection"),
        "--figure-output", str(SRC / "out_pt" / "Figure_Appendix_B" / "Fig3.png"),
    ]
    run(tinst_cmd, args.dry_run)

    run([sys.executable, str(SRC / "render_figures.py"), "--section", "appendix-b"], args.dry_run)


if __name__ == "__main__":
    main()
