#!/usr/bin/env python3
"""
Run the full clean/injection projection suite.

One command runs:
  1. clean t_inst harmfulness projection for all configured datasets
  2. clean t_post refusal projection for all configured datasets
  3. harmfulness injection at t_inst, readout at t_post/refusal for benign + jailbreak datasets
  4. harmfulness injection at t_inst, readout at t_inst/harmfulness for benign + jailbreak datasets
  5. full injection-layer x readout-layer heatmap grids

Outputs are written by the underlying scripts under:
  out_pt/clean_projection_suite/<run_name>/
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECTION_SCRIPT = SCRIPT_DIR / "experiments" / "clean_tinst_harmfulness_projection.py"
HEATMAP_GRID_SCRIPT = SCRIPT_DIR / "figures" / "plot_injection_delta_heatmap_grid.py"
PAPER_CLEAN_PROJECTION_SCRIPT = SCRIPT_DIR / "figures" / "plot_clean_projection_paper_figures.py"

DEFAULT_CLEAN_DATASETS = "advbench,alpaca,misrepresentation,authority_endorsement,expert_endorsement"
DEFAULT_INJECTION_DATASETS = "alpaca,misrepresentation,authority_endorsement,expert_endorsement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all clean projection, injection projection, and heatmap grid jobs."
    )
    parser.add_argument("--models", default="both", help="Model aliases. Default: both (= llama31,qwen25).")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--injection-layers", default=None)
    parser.add_argument("--base-dir", default="out_pt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Suite output root. Default: <base-dir>/clean_projection_suite/<timestamp>.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suite run directory name under <base-dir>/clean_projection_suite.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-dpi", type=int, default=200)
    parser.add_argument(
        "--clean-datasets",
        default=DEFAULT_CLEAN_DATASETS,
        help="Datasets for clean-only jobs.",
    )
    parser.add_argument(
        "--injection-datasets",
        default=DEFAULT_INJECTION_DATASETS,
        help="Datasets for injection jobs and heatmap grids.",
    )
    parser.add_argument(
        "--plot-measures",
        default="refusal_tpost",
        help="Comma-separated injection measures for heatmap grid plots. Default: refusal_tpost.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip clean-only jobs.",
    )
    parser.add_argument(
        "--skip-injection",
        action="store_true",
        help="Skip injection jobs.",
    )
    parser.add_argument(
        "--skip-post-plots",
        action="store_true",
        help="Skip post-hoc heatmap grid plots.",
    )
    parser.add_argument(
        "--auto-paper-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the suite finishes, render the refusal_tpost heatmap grid as out_pt/Figure/Fig9.png.",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Pass --parallel-models to model jobs. Use only if GPU memory is sufficient.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Forward --save-npy to projection jobs. Disabled by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def resolve_suite_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    base = Path(args.base_dir).expanduser()
    if not base.is_absolute():
        base = SCRIPT_DIR / base
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / "clean_projection_suite" / run_name


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def add_common_projection_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend([
        "--models",
        args.models,
        "--count",
        str(args.count),
        "--batch-size",
        str(args.batch_size),
        "--layers",
        args.layers,
        "--base-dir",
        args.base_dir,
        "--seed",
        str(args.seed),
        "--plot-dpi",
        str(args.plot_dpi),
    ])
    if args.label:
        cmd.extend(["--label", args.label])
    if args.parallel_models:
        cmd.append("--parallel-models")
    if args.save_npy:
        cmd.append("--save-npy")
    return cmd


def projection_cmd(
    args: argparse.Namespace,
    measure: str,
    datasets: str,
    run_injection: bool,
) -> list[str]:
    cmd = [sys.executable, str(PROJECTION_SCRIPT)]
    add_common_projection_args(cmd, args)
    cmd.extend([
        "--datasets",
        datasets,
        "--measure",
        measure,
    ])
    if run_injection:
        cmd.extend(["--run-injection", "--alpha", str(args.alpha)])
        if args.injection_layers:
            cmd.extend(["--injection-layers", args.injection_layers])
    return cmd


def heatmap_grid_cmd(args: argparse.Namespace, measure: str) -> list[str]:
    base_dir = getattr(args, "_suite_models_dir", args.base_dir)
    cmd = [
        sys.executable,
        str(HEATMAP_GRID_SCRIPT),
        "--models",
        args.models,
        "--measure",
        measure,
        "--base-dir",
        str(base_dir),
        "--dpi",
        str(args.plot_dpi),
        "--datasets",
        args.injection_datasets,
    ]
    return cmd


def build_commands(args: argparse.Namespace, suite_output_dir: Path) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    projection_root = suite_output_dir / "models"
    args._suite_models_dir = projection_root

    if not args.skip_clean:
        commands.extend([
            (
                "clean harmfulness_tinst",
                projection_cmd(
                    args=args,
                    measure="harmfulness_tinst",
                    datasets=args.clean_datasets,
                    run_injection=False,
                ),
            ),
            (
                "clean refusal_tpost",
                projection_cmd(
                    args=args,
                    measure="refusal_tpost",
                    datasets=args.clean_datasets,
                    run_injection=False,
                ),
            ),
        ])

    if not args.skip_injection:
        commands.extend([
            (
                "injection refusal_tpost",
                projection_cmd(
                    args=args,
                    measure="refusal_tpost",
                    datasets=args.injection_datasets,
                    run_injection=True,
                ),
            ),
            (
                "injection harmfulness_tinst",
                projection_cmd(
                    args=args,
                    measure="harmfulness_tinst",
                    datasets=args.injection_datasets,
                    run_injection=True,
                ),
            ),
        ])

    if not args.skip_post_plots:
        for measure in split_csv(args.plot_measures):
            cmd = heatmap_grid_cmd(args, measure)
            cmd.extend(["--output-dir", str(suite_output_dir / "plots" / measure_safe(measure))])
            commands.append((f"heatmap grid {measure}", cmd))

    for _, cmd in commands:
        if str(PROJECTION_SCRIPT) in cmd:
            cmd.extend(["--output-dir", str(projection_root)])

    return commands


def measure_safe(measure: str) -> str:
    return measure.replace("/", "_")


def run_command(label: str, cmd: list[str], dry_run: bool) -> None:
    print(f"\n[suite] {label}", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def figure_dir_for_base(args: argparse.Namespace) -> Path:
    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.is_absolute():
        base_dir = SCRIPT_DIR / base_dir
    figure_dir = base_dir / "Figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def compose_paper_fig7_fig8(args: argparse.Namespace, suite_output_dir: Path) -> None:
    figure_dir = figure_dir_for_base(args)
    cmd = [
        sys.executable,
        str(PAPER_CLEAN_PROJECTION_SCRIPT),
        "--fig",
        "all",
        "--base-dir",
        str(suite_output_dir / "models"),
        "--dpi",
        str(args.plot_dpi),
        "--output-dir",
        str(figure_dir),
    ]
    print("\n[suite] composing paper Fig7/Fig8:", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def compose_paper_fig9(args: argparse.Namespace, suite_output_dir: Path) -> None:
    figure_dir = figure_dir_for_base(args)
    cmd = [
        sys.executable,
        str(HEATMAP_GRID_SCRIPT),
        "--models",
        args.models,
        "--measure",
        "refusal_tpost",
        "--base-dir",
        str(suite_output_dir / "models"),
        "--dpi",
        str(args.plot_dpi),
        "--datasets",
        args.injection_datasets,
        "--output-dir",
        str(figure_dir),
        "--output-name",
        "Fig9",
    ]
    print("\n[suite] composing paper Fig9:", flush=True)
    print("  " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not PROJECTION_SCRIPT.exists():
        raise FileNotFoundError(PROJECTION_SCRIPT)
    if not HEATMAP_GRID_SCRIPT.exists():
        raise FileNotFoundError(HEATMAP_GRID_SCRIPT)

    suite_output_dir = resolve_suite_output_dir(args)
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[suite] output root: {suite_output_dir}", flush=True)
    commands = build_commands(args, suite_output_dir)
    print(f"[suite] running {len(commands)} job(s)", flush=True)
    for label, cmd in commands:
        run_command(label, cmd, args.dry_run)
    if args.auto_paper_figures:
        compose_paper_fig7_fig8(args, suite_output_dir)
        compose_paper_fig9(args, suite_output_dir)
    print("\n[suite] done", flush=True)


if __name__ == "__main__":
    main()
