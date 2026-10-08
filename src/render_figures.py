#!/usr/bin/env python3
"""Single entry point for rendering all paper and appendix figures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PT = SCRIPT_DIR / "out_pt"
TINST_RESULTS = SCRIPT_DIR / "results" / "steered_tinst_restore_refusal_projection"


SECTIONS = ("main", "appendix-a", "appendix-b", "appendix-c")


def _run(args: list[str]) -> None:
    print("\n$ " + " ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(args, cwd=SCRIPT_DIR.parent, check=True, env=env)


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_file() or child.is_symlink():
            child.unlink()


def render_main(dpi: int | None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "figures" / "plot_main_all_figures.py"),
        "--output-dir",
        str(OUT_PT / "Figure"),
    ]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    _run(cmd)
    _run([sys.executable, str(SCRIPT_DIR / "figures" / "make_fig7_2.py")])
    tinst_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "steered_tinst_restore_refusal_projection.py"),
        "--plot-existing",
        "--output-dir",
        str(TINST_RESULTS),
        "--figure-output",
        str(OUT_PT / "Figure" / "Fig4.png"),
    ]
    if dpi is not None:
        tinst_cmd += ["--plot-dpi", str(dpi)]
    _run(tinst_cmd)


def render_appendix_a(dpi: int | None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "figures" / "plot_appendix_a.py"),
        "--output-dir",
        str(OUT_PT / "Figure_Appendix_A"),
        "--results-dir",
        str(TINST_RESULTS),
    ]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    _run(cmd)

    tinst_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "steered_tinst_restore_refusal_projection_appendix_A.py"),
        "--plot-existing",
        "--output-dir",
        str(TINST_RESULTS),
        "--figure-output",
        str(OUT_PT / "Figure_Appendix_A" / "Fig3.png"),
    ]
    if dpi is not None:
        tinst_cmd += ["--plot-dpi", str(dpi)]
    _run(tinst_cmd)


def render_appendix_b(dpi: int | None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "figures" / "plot_appendix_b.py"),
        "--output-dir",
        str(OUT_PT / "Figure_Appendix_B"),
    ]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    _run(cmd)

    tinst_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "experiments" / "steered_tinst_restore_refusal_projection_appendix_B.py"),
        "--plot-existing",
        "--output-dir",
        str(TINST_RESULTS),
        "--figure-output",
        str(OUT_PT / "Figure_Appendix_B" / "Fig3.png"),
    ]
    if dpi is not None:
        tinst_cmd += ["--plot-dpi", str(dpi)]
    _run(tinst_cmd)


def render_appendix_c(dpi: int | None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "figures" / "plot_appendix_c.py"),
        "--output-dir",
        str(OUT_PT / "Figure"),
    ]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render main and appendix figures from regenerated JSON results.")
    parser.add_argument(
        "--section",
        action="append",
        choices=SECTIONS + ("all",),
        help="Section to render. Repeatable. Default: all.",
    )
    parser.add_argument("--all", action="store_true", help="Render all sections.")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=None, help="Override DPI for all renderers.")
    return parser.parse_args()


def clean_dirs_for_sections(sections: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for section in sections:
        if section in {"main", "appendix-c"}:
            dirs.append(OUT_PT / "Figure")
        elif section == "appendix-a":
            dirs.append(OUT_PT / "Figure_Appendix_A")
        elif section == "appendix-b":
            dirs.append(OUT_PT / "Figure_Appendix_B")
    return list(dict.fromkeys(dirs))


def main() -> None:
    args = parse_args()
    requested = args.section or []
    if args.all or not requested or "all" in requested:
        requested = list(SECTIONS)

    if args.clean:
        for d in clean_dirs_for_sections(requested):
            _clean_dir(d)

    for section in requested:
        if section == "main":
            render_main(args.dpi)
        elif section == "appendix-a":
            render_appendix_a(args.dpi)
        elif section == "appendix-b":
            render_appendix_b(args.dpi)
        elif section == "appendix-c":
            render_appendix_c(args.dpi)

    print("\nDone. Figures are under src/out_pt/Figure*.", flush=True)


if __name__ == "__main__":
    main()
