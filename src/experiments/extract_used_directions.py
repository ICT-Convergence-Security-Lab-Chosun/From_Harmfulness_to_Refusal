#!/usr/bin/env python3
"""
Extract only the direction tensors consumed by the coupling/steering scripts.

Default outputs per model:
  out_pt/<model-output-name>/hf-hidden-harmful-minus-harmless.pt
  out_pt/<model-output-name>/refuse-hidden-refuse-minus-accept.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import tempfile
from pathlib import Path

import torch

import extract_hidden
from extract_hidden import (
    build_standard_run_params,
    generate_directions,
    load_model_and_tokenizer,
    maybe_set_instruction_positions,
)
from model_utils import resolve_model_spec
from utils import read_row


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODELS = "qwen25,llama31"
DEFAULT_HARMFUL_DATA = REPO_ROOT / "data" / "advbench_extract.json"
DEFAULT_HARMLESS_DATA = REPO_ROOT / "data" / "alpaca_extract.json"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def select_rows(params: dict) -> tuple[list, list]:
    label_a_rows = read_row(params["harmful_pth"])
    if params.get("random_sample_harmful"):
        random.seed(params["left"] % len(label_a_rows))
        label_a_rows = random.sample(label_a_rows, 1)
    elif params["left"] < len(label_a_rows):
        label_a_rows = label_a_rows[params["left"] : params["right"]]
    else:
        start_idx = params["left"] % len(label_a_rows)
        label_a_rows = label_a_rows[start_idx : start_idx + 1]

    label_b_rows = read_row(params["harmless_pth"])[params["left"] : params["right"]]
    return label_a_rows, label_b_rows


def add_runtime_defaults(params: dict, args: argparse.Namespace) -> dict:
    params = copy.deepcopy(params)
    params.update(
        {
            "batch_size": args.batch_size,
            "model": params.get("model"),
            "model_id": params.get("model_id"),
            "harmful_pth": str(Path(args.harmful_data).expanduser()),
            "harmless_pth": str(Path(args.harmless_data).expanduser()),
            "extract_component": "hidden",
            "extract_only": 0,
            "ret_whole_seq": 0,
            "use_persuade_harmful": 0,
            "use_persuade_harmless": 0,
            "use_sys_harmful": 0,
            "random_sample_harmful": 0,
            "seed": args.seed,
            "left": args.left,
            "right": args.right,
        }
    )
    return params


def extract_one_mode(model, tokenizer, model_name: str, model_id: str, mode_dir: str, args: argparse.Namespace) -> Path:
    final_params = add_runtime_defaults(
        build_standard_run_params(model_name, model_id, mode_dir, output_base_dir=args.output_base_dir),
        args,
    )
    final_params["model"] = model_name
    final_params["model_id"] = model_id
    maybe_set_instruction_positions(final_params, tokenizer)

    final_output = Path(final_params["output_pth_ab"])
    final_output.parent.mkdir(parents=True, exist_ok=True)
    label_a_rows, label_b_rows = select_rows(final_params)

    with tempfile.TemporaryDirectory(prefix=f"extract-used-{mode_dir}-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        run_params = copy.deepcopy(final_params)
        run_params.update(
            {
                "output_pth_ab": str(tmp_dir_path / final_output.name),
                "output_pth_ba": str(tmp_dir_path / final_output.name.replace("-minus-", "-reverse-minus-")),
                "output_pth_label_a": str(tmp_dir_path / f"{mode_dir}-label-a.pt"),
                "output_pth_label_b": str(tmp_dir_path / f"{mode_dir}-label-b.pt"),
                "output_pth_mean_label_a": str(tmp_dir_path / f"{mode_dir}-label-a-mean.pt"),
                "output_pth_mean_label_b": str(tmp_dir_path / f"{mode_dir}-label-b-mean.pt"),
            }
        )
        direction = generate_directions(model, tokenizer, label_a_rows, label_b_rows, run_params)

    if direction is None:
        raise RuntimeError(f"Failed to extract {mode_dir} direction for {model_id}")
    torch.save(direction.to("cpu"), final_output)
    return final_output


def run_for_model(model_alias: str | None, model_id: str | None, args: argparse.Namespace) -> list[Path]:
    spec = resolve_model_spec(model=model_alias, model_id=model_id)
    extract_hidden.MODEL = spec.model_name
    print(f"[extract-used] loading {spec.model_id}", flush=True)
    model, tokenizer = load_model_and_tokenizer(spec.model_name, spec.model_id)
    model.eval()

    outputs = []
    for mode_dir in parse_csv(args.modes):
        if mode_dir not in {"hf", "refuse"}:
            raise ValueError(f"Unsupported mode: {mode_dir}")
        print(f"[extract-used] extracting {spec.output_name} / {mode_dir} / hidden", flush=True)
        outputs.append(extract_one_mode(model, tokenizer, spec.model_name, spec.model_id, mode_dir, args))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract only hidden harmfulness/refusal direction vectors used by steering/coupling scripts."
    )
    parser.add_argument("--model", default=None, help="Single model alias, e.g. qwen25, llama31, gemma.")
    parser.add_argument("--model-id", default=None, help="Single Hugging Face model id.")
    parser.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated model aliases. Default: {DEFAULT_MODELS}. Ignored when --model or --model-id is set.",
    )
    parser.add_argument("--modes", default="hf,refuse", help="Comma-separated subset of hf,refuse.")
    parser.add_argument(
        "--harmful-data",
        default=str(DEFAULT_HARMFUL_DATA),
        help="Dataset for the harmful side of the harmful-minus-harmless direction.",
    )
    parser.add_argument(
        "--harmless-data",
        default=str(DEFAULT_HARMLESS_DATA),
        help="Dataset for the harmless/accept side of both standard directions.",
    )
    parser.add_argument("--output-base-dir", default="out_pt", help="Base output directory.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.harmful_data).expanduser().exists():
        raise FileNotFoundError(f"Missing harmful dataset: {args.harmful_data}")
    if not Path(args.harmless_data).expanduser().exists():
        raise FileNotFoundError(f"Missing harmless dataset: {args.harmless_data}")

    print(f"[extract-used] harmful data : {Path(args.harmful_data).expanduser()}", flush=True)
    print(f"[extract-used] harmless data: {Path(args.harmless_data).expanduser()}", flush=True)

    all_outputs: list[Path] = []
    if args.model_id:
        all_outputs.extend(run_for_model(args.model, args.model_id, args))
    else:
        model_list = args.model or args.models or DEFAULT_MODELS
        models = parse_csv(model_list)
        for model_alias in models:
            all_outputs.extend(run_for_model(model_alias, None, args))

    print("[extract-used] wrote:", flush=True)
    for output in all_outputs:
        print(f"  {output}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
