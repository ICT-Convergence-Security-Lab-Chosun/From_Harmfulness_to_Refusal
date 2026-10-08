#!/usr/bin/env python3
"""
eval_direct_refusal.py - response generation only

Generate model responses for harmful and benign datasets, then save them under
out_responses/. Run judging separately with judge_direct_refusal.py.

Usage:
    python eval_direct_refusal.py
    python eval_direct_refusal.py --models llama3,qwen2.5
    python eval_direct_refusal.py --models my_model=org/model-id
    python eval_direct_refusal.py --datasets advbench,harmbench --count 50
    python eval_direct_refusal.py --list-models

# Step 1: generate responses
python eval_direct_refusal.py --models llama3,qwen2.5

# Step 2: judge + plot (default)
export OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
python judge_direct_refusal.py --input out_responses/direct_refusal_XXX/

# Judge only
python judge_direct_refusal.py --input out_responses/direct_refusal_XXX/ --mode judge

# Plot only, after judging is done
python judge_direct_refusal.py --input out_responses/direct_refusal_XXX/ --mode plot

# If interrupted, rerun the command; completed rows are skipped.
python judge_direct_refusal.py --input out_responses/direct_refusal_XXX/


judge_direct_refusal.py 
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import GenerationConfig

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import load_model_and_tokenizer

# Model Registry

MODEL_REGISTRY: dict[str, str] = {
    "llama3":   "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama2":   "meta-llama/Llama-2-7b-chat-hf",
    "qwen2.5":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "gemma":    "google/gemma-2-9b-it",
    "gemma2":   "google/gemma-2-9b-it",
}

DEFAULT_MODELS   = "llama3,qwen2.5"
DEFAULT_DATASETS = (
    "advbench-common,alpaca-common,"
    "sorry-misrepresentation,sorry-authority-endorsement,sorry-expert-endorsement"
)

DATA_PATHS: dict[str, str] = {
    "advbench-common":         "../data/advbench_common.json",
    "alpaca-common":           "../data/alpaca_common.json",
    "sorry-misrepresentation": "../data/sorry-misrepresentation.json",
    "sorry-authority-endorsement": "../data/sorry-authority-endorsement.json",
    "sorry-expert-endorsement":   "../data/sorry-expert-endorsement.json",
}

# Data loading

def _load_json_or_jsonl(path: Path, max_rows: int) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[:max_rows]
        return []
    except json.JSONDecodeError:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                    if len(rows) >= max_rows:
                        break
        return rows


def extract_instruction(row) -> Optional[str]:
    if not isinstance(row, dict):
        return str(row)
    if "turns" in row and row["turns"] is not None:
        turns = row["turns"]
        if isinstance(turns, list) and turns:
            return str(turns[0])
        return str(turns) if turns else None
    for key in ("bad_q", "instruction", "question", "prompt", "text"):
        if key in row and row[key]:
            return str(row[key])
    return None


def load_dataset(name: str, data_dir: Path, max_rows: int = 500) -> list[str]:
    rel_path = DATA_PATHS.get(name)
    if rel_path is None:
        print(f"  [warn] unknown dataset: {name}")
        return []
    path = (data_dir / rel_path).resolve()
    if not path.exists():
        print(f"  [warn] missing file: {path}")
        return []
    rows = _load_json_or_jsonl(path, max_rows)
    return [inst for row in rows if (inst := extract_instruction(row))]


# Prompt construction / response generation

def build_chat_prompts(tokenizer, instructions: list[str]) -> list[str]:
    prompts = []
    for inst in instructions:
        messages = [{"role": "user", "content": inst}]
        try:
            # enable_thinking=False forces NON-thinking output on thinking models
            # (e.g. Qwen3.5); harmless on models whose template lacks the kwarg.
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = inst
        prompts.append(prompt)
    return prompts


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 128,
) -> list[str]:
    device  = next(model.parameters()).device
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        temperature=None, top_p=None, top_k=None,
    )
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="  generating", leave=False):
        batch  = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, return_tensors="pt",
            truncation=True, max_length=1024,
        )
        ids  = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg)
        new_tok = out[:, ids.shape[1]:]
        responses.extend(
            t.strip() for t in tokenizer.batch_decode(new_tok, skip_special_tokens=True)
        )
    return responses


# CLI parsing

def parse_models_arg(models_str: str) -> dict[str, str]:
    """
    Parse a comma-separated model list.
      registered aliases: llama3,qwen2.5
      direct HF ID:       meta-llama/Llama-3.1-8B-Instruct
      alias=hf_id:       my_model=org/model-name
    """
    registry = dict(MODEL_REGISTRY)
    result: dict[str, str] = {}
    for token in models_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            alias, hf_id = token.split("=", 1)
            registry[alias.strip()] = hf_id.strip()
            result[alias.strip()]   = hf_id.strip()
        elif token in registry:
            result[token] = registry[token]
        elif "/" in token:
            alias = token.split("/")[-1].lower()
            result[alias] = token
        else:
            print(f"[warn] unknown model alias: {token!r}. registered aliases: {list(registry.keys())}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate model responses and save them under out_responses/. Run judging with judge_direct_refusal.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default=DEFAULT_MODELS,
        help=f"Models to evaluate, comma-separated. Add a model with alias=hf_id. registered aliases: {list(MODEL_REGISTRY.keys())}")
    p.add_argument("--datasets", default=DEFAULT_DATASETS,
        help=f"Datasets to evaluate, comma-separated. Available: {list(DATA_PATHS.keys())}")
    p.add_argument("--count",          type=int, default=100,  help="Samples per dataset")
    p.add_argument("--offset",         type=int, default=0,    help="Dataset start offset")
    p.add_argument("--max-rows",       type=int, default=500,  help="Maximum rows to load from each dataset")
    p.add_argument("--batch-size",     type=int, default=4,    help="Generation batch size")
    p.add_argument("--max-new-tokens", type=int, default=128,  help="Maximum generated tokens")
    p.add_argument("--data-dir",   type=Path, default=SCRIPT_DIR.parent / "data",
        help="Data directory")
    p.add_argument("--output-dir", type=Path, default=None,
        help="Output directory (default: out_responses/direct_refusal_TIMESTAMP)")
    p.add_argument("--list-models", action="store_true",
        help="List registered model aliases and exit")
    return p.parse_args()


# Main

def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Registered model aliases:")
        for alias, hf_id in MODEL_REGISTRY.items():
            print(f"  {alias:20s} → {hf_id}")
        return

    models_map = parse_models_arg(args.models)
    if not models_map:
        print("No models to evaluate.")
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = SCRIPT_DIR / "out_responses" / f"direct_refusal_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Load datasets.
    print("=== Load datasets ===")
    all_instructions: dict[str, list[str]] = {}
    for ds in datasets:
        insts = load_dataset(ds, args.data_dir, args.max_rows)
        if insts:
            all_instructions[ds] = insts[args.offset : args.offset + args.count]
            print(f"  {ds}: {len(all_instructions[ds])} samples")
        else:
            print(f"  {ds}: no data; skipping")

    if not all_instructions:
        print("No data loaded.")
        return

    # Generate responses as a flat list: [{model, dataset, instruction, response}, ...].
    all_items: list[dict] = []

    for alias, hf_id in models_map.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {alias}  ({hf_id})")
        print(f"{'='*60}")

        try:
            model, tokenizer, spec = load_model_and_tokenizer(model_id=hf_id)
            model.eval()
            print(f"  loaded: {spec.model_id}")
        except Exception as e:
            print(f"  load failed: {e}")
            continue

        for ds, instructions in all_instructions.items():
            print(f"\n  [{ds}] generating {len(instructions)} samples...")
            prompts   = build_chat_prompts(tokenizer, instructions)
            responses = generate_responses(
                model, tokenizer, prompts, args.batch_size, args.max_new_tokens,
            )
            for inst, resp in zip(instructions, responses):
                all_items.append({
                    "model":       alias,
                    "dataset":     ds,
                    "instruction": inst,
                    "response":    resp,
                })
            print(f"    done ({len(responses)})")

        del model
        torch.cuda.empty_cache()

    if not all_items:
        print("\nNo responses generated.")
        return

    out_path = args.output_dir / "responses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\nSaved responses: {out_path}  ({len(all_items)} items)")
    print(f"\nNext step - judge + plot:")
    print(f"  python judge_direct_refusal.py --input {out_path}")


if __name__ == "__main__":
    main()
