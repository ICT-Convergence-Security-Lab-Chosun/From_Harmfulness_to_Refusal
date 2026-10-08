#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import (
    get_assistant_suffix_text,
    get_output_root,
    get_template_parts,
    get_transformer_layers,
    load_model_and_tokenizer as load_model_and_tokenizer_common,
    normalize_model_name,
    resolve_model_spec,
)


NUM_TOKEN_HIDDEN = 2

def read_rows(file_path: str):
    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for row in f:
                rows.append(json.loads(row))
    except Exception:
        with open(file_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    return rows


def format_prompt(entry, model: str | None = None, model_id: str | None = None) -> str:
    if isinstance(entry, dict) and "prompt" in entry:
        entry = entry["prompt"]

    if isinstance(entry, dict):
        if "instruction" in entry:
            text = entry["instruction"]
        elif "question" in entry:
            text = entry["question"]
        elif "bad_q" in entry:
            text = entry["bad_q"]
        else:
            text = str(entry)
    else:
        text = str(entry)

    prefix, suffix = get_template_parts(model=model, model_id=model_id, use_template=True)
    return f"{prefix}{text}{suffix}"


def build_component_modules(model: AutoModelForCausalLM, component: str) -> List[torch.nn.Module]:
    layers = get_transformer_layers(model)
    if component == "hidden":
        return layers
    if component == "attn":
        return [layer.self_attn for layer in layers]
    if component == "mlp":
        return [layer.mlp for layer in layers]
    raise ValueError(f"Unsupported component: {component}")


def resolve_module_output_tensor(value) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim < 3:
            raise TypeError(f"Tensor rank too small for activations: {tuple(value.shape)}")
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor) and item.ndim >= 3:
                return item
    raise TypeError(f"Unsupported module output type: {type(value)}")


def resolve_positions_from_mask(
    seq_len: int,
    positions: List[int],
    attention_mask_row: Tensor | None = None,
) -> List[int]:
    effective_len = seq_len
    left_pad = 0
    if attention_mask_row is not None:
        attention_mask_row = attention_mask_row.to(dtype=torch.int64, device="cpu")
        effective_len = int(attention_mask_row.sum().item())
        if effective_len <= 0:
            raise ValueError("Attention mask indicates an empty sequence.")
        left_pad = seq_len - effective_len

    resolved = []
    for pos in positions:
        idx = pos if pos >= 0 else effective_len + pos
        if idx < 0 or idx >= effective_len:
            raise IndexError(
                f"Resolved position {pos} is out of bounds for effective sequence length {effective_len}"
            )
        resolved.append(left_pad + idx)
    return resolved


def gather_positions_per_sample(
    activation: Tensor,
    position_rows: List[List[int]],
) -> Tensor:
    gathered_rows = []
    for sample_idx, indices in enumerate(position_rows):
        index_tensor = torch.tensor(indices, device=activation.device, dtype=torch.long)
        gathered_rows.append(activation[sample_idx].index_select(0, index_tensor))
    return torch.stack(gathered_rows, dim=0)


def slice_activations(
    activation: Tensor,
    positions: List[int],
    attention_mask: Tensor | None = None,
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN,
) -> Tensor:
    if whole_seq:
        return activation.clone().detach().cpu()

    batch_size, seq_len, _ = activation.shape
    if seq_len < len(positions):
        raise ValueError(f"Sequence too short for positions: seq_len={seq_len}, positions={positions}")

    if attention_mask is not None and attention_mask.shape[:2] != activation.shape[:2]:
        raise ValueError(
            f"attention_mask shape mismatch: mask={tuple(attention_mask.shape)}, activation={tuple(activation.shape)}"
        )

    position_rows = []
    context_rows = []
    for sample_idx in range(batch_size):
        mask_row = None if attention_mask is None else attention_mask[sample_idx]
        resolved_positions = resolve_positions_from_mask(seq_len, positions, mask_row)
        first_position = resolved_positions[0]
        if step > first_position:
            raise IndexError(
                f"Not enough non-padding context tokens before positions {positions}; "
                f"need {step}, first resolved position is {first_position}"
            )
        context_rows.append(list(range(first_position - step, first_position)))
        position_rows.append(resolved_positions)

    context = gather_positions_per_sample(activation, context_rows) if step > 0 else activation[:, :0, :]
    pos_activations = gather_positions_per_sample(activation, position_rows)
    merged_activation = torch.cat([context, pos_activations], dim=1)
    return merged_activation.clone().detach().cpu()


class ForwardHookCache:
    def __init__(self, component: str, block_modules: List[torch.nn.Module], positions: List[int], whole_seq: bool):
        self.component = component
        self.block_modules = block_modules
        self.positions = positions
        self.whole_seq = whole_seq
        self.cache = [[] for _ in range(len(block_modules))]
        self.handles = []
        self.attention_mask = None

    def _hook(self, layer_idx: int):
        def fn(module, inputs, output):
            activation = resolve_module_output_tensor(output).half()
            self.cache[layer_idx].append(
                slice_activations(
                    activation,
                    self.positions,
                    attention_mask=self.attention_mask,
                    whole_seq=self.whole_seq,
                )
            )
        return fn

    def __enter__(self):
        if self.component in ("attn", "mlp"):
            for layer_idx, module in enumerate(self.block_modules):
                self.handles.append(module.register_forward_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()


def load_model_and_tokenizer(model_name: str | None = None, model_id: str | None = None):
    model, tokenizer, _ = load_model_and_tokenizer_common(model=model_name, model_id=model_id)
    return model, tokenizer


def get_instruction_positions(tokenizer, model_name: str | None = None, model_id: str | None = None):
    inst_token = get_assistant_suffix_text(model=model_name, model_id=model_id)
    tokenized_inst = tokenizer(inst_token, return_tensors="pt", add_special_tokens=False)
    print("inst_token", tokenizer.decode(tokenized_inst.input_ids[0]))
    return [i for i in range(-len(tokenized_inst.input_ids[0]), 0, 1)]


def extract_component_activations(
    model,
    tokenizer,
    prompts: List[str],
    component: str,
    batch_size: int,
    positions: List[int],
    whole_seq: bool,
):
    block_modules = build_component_modules(model, component)
    cache = ForwardHookCache(component, block_modules, positions, whole_seq)

    def tokenize(batch_prompts):
        return tokenizer(batch_prompts, padding=True, return_tensors="pt")

    with cache:
        for start in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenize(batch_prompts)
            cache.attention_mask = inputs.attention_mask
            model_kwargs = {
                "input_ids": inputs.input_ids.to(model.device),
                "attention_mask": inputs.attention_mask.to(model.device),
            }
            if component == "hidden":
                outputs = model(output_hidden_states=True, **model_kwargs)
                hidden_states = outputs.hidden_states[1:]
                for layer_idx, activation in enumerate(hidden_states):
                    cache.cache[layer_idx].append(
                        slice_activations(
                            activation.half(),
                            positions,
                            attention_mask=inputs.attention_mask,
                            whole_seq=whole_seq,
                        )
                    )
            else:
                model(**model_kwargs)

    flat_list = [torch.cat(layer_cache, dim=0) for layer_cache in cache.cache]
    result = torch.stack(flat_list, dim=0)
    mean_activations = result.mean(dim=1)
    return mean_activations, result


def parse_components(value: str) -> List[str]:
    if value == "all":
        return ["hidden", "attn", "mlp"]
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Optional model alias")
    parser.add_argument("--model-id", default=None, help="HF model id; preferred over --model")
    parser.add_argument("--input", default="../data/jbb-harmful.json", help="Input dataset path")
    parser.add_argument("--label", default="jbb", help="Saved label name, e.g. refuse or test")
    parser.add_argument("--components", default="all", help="One of hidden, attn, mlp, all, or comma-separated list")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to extract")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output-dir", default="out_pt", help="Base output directory")
    parser.add_argument("--ret-whole-seq", type=int, default=0, help="Save whole sequence instead of selected tokens")
    args = parser.parse_args()

    spec = resolve_model_spec(model=args.model, model_id=args.model_id)
    args.model = spec.model_name
    args.model_id = spec.model_id

    model, tokenizer = load_model_and_tokenizer(args.model, args.model_id)
    rows = read_rows(args.input)
    rows = rows[args.start : args.start + args.count]
    prompts = [format_prompt(row, args.model, args.model_id) for row in rows]
    positions = get_instruction_positions(tokenizer, args.model, args.model_id)
    print('positions: ', positions)

    output_dir = get_output_root(args.output_dir, model=args.model, model_id=args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    for component in parse_components(args.components):
        mean_activations, full_activations = extract_component_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            component=component,
            batch_size=args.batch_size,
            positions=positions,
            whole_seq=bool(args.ret_whole_seq),
        )

        full_path = output_dir / f"{component}-{args.label}.pt"
        mean_path = output_dir / f"{component}-{args.label}-mean.pt"
        prompts_path = output_dir / f"{component}-{args.label}_prompts_used.json"

        torch.save(full_activations.to("cpu"), full_path)
        torch.save(mean_activations.to("cpu"), mean_path)
        with prompts_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"saved full activations to {full_path}")
        print(f"saved mean activations to {mean_path}")
        print(f"saved prompts to {prompts_path}")


if __name__ == "__main__":
    main()
