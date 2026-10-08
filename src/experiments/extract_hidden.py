import json
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional
from torch import Tensor
from tqdm import tqdm
from utils import read_row, formatInp_llama_persuasion
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import functools
import random
import tempfile
from pathlib import Path

from model_utils import (
    get_default_binary_dataset_paths,
    get_assistant_suffix_text,
    get_output_root,
    get_transformer_layers,
    load_model_and_tokenizer as load_model_and_tokenizer_common,
    resolve_model_spec,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
MODEL = ''
NUM_TOKEN_HIDDEN = 2  # by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens
DEFAULT_HARMFUL_DATA, DEFAULT_HARMLESS_DATA = get_default_binary_dataset_paths()
STANDARD_VECTOR_HARMFUL_DATA = str(REPO_ROOT / "data" / "advbench.json")
STANDARD_VECTOR_HARMLESS_DATA = str(REPO_ROOT / "data" / "alpaca_data_instruction.json")
STANDARD_VECTOR_EXAMPLE_COUNT = 100

def parse_components(value: str) -> List[str]:
    """Support extracting one, many, or all components in a single run."""
    if value == 'all':
        return ['hidden', 'attn', 'mlp']
    return [item.strip() for item in value.split(',') if item.strip()]


def append_component_to_path(path: str, component: str) -> str:
    """Derive standard component-specific output paths when multiple components are requested."""
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    for existing_component in ("hidden", "attn", "mlp"):
        token = f"-{existing_component}-"
        if token in filename:
            filename = filename.replace(token, f"-{component}-", 1)
            return os.path.join(dirname, filename)

    root, ext = os.path.splitext(path)
    return f"{root}-{component}{ext}"


def build_standard_run_params(model: str, model_id: str, mode_dir: str, output_base_dir: str = 'out_pt') -> dict:
    """Return the standardized dataset, label, and output layout for one run."""
    if mode_dir == 'hf':
        label_a, label_b = 'harmful', 'harmless'
        extract_hidden_inst_token = 1
        extract_harmful_token_only = 1
        positions = [-1]
    elif mode_dir == 'refuse':
        label_a, label_b = 'refuse', 'accept'
        extract_hidden_inst_token = 0
        extract_harmful_token_only = 0
        positions = [-1]
    else:
        raise ValueError(f"Unsupported standard mode_dir: {mode_dir}")

    output_dir = str(get_output_root(output_base_dir, model=model, model_id=model_id))
    return {
        'harmful_pth': STANDARD_VECTOR_HARMFUL_DATA,
        'harmless_pth': STANDARD_VECTOR_HARMLESS_DATA,
        'label_a': label_a,
        'label_b': label_b,
        'extract_component': 'all',
        'extract_hidden_inst_token': extract_hidden_inst_token,
        'extract_harmful_token_only': extract_harmful_token_only,
        'positions': positions,
        'num_context_tokens': 0,
        'left': 0,
        'right': STANDARD_VECTOR_EXAMPLE_COUNT,
        'mode_dir': mode_dir,
        'output_pth_ab': os.path.join(output_dir, f"{mode_dir}-hidden-{label_a}-minus-{label_b}.pt"),
        'output_pth_ba': os.path.join(output_dir, f"{mode_dir}-hidden-{label_b}-minus-{label_a}.pt"),
        'output_pth_label_a': os.path.join(output_dir, f"hidden-{label_a}.pt"),
        'output_pth_label_b': os.path.join(output_dir, f"hidden-{label_b}.pt"),
        'output_pth_mean_label_a': os.path.join(output_dir, f"hidden-{label_a}-mean.pt"),
        'output_pth_mean_label_b': os.path.join(output_dir, f"hidden-{label_b}-mean.pt"),
    }


def build_component_modules(model: AutoModelForCausalLM, component: str) -> List[torch.nn.Module]:
    """Return the per-layer modules used for component extraction."""
    layers = get_transformer_layers(model)
    if component == 'hidden':
        return layers
    if component == 'attn':
        return [layer.self_attn for layer in layers]
    if component == 'mlp':
        return [layer.mlp for layer in layers]
    raise ValueError(f"Unsupported component: {component}")


def resolve_module_output_tensor(value) -> Tensor:
    """Extract the sequence activation tensor from a module forward output."""
    if isinstance(value, Tensor):
        if value.ndim < 3:
            raise TypeError(f"Tensor rank too small for activations: {tuple(value.shape)}")
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, Tensor) and item.ndim >= 3:
                return item
    raise TypeError(f"Unsupported module output type: {type(value)}")


def select_mode_slice(mean_activations: Tensor, mode_dir: str) -> Tensor:
    """Select the token slice used for direction comparisons."""
    if mode_dir == 'hf':
        return mean_activations[:, NUM_TOKEN_HIDDEN - 1]
    if mode_dir == 'refuse':
        return mean_activations[:, -1]
    raise ValueError(f"Unsupported mode_dir: {mode_dir}")


def recover_direction_from_saved_means(args: dict) -> Optional[Tensor]:
    """Rebuild a missing direction tensor from the per-label mean activation files."""
    mean_label_a_path = Path(args['output_pth_mean_label_a'])
    mean_label_b_path = Path(args['output_pth_mean_label_b'])
    if not mean_label_a_path.exists() or not mean_label_b_path.exists():
        return None

    mean_label_a = torch.load(mean_label_a_path, map_location='cpu')
    mean_label_b = torch.load(mean_label_b_path, map_location='cpu')
    if not isinstance(mean_label_a, Tensor):
        mean_label_a = torch.as_tensor(mean_label_a)
    if not isinstance(mean_label_b, Tensor):
        mean_label_b = torch.as_tensor(mean_label_b)

    if mean_label_a.shape != mean_label_b.shape:
        raise ValueError(
            "Cannot recover direction from mean activations with different shapes: "
            f"{tuple(mean_label_a.shape)} vs {tuple(mean_label_b.shape)}"
        )

    if mean_label_a.ndim == 3:
        # Existing saved direction files align with the last extracted token slice.
        mean_label_a = mean_label_a[:, -1, :]
        mean_label_b = mean_label_b[:, -1, :]
    elif mean_label_a.ndim != 2:
        raise ValueError(
            "Cannot recover direction from mean activations with unsupported shape: "
            f"{tuple(mean_label_a.shape)}"
        )

    mean_diffs = (mean_label_a - mean_label_b).to(dtype=torch.float32, device='cpu')
    mean_diffs_reverse = (mean_label_b - mean_label_a).to(dtype=torch.float32, device='cpu')
    torch.save(mean_diffs, args['output_pth_ab'])
    torch.save(mean_diffs_reverse, args['output_pth_ba'])
    return mean_diffs


def resolve_positions_from_mask(
    seq_len: int,
    positions: List[int],
    attention_mask_row: Optional[Tensor] = None,
) -> List[int]:
    """Resolve token positions against the non-padding span for one sample."""
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

@contextlib.contextmanager
def add_forward_hooks(
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
) -> None:
    """Context manager for temporarily adding forward hooks to modules."""
    handles = []
    try:
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def slice_activations(
    activation: Tensor,
    positions: List[int],
    attention_mask: Optional[Tensor] = None,
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN,
) -> Tensor:
    """Keep either the full sequence or the target token slice used downstream."""
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


def get_module_output_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    attention_mask: Optional[Tensor] = None,
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN,
) -> Callable:
    """Create a forward hook that stores one module output per batch."""
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...], output) -> None:
        activation = resolve_module_output_tensor(output).half()
        cache_full[layer].append(slice_activations(activation, positions, attention_mask, whole_seq, step))
    return hook_fn

def get_mean_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    tokenize_instructions_fn: Callable,
    component: str,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    ret_whole_seq: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Extracts mean activations from model for given instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        instructions: List of input instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (mean activations, full activations)
    """
    torch.cuda.empty_cache()

    n_modules = len(block_modules)
    full_activations = [[] for _ in range(n_modules)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        model_kwargs = {
            "input_ids": inputs.input_ids.to(model.device),
            "attention_mask": inputs.attention_mask.to(model.device),
        }
        if component == 'hidden':
            outputs = model(output_hidden_states=True, **model_kwargs)
            hidden_states = outputs.hidden_states[1:]
            for layer_idx, activation in enumerate(hidden_states):
                full_activations[layer_idx].append(
                    slice_activations(
                        activation.half(),
                        positions,
                        attention_mask=inputs.attention_mask,
                        whole_seq=ret_whole_seq,
                    )
                )
        else:
            fwd_hooks = [
                (block_modules[layer], get_module_output_hook(
                    layer=layer,
                    cache_full=full_activations,
                    positions=positions,
                    attention_mask=inputs.attention_mask,
                    whole_seq=ret_whole_seq
                )) for layer in range(n_modules)
            ]
            with add_forward_hooks(module_forward_hooks=fwd_hooks):
                model(**model_kwargs)

    # cat across batches to get [n_samples, num_pos, hidden] per layer,
    # then stack layers → [n_layers, n_samples, num_pos, hidden]
    flat_list = [torch.cat(inner_list, dim=0) for inner_list in full_activations]
    result = torch.stack(flat_list, dim=0)

    mean_activations = result.mean(dim=1)
    print('mean shape', mean_activations.shape)

    return mean_activations, result

def get_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    tokenize_instructions_fn: Callable,
    component: str,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    extract_only: bool = False,
    use_persuade_harmful: bool = False,
    use_persuade_harmless: bool = False,
    use_sys_harmful: bool = False,
    ret_whole_seq: bool = False,
    temp_cache_dir: Optional[str] = None,
    temp_cache_prefix: str = "extract-hidden",
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Computes mean activation differences between harmful and harmless instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        extract_only: Whether to only extract harmful activations
        use_persuade_harmful: Whether to use persuasion for harmful
        use_persuade_harmless: Whether to use persuasion for harmless
        use_sys_harmful: Whether to use system prompt for harmful
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (harmful mean activations, harmless mean activations,
                harmful full activations, harmless full activations)
    """
    if temp_cache_dir:
        os.makedirs(temp_cache_dir, exist_ok=True)

    mean_tmp_path = None
    full_tmp_path = None
    try:
        mean_activations_harmful, full_activations_harmful = get_mean_activations(
            model, tokenizer, harmful_instructions,
            functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmful, use_sys=use_sys_harmful),
            component, block_modules, batch_size=batch_size, positions=positions,
            ret_whole_seq=ret_whole_seq
        )

        mean_fd, mean_tmp_path = tempfile.mkstemp(
            dir=temp_cache_dir,
            prefix=f"{temp_cache_prefix}-mean-harmful-",
            suffix=".pt",
        )
        os.close(mean_fd)
        full_fd, full_tmp_path = tempfile.mkstemp(
            dir=temp_cache_dir,
            prefix=f"{temp_cache_prefix}-full-harmful-",
            suffix=".pt",
        )
        os.close(full_fd)

        torch.save(mean_activations_harmful, mean_tmp_path)
        torch.save(full_activations_harmful, full_tmp_path)
        del mean_activations_harmful, full_activations_harmful
        torch.cuda.empty_cache()

        if not extract_only:
            mean_activations_harmless, full_activations_harmless = get_mean_activations(
                model, tokenizer, harmless_instructions,
                functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmless),
                component, block_modules, batch_size=batch_size, positions=positions,
                ret_whole_seq=ret_whole_seq
            )
            mean_activations_harmful = torch.load(mean_tmp_path)
            full_activations_harmful = torch.load(full_tmp_path)

            print('mean_activations_harmful shape', mean_activations_harmful.shape)
            print('mean_activations_harmless shape', mean_activations_harmless.shape)
        else:
            mean_activations_harmless = None
            full_activations_harmless = None

        return mean_activations_harmful, mean_activations_harmless, full_activations_harmful, full_activations_harmless
    finally:
        for temp_path in (mean_tmp_path, full_tmp_path):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

def generate_directions(
    model_base: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    args: dict
) -> Optional[Tensor]:
    """
    Generates direction vectors from model activations.

    Args:
        model_base: Base model
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        args: Arguments dictionary

    Returns:
        Mean difference tensor or None if computation fails
    """
    output_paths = [
        args['output_pth_label_a'],
        args['output_pth_label_b'],
        args['output_pth_mean_label_a'],
        args['output_pth_mean_label_b'],
        args['output_pth_ab'],
        args['output_pth_ba'],
    ]
    for output_path in output_paths:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def tokenize_instructions_fn(instructions: List[str], use_persuade: bool = False, use_sys: bool = False) -> dict:
        prompt_model = MODEL
        inps = [
            formatInp_llama_persuasion(
                i,
                use_persuade,
                use_ss=use_sys,
                model=prompt_model,
                model_id=args.get('model_id'),
                use_template=True,
            ) for i in instructions
        ]
        return tokenizer(inps, padding=True, return_tensors="pt")

    temp_cache_dir = os.path.join(
        os.path.dirname(args['output_pth_mean_label_a']) or 'output',
        '.tmp_extract_hidden',
    )
    temp_cache_prefix = (
        f"{args['label_a']}-{args['label_b']}-"
        f"{args['extract_component']}-{args['mode_dir']}-{os.getpid()}"
    )
    model_block_modules = build_component_modules(model_base, args['extract_component'])
    mean_activations_harmful, mean_activations_harmless, all_harmful, all_harmless = get_mean_diff(
        model_base, tokenizer, harmful_instructions, harmless_instructions,
        tokenize_instructions_fn, args['extract_component'], model_block_modules, args['batch_size'],
        args['positions'], args['extract_only'], args['use_persuade_harmful'],
        args['use_persuade_harmless'], args['use_sys_harmful'], args['ret_whole_seq'],
        temp_cache_dir=temp_cache_dir, temp_cache_prefix=temp_cache_prefix,
    )

    torch.save(all_harmful, args['output_pth_label_a'])
    torch.save(all_harmless, args['output_pth_label_b'])
    torch.save(mean_activations_harmful.to('cpu'), args['output_pth_mean_label_a'])
    torch.save(mean_activations_harmless.to('cpu'), args['output_pth_mean_label_b'])

    try:
        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
        mean_activations_harmful = select_mode_slice(mean_activations_harmful, args['mode_dir'])
        mean_activations_harmless = select_mode_slice(mean_activations_harmless, args['mode_dir'])
        if mean_activations_harmful.shape != mean_activations_harmless.shape:
            raise ValueError(
                "Incompatible extracted activation shapes after token selection: "
                f"{tuple(mean_activations_harmful.shape)} vs {tuple(mean_activations_harmless.shape)}. "
                "This usually means artifacts were mixed across different models or concurrent runs. "
                "Re-run extract_hidden.py for this model after clearing the bad outputs."
            )
        mean_diffs = mean_activations_harmful - mean_activations_harmless
        mean_diffs_reverse = mean_activations_harmless - mean_activations_harmful
        assert not mean_diffs.isnan().any()
        assert not mean_diffs_reverse.isnan().any()
        torch.save(mean_diffs.to('cpu'), args['output_pth_ab'])
        torch.save(mean_diffs_reverse.to('cpu'), args['output_pth_ba'])
    except Exception as e:
        print(e)
        recovered = recover_direction_from_saved_means(args)
        if recovered is None:
            mean_diffs = None
        else:
            print(f"Recovered direction tensor from saved means: {args['output_pth_ab']}")
            mean_diffs = recovered

    return mean_diffs


def load_model_and_tokenizer(model_name: str | None = None, model_id: str | None = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model, tokenizer, _ = load_model_and_tokenizer_common(model=model_name, model_id=model_id)
    return model, tokenizer


def maybe_set_instruction_positions(params: dict, tokenizer: AutoTokenizer) -> None:
    """Derive extraction positions from the assistant prefix token span when requested."""
    global NUM_TOKEN_HIDDEN

    NUM_TOKEN_HIDDEN = int(params.get('num_context_tokens', NUM_TOKEN_HIDDEN))

    if not params['extract_hidden_inst_token']:
        return

    if params['model'] == 'vicuna':
        inst_token = 'ASSISTANT:\n'
    else:
        inst_token = get_assistant_suffix_text(model=MODEL, model_id=params.get('model_id'))

    tokenized_inst = tokenizer(inst_token, return_tensors='pt', add_special_tokens=False)
    print('inst_token', tokenizer.decode(tokenized_inst.input_ids[0]))
    params['positions'] = [i for i in range(-len(tokenized_inst.input_ids[0]), 0, 1)]
    if params['extract_harmful_token_only']:
        params['positions'] = [-len(tokenized_inst.input_ids[0]) - 1]
        NUM_TOKEN_HIDDEN = 0


def run_extraction(params: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Run one extraction configuration on already loaded model/tokenizer."""
    harmful_train = read_row(params['harmful_pth'])

    if params['random_sample_harmful']:
        random.seed(params['left'] % len(harmful_train))
        harmful_train = random.sample(harmful_train, 1)
    else:
        if params['left'] < len(harmful_train):
            harmful_train = harmful_train[params['left']:params['right']]
        else:
            start_idx = params['left'] % len(harmful_train)
            harmful_train = harmful_train[start_idx:start_idx + 1]

    harmless_train = read_row(params['harmless_pth'])[params['left']:params['right']]

    prompts_used_path = params['output_pth_ab'].replace('.pt', '_prompts_used.json')
    prompts_used_dir = os.path.dirname(prompts_used_path)
    if prompts_used_dir:
        os.makedirs(prompts_used_dir, exist_ok=True)

    with open(prompts_used_path, 'w') as f:
        json.dump({params['label_a']: harmful_train, params['label_b']: harmless_train}, f, indent=4)

    components = parse_components(params['extract_component'])
    single_component = len(components) == 1

    for component in components:
        run_params = copy.deepcopy(params)
        run_params['extract_component'] = component
        if not single_component:
            run_params['output_pth_ab'] = append_component_to_path(params['output_pth_ab'], component)
            run_params['output_pth_ba'] = append_component_to_path(params['output_pth_ba'], component)
            run_params['output_pth_label_a'] = append_component_to_path(params['output_pth_label_a'], component)
            run_params['output_pth_label_b'] = append_component_to_path(params['output_pth_label_b'], component)
            run_params['output_pth_mean_label_a'] = append_component_to_path(params['output_pth_mean_label_a'], component)
            run_params['output_pth_mean_label_b'] = append_component_to_path(params['output_pth_mean_label_b'], component)

        candidate_directions = generate_directions(model, tokenizer, harmful_train, harmless_train, run_params)
        if candidate_directions is not None:
            print(component, candidate_directions.shape)

def main() -> None:
    """Run the full pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Optional model alias")
    parser.add_argument("--model-id", default=None, type=str, help="HF model id; preferred over --model")
    parser.add_argument("--harmful_pth", default='data/medcq.json', type=str, help="Path to harmful examples")
    parser.add_argument("--harmless_pth", default='data/medcq.json', type=str, help="Path to harmless examples")
    parser.add_argument("--output_pth_label_a", default='output/label_a.pt', type=str, help="Output path for label A full activations")
    parser.add_argument("--output_pth_label_b", default='output/label_b.pt', type=str, help="Output path for label B full activations")
    parser.add_argument("--output_pth_mean_label_a", default='output/label_a_mean.pt', type=str, help="Output path for label A mean activations")
    parser.add_argument("--output_pth_mean_label_b", default='output/label_b_mean.pt', type=str, help="Output path for label B mean activations")
    parser.add_argument('--use_persuade_harmful', default=0, type=int, help='Use persuasion for harmful examples')
    parser.add_argument('--use_persuade_harmless', default=0, type=int, help='Use persuasion for harmless examples')
    parser.add_argument('--use_sys_harmful', default=0, type=int, help='Use system prompt for harmful examples')
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=100, type=int, help='Right index for data slicing')
    parser.add_argument('--random_sample_harmful', default=0, type=int, help='Randomly sample harmful examples')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument("--output_pth_ab", default='output/label_a-minus-label_b.pt', type=str, help="Output path for label A minus label B direction")
    parser.add_argument("--output_pth_ba", default='output/label_b-minus-label_a.pt', type=str, help="Output path for label B minus label A direction")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument('--mode', default='diff-mean', type=str, help='Mode')
    parser.add_argument('--positions', default='-1', type=str, help='Positions to extract')
    parser.add_argument('--extract_only', default=0, type=int, help='Only extract harmful activations')
    parser.add_argument('--ret_whole_seq', default=0, type=int, help='Return whole sequence')
    parser.add_argument('--extract_hidden_inst_token', default=0, type=int, help="Extract hidden state of instruction tokens")
    parser.add_argument('--extract_harmful_token_only', default=0, type=int, help="Extract harmful token only")
    parser.add_argument('--mode_dir', default='hf', type=str, help="Mode for direction extraction: 'hf' or 'refuse'")
    parser.add_argument('--extract_component', default='hidden', type=str, help="Component to extract: hidden, attn, mlp, all, or comma-separated list")
    parser.add_argument('--label_a', default='harmful', type=str, help="Name for first dataset label")
    parser.add_argument('--label_b', default='harmless', type=str, help="Name for second dataset label")
    parser.add_argument('--auto_standard', default=0, type=int, help="If 1, run both hf and refuse with standardized paths and outputs")
    parser.add_argument('--num_context_tokens', default=2, type=int, help="Number of context tokens to keep before extracted positions")
    parser.add_argument('--output-base-dir', default='out_pt', type=str, dest='output_base_dir', help="Base output directory used with --auto_standard")
    
    args = parser.parse_args()
    params = vars(args)
    
    global MODEL
    global NUM_TOKEN_HIDDEN
    
    params['positions'] = list(map(int, params['positions'].split()))
    spec = resolve_model_spec(model=params['model'], model_id=params.get('model_id'))
    params['model'] = spec.model_name
    params['model_id'] = spec.model_id
    MODEL = spec.model_name
    model, tokenizer = load_model_and_tokenizer(MODEL, spec.model_id)

    if params['auto_standard']:
        for mode_dir in ('hf', 'refuse'):
            run_params = copy.deepcopy(params)
            run_params.update(build_standard_run_params(MODEL, spec.model_id, mode_dir, output_base_dir=params['output_base_dir']))
            maybe_set_instruction_positions(run_params, tokenizer)
            print(f"Running standard extraction for {MODEL} / {mode_dir}")
            run_extraction(run_params, model, tokenizer)
    else:
        maybe_set_instruction_positions(params, tokenizer)
        run_extraction(params, model, tokenizer)

if __name__ == "__main__":
    main()
