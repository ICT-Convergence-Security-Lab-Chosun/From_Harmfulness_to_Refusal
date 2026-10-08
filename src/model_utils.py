from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_HARMFUL_DATA = REPO_ROOT / "data" / "jbb-harmful.json"
DEFAULT_HARMLESS_DATA = REPO_ROOT / "data" / "alpaca_common.json"

_ALIAS_TO_MODEL_ID = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31_70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "gemma2": "google/gemma-2-9b-it",
    "granite": "ibm-granite/granite-3.1-8b-instruct",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
    "yi": "01-ai/Yi-1.5-9B-Chat",
    "yi15-6b": "01-ai/Yi-1.5-6B-Chat",
    "aya23": "CohereLabs/aya-23-8B",
    "aya23-8b": "CohereLabs/aya-23-8B",
    "aya23_8b": "CohereLabs/aya-23-8B",
    "internlm25": "internlm/internlm2_5-7b-chat",
    "internlm25-7b": "internlm/internlm2_5-7b-chat",
    "internlm25_7b": "internlm/internlm2_5-7b-chat",
    "internlm2_5": "internlm/internlm2_5-7b-chat",
    "internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
    "olmo2": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo2-7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_id: str
    output_name: str
    trust_remote_code: bool


def normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if model_name is None:
        return None
    lower_name = model_name.lower().strip()
    if lower_name in {"llama", "llama2", "llama-2", "llama-2-7b", "llama-2-7b-chat"}:
        return "llama2"
    if lower_name in {"llama3", "llama-3", "llama31", "llama3.1", "llama-3.1", "llama-3.1-8b"}:
        return "llama31"
    if lower_name in {"llama31_70b", "llama3.1-70b", "llama-3.1-70b", "llama-3.1-70b-instruct", "llama3.1-70b-instruct"}:
        return "llama31_70b"
    if lower_name in {
        "gemma",
        "gemma2",
        "gemma-2",
        "gemma2-9b",
        "gemma-2-9b",
        "gemma2-9b-it",
        "gemma-2-9b-it",
        "gemma3",
        "gemma-3",
        "gemma3-12b",
        "gemma-3-12b",
        "gemma3-12b-it",
        "gemma-3-12b-it",
    }:
        return "gemma"
    if lower_name in {"glm", "chatglm", "glm4", "glm-4", "glm-4-9b-chat", "chatglm-4"}:
        return "glm"
    if lower_name in {"granite", "granite3", "granite-3", "granite3.1", "granite-3.1", "granite-3.1-8b", "granite-3.1-8b-instruct"}:
        return "granite"
    if lower_name in {"falcon", "falcon3", "falcon-3", "falcon3-7b", "falcon3-7b-instruct", "falcon-3-7b-instruct"}:
        return "falcon3"
    if lower_name in {"yi", "yi-1.5", "yi-1.5-6b-chat", "yi-1.5-9b-chat", "yi-1.5-34b-chat"}:
        return "yi"
    if lower_name in {"aya", "aya23", "aya-23", "aya23-8b", "aya23_8b", "aya-23-8b"}:
        return "aya23"
    if lower_name in {
        "internlm",
        "internlm2",
        "internlm2.5",
        "internlm2_5",
        "internlm25",
        "internlm25-7b",
        "internlm25_7b",
        "internlm2_5-7b",
        "internlm2_5-7b-chat",
    }:
        return "internlm25"
    if lower_name in {
        "olmo",
        "olmo2",
        "olmo-2",
        "olmo2-7b",
        "olmo2_7b",
        "olmo-2-7b",
        "olmo-2-1124-7b",
        "olmo-2-1124-7b-instruct",
    }:
        return "olmo2"
    if lower_name in {"qwen", "qwen25", "qwen2", "qwen2-7b", "qwen2-7b-instruct", "qwen2.5", "qwen2.5-7b", "qwen2.5-7b-instruct", "qwen2.5-8b", "qwen2.5-8b-instruct"}:
        return "qwen25"
    if lower_name in {"qwen14b", "qwen25_14b", "qwen2.5-14b", "qwen2.5-14b-instruct"}:
        return "qwen14b"
    if lower_name in {"qwen32b", "qwen25_32b", "qwen2.5-32b", "qwen2.5-32b-instruct"}:
        return "qwen32b"
    if lower_name in {"qwen72b", "qwen25_72b", "qwen2.5-72b", "qwen2.5-72b-instruct"}:
        return "qwen72b"
    return lower_name


def sanitize_model_id(model_id: str) -> str:
    name = model_id.split("/")[-1].strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip("-")


def infer_model_name_from_id(model_id: str) -> str:
    lowered = model_id.lower()
    if "llama-2" in lowered or "llama2" in lowered:
        return "llama2"
    if "llama-3" in lowered or "llama3" in lowered:
        if "70b" in lowered:
            return "llama31_70b"
        return "llama31"
    if "gemma-2" in lowered or "gemma2" in lowered or "gemma-3" in lowered or "gemma3" in lowered:
        return "gemma"
    if "chatglm" in lowered or "/glm" in lowered or "glm-4" in lowered:
        return "glm"
    if "granite" in lowered:
        return "granite"
    if "falcon3" in lowered or "falcon-3" in lowered:
        return "falcon3"
    if "/yi" in lowered or lowered.startswith("yi-") or "yi-1.5" in lowered:
        return "yi"
    if "aya-23" in lowered or "aya23" in lowered:
        return "aya23"
    if "internlm2_5" in lowered or "internlm2.5" in lowered or "internlm25" in lowered:
        return "internlm25"
    if "olmo-2" in lowered or "olmo2" in lowered:
        return "olmo2"
    if "qwen" in lowered:
        if "72b" in lowered:
            return "qwen72b"
        if "32b" in lowered:
            return "qwen32b"
        if "14b" in lowered:
            return "qwen14b"
        return "qwen25"
    return sanitize_model_id(model_id)


def requires_trust_remote_code(model_name: str, model_id: str) -> bool:
    lowered_model_name = model_name.lower()
    lowered_model_id = model_id.lower()
    return (
        lowered_model_name in {"qwen", "qwen25", "qwen14b", "qwen32b", "qwen72b", "yi", "glm", "internlm25"}
        or "qwen" in lowered_model_id
        or "yi-" in lowered_model_id
        or "chatglm" in lowered_model_id
        or "glm-4" in lowered_model_id
        or "internlm" in lowered_model_id
    )


def resolve_model_spec(model: Optional[str] = None, model_id: Optional[str] = None) -> ModelSpec:
    normalized_model = normalize_model_name(model)
    if model_id:
        resolved_model_id = model_id
        if normalized_model is None:
            normalized_model = infer_model_name_from_id(model_id)
    else:
        if normalized_model is None:
            normalized_model = "qwen25"
        if normalized_model not in _ALIAS_TO_MODEL_ID:
            raise ValueError(f"Unsupported model alias: {model}")
        resolved_model_id = _ALIAS_TO_MODEL_ID[normalized_model]

    if normalized_model is None:
        normalized_model = infer_model_name_from_id(resolved_model_id)

    return ModelSpec(
        model_name=normalized_model,
        model_id=resolved_model_id,
        output_name=sanitize_model_id(resolved_model_id),
        trust_remote_code=requires_trust_remote_code(normalized_model, resolved_model_id),
    )


def default_alpha_for_model(model: Optional[str] = None, model_id: Optional[str] = None) -> float:
    return 3.0


def get_output_root(base_dir: str | Path, model: Optional[str] = None, model_id: Optional[str] = None) -> Path:
    spec = resolve_model_spec(model=model, model_id=model_id)
    return Path(base_dir) / spec.output_name


def get_default_binary_dataset_paths() -> tuple[Path, Path]:
    return DEFAULT_HARMFUL_DATA, DEFAULT_HARMLESS_DATA


def get_template_parts(
    model: Optional[str] = None,
    model_id: Optional[str] = None,
    use_template: bool = True,
    do_not_use_last_inst_tok: bool = False,
) -> tuple[str, str]:
    if not use_template:
        return "", ""

    spec = resolve_model_spec(model=model, model_id=model_id)
    if spec.model_name in {"llama3", "llama31", "llama31_70b"}:
        prefix = "<|start_header_id|>user<|end_header_id|>\n"
        suffix = "" if do_not_use_last_inst_tok else "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        return prefix, suffix
    if spec.model_name in {"qwen", "qwen25", "qwen14b", "qwen32b", "qwen72b"}:
        prefix = "<|im_start|>user\n"
        suffix = "" if do_not_use_last_inst_tok else "<|im_end|>\n<|im_start|>assistant"
        return prefix, suffix
    if spec.model_name == "yi":
        prefix = "<|im_start|>user\n"
        suffix = "" if do_not_use_last_inst_tok else "<|im_end|>\n<|im_start|>assistant\n"
        return prefix, suffix
    if spec.model_name in {"gemma", "gemma2"}:
        prefix = "<start_of_turn>user\n"
        suffix = "" if do_not_use_last_inst_tok else "<end_of_turn>\n<start_of_turn>model\n"
        return prefix, suffix
    if spec.model_name == "granite":
        prefix = "<|start_of_role|>user<|end_of_role|>"
        suffix = "" if do_not_use_last_inst_tok else "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        return prefix, suffix
    if spec.model_name == "falcon3":
        prefix = "<|user|>\n"
        suffix = "" if do_not_use_last_inst_tok else "\n<|assistant|>\n"
        return prefix, suffix
    if spec.model_name == "aya23":
        prefix = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        suffix = "" if do_not_use_last_inst_tok else "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        return prefix, suffix
    if spec.model_name == "internlm25":
        prefix = "<|im_start|>user\n"
        suffix = "" if do_not_use_last_inst_tok else "<|im_end|>\n<|im_start|>assistant\n"
        return prefix, suffix
    if spec.model_name == "olmo2":
        prefix = "<|endoftext|><|user|>\n"
        suffix = "" if do_not_use_last_inst_tok else "\n<|assistant|>\n"
        return prefix, suffix
    prefix = "[INST] "
    suffix = "" if do_not_use_last_inst_tok else " [/INST]"
    return prefix, suffix


def get_assistant_suffix_text(
    model: Optional[str] = None,
    model_id: Optional[str] = None,
    use_template: bool = True,
    do_not_use_last_inst_tok: bool = False,
) -> str:
    _, suffix = get_template_parts(
        model=model,
        model_id=model_id,
        use_template=use_template,
        do_not_use_last_inst_tok=do_not_use_last_inst_tok,
    )
    return suffix


def get_assistant_suffix_auto(
    tokenizer: "AutoTokenizer",
    *,
    model: Optional[str] = None,
    model_id: Optional[str] = None,
) -> tuple[str, list[int]]:
    dummy = [{"role": "user", "content": "X"}]
    suffix_text = ""
    try:
        with_prompt = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
        without_prompt = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
        suffix_text = with_prompt[len(without_prompt):]
    except Exception:
        suffix_text = ""
    if not suffix_text:
        suffix_text = get_assistant_suffix_text(
            model=model,
            model_id=model_id,
            use_template=True,
            do_not_use_last_inst_tok=False,
        )
    if not suffix_text:
        raise ValueError("Could not determine assistant suffix from tokenizer chat template or fallback template")
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    if not suffix_ids:
        raise ValueError("Assistant suffix tokenization is empty")
    return suffix_text, suffix_ids


def resolve_positions_from_mask(
    seq_len: int,
    positions: list[int],
    attention_mask_row: torch.Tensor | None = None,
) -> list[int]:
    effective_len = seq_len
    start_idx = 0
    if attention_mask_row is not None:
        attention_mask_row = attention_mask_row.to(dtype=torch.int64, device="cpu")
        nonzero = torch.nonzero(attention_mask_row, as_tuple=False).flatten()
        effective_len = int(nonzero.numel())
        if effective_len <= 0:
            raise ValueError("Attention mask indicates an empty sequence.")
        start_idx = int(nonzero[0].item())

    resolved = []
    for pos in positions:
        idx = pos if pos >= 0 else effective_len + pos
        if idx < 0 or idx >= effective_len:
            raise IndexError(
                f"Resolved position {pos} is out of bounds for effective sequence length {effective_len}"
            )
        resolved.append(start_idx + idx)
    return resolved


def find_suffix_start(
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    suffix_ids: list[int],
) -> int:
    seq_len = input_ids_row.shape[0]
    attention_mask_row = attention_mask_row.to(dtype=torch.int64, device="cpu")
    nonzero = torch.nonzero(attention_mask_row, as_tuple=False).flatten()
    effective_len = int(nonzero.numel())
    if effective_len <= 0:
        raise ValueError("Attention mask indicates an empty sequence.")
    start_idx = int(nonzero[0].item())
    end_idx = int(nonzero[-1].item()) + 1
    seq = input_ids_row[start_idx:end_idx].tolist()
    if len(seq) < len(suffix_ids):
        raise ValueError(f"Sequence too short for suffix: seq_len={seq_len}, suffix_len={len(suffix_ids)}")
    tail = seq[-len(suffix_ids):]
    if tail == suffix_ids:
        return effective_len - len(suffix_ids)

    for start in range(effective_len - len(suffix_ids), -1, -1):
        if seq[start:start + len(suffix_ids)] == suffix_ids:
            return start
    raise ValueError("Failed to find assistant suffix in tokenized prompt.")


def collect_prompt_boundary_positions(
    tokenizer: "AutoTokenizer",
    prompts: list[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    model: Optional[str] = None,
    model_id: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    suffix_text, suffix_ids = get_assistant_suffix_auto(
        tokenizer,
        model=model,
        model_id=model_id,
    )
    harmful_positions = []
    refusal_positions = []
    seq_len = input_ids.shape[1]
    for batch_idx in range(input_ids.shape[0]):
        mask_row = attention_mask[batch_idx]
        try:
            suffix_start = find_suffix_start(input_ids[batch_idx], mask_row, suffix_ids)
        except ValueError:
            prompt = prompts[batch_idx]
            suffix_char_idx = prompt.rfind(suffix_text)
            if suffix_char_idx < 0:
                raise
            prefix_text = prompt[:suffix_char_idx]
            prefix_ids = tokenizer(prefix_text, add_special_tokens=True, return_tensors="pt").input_ids[0]
            suffix_start = int(prefix_ids.shape[0])
        harmful_positions.append(
            resolve_positions_from_mask(seq_len, [suffix_start - 1], mask_row)[0]
        )
        refusal_positions.append(
            resolve_positions_from_mask(seq_len, [-1], mask_row)[0]
        )
    return (
        torch.tensor(harmful_positions, dtype=torch.long, device=input_ids.device),
        torch.tensor(refusal_positions, dtype=torch.long, device=input_ids.device),
    )


def load_model_and_tokenizer(
    model: Optional[str] = None,
    model_id: Optional[str] = None,
    *,
    device_map: str = "auto",
    trust_remote_code: Optional[bool] = None,
) -> tuple["AutoModelForCausalLM", "AutoTokenizer", ModelSpec]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = resolve_model_spec(model=model, model_id=model_id)
    resolved_trust_remote_code = spec.trust_remote_code if trust_remote_code is None else trust_remote_code

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=resolved_trust_remote_code)
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": resolved_trust_remote_code,
    }
    if spec.model_name == "llama2":
        model_kwargs["torch_dtype"] = torch.float16
    elif spec.model_name in {"gemma", "yi"}:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model_obj = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if getattr(model_obj.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model_obj.config.pad_token_id = tokenizer.pad_token_id

    return model_obj, tokenizer, spec


def get_transformer_layers(model_obj) -> list[torch.nn.Module]:
    candidates = [
        getattr(getattr(model_obj, "model", None), "layers", None),
        getattr(getattr(getattr(model_obj, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model_obj, "language_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise AttributeError(
        f"Could not find transformer layers on model type {type(model_obj).__name__}. "
        "Tried model.layers, model.language_model.layers, and language_model.layers."
    )
