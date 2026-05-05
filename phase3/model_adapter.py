import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_configs import get_model_config, get_module_names


def get_decoder_layers(model, arch):
    if arch == "opt":
        return model.model.decoder.layers
    elif arch == "llama":
        return model.model.layers
    else:
        raise ValueError(f"Unknown arch: '{arch}'")


def get_model_device(model, arch):
    if arch == "opt":
        return model.model.decoder.embed_tokens.weight.device
    elif arch == "llama":
        return model.model.embed_tokens.weight.device
    else:
        raise ValueError(f"Unknown arch: '{arch}'")


def _resolve_parent(layer, module_name):
    # "self_attn.q_proj" → (layer.self_attn, "q_proj")
    # "fc1"              → (layer, "fc1")
    # "mlp.gate_proj"    → (layer.mlp, "gate_proj")
    parts = module_name.split(".")
    parent = layer
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_named_linear(layer, module_name):
    parent, attr = _resolve_parent(layer, module_name)
    return getattr(parent, attr)


def set_named_linear(layer, module_name, new_module):
    parent, attr = _resolve_parent(layer, module_name)
    setattr(parent, attr, new_module)


def parse_module_names(s, arch):
    valid = set(get_module_names(arch))
    names = [x.strip() for x in s.split(",") if x.strip()]
    for n in names:
        if n not in valid:
            raise ValueError(
                f"Invalid module name '{n}' for arch '{arch}'. "
                f"Valid: {sorted(valid)}"
            )
    return names


def resolve_target_layers(model, arch, replace_scope, one_layer_idx, custom_layer_indices=""):
    layers = get_decoder_layers(model, arch)
    num_layers = len(layers)

    if replace_scope == "one":
        if one_layer_idx < 0 or one_layer_idx >= num_layers:
            raise ValueError(f"one_layer_idx must be in [0, {num_layers - 1}]")
        return [one_layer_idx]
    elif replace_scope == "all":
        return list(range(num_layers))
    elif replace_scope == "custom":
        if not custom_layer_indices:
            raise ValueError("--custom_layer_indices 0,2,4 형식으로 입력해주세요.")
        return [int(i.strip()) for i in custom_layer_indices.split(",")]
    else:
        raise ValueError(f"Unsupported replace_scope: '{replace_scope}'")


def load_model_and_tokenizer(model_name, model_source="auto"):
    cfg = get_model_config(model_name)
    arch      = cfg["arch"]
    omni_ckpt = cfg["omni_ckpt"]
    hf_id     = cfg["hf_id"]

    if model_source not in ("auto", "fp16", "omni"):
        raise ValueError(f"Unknown model_source: '{model_source}'")

    if model_source == "auto":
        load_path = omni_ckpt if os.path.isdir(omni_ckpt) else hf_id
    elif model_source == "fp16":
        load_path = hf_id
    else:
        if not os.path.isdir(omni_ckpt):
            raise FileNotFoundError(f"OmniQuant checkpoint가 없습니다: {omni_ckpt}")
        load_path = omni_ckpt

    # model_source=fp16이면 원본 tokenizer/model을, omni이면 OmniQuant checkpoint를 강제 사용한다.
    tok_path = load_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    seqlen = model.config.max_position_embeddings

    # load_path: run_qr_ppl.py에서 weight_backend 판별 및 로깅에 사용
    return model, tokenizer, seqlen, arch, load_path
