### 이 파일은 run_grouped_ppl_test.py 기반의 코드를 가져와서 quant를 추가해 ppl 평가하는 코드
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# 0. Utility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_main_device(model):
    return model.model.decoder.embed_tokens.weight.device


def save_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for line in lines:
                f.write(str(line) + "\n")


def resolve_axis(dim: int, axis: int) -> int:
    if axis < 0:
        axis = dim + axis
    if axis < 0 or axis >= dim:
        raise ValueError(f"Invalid axis={axis} for tensor dim={dim}")
    return axis


# =========================================================
# 1. Dataset / PPL Eval Input Loader
# =========================================================
def load_wikitext2_testenc(model_id, split="test"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids

    return tokenizer, testenc


def get_probe_text():
    return "A transformer layer takes an input hidden state, projects it, mixes information, and returns a new hidden state."


# =========================================================
# 2. Fake Quantization
# =========================================================
def fake_quant_symmetric(
    x: torch.Tensor,
    n_bits: int = 8,
    mode: str = "tensor",
    ch_axis: int = -1,
    group_size: int = None,
    eps: float = 1e-8,
):
    """
    mode:
        - tensor
        - per_channel
        - per_token : 마지막 dim(H)을 한 벡터로 보고, 각 [B,S,:] 또는 [B,:]마다 scale 1개
        - group      : 마지막 dim 기준 contiguous group-wise quantization
    """
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")

    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1

    x_fp = x.float()

    if mode == "tensor":
        max_abs = x_fp.abs().max()
        scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(x_fp / scale).clamp(qmin, qmax)
        return (q * scale).to(x.dtype)

    elif mode == "per_channel":
        axis = resolve_axis(x_fp.dim(), ch_axis)
        reduce_dims = tuple(d for d in range(x_fp.dim()) if d != axis)
        max_abs = x_fp.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(x_fp / scale).clamp(qmin, qmax)
        return (q * scale).to(x.dtype)

    elif mode == "per_token":
        # activation x가 [B,S,H] 또는 [B,H]일 때
        # 마지막 dim(H)에 대해, 각 token/vector마다 scale 1개
        if x_fp.dim() == 1:
            max_abs = x_fp.abs().max()
            scale = (max_abs / qmax).clamp_min(eps)
        else:
            max_abs = x_fp.abs().amax(dim=-1, keepdim=True)
            scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(x_fp / scale).clamp(qmin, qmax)
        return (q * scale).to(x.dtype)

    elif mode == "group":
        if group_size is None:
            raise ValueError("group_size must be provided when mode='group'")

        if x_fp.shape[-1] % group_size != 0:
            raise ValueError(
                f"Last dim ({x_fp.shape[-1]}) must be divisible by group_size ({group_size})"
            )

        num_groups = x_fp.shape[-1] // group_size
        new_shape = x_fp.shape[:-1] + (num_groups, group_size)
        xg = x_fp.reshape(new_shape)

        max_abs = xg.abs().amax(dim=-1, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(xg / scale).clamp(qmin, qmax)
        dq = q * scale

        return dq.reshape_as(x_fp).to(x.dtype)

    else:
        raise ValueError(f"Unsupported quant mode: {mode}")
# =========================================================
# 2.1 Fake Quantization + HW like -> dequant after mac
#    - activation quant
#    - weight quant
# =========================================================
def quantize_symmetric_int(
    x: torch.Tensor,
    n_bits: int = 8,
    mode: str = "tensor",
    ch_axis: int = -1,
    group_size: int = None,
    eps: float = 1e-8,
):
    """
    반환:
      q_int  : int32 quantized tensor
      scale  : float tensor
    지원:
      activation hw_like -> tensor / per_token / group
      weight     hw_like -> tensor / per_channel / group
    """
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")

    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    q_dtype = torch.int8 if n_bits <= 8 else torch.int16
    x_fp = x.float()

    if mode == "tensor":
        max_abs = x_fp.abs().max()
        scale = (max_abs / qmax).clamp_min(eps)
        q = torch.round(x_fp / scale).clamp(qmin, qmax).to(q_dtype)
        return q, scale

    elif mode == "per_channel":
        axis = resolve_axis(x_fp.dim(), ch_axis)
        reduce_dims = tuple(d for d in range(x_fp.dim()) if d != axis)
        max_abs = x_fp.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)
        q = torch.round(x_fp / scale).clamp(qmin, qmax).to(q_dtype)
        return q, scale

    elif mode == "per_token":
        if x_fp.dim() == 1:
            max_abs = x_fp.abs().max()
            scale = (max_abs / qmax).clamp_min(eps)
        else:
            max_abs = x_fp.abs().amax(dim=-1, keepdim=True)
            scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(x_fp / scale).clamp(qmin, qmax).to(q_dtype)
        return q, scale

    elif mode == "group":
        if group_size is None:
            raise ValueError("group_size must be provided when mode='group'")

        if x_fp.shape[-1] % group_size != 0:
            raise ValueError(
                f"Last dim ({x_fp.shape[-1]}) must be divisible by group_size ({group_size})"
            )

        num_groups = x_fp.shape[-1] // group_size
        new_shape = x_fp.shape[:-1] + (num_groups, group_size)
        xg = x_fp.reshape(new_shape)

        max_abs = xg.abs().amax(dim=-1, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)

        q = torch.round(xg / scale).clamp(qmin, qmax).to(q_dtype)
        return q.reshape_as(x_fp), scale  # scale shape: [..., num_groups, 1]

    else:
        raise ValueError(f"Unsupported quant mode: {mode}")

def quantize_weight_symmetric_int_chunked(
    w: torch.Tensor,
    n_bits: int = 8,
    mode: str = "tensor",
    ch_axis: int = 0,
    group_size: int = None,
    row_chunk: int = 128,
    cache_device: str = "cpu",   # "cpu" or "cuda"
):
    """
    weight 전용:
    - CPU에서 chunk 단위로 quant
    - q_int는 int8/int16 저장
    - scale은 fp16/float32 저장
    """

    if w.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape={tuple(w.shape)}")

    # CPU에서 처리해서 GPU peak memory를 줄임
    w_cpu = w.detach().to("cpu", dtype=torch.float32).contiguous()

    q_chunks = []
    s_chunks = []

    scale_dtype = torch.float16 if w.dtype in (torch.float16, torch.bfloat16) else torch.float32

    for start in range(0, w_cpu.size(0), row_chunk):
        wc = w_cpu[start:start + row_chunk].contiguous()

        q_c, s_c = quantize_symmetric_int(
            wc,
            n_bits=n_bits,
            mode=mode,
            ch_axis=ch_axis,
            group_size=group_size,
        )

        q_chunks.append(q_c.contiguous())
        s_chunks.append(s_c.to(scale_dtype).contiguous())

    q = torch.cat(q_chunks, dim=0)
    s = torch.cat(s_chunks, dim=0)

    if cache_device == "cuda":
        q = q.to(w.device, non_blocking=True)
        s = s.to(w.device, dtype=w.dtype, non_blocking=True)

    return q, s
# =========================================================
# 3. Experiment Linear
#    - exact grouping
#    - weight quant
#    - activation quant
# =========================================================
class ExperimentLinear(nn.Module):
    """
    지원:
    1) exact grouped linear (기존)
    2) fake quant path (기존)
    3) hw_like path:
       quant -> int MAC emulation -> post-MAC dequant
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        enable_grouping: bool = False,
        linear_group_size: int = 128,
        enable_weight_quant: bool = False,
        weight_bits: int = 8,
        weight_quant_mode: str = "tensor",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        enable_act_quant: bool = False,
        act_bits: int = 8,
        act_quant_mode: str = "tensor",
        act_ch_axis: int = -1,
        act_group_size: int = 128,
        quant_impl: str = "fake",   # "fake" or "hw_like"
        weight_cache_device: str = "cpu",
        weight_quant_chunk_rows: int = 128,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.enable_grouping = enable_grouping
        self.linear_group_size = linear_group_size

        self.enable_weight_quant = enable_weight_quant
        self.weight_bits = weight_bits
        self.weight_quant_mode = weight_quant_mode
        self.weight_ch_axis = weight_ch_axis
        self.weight_group_size = weight_group_size
        self.weight_cache_device = weight_cache_device
        self.weight_quant_chunk_rows = weight_quant_chunk_rows
        self.enable_act_quant = enable_act_quant
        self.act_bits = act_bits
        self.act_quant_mode = act_quant_mode
        self.act_ch_axis = act_ch_axis
        self.act_group_size = act_group_size

        self.quant_impl = quant_impl

        if self.enable_grouping and (self.in_features % self.linear_group_size != 0):
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by linear_group_size ({self.linear_group_size})"
            )

        weight_fp = base_linear.weight.detach()

        if self.quant_impl == "fake":
            if self.enable_weight_quant:
                w_q = fake_quant_symmetric(
                    weight_fp,
                    n_bits=self.weight_bits,
                    mode=self.weight_quant_mode,
                    ch_axis=self.weight_ch_axis,
                    group_size=self.weight_group_size,
                )
                self.weight = nn.Parameter(w_q, requires_grad=False)
            else:
                self.weight = nn.Parameter(weight_fp.clone(), requires_grad=False)
                
            self.register_buffer("weight_q_int_cache", torch.empty(0, dtype=torch.int8), persistent=False)
            self.register_buffer("weight_scale_cache", torch.empty(0), persistent=False)

        elif self.quant_impl == "hw_like":
            if self.enable_weight_quant:
                q_int, scale = quantize_weight_symmetric_int_chunked(
                    weight_fp,
                    n_bits=self.weight_bits,
                    mode=self.weight_quant_mode,
                    ch_axis=self.weight_ch_axis,
                    group_size=self.weight_group_size,
                    row_chunk=self.weight_quant_chunk_rows,
                    cache_device=self.weight_cache_device,
                )
                self.register_buffer("weight_q_int_cache", q_int.detach(), persistent=False)
                self.register_buffer("weight_scale_cache", scale.detach(), persistent=False)
            else:
                self.register_buffer("weight_q_int_cache", torch.empty(0, device="cpu", dtype=torch.int8), persistent=False)
                self.register_buffer("weight_scale_cache", torch.empty(0, device="cpu", dtype=weight_fp.dtype), persistent=False)
            
            # hw-like 모드에서는 정수 가중치만 있으면 되므로 float 가중치를 버려서 메모리를 폭발적으로 줄임 (OOM 방지)
            self.weight = nn.Parameter(torch.empty(0, device=weight_fp.device, dtype=weight_fp.dtype), requires_grad=False)

        else:
            raise ValueError(f"Unsupported quant_impl: {self.quant_impl}")

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    # -----------------------------------------------------
    # fake path
    # -----------------------------------------------------
    def _apply_activation_quant_fake(self, x):
        if not self.enable_act_quant:
            return x

        x_q = fake_quant_symmetric(
            x,
            n_bits=self.act_bits,
            mode=self.act_quant_mode,
            ch_axis=self.act_ch_axis,
            group_size=self.act_group_size,
        )
        return x_q

    def _get_weight_for_forward_fake(self):
        return self.weight

    def _forward_grouped_fake(self, x, w):
        x_groups = torch.split(x, self.linear_group_size, dim=-1)
        w_groups = torch.split(w, self.linear_group_size, dim=1)

        out = None
        for xg, wg in zip(x_groups, w_groups):
            part = F.linear(xg, wg, bias=None)
            out = part if out is None else out + part

        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_normal_fake(self, x, w):
        return F.linear(x, w, self.bias)

    # -----------------------------------------------------
    # hw-like helpers -> Exception CHECK
    # -----------------------------------------------------
    def _check_hw_like_config(self):
        if self.enable_grouping:
            raise ValueError(
                "quant_impl='hw_like'에서는 일단 --enable_grouping을 같이 쓰지 마세요."
            )

        if not (self.enable_act_quant and self.enable_weight_quant):
            raise ValueError(
                "quant_impl='hw_like' 최소 버전은 현재 activation/weight quant 둘 다 켠 경우만 지원합니다."
            )

        if self.act_quant_mode == "per_channel":
            raise ValueError(
                "activation per_channel은 post-MAC dequant로 factor-out이 안 되어 hw_like 최소 버전에서 지원하지 않습니다."
            )

        if self.act_quant_mode not in {"tensor", "per_token", "group"}:
            raise ValueError(f"Unsupported act_quant_mode for hw_like: {self.act_quant_mode}")

        if self.weight_quant_mode not in {"tensor", "per_channel", "group"}:
            raise ValueError(f"Unsupported weight_quant_mode for hw_like: {self.weight_quant_mode}")

        if self.act_quant_mode == "group" and self.weight_quant_mode == "group":
            if self.act_group_size != self.weight_group_size:
                raise ValueError(
                    f"hw_like에서 act_group_size({self.act_group_size})와 "
                    f"weight_group_size({self.weight_group_size})는 같아야 합니다."
                )

    def _quantize_activation_hw(self, x):
        q_int, scale = quantize_symmetric_int(
            x,
            n_bits=self.act_bits,
            mode=self.act_quant_mode,
            ch_axis=self.act_ch_axis,
            group_size=self.act_group_size,
        )
        return q_int, scale

    def _get_hw_mac_group_size(self):
        group_sizes = []
        if self.act_quant_mode == "group":
            group_sizes.append(self.act_group_size)
        if self.weight_quant_mode == "group":
            group_sizes.append(self.weight_group_size)

        if len(group_sizes) == 0:
            return self.in_features

        if len(set(group_sizes)) != 1:
            raise ValueError(f"hw_like MAC group sizes mismatch: {group_sizes}")

        return group_sizes[0]

    def _get_act_scale_for_group(self, act_scale, g_idx):
        if self.act_quant_mode == "tensor":
            return act_scale
        elif self.act_quant_mode == "per_token":
            return act_scale                 # [..., 1]
        elif self.act_quant_mode == "group":
            return act_scale[..., g_idx, :] # [..., 1]
        else:
            raise ValueError(f"Unsupported act_quant_mode for hw_like: {self.act_quant_mode}")

    def _get_weight_scale_for_group(self, weight_scale, g_idx, acc_dim):
        if self.weight_quant_mode == "tensor":
            return weight_scale
        elif self.weight_quant_mode == "per_channel":
            # shape [out_features, 1] -> broadcast to [..., out_features]
            return weight_scale[:, 0].view(*([1] * (acc_dim - 1)), -1)
        elif self.weight_quant_mode == "group":
            # shape [out_features, num_groups, 1]
            return weight_scale[:, g_idx, 0].view(*([1] * (acc_dim - 1)), -1)
        else:
            raise ValueError(f"Unsupported weight_quant_mode for hw_like: {self.weight_quant_mode}")

    def _forward_hw_like(self, x):
        self._check_hw_like_config()

        orig_dtype = x.dtype

        # activation -> int + scale
        qx_int, act_scale = self._quantize_activation_hw(x)

        # weight cache: int + scale
        qw_int = self.weight_q_int_cache
        weight_scale = self.weight_scale_cache
        # cache가 CPU에 있으면 현재 activation device로 이동
        if qw_int.device != x.device:
            qw_int = qw_int.to(x.device, non_blocking=True)

        if weight_scale.device != x.device:
            weight_scale = weight_scale.to(
                x.device,
                dtype=torch.float16 if x.dtype in (torch.float16, torch.bfloat16) else torch.float32,
                non_blocking=True,
            )
        mac_group_size = self._get_hw_mac_group_size()

        x_groups = torch.split(qx_int, mac_group_size, dim=-1)
        w_groups = torch.split(qw_int, mac_group_size, dim=1)

        out = None
        for g_idx, (qxg, qwg) in enumerate(zip(x_groups, w_groups)):
            # int MAC emulation
            # FP32는 이 범위의 작은 int 누산을 정확히 표현 가능
            acc_int = F.linear(qxg.float(), qwg.float(), bias=None)

            act_scale_g = self._get_act_scale_for_group(act_scale, g_idx)
            weight_scale_g = self._get_weight_scale_for_group(weight_scale, g_idx, acc_int.dim())

            part = acc_int * act_scale_g * weight_scale_g
            out = part if out is None else out + part

        if self.bias is not None:
            out = out + self.bias.float()

        return out.to(orig_dtype)

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(self, x):
        if self.quant_impl == "fake":
            x_q = self._apply_activation_quant_fake(x)
            w_q = self._get_weight_for_forward_fake()

            if self.enable_grouping:
                return self._forward_grouped_fake(x_q, w_q)
            else:
                return self._forward_normal_fake(x_q, w_q)

        elif self.quant_impl == "hw_like":
            return self._forward_hw_like(x)

        else:
            raise ValueError(f"Unsupported quant_impl: {self.quant_impl}")


# =========================================================
# 4. Module Access / Replacement
# =========================================================
def get_named_linear_module(layer, module_name: str):
    mapping = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1": layer.fc1,
        "fc2": layer.fc2,
    }
    if module_name not in mapping:
        raise ValueError(f"Unsupported module_name: {module_name}")
    return mapping[module_name]


def set_named_linear_module(layer, module_name: str, new_module: nn.Module):
    if module_name == "self_attn.q_proj":
        layer.self_attn.q_proj = new_module
    elif module_name == "self_attn.k_proj":
        layer.self_attn.k_proj = new_module
    elif module_name == "self_attn.v_proj":
        layer.self_attn.v_proj = new_module
    elif module_name == "self_attn.out_proj":
        layer.self_attn.out_proj = new_module
    elif module_name == "fc1":
        layer.fc1 = new_module
    elif module_name == "fc2":
        layer.fc2 = new_module
    else:
        raise ValueError(f"Unsupported module_name: {module_name}")


def parse_module_names(s: str):
    names = [x.strip() for x in s.split(",") if len(x.strip()) > 0]
    valid = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    }
    for n in names:
        if n not in valid:
            raise ValueError(f"Invalid module name: {n}")
    return names


def resolve_target_layers(model, replace_scope: str, one_layer_idx: int):
    num_layers = len(model.model.decoder.layers)

    if replace_scope == "one":
        if one_layer_idx < 0 or one_layer_idx >= num_layers:
            raise ValueError(f"one_layer_idx must be in [0, {num_layers-1}]")
        return [one_layer_idx]
    elif replace_scope == "all":
        return list(range(num_layers))
    else:
        raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules_with_experiment_linear(
    model,
    layer_indices,
    module_names,
    enable_grouping,
    linear_group_size,
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    enable_act_quant,
    act_bits,
    act_quant_mode,
    act_ch_axis,
    act_group_size,
    quant_impl, # "fake" or "hw_like"
    weight_cache_device,
    weight_quant_chunk_rows,
):
    replaced_names = []

    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)

            new_module = ExperimentLinear(
                base_linear=old_module,
                enable_grouping=enable_grouping,
                linear_group_size=linear_group_size,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                enable_act_quant=enable_act_quant,
                act_bits=act_bits,
                act_quant_mode=act_quant_mode,
                act_ch_axis=act_ch_axis,
                act_group_size=act_group_size,
                quant_impl=quant_impl, # "fake" or "hw_like"
                weight_cache_device=weight_cache_device,
                weight_quant_chunk_rows=weight_quant_chunk_rows,
            )

            set_named_linear_module(layer, module_name, new_module)
            replaced_names.append(f"layer{layer_idx}.{module_name}")

    return replaced_names


# =========================================================
# 5. Probe 비교용 Hook
# =========================================================
def collect_module_outputs(model, tokenizer, text, target_layer_indices, target_module_names):
    outputs_dict = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            outputs_dict[name] = x.detach().float().cpu()
        return hook

    for layer_idx in target_layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in target_module_names:
            module = get_named_linear_module(layer, module_name)
            name = f"layer{layer_idx}.{module_name}"
            h = module.register_forward_hook(make_hook(name))
            hook_handles.append(h)

    device = get_model_main_device(model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits.detach().float().cpu()

    for h in hook_handles:
        h.remove()

    return outputs_dict, logits


def compare_tensor_dicts(ref_dict, new_dict):
    lines = []
    common_keys = sorted(set(ref_dict.keys()) & set(new_dict.keys()))

    for k in common_keys:
        a = ref_dict[k]
        b = new_dict[k]

        diff = a - b
        mse = (diff ** 2).mean().item()
        max_abs = diff.abs().max().item()
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

        lines.append(f"{k}")
        lines.append(f"  mse      : {mse:.12e}")
        lines.append(f"  max_abs  : {max_abs:.12e}")
        lines.append(f"  cosine   : {cos:.12f}")
        lines.append("")

    return lines


def compare_logits(ref_logits, new_logits):
    diff = ref_logits - new_logits
    mse = (diff ** 2).mean().item()
    max_abs = diff.abs().max().item()
    cos = F.cosine_similarity(ref_logits.flatten(), new_logits.flatten(), dim=0).item()

    lines = [
        "[Probe Logits Comparison]",
        f"mse      : {mse:.12e}",
        f"max_abs  : {max_abs:.12e}",
        f"cosine   : {cos:.12f}",
        "",
    ]
    return lines


# =========================================================
# 6. Perplexity
# =========================================================
def compute_perplexity(model, testenc, dev):
    print("\nEvaluating perplexity ...")

    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough tokens for one eval chunk: tokens={testenc.numel()}, seqlen={model.seqlen}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)

            outputs = model(batch, use_cache=False)
            lm_logits = outputs.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache
    return ppl.item()


# =========================================================
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir", type=str, default="quant_ppl_results")
    parser.add_argument("--seed", type=int, default=42)

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="one", choices=["one", "all"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="fc1",
        help="Comma-separated module names, e.g. fc1 or self_attn.q_proj,fc1",
    )

    # exact grouping
    parser.add_argument("--enable_grouping", action="store_true")
    parser.add_argument("--linear_group_size", type=int, default=128)

    # weight quant
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument(
        "--weight_quant_mode",
        type=str,
        default="tensor",
        choices=["tensor", "per_channel", "group"],
    )
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)

    # activation quant
    parser.add_argument("--enable_act_quant", action="store_true")
    parser.add_argument("--act_bits", type=int, default=8)
    parser.add_argument(
        "--act_quant_mode",
        type=str,
        default="tensor",
        choices=["tensor", "per_channel", "per_token", "group"],
    )
    parser.add_argument("--act_ch_axis", type=int, default=-1)
    parser.add_argument("--act_group_size", type=int, default=128)
    # quantization implementation
    parser.add_argument("--quant_impl", type=str, default='fake', choices=['fake', 'hw_like'],)
    parser.add_argument(
        "--weight_cache_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--weight_quant_chunk_rows",
        type=int,
        default=128,
    )

    # eval
    parser.add_argument("--eval_split", type=str, default="test")

    # probe
    parser.add_argument("--do_probe_compare", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)

    lines = []
    lines.append("[Config]")
    lines.append(f"model_id           : {args.model_id}")
    lines.append(f"replace_scope      : {args.replace_scope}")
    lines.append(f"one_layer_idx      : {args.one_layer_idx}")
    lines.append(f"target_modules     : {module_names}")
    lines.append(f"eval_split         : {args.eval_split}")
    lines.append("")
    lines.append("[Experiment Switches]")
    lines.append(f"enable_grouping    : {args.enable_grouping}")
    lines.append(f"linear_group_size  : {args.linear_group_size}")
    lines.append(f"enable_weight_quant: {args.enable_weight_quant}")
    lines.append(f"weight_bits        : {args.weight_bits}")
    lines.append(f"weight_quant_mode  : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis     : {args.weight_ch_axis}")
    lines.append(f"weight_group_size  : {args.weight_group_size}")
    lines.append(f"enable_act_quant   : {args.enable_act_quant}")
    lines.append(f"act_bits           : {args.act_bits}")
    lines.append(f"act_quant_mode     : {args.act_quant_mode}")
    lines.append(f"act_ch_axis        : {args.act_ch_axis}")
    lines.append(f"act_group_size     : {args.act_group_size}")
    lines.append(f"quant_impl         : {args.quant_impl}")
    lines.append("")

    print("[1] Loading WikiText2 + tokenizer...")
    tokenizer, testenc = load_wikitext2_testenc(args.model_id, split=args.eval_split)

    print("[2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.seqlen = model.config.max_position_embeddings

    probe_text = get_probe_text()
    target_layer_indices = resolve_target_layers(model, args.replace_scope, args.one_layer_idx)

    # -----------------------------------------------------
    # Baseline probe
    # -----------------------------------------------------
    if args.do_probe_compare:
        print("[3] Collecting baseline probe outputs/logits...")
        ref_module_outputs, ref_logits = collect_module_outputs(
            model=model,
            tokenizer=tokenizer,
            text=probe_text,
            target_layer_indices=target_layer_indices,
            target_module_names=module_names,
        )
    else:
        ref_module_outputs, ref_logits = None, None

    # -----------------------------------------------------
    # Baseline PPL
    # -----------------------------------------------------
    print("[4] Computing baseline PPL...")
    input_device = get_model_main_device(model)
    baseline_ppl = compute_perplexity(
        model=model,
        testenc=testenc,
        dev=input_device,
    )

    lines.append("[Baseline]")
    lines.append(f"baseline_ppl : {baseline_ppl:.8f}")
    lines.append("")

    # -----------------------------------------------------
    # Replace modules with ExperimentLinear
    # -----------------------------------------------------
    print("[5] Replacing modules with ExperimentLinear...")
    replaced_names = replace_modules_with_experiment_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        enable_grouping=args.enable_grouping,
        linear_group_size=args.linear_group_size,
        enable_weight_quant=args.enable_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        enable_act_quant=args.enable_act_quant,
        act_bits=args.act_bits,
        act_quant_mode=args.act_quant_mode,
        act_ch_axis=args.act_ch_axis,
        act_group_size=args.act_group_size,
        quant_impl=args.quant_impl, 
        weight_cache_device=args.weight_cache_device,
        weight_quant_chunk_rows=args.weight_quant_chunk_rows,
    )

    lines.append("[Replacement]")
    lines.append(f"replaced_count : {len(replaced_names)}")
    for name in replaced_names:
        lines.append(f"  - {name}")
    lines.append("")

    # -----------------------------------------------------
    # Modified probe
    # -----------------------------------------------------
    if args.do_probe_compare:
        print("[6] Collecting modified probe outputs/logits...")
        new_module_outputs, new_logits = collect_module_outputs(
            model=model,
            tokenizer=tokenizer,
            text=probe_text,
            target_layer_indices=target_layer_indices,
            target_module_names=module_names,
        )

        lines.extend(compare_logits(ref_logits, new_logits))
        lines.append("[Module Output Comparison]")
        lines.extend(compare_tensor_dicts(ref_module_outputs, new_module_outputs))

    # -----------------------------------------------------
    # Modified PPL
    # -----------------------------------------------------
    print("[7] Computing modified PPL...")
    modified_ppl = compute_perplexity(
        model=model,
        testenc=testenc,
        dev=input_device,
    )

    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines.append("[Modified]")
    lines.append(f"modified_ppl : {modified_ppl:.8f}")
    lines.append(f"ppl_diff     : {ppl_diff:.12e}")
    lines.append(f"relative_diff: {rel_diff:.12e}")
    lines.append("")

    print("\n".join(lines))
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] Summary saved to {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
