import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
        - per_token
        - group
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
# 3. Quotient / Remainder Activation + Weight Quant Linear
# =========================================================
class QuotRemLinear(nn.Module):
    """
    naive activation decomposition:

        q = round(x / base)
        r = x - base * q

    그리고
        q -> signed n_bits fake quant
        r -> signed n_bits fake quant
        w -> signed weight_bits fake quant

    최종:
        x_recon = base * q_dq + r_dq
        y = F.linear(x_recon, w_dq)

    목적:
    - activation을 quotient / remainder로 나눴을 때
      baseline 대비 방향성 검증
    - naive first prototype
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        qr_base: float = 16.0,
        enable_weight_quant: bool = True,
        weight_bits: int = 4,
        weight_quant_mode: str = "group",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        q_bits: int = 4,
        q_quant_mode: str = "group",
        q_ch_axis: int = -1,
        q_group_size: int = 128,
        r_bits: int = 4,
        r_quant_mode: str = "group",
        r_ch_axis: int = -1,
        r_group_size: int = 128,
        debug_name: str = "",
        collect_q_stats: bool = False,
        split_quant: bool = False,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.debug_name = debug_name
        self.split_quant = split_quant
        # q value check
        self.collect_q_stats = collect_q_stats
        # q = round(x / base) 통계
        self.q_count = 0
        self.q_sum = 0.0
        self.q_sumsq = 0.0
        self.q_min = None
        self.q_max = None
        self.q_absmax = 0.0

        # raw q histogram
        self.q_hist_raw = {}

        # int4 signed range로 clip했을 때 histogram
        self.q_hist_clip = {}

        # int4 overflow 통계
        self.q_overflow_low = 0   # q < -8
        self.q_overflow_high = 0  # q > 7

        self.qr_base = float(qr_base)

        self.enable_weight_quant = enable_weight_quant
        self.weight_bits = weight_bits
        self.weight_quant_mode = weight_quant_mode
        self.weight_ch_axis = weight_ch_axis
        self.weight_group_size = weight_group_size

        self.q_bits = q_bits
        self.q_quant_mode = q_quant_mode
        self.q_ch_axis = q_ch_axis
        self.q_group_size = q_group_size

        self.r_bits = r_bits
        self.r_quant_mode = r_quant_mode
        self.r_ch_axis = r_ch_axis
        self.r_group_size = r_group_size

        if self.enable_weight_quant:
            w_q = fake_quant_symmetric(
                base_linear.weight.detach(),
                n_bits=self.weight_bits,
                mode=self.weight_quant_mode,
                ch_axis=self.weight_ch_axis,
                group_size=self.weight_group_size,
            )
            self.weight = nn.Parameter(w_q, requires_grad=False)
        else:
            self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    # q value check helper function
    def reset_q_stats(self):
        self.q_count = 0
        self.q_sum = 0.0
        self.q_sumsq = 0.0
        self.q_min = None
        self.q_max = None
        self.q_absmax = 0.0
        self.q_hist_raw = {}
        self.q_hist_clip = {}
        self.q_overflow_low = 0
        self.q_overflow_high = 0

    def _update_q_stats(self, q: torch.Tensor):
        """
        q는 이미 round(x / base) 형태라서 '정수값'입니다.
        아직 q fake quant 전에 raw quotient 분포를 저장합니다.
        """
        q_cpu = q.detach().float().cpu().reshape(-1)
        q_int = q_cpu.round().to(torch.int32)

        if q_int.numel() == 0:
            return

        # global stats
        self.q_count += q_int.numel()
        self.q_sum += q_int.float().sum().item()
        self.q_sumsq += (q_int.float() ** 2).sum().item()

        cur_min = int(q_int.min().item())
        cur_max = int(q_int.max().item())
        cur_absmax = float(q_int.abs().max().item())

        if self.q_min is None:
            self.q_min = cur_min
        else:
            self.q_min = min(self.q_min, cur_min)

        if self.q_max is None:
            self.q_max = cur_max
        else:
            self.q_max = max(self.q_max, cur_max)

        self.q_absmax = max(self.q_absmax, cur_absmax)

        # raw histogram
        uniq, cnt = torch.unique(q_int, return_counts=True)
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            self.q_hist_raw[u] = self.q_hist_raw.get(u, 0) + c

        # signed int4 clip histogram
        q_clip = q_int.clamp(-8, 7)
        uniq_c, cnt_c = torch.unique(q_clip, return_counts=True)
        for u, c in zip(uniq_c.tolist(), cnt_c.tolist()):
            self.q_hist_clip[u] = self.q_hist_clip.get(u, 0) + c

        self.q_overflow_low += int((q_int < -8).sum().item())
        self.q_overflow_high += int((q_int > 7).sum().item())

    def _decompose_activation(self, x: torch.Tensor):
        """
        q = round(x / base)
        r = x - base * q
        """
        x_fp = x.float()
        q = torch.round(x_fp / self.qr_base) # 몫을 구하는 부분
        r = x_fp - self.qr_base * q          # 나머지를 구하는 부분
        return q.to(x.dtype), r.to(x.dtype)

    def _quantize_q(self, q: torch.Tensor):
        return fake_quant_symmetric(
            q,
            n_bits=self.q_bits,
            mode=self.q_quant_mode,
            ch_axis=self.q_ch_axis,
            group_size=self.q_group_size,
        )

    def _quantize_r(self, r: torch.Tensor):
        return fake_quant_symmetric(
            r,
            n_bits=self.r_bits,
            mode=self.r_quant_mode,
            ch_axis=self.r_ch_axis,
            group_size=self.r_group_size,
        )

    def _quantize_r_split(self, r: torch.Tensor, q_int: torch.Tensor) -> torch.Tensor:
        """
        q==0 (normal) / q!=0 (outlier) 를 분리해서 각각 독립적인 scale로 group fake quant.

        normal  그룹: outlier 채널이 scale을 당기지 않으므로 더 세밀한 resolution 확보
        outlier 그룹: normal 채널과 무관하게 자체 scale 사용
        """
        qmax = (2 ** (self.r_bits - 1)) - 1
        qmin = -qmax - 1
        eps  = 1e-8
        gs   = self.r_group_size

        x_fp      = r.float()
        orig_shape = x_fp.shape
        H          = orig_shape[-1]
        assert H % gs == 0, f"Last dim {H} not divisible by r_group_size {gs}"

        # [..., num_groups, group_size]
        xg       = x_fp.reshape(orig_shape[:-1] + (H // gs, gs))
        normal_m = (q_int == 0).reshape(orig_shape[:-1] + (H // gs, gs)).float()
        outlier_m = 1.0 - normal_m

        def _group_quant_masked(vals_masked: torch.Tensor) -> torch.Tensor:
            # masked 위치는 0이므로 max_abs 계산에 영향을 주지 않음
            max_abs = vals_masked.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
            scale   = max_abs / qmax
            return torch.round(vals_masked / scale).clamp(qmin, qmax) * scale

        dq_normal  = _group_quant_masked(xg * normal_m)
        dq_outlier = _group_quant_masked(xg * outlier_m)

        # 두 결과를 합산 (각 위치는 둘 중 하나만 non-zero)
        return (dq_normal + dq_outlier).reshape_as(x_fp).to(r.dtype)

    def forward(self, x):
        # w_q는 __init__에서 이미 양자화되어 self.weight에 덮어씌워져 있습니다.
        w_q = self.weight

        # 1) decompose activation
        q, r = self._decompose_activation(x)
        # 1.1) q value check
        if self.collect_q_stats:
            self._update_q_stats(q)
        
        # 2) quantize q
        q_dq = self._quantize_q(q)

        # 3) quantize r
        if self.split_quant:
            # q==0 (normal) / q!=0 (outlier) 분리 양자화
            r_dq = self._quantize_r_split(r, q)
        else:
            r_dq = self._quantize_r(r)

        # 4) reconstruct activation
        x_recon = self.qr_base * q_dq + r_dq

        # 4) single linear using reconstructed activation
        out = F.linear(x_recon, w_q, self.bias)
        return out


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


def resolve_target_layers(model, replace_scope: str, one_layer_idx: int, custom_layer_indices: str = ""):
    num_layers = len(model.model.decoder.layers)

    if replace_scope == "one":
        if one_layer_idx < 0 or one_layer_idx >= num_layers:
            raise ValueError(f"one_layer_idx must be in [0, {num_layers-1}]")
        return [one_layer_idx]
    elif replace_scope == "all":
        return list(range(num_layers))
    elif replace_scope == "custom":
        # 예시: --custom_layer_indices 0,2,4
        if not custom_layer_indices:
            raise ValueError("custom_layer_indices 값이 입력되지 않았습니다. --custom_layer_indices 0,2,4 형식으로 입력해주세요.")
        indices = [int(i.strip()) for i in custom_layer_indices.split(",")]
        return indices
    else:
        raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules_with_quotrem_linear(
    model,
    layer_indices,
    module_names,
    qr_base,
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    q_bits,
    q_quant_mode,
    q_ch_axis,
    q_group_size,
    r_bits,
    r_quant_mode,
    r_ch_axis,
    r_group_size,
    collect_q_stats,
    split_quant,
):
    replaced_names = []

    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)

            new_module = QuotRemLinear(
                base_linear=old_module,
                qr_base=qr_base,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                q_bits=q_bits,
                q_quant_mode=q_quant_mode,
                q_ch_axis=q_ch_axis,
                q_group_size=q_group_size,
                r_bits=r_bits,
                r_quant_mode=r_quant_mode,
                r_ch_axis=r_ch_axis,
                r_group_size=r_group_size,
                debug_name=f"layer{layer_idx}.{module_name}",
                collect_q_stats=collect_q_stats,
                split_quant=split_quant,
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
# Appendix. Check quotient values
# =========================================================
def dump_q_stats(model, output_dir, topk=20):
    lines = []
    lines.append("[Quotient q Statistics]")
    lines.append("")

    found_any = False

    for module in model.modules():
        if not isinstance(module, QuotRemLinear):
            continue

        if module.q_count == 0:
            continue

        found_any = True
        name = module.debug_name if len(module.debug_name) > 0 else "<unnamed>"

        q_mean = module.q_sum / max(module.q_count, 1)
        q_var = module.q_sumsq / max(module.q_count, 1) - (q_mean ** 2)
        q_std = max(q_var, 0.0) ** 0.5

        lines.append(f"{name}")
        lines.append(f"  qr_base           : {module.qr_base}")
        lines.append(f"  q_count           : {module.q_count}")
        lines.append(f"  q_mean            : {q_mean:.8f}")
        lines.append(f"  q_std             : {q_std:.8f}")
        lines.append(f"  q_min             : {module.q_min}")
        lines.append(f"  q_max             : {module.q_max}")
        lines.append(f"  q_absmax          : {module.q_absmax:.8f}")
        lines.append(f"  overflow_low(<-8) : {module.q_overflow_low}")
        lines.append(f"  overflow_high(>7) : {module.q_overflow_high}")
        lines.append("")

        # raw histogram
        lines.append("  [q_hist_raw]")
        for k in sorted(module.q_hist_raw.keys()):
            lines.append(f"    {k:>4d} : {module.q_hist_raw[k]}")
        lines.append("")

        # clipped histogram
        lines.append("  [q_hist_clip_to_int4]")
        for k in sorted(module.q_hist_clip.keys()):
            lines.append(f"    {k:>4d} : {module.q_hist_clip[k]}")
        lines.append("")

    if not found_any:
        lines.append("No q statistics were collected.")
        lines.append("Check that:")
        lines.append("  - --collect_q_stats is enabled")
        lines.append("  - forward actually ran after replacement")
        lines.append("")

    save_txt(lines, Path(output_dir) / "q_stats_summary.txt")
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
    parser.add_argument("--output_dir", type=str, default="quotrem_ppl_results")
    parser.add_argument("--seed", type=int, default=42)

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="one", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="fc1",
        help="Comma-separated module names, e.g. fc1 or self_attn.q_proj,fc1",
    )
    ## custom layer indices for selection
    parser.add_argument(
        "--custom_layer_indices",
        type=str,
        default=None,
        help="Comma-separated layer indices for custom selection, e.g. 0,2,4",
    )

    # quotient / remainder
    parser.add_argument("--qr_base", type=float, default=16.0)

    # weight quant
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument(
        "--weight_quant_mode",
        type=str,
        default="group",
        choices=["tensor", "per_channel", "group"],
    )
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)

    # quotient quant
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument(
        "--q_quant_mode",
        type=str,
        default="group",
        choices=["tensor", "per_channel", "per_token", "group"],
    )
    parser.add_argument("--q_ch_axis", type=int, default=-1)
    parser.add_argument("--q_group_size", type=int, default=128)

    # remainder quant
    parser.add_argument("--r_bits", type=int, default=4)
    parser.add_argument(
        "--r_quant_mode",
        type=str,
        default="group",
        choices=["tensor", "per_channel", "per_token", "group"],
    )
    parser.add_argument("--r_ch_axis", type=int, default=-1)
    parser.add_argument("--r_group_size", type=int, default=128)

    # eval
    parser.add_argument("--eval_split", type=str, default="test")
    # check q value
    parser.add_argument("--collect_q_stats", action="store_true")
    parser.add_argument("--q_stats_topk", type=int, default=20)
    # probe
    parser.add_argument("--do_probe_compare", action="store_true")
    # split quant: q==0 (normal) / q!=0 (outlier) 분리 양자화
    parser.add_argument(
        "--split_quant",
        action="store_true",
        help="q==0 그룹(normal)과 q!=0 그룹(outlier)을 분리해서 각각 독립 scale로 양자화",
    )

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
    lines.append("[Quotient / Remainder]")
    lines.append(f"qr_base            : {args.qr_base}")
    lines.append("")
    lines.append("[Weight Quant]")
    lines.append(f"enable_weight_quant: {args.enable_weight_quant}")
    lines.append(f"weight_bits        : {args.weight_bits}")
    lines.append(f"weight_quant_mode  : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis     : {args.weight_ch_axis}")
    lines.append(f"weight_group_size  : {args.weight_group_size}")
    lines.append("")
    lines.append("[Q Quant]")
    lines.append(f"q_bits             : {args.q_bits}")
    lines.append(f"q_quant_mode       : {args.q_quant_mode}")
    lines.append(f"q_ch_axis          : {args.q_ch_axis}")
    lines.append(f"q_group_size       : {args.q_group_size}")
    lines.append("")
    lines.append("[R Quant]")
    lines.append(f"r_bits             : {args.r_bits}")
    lines.append(f"r_quant_mode       : {args.r_quant_mode}")
    lines.append(f"r_ch_axis          : {args.r_ch_axis}")
    lines.append(f"r_group_size       : {args.r_group_size}")
    lines.append(f"collect_q_stats    : {args.collect_q_stats}")
    lines.append(f"q_stats_topk       : {args.q_stats_topk}")
    lines.append(f"split_quant        : {args.split_quant}")
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
    target_layer_indices = resolve_target_layers(
        model, 
        args.replace_scope, 
        args.one_layer_idx, 
        args.custom_layer_indices
    )

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
    # Replace modules with QuotRemLinear
    # -----------------------------------------------------
    print("[5] Replacing modules with QuotRemLinear...")
    replaced_names = replace_modules_with_quotrem_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        qr_base=args.qr_base,
        enable_weight_quant=args.enable_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        q_bits=args.q_bits,
        q_quant_mode=args.q_quant_mode,
        q_ch_axis=args.q_ch_axis,
        q_group_size=args.q_group_size,
        r_bits=args.r_bits,
        r_quant_mode=args.r_quant_mode,
        r_ch_axis=args.r_ch_axis,
        r_group_size=args.r_group_size,
        collect_q_stats=args.collect_q_stats,
        split_quant=args.split_quant,
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
    if args.collect_q_stats:
        for m in model.modules():
            if isinstance(m, QuotRemLinear):
                m.reset_q_stats()
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
    if args.collect_q_stats:
        q_lines = dump_q_stats(
            model=model,
            output_dir=args.output_dir,
            topk=args.q_stats_topk,
        )
        lines.extend(q_lines)
        lines.append("")
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] Summary saved to {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()