import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# ver2_cali: ver2와 동일 알고리즘
# calibration 128 샘플로 per-group max 수집 → 고정 scale로 eval
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
# 1. Dataset
# =========================================================
def load_wikitext2_testenc(model_id, split="test"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids
    return tokenizer, testenc


def load_wikitext2_calib_samples(model_id, n_samples=128, seqlen=2048, seed=42):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(ds["text"])
    enc = tokenizer(full_text, return_tensors="pt").input_ids
    total_tokens = enc.shape[1]
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, total_tokens - seqlen - 1)
        samples.append(enc[:, start: start + seqlen])
    return tokenizer, samples


# =========================================================
# 2. Fake Quantization (weight quant 용)
# =========================================================
def fake_quant_symmetric(x, n_bits=8, mode="tensor", ch_axis=-1, group_size=None, eps=1e-8):
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    x_fp = x.float()

    if mode == "tensor":
        max_abs = x_fp.abs().max()
        scale = (max_abs / qmax).clamp_min(eps)
        return (torch.round(x_fp / scale).clamp(qmin, qmax) * scale).to(x.dtype)

    elif mode == "per_channel":
        axis = resolve_axis(x_fp.dim(), ch_axis)
        reduce_dims = tuple(d for d in range(x_fp.dim()) if d != axis)
        max_abs = x_fp.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)
        return (torch.round(x_fp / scale).clamp(qmin, qmax) * scale).to(x.dtype)

    elif mode == "group":
        if group_size is None:
            raise ValueError("group_size required")
        if x_fp.shape[-1] % group_size != 0:
            raise ValueError(f"Last dim ({x_fp.shape[-1]}) not divisible by group_size ({group_size})")
        num_groups = x_fp.shape[-1] // group_size
        xg = x_fp.reshape(x_fp.shape[:-1] + (num_groups, group_size))
        max_abs = xg.abs().amax(dim=-1, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)
        return (torch.round(xg / scale).clamp(qmin, qmax) * scale).reshape_as(x_fp).to(x.dtype)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# 3. Power-of-2 base 선택 (조건 A)
# =========================================================
def select_pow2_base(threshold: torch.Tensor) -> torch.Tensor:
    """{2,4,8,16,32,64} 중 threshold 이상을 만족하는 최솟값"""
    base = torch.full_like(threshold, 64.0)
    base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
    base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
    base = torch.where(threshold <=  8.0, torch.full_like(threshold,  8.0), base)
    base = torch.where(threshold <=  4.0, torch.full_like(threshold,  4.0), base)
    base = torch.where(threshold <=  2.0, torch.full_like(threshold,  2.0), base)
    return base


# =========================================================
# 4. QR forward (공통 로직)
# =========================================================
def qr_forward_dynamic(xg, q_bits, q_max_val, r_bits, r_max_val, r_min_val, eps):
    """ver2와 동일: 매 입력마다 dynamic하게 base, scale_r 결정"""
    max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(eps)  # [..., G, 1]

    if q_bits == 1:
        # q ∈ {-1, +1}: 부호만 표현, base = max_abs
        base = select_pow2_base(max_abs)
        q = torch.sign(xg)
        q = torch.where(q == 0, torch.ones_like(q), q)  # 0은 +1로 처리
    else:
        base = select_pow2_base(max_abs / q_max_val)
        q = torch.round(xg / base).clamp(-q_max_val, q_max_val)

    r = xg - base * q
    max_abs_r = r.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale_r   = max_abs_r / r_max_val
    r_q       = torch.round(r / scale_r).clamp(r_min_val, r_max_val)
    r_dq      = r_q * scale_r

    return base * q + r_dq, max_abs


def qr_forward_static(xg, q_bits, q_max_val, r_bits, r_max_val, r_min_val, fixed_base, fixed_scale_r):
    """첫 배치 후 고정된 base, scale_r 사용"""
    if q_bits == 1:
        # q ∈ {-1, +1}: 부호만 표현, base에 무관하게 항상 올바름
        q = torch.sign(xg)
        q = torch.where(q == 0, torch.ones_like(q), q)  # 0은 +1로 처리
    else:
        q = torch.round(xg / fixed_base).clamp(-q_max_val, q_max_val)

    r    = xg - fixed_base * q
    r_q  = torch.round(r / fixed_scale_r).clamp(r_min_val, r_max_val)
    r_dq = r_q * fixed_scale_r

    return fixed_base * q + r_dq


# =========================================================
# 5. QuotRemLinear
# =========================================================
class QuotRemLinear(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        enable_weight_quant=True,
        weight_bits=4,
        weight_quant_mode="group",
        weight_ch_axis=0,
        weight_group_size=128,
        q_bits=4,
        r_bits=4,
        r_group_size=128,
        debug_name="",
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.debug_name   = debug_name
        self.q_bits       = q_bits
        self.r_bits       = r_bits
        self.r_group_size = r_group_size

        G = self.in_features // r_group_size  # num_groups

        # 첫 번째 forward에서 동적으로 결정 후 고정 (oa-lama Quantizer 방식)
        self.init = True

        # 고정 scale (register_buffer → 모델 device 따라감)
        self.register_buffer('fixed_base',    torch.ones(G))   # [G]
        self.register_buffer('fixed_scale_r', torch.ones(G))   # [G]

        if enable_weight_quant:
            w_q = fake_quant_symmetric(
                base_linear.weight.detach(),
                n_bits=weight_bits, mode=weight_quant_mode,
                ch_axis=weight_ch_axis, group_size=weight_group_size,
            )
            self.weight = nn.Parameter(w_q, requires_grad=False)
        else:
            self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        gs        = self.r_group_size
        q_max_val = (2 ** (self.q_bits - 1)) - 1 if self.q_bits > 1 else 1
        r_max_val = (2 ** (self.r_bits - 1)) - 1
        r_min_val = -r_max_val - 1
        eps       = 1e-8

        x_fp = x.float()
        H    = x_fp.shape[-1]
        xg   = x_fp.reshape(x_fp.shape[:-1] + (H // gs, gs))  # [..., G, gs]

        if self.init:
            # 첫 번째 forward: 이 입력으로 동적으로 base/scale_r 결정 후 고정
            x_recon_g, max_abs = qr_forward_dynamic(
                xg, self.q_bits, q_max_val, self.r_bits, r_max_val, r_min_val, eps
            )
            # max_abs: [..., G, 1] → per-group max → [G]
            fb = max_abs.squeeze(-1).reshape(-1, H // gs).amax(dim=0).detach()
            if self.q_bits == 1:
                fb = select_pow2_base(fb)
                # q ∈ {-1,+1}: r = x - sign(x)*base, |r| <= base (x와 base 같은 부호이므로)
                fs = fb / r_max_val
            else:
                fb = select_pow2_base(fb / q_max_val)
                fs = fb / 2.0 / r_max_val

            self.fixed_base.copy_(fb)
            self.fixed_scale_r.copy_(fs)
            self.init = False

        else:
            # 이후 forward: 고정된 base/scale_r 사용
            n_extra    = xg.dim() - 2
            vshape     = (1,) * n_extra + (H // gs, 1)
            fixed_base = self.fixed_base.to(x.device).view(vshape)
            fixed_sr   = self.fixed_scale_r.to(x.device).view(vshape)

            x_recon_g = qr_forward_static(
                xg, self.q_bits, q_max_val, self.r_bits, r_max_val, r_min_val,
                fixed_base, fixed_sr
            )

        x_recon = x_recon_g.reshape_as(x_fp).to(x.dtype)
        return F.linear(x_recon, self.weight, self.bias)


# =========================================================
# 6. Module Access / Replacement
# =========================================================
def get_named_linear_module(layer, module_name):
    mapping = {
        "self_attn.q_proj":  layer.self_attn.q_proj,
        "self_attn.k_proj":  layer.self_attn.k_proj,
        "self_attn.v_proj":  layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1": layer.fc1,
        "fc2": layer.fc2,
    }
    if module_name not in mapping:
        raise ValueError(f"Unsupported module_name: {module_name}")
    return mapping[module_name]


def set_named_linear_module(layer, module_name, new_module):
    if   module_name == "self_attn.q_proj":  layer.self_attn.q_proj  = new_module
    elif module_name == "self_attn.k_proj":  layer.self_attn.k_proj  = new_module
    elif module_name == "self_attn.v_proj":  layer.self_attn.v_proj  = new_module
    elif module_name == "self_attn.out_proj": layer.self_attn.out_proj = new_module
    elif module_name == "fc1":               layer.fc1               = new_module
    elif module_name == "fc2":               layer.fc2               = new_module
    else: raise ValueError(f"Unsupported module_name: {module_name}")


def parse_module_names(s):
    names = [x.strip() for x in s.split(",") if x.strip()]
    valid = {"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
             "self_attn.out_proj","fc1","fc2"}
    for n in names:
        if n not in valid:
            raise ValueError(f"Invalid module name: {n}")
    return names


def resolve_target_layers(model, replace_scope, one_layer_idx, custom_layer_indices=""):
    num_layers = len(model.model.decoder.layers)
    if replace_scope == "one":    return [one_layer_idx]
    elif replace_scope == "all":  return list(range(num_layers))
    elif replace_scope == "custom":
        return [int(i.strip()) for i in custom_layer_indices.split(",")]
    else: raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules(model, layer_indices, module_names,
                    enable_weight_quant, weight_bits, weight_quant_mode,
                    weight_ch_axis, weight_group_size, q_bits, r_bits, r_group_size):
    replaced = []
    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old = get_named_linear_module(layer, module_name)
            new = QuotRemLinear(
                base_linear=old,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits, weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis, weight_group_size=weight_group_size,
                q_bits=q_bits, r_bits=r_bits, r_group_size=r_group_size,
                debug_name=f"layer{layer_idx}.{module_name}",
            )
            set_named_linear_module(layer, module_name, new)
            replaced.append(f"layer{layer_idx}.{module_name}")
    return replaced


# =========================================================
# 7. Perplexity
# =========================================================
def compute_perplexity(model, testenc, dev):
    print("\nEvaluating perplexity ...")
    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens: {testenc.numel()}, seqlen={model.seqlen}")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, i * model.seqlen:(i + 1) * model.seqlen].to(dev)
            lm_logits = model(batch, use_cache=False).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()


# =========================================================
# 9. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",            type=str,  default="facebook/opt-6.7b")
    parser.add_argument("--output_dir",           type=str,  default="quotrem_ppl_results_v2_cali")
    parser.add_argument("--seed",                 type=int,  default=42)
    parser.add_argument("--replace_scope",        type=str,  default="all",
                        choices=["one","all","custom"])
    parser.add_argument("--one_layer_idx",        type=int,  default=10)
    parser.add_argument("--target_modules",       type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices", type=str,  default=None)
    parser.add_argument("--enable_weight_quant",  action="store_true")
    parser.add_argument("--weight_bits",          type=int,  default=4)
    parser.add_argument("--weight_quant_mode",    type=str,  default="group",
                        choices=["tensor","per_channel","group"])
    parser.add_argument("--weight_ch_axis",       type=int,  default=0)
    parser.add_argument("--weight_group_size",    type=int,  default=128)
    parser.add_argument("--q_bits",               type=int,  default=4)
    parser.add_argument("--r_bits",               type=int,  default=4)
    parser.add_argument("--r_group_size",         type=int,  default=128)
    parser.add_argument("--n_calib_samples",      type=int,  default=128)
    parser.add_argument("--calib_seqlen",         type=int,  default=2048)
    parser.add_argument("--eval_split",           type=str,  default="test")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    module_names = parse_module_names(args.target_modules)

    lines = ["[Config]",
             f"model_id           : {args.model_id}",
             f"replace_scope      : {args.replace_scope}",
             f"target_modules     : {module_names}",
             f"n_calib_samples    : {args.n_calib_samples}",
             f"calib_seqlen       : {args.calib_seqlen}",
             "",
             "[Weight Quant]",
             f"enable_weight_quant: {args.enable_weight_quant}",
             f"weight_bits        : {args.weight_bits}",
             f"weight_quant_mode  : {args.weight_quant_mode}",
             f"weight_group_size  : {args.weight_group_size}",
             "",
             "[QR Config]",
             f"q_bits             : {args.q_bits}",
             f"r_bits             : {args.r_bits}",
             f"r_group_size       : {args.r_group_size}",
             ""]

    print("[1] Loading data ...")
    tokenizer, testenc = load_wikitext2_testenc(args.model_id, split=args.eval_split)

    print("[2] Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.seqlen = model.config.max_position_embeddings
    dev = get_model_main_device(model)

    target_layer_indices = resolve_target_layers(
        model, args.replace_scope, args.one_layer_idx, args.custom_layer_indices)

    print("[3] Baseline PPL ...")
    baseline_ppl = compute_perplexity(model, testenc, dev)
    lines += ["[Baseline]", f"baseline_ppl : {baseline_ppl:.8f}", ""]

    print("[4] Replacing modules ...")
    replaced = replace_modules(
        model, target_layer_indices, module_names,
        args.enable_weight_quant, args.weight_bits, args.weight_quant_mode,
        args.weight_ch_axis, args.weight_group_size,
        args.q_bits, args.r_bits, args.r_group_size,
    )
    lines += ["[Replacement]", f"replaced_count : {len(replaced)}", ""]

    print("[5] Modified PPL (첫 번째 배치에서 scale 자동 결정) ...")
    modified_ppl = compute_perplexity(model, testenc, dev)
    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines += ["[Modified (first-batch static, oa-lama style)]",
              f"modified_ppl : {modified_ppl:.8f}",
              f"ppl_diff     : {ppl_diff:.12e}",
              f"relative_diff: {rel_diff:.12e}", ""]

    print("\n".join(lines))
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
