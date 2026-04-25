# Amove (MICRO'25) reproduce — activation 에만 residual approximation 적용, weight 는 기존 fake_quant_symmetric 사용
# 평가 환경: 기존 quant_ver2_up_sep.compute_perplexity (wikitext2, seqlen=2048, use_cache=False) 동일
# Config: Amove-Conservative (K=32, C=4, E=2), scale/residual 는 FP16 (추후 FP8 교체 예정)
# 적용 범위: Linear layers (self_attn.q/k/v/out_proj, fc1, fc2) — attention K/V 경로는 제외
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

from quant_ver2_up_sep import (
    set_seed,
    ensure_dir,
    save_txt,
    get_model_main_device,
    load_wikitext2_testenc,
    compute_perplexity,
    fake_quant_symmetric,
    get_named_linear_module,
    set_named_linear_module,
    resolve_target_layers,
    parse_module_names,
)


# =========================================================
# A. Amove activation fake-quantize (Algorithm 1 + Eq.3, online mode)
#    - Coarse group size K, cluster size C, encoding bits E
#    - 각 cluster Δ_i = max(|x_ci|) / (2^(b-1) - 1)
#    - Δ_base = max_i Δ_i  (= group 전체의 symmetric scale)
#    - R = Σ |Δ_i − Δ_base| / (num_clusters · E)     [Algorithm 1 line 13, activation 경로]
#    - 내부 encoding 은 non-negative {0,..,2^E-1} 로 저장; paper 의 non-positive 표기와는 부호만 다름
#    - Δ_ci_approx = Δ_base − e_i · R                [Eq.3, 부호 조정]
# =========================================================
def amove_fake_quantize_activation(
    x,
    a_bits,
    group_size,
    cluster_size,
    encoding_bits,
):
    eps = 1e-8
    qmax = (2 ** (a_bits - 1)) - 1
    qmin = -qmax - 1
    e_max = (2 ** encoding_bits) - 1

    x_fp = x.float()
    orig_shape = x_fp.shape
    H = orig_shape[-1]

    assert H % group_size == 0, f"H={H} not divisible by group_size={group_size}"
    assert group_size % cluster_size == 0, f"group_size={group_size} not divisible by cluster_size={cluster_size}"

    num_groups = H // group_size
    num_clusters = group_size // cluster_size

    # [..., num_groups, num_clusters, cluster_size]
    xg = x_fp.reshape(orig_shape[:-1] + (num_groups, num_clusters, cluster_size))

    # Step 1: per-cluster Δ_i
    max_abs_cl = xg.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    delta_i = max_abs_cl / qmax                                          # [..., G, NC, 1]

    # Step 2: Δ_base = max over clusters in each group
    delta_base = delta_i.amax(dim=-2, keepdim=True)                      # [..., G, 1,  1]

    # Step 3: R (activation online average deviation)
    abs_dev = (delta_base - delta_i).abs()
    R = abs_dev.sum(dim=-2, keepdim=True) / (num_clusters * encoding_bits)   # [..., G, 1, 1]
    R_safe = R.clamp_min(eps)

    # Step 4: encoding e_i (non-negative representation ∈ [0, e_max])
    e_i = torch.round((delta_base - delta_i) / R_safe).clamp(0, e_max)   # [..., G, NC, 1]

    # Step 5: reconstructed per-cluster scale (R≈0 이면 fallback)
    delta_ci = torch.where(R > eps, delta_base - e_i * R, delta_base)
    delta_ci = delta_ci.clamp_min(eps)

    # Step 6: per-cluster quantize / dequantize
    xq   = torch.round(xg / delta_ci).clamp(qmin, qmax)
    x_dq = (xq * delta_ci).reshape(orig_shape)
    return x_dq.to(x.dtype)


# =========================================================
# B. AmoveLinear: weight 는 기존 fake_quant_symmetric, activation 은 Amove
# =========================================================
class AmoveLinear(nn.Module):

    def __init__(
        self,
        base_linear,
        enable_weight_quant=True,
        weight_bits=4,
        weight_quant_mode="group",
        weight_ch_axis=0,
        weight_group_size=16,
        a_bits=4,
        a_group_size=32,
        a_cluster_size=4,
        a_encoding_bits=2,
        debug_name="",
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.debug_name   = debug_name

        self.a_bits          = a_bits
        self.a_group_size    = a_group_size
        self.a_cluster_size  = a_cluster_size
        self.a_encoding_bits = a_encoding_bits

        if enable_weight_quant:
            w_q = fake_quant_symmetric(
                base_linear.weight.detach(),
                n_bits=weight_bits,
                mode=weight_quant_mode,
                ch_axis=weight_ch_axis,
                group_size=weight_group_size,
            )
            self.weight = nn.Parameter(w_q, requires_grad=False)
        else:
            self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        x_dq = amove_fake_quantize_activation(
            x,
            a_bits=self.a_bits,
            group_size=self.a_group_size,
            cluster_size=self.a_cluster_size,
            encoding_bits=self.a_encoding_bits,
        )
        return F.linear(x_dq, self.weight, self.bias)


# =========================================================
# C. Module replacement
# =========================================================
def replace_modules_with_amove_linear(
    model,
    layer_indices,
    module_names,
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    a_bits,
    a_group_size,
    a_cluster_size,
    a_encoding_bits,
):
    replaced_names = []
    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)
            new_module = AmoveLinear(
                base_linear=old_module,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                a_bits=a_bits,
                a_group_size=a_group_size,
                a_cluster_size=a_cluster_size,
                a_encoding_bits=a_encoding_bits,
                debug_name=f"layer{layer_idx}.{module_name}",
            )
            set_named_linear_module(layer, module_name, new_module)
            replaced_names.append(f"layer{layer_idx}.{module_name}")
    return replaced_names


# =========================================================
# D. Main (평가 환경은 quant_ver2_up_sep.compute_perplexity 그대로 사용)
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir", type=str, default="amove_ppl_results")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--replace_scope", type=str, default="all", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument("--target_modules", type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices", type=str, default=None)

    # Weight quant (QuotRem 과 동일 조건)
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--weight_quant_mode", type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=16)

    # Amove activation config (Amove-Conservative 기본)
    parser.add_argument("--a_bits",          type=int, default=4)
    parser.add_argument("--a_group_size",    type=int, default=32)
    parser.add_argument("--a_cluster_size",  type=int, default=4)
    parser.add_argument("--a_encoding_bits", type=int, default=2)

    parser.add_argument("--eval_split", type=str, default="test")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)

    lines = []
    lines.append("[Config]")
    lines.append(f"model_id            : {args.model_id}")
    lines.append(f"replace_scope       : {args.replace_scope}")
    lines.append(f"one_layer_idx       : {args.one_layer_idx}")
    lines.append(f"target_modules      : {module_names}")
    lines.append(f"eval_split          : {args.eval_split}")
    lines.append("")
    lines.append("[Weight Quant]")
    lines.append(f"enable_weight_quant : {args.enable_weight_quant}")
    lines.append(f"weight_bits         : {args.weight_bits}")
    lines.append(f"weight_quant_mode   : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis      : {args.weight_ch_axis}")
    lines.append(f"weight_group_size   : {args.weight_group_size}")
    lines.append("")
    lines.append("[Amove Activation]")
    lines.append(f"a_bits              : {args.a_bits}")
    lines.append(f"a_group_size  (K)   : {args.a_group_size}")
    lines.append(f"a_cluster_size(C)   : {args.a_cluster_size}")
    lines.append(f"a_encoding_bits(E)  : {args.a_encoding_bits}")
    lines.append(f"num_clusters/group  : {args.a_group_size // args.a_cluster_size}")
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

    target_layer_indices = resolve_target_layers(
        model,
        args.replace_scope,
        args.one_layer_idx,
        args.custom_layer_indices,
    )

    input_device = get_model_main_device(model)

    print("[3] Baseline PPL...")
    baseline_ppl = compute_perplexity(model, testenc, input_device)
    lines.append("[Baseline]")
    lines.append(f"baseline_ppl : {baseline_ppl:.8f}")
    lines.append("")

    print("[4] Replacing Linear modules with AmoveLinear...")
    replaced_names = replace_modules_with_amove_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        enable_weight_quant=args.enable_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        a_bits=args.a_bits,
        a_group_size=args.a_group_size,
        a_cluster_size=args.a_cluster_size,
        a_encoding_bits=args.a_encoding_bits,
    )

    lines.append("[Replacement]")
    lines.append(f"replaced_count : {len(replaced_names)}")
    for n in replaced_names:
        lines.append(f"  - {n}")
    lines.append("")

    print("[5] Modified PPL...")
    modified_ppl = compute_perplexity(model, testenc, input_device)

    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines.append("[Modified]")
    lines.append(f"modified_ppl : {modified_ppl:.8f}")
    lines.append(f"ppl_diff     : {ppl_diff:.12e}")
    lines.append(f"relative_diff: {rel_diff:.12e}")
    lines.append("")

    print("\n".join(lines))
    out_path = Path(args.output_dir) / "summary.txt"
    save_txt(lines, out_path)
    print(f"[Done] Summary saved to {out_path}")


if __name__ == "__main__":
    main()
