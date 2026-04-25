#quant_ver2_up_sep.py의 양자화 방식을 그대로 가져오고, 여기에 kv cache 양자화 방식까지 추가
# 목표:
#   - Linear 양자화는 ver2_up_sep의 QuotRemLinear를 재사용
#   - Attention 내부의 K / V tensor에 QuotRem fake-quant를 적용
#     (KV cache에 INT 저장 후 복원하는 상황을 시뮬레이션)
#   - OPTAttention 클래스 자체를 커스텀 QuotRemKVOPTAttention 으로 교체 (방식 2)
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
from transformers.models.opt.modeling_opt import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from quant_ver2_up_sep import (
    set_seed,
    ensure_dir,
    save_txt,
    get_model_main_device,
    load_wikitext2_testenc,
    compute_perplexity,
    QuotRemLinear,
    replace_modules_with_quotrem_linear,
    resolve_target_layers,
    parse_module_names,
)


# =========================================================
# A. KV용 QuotRem fake-quantize (단일 tensor 반환)
#    ver2_up_sep 의 _adaptive_base_forward 와 동일한 수치 로직을 따르되,
#    matmul split (q_out + r_out) 이 불가능한 attention 경로용이므로
#    dequantized 값 (sign*Q*base + r_dq) 하나만 반환.
# =========================================================
def quotrem_fake_quantize(
    x: torch.Tensor,
    q_bits: int,
    r_bits: int,
    base_group_size: int,
    r_group_size: int,
) -> torch.Tensor:
    r_max_val = (2 ** (r_bits - 1)) - 1
    r_min_val = -r_max_val - 1
    eps = 1e-8

    x_fp = x.float()
    orig_shape = x_fp.shape
    H = orig_shape[-1]

    base_gs = H if base_group_size == -1 else base_group_size
    r_gs    = H if r_group_size    == -1 else r_group_size

    assert H % base_gs == 0, f"H={H} not divisible by base_group_size={base_gs}"
    assert H % r_gs    == 0, f"H={H} not divisible by r_group_size={r_gs}"

    G_base = H // base_gs
    xg = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))
    max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(eps)

    if q_bits == 1:
        log2_max   = torch.log2(max_abs)
        log2_floor = torch.floor(log2_max)
        log2_ceil  = torch.ceil(log2_max)
        base_floor = torch.pow(2.0, log2_floor)
        base_ceil  = torch.pow(2.0, log2_ceil)
        dist_floor = torch.abs(max_abs - base_floor)
        dist_ceil  = torch.abs(base_ceil - max_abs)
        base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)
        base = base.clamp(min=1.0, max=128.0)

        max_pos_val = xg.amax(dim=-1, keepdim=True)
        min_val     = xg.amin(dim=-1, keepdim=True)
        sign_flag   = torch.where(
            max_pos_val.abs() >= min_val.abs(),
            torch.ones_like(max_pos_val),
            -torch.ones_like(max_pos_val),
        )
        q = (xg * sign_flag >= base / 2.0).float()
        r = xg - sign_flag * base * q
        q_scaled_base = q * sign_flag * base
    else:
        q_max_val = (2 ** (q_bits - 1)) - 1
        threshold = max_abs / q_max_val
        base = torch.full_like(threshold, 64.0)
        base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
        base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
        base = torch.where(threshold <=  8.0, torch.full_like(threshold,  8.0), base)
        base = torch.where(threshold <=  4.0, torch.full_like(threshold,  4.0), base)
        base = torch.where(threshold <=  2.0, torch.full_like(threshold,  2.0), base)
        q = torch.round(xg / base).clamp(-q_max_val, q_max_val)
        r = xg - base * q
        q_scaled_base = q * base

    r_flat = r.reshape(orig_shape)
    G_r = H // r_gs
    r_for_quant = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
    max_abs_r   = r_for_quant.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale_r     = max_abs_r / r_max_val
    r_q         = torch.round(r_for_quant / scale_r).clamp(r_min_val, r_max_val)
    r_dq_grp    = r_q * scale_r
    r_dq        = r_dq_grp.reshape(orig_shape)

    q_scaled = q_scaled_base.reshape(orig_shape)
    x_dq = q_scaled + r_dq
    return x_dq.to(x.dtype)


# =========================================================
# B. OPTAttention 전체를 교체하는 커스텀 클래스
#    기존 OPTAttention (transformers 5.4.0) 의 forward 흐름을 그대로 복제한 뒤,
#    K / V 의 view+transpose 이후 (shape [bsz, num_heads, seq_len, head_dim])
#    에서 QuotRem fake-quant를 적용.
# =========================================================
class QuotRemKVOPTAttention(nn.Module):

    def __init__(
        self,
        base_attn: nn.Module,
        kv_q_bits: int = 1,
        kv_r_bits: int = 3,
        kv_base_group_size: int = 16,
        kv_r_group_size: int = 16,
        kv_target: str = "both",
    ):
        super().__init__()
        # 기존 attention 모듈의 속성 복사 (state 공유)
        self.config     = base_attn.config
        self.embed_dim  = base_attn.embed_dim
        self.num_heads  = base_attn.num_heads
        self.dropout    = base_attn.dropout
        self.enable_bias = base_attn.enable_bias
        self.layer_idx  = base_attn.layer_idx
        self.head_dim   = base_attn.head_dim
        self.is_causal  = base_attn.is_causal
        self.scaling    = base_attn.scaling

        # projection layer 들은 그대로 참조 (QuotRemLinear 로 교체되어 있을 수도 있음)
        self.q_proj   = base_attn.q_proj
        self.k_proj   = base_attn.k_proj
        self.v_proj   = base_attn.v_proj
        self.out_proj = base_attn.out_proj

        # KV QR 설정
        self.kv_q_bits           = kv_q_bits
        self.kv_r_bits           = kv_r_bits
        self.kv_base_group_size  = kv_base_group_size
        self.kv_r_group_size     = kv_r_group_size
        if kv_target not in ("k", "v", "both"):
            raise ValueError(f"kv_target must be one of k/v/both, got {kv_target}")
        self.kv_target = kv_target

    def _qr_kv(self, t: torch.Tensor) -> torch.Tensor:
        return quotrem_fake_quantize(
            t,
            q_bits=self.kv_q_bits,
            r_bits=self.kv_r_bits,
            base_group_size=self.kv_base_group_size,
            r_group_size=self.kv_r_group_size,
        )

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states   = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # KV fake-quant: [bsz, num_heads, seq_len, head_dim] 상태에서 적용
        if self.kv_target in ("k", "both"):
            key_states = self._qr_kv(key_states)
        if self.kv_target in ("v", "both"):
            value_states = self._qr_kv(value_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# =========================================================
# C. 모델 내 OPTAttention 들을 QuotRemKVOPTAttention 으로 교체
# =========================================================
def replace_attn_with_quotrem_kv(
    model,
    layer_indices,
    kv_q_bits,
    kv_r_bits,
    kv_base_group_size,
    kv_r_group_size,
    kv_target,
):
    replaced = []
    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        base_attn = layer.self_attn
        new_attn = QuotRemKVOPTAttention(
            base_attn=base_attn,
            kv_q_bits=kv_q_bits,
            kv_r_bits=kv_r_bits,
            kv_base_group_size=kv_base_group_size,
            kv_r_group_size=kv_r_group_size,
            kv_target=kv_target,
        )
        # device / dtype 유지
        dev = next(base_attn.parameters()).device
        new_attn = new_attn.to(dev)
        layer.self_attn = new_attn
        replaced.append(f"layer{layer_idx}.self_attn")
    return replaced


# =========================================================
# D. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir", type=str, default="quotrem_ppl_results_v3_kv")
    parser.add_argument("--seed", type=int, default=42)

    # 적용 범위 (Linear QR 과 KV QR 모두 동일 범위 사용)
    parser.add_argument("--replace_scope", type=str, default="all", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument("--custom_layer_indices", type=str, default=None)

    # -------- Linear QuotRem (ver2_up_sep 재사용) --------
    parser.add_argument("--enable_linear_quant", action="store_true",
                        help="True면 Linear 모듈들을 QuotRemLinear로 교체")
    parser.add_argument("--target_modules", type=str, default="fc1,fc2,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj",
                        help="Linear QR를 적용할 모듈 목록")

    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--weight_quant_mode", type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)

    parser.add_argument("--q_bits", type=int, default=1)
    parser.add_argument("--r_bits", type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--r_group_size", type=int, default=128)

    # -------- KV cache QuotRem (신규) --------
    parser.add_argument("--enable_kv_quant", action="store_true",
                        help="True면 OPTAttention을 QuotRemKVOPTAttention으로 교체")
    parser.add_argument("--kv_q_bits", type=int, default=1)
    parser.add_argument("--kv_r_bits", type=int, default=3)
    parser.add_argument("--kv_base_group_size", type=int, default=16,
                        help="head_dim(=128)의 약수여야 함")
    parser.add_argument("--kv_r_group_size", type=int, default=16)
    parser.add_argument("--kv_target", type=str, default="both",
                        choices=["k", "v", "both"])

    # eval
    parser.add_argument("--eval_split", type=str, default="test")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules) if args.enable_linear_quant else []

    lines = []
    lines.append("[Config]")
    lines.append(f"model_id             : {args.model_id}")
    lines.append(f"replace_scope        : {args.replace_scope}")
    lines.append(f"one_layer_idx        : {args.one_layer_idx}")
    lines.append(f"custom_layer_indices : {args.custom_layer_indices}")
    lines.append(f"eval_split           : {args.eval_split}")
    lines.append("")
    lines.append("[Linear QR]")
    lines.append(f"enable_linear_quant  : {args.enable_linear_quant}")
    lines.append(f"target_modules       : {module_names}")
    lines.append(f"enable_weight_quant  : {args.enable_weight_quant}")
    lines.append(f"weight_bits          : {args.weight_bits}")
    lines.append(f"weight_quant_mode    : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis       : {args.weight_ch_axis}")
    lines.append(f"weight_group_size    : {args.weight_group_size}")
    lines.append(f"q_bits               : {args.q_bits}")
    lines.append(f"r_bits               : {args.r_bits}")
    lines.append(f"base_group_size      : {args.base_group_size}")
    lines.append(f"r_group_size         : {args.r_group_size}")
    lines.append("")
    lines.append("[KV QR]")
    lines.append(f"enable_kv_quant      : {args.enable_kv_quant}")
    lines.append(f"kv_target            : {args.kv_target}")
    lines.append(f"kv_q_bits            : {args.kv_q_bits}")
    lines.append(f"kv_r_bits            : {args.kv_r_bits}")
    lines.append(f"kv_base_group_size   : {args.kv_base_group_size}")
    lines.append(f"kv_r_group_size      : {args.kv_r_group_size}")
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

    linear_replaced = []
    kv_replaced = []

    if args.enable_linear_quant:
        print("[4a] Replacing Linear modules with QuotRemLinear...")
        linear_replaced = replace_modules_with_quotrem_linear(
            model=model,
            layer_indices=target_layer_indices,
            module_names=module_names,
            enable_weight_quant=args.enable_weight_quant,
            weight_bits=args.weight_bits,
            weight_quant_mode=args.weight_quant_mode,
            weight_ch_axis=args.weight_ch_axis,
            weight_group_size=args.weight_group_size,
            q_bits=args.q_bits,
            r_bits=args.r_bits,
            base_group_size=args.base_group_size,
            r_group_size=args.r_group_size,
        )

    if args.enable_kv_quant:
        print("[4b] Replacing OPTAttention with QuotRemKVOPTAttention...")
        kv_replaced = replace_attn_with_quotrem_kv(
            model=model,
            layer_indices=target_layer_indices,
            kv_q_bits=args.kv_q_bits,
            kv_r_bits=args.kv_r_bits,
            kv_base_group_size=args.kv_base_group_size,
            kv_r_group_size=args.kv_r_group_size,
            kv_target=args.kv_target,
        )

    lines.append("[Replacement]")
    lines.append(f"linear_replaced_count : {len(linear_replaced)}")
    for n in linear_replaced:
        lines.append(f"  - {n}")
    lines.append(f"kv_replaced_count     : {len(kv_replaced)}")
    for n in kv_replaced:
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
