import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# ver 3
# base 결정은 adaptive하게 
# R scale은 자기만의 타이트하게 결정, 몫의 개수 분포 출력

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
# 2. Fake Quantization (weight quant 용)
# =========================================================
def fake_quant_symmetric(
    x: torch.Tensor,
    n_bits: int = 8,
    mode: str = "tensor",
    ch_axis: int = -1,
    group_size: int = None,
    eps: float = 1e-8,
):
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
# 3. Power-of-2 Adaptive QR Linear
#
# 아이디어:
#   각 group의 max_abs를 기반으로 qr_base를 {2, 4, 8, 16} 중에서 선택
#   조건: round(x / qr_base)가 q_bits 범위 안에 들어오는 가장 작은 2의 승수
#     → qr_base >= max_abs / q_max 를 만족하는 adaptive base 선택
#
#   qr_base가 2의 승수이므로:
#     - 별도 float scale 저장 불필요 (2-bit index로 충분)
#     - CIM에서 shift 연산으로 처리 가능
#
#   R scale은 실제 R값의 max_abs 기반으로 결정 (tight scale):
#     scale_r = actual_max_abs_r / r_max
# =========================================================
class QuotRemLinear(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        enable_weight_quant: bool = True,
        weight_bits: int = 4,
        weight_quant_mode: str = "group",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        q_bits: int = 4,
        r_bits: int = 4,
        r_group_size: int = 128,
        debug_name: str = "",
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
            self.weight = nn.Parameter(
                base_linear.weight.detach().clone(), requires_grad=False
            )

        if base_linear.bias is not None:
            self.bias = nn.Parameter(
                base_linear.bias.detach().clone(), requires_grad=False
            )
        else:
            self.bias = None

        self.q_counts = {}
        self.total_q = 0

    def _adaptive_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Power-of-2 adaptive base QR quantization.

        각 group마다 {2, 4, 8, 16} 중 조건을 만족하는 가장 작은 값을 qr_base로 선택:
          조건: max_abs / qr_base <= q_max  (q가 q_bits 범위를 초과하지 않음)

        qr_base는 2의 승수이므로 별도 float quant 불필요.
        R scale은 실제 R값의 max_abs로 결정 (tight scale).
        """
        q_max_val = (2 ** (self.q_bits - 1)) - 1   # e.g. 3-bit: 3, 4-bit: 7
        r_max_val = (2 ** (self.r_bits - 1)) - 1   # e.g. 5-bit: 15, 4-bit: 7
        r_min_val = -r_max_val - 1
        eps = 1e-8
        gs  = self.r_group_size

        x_fp       = x.float()
        orig_shape = x_fp.shape
        H          = orig_shape[-1]
        assert H % gs == 0, f"Last dim {H} not divisible by r_group_size {gs}"

        xg = x_fp.reshape(orig_shape[:-1] + (H // gs, gs))  # [..., G, gs]

        # 1) per-group continuous adaptive base (Q용 scale)
        max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(eps)  # [..., G, 1]
        base    = max_abs / q_max_val  # Q가 항상 [-q_max, q_max] 범위를 꽉 채움

        # 2) QR decompose (asymmetric clamp: e.g. -4~3 for 3-bit)
        q = torch.round(xg / base).clamp(-q_max_val - 1, q_max_val)
        r = xg - base * q

        # 3) Q distribution 통계
        unique_vals, counts = torch.unique(q, return_counts=True)
        for val, cnt in zip(unique_vals.cpu().tolist(), counts.cpu().tolist()):
            self.q_counts[val] = self.q_counts.get(val, 0) + cnt
        self.total_q += q.numel()

        # 4) R quantize: 실제 R max 기반 독립 scale (R용 scale, base와 무관)
        #    base/2 bound를 쓰지 않고 실제 분포에서 tight하게 결정
        max_abs_r = r.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale_r   = max_abs_r / r_max_val
        r_q       = torch.round(r / scale_r).clamp(r_min_val, r_max_val)
        r_dq      = r_q * scale_r

        # 5) reconstruct  (scale 2개: base, scale_r 각각 독립)
        x_recon = base * q + r_dq                                      # [..., G, gs]
        return x_recon.reshape_as(x_fp).to(x.dtype)

    def forward(self, x):
        x_recon = self._adaptive_base_forward(x)
        return F.linear(x_recon, self.weight, self.bias)


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
        if not custom_layer_indices:
            raise ValueError("--custom_layer_indices 0,2,4 형식으로 입력해주세요.")
        return [int(i.strip()) for i in custom_layer_indices.split(",")]
    else:
        raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules_with_quotrem_linear(
    model,
    layer_indices,
    module_names,
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    q_bits,
    r_bits,
    r_group_size,
):
    replaced_names = []

    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)

            new_module = QuotRemLinear(
                base_linear=old_module,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                q_bits=q_bits,
                r_bits=r_bits,
                r_group_size=r_group_size,
                debug_name=f"layer{layer_idx}.{module_name}",
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
            x = output[0] if isinstance(output, tuple) else output
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
    for k in sorted(set(ref_dict.keys()) & set(new_dict.keys())):
        a, b = ref_dict[k], new_dict[k]
        diff = a - b
        lines.append(f"{k}")
        lines.append(f"  mse      : {(diff**2).mean().item():.12e}")
        lines.append(f"  max_abs  : {diff.abs().max().item():.12e}")
        lines.append(f"  cosine   : {F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.12f}")
        lines.append("")
    return lines


def compare_logits(ref_logits, new_logits):
    diff = ref_logits - new_logits
    return [
        "[Probe Logits Comparison]",
        f"mse      : {(diff**2).mean().item():.12e}",
        f"max_abs  : {diff.abs().max().item():.12e}",
        f"cosine   : {F.cosine_similarity(ref_logits.flatten(), new_logits.flatten(), dim=0).item():.12f}",
        "",
    ]


def get_q_distribution(model):
    total_counts = {}
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QuotRemLinear):
            for k, v in module.q_counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
            total_elements += module.total_q
            
    dist_lines = []
    dist_lines.append("")
    dist_lines.append("="*40)
    dist_lines.append(" Quotient (Q) Distribution Summary")
    dist_lines.append("="*40)
    
    if total_elements == 0:
        dist_lines.append("수집된 Q 값이 없습니다.")
        return dist_lines

    for k in sorted(total_counts.keys()):
        cnt = total_counts[k]
        pct = (cnt / total_elements) * 100
        dist_lines.append(f" Q = {k:>3.0f} : {cnt:>15d} 개 ({pct:>6.2f}%)")
        
    dist_lines.append("-" * 40)
    dist_lines.append(f" Total Elements : {total_elements}")
    dist_lines.append("="*40)
    dist_lines.append("")
    
    return dist_lines


# =========================================================
# 6. Perplexity
# =========================================================
def compute_perplexity(model, testenc, dev):
    print("\nEvaluating perplexity ...")

    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough tokens: tokens={testenc.numel()}, seqlen={model.seqlen}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
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
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir", type=str, default="quotrem_ppl_results_v2")
    parser.add_argument("--seed", type=int, default=42)

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="one", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument("--target_modules", type=str, default="fc1",
                        help="e.g. fc1 or self_attn.q_proj,fc1")
    parser.add_argument("--custom_layer_indices", type=str, default=None,
                        help="e.g. 0,2,4")

    # weight quant
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--weight_quant_mode", type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)

    # QR bits
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument("--r_bits", type=int, default=4)
    parser.add_argument("--r_group_size", type=int, default=128)

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
    lines.append("[Weight Quant]")
    lines.append(f"enable_weight_quant: {args.enable_weight_quant}")
    lines.append(f"weight_bits        : {args.weight_bits}")
    lines.append(f"weight_quant_mode  : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis     : {args.weight_ch_axis}")
    lines.append(f"weight_group_size  : {args.weight_group_size}")
    lines.append("")
    lines.append("[Adaptive QR]")
    lines.append(f"q_bits             : {args.q_bits}")
    lines.append(f"r_bits             : {args.r_bits}")
    lines.append(f"r_group_size       : {args.r_group_size}")
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

    # Baseline probe
    if args.do_probe_compare:
        print("[3] Collecting baseline probe outputs/logits...")
        ref_module_outputs, ref_logits = collect_module_outputs(
            model, tokenizer, get_probe_text(),
            target_layer_indices, module_names,
        )
    else:
        ref_module_outputs, ref_logits = None, None

    # Baseline PPL
    print("[4] Computing baseline PPL...")
    input_device = get_model_main_device(model)
    baseline_ppl = compute_perplexity(model, testenc, input_device)

    lines.append("[Baseline]")
    lines.append(f"baseline_ppl : {baseline_ppl:.8f}")
    lines.append("")

    # Replace modules
    print("[5] Replacing modules with QuotRemLinear (adaptive base)...")
    replaced_names = replace_modules_with_quotrem_linear(
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
        r_group_size=args.r_group_size,
    )

    lines.append("[Replacement]")
    lines.append(f"replaced_count : {len(replaced_names)}")
    for name in replaced_names:
        lines.append(f"  - {name}")
    lines.append("")

    # Modified probe
    if args.do_probe_compare:
        print("[6] Collecting modified probe outputs/logits...")
        new_module_outputs, new_logits = collect_module_outputs(
            model, tokenizer, get_probe_text(),
            target_layer_indices, module_names,
        )
        lines.extend(compare_logits(ref_logits, new_logits))
        lines.append("[Module Output Comparison]")
        lines.extend(compare_tensor_dicts(ref_module_outputs, new_module_outputs))

    # Modified PPL
    print("[7] Computing modified PPL...")
    modified_ppl = compute_perplexity(model, testenc, input_device)

    q_lines = get_q_distribution(model)
    for q_line in q_lines:
        print(q_line)
    lines.extend(q_lines)

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
