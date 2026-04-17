import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# ver 2 + q distribution 수집 + 코드 정리
# base가 2의 승수 -> CIM SHIFT 연산 간단화는 간단하지만, PPL 살짝 손해 값 선택 범위는 (2,~64)
# R scale은 자기만의 타이트하게 결정, 몫은 분포 출력은 x 
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
#   각 group의 max_abs를 기반으로 qr_base를 {2,4,8,16,32,64} 중에서 선택 (조건 A)
#   조건: max_abs / qr_base <= q_max 를 만족하는 가장 작은 2의 승수
#     → Q overflow 없음, R 정밀도 최대화
#
#   qr_base가 2의 승수이므로:
#     - 별도 float scale 저장 불필요 (3-bit index로 충분)
#     - CIM에서 shift 연산으로 처리 가능
#
#   R scale은 실제 R값의 max_abs 기반으로 독립적으로 결정 (tight scale):
#     scale_r = actual_max_abs_r / r_max  (base와 무관)
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

    def _adaptive_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Power-of-2 adaptive base QR quantization (조건 A).

        q_bits >= 2:
          {2,4,8,16,32,64} 중 max_abs/b <= q_max 를 만족하는 가장 작은 b 선택.
          Q ∈ [-q_max, q_max]

        q_bits == 1:
          Q ∈ {0, 1} (unsigned binary)
          base = {2,4,8,16,32,64} 중 max_abs <= b 를 만족하는 가장 작은 값
          Q = 1 if x >= base/2 else 0
          R = x - base * Q

        R scale은 실제 R 분포의 max_abs로 독립적으로 결정.
        """
        r_max_val = (2 ** (self.r_bits - 1)) - 1
        r_min_val = -r_max_val - 1
        eps = 1e-8
        gs  = self.r_group_size

        x_fp       = x.float()
        orig_shape = x_fp.shape
        H          = orig_shape[-1]
        assert H % gs == 0, f"Last dim {H} not divisible by r_group_size {gs}"

        xg = x_fp.reshape(orig_shape[:-1] + (H // gs, gs))  # [..., G, gs]

        max_abs = xg.abs().amax(dim=-1, keepdim=True).clamp_min(eps)  # [..., G, 1]

        if self.q_bits == 1:
            # ceiling method
            #base = torch.full_like(max_abs, 64.0)
            #base = torch.where(max_abs <= 32.0, torch.full_like(max_abs, 32.0), base)
            #base = torch.where(max_abs <= 16.0, torch.full_like(max_abs, 16.0), base)
            #base = torch.where(max_abs <=  8.0, torch.full_like(max_abs,  8.0), base)
            #base = torch.where(max_abs <=  4.0, torch.full_like(max_abs,  4.0), base)
            #base = torch.where(max_abs <=  2.0, torch.full_like(max_abs,  2.0), base)
            
            
            #1-bit Q: {0, 1}, max_abs에 가장 가까운 2의 제곱 선택
            log2_max = torch.log2(max_abs)
            log2_floor = torch.floor(log2_max)  # floor 값과 ceil 값 두개 모두 계산
            log2_ceil = torch.ceil(log2_max)

            base_floor = torch.pow(2.0, log2_floor)     # 마찬가지로 base 값도 두종류 
            base_ceil = torch.pow(2.0, log2_ceil)
    
            dist_floor = torch.abs(max_abs - base_floor)     # dist 계산해서 더 가까운 걸로 결정
            dist_ceil = torch.abs(base_ceil - max_abs)
            base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)
            base = base.clamp(min=2.0, max=64.0)

            # Hardware 방식: dist 이용
            #candidates = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0, 64.0], device=xg.device)
            #cand = candidates.view(*([1] * (max_abs.dim() - 1)), -1)   # broadcast shape
            #dist = torch.abs(max_abs.squeeze(-1).unsqueeze(-1) - cand)
            #base = candidates[dist.argmin(dim=-1)].unsqueeze(-1)            

            # signed base: group 내 abs 최대 원소의 부호를 base에 흡수
            # Q ∈ {0,1}, 복원식: x = base_signed * Q + r (sign 의존 없음)
#
            q = (xg.abs() >= base / 2.0).float()
            #base_sign   = torch.sign(
            #    xg.gather(-1, xg.abs().argmax(dim=-1, keepdim=True))
            #)  # [..., G, 1]
            #base = base * base_sign
            #r = xg - base * q

            # negative 고려 x
            q = (xg >= base / 2.0).float()
            r = xg - base * q

            # --- [Folded 방식] ---
            # ceiling base 필수 (nearest 사용 시 base < max_abs 이면 → 복원 공식 깨짐)
            #base_ceiling = torch.full_like(max_abs, 64.0)
            #base_ceiling = torch.where(max_abs <= 32.0, torch.full_like(max_abs, 32.0), base_ceiling)
            #base_ceiling = torch.where(max_abs <= 16.0, torch.full_like(max_abs, 16.0), base_ceiling)
            #base_ceiling = torch.where(max_abs <=  8.0, torch.full_like(max_abs,  8.0), base_ceiling)
            #base_ceiling = torch.where(max_abs <=  4.0, torch.full_like(max_abs,  4.0), base_ceiling)
            #base_ceiling = torch.where(max_abs <=  2.0, torch.full_like(max_abs,  2.0), base_ceiling)
            #base = base_ceiling
            ##
            #xg_abs = xg.abs()
            #q = (xg_abs >= base / 2.0).float()
            #r = torch.sign(xg) * (xg_abs - base * q)
            # --- [Folded 방식 끝] ---


        else:
            q_max_val = (2 ** (self.q_bits - 1)) - 1
            threshold = max_abs / q_max_val  # qr_base >= threshold 조건

            # {2,4,8,16,32,64} 중 threshold를 만족하는 가장 작은 2의 승수 선택
            base = torch.full_like(threshold, 64.0)
            base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
            base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
            base = torch.where(threshold <=  8.0, torch.full_like(threshold,  8.0), base)
            base = torch.where(threshold <=  4.0, torch.full_like(threshold,  4.0), base)
            base = torch.where(threshold <=  2.0, torch.full_like(threshold,  2.0), base)

            q = torch.round(xg / base).clamp(-q_max_val, q_max_val)
            r = xg - base * q

        # R quantize: 실제 R max 기반 독립 scale (tight)
        max_abs_r = r.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale_r   = max_abs_r / r_max_val
        r_q       = torch.round(r / scale_r).clamp(r_min_val, r_max_val)
        r_dq      = r_q * scale_r

        # reconstruct
        # q_bits==1 signed base: x = base_signed * Q + r_dq
        # q_bits>=2 standard:    x = base * Q + r_dq
        if self.q_bits == 1:
            #base_sign = torch.sign(xg.gather(-1, xg.abs().argmax(dim=-1, keepdim=True)))
            #sign_safe = torch.where(r_dq != 0, torch.sign(r_dq), -base_sign)
            # x_recon = r_dq - q * sign_safe * base = q_scaled + r_dq
            # q_scaled: Q가 weight에 곱해질 때의 실질적 기여값 (부호 포함)
            #q_scaled = -q * sign_safe * base                  # [..., G, gs]
            q_scaled = q * base # signed base method
        else:
            q_scaled = base * q                               # [..., G, gs]

        # (q_scaled, r_dq) 튜플 반환 — forward에서 각각 matmul
        return (
            q_scaled.reshape_as(x_fp).to(x.dtype),
            r_dq.reshape_as(x_fp).to(x.dtype),
        )

    def forward(self, x):
        q_scaled, r_dq = self._adaptive_base_forward(x)
        # 몫 기여: Q_scaled @ W  (= Q * base * W, 부호 포함)
        q_out = F.linear(q_scaled, self.weight, None)
        # 나머지 기여: R_dq @ W
        r_out = F.linear(r_dq, self.weight, None)
        out = q_out + r_out
        if self.bias is not None:
            out = out + self.bias
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
