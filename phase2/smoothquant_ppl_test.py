import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# SmoothQuant 아이디어를 OPT-6.7b 에 적용해 PPL 측정
# smooth.py 원본 충실히 이식:
#   - per-channel act abs max 수집 (calibration)
#   - smooth_ln_fcs: scales 를 LayerNorm 과 Linear weight 에 흡수
#   - OPT smoothing 쌍: self_attn_layer_norm→[q,k,v], final_layer_norm→[fc1]
# weight quantization: fake_quant_symmetric (uniform, group 방식)
# activation quantization (논문 Table 2 기준):
#   O1 (per_token  dynamic): 토큰마다 Δ[t] = max(|X[t,:]|)/(2^7-1)  ← 가장 세밀
#   O2 (per_tensor dynamic): 텐서 전체 Δ  = max(|X|)    /(2^7-1)  ← 가장 단순

import argparse
import functools
import math
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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_main_device(model):
    return model.model.decoder.embed_tokens.weight.device


def save_txt(lines, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


# =========================================================
# 1. Dataset
# =========================================================
def load_wikitext2_testenc(model_id, split="test"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids
    return tokenizer, testenc


def get_calib_dataloader(model_id, nsamples, seed, seqlen):
    """datautils.py get_wikitext2 와 동일 방식"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1")
    trainenc  = tokenizer("\n\n".join(dataset["train"]["text"]), return_tensors="pt")
    random.seed(seed)
    loader = []
    for _ in range(nsamples):
        i   = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader


# =========================================================
# 2. SmoothQuant calibration — per-channel abs max 수집
# =========================================================
@torch.no_grad()
def collect_ln_input_max(model, dataloader, device):
    """
    q_proj / k_proj / v_proj / fc1 입력의 per-channel abs max 수집.
    (== 각 레이어의 self_attn_layer_norm / final_layer_norm 출력과 동일)

    반환: dict  key="layer{i}.{mname}"  value=Tensor[in_features]
    """
    act_max = {}

    def make_hook(key):
        def hook(m, x, y):
            inp  = x[0] if isinstance(x, tuple) else x
            vals = inp.view(-1, inp.shape[-1]).abs().max(dim=0).values.float().cpu()
            if key in act_max:
                act_max[key] = torch.max(act_max[key], vals)
            else:
                act_max[key] = vals
        return hook

    hooks = []
    for i, layer in enumerate(model.model.decoder.layers):
        for mname, mod in [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("fc1",    layer.fc1),
        ]:
            hooks.append(mod.register_forward_hook(make_hook(f"layer{i}.{mname}")))

    # ── Catcher 패턴으로 layer-by-layer 통과 ──────────────────────
    nsamples = len(dataloader)
    layers   = model.model.decoder.layers

    model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps  = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.squeeze(0)
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    torch.cuda.empty_cache()

    outs           = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="Calibration (smooth)"):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    for h in hooks:
        h.remove()

    return act_max


# =========================================================
# 3. SmoothQuant 적용 (smooth.py 충실히 이식)
# =========================================================
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """
    smooth.py smooth_ln_fcs 충실히 이식.

    scales[i] = act_scales[i]^alpha / weight_max[i]^(1-alpha)

    - LayerNorm weight (+ bias) 를 scales 로 나눔  → activation 난이도 낮춤
    - 각 Linear weight 를 scales 로 곱함           → weight 쪽으로 이동
    forward 시 추가 연산 없이 smoothing 효과 발생.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel(), (
            f"Dim mismatch: ln={ln.weight.numel()}, "
            f"fc.in={fc.in_features}, act={act_scales.numel()}"
        )

    device = fcs[0].weight.device
    dtype  = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # per input-channel weight max (모든 fc 를 concat 후 max)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True).values for fc in fcs], dim=0
    ).max(dim=0).values.clamp(min=1e-5)

    scales = (
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
    ).clamp(min=1e-5).to(device=device, dtype=dtype)

    # LayerNorm 에 흡수
    ln.weight.data.div_(scales)
    if ln.bias is not None:
        ln.bias.data.div_(scales)

    # Linear weight 에 흡수
    for fc in fcs:
        fc.weight.data.mul_(scales.view(1, -1))


@torch.no_grad()
def apply_smoothquant_opt(model, act_max, alpha=0.5):
    """
    OPT 모델에 SmoothQuant 적용.
    smoothing 쌍:
      self_attn_layer_norm  →  [q_proj, k_proj, v_proj]
      final_layer_norm      →  [fc1]
    """
    n_applied = 0
    for i, layer in enumerate(model.model.decoder.layers):
        # 1. self_attn_layer_norm → q, k, v
        q_key = f"layer{i}.q_proj"
        if q_key in act_max:
            smooth_ln_fcs(
                layer.self_attn_layer_norm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                act_max[q_key],
                alpha=alpha,
            )
            n_applied += 1

        # 2. final_layer_norm → fc1
        fc1_key = f"layer{i}.fc1"
        if fc1_key in act_max:
            smooth_ln_fcs(
                layer.final_layer_norm,
                [layer.fc1],
                act_max[fc1_key],
                alpha=alpha,
            )
            n_applied += 1

    print(f"  → SmoothQuant: {n_applied} LayerNorm→Linear 쌍에 적용 완료")
    return n_applied


# =========================================================
# 4. Activation quantization (논문 Eq.1 기반)
# =========================================================
def fake_quant_act(x: torch.Tensor, n_bits: int = 8, mode: str = "per_token",
                   eps: float = 1e-8) -> torch.Tensor:
    """
    논문 Eq.(1) symmetric activation quantization (fake-quant).

    Δ = max(|X|) / (2^(N-1) - 1)
    X_q = round(X / Δ) · Δ     (floating-point 시뮬레이션)

    mode="per_token"  [논문 O1 setting]:
        각 토큰(token position)마다 독립적인 Δ 계산.
        X shape: [..., seq_len, hidden_dim]
        Δ shape: [..., seq_len, 1]   ← 채널 방향으로 amax

        왜 per-token이 좋은가:
          outlier 는 특정 채널에 고정적으로 나타나지만,
          그 크기는 토큰마다 조금씩 다르다.
          per-token 은 각 토큰의 실제 범위를 반영해
          per-tensor 보다 유효 bit 수가 높다.

    mode="per_tensor" [논문 O2 setting]:
        텐서 전체에 단일 Δ. 가장 구현이 단순하고 hardware 친화적.
        outlier 가 하나라도 있으면 모든 값의 유효 bit 수가 낮아진다.
    """
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    x_fp = x.float()

    if mode == "per_token":
        # 마지막 차원(hidden_dim) 방향으로 max → 토큰별 scale
        max_abs = x_fp.abs().amax(dim=-1, keepdim=True)   # [..., seq, 1]
        scale   = (max_abs / qmax).clamp_min(eps)
    elif mode == "per_tensor":
        max_abs = x_fp.abs().max()
        scale   = (max_abs / qmax).clamp_min(eps)
    else:
        raise ValueError(f"Unsupported act_quant_mode: {mode}")

    return (torch.round(x_fp / scale).clamp(qmin, qmax) * scale).to(x.dtype)


class SQLinear(nn.Module):
    """
    SmoothQuant W(n)A(n) Linear.

    forward 에서 입력 activation 을 먼저 fake_quant_act 로 양자화한 뒤
    linear 연산을 수행한다.

    weight 는 이미 apply_weight_quant_opt 로 fake-quant 된 상태로 받는다.
    (weight 자체를 다시 양자화하지 않음 — 중복 방지)
    """
    def __init__(self, base_linear: nn.Linear, act_bits: int = 8,
                 act_mode: str = "per_token"):
        super().__init__()
        self.act_bits = act_bits
        self.act_mode = act_mode
        self.weight = nn.Parameter(base_linear.weight.detach().clone(),
                                   requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(),
                                     requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_quant_act(x, n_bits=self.act_bits, mode=self.act_mode)
        return F.linear(x_q, self.weight, self.bias)


@torch.no_grad()
def apply_act_quant_opt(model, n_bits: int, mode: str, target_modules: list):
    """
    target_modules 에 해당하는 모든 Linear 를 SQLinear 로 교체.
    반드시 apply_weight_quant_opt 이후에 호출해야
    weight 가 이미 fake-quant 된 상태로 SQLinear 에 복사된다.
    """
    name_to_getter = {
        "self_attn.q_proj":   lambda l: (l.self_attn, "q_proj"),
        "self_attn.k_proj":   lambda l: (l.self_attn, "k_proj"),
        "self_attn.v_proj":   lambda l: (l.self_attn, "v_proj"),
        "self_attn.out_proj": lambda l: (l.self_attn, "out_proj"),
        "fc1": lambda l: (l, "fc1"),
        "fc2": lambda l: (l, "fc2"),
    }
    count = 0
    for layer in model.model.decoder.layers:
        for mname in target_modules:
            parent, attr = name_to_getter[mname](layer)
            old = getattr(parent, attr)
            setattr(parent, attr, SQLinear(old, act_bits=n_bits, act_mode=mode))
            count += 1
    print(f"  → Act quant ({n_bits}b {mode}): {count} Linear → SQLinear")


# =========================================================
# 6. Weight quantization (fake_quant_symmetric, ver2와 동일)
# =========================================================
def fake_quant_symmetric(x, n_bits=8, mode="tensor", ch_axis=-1, group_size=None, eps=1e-8):
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    x_fp = x.float()

    if mode == "tensor":
        max_abs = x_fp.abs().max()
        scale   = (max_abs / qmax).clamp_min(eps)
        return (torch.round(x_fp / scale).clamp(qmin, qmax) * scale).to(x.dtype)

    elif mode == "per_channel":
        axis      = ch_axis if ch_axis >= 0 else x_fp.dim() + ch_axis
        rdims     = tuple(d for d in range(x_fp.dim()) if d != axis)
        max_abs   = x_fp.abs().amax(dim=rdims, keepdim=True)
        scale     = (max_abs / qmax).clamp_min(eps)
        return (torch.round(x_fp / scale).clamp(qmin, qmax) * scale).to(x.dtype)

    elif mode == "group":
        num_groups = x_fp.shape[-1] // group_size
        xg         = x_fp.reshape(x_fp.shape[:-1] + (num_groups, group_size))
        max_abs    = xg.abs().amax(dim=-1, keepdim=True)
        scale      = (max_abs / qmax).clamp_min(eps)
        return (torch.round(xg / scale).clamp(qmin, qmax) * scale).reshape_as(x_fp).to(x.dtype)

    raise ValueError(f"Unsupported mode: {mode}")


@torch.no_grad()
def apply_weight_quant_opt(model, n_bits, mode, group_size, target_modules):
    """
    모델의 target_modules 에 해당하는 모든 Linear weight 를 fake-quant.
    """
    name_to_attr = {
        "self_attn.q_proj":   lambda l: l.self_attn.q_proj,
        "self_attn.k_proj":   lambda l: l.self_attn.k_proj,
        "self_attn.v_proj":   lambda l: l.self_attn.v_proj,
        "self_attn.out_proj": lambda l: l.self_attn.out_proj,
        "fc1": lambda l: l.fc1,
        "fc2": lambda l: l.fc2,
    }
    count = 0
    for layer in model.model.decoder.layers:
        for mname in target_modules:
            m = name_to_attr[mname](layer)
            m.weight.data = fake_quant_symmetric(
                m.weight.data,
                n_bits=n_bits, mode=mode, group_size=group_size,
            )
            count += 1
    print(f"  → Weight quant ({n_bits}b {mode}): {count} Linear layers")


# =========================================================
# 7. Perplexity (ver2와 동일)
# =========================================================
def compute_perplexity(model, testenc, dev):
    print("\nEvaluating perplexity ...")
    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens: {testenc.numel()}, seqlen={model.seqlen}")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls      = []
    loss_fct  = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch       = testenc[:, i * model.seqlen:(i + 1) * model.seqlen].to(dev)
            lm_logits   = model(batch, use_cache=False).logits
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
# 8. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="SmoothQuant PPL test on OPT")
    parser.add_argument("--model_id",         type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir",       type=str, default="results_smoothquant")
    parser.add_argument("--seed",             type=int, default=42)
    # SmoothQuant
    parser.add_argument("--smooth_alpha",     type=float, default=0.5,
                        help="migration strength α (0=activation 쪽, 1=weight 쪽)")
    parser.add_argument("--n_calib_samples",  type=int, default=128)
    parser.add_argument("--calib_seqlen",     type=int, default=2048)
    # Weight quant
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits",      type=int, default=8)
    parser.add_argument("--weight_quant_mode", type=str, default="per_tensor",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_group_size", type=int, default=128)
    # Activation quant (논문 Table 2)
    parser.add_argument("--enable_act_quant", action="store_true",
                        help="activation 을 fake-quant 로 양자화 (SQLinear 교체)")
    parser.add_argument("--act_bits",         type=int, default=8,
                        help="activation quantization bit width (논문: 8)")
    parser.add_argument("--act_quant_mode",   type=str, default="per_token",
                        choices=["per_token", "per_tensor"],
                        help="per_token=O1(논문 권장), per_tensor=O2")
    parser.add_argument("--target_modules",   type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,"
                                "self_attn.out_proj,fc1,fc2")
    parser.add_argument("--eval_split",       type=str, default="test")
    args = parser.parse_args()

    set_seed(args.seed)
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    lines = [
        "[Config]",
        f"model_id          : {args.model_id}",
        f"smooth_alpha      : {args.smooth_alpha}",
        f"n_calib_samples   : {args.n_calib_samples}",
        f"calib_seqlen      : {args.calib_seqlen}",
        "",
        "[Weight Quant]",
        f"enable_weight_quant: {args.enable_weight_quant}",
        f"weight_bits        : {args.weight_bits}",
        f"weight_quant_mode  : {args.weight_quant_mode}",
        f"weight_group_size  : {args.weight_group_size}",
        "",
        "[Act Quant]",
        f"enable_act_quant   : {args.enable_act_quant}",
        f"act_bits           : {args.act_bits}",
        f"act_quant_mode     : {args.act_quant_mode} "
        f"({'O1-per-token' if args.act_quant_mode == 'per_token' else 'O2-per-tensor'})",
        "",
    ]

    print("[1] Loading data ...")
    _, testenc = load_wikitext2_testenc(args.model_id, split=args.eval_split)

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

    # ── Baseline PPL ─────────────────────────────────────────────
    print("[3] Baseline PPL ...")
    baseline_ppl = compute_perplexity(model, testenc, dev)
    lines += ["[Baseline]", f"baseline_ppl : {baseline_ppl:.8f}", ""]
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ── SmoothQuant calibration ───────────────────────────────────
    print("[4] Calibration (collecting act max) ...")
    calib_loader = get_calib_dataloader(
        args.model_id, args.n_calib_samples, args.seed, args.calib_seqlen
    )
    act_max = collect_ln_input_max(model, calib_loader, dev)

    # calibration 후 model 을 device 로 복원
    model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    for i in range(len(model.model.decoder.layers)):
        model.model.decoder.layers[i] = model.model.decoder.layers[i].to(dev)

    # ── SmoothQuant 적용 ─────────────────────────────────────────
    print(f"[5] Applying SmoothQuant (alpha={args.smooth_alpha}) ...")
    n_applied = apply_smoothquant_opt(model, act_max, alpha=args.smooth_alpha)
    lines += ["[SmoothQuant]", f"n_applied : {n_applied}", ""]

    # ── (선택) weight quantization ────────────────────────────────
    # 반드시 act quant (SQLinear 교체) 보다 먼저 실행해야
    # SQLinear 생성 시 이미 fake-quant 된 weight 를 복사함
    if args.enable_weight_quant:
        print(f"[6] Weight quantization ({args.weight_bits}b {args.weight_quant_mode}) ...")
        apply_weight_quant_opt(
            model, args.weight_bits, args.weight_quant_mode,
            args.weight_group_size, target_modules,
        )

    # ── (선택) activation quantization ───────────────────────────
    # Linear → SQLinear 교체: forward 시 입력 activation 을 fake-quant
    if args.enable_act_quant:
        print(f"[6b] Activation quantization ({args.act_bits}b {args.act_quant_mode}) ...")
        apply_act_quant_opt(model, args.act_bits, args.act_quant_mode, target_modules)

    # ── PPL 측정 ─────────────────────────────────────────────────
    print("[7] SmoothQuant PPL ...")
    sq_ppl   = compute_perplexity(model, testenc, dev)
    ppl_diff = sq_ppl - baseline_ppl

    lines += [
        "[SmoothQuant PPL]",
        f"sq_ppl      : {sq_ppl:.8f}",
        f"ppl_diff    : {ppl_diff:.12e}",
        "",
    ]
    print(f"SmoothQuant PPL : {sq_ppl:.4f}  (diff {ppl_diff:+.4f})")

    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] {Path(args.output_dir) / 'summary.txt'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
