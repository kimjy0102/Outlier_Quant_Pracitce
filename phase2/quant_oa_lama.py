import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# OA-LAMA 방식 activation quantization을 ver2 환경에서 실행
# - activation: exponent-based mixed 3-bit/4-bit (OA-LAMA quant.py 그대로)
# - per-group dynamic scale + 첫 배치에서 global clamp 하한선 결정 (Quantizer.init 패턴)
# - channel reordering: outlier.py get_act_stats_opt / reorder_tensor 충실히 이식
# - weight: 선택적 (--enable_weight_quant, fake_quant_symmetric 사용)
# - PPL: ver2와 동일한 방식

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
        for line in lines:
            f.write(str(line) + "\n")

def resolve_axis(dim, axis):
    if axis < 0:
        axis = dim + axis
    return axis


# =========================================================
# 1. Dataset (PPL 평가용 - test split)
# =========================================================
def load_wikitext2_testenc(model_id, split="test"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids
    return tokenizer, testenc


# =========================================================
# 1b. Calibration dataloader (outlier.py / datautils.py get_wikitext2 충실히 이식)
# =========================================================
def get_calib_dataloader(model_id, nsamples, seed, seqlen):
    """
    datautils.py get_wikitext2 와 동일한 방식으로 wikitext2 train split에서
    nsamples개의 (inp, tar) 쌍을 랜덤 샘플링해 반환.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    dataset   = load_dataset('wikitext', 'wikitext-2-raw-v1')
    traindata = dataset['train']
    trainenc  = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


# =========================================================
# 2. OA-LAMA activation quantization (quant.py 충실히 이식)
# =========================================================
@torch.no_grad()
def oa_lama_quantize(x: torch.Tensor, scale: int, group_size: int, threshold: int = 0) -> torch.Tensor:
    """
    OA-LAMA의 quantize_tensor(exponential=True) 이식.

    exponent 추출: FP16 기준 (view(int16) >> 10)  - FP16: S1E5M10, >> 10 = exponent
    산술 연산:     float32 기준 (FP16 overflow 방지: 2^17 > FP16 max 65504)

    threshold==0 (normal):
      group 내 앞 4채널 → 3-bit, 나머지 12채널 → 4-bit

    threshold>0 (outlier group, channel reorder 후 사용):
      뒤 threshold개 group → 앞 8채널 3-bit, 중간 7채널 4-bit, 마지막 1채널 8-bit
      앞 (group_num - threshold)개 group → normal 처리
    """
    saved_shape = x.shape
    assert saved_shape[-1] % group_size == 0
    group_num = saved_shape[-1] // group_size

    # FP16: exponent 추출 전용
    x_fp16 = x.half().view(-1, group_num, group_size)
    # float32: 산술 연산 전용 (overflow 없음)
    x_f32  = x.float().view(-1, group_num, group_size)

    def get_shift(fp16_slice):
        """FP16 슬라이스 [..., group, gs] → per-group float32 shift [..., group, 1]"""
        gmax = fp16_slice.abs().amax(dim=-1)                        # [..., group]
        exp  = 13 - (gmax.view(torch.int16) >> 10)                  # FP16 exponent 추출
        return exp.clamp(min=scale, max=scale + 15).float().unsqueeze(-1)  # [..., group, 1]

    # ── outlier group (뒤 threshold개) ──────────────────────────
    if threshold > 0:
        # 원본(quant.py L111-113)과 동일:
        # 마지막 채널을 0.25배 축소한 뒤 그 값 포함하여 grouped_max 계산 → shift 결정
        # (마지막 채널이 큰 outlier일 때 shift가 과도하게 커지는 것을 방지)
        x_fp16_ol = x_fp16[:, -threshold:].clone()
        x_fp16_ol[..., -1] = (x_fp16_ol[..., -1].float() * 0.25).half()
        shift = get_shift(x_fp16_ol)  # [B, threshold, 1]

        # 앞 8채널: 3-bit, 중간 7채널: 4-bit
        seg = x_f32[:, -threshold:, :-1].clone()
        seg[..., :8] *= (2.0 ** (shift + 3))
        seg[..., 8:] *= (2.0 ** (shift + 4))
        seg = seg.round()
        seg[..., :8] = seg[..., :8].clamp(-4, 3)    # 3-bit: [-4, 3]
        seg[..., 8:] = seg[..., 8:].clamp(-8, 7)    # 4-bit: [-8, 7]
        seg[..., :8] *= (2.0 ** -(shift + 3))
        seg[..., 8:] *= (2.0 ** -(shift + 4))
        x_f32[:, -threshold:, :-1] = seg

        # 마지막 1채널: 8-bit (/4 축소 후 shift+8 적용, 역변환 shift+6)
        last = x_f32[:, -threshold:, -1:].clone() / 4.0
        last *= (2.0 ** (shift + 8))
        last = last.round().clamp(-128, 127)
        last *= (2.0 ** -(shift + 6))  # -(shift+8) + log2(4) = -(shift+6)
        x_f32[:, -threshold:, -1:] = last

    # ── normal group (앞 group_num - threshold개) ────────────────
    if threshold < group_num:
        sl  = slice(None) if threshold == 0 else slice(None, -threshold)
        seg = x_f32[:, sl, :].clone()
        shift = get_shift(x_fp16[:, sl, :])  # [B, normal_groups, 1]

        # 앞 4채널: 3-bit, 나머지: 4-bit
        seg[..., :4] *= (2.0 ** (shift + 3))
        seg[..., 4:] *= (2.0 ** (shift + 4))
        seg = seg.round()
        seg[..., :4] = seg[..., :4].clamp(-4, 3)    # 3-bit: [-4, 3]
        seg[..., 4:] = seg[..., 4:].clamp(-8, 7)    # 4-bit: [-8, 7]
        seg[..., :4] *= (2.0 ** -(shift + 3))
        seg[..., 4:] *= (2.0 ** -(shift + 4))
        x_f32[:, sl, :] = seg

    return x_f32.view(saved_shape).to(x.dtype)


# =========================================================
# 2b. Channel reordering (outlier.py 충실히 이식)
# =========================================================
@torch.no_grad()
def get_act_stats_opt(model, dataloader, device, metric='hessian'):
    """
    outlier.py get_act_stats_opt 충실히 이식.
    모든 nn.Linear의 input/output에 hook을 등록하고
    layer-by-layer로 calibration data를 통과시켜 per-channel 통계 수집.
    """
    nsamples  = len(dataloader)
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            tensorH        = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H      = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        else:
            # abs mean: outlier.py에서 symmetric quant의 absmax 사용 이유로 abs 사용
            comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input",  x)
        stat_tensor(name + ".output", y)

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps  = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.squeeze(0)
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    assert cache['i'] == nsamples, "Captured samples should be equal to nsamples"

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs            = torch.zeros_like(inps)
    attention_mask  = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="Collecting act stats"):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    for h in hooks:
        h.remove()

    return act_scales


def reorder_tensor(tensor: torch.Tensor, group_size: int):
    """
    outlier.py reorder_tensor 충실히 이식.
    tensor: 1-D per-channel scale 벡터
    반환: (sorted_index, threshold)
      - sorted_index: 소→대 정렬 후 view(group_size,-1).T.reshape(-1) 인터리빙
      - threshold: mean+3*std 초과 채널 수 (최대 group_num-1)
    """
    assert tensor.dim() == 1, "reorder_tensor: input must be 1-D"
    _, sorted_index = torch.sort(tensor, descending=False)  # 작은 값 먼저 → outlier 뒤로
    group_num   = tensor.shape[0] // group_size
    # interleaving: 각 그룹이 모든 tier에서 1개씩 갖도록
    sorted_index = sorted_index.view(group_size, -1).transpose(-2, -1).reshape(-1)
    try:
        threshold = (
            (tensor > (tensor.mean() + 3 * tensor.std()))
            .unique(return_counts=True)[1][1]
            .clamp(max=group_num - 1)
        )
    except IndexError:
        # outlier가 없는 경우 (모든 채널이 mean+3*std 이하)
        threshold = torch.tensor(0)
    return sorted_index, threshold


def compute_reorder_indices(act_scales, target_layer_indices, module_names, group_size):
    """
    각 (layer_idx, module_name) 쌍의 input scale로부터 reorder_index, threshold 계산.
    act_scales key 형식: "decoder.layers.{i}.{module_name}.input"

    reorder_tensor의 threshold는 outlier 채널 수 (채널 단위).
    oa_lama_quantize의 threshold는 outlier 그룹 수 (그룹 단위).
    → ceil(threshold_channels / group_size) 로 변환.
    """
    reorder_indices = {}
    for layer_idx in target_layer_indices:
        for module_name in module_names:
            key = f"decoder.layers.{layer_idx}.{module_name}.input"
            if key not in act_scales:
                print(f"  [Warning] key not found in act_scales: {key}")
                continue
            sorted_index, threshold_channels = reorder_tensor(act_scales[key], group_size)
            # 채널 수 → 그룹 수 변환
            threshold_groups = math.ceil(int(threshold_channels.item()) / group_size)
            reorder_indices[(layer_idx, module_name)] = (
                sorted_index.cpu(),
                threshold_groups,
            )
    return reorder_indices


# =========================================================
# 3. Weight fake-quant (ver2와 동일)
# =========================================================
def fake_quant_symmetric(x, n_bits=8, mode="tensor", ch_axis=-1, group_size=None, eps=1e-8):
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
        num_groups = x_fp.shape[-1] // group_size
        xg = x_fp.reshape(x_fp.shape[:-1] + (num_groups, group_size))
        max_abs = xg.abs().amax(dim=-1, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)
        return (torch.round(xg / scale).clamp(qmin, qmax) * scale).reshape_as(x_fp).to(x.dtype)

    raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# 4. OALAMALinear
# =========================================================
class OALAMALinear(nn.Module):
    """
    activation: OA-LAMA 방식 (exponent-based mixed 3/4-bit, per-group dynamic)
                첫 번째 forward에서 global scale 하한선 결정 (Quantizer.init 패턴)
    reorder_index가 주어지면:
      - weight 열(column)을 미리 reorder (qLinearLayer.reorder 와 동일)
      - forward에서 x를 동일 인덱스로 reorder 후 양자화
    threshold: outlier group 수 (reorder 시 calibration에서 자동 결정)
    weight: 선택적으로 fake_quant_symmetric 적용
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        act_group_size: int = 16,
        act_threshold: int = 0,
        reorder_index: torch.Tensor = None,   # 1-D long tensor or None
        enable_weight_quant: bool = False,
        weight_group_size: int = 16,
        debug_name: str = "",
    ):
        super().__init__()
        self.act_group_size = act_group_size
        self.act_threshold  = act_threshold
        self.debug_name     = debug_name

        # OA-LAMA Quantizer.init 패턴
        self.init  = True
        self.scale = 0  # global shift 하한선 (첫 배치에서 결정)

        # ── reorder index ────────────────────────────────────────
        if reorder_index is not None:
            self.register_buffer('reorder_index', reorder_index)
        else:
            self.register_buffer('reorder_index', None)

        # ── weight 준비 ──────────────────────────────────────────
        weight = base_linear.weight.detach().clone()

        # reorder: weight 열(column) 재배치 (qLinearLayer.reorder dim=1)
        if reorder_index is not None:
            ri     = reorder_index.to(weight.device)
            weight = torch.index_select(weight, 1, ri)

        if enable_weight_quant:
            # OA-LAMA 방식: activation과 동일한 exponent-based mixed 3/4-bit 그룹 양자화
            # reorder 후 마지막 act_threshold개 그룹이 outlier 그룹 → activation과 동일 threshold 사용
            # weight의 global shift 하한선을 정적으로 계산 (activation은 첫 배치에서 동적 결정)
            w_scale = int(13 - (weight.abs().max().half().view(torch.int16).item() >> 10))
            weight = oa_lama_quantize(
                weight,
                scale=w_scale,
                group_size=weight_group_size,
                threshold=act_threshold,
            )

        self.weight = nn.Parameter(weight, requires_grad=False)

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.init:
            # 첫 배치의 global max에서 shift 하한선 결정 (OA-LAMA Quantizer.forward 동일)
            self.scale = int(13 - (x.abs().max().half().view(torch.int16).item() >> 10))
            self.init  = False

        # channel reorder 적용 (reorder_model_opt 의 LayerNorm reorder 와 동일 효과)
        if self.reorder_index is not None:
            x = x[..., self.reorder_index]

        x_q = oa_lama_quantize(
            x,
            scale=self.scale,
            group_size=self.act_group_size,
            threshold=self.act_threshold,
        )
        return F.linear(x_q, self.weight, self.bias)


# =========================================================
# 5. Module Access / Replacement
# =========================================================
def get_named_linear_module(layer, module_name):
    mapping = {
        "self_attn.q_proj":   layer.self_attn.q_proj,
        "self_attn.k_proj":   layer.self_attn.k_proj,
        "self_attn.v_proj":   layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1": layer.fc1,
        "fc2": layer.fc2,
    }
    if module_name not in mapping:
        raise ValueError(f"Unsupported module_name: {module_name}")
    return mapping[module_name]


def set_named_linear_module(layer, module_name, new_module):
    if   module_name == "self_attn.q_proj":   layer.self_attn.q_proj   = new_module
    elif module_name == "self_attn.k_proj":   layer.self_attn.k_proj   = new_module
    elif module_name == "self_attn.v_proj":   layer.self_attn.v_proj   = new_module
    elif module_name == "self_attn.out_proj": layer.self_attn.out_proj = new_module
    elif module_name == "fc1":                layer.fc1                = new_module
    elif module_name == "fc2":                layer.fc2                = new_module
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
    if replace_scope == "one":
        return [one_layer_idx]
    elif replace_scope == "all":
        return list(range(num_layers))
    elif replace_scope == "custom":
        return [int(i.strip()) for i in custom_layer_indices.split(",")]
    raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules(
    model, layer_indices, module_names,
    act_group_size, act_threshold,
    reorder_indices,                   # dict: (layer_idx, module_name) → (sorted_index, threshold)
    enable_weight_quant, weight_group_size,
):
    replaced = []
    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old = get_named_linear_module(layer, module_name)

            # reorder index / threshold (calibration 결과 또는 None/0)
            ri        = None
            threshold = act_threshold
            if reorder_indices and (layer_idx, module_name) in reorder_indices:
                ri, threshold = reorder_indices[(layer_idx, module_name)]

            new = OALAMALinear(
                base_linear=old,
                act_group_size=act_group_size,
                act_threshold=threshold,
                reorder_index=ri,
                enable_weight_quant=enable_weight_quant,
                weight_group_size=weight_group_size,
                debug_name=f"layer{layer_idx}.{module_name}",
            )
            set_named_linear_module(layer, module_name, new)
            replaced.append(f"layer{layer_idx}.{module_name}(th={threshold})")
    return replaced


# =========================================================
# 6. Perplexity (ver2와 동일)
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
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",            type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir",           type=str, default="results_oa_lama")
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--replace_scope",        type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx",        type=int, default=10)
    parser.add_argument("--target_modules",       type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices", type=str, default=None)
    parser.add_argument("--act_group_size",       type=int, default=16)
    parser.add_argument("--act_threshold",        type=int, default=0,
                        help="reorder 미사용 시 수동 지정. reorder 사용 시 calibration에서 자동 결정.")
    # ── channel reordering ──────────────────────────────────────
    parser.add_argument("--reorder",              action="store_true",
                        help="OA-LAMA channel reordering 활성화 (outlier.py get_act_stats_opt 이식)")
    parser.add_argument("--n_calib_samples",      type=int, default=128,
                        help="calibration sample 수 (datautils.py 기본값 128)")
    parser.add_argument("--calib_seqlen",         type=int, default=2048)
    parser.add_argument("--act_sort_metric",      type=str, default="hessian",
                        choices=["hessian", "abs_mean"],
                        help="채널 정렬 metric (outlier.py 기본값 hessian)")
    # ── weight quant ────────────────────────────────────────────
    parser.add_argument("--enable_weight_quant",  action="store_true")

    parser.add_argument("--weight_group_size",    type=int, default=16)
    parser.add_argument("--eval_split",           type=str, default="test")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    module_names = parse_module_names(args.target_modules)

    lines = [
        "[Config]",
        f"model_id           : {args.model_id}",
        f"replace_scope      : {args.replace_scope}",
        f"target_modules     : {module_names}",
        f"act_group_size     : {args.act_group_size}",
        f"act_threshold      : {args.act_threshold}",
        f"reorder            : {args.reorder}",
        f"n_calib_samples    : {args.n_calib_samples}",
        f"act_sort_metric    : {args.act_sort_metric}",
        "",
        "[Weight Quant]",
        f"enable_weight_quant: {args.enable_weight_quant}",

        f"weight_group_size  : {args.weight_group_size}",
        "",
    ]

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
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ── channel reordering (calibration) ────────────────────────
    reorder_indices = {}
    if args.reorder:
        print("[4a] Loading calibration data ...")
        calib_loader = get_calib_dataloader(
            args.model_id, args.n_calib_samples, args.seed, args.calib_seqlen
        )
        print(f"[4b] Getting activation stats (metric={args.act_sort_metric}) ...")
        act_scales = get_act_stats_opt(model, calib_loader, dev, metric=args.act_sort_metric)

        # calibration 후 model 을 device 로 복원
        model.model.decoder.embed_tokens    = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        for i in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[i] = model.model.decoder.layers[i].to(dev)

        print("[4c] Computing reorder indices ...")
        reorder_indices = compute_reorder_indices(
            act_scales, target_layer_indices, module_names, args.act_group_size
        )
        thresholds = {k: v[1] for k, v in reorder_indices.items()}
        n_outlier_layers = sum(1 for th in thresholds.values() if th > 0)
        th_values = sorted(thresholds.values())
        print(f"  → reorder indices computed for {len(reorder_indices)} modules, "
              f"{n_outlier_layers} have threshold>0")
        print(f"  → threshold stats: min={min(th_values)}, max={max(th_values)}, "
              f"mean={sum(th_values)/len(th_values):.1f}")
        for k, th in sorted(thresholds.items()):
            if th > 0:
                print(f"     layer{k[0]}.{k[1]}: threshold={th}")
        lines += [
            "[Reorder]",
            f"n_modules_reordered: {len(reorder_indices)}",
            f"n_outlier_threshold>0: {n_outlier_layers}",
            f"threshold_min: {min(th_values)}",
            f"threshold_max: {max(th_values)}",
            f"threshold_mean: {sum(th_values)/len(th_values):.1f}",
            "",
        ]

    print("[4] Replacing modules with OALAMALinear ...")
    replaced = replace_modules(
        model, target_layer_indices, module_names,
        args.act_group_size, args.act_threshold,
        reorder_indices,
        args.enable_weight_quant, args.weight_group_size,
    )
    lines += ["[Replacement]", f"replaced_count : {len(replaced)}", ""]

    print("[5] OA-LAMA quant PPL ...")
    modified_ppl = compute_perplexity(model, testenc, dev)
    ppl_diff = modified_ppl - baseline_ppl
    rel_diff  = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines += [
        "[OA-LAMA Quant]",
        f"modified_ppl : {modified_ppl:.8f}",
        f"ppl_diff     : {ppl_diff:.12e}",
        f"relative_diff: {rel_diff:.12e}",
        "",
    ]
    print(f"OA-LAMA PPL : {modified_ppl:.4f}  (diff {ppl_diff:+.4f})")

    print("\n".join(lines))
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
