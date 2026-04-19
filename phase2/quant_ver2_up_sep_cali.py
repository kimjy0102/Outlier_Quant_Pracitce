import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# ver2_up_sep_cali: quant_ver2_up_sep.py 의 양자화 방식을 calibration 기반으로 적용
# - sign-aware Q (q_bits=1): group 내 max_abs 방향에 따라 양수/음수 outlier 포착
# - nearest power-of-2 base
# - base_group_size / r_group_size 분리
# - calibration 데이터로 base/sign_flag/scale_r 사전 결정 후 고정
# - split matmul: F.linear(q_scaled, W) + F.linear(r_dq, W)
#
# [이력]
# - v1: 128 calib sample의 running max로 base/sign_flag/scale_r 모두 고정 → PPL=1300
#   원인: scale_r가 outlier로 과대 설정 → 일반 token R이 round=0으로 손실
# - v2: oa-lama 스타일 (첫 호출 한 번만 stats) + scale_r runtime → PPL=11.36
#   문제: 각 모듈이 첫 sample 1회만 반영 → n_calib_samples 의미 없음 ("대표값 고정")
# - v3: traditional calibration (running max) + scale_r runtime → v2보다 약간 저조
#   원인: running max가 outlier로 부풀려져 base가 과대 → Q 활성화 감소 → R 부담↑ (r_bits=3 한계)
# - v4: median base + majority vote sign + scale_r runtime → PPL=11.333
# - v5(현재): median base + base-따르기 sign + scale_r runtime
#   * calibration phase: 매 batch의 group-wise max를 list에 누적 (fp pass, 양자화 X)
#   * finalize_calib: batch 축으로 median 취해 base 결정 (outlier에 robust)
#                     sign_flag는 q_pos_median >= q_neg_median 에 따라 결정
#                     (base magnitude가 온 쪽과 부호를 일치시켜 signed_base 방향 정합)
#   * inference phase: 양자화 적용 (scale_r는 runtime per-token group)

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
        start = rng.randint(0, total_tokens - seqlen - 1)   # 랜덤 위치에서 seqlen 만큼 추출
        samples.append(enc[:, start: start + seqlen])
    return samples


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
# 3. QuotRemLinear (calibration 방식)
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
        base_group_size=128,    # base/Q 결정 group 크기 (-1이면 per-tensor)
        r_group_size=128,       # R 양자화 group 크기 (-1이면 per-tensor)
        debug_name="",
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features     = base_linear.in_features
        self.out_features    = base_linear.out_features
        self.debug_name      = debug_name
        self.q_bits          = q_bits
        self.r_bits          = r_bits
        self.base_group_size = base_group_size   # base 결정 group 크기
        self.r_group_size    = r_group_size      # R 양자화 group 크기

        H = self.in_features
        # per-tensor(-1)이면 group 수 = 1
        base_gs = H if base_group_size == -1 else base_group_size
        r_gs    = H if r_group_size    == -1 else r_group_size
        assert H % base_gs == 0, f"in_features={H} not divisible by base_group_size={base_gs}"
        assert H % r_gs    == 0, f"in_features={H} not divisible by r_group_size={r_gs}"
        G_base = H // base_gs   # base-group 수
        G_r    = H // r_gs      # R-group 수

        # 캘리브레이션 전까지 calibrated=False → forward에서 통계 수집 후 fp pass
        self.calibrated = False

        # [v4] 매 batch의 group-wise (max_pos, max_neg_abs)를 list에 누적 (running max 대신)
        # finalize_calib에서 median/majority vote로 집약.
        self._batch_max_pos_list     = []    # 각 원소: [G_base] tensor
        self._batch_max_neg_abs_list = []    # 각 원소: [G_base] tensor

        # (디버그용 running max — 분포 비교 목적으로만 유지)
        self.register_buffer('calib_max_pos',     torch.zeros(G_base))
        self.register_buffer('calib_max_neg_abs', torch.zeros(G_base))
        self.register_buffer('calib_max_r_abs',   torch.zeros(G_r))

        # 캘리브레이션 완료 후 고정되는 파라미터
        self.register_buffer('fixed_base',      torch.ones(G_base))      # [G_base] 선택된 2의 승수
        self.register_buffer('fixed_sign_flag', torch.ones(G_base))      # [G_base] +1 or -1
        self.register_buffer('fixed_scale_r',   torch.ones(G_r))         # [G_r] R scale

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

    def _update_calib_stats(self, x: torch.Tensor):
        """
        [v4: running max → per-batch list 누적]
        이유: running max는 outlier 1개로 base가 부풀려져 Q 활성화가 감소 → R 부담 증가.
              per-batch max를 list로 모아 finalize에서 median/majority로 robust하게 집약.
        주의: scale_r은 runtime per-token group이라 여기 stats에 영향 X.
        """
        eps  = 1e-8
        x_fp = x.float()
        H    = x_fp.shape[-1]
        dev  = x.device

        # (디버그용) running max 버퍼를 input device로 이동
        if self.calib_max_pos.device != dev:
            self.calib_max_pos.data     = self.calib_max_pos.data.to(dev)
            self.calib_max_neg_abs.data = self.calib_max_neg_abs.data.to(dev)

        base_gs = H if self.base_group_size == -1 else self.base_group_size
        G_base  = H // base_gs

        xg = x_fp.reshape(-1, G_base, base_gs)                                  # [N, G_base, base_gs]

        # 이번 batch의 group-wise max (torch scalar tensors, CPU로 옮겨서 누적)
        max_pos_batch     = xg.amax(dim=-1).amax(dim=0).clamp_min(0.0)          # [G_base]
        max_neg_abs_batch = xg.amin(dim=-1).abs().amax(dim=0).clamp_min(0.0)    # [G_base]

        # list 누적 (메모리 절감: CPU로 detach)
        self._batch_max_pos_list.append(max_pos_batch.detach().cpu())
        self._batch_max_neg_abs_list.append(max_neg_abs_batch.detach().cpu())

        # 디버그용 running max도 함께 업데이트 (비교를 위해)
        self.calib_max_pos.data     = torch.maximum(self.calib_max_pos,     max_pos_batch.detach())
        self.calib_max_neg_abs.data = torch.maximum(self.calib_max_neg_abs, max_neg_abs_batch.detach())

    def finalize_calib(self, base_quantile: float = 0.5):
        """
        [v4: median base + majority vote sign]
        run_calibration 마지막 1회 호출.
          - base magnitude: batch 축으로 per-group max의 quantile(default median=0.5)
                             → outlier 1~2개 batch에 끌려가지 않음
          - sign_flag     : batch 축에서 dominant sign(+/-) count의 majority vote
          - nearest pow2 base로 양자화 (2..128)
        p90 등으로 튜닝하려면 base_quantile 인자만 변경.
        """
        eps = 1e-8

        if len(self._batch_max_pos_list) == 0:
            # calibration 데이터가 없는 극단적 상황: fallback으로 1.0 유지
            self.calibrated = True
            return

        # [n_batches, G_base]
        stacked_pos = torch.stack(self._batch_max_pos_list, dim=0)
        stacked_neg = torch.stack(self._batch_max_neg_abs_list, dim=0)

        # per-group magnitude (batch 축 quantile) — median이 base_quantile=0.5
        q_pos = torch.quantile(stacked_pos, base_quantile, dim=0)               # [G_base]
        q_neg = torch.quantile(stacked_neg, base_quantile, dim=0)               # [G_base]

        # group별 크기 대표값: 양/음 quantile 중 큰 쪽
        max_abs = torch.maximum(q_pos, q_neg).clamp_min(eps)

        # sign_flag: base magnitude가 결정된 쪽의 부호를 그대로 따라감
        # (majority vote는 median과 어긋나면 signed_base가 outlier 반대쪽을 가리켜 Q 무용지물)
        sign_flag = torch.where(q_pos >= q_neg,
                                torch.ones_like(max_abs),
                                -torch.ones_like(max_abs))

        # nearest power-of-2 base
        log2_max   = torch.log2(max_abs)
        log2_floor = torch.floor(log2_max)
        log2_ceil  = torch.ceil(log2_max)
        base_floor = torch.pow(2.0, log2_floor)
        base_ceil  = torch.pow(2.0, log2_ceil)
        dist_floor = (max_abs - base_floor).abs()
        dist_ceil  = (base_ceil - max_abs).abs()
        # min을 1.0으로 낮춤: median이 1 근처인 group(작은 magnitude)에서 base=1 허용
        # → threshold=0.5로 더 많은 token이 Q=1 → R 부담 감소
        base       = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil).clamp(1.0, 128.0)

        # fixed_* 저장 (CPU) — scale_r은 runtime이므로 저장 안함
        self.fixed_base.data      = base.detach().cpu()
        self.fixed_sign_flag.data = sign_flag.detach().cpu()
        self.calibrated = True                                                  # 이후 forward는 양자화 path

        # list 메모리 해제 (더 이상 불필요)
        self._batch_max_pos_list     = []
        self._batch_max_neg_abs_list = []

    def forward(self, x: torch.Tensor):
        # [변경: 첫 호출 양자화 → calibration 단계 분리]
        # calibration phase (calibrated=False): stats 수집 + fp pass (양자화 X)
        # inference phase  (calibrated=True): 양자화 적용
        # 이전(oa-lama 스타일 self-consistent)은 첫 호출 양자화했지만, n_calib_samples 무의미.
        # 현재는 traditional calibration: 모든 sample fp activation으로 통계 수집.
        if not self.calibrated:
            self._update_calib_stats(x)
            return F.linear(x, self.weight, self.bias)                          # fp pass (weight만 양자화됨)

        # ---- 추론: 고정 파라미터로 양자화 + split matmul ----
        r_max_val  = (2 ** (self.r_bits - 1)) - 1
        r_min_val  = -r_max_val - 1
        eps        = 1e-8

        x_fp       = x.float()
        orig_shape = x_fp.shape
        H          = orig_shape[-1]
        n_extra    = x_fp.dim() - 1    # leading dims 수 (batch, seqlen 등)

        base_gs = H if self.base_group_size == -1 else self.base_group_size
        r_gs    = H if self.r_group_size    == -1 else self.r_group_size
        G_base  = H // base_gs
        G_r     = H // r_gs

        # fixed_base/sign_flag: [G_base] → (..., G_base, 1) for broadcasting
        base_view       = (1,) * n_extra + (G_base, 1)
        fixed_base      = self.fixed_base.to(x.device).view(base_view)         # [..., G_base, 1]
        fixed_sign_flag = self.fixed_sign_flag.to(x.device).view(base_view)    # [..., G_base, 1]

        # Step 1: base-group 단위 Q 계산 (sign-aware)
        xg_base   = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))          # [..., G_base, base_gs]
        q         = (xg_base * fixed_sign_flag >= fixed_base / 2.0).float()    # sign-aware threshold
        r         = xg_base - fixed_sign_flag * fixed_base * q                 # r = x - sign*base*q
        q_scaled  = (q * fixed_sign_flag * fixed_base).reshape(orig_shape)     # [..., H]

        # Step 2: R-group 단위 R 양자화 (scale_r은 runtime per-token group max — oa-lama 패턴)
        r_flat    = r.reshape(orig_shape)                                       # [..., H]
        r_grp     = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))              # [..., G_r, r_gs]
        max_r     = r_grp.abs().amax(dim=-1, keepdim=True).clamp_min(eps)      # [..., G_r, 1] runtime!
        scale_r   = max_r / r_max_val                                          # tight scale per token-group
        r_q       = torch.round(r_grp / scale_r).clamp(r_min_val, r_max_val)
        r_dq      = (r_q * scale_r).reshape(orig_shape)                        # [..., H]

        # split matmul: quant_ver2_up_sep.py 방식 그대로
        q_out = F.linear(q_scaled.to(x.dtype), self.weight, None)              # Q 기여
        r_out = F.linear(r_dq.to(x.dtype),     self.weight, None)              # R 기여
        out   = q_out + r_out
        if self.bias is not None:
            out = out + self.bias
        return out


# =========================================================
# 4. Module Access / Replacement
# =========================================================
def get_named_linear_module(layer, module_name):
    mapping = {
        "self_attn.q_proj":   layer.self_attn.q_proj,
        "self_attn.k_proj":   layer.self_attn.k_proj,
        "self_attn.v_proj":   layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1":                layer.fc1,
        "fc2":                layer.fc2,
    }
    if module_name not in mapping:
        raise ValueError(f"Unsupported module_name: {module_name}")
    return mapping[module_name]


def set_named_linear_module(layer, module_name, new_module):
    if   module_name == "self_attn.q_proj":   layer.self_attn.q_proj   = new_module
    elif module_name == "self_attn.k_proj":   layer.self_attn.k_proj   = new_module
    elif module_name == "self_attn.v_proj":   layer.self_attn.v_proj   = new_module
    elif module_name == "self_attn.out_proj": layer.self_attn.out_proj  = new_module
    elif module_name == "fc1":                layer.fc1                 = new_module
    elif module_name == "fc2":                layer.fc2                 = new_module
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
    else:
        raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules(model, layer_indices, module_names,
                    enable_weight_quant, weight_bits, weight_quant_mode,
                    weight_ch_axis, weight_group_size,
                    q_bits, r_bits, base_group_size, r_group_size):
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
                q_bits=q_bits, r_bits=r_bits,
                base_group_size=base_group_size,   # base 결정 group
                r_group_size=r_group_size,         # R 양자화 group
                debug_name=f"layer{layer_idx}.{module_name}",
            )
            set_named_linear_module(layer, module_name, new)
            replaced.append(f"layer{layer_idx}.{module_name}")
    return replaced


# =========================================================
# 5. Calibration
# =========================================================
def get_all_quotrem_modules(model):
    """모델 내 QuotRemLinear 모듈 전체 수집"""
    return {name: m for name, m in model.named_modules() if isinstance(m, QuotRemLinear)}


def run_calibration(model, calib_samples, device, n_calib_samples):
    """
    캘리브레이션 실행:
      1. calib_samples 로 forward → 각 모듈의 _update_calib_stats 자동 호출
      2. 모든 모듈의 finalize_calib() 호출 → 이후 추론은 고정 파라미터 사용
    """
    model.eval()
    print(f"  Running {n_calib_samples} calibration forward passes ...")
    with torch.no_grad():
        for batch in tqdm(calib_samples[:n_calib_samples], desc="Calibrating"):
            model(batch.to(device), use_cache=False)    # forward → _update_calib_stats 호출

    # 모든 모듈 finalize
    qr_modules = get_all_quotrem_modules(model)
    for name, m in qr_modules.items():
        m.finalize_calib()                              # fixed_base/sign_flag/scale_r 확정
    print(f"  Finalized {len(qr_modules)} QuotRemLinear modules.")

    # [DEBUG] 각 모듈 타입별 fixed_base 분포 — running max(v3) vs median(v4) 비교
    for target_key in ["layers.0.self_attn.k_proj", "layers.0.fc1", "layers.15.fc1", "layers.15.fc2"]:
        match = [(n, m) for n, m in qr_modules.items() if target_key in n]
        if not match:
            continue
        name, m = match[0]
        fb  = m.fixed_base
        fsf = m.fixed_sign_flag
        print(f"  [DEBUG] {name}")
        print(f"    fixed_base      : min={fb.min().item():.2f}, max={fb.max().item():.2f}, "
              f"mean={fb.mean().item():.2f}, unique={torch.unique(fb).tolist()}")
        print(f"    fixed_sign_flag : +1 count={(fsf>0).sum().item()}, -1 count={(fsf<0).sum().item()}")
        print(f"    (running-max ref) calib_max_pos: min={m.calib_max_pos.min().item():.2f}, "
              f"max={m.calib_max_pos.max().item():.2f}")
        print(f"    (running-max ref) calib_max_neg_abs: min={m.calib_max_neg_abs.min().item():.2f}, "
              f"max={m.calib_max_neg_abs.max().item():.2f}")


# =========================================================
# 6. Perplexity
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
    parser.add_argument("--model_id",             type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir",            type=str, default="quotrem_ppl_results_v2_sep_cali")
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--replace_scope",         type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx",         type=int, default=10)
    parser.add_argument("--target_modules",        type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices",  type=str, default=None)
    parser.add_argument("--enable_weight_quant",   action="store_true")
    parser.add_argument("--weight_bits",           type=int, default=4)
    parser.add_argument("--weight_quant_mode",     type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis",        type=int, default=0)
    parser.add_argument("--weight_group_size",     type=int, default=16)
    parser.add_argument("--q_bits",                type=int, default=1)
    parser.add_argument("--r_bits",                type=int, default=3)
    # base 결정 group (-1이면 per-tensor)
    parser.add_argument("--base_group_size",       type=int, default=64)
    # R 양자화 group (-1이면 per-tensor)
    parser.add_argument("--r_group_size",          type=int, default=16)
    parser.add_argument("--n_calib_samples",       type=int, default=128)
    parser.add_argument("--calib_seqlen",          type=int, default=2048)
    parser.add_argument("--eval_split",            type=str, default="test")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    module_names = parse_module_names(args.target_modules)

    lines = [
        "[Config]",
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
        f"base_group_size    : {args.base_group_size}",
        f"r_group_size       : {args.r_group_size}",
        "",
    ]

    print("[1] Loading test data ...")
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
        args.q_bits, args.r_bits, args.base_group_size, args.r_group_size,
    )
    lines += ["[Replacement]", f"replaced_count : {len(replaced)}", ""]

    print("[5] Loading calibration data ...")
    calib_samples = load_wikitext2_calib_samples(
        args.model_id, n_samples=args.n_calib_samples,
        seqlen=args.calib_seqlen, seed=args.seed,
    )

    print("[6] Running calibration ...")
    run_calibration(model, calib_samples, dev, n_calib_samples=args.n_calib_samples)

    print("[7] Modified PPL (calibrated) ...")
    modified_ppl = compute_perplexity(model, testenc, dev)
    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines += [
        "[Modified (calibration-based static)]",
        f"modified_ppl : {modified_ppl:.8f}",
        f"ppl_diff     : {ppl_diff:.12e}",
        f"relative_diff: {rel_diff:.12e}",
        "",
    ]

    print("\n".join(lines))
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()
