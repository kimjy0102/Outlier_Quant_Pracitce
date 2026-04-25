import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# OmniQuant W4A16g128로 저장된 weight를 불러오고, weight를 다시 fake quant하지 않은 상태에서
# selective activation QR만 적용하는 standalone variant.
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
    scale_method: str = "max",
    scale_shrink_factors = None,
    eps: float = 1e-8,
):
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")

    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    x_fp = x.float()
    if scale_shrink_factors is None:
        scale_shrink_factors = [1.0, 0.95, 0.9, 0.85, 0.8]

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
        if scale_method == "mse":
            best_scale = scale
            best_err = torch.full_like(max_abs, float("inf"))
            for shrink in scale_shrink_factors:
                cand_scale = (scale * float(shrink)).clamp_min(eps)
                q = torch.round(xg / cand_scale).clamp(qmin, qmax)
                dq = q * cand_scale
                err = (xg - dq).pow(2).mean(dim=-1, keepdim=True)
                choose = err < best_err
                best_err = torch.where(choose, err, best_err)
                best_scale = torch.where(choose, cand_scale, best_scale)
            scale = best_scale
        elif scale_method != "max":
            raise ValueError(f"Unsupported scale_method: {scale_method}")
        q = torch.round(xg / scale).clamp(qmin, qmax)
        dq = q * scale
        return dq.reshape_as(x_fp).to(x.dtype)

    else:
        raise ValueError(f"Unsupported quant mode: {mode}")


# =========================================================
# 3. Power-of-2 Adaptive QR Linear
#
# 아이디어:
#   각 group의 max_abs를 기반으로 qr_base를 {1,2,4,8,16,32,64} 중에서 선택 (조건 A)
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
        weight_scale_method: str = "max",
        weight_scale_shrink_factors = None,
        q_bits: int = 4,
        r_bits: int = 4,
        base_group_size: int = 128,        # 추가: base(나눌 값)를 결정할 group 크기 (-1이면 per-tensor)
        r_group_size: int = 128,           # R quantization을 위한 group 크기 (-1이면 per-tensor)
        selective_base_threshold: float = 1.0,
        selective_int_bits: int = 4,
        residual_clip_alpha: float = 0.0,
        collect_residuals: bool = False,   # 분석용: True이면 R 값을 _r_buf에 수집
        debug_name: str = "",
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.debug_name   = debug_name

        self.q_bits              = q_bits
        self.r_bits              = r_bits
        self.base_group_size     = base_group_size    # base 결정 group (R group과 독립)
        self.r_group_size        = r_group_size       # R quant group
        if selective_int_bits < 2:
            raise ValueError(f"selective_int_bits must be >= 2, got {selective_int_bits}")
        self.selective_base_threshold = selective_base_threshold  # base가 이 값보다 작으면 QR 대신 INT fallback
        self.selective_int_bits       = selective_int_bits        # fallback group에서 사용할 activation INT bit
        self.residual_clip_alpha      = residual_clip_alpha       # >0이면 R scale을 alpha*base로 제한
        self.collect_residuals   = collect_residuals  # 분석 모드 플래그
        self._r_buf: list        = []                 # R 값 누적 버퍼 (분석 모드에서만 채워짐)
        self._q_buf: list        = []                 # Q 값 누적 버퍼 (분석 모드에서만 채워짐)
        self._base_buf: list     = []                 # base 값 누적 버퍼 (분석 모드에서만 채워짐)
        self._base_group_buf: list = []               # 분석용: base group별 [base, Q=1 count, Q=0 count, Q=1 fraction]
        self._max_r_samples: int = 100_000            # 버퍼 최대 원소 수 (메모리 방지)

        if enable_weight_quant:
            w_q = fake_quant_symmetric(
                base_linear.weight.detach(),
                n_bits=weight_bits,
                mode=weight_quant_mode,
                ch_axis=weight_ch_axis,
                group_size=weight_group_size,
                scale_method=weight_scale_method,
                scale_shrink_factors=weight_scale_shrink_factors,
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
          base = {1,2,4,8,16,32,64,128} 중 max_abs <= b 를 만족하는 가장 작은 값
          Q = 1 if x >= base/2 else 0
          R = x - base * Q

        R scale은 실제 R 분포의 max_abs로 독립적으로 결정.
        """
        r_max_val = (2 ** (self.r_bits - 1)) - 1     # R 양자화 최대값 (signed)
        r_min_val = -r_max_val - 1                   # R 양자화 최소값 (signed)
        eps = 1e-8                                   # log/division 안전용 작은 값

        x_fp       = x.float()                       # 계산은 float32로 정확하게
        orig_shape = x_fp.shape                      # 마지막에 원래 shape으로 복원
        H          = orig_shape[-1]                  # hidden dim (마지막 차원 길이)

        # base/r group size 해석: -1이면 per-tensor (= H 전체를 한 group으로)
        base_gs = H if self.base_group_size == -1 else self.base_group_size  # base 결정 group
        r_gs    = H if self.r_group_size    == -1 else self.r_group_size     # R quant group

        # 두 group은 서로 정렬될 필요 없음 — 각자 H의 약수이기만 하면 됨
        assert H % base_gs == 0, f"H={H} not divisible by base_group_size={base_gs}"
        assert H % r_gs    == 0, f"H={H} not divisible by r_group_size={r_gs}"

        # =================== Step 1: base 결정 (base_gs 단위) ===================
        G_base  = H // base_gs                                                      # base group 개수
        xg_base = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))                 # [..., G_base, base_gs]
        max_abs = xg_base.abs().amax(dim=-1, keepdim=True).clamp_min(eps)           # group별 max|x|

        if self.q_bits == 1:
            # --- 1-bit Q: nearest 2의 승수 base 선택 (현재 active 방식) ---
            log2_max   = torch.log2(max_abs)                                        # log2(max_abs) — 어느 2의 승수에 가까운지 판단용
            log2_floor = torch.floor(log2_max)                                      # 아래쪽 가장 가까운 정수
            log2_ceil  = torch.ceil(log2_max)                                       # 위쪽 가장 가까운 정수
            base_floor = torch.pow(2.0, log2_floor)                                 # floor 후보 base
            base_ceil  = torch.pow(2.0, log2_ceil)                                  # ceil 후보 base
            dist_floor = torch.abs(max_abs - base_floor)                            # max_abs와 floor 후보의 거리
            dist_ceil  = torch.abs(base_ceil  - max_abs)                            # max_abs와 ceil 후보의 거리
            base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)      # 더 가까운 후보 선택 (nearest)
            base = base.clamp(min=2**(-4), max=128.0)                                    # base 값 범위를 {1,2,...,128}로 제한
            
            # --- sign-aware: group 내 max_abs가 양수/음수 어느 쪽에서 왔는지 판단 ---
            max_pos_val = xg_base.amax(dim=-1, keepdim=True)                        # group 내 최대값 (signed, 양수 쪽 대표)
            min_val     = xg_base.amin(dim=-1, keepdim=True)                        # group 내 최솟값 (signed, 음수 쪽 대표)
            # 양수 쪽 abs >= 음수 쪽 abs이면 +1 (양수 outlier), 반대면 -1 (음수 outlier)
            sign_flag   = torch.where(max_pos_val.abs() >= min_val.abs(),
                                      torch.ones_like(max_pos_val),
                                      -torch.ones_like(max_pos_val))               # shape: [..., G_base, 1]
            # xg * sign_flag >= base/2 : 양수 outlier면 기존과 동일, 음수 outlier면 xg <= -base/2 와 동치
            q = (xg_base * sign_flag >= base / 2.0).float()                        # 해당 방향의 큰 값만 q=1
            r = xg_base - sign_flag * base * q                                     # r = x - sign*base*q

            # 분석 모드: base group 단위로 base와 group 내부 Q=1/Q=0 개수를 같은 index에서 저장
            group_buf_n = sum(t.shape[0] for t in self._base_group_buf)
            if self.collect_residuals and group_buf_n < self._max_r_samples:
                base_cpu    = base.detach().float().cpu().flatten()                 # [..., G_base, 1] → group별 base
                q1_count_cpu = (q == 1).sum(dim=-1).detach().float().cpu().flatten() # [..., G_base] → group별 Q=1 개수
                q0_count_cpu = float(base_gs) - q1_count_cpu                        # group별 Q=0 개수
                q1_frac_cpu  = q1_count_cpu / float(base_gs)                        # group 내부 Q=1 비율
                remain       = self._max_r_samples - group_buf_n
                n_take       = min(len(base_cpu), remain, 10_000)
                idx          = torch.randperm(len(base_cpu))[:n_take]
                group_stats  = torch.stack(
                    [base_cpu[idx], q1_count_cpu[idx], q0_count_cpu[idx], q1_frac_cpu[idx]], dim=1
                )
                self._base_group_buf.append(group_stats)

            # 분석 모드: 선택된 base 값들을 버퍼에 subsampling하여 저장 (per-group → flatten)
            if self.collect_residuals and len(self._base_buf) < self._max_r_samples:
                base_cpu = base.detach().float().cpu().flatten()                     # [..., G_base, 1] → 1-D
                remain   = self._max_r_samples - len(self._base_buf)                # 버퍼 잔여 슬롯
                n_take   = min(len(base_cpu), remain, 10_000)                       # 한 번에 최대 10000개
                idx      = torch.randperm(len(base_cpu))[:n_take]                  # 랜덤 subsampling
                self._base_buf.append(base_cpu[idx])                               # 버퍼에 추가

        else:
            sign_flag = 1.0                                                         # q_bits>=2: q 자체가 signed이므로 sign_flag 불필요
            q_max_val = (2 ** (self.q_bits - 1)) - 1                                # signed Q의 최대 절대값
            threshold = max_abs / q_max_val                                         # max_abs/Q_max <= base 조건
            # {2,4,8,16,32,64} 중 threshold 이상이 되는 가장 작은 2의 승수 선택
            base = torch.full_like(threshold, 64.0)
            base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
            base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
            base = torch.where(threshold <=  8.0, torch.full_like(threshold,  8.0), base)
            base = torch.where(threshold <=  4.0, torch.full_like(threshold,  4.0), base)
            base = torch.where(threshold <=  2.0, torch.full_like(threshold,  2.0), base)

            q = torch.round(xg_base / base).clamp(-q_max_val, q_max_val)            # 정수 Q 양자화
            r = xg_base - base * q                                                  # 잔여 R

        # selective fallback: base가 작은 group은 QR 분해 대신 x 자체를 INT4 fake-quant로 표현
        selective_mask = base < self.selective_base_threshold                        # [..., G_base, 1]
        int_qmax_val   = (2 ** (self.selective_int_bits - 1)) - 1
        int_qmin_val   = -int_qmax_val - 1
        int_scale      = (max_abs / int_qmax_val).clamp_min(eps)
        x_int_q        = torch.round(xg_base / int_scale).clamp(int_qmin_val, int_qmax_val)
        x_int_dq       = x_int_q * int_scale                                         # [..., G_base, base_gs]

        # q_scaled: Q가 weight에 곱해질 때의 실질적 기여값 (= sign * Q * base)
        q_scaled_base = q * sign_flag * base                                        # [..., G_base, base_gs]

        # =================== Step 2: R quantize (r_gs 단위, base와 독립) ===================
        # r을 원래 shape으로 복원 → r_gs 단위로 다시 reshape (base_gs와 무관)
        r_flat      = r.reshape(orig_shape)                                         # [..., H]

        # 분석 모드: R 양자화 전 raw R 값을 버퍼에 subsampling하여 저장
        if self.collect_residuals and len(self._r_buf) < self._max_r_samples:
            r_cpu  = r_flat.detach().float().cpu().flatten()                        # GPU → CPU, 1-D
            remain = self._max_r_samples - len(self._r_buf)                        # 버퍼에 남은 여유 슬롯 수
            n_take = min(len(r_cpu), remain, 10_000)                               # 한 번에 최대 10000개씩만 추가
            idx    = torch.randperm(len(r_cpu))[:n_take]                           # 랜덤 인덱스로 subsampling
            self._r_buf.append(r_cpu[idx])                                         # 버퍼에 추가 (cat은 분석 시점에)

        # 분석 모드: Q 값도 동일한 방식으로 버퍼에 저장 (q는 base_gs shape → flatten)
        if self.collect_residuals and len(self._q_buf) < self._max_r_samples:
            q_cpu  = q.detach().float().cpu().flatten()                             # Q 값 CPU로 이동
            remain = self._max_r_samples - len(self._q_buf)
            n_take = min(len(q_cpu), remain, 10_000)
            idx    = torch.randperm(len(q_cpu))[:n_take]
            self._q_buf.append(q_cpu[idx])
        G_r         = H // r_gs                                                     # R group 개수
        r_for_quant = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))                 # [..., G_r, r_gs]
        max_abs_r   = r_for_quant.abs().amax(dim=-1, keepdim=True).clamp_min(eps)   # R group별 max|r|
        r_input      = r_for_quant
        if self.residual_clip_alpha > 0:
            # QR에서는 R이 base가 처리한 coarse unit의 잔여분이므로,
            # 극소수 residual tail이 scale_r 전체를 키우지 못하게 alpha*base로 R 범위를 제한한다.
            base_flat = base.expand_as(xg_base).reshape(orig_shape)                  # [..., H]
            base_for_r = base_flat.reshape(orig_shape[:-1] + (G_r, r_gs))            # [..., G_r, r_gs]
            clip_bound_elem = (float(self.residual_clip_alpha) * base_for_r).clamp_min(eps)
            clip_bound_grp  = clip_bound_elem.amax(dim=-1, keepdim=True).clamp_min(eps)
            max_abs_r       = torch.minimum(max_abs_r, clip_bound_grp).clamp_min(eps)
            r_input         = torch.maximum(
                torch.minimum(r_for_quant, clip_bound_elem),
                -clip_bound_elem,
            )
        scale_r     = max_abs_r / r_max_val                                         # tight scale (R 분포 기준, 선택적으로 base-bound)
        r_q         = torch.round(r_input / scale_r).clamp(r_min_val, r_max_val)     # 정수 R 양자화
        r_dq_grp    = r_q * scale_r                                                 # 역양자화 (group shape 유지)
        r_dq        = r_dq_grp.reshape(orig_shape)                                  # [..., H]로 복원

        # q_scaled도 원래 shape으로 복원해서 반환 (forward에서 matmul용)
        q_scaled    = q_scaled_base.reshape(orig_shape)                             # [..., H]

        # selective fallback group은 q path를 끄고, r path에 INT4(x)를 직접 넣는다.
        selective_flat = selective_mask.expand_as(xg_base).reshape(orig_shape)
        x_int_dq_flat  = x_int_dq.reshape(orig_shape)
        q_scaled       = torch.where(selective_flat, torch.zeros_like(q_scaled), q_scaled)
        r_dq           = torch.where(selective_flat, x_int_dq_flat, r_dq)

        # (q_scaled, r_dq) 튜플 반환 — forward에서 각각 matmul 후 합산
        return (
            q_scaled.to(x.dtype),
            r_dq.to(x.dtype),
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


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if len(x.strip()) > 0]


def parse_module_float_map(s: str):
    if s is None or len(s.strip()) == 0:
        return {}
    valid = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    }
    parsed = {}
    for item in s.split(","):
        item = item.strip()
        if len(item) == 0:
            continue
        if ":" not in item:
            raise ValueError(
                "--module_selective_base_thresholds must use module:value format, "
                f"got '{item}'"
            )
        module_name, value = item.split(":", 1)
        module_name = module_name.strip()
        if module_name not in valid:
            raise ValueError(f"Invalid module name in module threshold map: {module_name}")
        parsed[module_name] = float(value.strip())
    return parsed


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
    weight_scale_method,
    weight_scale_shrink_factors,
    q_bits,
    r_bits,
    base_group_size,                    # base 결정 group (R group과 독립)
    r_group_size,
    selective_base_threshold,
    module_selective_base_thresholds,
    selective_int_bits,
    residual_clip_alpha,
    collect_residuals: bool = False,    # 추가: 분석 모드 플래그 (data 스크립트에서 True로 전달)
):
    replaced_names = []

    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)
            # module-wise threshold가 지정된 경우 해당 module만 덮어쓰고,
            # 지정되지 않은 module은 기존 전역 threshold를 그대로 사용한다.
            module_threshold = module_selective_base_thresholds.get(
                module_name,
                selective_base_threshold,
            )

            new_module = QuotRemLinear(
                base_linear=old_module,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                weight_scale_method=weight_scale_method,
                weight_scale_shrink_factors=weight_scale_shrink_factors,
                q_bits=q_bits,
                r_bits=r_bits,
                base_group_size=base_group_size,
                r_group_size=r_group_size,
                selective_base_threshold=module_threshold,
                selective_int_bits=selective_int_bits,
                residual_clip_alpha=residual_clip_alpha,
                collect_residuals=collect_residuals,    # 추가
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

    parser.add_argument("--model_id", type=str,
                        default="/home2/juneyeop/OmniQuant/checkpoint/opt-6.7b-w4a16g128",
                        help="OmniQuant W4A16g128 save_pretrained checkpoint path")
    parser.add_argument("--output_dir", type=str, default="quotrem_ppl_results_v3_sel_omniweight")
    parser.add_argument("--seed", type=int, default=42)

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="one", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument("--target_modules", type=str, default="fc1",
                        help="e.g. fc1 or self_attn.q_proj,fc1")
    parser.add_argument("--custom_layer_indices", type=str, default=None,
                        help="e.g. 0,2,4")

    # OmniQuant checkpoint already contains quantized-dequantized weights, so do not quantize weights again.
    parser.add_argument("--enable_weight_quant", action="store_true",
                        help="Deprecated in this file: ignored to avoid double-quantizing OmniQuant weights")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--weight_quant_mode", type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)
    parser.add_argument("--weight_scale_method", type=str, default="max",
                        choices=["max", "mse"])
    parser.add_argument("--weight_scale_shrink_factors", type=str,
                        default="1.0,0.95,0.9,0.85,0.8")

    # QR bits
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument("--r_bits", type=int, default=4)
    # base 결정 group (R group과 독립적으로 설정 가능, -1이면 per-tensor)
    parser.add_argument("--base_group_size", type=int, default=128)
    # R quantization group (-1이면 per-tensor)
    parser.add_argument("--r_group_size", type=int, default=128)
    parser.add_argument("--selective_base_threshold", type=float, default=1.0,
                        help="base가 이 값보다 작은 group은 QR 대신 INT fallback 적용")
    parser.add_argument("--module_selective_base_thresholds", type=str, default="",
                        help="module별 threshold 덮어쓰기. 예: fc1:8,fc2:4,self_attn.v_proj:8")
    parser.add_argument("--selective_int_bits", type=int, default=4,
                        help="selective fallback group에서 사용할 activation INT bit")
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0,
                        help=">0이면 R quant scale을 alpha*base로 cap하고 R을 clip. 0이면 기존 max_abs_r 방식")

    # eval
    parser.add_argument("--eval_split", type=str, default="test")
    # probe
    parser.add_argument("--do_probe_compare", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)
    weight_scale_shrink_factors = parse_float_list(args.weight_scale_shrink_factors)
    use_weight_quant = False
    module_selective_base_thresholds = parse_module_float_map(args.module_selective_base_thresholds)

    lines = []
    lines.append("[Config]")
    lines.append(f"model_id           : {args.model_id}")
    lines.append(f"replace_scope      : {args.replace_scope}")
    lines.append(f"one_layer_idx      : {args.one_layer_idx}")
    lines.append(f"target_modules     : {module_names}")
    lines.append(f"eval_split         : {args.eval_split}")
    lines.append("")
    lines.append("[OmniQuant Weight]")
    lines.append("source             : OmniQuant checkpoint loaded by --model_id")
    lines.append("enable_weight_quant: False  # forced off to avoid double quantization")
    lines.append(f"ignored_cli_enable_weight_quant: {args.enable_weight_quant}")
    lines.append(f"weight_bits        : {args.weight_bits}")
    lines.append(f"weight_quant_mode  : {args.weight_quant_mode}")
    lines.append(f"weight_ch_axis     : {args.weight_ch_axis}")
    lines.append(f"weight_group_size  : {args.weight_group_size}")
    lines.append(f"weight_scale_method: {args.weight_scale_method}")
    lines.append(f"weight_scale_shrink_factors: {weight_scale_shrink_factors}")
    lines.append("")
    lines.append("[Adaptive QR]")
    lines.append(f"q_bits             : {args.q_bits}")
    lines.append(f"r_bits             : {args.r_bits}")
    lines.append(f"base_group_size    : {args.base_group_size}")   # 추가
    lines.append(f"r_group_size       : {args.r_group_size}")
    lines.append(f"selective_base_threshold : {args.selective_base_threshold}")
    lines.append(f"module_selective_base_thresholds : {module_selective_base_thresholds}")
    lines.append(f"selective_int_bits       : {args.selective_int_bits}")
    lines.append(f"residual_clip_alpha      : {args.residual_clip_alpha}")
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
        enable_weight_quant=use_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        weight_scale_method=args.weight_scale_method,
        weight_scale_shrink_factors=weight_scale_shrink_factors,
        q_bits=args.q_bits,
        r_bits=args.r_bits,
        base_group_size=args.base_group_size,   # 추가
        r_group_size=args.r_group_size,
        selective_base_threshold=args.selective_base_threshold,
        module_selective_base_thresholds=module_selective_base_thresholds,
        selective_int_bits=args.selective_int_bits,
        residual_clip_alpha=args.residual_clip_alpha,
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
