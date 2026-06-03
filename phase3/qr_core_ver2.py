import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_axis(dim: int, axis: int) -> int:
    if axis < 0:
        axis = dim + axis
    if axis < 0 or axis >= dim:
        raise ValueError(f"Invalid axis={axis} for tensor dim={dim}")
    return axis


def fake_quant_symmetric(
    x: torch.Tensor,
    n_bits: int = 8,
    mode: str = "tensor",
    ch_axis: int = -1,
    group_size: int = None,
    scale_method: str = "max",
    scale_shrink_factors=None,
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
# Pure QR decomposition (v2)
#
# qr_core.py 대비 변경점:
#   - selective_base_threshold / selective_int_bits 제거.
#   - per-group fallback 분기 없음. 모든 group을 동일하게 QR 분해.
#   - 모듈 단위로 QuotRemLinearV2 또는 IntActLinear 중 하나를 선택하는 방식이므로
#     group-level dynamic selection이 불필요해짐.
# =========================================================
def qr_decompose_activation_v2(
    x: torch.Tensor,
    q_bits: int = 1,
    r_bits: int = 3,
    base_group_size: int = 128,
    r_group_size: int = 128,
    residual_clip_alpha: float = 0.0,
    eps: float = 1e-8,
):
    r_max_val = (2 ** (r_bits - 1)) - 1
    r_min_val = -r_max_val - 1

    x_fp = x.float()
    orig_shape = x_fp.shape
    H = orig_shape[-1]

    base_gs = H if base_group_size == -1 else base_group_size
    r_gs    = H if r_group_size    == -1 else r_group_size

    assert H % base_gs == 0, f"H={H} not divisible by base_group_size={base_gs}"
    assert H % r_gs    == 0, f"H={H} not divisible by r_group_size={r_gs}"

    # Step 1: base 결정 및 Q/R 분해
    G_base  = H // base_gs
    xg_base = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))
    max_abs = xg_base.abs().amax(dim=-1, keepdim=True).clamp_min(eps)

    if q_bits == 1:
        # 1-bit Q: nearest power-of-two base, sign-aware
        log2_max   = torch.log2(max_abs)
        log2_floor = torch.floor(log2_max)
        log2_ceil  = torch.ceil(log2_max)
        base_floor = torch.pow(2.0, log2_floor)
        base_ceil  = torch.pow(2.0, log2_ceil)
        dist_floor = torch.abs(max_abs - base_floor)
        dist_ceil  = torch.abs(base_ceil  - max_abs)
        base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)
        base = base.clamp(min=2**(-7), max=128.0)

        max_pos_val = xg_base.amax(dim=-1, keepdim=True)
        min_val     = xg_base.amin(dim=-1, keepdim=True)
        sign_flag   = torch.where(
            max_pos_val.abs() >= min_val.abs(),
            torch.ones_like(max_pos_val),
            -torch.ones_like(max_pos_val),
        )
        q = (xg_base * sign_flag >= base / 2.0).float()
        r = xg_base - sign_flag * base * q
    else:
        # q_bits >= 2: signed Q
        q_max_val = (2 ** (q_bits - 1)) - 1
        threshold = max_abs / q_max_val
        base = torch.full_like(threshold, 64.0)
        base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
        base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
        base = torch.where(threshold <=  8.0, torch.full_like(threshold,  8.0), base)
        base = torch.where(threshold <=  4.0, torch.full_like(threshold,  4.0), base)
        base = torch.where(threshold <=  2.0, torch.full_like(threshold,  2.0), base)
        sign_flag = 1.0
        q = torch.round(xg_base / base).clamp(-q_max_val, q_max_val)
        r = xg_base - base * q

    q_scaled_base = q * sign_flag * base

    # Step 2: residual R quantization
    r_flat      = r.reshape(orig_shape)
    G_r         = H // r_gs
    r_for_quant = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
    max_abs_r   = r_for_quant.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    r_input     = r_for_quant

    if residual_clip_alpha > 0:
        # 극소수 residual tail이 scale_r 전체를 키우는 것을 막기 위한 base-relative cap
        base_flat       = base.expand_as(xg_base).reshape(orig_shape)
        base_for_r      = base_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
        clip_bound_elem = (float(residual_clip_alpha) * base_for_r).clamp_min(eps)
        clip_bound_grp  = clip_bound_elem.amax(dim=-1, keepdim=True).clamp_min(eps)
        max_abs_r       = torch.minimum(max_abs_r, clip_bound_grp).clamp_min(eps)
        r_input         = torch.maximum(
            torch.minimum(r_for_quant, clip_bound_elem),
            -clip_bound_elem,
        )

    scale_r  = max_abs_r / r_max_val
    r_q      = torch.round(r_input / scale_r).clamp(r_min_val, r_max_val)
    r_dq_grp = r_q * scale_r
    r_dq     = r_dq_grp.reshape(orig_shape)

    q_scaled = q_scaled_base.reshape(orig_shape)

    return q_scaled.to(x.dtype), r_dq.to(x.dtype)


# =========================================================
# QuotRemLinearV2: pure QR, no per-group fallback
# qr_core.py의 QuotRemLinear에서 selective 관련 분기/파라미터를 모두 제거.
# 모듈 단위로 QR을 적용할지 INT만 적용할지는 외부에서 결정.
# =========================================================
class QuotRemLinearV2(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        enable_weight_quant: bool = True,
        weight_bits: int = 4,
        weight_quant_mode: str = "group",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        weight_scale_method: str = "max",
        weight_scale_shrink_factors=None,
        q_bits: int = 1,
        r_bits: int = 3,
        base_group_size: int = 128,
        r_group_size: int = 128,
        residual_clip_alpha: float = 0.0,
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
        self.base_group_size     = base_group_size
        self.r_group_size        = r_group_size
        self.residual_clip_alpha = residual_clip_alpha

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
            # OmniQuant 사전 양자화 weight를 그대로 재사용. 큰 모델 메모리 절약.
            self.weight = base_linear.weight
            self.weight.requires_grad_(False)

        if base_linear.bias is not None:
            self.bias = base_linear.bias
            self.bias.requires_grad_(False)
        else:
            self.bias = None

    def forward(self, x):
        q_scaled, r_dq = qr_decompose_activation_v2(
            x,
            q_bits=self.q_bits,
            r_bits=self.r_bits,
            base_group_size=self.base_group_size,
            r_group_size=self.r_group_size,
            residual_clip_alpha=self.residual_clip_alpha,
        )
        q_out = F.linear(q_scaled, self.weight, None)
        r_out = F.linear(r_dq, self.weight, None)
        out = q_out + r_out
        if self.bias is not None:
            out = out + self.bias
        return out


# =========================================================
# IntActLinear: pure INT (no QR)
# 데이터 분석상 QR이 의미 없는 모듈(out_proj, o_proj 등)을 위한 단순 INT 양자화 경로.
# activation을 group-wise INT-N fake quant 후 linear 통과.
# =========================================================
class IntActLinear(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        enable_weight_quant: bool = True,
        weight_bits: int = 4,
        weight_quant_mode: str = "group",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        weight_scale_method: str = "max",
        weight_scale_shrink_factors=None,
        act_bits: int = 4,
        act_group_size: int = 128,
        debug_name: str = "",
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.debug_name   = debug_name

        self.act_bits       = act_bits
        self.act_group_size = act_group_size

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
            self.weight = base_linear.weight
            self.weight.requires_grad_(False)

        if base_linear.bias is not None:
            self.bias = base_linear.bias
            self.bias.requires_grad_(False)
        else:
            self.bias = None

    def forward(self, x):
        x_q = fake_quant_symmetric(
            x,
            n_bits=self.act_bits,
            mode="group",
            group_size=self.act_group_size,
        )
        out = F.linear(x_q, self.weight, self.bias)
        return out
