import torch
import torch.nn as nn
import torch.nn.functional as F

from qr_core_ver2 import fake_quant_symmetric, qr_decompose_activation_v2


def nearest_pow2_base(max_abs: torch.Tensor) -> torch.Tensor:
    # qr_core_ver2의 q_bits=1 base 선택과 동일한 nearest-P2 규칙.
    log2_max   = torch.log2(max_abs.clamp_min(1e-8))
    log2_floor = torch.floor(log2_max)
    log2_ceil  = torch.ceil(log2_max)
    base_floor = torch.pow(2.0, log2_floor)
    base_ceil  = torch.pow(2.0, log2_ceil)
    dist_floor = torch.abs(max_abs - base_floor)
    dist_ceil  = torch.abs(base_ceil  - max_abs)
    base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)
    base = base.clamp(min=2 ** (-7), max=128.0)
    return base


def compute_static_mask(per_group_max_abs: torch.Tensor, threshold: float) -> torch.Tensor:
    # Calibration에서 측정한 group별 max_abs 기준으로 nearest-P2 base를 구한 뒤,
    # base >= threshold 인 group은 QR, 그 외는 INT4 경로로 정적 분류한다.
    base_est = nearest_pow2_base(per_group_max_abs)
    return base_est >= float(threshold)


# =========================================================
# Phase 1: static base + static INT4 scale
#
# Calibration의 per_group_max_abs를 기반으로 base / INT scale을 미리 계산해 buffer로 저장.
# Runtime에는 group별 max reduction과 nearest-P2 계산이 모두 사라진다.
# Sign_flag와 scale_r는 여전히 dynamic (Phase 2/3 대상).
# =========================================================
def qr_decompose_static_base_phase1(
    x: torch.Tensor,
    static_base: torch.Tensor,    # shape [G_base], group마다 미리 계산된 P2 base
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

    G_base  = H // base_gs
    xg_base = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))

    # static_base shape [G_base] → broadcast 가능한 형태 [1, ..., G_base, 1]
    base = static_base.view(*([1] * (len(orig_shape) - 1)), G_base, 1).to(x_fp.dtype)

    if q_bits == 1:
        # sign_flag는 여전히 dynamic (Phase 2에서 정적화 예정)
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
        q_max_val = (2 ** (q_bits - 1)) - 1
        sign_flag = 1.0
        q = torch.round(xg_base / base).clamp(-q_max_val, q_max_val)
        r = xg_base - base * q

    q_scaled_base = q * sign_flag * base

    # Residual scale_r는 여전히 dynamic (Phase 3에서 정적화 예정)
    r_flat      = r.reshape(orig_shape)
    G_r         = H // r_gs
    r_for_quant = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
    max_abs_r   = r_for_quant.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    r_input     = r_for_quant

    if residual_clip_alpha > 0:
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


def int_fake_quant_static_scale(
    x: torch.Tensor,
    static_scale: torch.Tensor,   # shape [G_act], group마다 미리 계산된 scale
    n_bits: int = 4,
    act_group_size: int = 128,
):
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1
    x_fp = x.float()
    orig_shape = x_fp.shape
    H = orig_shape[-1]
    G = H // act_group_size
    xg = x_fp.reshape(orig_shape[:-1] + (G, act_group_size))
    scale = static_scale.view(*([1] * (len(orig_shape) - 1)), G, 1).to(x_fp.dtype)
    q = torch.round(xg / scale).clamp(qmin, qmax)
    dq = q * scale
    return dq.reshape(orig_shape).to(x.dtype)


class QuotRemLinearV3(nn.Module):
    """
    Channel reordering + per-group static format (QR or INT4).
      - perm:                calibration에서 결정된 channel permutation. weight도 init 시 동일 permute.
      - per_group_max_abs:   calibration에서 측정된 group별 max_abs 통계.
      - threshold:           runtime에서 받아 mask 결정 (QR vs INT4 per group).
      - use_static_scales:   Phase 1. True면 base / INT scale도 calibration으로 미리 고정.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        perm: torch.Tensor,                  # [in_features]
        per_group_max_abs: torch.Tensor,     # [G_base] = in_features / base_group_size
        threshold: float,
        # weight quant
        enable_weight_quant: bool = True,
        weight_bits: int = 4,
        weight_quant_mode: str = "group",
        weight_ch_axis: int = 0,
        weight_group_size: int = 128,
        weight_scale_method: str = "max",
        weight_scale_shrink_factors=None,
        # QR params
        q_bits: int = 1,
        r_bits: int = 3,
        base_group_size: int = 128,
        r_group_size: int = 128,
        residual_clip_alpha: float = 0.0,
        # INT4 params (mask=False group용)
        act_bits: int = 4,
        act_group_size: int = 128,
        # Phase 1 toggle
        use_static_scales: bool = False,
        debug_name: str = "",
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        in_features  = base_linear.in_features
        out_features = base_linear.out_features
        if perm.numel() != in_features:
            raise ValueError(
                f"perm length {perm.numel()} != in_features {in_features}"
            )
        if in_features % base_group_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by base_group_size={base_group_size}"
            )
        G_base = in_features // base_group_size
        if per_group_max_abs.numel() != G_base:
            raise ValueError(
                f"per_group_max_abs length {per_group_max_abs.numel()} != G_base {G_base}"
            )
        if act_group_size != base_group_size and use_static_scales:
            raise ValueError(
                "Phase 1: static INT4 scale은 act_group_size == base_group_size를 가정한다."
            )

        self.in_features         = in_features
        self.out_features        = out_features
        self.debug_name          = debug_name
        self.q_bits              = q_bits
        self.r_bits              = r_bits
        self.base_group_size     = base_group_size
        self.r_group_size        = r_group_size
        self.residual_clip_alpha = residual_clip_alpha
        self.act_bits            = act_bits
        self.act_group_size      = act_group_size
        self.use_static_scales   = use_static_scales

        # weight를 permute 후 (선택적으로) fake-quantize.
        w_perm = base_linear.weight.detach()[:, perm.long()]
        if enable_weight_quant:
            w_perm = fake_quant_symmetric(
                w_perm,
                n_bits=weight_bits,
                mode=weight_quant_mode,
                ch_axis=weight_ch_axis,
                group_size=weight_group_size,
                scale_method=weight_scale_method,
                scale_shrink_factors=weight_scale_shrink_factors,
            )
            self.weight = nn.Parameter(w_perm.contiguous(), requires_grad=False)
        else:
            self.weight = nn.Parameter(w_perm.contiguous(), requires_grad=False)
            self.weight.requires_grad_(False)

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
            self.bias.requires_grad_(False)
        else:
            self.bias = None

        target_device = self.weight.device
        mask = compute_static_mask(per_group_max_abs.float(), threshold).to(target_device)
        self.register_buffer("perm", perm.long().to(target_device))
        self.register_buffer("group_mask", mask)
        self.register_buffer(
            "per_group_max_abs", per_group_max_abs.float().to(target_device)
        )

        # Phase 1: 미리 계산된 static base와 static INT4 scale을 buffer로 저장.
        # use_static_scales=False여도 buffer는 등록해 두면 forward에서 같은 경로로 토글 가능.
        static_base = nearest_pow2_base(per_group_max_abs.float()).to(target_device)
        int_qmax    = (2 ** (act_bits - 1)) - 1
        static_int_scale = (per_group_max_abs.float() / int_qmax).clamp_min(1e-8).to(target_device)
        self.register_buffer("static_base", static_base)
        self.register_buffer("static_int_scale", static_int_scale)

    def forward(self, x):
        x_perm = x.index_select(-1, self.perm)

        if self.use_static_scales:
            # Phase 1: base / INT4 scale 모두 buffer 그대로 사용 (runtime max reduction 제거)
            q_scaled, r_dq = qr_decompose_static_base_phase1(
                x_perm,
                static_base=self.static_base,
                q_bits=self.q_bits,
                r_bits=self.r_bits,
                base_group_size=self.base_group_size,
                r_group_size=self.r_group_size,
                residual_clip_alpha=self.residual_clip_alpha,
            )
            int_dq = int_fake_quant_static_scale(
                x_perm,
                static_scale=self.static_int_scale,
                n_bits=self.act_bits,
                act_group_size=self.act_group_size,
            )
        else:
            # 기존 ver3 동작: base/scale 모두 dynamic
            q_scaled, r_dq = qr_decompose_activation_v2(
                x_perm,
                q_bits=self.q_bits,
                r_bits=self.r_bits,
                base_group_size=self.base_group_size,
                r_group_size=self.r_group_size,
                residual_clip_alpha=self.residual_clip_alpha,
            )
            int_dq = fake_quant_symmetric(
                x_perm,
                n_bits=self.act_bits,
                mode="group",
                group_size=self.act_group_size,
            )

        qr_dq = q_scaled + r_dq

        orig_shape = x_perm.shape
        H          = orig_shape[-1]
        G_base     = H // self.base_group_size
        new_shape  = orig_shape[:-1] + (G_base, self.base_group_size)
        qr_g       = qr_dq.reshape(new_shape)
        int_g      = int_dq.reshape(new_shape)
        mask_view  = self.group_mask.view(*([1] * (len(orig_shape) - 1)), G_base, 1)
        out_g      = torch.where(mask_view, qr_g, int_g)
        x_quant    = out_g.reshape(orig_shape)

        return F.linear(x_quant, self.weight, self.bias)
