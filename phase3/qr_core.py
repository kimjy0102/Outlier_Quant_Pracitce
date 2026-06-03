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
# Shared QR decomposition / analysis API
#
# 이 함수는 QuotRemLinear forward와 collect_qr_stats.py가 함께 쓰는
# 단일 QR 분해 구현입니다. 목적은 QR 수식을 여러 파일에 복사하지 않고,
# 실제 PPL 실행과 통계 수집이 항상 같은 base/q/r/fallback 규칙을 쓰게
# 만드는 것입니다.
#
# return_stats=False:
#   - 일반 PPL 실행 경로. q_scaled, r_dq만 반환합니다.
#   - stats용 중간 tensor dict를 만들지 않으므로 평소 실행 비용을 늘리지 않습니다.
#
# return_stats=True:
#   - collect_qr_stats.py 전용 분석 경로.
#   - base, q, raw residual, scale_r, fallback mask, r_q 등을 함께 반환합니다.
#   - 이 옵션은 통계 수집 스크립트에서만 켜는 것을 전제로 합니다.
# =========================================================
def qr_decompose_activation(
    x: torch.Tensor,
    q_bits: int = 4,
    r_bits: int = 4,
    base_group_size: int = 128,
    r_group_size: int = 128,
    selective_base_threshold: float = 1.0,
    selective_int_bits: int = 4,
    residual_clip_alpha: float = 0.0,
    return_stats: bool = False,
    eps: float = 1e-8,
):
    if selective_int_bits < 2:
        raise ValueError(f"selective_int_bits must be >= 2, got {selective_int_bits}")

    r_max_val = (2 ** (r_bits - 1)) - 1
    r_min_val = -r_max_val - 1

    x_fp = x.float()
    orig_shape = x_fp.shape
    H = orig_shape[-1]

    base_gs = H if base_group_size == -1 else base_group_size
    r_gs = H if r_group_size == -1 else r_group_size

    assert H % base_gs == 0, f"H={H} not divisible by base_group_size={base_gs}"
    assert H % r_gs == 0, f"H={H} not divisible by r_group_size={r_gs}"

    # =================== Step 1: base 결정 및 Q/R 분해 ===================
    G_base = H // base_gs
    xg_base = x_fp.reshape(orig_shape[:-1] + (G_base, base_gs))
    max_abs = xg_base.abs().amax(dim=-1, keepdim=True).clamp_min(eps)

    if q_bits == 1:
        # 1-bit Q는 unsigned {0, 1} correction path로 해석한다.
        # base는 group max_abs에 가장 가까운 power-of-two를 고른다.
        log2_max = torch.log2(max_abs)
        log2_floor = torch.floor(log2_max)
        log2_ceil = torch.ceil(log2_max)
        base_floor = torch.pow(2.0, log2_floor)
        base_ceil = torch.pow(2.0, log2_ceil)
        dist_floor = torch.abs(max_abs - base_floor)
        dist_ceil = torch.abs(base_ceil - max_abs)
        base = torch.where(dist_floor <= dist_ceil, base_floor, base_ceil)
        base = base.clamp(min=2**(-7), max=128.0)

        # sign_flag는 group의 가장 큰 절대값이 양수 쪽인지 음수 쪽인지 나타낸다.
        # q=1이면 sign*base를 sparse correction으로 더하고, q=0이면 residual만 남긴다.
        max_pos_val = xg_base.amax(dim=-1, keepdim=True)
        min_val = xg_base.amin(dim=-1, keepdim=True)
        sign_flag = torch.where(
            max_pos_val.abs() >= min_val.abs(),
            torch.ones_like(max_pos_val),
            -torch.ones_like(max_pos_val),
        )
        q = (xg_base * sign_flag >= base / 2.0).float()
        r = xg_base - sign_flag * base * q
    else:
        # q_bits>=2에서는 signed Q를 사용한다.
        q_max_val = (2 ** (q_bits - 1)) - 1
        threshold = max_abs / q_max_val
        base = torch.full_like(threshold, 64.0)
        base = torch.where(threshold <= 32.0, torch.full_like(threshold, 32.0), base)
        base = torch.where(threshold <= 16.0, torch.full_like(threshold, 16.0), base)
        base = torch.where(threshold <= 8.0, torch.full_like(threshold, 8.0), base)
        base = torch.where(threshold <= 4.0, torch.full_like(threshold, 4.0), base)
        base = torch.where(threshold <= 2.0, torch.full_like(threshold, 2.0), base)
        sign_flag = 1.0
        q = torch.round(xg_base / base).clamp(-q_max_val, q_max_val)
        r = xg_base - base * q

    # =================== Step 2: selective INT4 fallback ===================
    selective_mask = base < selective_base_threshold
    int_qmax_val = (2 ** (selective_int_bits - 1)) - 1
    int_qmin_val = -int_qmax_val - 1
    int_scale = (max_abs / int_qmax_val).clamp_min(eps)
    x_int_q = torch.round(xg_base / int_scale).clamp(int_qmin_val, int_qmax_val)
    x_int_dq = x_int_q * int_scale

    q_scaled_base = q * sign_flag * base

    # =================== Step 3: residual R quantization ===================
    r_flat = r.reshape(orig_shape)
    G_r = H // r_gs
    r_for_quant = r_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
    max_abs_r = r_for_quant.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    r_input = r_for_quant

    if residual_clip_alpha > 0:
        # residual clip은 QR 분해 후 극소수 residual tail이 scale_r 전체를 키우는 것을
        # 막기 위한 base-relative cap이다. 기본값 0이면 기존 max_abs scaling 그대로다.
        base_flat = base.expand_as(xg_base).reshape(orig_shape)
        base_for_r = base_flat.reshape(orig_shape[:-1] + (G_r, r_gs))
        clip_bound_elem = (float(residual_clip_alpha) * base_for_r).clamp_min(eps)
        clip_bound_grp = clip_bound_elem.amax(dim=-1, keepdim=True).clamp_min(eps)
        max_abs_r = torch.minimum(max_abs_r, clip_bound_grp).clamp_min(eps)
        r_input = torch.maximum(
            torch.minimum(r_for_quant, clip_bound_elem),
            -clip_bound_elem,
        )

    scale_r = max_abs_r / r_max_val
    r_q = torch.round(r_input / scale_r).clamp(r_min_val, r_max_val)
    r_dq_grp = r_q * scale_r
    r_dq = r_dq_grp.reshape(orig_shape)

    q_scaled = q_scaled_base.reshape(orig_shape)

    # fallback group은 q path를 끄고, residual path에 INT fake-quantized x를 넣는다.
    selective_flat = selective_mask.expand_as(xg_base).reshape(orig_shape)
    x_int_dq_flat = x_int_dq.reshape(orig_shape)
    q_scaled = torch.where(selective_flat, torch.zeros_like(q_scaled), q_scaled)
    r_dq = torch.where(selective_flat, x_int_dq_flat, r_dq)

    q_scaled = q_scaled.to(x.dtype)
    r_dq = r_dq.to(x.dtype)

    if not return_stats:
        return q_scaled, r_dq

    q_active_count = (q != 0).sum(dim=-1).float()
    q_active_count_effective = torch.where(
        selective_mask.squeeze(-1),
        torch.zeros_like(q_active_count),
        q_active_count,
    )

    stats = {
        "base": base,
        "q": q,
        "q_active_count": q_active_count,
        "q_active_count_effective": q_active_count_effective,
        "r_raw": r_flat,
        "r_input": r_input.reshape(orig_shape),
        "r_q": r_q,
        "r_dq": r_dq,
        "scale_r": scale_r,
        "max_abs": max_abs,
        "max_abs_r": max_abs_r,
        "selective_mask": selective_mask,
        "selective_flat": selective_flat,
        "x_int_dq": x_int_dq_flat.to(x.dtype),
        "base_group_size": base_gs,
        "r_group_size": r_gs,
        "num_base_groups": G_base,
        "num_r_groups": G_r,
    }
    return q_scaled, r_dq, stats


# =========================================================
# Power-of-2 Adaptive QR Linear
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
        weight_scale_shrink_factors=None,
        q_bits: int = 4,
        r_bits: int = 4,
        base_group_size: int = 128,        # base(나눌 값)를 결정할 group 크기 (-1이면 per-tensor)
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
            # In no-weight-quant mode, keep the original Parameter instead of cloning it.
            # This avoids a large temporary duplicate when replacing all OPT/Llama layers.
            self.weight = base_linear.weight
            self.weight.requires_grad_(False)

        if base_linear.bias is not None:
            self.bias = base_linear.bias
            self.bias.requires_grad_(False)
        else:
            self.bias = None

    def _adaptive_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        실제 forward 경로.
        QR 수식 자체는 qr_decompose_activation()에만 두고, 여기서는
        QuotRemLinear 인스턴스의 설정값을 넘겨 호출한다.
        """
        result = qr_decompose_activation(
            x,
            q_bits=self.q_bits,
            r_bits=self.r_bits,
            base_group_size=self.base_group_size,
            r_group_size=self.r_group_size,
            selective_base_threshold=self.selective_base_threshold,
            selective_int_bits=self.selective_int_bits,
            residual_clip_alpha=self.residual_clip_alpha,
            return_stats=self.collect_residuals,
        )

        if not self.collect_residuals:
            return result

        q_scaled, r_dq, stats = result
        self._append_analysis_buffers(stats)
        return q_scaled, r_dq

    def _num_buffer_values(self, buffers):
        return sum(t.numel() for t in buffers)

    def _append_sampled_1d(self, buffers, values):
        current = self._num_buffer_values(buffers)
        if current >= self._max_r_samples:
            return
        values = values.detach().float().cpu().flatten()
        remain = self._max_r_samples - current
        n_take = min(values.numel(), remain, 10_000)
        if n_take <= 0:
            return
        idx = torch.randperm(values.numel())[:n_take]
        buffers.append(values[idx])

    def _append_analysis_buffers(self, stats):
        """
        기존 collect_residuals=True 경로와의 호환을 위한 샘플링 버퍼 업데이트.
        collect_qr_stats.py는 return_stats=True dict를 직접 쓰지만, 예전처럼
        QuotRemLinear 내부 버퍼를 확인하는 분석 코드도 계속 동작하게 둔다.
        """
        base = stats["base"]
        q = stats["q"]
        r_raw = stats["r_raw"]
        base_gs = float(stats["base_group_size"])

        group_buf_n = sum(t.shape[0] for t in self._base_group_buf)
        if group_buf_n < self._max_r_samples:
            base_cpu = base.detach().float().cpu().flatten()
            q1_count_cpu = stats["q_active_count"].detach().float().cpu().flatten()
            q0_count_cpu = base_gs - q1_count_cpu
            q1_frac_cpu = q1_count_cpu / base_gs
            remain = self._max_r_samples - group_buf_n
            n_take = min(base_cpu.numel(), remain, 10_000)
            if n_take > 0:
                idx = torch.randperm(base_cpu.numel())[:n_take]
                group_stats = torch.stack(
                    [base_cpu[idx], q1_count_cpu[idx], q0_count_cpu[idx], q1_frac_cpu[idx]], dim=1
                )
                self._base_group_buf.append(group_stats)

        self._append_sampled_1d(self._base_buf, base)
        self._append_sampled_1d(self._r_buf, r_raw)
        self._append_sampled_1d(self._q_buf, q)

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
