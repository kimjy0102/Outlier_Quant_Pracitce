import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from eval_utils import load_eval_tokens
from model_adapter import (
    get_decoder_layers,
    get_model_device,
    get_named_linear,
    load_model_and_tokenizer,
    parse_module_names,
    resolve_target_layers,
)
from model_configs import get_module_names
from qr_core import fake_quant_symmetric, qr_decompose_activation


# =========================================================
# Small IO helpers
# =========================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    return finite_or_none(obj)


def tensor_sample_1d(tensor, max_take):
    flat = tensor.detach().flatten()
    if flat.numel() == 0 or max_take <= 0:
        return torch.empty(0)
    n_take = min(flat.numel(), max_take)
    if n_take == flat.numel():
        return flat.float().cpu()
    idx = torch.randperm(flat.numel(), device=flat.device)[:n_take]
    return flat[idx].float().cpu()


def concat_plot_samples(accumulators, sample_attr, max_total=2_000_000):
    # CSV/JSON 통계는 module별 cap 전체를 사용하지만, preview PNG는 너무 많은
    # sample을 한 번에 torch.quantile/hist에 넣으면 PyTorch 한계를 넘을 수 있다.
    # 그래서 plot용으로만 전체 sample을 균등하게 줄인다.
    chunks = []
    for acc in accumulators:
        samples = acc._cat_samples(getattr(acc, sample_attr))
        if samples.numel() > 0:
            chunks.append(samples)

    if not chunks:
        return torch.empty(0)

    total = sum(t.numel() for t in chunks)
    if total <= max_total:
        return torch.cat([t.float().cpu() for t in chunks])

    per_chunk = max(1, math.ceil(max_total / len(chunks)))
    sampled = [tensor_sample_1d(t, min(per_chunk, t.numel())) for t in chunks]
    merged = torch.cat([t for t in sampled if t.numel() > 0])
    if merged.numel() > max_total:
        merged = tensor_sample_1d(merged, max_total)
    return merged.float().cpu()


def quantile_dict(values, prefix):
    if values.numel() == 0:
        return {
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_p999": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    qs = torch.tensor([0.5, 0.9, 0.99, 0.999])
    qv = torch.quantile(values.float(), qs)
    return {
        f"{prefix}_p50": float(qv[0].item()),
        f"{prefix}_p90": float(qv[1].item()),
        f"{prefix}_p99": float(qv[2].item()),
        f"{prefix}_p999": float(qv[3].item()),
        f"{prefix}_max": float(values.max().item()),
    }


def safe_div(num, den):
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def sqnr_db(signal_sum, error_sum):
    if error_sum <= 0:
        return float("inf")
    if signal_sum <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal_sum / error_sum)


def base_threshold_to_max_abs(threshold, q_bits):
    """selective_base_threshold (= base 비교용 임계값)가 max_abs 축에서 어디에 해당하는지 계산.
    plot에서 'base >= threshold' 경계를 max_abs 축에 그릴 때 사용한다.

    qr_core.qr_decompose_activation 의 base 결정 로직과 정확히 정합:
      - q_bits == 1 (qr_core.py:131-142): nearest-power-of-two를 **선형(절대)거리** 기준으로 선택.
          base = floor_p2 if |max_abs - floor_p2| ≤ |ceil_p2 - max_abs| else ceil_p2
          → base가 target=ceil_p2(threshold) 가 되는 boundary는 prev=target/2 와의 산술 중점:
            max_abs = (prev + target) / 2     (예: thr=8 → target=8, prev=4 → boundary=6.0)
      - q_bits >= 2 (qr_core.py:155-167): t = max_abs / q_max_val 을 ceil-pow2 처리해 base 결정.
          base = target 이 되는 영역은 prev < t ≤ target → prev*q_max_val < max_abs ≤ target*q_max_val
          → base ≥ threshold boundary: max_abs > prev * q_max_val
    """
    if threshold <= 0:
        return 0.0
    # threshold 이상의 최소 2의 거듭제곱(= 실제 선택될 base 값)
    target = 2.0 ** math.ceil(math.log2(float(threshold)))
    prev = target / 2.0
    if q_bits == 1:
        return 0.5 * (prev + target)  # 선형 거리 기준 nearest-p2 boundary
    q_max_val = (2 ** (int(q_bits) - 1)) - 1
    if q_max_val <= 0:
        return float(target)
    return float(prev) * float(q_max_val)


# =========================================================
# Per-module accumulator
#
# Hook은 forward 중 Linear 입력 activation x를 받는다.
# 여기서 qr_core.qr_decompose_activation(return_stats=True)를 호출해
# 실제 phase3 QR 규칙과 동일한 base/q/r/fallback 결과를 얻고, 숫자 통계를
# 누적한다. PNG/CSV용 raw 값은 전부 저장하지 않고 module별 cap만 샘플한다.
# =========================================================
class ModuleAccumulator:
    def __init__(self, layer_idx, module_name, args):
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.args = args

        self.n_values = 0
        self.n_groups = 0
        self.fallback_groups = 0

        self.sum_x2 = 0.0
        self.sum_err_qr = 0.0
        self.sum_err_int4 = 0.0

        self.q_active_total = 0.0
        self.q_active_effective_total = 0.0
        self.q_total = 0
        self.q_count_sum = 0.0

        self.r_sat_count = 0
        self.r_q_count = 0

        self.base_sum = 0.0
        self.base_max = float("-inf")
        self.base_counts = defaultdict(int)

        self.x_abs_samples = []
        self.r_abs_samples = []
        self.x_signed_samples = []
        self.r_signed_samples = []
        self.q1_count_samples = []

        self.base_q = defaultdict(
            lambda: {
                "n_groups": 0,
                "q1_count_sum": 0.0,
                "fallback_groups": 0,
                "q1_samples": [],
            }
        )

        # === 채널별 signed 통계 (lazy init: 첫 update에서 H가 확정될 때 할당) ===
        # 모든 토큰을 channel(last dim)별로 누적해서 channel별 min/max/mean/std 계산.
        # H가 모듈마다 달라서 첫 update에서 X.size(-1)을 보고 buffer 할당.
        # NOTE: 채널별 kurtosis는 LLM activation outlier (특정 채널이 항상 큰 magnitude를 가지는
        # systematic outlier)를 잘 잡지 못한다. 대신 per-token cross-channel kurtosis를 따로 측정.
        self.channel_n = 0          # 채널별 element 수 (= 누적 token 수)
        self.channel_min = None     # [H], running min
        self.channel_max = None     # [H], running max
        self.channel_sum_1 = None   # [H], sum(x) (mean 계산용)
        self.channel_sum_2 = None   # [H], sum(x^2) (std 계산용)

        # === Per-token cross-channel kurtosis ===
        # 한 token의 H개 채널 값 분포의 kurtosis를 계산. 양자화 단위(token × channels)와 정합되고
        # SmoothQuant / AWQ 가 인용하는 표준 cross-channel outlier metric.
        # 한 update의 N tokens에서 N개의 kurt 값을 얻고, reservoir에 저장한다.
        self.per_token_kurt_samples = []   # list of 1D tensors (float)
        self.max_per_token_kurt_samples = 200_000

        # === 그룹별 sign-balance / opposite-sign / outlier 통계 ===
        # 각 base group g (= 한 토큰 × 한 그룹) 에 대해:
        #   pos_max_g = group.amax (음수만 있으면 0)
        #   neg_max_g = -group.amin (양수만 있으면 0)
        #   sign_balance = min(pos, neg) / max(pos, neg) ∈ [0,1]
        #     ≈0: 한쪽 부호로 outlier 쏠림 (QR 유리)
        #     ≈1: 양쪽 균형 — 단, 두 값이 둘 다 작으면 무해, 둘 다 크면 위험
        #   opposite_sign_large: dominant 부호의 base/2 이상인 minor 부호 값 존재 여부
        #   outlier_group: base >= selective_base_threshold 인 그룹
        #     (qr_core.py:170 의 selective_mask = (base < threshold) 와 정확히 정합되는 QR-active 그룹)
        #   raw_outlier_group: max_abs >= selective_base_threshold (참고용 sanity check)
        #     → q_bits=1 nearest-p2 일 때 max_abs ≈ thr/sqrt(2) 이상이면 base≥thr이 됨
        #   dangerous_group: outlier_group AND sign_balance >= 0.5 (양쪽 outlier 공존 위험)
        self.group_sign_balance_sum = 0.0
        self.group_opposite_large_count = 0   # base/2 이상 minor 부호 존재 group 수
        # outlier 기준은 qr_core.py:170의 selective_mask = (base < threshold) 와 정확히 정합되도록
        # `base_g >= selective_base_threshold` (= QR이 실제 적용되는 그룹) 로 정의한다.
        # 추가로 raw_outlier (max_abs >= threshold)는 보조 sanity check용.
        self.group_outlier_count = 0          # base >= threshold (QR 적용 그룹)
        self.group_raw_outlier_count = 0      # max_abs >= threshold (참고용)
        self.group_dangerous_count = 0        # base-outlier AND sign_balance >= dangerous_sb_threshold
        self.group_outlier_sign_balance_sum = 0.0  # base-outlier 그룹들만의 sign_balance 합
        # reservoir samples (sign_balance, pos_max, neg_max, max_abs, base, is_outlier, is_dangerous)
        self.group_sign_balance_samples = []  # 1D tensors
        self.group_pos_max_samples = []
        self.group_neg_max_samples = []
        self.group_max_abs_samples = []
        self.group_base_samples = []
        self.max_sign_balance_samples = 200_000
        self.dangerous_sb_threshold = 0.5     # dangerous 그룹 기준 (코드에서 고정, args.selective_base_threshold는 outlier 기준)

    def _sample_room(self, current_samples):
        current = sum(t.numel() for t in current_samples)
        return max(0, self.args.max_samples_per_module - current)

    def _append_sample(self, target, values):
        room = self._sample_room(target)
        if room <= 0:
            return
        target.append(tensor_sample_1d(values, min(room, self.args.sample_chunk)))

    def update(self, x):
        x_fp = x.detach().float()
        q_scaled, r_dq, stats = qr_decompose_activation(
            x,
            q_bits=self.args.q_bits,
            r_bits=self.args.r_bits,
            base_group_size=self.args.base_group_size,
            r_group_size=self.args.r_group_size,
            selective_base_threshold=self.args.selective_base_threshold,
            selective_int_bits=self.args.selective_int_bits,
            residual_clip_alpha=self.args.residual_clip_alpha,
            return_stats=True,
        )

        qr_hat = q_scaled.float() + r_dq.float()
        int_group_size = self.args.int_group_size
        x_int4 = fake_quant_symmetric(
            x,
            n_bits=self.args.int_bits,
            mode="group",
            group_size=int_group_size,
        ).float()

        self.n_values += x_fp.numel()
        self.sum_x2 += float(x_fp.pow(2).sum().item())
        self.sum_err_qr += float((x_fp - qr_hat).pow(2).sum().item())
        self.sum_err_int4 += float((x_fp - x_int4).pow(2).sum().item())

        q = stats["q"]
        q_scaled_eff = q_scaled.float()
        self.q_total += q.numel()
        self.q_active_total += float((q != 0).sum().item())
        self.q_active_effective_total += float((q_scaled_eff != 0).sum().item())

        selective_mask = stats["selective_mask"]
        self.n_groups += selective_mask.numel()
        self.fallback_groups += int(selective_mask.sum().item())

        q_count_eff = stats["q_active_count_effective"].detach().float()
        self.q_count_sum += float(q_count_eff.sum().item())

        r_q = stats["r_q"].detach()
        r_max_val = (2 ** (self.args.r_bits - 1)) - 1
        r_min_val = -r_max_val - 1
        self.r_sat_count += int(((r_q <= r_min_val) | (r_q >= r_max_val)).sum().item())
        self.r_q_count += r_q.numel()

        base = stats["base"].detach().float()
        self.base_sum += float(base.sum().item())
        self.base_max = max(self.base_max, float(base.max().item()))

        base_cpu = base.flatten().cpu()
        q_count_cpu = q_count_eff.flatten().cpu()
        fallback_cpu = selective_mask.flatten().cpu()
        for b, q_cnt, fb in zip(base_cpu.tolist(), q_count_cpu.tolist(), fallback_cpu.tolist()):
            key = float(b)
            self.base_counts[key] += 1
            item = self.base_q[key]
            item["n_groups"] += 1
            item["q1_count_sum"] += float(q_cnt)
            item["fallback_groups"] += int(bool(fb))
            if len(item["q1_samples"]) < self.args.max_q1_samples_per_base:
                item["q1_samples"].append(float(q_cnt))

        self._append_sample(self.x_abs_samples, x_fp.abs())
        self._append_sample(self.r_abs_samples, stats["r_raw"].detach().float().abs())
        self._append_sample(self.x_signed_samples, x_fp)
        self._append_sample(self.r_signed_samples, stats["r_raw"].detach().float())
        self._append_sample(self.q1_count_samples, q_count_eff)

        # =====================================================================
        # 채널별 signed 통계 누적 (모든 토큰을 channel별로 합산)
        # x shape: (..., H). flatten해서 (N_tokens, H)로 변형.
        # =====================================================================
        x_2d = x_fp.reshape(-1, x_fp.shape[-1])  # (N, H) on whatever device
        H = x_2d.shape[1]
        N = x_2d.shape[0]
        # buffer lazy init (CPU에 두어 GPU 메모리 절약)
        if self.channel_min is None:
            self.channel_min = torch.full((H,), float("inf"))
            self.channel_max = torch.full((H,), float("-inf"))
            self.channel_sum_1 = torch.zeros(H, dtype=torch.float64)
            self.channel_sum_2 = torch.zeros(H, dtype=torch.float64)

        x_cpu = x_2d.detach().float().cpu()
        self.channel_n += N
        cur_min = x_cpu.amin(dim=0)
        cur_max = x_cpu.amax(dim=0)
        self.channel_min = torch.minimum(self.channel_min, cur_min)
        self.channel_max = torch.maximum(self.channel_max, cur_max)
        x_cpu_64 = x_cpu.to(torch.float64)
        self.channel_sum_1 += x_cpu_64.sum(dim=0)
        self.channel_sum_2 += (x_cpu_64 * x_cpu_64).sum(dim=0)

        # === Per-token cross-channel kurtosis ===
        # 각 row (한 token)에 대해 H 채널 값들의 kurtosis 계산.
        # outlier channel이 dominant하면 한 token 안 분포의 꼬리가 두꺼움 → kurt 큼.
        # CIM/QR의 양자화 단위(한 token × channels)와 정합된 outlier 측정.
        # x_cpu_64: (N, H), per-row kurt
        row_mean = x_cpu_64.mean(dim=1, keepdim=True)              # (N, 1)
        x_centered = x_cpu_64 - row_mean
        row_var = (x_centered ** 2).mean(dim=1)                    # (N,)
        row_m4 = (x_centered ** 4).mean(dim=1)                     # (N,)
        # var가 거의 0인 row는 NaN 처리 (모든 채널이 같은 값일 때 발생 — 사실상 없음)
        kurt_per_token = torch.where(
            row_var > 1e-20,
            row_m4 / (row_var ** 2),
            torch.full_like(row_var, float("nan")),
        ).float()
        # reservoir-style: 매 update마다 sample을 모으고 cap 초과 시 random drop
        cap = self.max_per_token_kurt_samples
        if N > 0:
            kurt_finite = kurt_per_token[torch.isfinite(kurt_per_token)]
            if kurt_finite.numel() > 0:
                self.per_token_kurt_samples.append(kurt_finite)
                current = sum(t.numel() for t in self.per_token_kurt_samples)
                if current > cap:
                    all_kurt = torch.cat(self.per_token_kurt_samples)
                    keep_idx = torch.randperm(all_kurt.numel())[:cap]
                    self.per_token_kurt_samples = [all_kurt[keep_idx]]

        # =====================================================================
        # 그룹별 sign-balance / opposite-sign 통계
        # base group으로 reshape: (..., G, base_gs). G 차원으로 그룹 단위 통계.
        # =====================================================================
        base_gs = self.args.base_group_size
        if base_gs == -1:
            base_gs = H
        if H % base_gs == 0:
            num_g = H // base_gs
            xg = x_fp.reshape(-1, num_g, base_gs)  # (N, G, gs)
            pos_max = xg.amax(dim=-1)              # (N, G)
            neg_max = (-xg.amin(dim=-1))           # (N, G)
            # pos_max/neg_max 둘 다 음수가 될 수 있음 (그룹 전체가 한 부호로 쏠릴 때)
            pos_max = pos_max.clamp(min=0.0)
            neg_max = neg_max.clamp(min=0.0)
            sb_num = torch.minimum(pos_max, neg_max)
            sb_den = torch.maximum(pos_max, neg_max).clamp_min(1e-12)
            sign_balance = sb_num / sb_den         # (N, G), ∈ [0, 1]
            max_abs_g = torch.maximum(pos_max, neg_max)  # (N, G)

            # base는 stats["base"]에서 받음 (shape: (..., G, 1))
            base_g = stats["base"].detach().float().squeeze(-1)  # (..., G) usually (N, G)
            base_g = base_g.reshape(sign_balance.shape)
            # opposite-sign large: minor 부호 max가 base/2 이상이면 True
            minor_max = sb_num
            opposite_large = (minor_max >= base_g / 2.0)

            # outlier / dangerous group counts
            #   outlier_group: 그룹의 base가 selective threshold 이상 — qr_core.py:170의
            #                  selective_mask = (base < threshold) 와 정확히 정합 (QR 실제 적용 그룹).
            #   raw_outlier_group: max_abs >= threshold — 참고용 sanity check.
            #   dangerous_group: outlier_group 중 sign_balance가 큰 (양쪽 outlier 공존) 그룹.
            outlier_thr = float(self.args.selective_base_threshold)
            outlier_mask = (base_g >= outlier_thr)
            raw_outlier_mask = (max_abs_g >= outlier_thr)
            dangerous_mask = outlier_mask & (sign_balance >= self.dangerous_sb_threshold)

            n_groups_batch = int(sign_balance.numel())
            self.group_sign_balance_sum += float(sign_balance.sum().item())
            self.group_opposite_large_count += int(opposite_large.sum().item())
            self.group_outlier_count += int(outlier_mask.sum().item())
            self.group_raw_outlier_count += int(raw_outlier_mask.sum().item())
            self.group_dangerous_count += int(dangerous_mask.sum().item())
            # outlier 그룹들만의 sign_balance 합산 (평균 계산용)
            if outlier_mask.any():
                self.group_outlier_sign_balance_sum += float(
                    (sign_balance * outlier_mask.float()).sum().item()
                )

            # === reservoir-style sampling ===
            # 항상 batch에서 일정 수만큼 random pick하여 추가하고, cap 초과 시 합쳐서 random drop.
            # 이렇게 하면 calibration window 후반 데이터도 균등하게 반영됨 (앞쪽 bias 방지).
            cap = self.max_sign_balance_samples
            per_batch_max = 50_000
            sb_flat = sign_balance.detach().flatten()
            pm_flat = pos_max.detach().flatten()
            nm_flat = neg_max.detach().flatten()
            ma_flat = max_abs_g.detach().flatten()
            bg_flat = base_g.detach().flatten()
            n_avail = sb_flat.numel()
            take = min(n_avail, per_batch_max)
            if take > 0:
                idx = torch.randperm(n_avail, device=sb_flat.device)[:take]
                self.group_sign_balance_samples.append(sb_flat[idx].float().cpu())
                self.group_pos_max_samples.append(pm_flat[idx].float().cpu())
                self.group_neg_max_samples.append(nm_flat[idx].float().cpu())
                self.group_max_abs_samples.append(ma_flat[idx].float().cpu())
                self.group_base_samples.append(bg_flat[idx].float().cpu())

                # cap 초과 시: 모든 series 동일 index로 random drop (cross-series 정합성 보존)
                current = sum(t.numel() for t in self.group_sign_balance_samples)
                if current > cap:
                    all_sb = torch.cat(self.group_sign_balance_samples)
                    all_pm = torch.cat(self.group_pos_max_samples)
                    all_nm = torch.cat(self.group_neg_max_samples)
                    all_ma = torch.cat(self.group_max_abs_samples)
                    all_bg = torch.cat(self.group_base_samples)
                    keep_idx = torch.randperm(all_sb.numel())[:cap]
                    self.group_sign_balance_samples = [all_sb[keep_idx]]
                    self.group_pos_max_samples      = [all_pm[keep_idx]]
                    self.group_neg_max_samples      = [all_nm[keep_idx]]
                    self.group_max_abs_samples      = [all_ma[keep_idx]]
                    self.group_base_samples         = [all_bg[keep_idx]]

    def _cat_samples(self, samples):
        if not samples:
            return torch.empty(0)
        return torch.cat(samples)

    def module_row(self, config_row):
        x_abs = self._cat_samples(self.x_abs_samples)
        r_abs = self._cat_samples(self.r_abs_samples)
        row = dict(config_row)
        row.update(
            {
                "layer": self.layer_idx,
                "module": self.module_name,
                "n_values": self.n_values,
                "n_groups": self.n_groups,
            }
        )
        row.update(quantile_dict(x_abs, "x_abs"))
        row.update(quantile_dict(r_abs, "r_abs"))

        row["r_p99_shrink"] = safe_div(row["r_abs_p99"], row["x_abs_p99"])
        row["r_max_shrink"] = safe_div(row["r_abs_max"], row["x_abs_max"])
        row["base_mean"] = safe_div(self.base_sum, self.n_groups)
        row["base_mode"] = max(self.base_counts.items(), key=lambda kv: kv[1])[0] if self.base_counts else float("nan")
        row["base_max"] = self.base_max
        row["fallback_ratio"] = safe_div(self.fallback_groups, self.n_groups)
        row["q1_ratio_raw"] = safe_div(self.q_active_total, self.q_total)
        row["q1_ratio"] = safe_div(self.q_active_effective_total, self.q_total)
        row["avg_q1_count"] = safe_div(self.q_count_sum, self.n_groups)
        row["mse_int4"] = safe_div(self.sum_err_int4, self.n_values)
        row["mse_qr"] = safe_div(self.sum_err_qr, self.n_values)
        row["sqnr_int4"] = sqnr_db(self.sum_x2, self.sum_err_int4)
        row["sqnr_qr"] = sqnr_db(self.sum_x2, self.sum_err_qr)
        row["r_saturation_ratio"] = safe_div(self.r_sat_count, self.r_q_count)

        # === Per-token cross-channel kurtosis 요약 ===
        # 한 token 안의 H개 channel 분포의 kurtosis. CIM/QR의 양자화 단위와 정합.
        per_token_kurt = self._cat_samples(self.per_token_kurt_samples)
        if per_token_kurt.numel() > 0:
            row["per_token_kurt_mean"] = float(per_token_kurt.mean().item())
            qs = torch.quantile(per_token_kurt, torch.tensor([0.5, 0.9, 0.99]))
            row["per_token_kurt_p50"]  = float(qs[0].item())
            row["per_token_kurt_p90"]  = float(qs[1].item())
            row["per_token_kurt_p99"]  = float(qs[2].item())
        else:
            row["per_token_kurt_mean"] = float("nan")
            row["per_token_kurt_p50"]  = float("nan")
            row["per_token_kurt_p90"]  = float("nan")
            row["per_token_kurt_p99"]  = float("nan")

        # === 그룹별 sign-balance / opposite-sign / outlier / dangerous ===
        sb_samples = self._cat_samples(self.group_sign_balance_samples)
        if self.n_groups > 0:
            row["sign_balance_mean"] = safe_div(self.group_sign_balance_sum, self.n_groups)
            row["opposite_sign_ratio"] = safe_div(self.group_opposite_large_count, self.n_groups)
            # outlier_group_ratio: base >= threshold (QR이 실제로 적용되는 그룹). qr_core.py:170과 정확히 정합.
            row["outlier_group_ratio"] = safe_div(self.group_outlier_count, self.n_groups)
            # raw_outlier_group_ratio: max_abs >= threshold (참고용)
            row["raw_outlier_group_ratio"] = safe_div(self.group_raw_outlier_count, self.n_groups)
            row["dangerous_group_ratio"] = safe_div(self.group_dangerous_count, self.n_groups)
        else:
            row["sign_balance_mean"] = float("nan")
            row["opposite_sign_ratio"] = float("nan")
            row["outlier_group_ratio"] = float("nan")
            row["raw_outlier_group_ratio"] = float("nan")
            row["dangerous_group_ratio"] = float("nan")
        if sb_samples.numel() > 0:
            qs = torch.quantile(sb_samples, torch.tensor([0.5, 0.9, 0.99]))
            row["sign_balance_p50"] = float(qs[0].item())
            row["sign_balance_p90"] = float(qs[1].item())
            row["sign_balance_p99"] = float(qs[2].item())
        else:
            row["sign_balance_p50"] = float("nan")
            row["sign_balance_p90"] = float("nan")
            row["sign_balance_p99"] = float("nan")
        # outlier 그룹들만의 sign_balance (전체 그룹 기준 mean과 별도)
        row["outlier_sign_balance_mean"] = safe_div(
            self.group_outlier_sign_balance_sum, self.group_outlier_count
        )
        # outlier 그룹만의 sign_balance p90: 저장된 sample을 base >= threshold mask로 filter (QR 정합).
        base_samples = self._cat_samples(self.group_base_samples)
        if sb_samples.numel() > 0 and base_samples.numel() == sb_samples.numel():
            mask = base_samples >= float(self.args.selective_base_threshold)
            if mask.any():
                row["outlier_sign_balance_p90"] = float(
                    torch.quantile(sb_samples[mask], 0.9).item()
                )
            else:
                row["outlier_sign_balance_p90"] = float("nan")
        else:
            row["outlier_sign_balance_p90"] = float("nan")

        return row

    def base_q_rows(self, config_row):
        rows = []
        for base, item in sorted(self.base_q.items(), key=lambda kv: kv[0]):
            samples = torch.tensor(item["q1_samples"], dtype=torch.float32)
            if samples.numel() > 0:
                p50 = float(torch.quantile(samples, 0.5).item())
                p90 = float(torch.quantile(samples, 0.9).item())
            else:
                p50 = float("nan")
                p90 = float("nan")
            row = dict(config_row)
            row.update(
                {
                    "layer": self.layer_idx,
                    "module": self.module_name,
                    "base": base,
                    "n_groups": item["n_groups"],
                    "q1_count_mean": safe_div(item["q1_count_sum"], item["n_groups"]),
                    "q1_count_p50": p50,
                    "q1_count_p90": p90,
                    "q1_frac_mean": safe_div(
                        item["q1_count_sum"],
                        item["n_groups"] * self.args.base_group_size,
                    ),
                    "fallback_ratio": safe_div(item["fallback_groups"], item["n_groups"]),
                }
            )
            rows.append(row)
        return rows


# =========================================================
# CSV/plot materialization
# =========================================================
def make_hist_rows(accumulators, sample_attr, value_name, bins):
    rows = []
    for acc in accumulators:
        samples = acc._cat_samples(getattr(acc, sample_attr))
        if samples.numel() == 0:
            continue
        max_val = float(samples.max().item())
        if max_val <= 0:
            max_val = 1.0
        counts = torch.histc(samples.float(), bins=bins, min=0.0, max=max_val)
        width = max_val / bins
        for idx, count in enumerate(counts.tolist()):
            rows.append(
                {
                    "layer": acc.layer_idx,
                    "module": acc.module_name,
                    "value": value_name,
                    "bin_idx": idx,
                    "bin_left": idx * width,
                    "bin_right": (idx + 1) * width,
                    "count": int(count),
                }
            )
    return rows


def make_signed_hist_rows(accumulators, sample_attr, value_name, bins, fixed_width=None):
    """
    Signed activation/residual의 histogram을 0 중심으로 대칭 bin으로 생성.
    fixed_width=None이면 module별 max_abs에 맞춘 bin width 자동 계산.
    fixed_width=float이면 모든 module에 동일 bin width (예: 1.0 → -X.0~-(X-1).0, ... 0~1, 1~2, ...)
    → "0~2, 2~4, ..., 30~40, 50~60" 처럼 일정 폭의 bin을 원하면 fixed_width를 지정.
    """
    rows = []
    for acc in accumulators:
        samples = acc._cat_samples(getattr(acc, sample_attr))
        if samples.numel() == 0:
            continue
        max_abs_val = float(samples.float().abs().max().item())
        if max_abs_val <= 0:
            max_abs_val = 1.0

        if fixed_width is None:
            n_bins = bins
            width = (2.0 * max_abs_val) / n_bins
            edges_min = -max_abs_val
            edges_max = max_abs_val
        else:
            # fixed_width 단위로 max_abs를 덮을 수 있는 bin 수 계산 (양쪽 대칭)
            half = math.ceil(max_abs_val / fixed_width)
            n_bins = int(2 * half)
            width = float(fixed_width)
            edges_min = -half * width
            edges_max = half * width

        counts = torch.histc(samples.float(), bins=n_bins, min=edges_min, max=edges_max)
        for idx, count in enumerate(counts.tolist()):
            bin_left  = edges_min + idx * width
            bin_right = edges_min + (idx + 1) * width
            rows.append(
                {
                    "layer": acc.layer_idx,
                    "module": acc.module_name,
                    "value": value_name,
                    "bin_idx": idx,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "count": int(count),
                }
            )
    return rows


def make_channel_signed_rows(accumulators):
    """채널별 signed min/max/mean/std row. (layer, module, channel) 단위.
    NOTE: 채널별 kurtosis는 systematic outlier (특정 채널의 항상 큰 magnitude)를 못 잡아서 제외.
    Outlier 측정은 per-token cross-channel kurt (module_stats.csv의 per_token_kurt_*)로 통일."""
    rows = []
    for acc in accumulators:
        if acc.channel_n <= 0 or acc.channel_min is None:
            continue
        n = float(acc.channel_n)
        mean = (acc.channel_sum_1 / n).float()
        var = (acc.channel_sum_2 / n - (acc.channel_sum_1 / n) ** 2).float()
        var_safe = var.clamp_min(0.0)
        std = torch.sqrt(var_safe)
        cmin = acc.channel_min
        cmax = acc.channel_max
        H = cmin.numel()
        for ch in range(H):
            rows.append(
                {
                    "layer": acc.layer_idx,
                    "module": acc.module_name,
                    "channel": ch,
                    "min": float(cmin[ch].item()),
                    "max": float(cmax[ch].item()),
                    "mean": float(mean[ch].item()),
                    "std": float(std[ch].item()),
                }
            )
    return rows


def make_group_sample_rows(accumulators, max_per_module=5000):
    """그룹별 (sign_balance, pos_max, neg_max, max_abs, base) reservoir sample CSV row.
    is_outlier (base 기준, QR 적용 그룹) / is_dangerous (outlier AND high sign_balance) 컬럼 포함.
    layer × module × sample_idx 단위. scatter/heatmap 후처리 + threshold 재구성에 사용."""
    rows = []
    for acc in accumulators:
        sb = acc._cat_samples(acc.group_sign_balance_samples)
        pm = acc._cat_samples(acc.group_pos_max_samples)
        nm = acc._cat_samples(acc.group_neg_max_samples)
        ma = acc._cat_samples(acc.group_max_abs_samples)
        bs = acc._cat_samples(acc.group_base_samples)
        n = sb.numel()
        if n == 0:
            continue
        n = min(n, pm.numel(), nm.numel(), ma.numel(), bs.numel())
        if n > max_per_module:
            idx = torch.randperm(n)[:max_per_module]
            sb, pm, nm, ma, bs = sb[idx], pm[idx], nm[idx], ma[idx], bs[idx]
        out_thr = float(acc.args.selective_base_threshold)
        sb_thr = float(acc.dangerous_sb_threshold)
        for i in range(sb.numel()):
            base_i = float(bs[i].item())
            max_abs_i = float(ma[i].item())
            sb_i = float(sb[i].item())
            is_outlier = base_i >= out_thr           # qr_core selective와 정합 (base 기준)
            is_raw_outlier = max_abs_i >= out_thr    # 참고
            is_dangerous = bool(is_outlier and (sb_i >= sb_thr))
            rows.append(
                {
                    "layer": acc.layer_idx,
                    "module": acc.module_name,
                    "sample_idx": i,
                    "pos_max": float(pm[i].item()),
                    "neg_max": float(nm[i].item()),
                    "max_abs": max_abs_i,
                    "base": base_i,
                    "sign_balance": sb_i,
                    "is_outlier": int(bool(is_outlier)),
                    "is_raw_outlier": int(bool(is_raw_outlier)),
                    "is_dangerous": int(bool(is_dangerous)),
                }
            )
    return rows


def make_q1_hist_rows(accumulators):
    rows = []
    for acc in accumulators:
        samples = acc._cat_samples(acc.q1_count_samples)
        if samples.numel() == 0:
            continue
        counts = torch.bincount(samples.round().long(), minlength=acc.args.base_group_size + 1)
        for q1_count, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            rows.append(
                {
                    "layer": acc.layer_idx,
                    "module": acc.module_name,
                    "q1_count": q1_count,
                    "count": int(count),
                }
            )
    return rows


def make_layer_rows(module_rows):
    grouped = defaultdict(list)
    for row in module_rows:
        grouped[row["layer"]].append(row)

    layer_rows = []
    for layer, rows in sorted(grouped.items()):
        n_values = sum(int(r["n_values"]) for r in rows)
        n_groups = sum(int(r["n_groups"]) for r in rows)
        layer_rows.append(
            {
                "layer": layer,
                "n_modules": len(rows),
                "n_values": n_values,
                "n_groups": n_groups,
                "fallback_ratio": safe_div(
                    sum(float(r["fallback_ratio"]) * int(r["n_groups"]) for r in rows),
                    n_groups,
                ),
                "q1_ratio": safe_div(
                    sum(float(r["q1_ratio"]) * int(r["n_values"]) for r in rows),
                    n_values,
                ),
                "avg_q1_count": safe_div(
                    sum(float(r["avg_q1_count"]) * int(r["n_groups"]) for r in rows),
                    n_groups,
                ),
                "mse_int4": safe_div(
                    sum(float(r["mse_int4"]) * int(r["n_values"]) for r in rows),
                    n_values,
                ),
                "mse_qr": safe_div(
                    sum(float(r["mse_qr"]) * int(r["n_values"]) for r in rows),
                    n_values,
                ),
            }
        )
    return layer_rows


def write_plots(output_dir, accumulators):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warning] matplotlib import failed. Skip plots. ({exc})")
        return

    plot_dir = Path(output_dir) / "plots"
    ensure_dir(plot_dir)

    x_signed = concat_plot_samples(accumulators, "x_signed_samples")
    r_signed = concat_plot_samples(accumulators, "r_signed_samples")

    # Full-range signed histogram (outlier tail까지 그대로): paper figure 용
    if x_signed.numel() > 0:
        x_max_abs = float(x_signed.float().abs().max().item())
        # bin width 결정: --signed_bin_width와 동일한 폭으로
        bin_width = getattr(accumulators[0].args, "signed_bin_width", 2.0) or 2.0
        if bin_width <= 0:
            bin_width = 2.0
        n_bins_full = max(20, int(math.ceil((2.0 * x_max_abs) / bin_width)))
        x_range = (-x_max_abs, x_max_abs)
        x_mean_full = float(x_signed.float().mean().item())

        plt.figure(figsize=(10, 5))
        plt.hist(x_signed.numpy(), bins=n_bins_full, range=x_range, color="steelblue", alpha=0.85)
        plt.axvline(0, color="black", linestyle=":", alpha=0.5)
        plt.axvline(x_mean_full, color="red", linestyle="--", linewidth=1.0,
                    label=f"mean={x_mean_full:.4g}")
        plt.yscale("log")
        plt.xlabel(f"activation value (signed, bin_width={bin_width})")
        plt.ylabel("count (log scale)")
        plt.title(f"Activation Value Distribution (full range, |x|_max={x_max_abs:.3g})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "x_signed_full_range.png", dpi=200)
        plt.close()

    if x_signed.numel() > 0 and r_signed.numel() > 0:
        # QR 적용 전 activation x와 QR 적용 후 residual r의 값 분포를 같은 x축에서 비교한다.
        # 극단 outlier 하나가 histogram 전체를 눌러버리지 않도록 preview plot만 p0.1~p99.9
        # 범위로 그린다. mean 점선은 clipping 전 sampled 값 기준이다.
        combined = torch.cat([x_signed.float(), r_signed.float()])
        lo = float(torch.quantile(combined, 0.001).item())
        hi = float(torch.quantile(combined, 0.999).item())
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            lo = float(combined.min().item())
            hi = float(combined.max().item())
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6

        x_mean = float(x_signed.float().mean().item())
        r_mean = float(r_signed.float().mean().item())
        bins = accumulators[0].args.hist_bins if accumulators else 100

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].hist(x_signed.numpy(), bins=bins, range=(lo, hi), color="tab:blue", alpha=0.78)
        axes[0].axvline(x_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={x_mean:.4g}")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("count")
        axes[0].set_title("Before QR: activation x")
        axes[0].legend()

        axes[1].hist(r_signed.numpy(), bins=bins, range=(lo, hi), color="tab:orange", alpha=0.78)
        axes[1].axvline(r_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean={r_mean:.4g}")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("value")
        axes[1].set_ylabel("count")
        axes[1].set_title("After QR: residual r")
        axes[1].legend()

        fig.suptitle("Value Distribution Before/After QR (sampled, xlim=p0.1~p99.9)")
        fig.tight_layout()
        fig.savefig(plot_dir / "value_range_x_vs_r.png", dpi=200)
        plt.close(fig)

    x_samples = concat_plot_samples(accumulators, "x_abs_samples")
    r_samples = concat_plot_samples(accumulators, "r_abs_samples")
    if x_samples.numel() > 0 and r_samples.numel() > 0:
        max_val = float(torch.quantile(x_samples, 0.999).item())
        max_val = max(max_val, float(torch.quantile(r_samples, 0.999).item()), 1e-6)
        plt.figure(figsize=(8, 5))
        plt.hist(x_samples.clamp_max(max_val).numpy(), bins=100, alpha=0.55, label="abs(x)")
        plt.hist(r_samples.clamp_max(max_val).numpy(), bins=100, alpha=0.55, label="abs(r)")
        plt.yscale("log")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "x_vs_r_hist.png", dpi=200)
        plt.close()

        # Full-range magnitude plot. Outlier visibility가 목적이므로 p99.9 clipping을
        # 하지 않고, log-spaced x bins + log-count y축으로 |x| tail과 |r| tail을 비교한다.
        x_pos = x_samples.float()
        r_pos = r_samples.float()
        x_pos = x_pos[torch.isfinite(x_pos) & (x_pos > 0)]
        r_pos = r_pos[torch.isfinite(r_pos) & (r_pos > 0)]
        if x_pos.numel() > 0 and r_pos.numel() > 0:
            pos_all = torch.cat([x_pos, r_pos])
            xmin = float(torch.quantile(pos_all, 0.0001).item())
            xmax = float(pos_all.max().item())
            xmin = max(xmin, 1e-8)
            if math.isfinite(xmin) and math.isfinite(xmax) and xmin < xmax:
                bins_abs = torch.logspace(
                    math.log10(xmin),
                    math.log10(xmax),
                    steps=(accumulators[0].args.hist_bins if accumulators else 100) + 1,
                )
                x_mean_abs = float(x_pos.mean().item())
                r_mean_abs = float(r_pos.mean().item())

                plt.figure(figsize=(8, 5))
                plt.hist(x_pos.numpy(), bins=bins_abs.numpy(), alpha=0.55, label="abs(x)")
                plt.hist(r_pos.numpy(), bins=bins_abs.numpy(), alpha=0.55, label="abs(r)")
                plt.axvline(x_mean_abs, color="tab:blue", linestyle="--", linewidth=1.2,
                            label=f"mean abs(x)={x_mean_abs:.4g}")
                plt.axvline(r_mean_abs, color="tab:orange", linestyle="--", linewidth=1.2,
                            label=f"mean abs(r)={r_mean_abs:.4g}")
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("absolute value")
                plt.ylabel("count")
                plt.title("Absolute Value Distribution Before/After QR (full sampled range)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_dir / "abs_value_fullrange_x_vs_r.png", dpi=200)
                plt.close()

        # Tail visibility plot: P(|value| > threshold). Histogram보다 count가 작은
        # outlier tail을 보기 쉽고, QR 후 residual tail이 얼마나 줄었는지 직접 보인다.
        def ccdf_points(samples, max_points=5000):
            vals = samples.float()
            vals = vals[torch.isfinite(vals)]
            vals = vals[vals > 0]
            if vals.numel() == 0:
                return torch.empty(0), torch.empty(0)
            vals = torch.sort(vals).values
            n = vals.numel()
            keep = torch.linspace(0, n - 1, steps=min(max_points, n)).round().long()
            xs = vals[keep]
            ys = (n - keep.float()) / n
            return xs.cpu(), ys.cpu()

        x_ccdf_x, x_ccdf_y = ccdf_points(x_samples)
        r_ccdf_x, r_ccdf_y = ccdf_points(r_samples)
        if x_ccdf_x.numel() > 0 and r_ccdf_x.numel() > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(x_ccdf_x.numpy(), x_ccdf_y.numpy(), label="abs(x)")
            plt.plot(r_ccdf_x.numpy(), r_ccdf_y.numpy(), label="abs(r)")
            plt.yscale("log")
            plt.xlabel("absolute value threshold")
            plt.ylabel("P(|value| > threshold)")
            plt.title("Tail CCDF Before/After QR")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "tail_ccdf_x_vs_r.png", dpi=200)
            plt.close()

    base_counts = defaultdict(int)
    q1_counts = concat_plot_samples(accumulators, "q1_count_samples")
    labels = []
    q1_ratios = []
    for acc in accumulators:
        for base, count in acc.base_counts.items():
            base_counts[base] += count
        labels.append(f"L{acc.layer_idx}.{acc.module_name}")
        q1_ratios.append(acc.module_row({})["q1_ratio"])

    if base_counts:
        bases = sorted(base_counts)
        counts = [base_counts[b] for b in bases]
        plt.figure(figsize=(8, 5))
        plt.bar([str(b) for b in bases], counts)
        plt.xlabel("base")
        plt.ylabel("groups")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_dir / "base_hist.png", dpi=200)
        plt.close()

    if q1_counts.numel() > 0:
        counts = torch.bincount(q1_counts.round().long())
        plt.figure(figsize=(8, 5))
        plt.bar(list(range(len(counts))), counts.tolist())
        plt.xlabel("Q active count per group")
        plt.ylabel("groups")
        plt.tight_layout()
        plt.savefig(plot_dir / "q1_count_hist.png", dpi=200)
        plt.close()

    if labels:
        plt.figure(figsize=(max(10, len(labels) * 0.16), 5))
        plt.bar(range(len(labels)), q1_ratios)
        plt.xlabel("layer.module")
        plt.ylabel("effective Q active ratio")
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.tight_layout()
        plt.savefig(plot_dir / "module_q1_ratio.png", dpi=200)
        plt.close()

    # =====================================================================
    # 채널별 signed min/max line plot + 그룹별 sign_balance histogram
    # module × 대표 layer (0, mid, last)별로 1장씩.
    # paper figure로 "channel별 양/음 분포 + 그룹 내 부호 쏠림" 시각화.
    # =====================================================================
    if not accumulators:
        return
    by_module = defaultdict(list)
    for acc in accumulators:
        by_module[acc.module_name].append(acc)

    # 대표 layer 선택: 사용 가능한 layer 중 (min, mid, max) 3개
    for module_name, accs in by_module.items():
        accs_sorted = sorted(accs, key=lambda a: a.layer_idx)
        if not accs_sorted:
            continue
        n = len(accs_sorted)
        rep_indices = sorted(set([0, n // 2, n - 1]))
        rep_accs = [accs_sorted[i] for i in rep_indices]

        # 1) channel signed min/max line plot
        for acc in rep_accs:
            if acc.channel_min is None or acc.channel_n <= 0:
                continue
            H = acc.channel_min.numel()
            xs = list(range(H))
            cmin = acc.channel_min.numpy()
            cmax = acc.channel_max.numpy()
            plt.figure(figsize=(10, 4))
            plt.fill_between(xs, cmin, cmax, color="steelblue", alpha=0.35,
                             label="value range [min, max]")
            plt.plot(xs, cmax, color="tab:blue", linewidth=0.5, label="channel max")
            plt.plot(xs, cmin, color="tab:red", linewidth=0.5, label="channel min")
            plt.axhline(0, color="black", linestyle=":", alpha=0.6)
            plt.xlabel("channel index")
            plt.ylabel("activation value (signed)")
            plt.title(f"Per-channel signed range — {module_name} (layer {acc.layer_idx})")
            plt.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            fname = f"signed_channel_minmax_{module_name.replace('.','_')}_layer{acc.layer_idx}.png"
            plt.savefig(plot_dir / fname, dpi=200)
            plt.close()

        # 2) (pos_max, neg_max) 2D hexbin scatter — "한 토큰 × 한 그룹" 단위.
        #    대각선 위/위쪽: 양쪽 outlier 공존 (dangerous), 한 축 dominant: 한 부호 쏠림.
        for acc in rep_accs:
            pm = acc._cat_samples(acc.group_pos_max_samples)
            nm = acc._cat_samples(acc.group_neg_max_samples)
            if pm.numel() == 0 or nm.numel() == 0:
                continue
            n = min(pm.numel(), nm.numel())
            pm, nm = pm[:n], nm[:n]
            row = acc.module_row({})
            outlier_ratio = row.get("outlier_group_ratio", float("nan"))
            dangerous_ratio = row.get("dangerous_group_ratio", float("nan"))
            thr = float(acc.args.selective_base_threshold)

            plt.figure(figsize=(6, 6))
            # hexbin: 분포 밀도 색상으로
            hb = plt.hexbin(pm.numpy(), nm.numpy(), gridsize=60, mincnt=1,
                            cmap="viridis", bins="log")
            plt.colorbar(hb, label="log10(group count)")
            ax_max = float(max(pm.max().item(), nm.max().item())) * 1.05
            ax_max = max(ax_max, thr * 1.5)
            # 대각선 reference
            plt.plot([0, ax_max], [0, ax_max], color="white", linestyle="--", alpha=0.6,
                     label="pos_max = neg_max")
            # raw max_abs = threshold line (max_abs 축 그대로 비교용)
            plt.axvline(thr, color="red", linestyle=":", alpha=0.7,
                        label=f"raw max_abs = {thr}")
            plt.axhline(thr, color="red", linestyle=":", alpha=0.7)
            # base = threshold 등가 boundary line (실제 outlier_group_ratio 정의와 정합).
            # title의 outlier/dangerous % 도 이 base boundary 기준.
            base_eq_max_abs = base_threshold_to_max_abs(thr, acc.args.q_bits)
            plt.axvline(base_eq_max_abs, color="orange", linestyle="-", alpha=0.85,
                        label=f"base ≥ {thr} boundary (max_abs ≈ {base_eq_max_abs:.2f})")
            plt.axhline(base_eq_max_abs, color="orange", linestyle="-", alpha=0.85)
            plt.xlim(0, ax_max)
            plt.ylim(0, ax_max)
            plt.xlabel("pos_max (per (token, group))")
            plt.ylabel("neg_max (per (token, group))")
            plt.title(f"(pos_max, neg_max) per group — {module_name} L{acc.layer_idx}\n"
                      f"outlier(base≥{thr})={outlier_ratio*100:.2f}%  "
                      f"dangerous={dangerous_ratio*100:.3f}%")
            plt.legend(loc="upper right", fontsize=7)
            plt.tight_layout()
            fname = f"pos_neg_scatter_{module_name.replace('.','_')}_layer{acc.layer_idx}.png"
            plt.savefig(plot_dir / fname, dpi=200)
            plt.close()

        # 3) base별 stratified sign_balance histogram (각 base 색 다르게 overlay)
        base_choices = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        for acc in rep_accs:
            sb = acc._cat_samples(acc.group_sign_balance_samples)
            bs = acc._cat_samples(acc.group_base_samples)
            if sb.numel() == 0 or bs.numel() == 0 or sb.numel() != bs.numel():
                continue
            plt.figure(figsize=(8, 4.5))
            bins_edges = torch.linspace(0.0, 1.0, 21).numpy()
            any_plotted = False
            cmap = plt.get_cmap("plasma")
            for bi, b_val in enumerate(base_choices):
                mask = (bs == b_val)
                if not mask.any():
                    continue
                sb_b = sb[mask].numpy()
                plt.hist(sb_b, bins=bins_edges, histtype="step", linewidth=1.6,
                         color=cmap(bi / max(len(base_choices) - 1, 1)),
                         label=f"base={int(b_val)} ({int(mask.sum().item())} groups)")
                any_plotted = True
            if any_plotted:
                plt.xlabel("group sign-balance ratio")
                plt.ylabel("groups (log)")
                plt.yscale("log")
                plt.title(f"sign_balance by base — {module_name} L{acc.layer_idx}")
                plt.legend(fontsize=7, loc="upper right")
                plt.tight_layout()
                fname = f"sign_balance_by_base_{module_name.replace('.','_')}_layer{acc.layer_idx}.png"
                plt.savefig(plot_dir / fname, dpi=200)
            plt.close()

        # 4) max_abs vs sign_balance 2D heatmap (log10 max_abs × sign_balance)
        for acc in rep_accs:
            sb = acc._cat_samples(acc.group_sign_balance_samples)
            ma = acc._cat_samples(acc.group_max_abs_samples)
            if sb.numel() == 0 or ma.numel() == 0 or sb.numel() != ma.numel():
                continue
            ma_pos = ma.clamp_min(1e-6)
            log_ma = torch.log10(ma_pos).numpy()
            sb_np = sb.numpy()
            plt.figure(figsize=(7, 5))
            h = plt.hist2d(log_ma, sb_np, bins=[40, 20],
                           range=[[log_ma.min(), log_ma.max()], [0.0, 1.0]],
                           cmap="magma", norm=__import__("matplotlib").colors.LogNorm())
            plt.colorbar(h[3], label="group count (log)")
            thr = float(acc.args.selective_base_threshold)
            if thr > 0:
                # raw max_abs = threshold (직역; 실제 outlier_group_ratio와는 다름)
                plt.axvline(math.log10(thr), color="cyan", linestyle="--", alpha=0.8,
                            label=f"raw max_abs = {thr}")
                # base = threshold 등가 boundary (outlier_group_ratio 정의와 정합)
                base_eq = base_threshold_to_max_abs(thr, acc.args.q_bits)
                if base_eq > 0:
                    plt.axvline(math.log10(base_eq), color="orange", linestyle="-",
                                alpha=0.9,
                                label=f"base ≥ {thr} boundary (max_abs ≈ {base_eq:.2f})")
            plt.axhline(acc.dangerous_sb_threshold, color="white", linestyle=":", alpha=0.8,
                        label=f"dangerous sb = {acc.dangerous_sb_threshold}")
            plt.xlabel("log10(group max_abs)")
            plt.ylabel("sign_balance ratio")
            plt.title(f"max_abs × sign_balance heatmap — {module_name} L{acc.layer_idx}\n"
                      f"(outlier_group_ratio uses base≥{thr} criterion)")
            plt.legend(fontsize=7, loc="upper left")
            plt.tight_layout()
            fname = f"max_abs_sb_heatmap_{module_name.replace('.','_')}_layer{acc.layer_idx}.png"
            plt.savefig(plot_dir / fname, dpi=200)
            plt.close()

        # 5) per-token cross-channel kurtosis histogram
        # 한 token 안의 H개 channel 분포의 kurtosis. CIM/QR 양자화 단위와 정합.
        # x축 = kurt 값, y축 = 그 kurt 값을 가진 token 수.
        # mean이 3(normal)보다 훨씬 크면 cross-channel outlier 존재 — outlier-aware 양자화 정당화.
        for acc in rep_accs:
            samples = acc._cat_samples(acc.per_token_kurt_samples)
            if samples.numel() == 0:
                continue
            samples = samples[torch.isfinite(samples)]
            if samples.numel() == 0:
                continue
            kurt_np = samples.numpy()
            # 매우 긴 우측 꼬리 (kurt > 수천)는 p99로 clamp해서 시각화
            upper = float(torch.quantile(samples, 0.99).item())
            upper = max(upper, 10.0)
            kurt_clamped = kurt_np.clip(max=upper)
            mean_k = float(samples.mean().item())
            p50_k = float(torch.quantile(samples, 0.5).item())
            p90_k = float(torch.quantile(samples, 0.9).item())
            plt.figure(figsize=(8, 4))
            plt.hist(kurt_clamped, bins=60, color="seagreen", alpha=0.85)
            plt.axvline(3.0, color="black", linestyle=":", alpha=0.7,
                        label="normal=3")
            plt.axvline(mean_k, color="red", linestyle="--", linewidth=1.2,
                        label=f"mean={mean_k:.2f}")
            plt.axvline(p50_k, color="purple", linestyle="-.", linewidth=1.0,
                        label=f"median={p50_k:.2f}")
            plt.axvline(p90_k, color="orange", linestyle="--", linewidth=1.2,
                        label=f"p90={p90_k:.2f}")
            plt.xlabel(f"per-token kurtosis across {acc.channel_min.numel() if acc.channel_min is not None else 'H'} channels "
                       f"(clipped at p99={upper:.1f})")
            plt.ylabel("tokens")
            plt.title(f"per-token cross-channel kurtosis — {module_name} L{acc.layer_idx}\n"
                      f"(QR/CIM quantization unit; mean ≫ 3 ⇒ outlier channels)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            fname = f"per_token_kurt_{module_name.replace('.','_')}_layer{acc.layer_idx}.png"
            plt.savefig(plot_dir / fname, dpi=200)
            plt.close()


def save_summary(output_dir, args, load_path, arch, module_rows, layer_rows):
    total_values = sum(int(r["n_values"]) for r in module_rows)
    total_groups = sum(int(r["n_groups"]) for r in module_rows)
    avg_q1_ratio = safe_div(sum(float(r["q1_ratio"]) * int(r["n_values"]) for r in module_rows), total_values)
    avg_fallback = safe_div(sum(float(r["fallback_ratio"]) * int(r["n_groups"]) for r in module_rows), total_groups)
    mse_int4 = safe_div(sum(float(r["mse_int4"]) * int(r["n_values"]) for r in module_rows), total_values)
    mse_qr = safe_div(sum(float(r["mse_qr"]) * int(r["n_values"]) for r in module_rows), total_values)

    summary = {
        "config": vars(args),
        "load_path": str(load_path),
        "arch": arch,
        "n_modules": len(module_rows),
        "n_layers": len(layer_rows),
        "n_values": total_values,
        "n_groups": total_groups,
        "q1_ratio": avg_q1_ratio,
        "fallback_ratio": avg_fallback,
        "mse_int4": mse_int4,
        "mse_qr": mse_qr,
        "mse_qr_over_int4": safe_div(mse_qr, mse_int4),
    }

    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)

    lines = [
        "[QR Stats Summary]",
        f"model_name          : {args.model_name}",
        f"model_source        : {args.model_source}",
        f"load_path           : {load_path}",
        f"arch                : {arch}",
        f"eval_dataset        : {args.eval_dataset}",
        f"stats_nsamples      : {args.stats_nsamples}",
        f"n_modules           : {len(module_rows)}",
        f"n_values            : {total_values}",
        f"n_groups            : {total_groups}",
        f"q1_ratio            : {avg_q1_ratio:.8f}",
        f"fallback_ratio      : {avg_fallback:.8f}",
        f"mse_int4            : {mse_int4:.12e}",
        f"mse_qr              : {mse_qr:.12e}",
        f"mse_qr_over_int4    : {safe_div(mse_qr, mse_int4):.8f}",
        "",
        "[Outputs]",
        "module_stats.csv                        (+outlier_group_ratio, dangerous_group_ratio, outlier_sign_balance_mean/p90)",
        "layer_stats.csv",
        "base_q_stats.csv",
        "hist_x_abs.csv",
        "hist_r_abs.csv",
        "hist_q1_count.csv",
        "hist_x_signed.csv",
        "hist_r_signed.csv",
        "channel_signed_minmax.csv               (channel별 signed min/max/mean/std/kurt)",
        "group_pos_neg_samples.csv               (그룹별 pos_max/neg_max/base/sign_balance sample, scatter 재현용)",
        "summary.json",
        "plots/signed_channel_minmax_<module>_layer<i>.png",
        "plots/pos_neg_scatter_<module>_layer<i>.png         (한 토큰×그룹 단위 outlier 분포)",
        "plots/sign_balance_by_base_<module>_layer<i>.png    (base별 sign_balance overlay)",
        "plots/max_abs_sb_heatmap_<module>_layer<i>.png      (max_abs × sign_balance 2D 분포)",
        "plots/per_token_kurt_<module>_layer<i>.png          (token별 cross-channel kurtosis 분포, CIM/QR 단위 정합)",
        "plots/*.png",
    ]
    with open(Path(output_dir) / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="opt-6.7b")
    parser.add_argument("--model_source", type=str, default="omni", choices=["auto", "fp16", "omni"])
    parser.add_argument("--eval_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "c4_new", "c4_omni"])
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--stats_nsamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c4_cache_dir", type=str, default=str(Path(__file__).resolve().parent / "cache"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "results_qr_stats"))

    parser.add_argument("--replace_scope", type=str, default="all", choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=0)
    parser.add_argument("--custom_layer_indices", type=str, default="")
    parser.add_argument("--target_modules", type=str, default="auto")

    parser.add_argument("--q_bits", type=int, default=1)
    parser.add_argument("--r_bits", type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--r_group_size", type=int, default=128)
    parser.add_argument("--selective_base_threshold", type=float, default=8.0)
    parser.add_argument("--selective_int_bits", type=int, default=4)
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0)

    parser.add_argument("--int_bits", type=int, default=4)
    parser.add_argument("--int_group_size", type=int, default=128)
    parser.add_argument("--max_samples_per_module", type=int, default=200000)
    parser.add_argument("--max_q1_samples_per_base", type=int, default=50000)
    parser.add_argument("--sample_chunk", type=int, default=20000)
    parser.add_argument("--hist_bins", type=int, default=100)
    parser.add_argument("--signed_bin_width", type=float, default=2.0,
                        help="signed histogram의 bin 폭 (예: 2.0 → 0~2, 2~4, ..., -2~0, -4~-2 ...). "
                             "None/0 이면 max_abs 기준 자동 분할.")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir) / args.model_name / f"{args.eval_dataset}_{args.model_source}"
    ensure_dir(output_dir)

    print("[1] Loading model ...")
    model, tokenizer, seqlen, arch, load_path = load_model_and_tokenizer(
        args.model_name,
        model_source=args.model_source,
    )
    dev = get_model_device(model, arch)

    print("[2] Loading eval tokens ...")
    _, testenc = load_eval_tokens(
        load_path,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        seqlen=seqlen,
        nsamples=args.stats_nsamples,
        seed=args.seed,
        c4_cache_dir=args.c4_cache_dir,
    )
    n_available = testenc.numel() // seqlen
    n_run = min(args.stats_nsamples, n_available)
    if n_run <= 0:
        raise ValueError(f"No full {seqlen}-token window available.")

    if args.target_modules == "auto":
        module_names = get_module_names(arch)
    else:
        module_names = parse_module_names(args.target_modules, arch)

    layer_indices = resolve_target_layers(
        model,
        arch,
        args.replace_scope,
        args.one_layer_idx,
        args.custom_layer_indices,
    )

    print("[3] Registering hooks ...")
    layers = get_decoder_layers(model, arch)
    accumulators = {}
    hooks = []
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for module_name in module_names:
            module = get_named_linear(layer, module_name)
            key = (layer_idx, module_name)
            accumulators[key] = ModuleAccumulator(layer_idx, module_name, args)

            def pre_hook(mod, inputs, hook_key=key):
                x = inputs[0]
                accumulators[hook_key].update(x)

            hooks.append(module.register_forward_pre_hook(pre_hook))

    print(f"[4] Collecting stats on {n_run} windows ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            for i in tqdm(range(n_run)):
                batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
                out = model(batch, use_cache=False)
                del out
    finally:
        model.config.use_cache = use_cache
        for h in hooks:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[5] Writing CSV/JSON outputs ...")
    config_row = {
        "model_name": args.model_name,
        "model_source": args.model_source,
        "dataset": args.eval_dataset,
        "split": args.eval_split,
        "q_bits": args.q_bits,
        "r_bits": args.r_bits,
        "base_group_size": args.base_group_size,
        "r_group_size": args.r_group_size,
        "selective_base_threshold": args.selective_base_threshold,
    }

    acc_list = [accumulators[k] for k in sorted(accumulators)]
    module_rows = [acc.module_row(config_row) for acc in acc_list]
    layer_rows = make_layer_rows(module_rows)
    base_q_rows = []
    for acc in acc_list:
        base_q_rows.extend(acc.base_q_rows(config_row))

    module_fields = list(module_rows[0].keys()) if module_rows else []
    layer_fields = list(layer_rows[0].keys()) if layer_rows else []
    base_q_fields = list(base_q_rows[0].keys()) if base_q_rows else []

    write_csv(output_dir / "module_stats.csv", module_fields, module_rows)
    write_csv(output_dir / "layer_stats.csv", layer_fields, layer_rows)
    write_csv(output_dir / "base_q_stats.csv", base_q_fields, base_q_rows)

    hist_x_rows = make_hist_rows(acc_list, "x_abs_samples", "x_abs", args.hist_bins)
    hist_r_rows = make_hist_rows(acc_list, "r_abs_samples", "r_abs", args.hist_bins)
    hist_q_rows = make_q1_hist_rows(acc_list)
    # Signed (음수/양수 모두 표시) histogram: bin 폭을 args.signed_bin_width로 고정해서
    # "0~2, 2~4, ..., 50~60" 식의 일정 단위로 count를 뽑는다 (음수 쪽도 대칭).
    signed_width = args.signed_bin_width if args.signed_bin_width and args.signed_bin_width > 0 else None
    hist_x_signed_rows = make_signed_hist_rows(
        acc_list, "x_signed_samples", "x_signed",
        bins=args.hist_bins,
        fixed_width=signed_width,
    )
    hist_r_signed_rows = make_signed_hist_rows(
        acc_list, "r_signed_samples", "r_signed",
        bins=args.hist_bins,
        fixed_width=signed_width,
    )
    write_csv(output_dir / "hist_x_abs.csv", list(hist_x_rows[0].keys()) if hist_x_rows else [], hist_x_rows)
    write_csv(output_dir / "hist_r_abs.csv", list(hist_r_rows[0].keys()) if hist_r_rows else [], hist_r_rows)
    write_csv(output_dir / "hist_q1_count.csv", list(hist_q_rows[0].keys()) if hist_q_rows else [], hist_q_rows)
    write_csv(output_dir / "hist_x_signed.csv", list(hist_x_signed_rows[0].keys()) if hist_x_signed_rows else [], hist_x_signed_rows)
    write_csv(output_dir / "hist_r_signed.csv", list(hist_r_signed_rows[0].keys()) if hist_r_signed_rows else [], hist_r_signed_rows)

    # === 신규: 채널별 signed 통계 + 그룹별 sample (pos_max, neg_max, base, sign_balance) ===
    # 기존 단순 sign_balance histogram CSV는 제거됨. 정보는 group_pos_neg_samples.csv
    # (raw scatter sample) + module_stats의 outlier_group_ratio / dangerous_group_ratio
    # / sign_balance_mean / sign_balance_p50/p90/p99 / outlier_sign_balance_mean / p90 컬럼으로
    # 모두 대체된다 (재현 가능 + 그래프 자유 생성).
    channel_signed_rows = make_channel_signed_rows(acc_list)
    group_sample_rows = make_group_sample_rows(acc_list, max_per_module=5000)
    write_csv(
        output_dir / "channel_signed_minmax.csv",
        list(channel_signed_rows[0].keys()) if channel_signed_rows else [],
        channel_signed_rows,
    )
    write_csv(
        output_dir / "group_pos_neg_samples.csv",
        list(group_sample_rows[0].keys()) if group_sample_rows else [],
        group_sample_rows,
    )

    if not args.no_plots:
        print("[6] Writing preview PNG plots ...")
        write_plots(output_dir, acc_list)

    save_summary(output_dir, args, load_path, arch, module_rows, layer_rows)
    print(f"[Done] {output_dir}")


if __name__ == "__main__":
    main()
