import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval_utils import load_eval_tokens
from model_adapter import (
    get_decoder_layers,
    get_model_device,
    get_named_linear,
    load_model_and_tokenizer,
)


def nearest_pow2_base(max_abs):
    # qr_core_ver2.py:117-127 의 q_bits=1 base 선택 규칙과 동일.
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


def qr_split(x, base_group_size):
    # qr_core_ver2.py:117-137 의 q_bits=1 경로와 동일하게 base/sign_flag/q 결정 후
    # r은 quantize 전 raw 값을 그대로 반환 (분포 시각화 용도).
    H = x.shape[-1]
    base_gs = H if base_group_size == -1 else base_group_size
    assert H % base_gs == 0
    G = H // base_gs

    xg = x.float().reshape(x.shape[:-1] + (G, base_gs))
    max_abs   = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    base      = nearest_pow2_base(max_abs)
    pos_max   = xg.amax(dim=-1, keepdim=True)
    neg_max   = xg.amin(dim=-1, keepdim=True)
    sign_flag = torch.where(
        pos_max.abs() >= neg_max.abs(),
        torch.ones_like(pos_max),
        -torch.ones_like(pos_max),
    )
    q     = (xg * sign_flag >= base / 2.0).float()
    r_raw = xg - sign_flag * base * q

    return xg, r_raw, base.squeeze(-1), sign_flag.squeeze(-1), pos_max.squeeze(-1), neg_max.squeeze(-1)


def auto_pick_target(stats_csv):
    best = (-1.0, None, None)
    with open(stats_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ratio = float(row.get("outlier_group_ratio", 0.0))
            except (TypeError, ValueError):
                continue
            if ratio > best[0]:
                best = (ratio, int(row["layer"]), row["module"])
    return best


def capture_groups(model, testenc, seqlen, n_run, dev, arch, layer_idx, module_name, base_group_size):
    layers = get_decoder_layers(model, arch)
    module = get_named_linear(layers[layer_idx], module_name)

    xg_list, r_list, base_list, sign_list, pos_list, neg_list = [], [], [], [], [], []

    def pre_hook(mod, inputs):
        x = inputs[0].detach()
        xg, r_raw, base, sign_flag, pos_max, neg_max = qr_split(x, base_group_size)
        xg_list.append(xg.reshape(-1, xg.shape[-1]).cpu())
        r_list.append(r_raw.reshape(-1, r_raw.shape[-1]).cpu())
        base_list.append(base.reshape(-1).cpu())
        sign_list.append(sign_flag.reshape(-1).cpu())
        pos_list.append(pos_max.reshape(-1).cpu())
        neg_list.append(neg_max.reshape(-1).cpu())

    handle = module.register_forward_pre_hook(pre_hook)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            for i in tqdm(range(n_run), desc=f"capture L{layer_idx}.{module_name}"):
                batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
                model(batch, use_cache=False)
    finally:
        model.config.use_cache = use_cache
        handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        torch.cat(xg_list, dim=0),
        torch.cat(r_list, dim=0),
        torch.cat(base_list, dim=0),
        torch.cat(sign_list, dim=0),
        torch.cat(pos_list, dim=0),
        torch.cat(neg_list, dim=0),
    )


def compute_masks(x_groups, base, pos_max, neg_max, threshold, sb_max, dom_ratio_min):
    max_abs = torch.maximum(pos_max.abs(), neg_max.abs())
    pos_part = pos_max.clamp(min=0.0)
    neg_part = (-neg_max).clamp(min=0.0)
    sb_num = torch.minimum(pos_part, neg_part)
    sb_den = torch.maximum(pos_part, neg_part).clamp_min(1e-12)
    sign_balance = sb_num / sb_den

    # 그룹별 2nd-largest |x| (절대값 기준 두번째 큰 값)
    top2, _ = x_groups.abs().topk(2, dim=-1)
    second_largest = top2[:, 1]
    dom_ratio = max_abs / second_largest.clamp_min(1e-12)

    outlier_mask = base >= threshold
    favorable_mask = outlier_mask & (sign_balance < sb_max) & (dom_ratio >= dom_ratio_min)
    return outlier_mask, favorable_mask, max_abs, sign_balance, dom_ratio


def plot_figure1_strip(x_g, r_g, base_val, sign_val, max_abs_val,
                       int_bits, r_bits, output_path, title):
    int_qmax = (2 ** (int_bits - 1)) - 1
    r_qmax   = (2 ** (r_bits   - 1)) - 1

    x_vals = x_g.numpy()
    r_vals = r_g.numpy()
    x_max  = float(np.abs(x_vals).max())
    r_max  = float(np.abs(r_vals).max())
    step_x = x_max / int_qmax
    step_r = r_max / r_qmax
    ratio  = step_x / step_r if step_r > 0 else float("inf")

    # 수정 이유 (사용자 피드백 4차 반영):
    #   - band / 큰 박스 / inset / shared xlim 모두 제거.
    #   - 각 패널을 자기 범위(±x_max, ±r_max)로 auto-scale → 데이터가 패널을 채워서
    #     클러스터가 자연스럽게 보이고 INT 격자 간격도 또렷이 보임.
    #   - 두 패널은 다른 스케일이지만 xlabel 의 |x|_max / |r|_max / step 숫자로 비교 가능.
    # 수정 이유 (5차 → 6차 revert): cluster bracket 표현 제거하고 깔끔한 4차 버전 복원.

    x_abs = np.abs(x_vals)
    outlier_idx = int(np.argmax(x_abs))
    x_outlier = float(x_vals[outlier_idx])
    r_outlier = float(r_vals[outlier_idx])

    grid_x = np.arange(-int_qmax, int_qmax + 1) * step_x
    grid_r = np.arange(-r_qmax,   r_qmax   + 1) * step_r

    def _draw_panel(ax, vals, outlier_val, grid, step, xlim,
                    color, edge, panel_title, panel_xlabel):
        # 굵은 검은 가로축 = number line
        ax.axhline(0.5, color="black", linewidth=2.0, zorder=2)
        # INT grid: 축 위쪽 세로 tick + 라벨. xlim 내 visible 개수로 label_skip 결정.
        visible_grid = [g for g in grid if xlim[0] <= g <= xlim[1]]
        n_vis = len(visible_grid)
        label_skip = 1 if n_vis <= 9 else (2 if n_vis <= 15 else 3)
        for i, g in enumerate(grid):
            if xlim[0] <= g <= xlim[1]:
                ax.plot([g, g], [0.5, 0.66], color="dimgray", linewidth=1.1, zorder=3)
                # 데이터 절대값 범위에 따라 포맷 자동 (작은 값일 때 소수점 더)
                fmt = "{:.2f}" if abs(g) < 10 else "{:.1f}"
                if (i % label_skip) == 0:
                    ax.text(g, 0.69, fmt.format(g),
                            ha="center", va="bottom", fontsize=9, color="dimgray")
        # 데이터 점 (jitter 없이 축 위에)
        mask = np.ones_like(vals, dtype=bool)
        mask[outlier_idx] = False
        pts = vals[mask]
        ax.scatter(pts, np.full_like(pts, 0.5), s=34, alpha=0.75,
                   color=color, edgecolor=edge, linewidth=0.5, zorder=5)
        # outlier 별표
        ax.scatter([outlier_val], [0.5], s=170, marker="*",
                   color="red", edgecolor="darkred", linewidth=0.8, zorder=6)
        ax.annotate(f"outlier {outlier_val:+.3f}",
                    xy=(outlier_val, 0.5), xytext=(outlier_val, 0.88),
                    ha="center", fontsize=10, color="darkred", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8))
        # step 간격 화살표 (첫 두 양수 grid 사이) — 축 아래쪽에 배치
        pos_grid = [g for g in visible_grid if g > 0]
        if len(pos_grid) >= 2:
            g0, g1 = pos_grid[0], pos_grid[1]
            ax.annotate("", xy=(g1, 0.32), xytext=(g0, 0.32),
                        arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.3))
            ax.text((g0 + g1) / 2, 0.26, f"step = {step:.3f}",
                    ha="center", va="top", fontsize=10, color="dimgray")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel(panel_xlabel, fontsize=10.5)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.8),
                              gridspec_kw={"hspace": 0.85})

    # Top: Before QR — 자기 범위 (±x_max)
    xlim_top = (-x_max * 1.12, x_max * 1.12)
    _draw_panel(
        axes[0], x_vals, x_outlier, grid_x, step_x, xlim_top,
        "steelblue", "navy",
        panel_title="Before QR — 128 raw values on the number line  (★ outlier)",
        panel_xlabel=f"x value     |x|_max = {x_max:.3f},     INT{int_bits} step = {step_x:.3f}",
    )

    # Bottom: After QR — 좁아진 자기 범위 (±r_max)
    xlim_bot = (-r_max * 1.15, r_max * 1.15)
    _draw_panel(
        axes[1], r_vals, r_outlier, grid_r, step_r, xlim_bot,
        "darkorange", "saddlebrown",
        panel_title=(f"After QR — residuals on a {x_max / max(r_max, 1e-8):.1f}× "
                     f"narrower number line  (base={float(base_val):g}, "
                     f"sign={int(sign_val):+d})"),
        panel_xlabel=(f"r value     |r|_max = {r_max:.3f},     "
                      f"r{r_bits} step = {step_r:.3f}     "
                      f"(step shrunk {ratio:.1f}×)"),
    )

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure2_distribution(x_all, r_all, mask, int_bits, r_bits, threshold,
                              output_path, title):
    int_qmax = (2 ** (int_bits - 1)) - 1
    r_qmax   = (2 ** (r_bits   - 1)) - 1

    x_sel = x_all[mask].reshape(-1).numpy()
    r_sel = r_all[mask].reshape(-1).numpy()

    x_max = float(np.abs(x_sel).max())
    r_max = float(np.abs(r_sel).max())
    step_x = x_max / int_qmax
    step_r = r_max / r_qmax
    ratio  = step_x / step_r if step_r > 0 else float("inf")

    n_bins   = 100
    bins_full = np.linspace(-x_max, x_max, n_bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.hist(x_sel, bins=bins_full, color="steelblue", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Before QR (raw x, outlier-favorable groups)  "
                 f"|x|_max={x_max:.3g}, INT{int_bits} step={step_x:.4g}")

    ax = axes[1]
    ax.hist(r_sel, bins=bins_full, color="darkorange", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("value (shared axis for both panels)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"After QR (residual r, same groups)  "
                 f"|r|_max={r_max:.3g}, r{r_bits}-bit step={step_r:.4g}  "
                 f"→ step shrunk {ratio:.1f}×")

    fig.suptitle(title + f"  (threshold={threshold}, n_groups={int(mask.sum())})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    # auto-scale 동반 그림: 각 panel이 자기 범위를 다 쓰는 버전 (Gaussian 모양 강조).
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6))
    axes2[0].hist(x_sel, bins=np.linspace(-x_max, x_max, n_bins + 1),
                  color="steelblue", alpha=0.85)
    axes2[0].axvline(0, color="black", linewidth=0.6, alpha=0.6)
    axes2[0].set_yscale("log")
    axes2[0].set_xlabel(f"x value (auto-scaled)  |x|_max={x_max:.3g}")
    axes2[0].set_ylabel("count (log)")
    axes2[0].set_title("Before QR — raw x distribution (own scale)")

    axes2[1].hist(r_sel, bins=np.linspace(-r_max, r_max, n_bins + 1),
                  color="darkorange", alpha=0.85)
    axes2[1].axvline(0, color="black", linewidth=0.6, alpha=0.6)
    axes2[1].set_yscale("log")
    axes2[1].set_xlabel(f"r value (auto-scaled)  |r|_max={r_max:.3g}")
    axes2[1].set_ylabel("count (log)")
    axes2[1].set_title(f"After QR — residual r distribution (own scale)  "
                       f"INT-step ratio (vs same-bits): "
                       f"{(x_max/int_qmax)/(r_max/int_qmax):.1f}× "
                       f"vs r{r_bits}: {ratio:.1f}×")

    fig2.suptitle(title + " — auto-scaled view")
    fig2.tight_layout()
    out_auto = output_path.with_name(output_path.stem + "_autoscale.png")
    fig2.savefig(out_auto, dpi=200)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   type=str, default="opt-6.7b")
    parser.add_argument("--model_source", type=str, default="omni",
                        choices=["auto", "fp16", "omni"])
    parser.add_argument("--eval_dataset", type=str, default="wikitext2")
    parser.add_argument("--eval_split",   type=str, default="test")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--nsamples",     type=int, default=4)
    parser.add_argument("--c4_cache_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "cache"))

    parser.add_argument("--layer",  type=str, default="auto")
    parser.add_argument("--module", type=str, default="auto")
    parser.add_argument("--base_group_size",          type=int,   default=128)
    parser.add_argument("--selective_base_threshold", type=float, default=8.0)
    parser.add_argument("--int_bits", type=int, default=4,
                        help="baseline INT 비트 (raw x 격자 계산)")
    parser.add_argument("--r_bits",   type=int, default=3,
                        help="실제 r 양자화 비트 (residual 격자 계산)")
    parser.add_argument("--sb_max",       type=float, default=0.2,
                        help="favorable mask: sign_balance 상한")
    parser.add_argument("--dom_ratio_min", type=float, default=4.0,
                        help="favorable mask: max_abs/2nd_largest 하한")

    parser.add_argument("--stats_csv", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "results_qr_stats/opt-6.7b/wikitext2_omni/module_stats.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results_dist_plots"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) 타겟 (layer, module) 결정
    if args.layer == "auto" or args.module == "auto":
        ratio, auto_layer, auto_module = auto_pick_target(args.stats_csv)
        if auto_layer is None:
            raise RuntimeError(f"auto-pick 실패: {args.stats_csv} 에서 outlier_group_ratio 못 읽음")
        layer_idx   = auto_layer  if args.layer  == "auto" else int(args.layer)
        module_name = auto_module if args.module == "auto" else args.module
        print(f"[auto] layer={layer_idx}, module={module_name}  "
              f"(top outlier_group_ratio={ratio:.4f})")
    else:
        layer_idx   = int(args.layer)
        module_name = args.module

    # 2) 모델 로드
    print("[1] loading model ...")
    model, tokenizer, seqlen, arch, load_path = load_model_and_tokenizer(
        args.model_name, model_source=args.model_source,
    )
    dev = get_model_device(model, arch)

    print("[2] loading eval tokens ...")
    _, testenc = load_eval_tokens(
        load_path,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        seqlen=seqlen,
        nsamples=args.nsamples,
        seed=args.seed,
        c4_cache_dir=args.c4_cache_dir,
    )
    n_available = testenc.numel() // seqlen
    n_run = min(args.nsamples, n_available)
    if n_run <= 0:
        raise ValueError("eval 데이터에 full window 없음")

    # 3) 그룹 캡쳐
    print(f"[3] capturing groups at L{layer_idx}.{module_name} ...")
    x_groups, r_groups, base, sign_flag, pos_max, neg_max = capture_groups(
        model, testenc, seqlen, n_run, dev, arch,
        layer_idx, module_name, args.base_group_size,
    )
    print(f"    -> {x_groups.shape[0]} groups × {x_groups.shape[1]} values captured")

    # 4) outlier / favorable mask
    outlier_mask, favorable_mask, max_abs, sign_balance, dom_ratio = compute_masks(
        x_groups, base, pos_max, neg_max,
        threshold=args.selective_base_threshold,
        sb_max=args.sb_max,
        dom_ratio_min=args.dom_ratio_min,
    )
    n_total     = base.numel()
    n_outlier   = int(outlier_mask.sum())
    n_favorable = int(favorable_mask.sum())
    print(f"    outlier groups (base>={args.selective_base_threshold}): {n_outlier}/{n_total}")
    print(f"    favorable (sb<{args.sb_max}, dom_ratio>={args.dom_ratio_min}): {n_favorable}/{n_total}")

    output_dir = Path(args.output_dir) / args.model_name / f"L{layer_idx}_{module_name.replace('.', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5) Figure 1: 대표 favorable group 1개 strip plot
    #
    # 수정 이유 (사용자 피드백): "내 아이디어가 정확히 빛을 발하는 case"를 보여줘야 하는데
    # 단순히 median max_abs 만으로 뽑으면 cluster 가 너무 작아서 INT4 step 이든 r-step 이든
    # 둘 다 cluster 를 0 으로 죽이는 그룹이 뽑힘. 그래서 다음 조건들을 함께 만족하는 그룹을
    # 우선 픽한다:
    #   (a) max_abs / base 가 1.0 에 가까움  → outlier 잔차 ≈ 0, "range 좁아짐"이 극적
    #   (b) 2nd_largest 가 step_r/2 보다 크고 step_x/2 보다 작음
    #        → INT4 baseline 은 2nd_largest 를 0 으로 죽이지만 r-bit 는 살림 (QR 이득 영역)
    # 위 두 조건 만족 그룹이 없으면 favorable, 그것도 없으면 outlier 로 fallback.
    int_qmax = (2 ** (args.int_bits - 1)) - 1
    r_qmax   = (2 ** (args.r_bits   - 1)) - 1
    abs_g = x_groups.abs()
    top2_vals, _ = abs_g.topk(2, dim=-1)
    second_largest_abs = top2_vals[:, 1]
    base_t = base
    max_abs_t = max_abs
    # outlier 잔차 비율: |max_abs - base| / max_abs.  0 에 가까울수록 좋음 (≤ 0.15 권장).
    residual_ratio = (max_abs_t - base_t).abs() / max_abs_t.clamp_min(1e-8)
    # 그룹별 step 예상치 (이 그룹이 단독으로 quantize 됐을 때):
    step_x_per_group = max_abs_t / int_qmax
    # r_max 는 max(residual_outlier, second_largest_abs) 로 근사 (r3-bit 잔차는 이보다 작거나 같음).
    r_max_approx = torch.maximum((max_abs_t - base_t).abs(), second_largest_abs)
    step_r_per_group = r_max_approx / r_qmax
    # QR 이 cluster 를 살리는 영역: step_r/2 < 2nd_largest < step_x/2
    cluster_gain_mask = (second_largest_abs > step_r_per_group / 2) & \
                         (second_largest_abs < step_x_per_group / 2)
    tight_residual_mask = residual_ratio < 0.15
    ideal_mask = favorable_mask & tight_residual_mask & cluster_gain_mask
    if ideal_mask.any():
        pick_mask = ideal_mask
        pick_reason = "ideal (favorable & tight residual & cluster-gain)"
    elif favorable_mask.any():
        pick_mask = favorable_mask
        pick_reason = "favorable (no ideal match)"
    else:
        pick_mask = outlier_mask
        pick_reason = "outlier fallback (no favorable)"
    print(f"    ideal groups (tight residual & cluster gain): {int(ideal_mask.sum())}/{n_total}")
    print(f"    Figure 1 picking from: {pick_reason}")

    if pick_mask.any():
        # 더 극단적인 group 선택 (사용자 요청): pool 안에서 실제 step shrinkage ratio 최대.
        # shrink_ratio_actual = max_abs / r_max_actual 이 클수록 QR 의 이득이 큼.
        r_max_actual = r_groups.abs().amax(dim=-1)  # shape: (N_groups,)
        shrink_score = max_abs / r_max_actual.clamp_min(1e-8)
        shrink_score = shrink_score.masked_fill(~pick_mask, float("-inf"))
        rep_idx = int(shrink_score.argmax().item())

        title1 = (f"L{layer_idx}.{module_name} — group #{rep_idx}  "
                  f"(max|x|={float(max_abs[rep_idx]):.3g}, "
                  f"base={float(base[rep_idx]):g}, "
                  f"sign_balance={float(sign_balance[rep_idx]):.3f}, "
                  f"dom_ratio={float(dom_ratio[rep_idx]):.2f}, "
                  f"2nd_largest={float(second_largest_abs[rep_idx]):.2f}, "
                  f"|r|_max={float(r_max_actual[rep_idx]):.3f}, "
                  f"shrink={float(shrink_score[rep_idx]):.1f}×; "
                  f"picked = {pick_reason} / max-shrink)")
        out1 = output_dir / "fig1_group_strip.png"
        plot_figure1_strip(
            x_groups[rep_idx], r_groups[rep_idx],
            base[rep_idx].item(), sign_flag[rep_idx].item(),
            float(max_abs[rep_idx]),
            args.int_bits, args.r_bits, out1, title1,
        )
        print(f"[4] Figure 1 -> {out1}")
    else:
        print("[4] outlier 그룹이 하나도 없어서 Figure 1 생략")

    # 6) Figure 2: 분포 비교 (favorable mask 우선, 없으면 outlier mask)
    title2_fav = (f"L{layer_idx}.{module_name} — outlier-favorable groups: "
                  f"value distribution before vs after QR")
    if favorable_mask.any():
        out2 = output_dir / "fig2_dist_favorable.png"
        plot_figure2_distribution(
            x_groups, r_groups, favorable_mask,
            args.int_bits, args.r_bits, args.selective_base_threshold,
            out2, title2_fav,
        )
        print(f"[5] Figure 2 (favorable) -> {out2}")

    # 보조: favorable 조건이 너무 빡빡할 수 있어서 outlier mask 버전도 동반 출력
    if outlier_mask.any():
        title2_out = (f"L{layer_idx}.{module_name} — all outlier groups (base>=thr): "
                      f"value distribution before vs after QR")
        out3 = output_dir / "fig2_dist_outlier.png"
        plot_figure2_distribution(
            x_groups, r_groups, outlier_mask,
            args.int_bits, args.r_bits, args.selective_base_threshold,
            out3, title2_out,
        )
        print(f"[5] Figure 2 (outlier) -> {out3}")

    print(f"[Done] {output_dir}")


if __name__ == "__main__":
    main()
