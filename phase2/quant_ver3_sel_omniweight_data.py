import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# quant_ver3_sel_omniweight.py 의 R/Q/base/selective 분포 수집 및 분석 스크립트
# - OmniQuant W4A16g128 weight + Selective QR 모듈 교체
# - collect_residuals=True 로 R/Q/base/_base_group_buf 수집
# - base 분포로부터 selective routing (INT4 vs QR) 비율도 분석

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_ver3_sel_omniweight import (
    QuotRemLinear,
    replace_modules_with_quotrem_linear,
    resolve_target_layers,
    parse_module_names,
    parse_module_float_map,
    get_model_main_device,
    ensure_dir,
    save_txt,
    set_seed,
)


# =========================================================
# 1. 데이터 로드
# =========================================================
def load_calib_data(model_id, n_samples, seq_len=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join(ds["text"])
    enc       = tokenizer(full_text, return_tensors="pt").input_ids
    samples   = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > enc.shape[1]:
            break
        samples.append(enc[:, start: start + seq_len])
    return samples


# =========================================================
# 2. Forward pass
# =========================================================
def run_forward_passes(model, calib_samples, device, n_batches):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_samples[:n_batches], desc="Collecting"):
            model(batch.to(device), use_cache=False)


# =========================================================
# 3. 모듈 수집
# =========================================================
def get_all_quotrem_modules(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, QuotRemLinear)}


# =========================================================
# 4. 통계 함수들
# =========================================================
def compute_stats(values):
    v = values.float()
    percs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pvals = torch.quantile(v, torch.tensor(percs, dtype=torch.float32) / 100.0).tolist()
    return {
        "n"            : len(v),
        "mean"         : v.mean().item(),
        "std"          : v.std().item(),
        "min"          : v.min().item(),
        "max"          : v.max().item(),
        "abs_mean"     : v.abs().mean().item(),
        "abs_max"      : v.abs().max().item(),
        "frac_zero"    : (v == 0).float().mean().item(),
        "frac_abs_lt1" : (v.abs() < 1.0).float().mean().item(),
        "frac_abs_lt2" : (v.abs() < 2.0).float().mean().item(),
        "percentiles"  : dict(zip(percs, pvals)),
    }


def compute_q_stats(values):
    v      = values.float()
    total  = len(v)
    unique = v.unique().tolist()
    counts = {int(val): (v == val).sum().item() for val in unique}
    fracs  = {k: cnt / total for k, cnt in counts.items()}
    return {"n": total, "counts": counts, "fracs": fracs}


def compute_base_stats(values):
    v      = values.float()
    total  = len(v)
    unique = sorted(v.unique().tolist())
    counts = {b: (v == b).sum().item() for b in unique}
    fracs  = {b: counts[b] / total for b in unique}
    return {"n": total, "counts": counts, "fracs": fracs}


def compute_selective_stats(base_vals, threshold):
    v         = base_vals.float()
    total     = len(v)
    n_qr      = (v >= threshold).sum().item()
    n_int4    = (v <  threshold).sum().item()
    return {
        "n"         : total,
        "threshold" : threshold,
        "n_qr"      : n_qr,
        "n_int4"    : n_int4,
        "frac_qr"   : n_qr   / total if total > 0 else 0.0,
        "frac_int4" : n_int4 / total if total > 0 else 0.0,
    }


def compute_base_group_q1_stats(group_values):
    v        = group_values.float()
    q1_frac  = v[:, 3]
    q1_count = v[:, 1]
    base     = v[:, 0]
    percs    = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pvals    = torch.quantile(q1_frac, torch.tensor(percs, dtype=torch.float32) / 100.0).tolist()
    by_base  = {}
    for b in sorted(base.unique().tolist()):
        mask   = base == b
        frac_b = q1_frac[mask]
        by_base[b] = {
            "n_groups"     : int(mask.sum().item()),
            "q1_frac_mean" : frac_b.mean().item(),
            "q1_frac_p50"  : torch.quantile(frac_b, 0.50).item(),
            "q1_frac_max"  : frac_b.max().item(),
            "frac_no_q1"   : (frac_b == 0).float().mean().item(),
        }
    return {
        "n_groups"           : len(v),
        "q1_count_mean"      : q1_count.mean().item(),
        "q1_count_max"       : q1_count.max().item(),
        "q1_frac_mean"       : q1_frac.mean().item(),
        "frac_groups_q1_0"   : (q1_count == 0).float().mean().item(),
        "frac_groups_q1_1p"  : (q1_count > 0).float().mean().item(),
        "q1_frac_percentiles": dict(zip(percs, pvals)),
        "by_base"            : by_base,
    }


# =========================================================
# 5. 텍스트 변환 함수들
# =========================================================
def fmt_base(b):
    return f"{int(b):3d}" if float(b).is_integer() and b >= 1.0 else f"{b:.6g}"


def stats_to_lines(name, stats):
    lines = [
        f"[{name}]",
        f"  n            : {stats['n']:,}",
        f"  mean         : {stats['mean']:.6f}",
        f"  std          : {stats['std']:.6f}",
        f"  min/max      : {stats['min']:.6f} / {stats['max']:.6f}",
        f"  abs_mean     : {stats['abs_mean']:.6f}",
        f"  abs_max      : {stats['abs_max']:.6f}",
        f"  frac_zero    : {stats['frac_zero']:.4f}",
        f"  frac_|r|<1   : {stats['frac_abs_lt1']:.4f}",
        f"  frac_|r|<2   : {stats['frac_abs_lt2']:.4f}",
    ]
    for p, v in stats["percentiles"].items():
        lines.append(f"  p{p:3d}         : {v:.6f}")
    lines.append("")
    return lines


def q_stats_to_lines(name, stats):
    lines = [f"[{name}] Q distribution", f"  total : {stats['n']:,}"]
    for val in sorted(stats["counts"].keys()):
        lines.append(f"  Q={val:2d}  : {stats['counts'][val]:>10,}  ({stats['fracs'][val]*100:.2f}%)")
    lines.append("")
    return lines


def base_stats_to_lines(name, stats):
    lines = [f"[{name}] Base distribution", f"  total : {stats['n']:,}"]
    for b in sorted(stats["counts"].keys()):
        lines.append(f"  base={fmt_base(b):>8s}  : {stats['counts'][b]:>10,}  ({stats['fracs'][b]*100:.2f}%)")
    lines.append("")
    return lines


def selective_stats_to_lines(name, stats):
    lines = [
        f"[{name}] Selective routing (threshold={stats['threshold']})",
        f"  total groups : {stats['n']:,}",
        f"  QR   (base>= thr) : {stats['n_qr']:>10,}  ({stats['frac_qr']*100:.2f}%)",
        f"  INT4 (base<  thr) : {stats['n_int4']:>10,}  ({stats['frac_int4']*100:.2f}%)",
        "",
    ]
    return lines


def base_group_q1_stats_to_lines(name, stats):
    lines = [
        f"[{name}] Base-group Q=1 ratio",
        f"  total groups          : {stats['n_groups']:,}",
        f"  mean Q=1 count/group  : {stats['q1_count_mean']:.4f}",
        f"  max  Q=1 count/group  : {stats['q1_count_max']:.0f}",
        f"  mean Q=1 frac/group   : {stats['q1_frac_mean']:.6f}",
        f"  groups with no Q=1    : {stats['frac_groups_q1_0']:.4f}",
        f"  groups with any Q=1   : {stats['frac_groups_q1_1p']:.4f}",
        "  Q=1 fraction percentiles:",
    ]
    for p, v in stats["q1_frac_percentiles"].items():
        lines.append(f"    p{p:3d}                : {v:.6f}")
    lines.append("  by base:")
    for b, row in stats["by_base"].items():
        lines.append(
            f"    base={fmt_base(b):>8s}  groups={row['n_groups']:>8,}  "
            f"mean={row['q1_frac_mean']:.6f}  p50={row['q1_frac_p50']:.6f}  "
            f"max={row['q1_frac_max']:.6f}  no_q1={row['frac_no_q1']:.4f}"
        )
    lines.append("")
    return lines


# =========================================================
# 6. 히스토그램 및 overview 플롯
# =========================================================
def plot_histogram(values, name, save_path, n_bins=100):
    v   = values.float().numpy()
    p1  = np.percentile(v, 1)
    p99 = np.percentile(v, 99)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"R distribution: {name}", fontsize=11)
    axes[0].hist(v, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_title("Full range")
    axes[0].set_xlabel("R value")
    axes[0].set_ylabel("Count")
    axes[1].hist(v, bins=n_bins, range=(p1, p99), color="coral", edgecolor="none", alpha=0.8)
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_title(f"Zoomed [p1={p1:.2f}, p99={p99:.2f}]")
    axes[1].set_xlabel("R value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_overview(layer_stats, selective_stats_all, save_path):
    names    = list(layer_stats.keys())
    abs_mean = [layer_stats[n]["abs_mean"] for n in names]
    abs_max  = [layer_stats[n]["abs_max"]  for n in names]
    frac_qr  = [selective_stats_all.get(n, {}).get("frac_qr", 1.0) for n in names]

    x   = list(range(len(names)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.35), 10))

    ax1.bar(x, abs_mean, label="R abs_mean", color="steelblue", alpha=0.85)
    ax1.plot(x, abs_max, label="R abs_max", color="coral", marker="o", linewidth=1, markersize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=7)
    ax1.set_ylabel("R value magnitude")
    ax1.set_title("R abs_mean & abs_max per layer")
    ax1.legend()

    ax2.bar(x, frac_qr,  label="QR path",   color="steelblue", alpha=0.85)
    ax2.bar(x, [1 - v for v in frac_qr], bottom=frac_qr, label="INT4 path", color="orange", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=7)
    ax2.set_ylabel("Group ratio")
    ax2.set_title("Selective routing ratio (QR vs INT4) per layer")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =========================================================
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     type=str,
                        default="/home2/juneyeop/OmniQuant/checkpoint/opt-6.7b-w4a16g128")
    parser.add_argument("--output_dir",   type=str, default="results/r_analysis_ver3_sel_omni")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_batches",    type=int, default=10)
    parser.add_argument("--n_bins",       type=int, default=100)

    parser.add_argument("--replace_scope",       type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx",       type=int, default=10)
    parser.add_argument("--target_modules",      type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices", type=str, default=None)

    parser.add_argument("--q_bits",                  type=int,   default=1)
    parser.add_argument("--r_bits",                  type=int,   default=3)
    parser.add_argument("--base_group_size",         type=int,   default=128)
    parser.add_argument("--r_group_size",            type=int,   default=128)
    parser.add_argument("--selective_base_threshold", type=float, default=8.0)
    parser.add_argument("--module_selective_base_thresholds", type=str, default="")
    parser.add_argument("--selective_int_bits",      type=int,   default=4)
    parser.add_argument("--residual_clip_alpha",     type=float, default=0.0)

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)
    module_selective_base_thresholds = parse_module_float_map(args.module_selective_base_thresholds)

    print("[1] Loading model (OmniQuant W4A16g128 checkpoint) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    target_layer_indices = resolve_target_layers(
        model, args.replace_scope, args.one_layer_idx, args.custom_layer_indices
    )

    print("[2] Replacing modules (enable_weight_quant=False, OmniQuant weight 보존) ...")
    replace_modules_with_quotrem_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        enable_weight_quant=False,          # OmniQuant weight double-quant 방지
        weight_bits=4,
        weight_quant_mode="group",
        weight_ch_axis=0,
        weight_group_size=128,
        weight_scale_method="max",
        weight_scale_shrink_factors=None,
        q_bits=args.q_bits,
        r_bits=args.r_bits,
        base_group_size=args.base_group_size,
        r_group_size=args.r_group_size,
        selective_base_threshold=args.selective_base_threshold,
        module_selective_base_thresholds=module_selective_base_thresholds,
        selective_int_bits=args.selective_int_bits,
        residual_clip_alpha=args.residual_clip_alpha,
    )

    # collect_residuals 활성화 (replace 함수 인자로 없으므로 교체 후 직접 세팅)
    for _, m in model.named_modules():
        if isinstance(m, QuotRemLinear):
            m.collect_residuals = True

    print("[3] Loading calibration data (wikitext2 train) ...")
    calib_samples = load_calib_data(args.model_id, n_samples=args.n_batches + 5)

    device = get_model_main_device(model)
    print(f"[4] Running {args.n_batches} forward passes ...")
    run_forward_passes(model, calib_samples, device, n_batches=args.n_batches)

    print("[5] Analyzing distributions ...")
    qr_modules = get_all_quotrem_modules(model)

    all_r_stats       = {}
    all_selective_stats = {}
    stat_lines        = ["[R / Q / Base / Selective Statistics — quant_ver3_sel_omniweight]\n"]

    for name, module in qr_modules.items():
        if not module._r_buf:
            continue

        r_vals = torch.cat(module._r_buf)
        stats  = compute_stats(r_vals)
        all_r_stats[name] = stats
        stat_lines.extend(stats_to_lines(name, stats))

        if module._q_buf:
            q_vals = torch.cat(module._q_buf)
            stat_lines.extend(q_stats_to_lines(name, compute_q_stats(q_vals)))

        if module._base_buf:
            base_vals  = torch.cat(module._base_buf)
            base_stats = compute_base_stats(base_vals)
            stat_lines.extend(base_stats_to_lines(name, base_stats))

            # selective routing 비율 (base 분포에서 threshold 기준 분리)
            module_thr = module_selective_base_thresholds.get(name.split(".")[-2] + "." + name.split(".")[-1],
                         args.selective_base_threshold)
            sel_stats  = compute_selective_stats(base_vals, module_thr)
            all_selective_stats[name] = sel_stats
            stat_lines.extend(selective_stats_to_lines(name, sel_stats))

        if hasattr(module, "_base_group_buf") and module._base_group_buf:
            group_vals  = torch.cat(module._base_group_buf)
            group_stats = compute_base_group_q1_stats(group_vals)
            stat_lines.extend(base_group_q1_stats_to_lines(name, group_stats))

            safe = name.replace(".", "_").replace("/", "_")
            csv_path = str(Path(args.output_dir) / f"base_group_q1_{safe}.csv")
            arr    = group_vals.float().cpu().numpy()
            header = "idx,base,q1_count,q0_count,q1_fraction"
            np.savetxt(csv_path, np.column_stack([np.arange(len(arr)), arr]),
                       delimiter=",", header=header, comments="",
                       fmt=["%d", "%.8g", "%.0f", "%.0f", "%.8f"])
            print(f"  → {csv_path}")

        safe     = name.replace(".", "_").replace("/", "_")
        hist_path = str(Path(args.output_dir) / f"r_hist_{safe}.png")
        plot_histogram(r_vals.cpu(), name, hist_path, n_bins=args.n_bins)
        print(f"  → {hist_path}")

    if all_r_stats:
        overview_path = str(Path(args.output_dir) / "overview.png")
        plot_overview(all_r_stats, all_selective_stats, overview_path)
        print(f"\n[Overview] → {overview_path}")

    txt_path = str(Path(args.output_dir) / "r_stats.txt")
    save_txt(stat_lines, txt_path)
    print(f"[Stats]    → {txt_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
