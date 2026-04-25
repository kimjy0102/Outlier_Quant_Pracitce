import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# R(나머지) 값 분포 수집 및 분석 스크립트
# quant_ver2_up_sep.py 의 QuotRemLinear 를 collect_residuals=True 로 교체한 뒤
# calibration 데이터를 n_batches 만큼 forward pass → R 분포 히스토그램 + 통계 출력

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # 서버 환경: 디스플레이 없이 이미지 저장
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# quant_ver2_up_sep.py 에서 필요한 것만 import
from quant_ver2_up_sep import (
    QuotRemLinear,
    replace_modules_with_quotrem_linear,
    resolve_target_layers,
    parse_module_names,
    set_seed,
    ensure_dir,
    get_model_main_device,
    save_txt,
)


# =========================================================
# 1. Calibration 데이터 로드 (wikitext2 train split)
# =========================================================
def load_calib_data(model_id, n_samples, seq_len=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)     # tokenizer 로드
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")# wikitext2 train
    full_text = "\n\n".join(ds["text"])                                      # 전체 텍스트 합치기
    enc       = tokenizer(full_text, return_tensors="pt").input_ids          # 토크나이즈

    samples = []
    for i in range(n_samples):
        start = i * seq_len                                                  # 시작 위치
        if start + seq_len > enc.shape[1]:                                   # 토큰 부족하면 종료
            break
        samples.append(enc[:, start : start + seq_len])                     # seq_len 단위로 잘라서 추가
    return samples


# =========================================================
# 2. Forward pass 실행 → R 버퍼 채우기
# =========================================================
def run_forward_passes(model, calib_samples, device, n_batches):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_samples[:n_batches], desc="Collecting R values"):
            batch = batch.to(device)                                         # GPU로 이동
            model(batch, use_cache=False)                                    # forward (출력 불필요)


# =========================================================
# 3. 모델에서 QuotRemLinear 모듈 전체 수집
# =========================================================
def get_all_quotrem_modules(model):
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, QuotRemLinear):                                # QuotRemLinear만 선택
            modules[name] = module
    return modules


# =========================================================
# 4. 통계 계산
# =========================================================
def compute_stats(values: torch.Tensor) -> dict:
    v     = values.float()                                                   # float32로 변환
    percs = [1, 5, 10, 25, 50, 75, 90, 95, 99]                              # 확인할 백분위수 목록
    pvals = torch.quantile(v, torch.tensor(percs, dtype=torch.float32) / 100.0).tolist()

    return {
        "n"            : len(v),
        "mean"         : v.mean().item(),
        "std"          : v.std().item(),
        "min"          : v.min().item(),
        "max"          : v.max().item(),
        "abs_mean"     : v.abs().mean().item(),                              # 절댓값 평균 (R 크기 대표값)
        "abs_max"      : v.abs().max().item(),                               # 절댓값 최대
        "frac_zero"    : (v == 0).float().mean().item(),                     # R=0 비율 (Q가 정확히 맞은 경우)
        "frac_abs_lt1" : (v.abs() < 1.0).float().mean().item(),              # |R|<1 비율
        "frac_abs_lt2" : (v.abs() < 2.0).float().mean().item(),              # |R|<2 비율
        "percentiles"  : dict(zip(percs, pvals)),
    }


# =========================================================
# 4-2. Q 통계 계산
# =========================================================
def compute_q_stats(values: torch.Tensor) -> dict:
    v      = values.float()
    total  = len(v)                                          # 전체 Q 원소 수
    unique = v.unique().tolist()                             # 존재하는 Q 값 목록
    counts = {int(val): (v == val).sum().item() for val in unique}  # 값별 개수
    fracs  = {k: cnt / total for k, cnt in counts.items()}  # 값별 비율

    return {
        "n"      : total,
        "counts" : counts,   # {0: 개수, 1: 개수} 형태
        "fracs"  : fracs,    # {0: 비율, 1: 비율} 형태
    }


# =========================================================
# 4-3. Base 통계 계산 (실제로 관측된 base 값을 전부 count/fraction으로 출력)
# =========================================================
def compute_base_stats(values: torch.Tensor) -> dict:
    v      = values.float()
    total  = len(v)                                                 # 전체 base 샘플 수 (per-group 단위)
    unique = sorted(v.unique().tolist())                            # 실제 관측된 base 값 전체
    counts = {}
    fracs  = {}
    for b in unique:
        cnt        = (v == b).sum().item()                          # 해당 base가 선택된 횟수
        counts[b]  = cnt
        fracs[b]   = cnt / total if total > 0 else 0.0             # 비율
    return {
        "n"      : total,
        "counts" : counts,   # {base값: 개수}
        "fracs"  : fracs,    # {base값: 비율}
    }


def format_base_value(value: float) -> str:
    if float(value).is_integer() and value >= 1.0:
        return f"{int(value):3d}"
    return f"{value:.6g}"


def print_base_stats(name: str, stats: dict):
    print(f"\n{'='*60}")
    print(f"[{name}] Base distribution")
    print(f"  total : {stats['n']:,}")
    for b in sorted(stats["counts"].keys()):
        cnt  = stats["counts"][b]
        frac = stats["fracs"][b]
        print(f"  base={format_base_value(b):>8s}  : {cnt:>10,}  ({frac*100:.2f}%)")


def base_stats_to_lines(name: str, stats: dict) -> list:
    lines = [f"[{name}] Base distribution", f"  total : {stats['n']:,}"]
    for b in sorted(stats["counts"].keys()):
        cnt  = stats["counts"][b]
        frac = stats["fracs"][b]
        lines.append(f"  base={format_base_value(b):>8s}  : {cnt:>10,}  ({frac*100:.2f}%)")
    lines.append("")
    return lines


# =========================================================
# 4-4. Base group별 Q=1 비율 통계 계산
# group_values columns: [base, q1_count, q0_count, q1_fraction]
# =========================================================
def compute_base_group_q1_stats(group_values: torch.Tensor) -> dict:
    v        = group_values.float()
    base     = v[:, 0]
    q1_count = v[:, 1]
    q0_count = v[:, 2]
    q1_frac  = v[:, 3]
    percs    = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pvals    = torch.quantile(q1_frac, torch.tensor(percs, dtype=torch.float32) / 100.0).tolist()

    count_unique = sorted(q1_count.unique().tolist())
    q1_count_counts = {int(c): (q1_count == c).sum().item() for c in count_unique}

    by_base = {}
    for b in sorted(base.unique().tolist()):
        mask = base == b
        frac_b = q1_frac[mask]
        q1_count_b = q1_count[mask]
        q0_count_b = q0_count[mask]
        count_pairs = []
        for c in sorted(q1_count_b.unique().tolist()):
            count_mask = q1_count_b == c
            n_groups = int(count_mask.sum().item())
            q0_value = int(q0_count_b[count_mask][0].item())
            count_pairs.append((int(c), q0_value, n_groups, n_groups / int(mask.sum().item())))
        by_base[b] = {
            "n_groups"       : int(mask.sum().item()),
            "q1_frac_mean"   : frac_b.mean().item(),
            "q1_frac_p50"    : torch.quantile(frac_b, 0.50).item(),
            "q1_frac_p90"    : torch.quantile(frac_b, 0.90).item(),
            "q1_frac_max"    : frac_b.max().item(),
            "frac_no_q1"     : (frac_b == 0).float().mean().item(),
            "count_pairs"    : count_pairs,
        }

    return {
        "n_groups"        : len(v),
        "q1_count_mean"   : q1_count.mean().item(),
        "q1_count_max"    : q1_count.max().item(),
        "q1_frac_mean"    : q1_frac.mean().item(),
        "frac_groups_q1_0": (q1_count == 0).float().mean().item(),
        "frac_groups_q1_1p": (q1_count > 0).float().mean().item(),
        "q1_frac_percentiles": dict(zip(percs, pvals)),
        "q1_count_counts" : q1_count_counts,
        "by_base"         : by_base,
    }


def print_base_group_q1_stats(name: str, stats: dict):
    print(f"\n{'='*60}")
    print(f"[{name}] Base-group Q=1 ratio")
    print(f"  total groups        : {stats['n_groups']:,}")
    print(f"  mean Q=1 count/group: {stats['q1_count_mean']:.4f}")
    print(f"  max  Q=1 count/group: {stats['q1_count_max']:.0f}")
    print(f"  mean Q=1 frac/group : {stats['q1_frac_mean']:.6f}")
    print(f"  groups with no Q=1  : {stats['frac_groups_q1_0']:.4f}")
    print(f"  groups with any Q=1 : {stats['frac_groups_q1_1p']:.4f}")
    print("  Q=1 fraction percentiles:")
    for p, v in stats["q1_frac_percentiles"].items():
        print(f"    p{p:3d}              : {v:.6f}")
    print("  by base:")
    for b, row in stats["by_base"].items():
        print(
            f"    base={format_base_value(b):>8s}  groups={row['n_groups']:>8,}  "
            f"mean={row['q1_frac_mean']:.6f}  p50={row['q1_frac_p50']:.6f}  "
            f"p90={row['q1_frac_p90']:.6f}  max={row['q1_frac_max']:.6f}  "
            f"no_q1={row['frac_no_q1']:.4f}"
        )
        for q1_cnt, q0_cnt, n_groups, frac in row["count_pairs"]:
            print(
                f"      Q1={q1_cnt:4d}, Q0={q0_cnt:4d} : "
                f"{n_groups:>8,} groups ({frac*100:.2f}%)"
            )


def base_group_q1_stats_to_lines(name: str, stats: dict) -> list:
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
    lines.append("  Q=1 count per group distribution:")
    for cnt, n in stats["q1_count_counts"].items():
        lines.append(f"    q1_count={cnt:4d}      : {n:>10,}  ({n / stats['n_groups'] * 100:.2f}%)")
    lines.append("  by base:")
    for b, row in stats["by_base"].items():
        lines.append(
            f"    base={format_base_value(b):>8s}  groups={row['n_groups']:>8,}  "
            f"mean={row['q1_frac_mean']:.6f}  p50={row['q1_frac_p50']:.6f}  "
            f"p90={row['q1_frac_p90']:.6f}  max={row['q1_frac_max']:.6f}  "
            f"no_q1={row['frac_no_q1']:.4f}"
        )
        for q1_cnt, q0_cnt, n_groups, frac in row["count_pairs"]:
            lines.append(
                f"      Q1={q1_cnt:4d}, Q0={q0_cnt:4d} : "
                f"{n_groups:>8,} groups ({frac*100:.2f}%)"
            )
    lines.append("")
    return lines


def save_base_group_csv(group_values: torch.Tensor, save_path: str):
    arr = group_values.float().cpu().numpy()
    header = "group_sample_idx,base,q1_count,q0_count,q1_fraction"
    rows = np.column_stack([np.arange(arr.shape[0]), arr])
    np.savetxt(
        save_path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.8g", "%.0f", "%.0f", "%.8f"],
    )


def print_q_stats(name: str, stats: dict):
    print(f"\n{'='*60}")
    print(f"[{name}] Q distribution")
    print(f"  total : {stats['n']:,}")
    for val in sorted(stats["counts"].keys()):
        cnt  = stats["counts"][val]
        frac = stats["fracs"][val]
        print(f"  Q={val:2d}  : {cnt:>10,}  ({frac*100:.2f}%)")


def q_stats_to_lines(name: str, stats: dict) -> list:
    lines = [f"[{name}] Q distribution", f"  total : {stats['n']:,}"]
    for val in sorted(stats["counts"].keys()):
        cnt  = stats["counts"][val]
        frac = stats["fracs"][val]
        lines.append(f"  Q={val:2d}  : {cnt:>10,}  ({frac*100:.2f}%)")
    lines.append("")
    return lines


# =========================================================
# 5. 통계 콘솔 출력
# =========================================================
def print_stats(name: str, stats: dict):
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"  n              : {stats['n']:,}")
    print(f"  mean           : {stats['mean']:.6f}")
    print(f"  std            : {stats['std']:.6f}")
    print(f"  min / max      : {stats['min']:.6f} / {stats['max']:.6f}")
    print(f"  abs_mean       : {stats['abs_mean']:.6f}")
    print(f"  abs_max        : {stats['abs_max']:.6f}")
    print(f"  frac_zero      : {stats['frac_zero']:.4f}  (R=0 비율)")
    print(f"  frac_|r|<1     : {stats['frac_abs_lt1']:.4f}")
    print(f"  frac_|r|<2     : {stats['frac_abs_lt2']:.4f}")
    print("  percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    p{p:3d}         : {v:.6f}")


# =========================================================
# 6. 통계 → 텍스트 라인 변환 (파일 저장용)
# =========================================================
def stats_to_lines(name: str, stats: dict) -> list:
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


# =========================================================
# 7. 레이어별 R 히스토그램 저장
# =========================================================
def plot_histogram(values: torch.Tensor, name: str, save_path: str, n_bins: int = 100):
    v    = values.float().numpy()
    p1   = np.percentile(v, 1)                                              # 하위 1% 값
    p99  = np.percentile(v, 99)                                             # 상위 1% 값

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"R distribution: {name}", fontsize=11)

    # 왼쪽: 전체 범위 (극단값 포함)
    axes[0].hist(v, bins=n_bins, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--")            # R=0 기준선
    axes[0].set_title("Full range")
    axes[0].set_xlabel("R value")
    axes[0].set_ylabel("Count")

    # 오른쪽: p1~p99 범위만 표시 (clip 아님 — 범위 밖 값은 무시, 경계에 쌓이지 않음)
    axes[1].hist(v, bins=n_bins, range=(p1, p99), color="coral", edgecolor="none", alpha=0.8)
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_title(f"Zoomed [p1={p1:.2f}, p99={p99:.2f}]")
    axes[1].set_xlabel("R value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =========================================================
# 8. 전체 레이어 overview (abs_mean 비교 막대 그래프)
# =========================================================
def plot_overview(layer_stats: dict, save_path: str):
    names    = list(layer_stats.keys())
    abs_mean = [layer_stats[n]["abs_mean"] for n in names]                  # 레이어별 abs_mean
    abs_max  = [layer_stats[n]["abs_max"]  for n in names]                  # 레이어별 abs_max

    x   = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.35), 5))
    ax.bar(x, abs_mean, label="abs_mean", color="steelblue", alpha=0.85)    # abs_mean 막대
    ax.plot(x, abs_max,  label="abs_max",  color="coral",    marker="o",
            linewidth=1, markersize=3)                                       # abs_max 라인
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("R value magnitude")
    ax.set_title("R abs_mean & abs_max per layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# =========================================================
# 9. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir",   type=str, default="results/r_analysis")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_batches",    type=int, default=10,   help="R 수집용 forward pass 횟수")
    parser.add_argument("--n_bins",       type=int, default=100,  help="히스토그램 bin 수")

    # 교체 범위 (quant_ver2_up_sep.py와 동일)
    parser.add_argument("--replace_scope",        type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx",         type=int, default=10)
    parser.add_argument("--target_modules",        type=str,
                        default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2")
    parser.add_argument("--custom_layer_indices",  type=str, default=None)

    # weight quant
    parser.add_argument("--enable_weight_quant",  action="store_true")
    parser.add_argument("--weight_bits",           type=int, default=4)
    parser.add_argument("--weight_quant_mode",     type=str, default="group")
    parser.add_argument("--weight_ch_axis",        type=int, default=0)
    parser.add_argument("--weight_group_size",     type=int, default=128)

    # QR 설정 (quant_ver2_up_sep.py와 동일)
    parser.add_argument("--q_bits",          type=int, default=1)
    parser.add_argument("--r_bits",          type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=64)
    parser.add_argument("--r_group_size",    type=int, default=16)

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)

    # 모델 로드
    print("[1] Loading model...")
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

    # collect_residuals=True 로 모듈 교체
    print("[2] Replacing modules with collect_residuals=True ...")
    replace_modules_with_quotrem_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        enable_weight_quant=args.enable_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        q_bits=args.q_bits,
        r_bits=args.r_bits,
        base_group_size=args.base_group_size,
        r_group_size=args.r_group_size,
        collect_residuals=True,                         # 분석 모드 활성화
    )

    # Calibration 데이터 로드 + forward pass 실행
    print("[3] Loading calibration data (wikitext2 train) ...")
    calib_samples = load_calib_data(args.model_id, n_samples=args.n_batches + 5)

    device = get_model_main_device(model)
    print(f"[4] Running {args.n_batches} forward passes ...")
    run_forward_passes(model, calib_samples, device, n_batches=args.n_batches)

    # 분석
    print("[5] Analyzing R distributions ...")
    qr_modules = get_all_quotrem_modules(model)

    all_stats  = {}
    stat_lines = ["[R Value Statistics]\n"]

    for name, module in qr_modules.items():
        if not module._r_buf:                                               # 수집된 값 없으면 skip
            continue

        r_vals = torch.cat(module._r_buf)                                  # 버퍼 전체를 하나의 텐서로 합산

        stats = compute_stats(r_vals)
        all_stats[name] = stats

        print_stats(name, stats)                                            # R 통계 콘솔 출력
        stat_lines.extend(stats_to_lines(name, stats))                     # R 통계 파일 저장용

        # Q 통계
        if module._q_buf:
            q_vals   = torch.cat(module._q_buf)                            # Q 버퍼 합산
            q_stats  = compute_q_stats(q_vals)
            print_q_stats(name, q_stats)                                   # Q 통계 콘솔 출력
            stat_lines.extend(q_stats_to_lines(name, q_stats))            # Q 통계 파일 저장용

        # Base 통계 (실제로 관측된 base 값 전체)
        if module._base_buf:
            base_vals  = torch.cat(module._base_buf)                       # base 버퍼 합산
            base_stats = compute_base_stats(base_vals)
            print_base_stats(name, base_stats)                             # base 통계 콘솔 출력
            stat_lines.extend(base_stats_to_lines(name, base_stats))      # base 통계 파일 저장용

        # Base group별 상세 통계: 각 group의 base와 그 group 안에서 Q=1인 비율을 함께 저장
        if hasattr(module, "_base_group_buf") and module._base_group_buf:
            group_vals  = torch.cat(module._base_group_buf)                # columns: [base, q1_count, q0_count, q1_fraction]
            group_stats = compute_base_group_q1_stats(group_vals)
            print_base_group_q1_stats(name, group_stats)
            stat_lines.extend(base_group_q1_stats_to_lines(name, group_stats))

            safe_name = name.replace(".", "_").replace("/", "_")
            csv_path = str(Path(args.output_dir) / f"base_group_q1_{safe_name}.csv")
            save_base_group_csv(group_vals, csv_path)
            print(f"  → {csv_path}")

        # 레이어별 히스토그램 저장
        safe_name = name.replace(".", "_").replace("/", "_")               # 파일명 안전하게 변환
        hist_path = str(Path(args.output_dir) / f"r_hist_{safe_name}.png")
        plot_histogram(r_vals.cpu(), name, hist_path, n_bins=args.n_bins)
        print(f"  → {hist_path}")

    # 전체 레이어 overview 그래프
    if all_stats:
        overview_path = str(Path(args.output_dir) / "r_overview.png")
        plot_overview(all_stats, overview_path)
        print(f"\n[Overview] → {overview_path}")

    # 통계 텍스트 파일 저장
    txt_path = str(Path(args.output_dir) / "r_stats.txt")
    save_txt(stat_lines, txt_path)
    print(f"[Stats]    → {txt_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
