import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from model_configs import MODEL_REGISTRY, get_model_config, get_module_names
from model_adapter import (
    load_model_and_tokenizer,
    get_model_device,
    parse_module_names,
    resolve_target_layers,
)
from eval_utils import load_eval_tokens, compute_perplexity
from replace_modules_ver3 import replace_modules_v3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_module_float_map(s, arch):
    if not s or not s.strip():
        return {}
    valid = set(get_module_names(arch))
    parsed = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"--module_selective_base_thresholds는 module:value 형식이어야 합니다. got '{item}'"
            )
        module_name, value = item.split(":", 1)
        module_name = module_name.strip()
        if module_name not in valid:
            raise ValueError(f"Invalid module name '{module_name}' for arch '{arch}'")
        parsed[module_name] = float(value.strip())
    return parsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model_source", type=str, default="auto",
                        choices=["auto", "fp16", "omni"])
    parser.add_argument("--output_dir", type=str, default="results_ver3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_only", action="store_true")

    # 적용 범위 (ver3는 모든 모듈에 동일 처리 — ver2의 qr_modules 분기 없음)
    parser.add_argument("--replace_scope", type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=0)
    parser.add_argument("--custom_layer_indices", type=str, default="")
    parser.add_argument("--target_modules", type=str, default="",
                        help="비우면 arch 전체 모듈 자동 선택")

    # calibration artifact + threshold
    parser.add_argument("--calib_artifact", type=str, required=True,
                        help="calibrate_qr_v3.py 결과 .pt 파일 경로")
    parser.add_argument("--selective_base_threshold", type=float, default=8.0,
                        help="전역 mask 임계값. group base >= threshold 면 QR, 아니면 INT4")
    parser.add_argument("--module_selective_base_thresholds", type=str, default="",
                        help="module별 threshold override. 예: fc1:8,fc2:64")
    parser.add_argument("--disable_permutation", action="store_true",
                        help="채널 reorder 비활성. perm을 identity로 override해 mask만 적용.")
    parser.add_argument("--use_static_scales", action="store_true",
                        help="Phase 1. base / INT4 scale을 calibration max_abs로 정적 고정.")

    # weight quant
    parser.add_argument("--enable_weight_quant", action="store_true")
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--weight_quant_mode", type=str, default="group",
                        choices=["tensor", "per_channel", "group"])
    parser.add_argument("--weight_ch_axis", type=int, default=0)
    parser.add_argument("--weight_group_size", type=int, default=128)
    parser.add_argument("--weight_scale_method", type=str, default="max",
                        choices=["max", "mse"])
    parser.add_argument("--weight_scale_shrink_factors", type=str,
                        default="1.0,0.95,0.9,0.85,0.8")

    # QR params
    parser.add_argument("--q_bits", type=int, default=1)
    parser.add_argument("--r_bits", type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--r_group_size", type=int, default=128)
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0)

    # INT4 params (mask=False group용)
    parser.add_argument("--act_bits", type=int, default=4)
    parser.add_argument("--act_group_size", type=int, default=128)

    # eval
    parser.add_argument("--eval_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "c4_new", "c4_omni"])
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_nsamples", type=int, default=256)
    parser.add_argument("--c4_cache_dir", type=str, default="/home/dataset/allenai_c4")
    parser.add_argument("--seqlen", type=int, default=2048)

    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.model_name
    ensure_dir(output_dir)

    print(f"[1] Loading model: {args.model_name} ...")
    model, tokenizer, model_seqlen, arch, load_path = load_model_and_tokenizer(
        args.model_name, model_source=args.model_source,
    )

    if not args.target_modules.strip():
        module_names = get_module_names(arch)
    else:
        module_names = parse_module_names(args.target_modules, arch)

    weight_scale_shrink_factors = parse_float_list(args.weight_scale_shrink_factors)
    module_selective_base_thresholds = parse_module_float_map(
        args.module_selective_base_thresholds, arch
    )

    cfg = get_model_config(args.model_name)
    weight_backend = "omni" if load_path == cfg["omni_ckpt"] else "fp16"
    if weight_backend == "omni" and not args.enable_weight_quant:
        effective_weight_mode = "omni_preweight"
    elif weight_backend == "omni" and args.enable_weight_quant:
        effective_weight_mode = "fake_weight_from_omni_double_quant"
    elif weight_backend == "fp16" and not args.enable_weight_quant:
        effective_weight_mode = "fp16_weight"
    else:
        effective_weight_mode = "fake_weight_from_fp16"

    print(f"[2] Loading calibration artifact: {args.calib_artifact} ...")
    artifact = torch.load(args.calib_artifact, map_location="cpu", weights_only=False)
    if artifact["model_name"] != args.model_name:
        raise ValueError(
            f"calib artifact model_name {artifact['model_name']} != {args.model_name}"
        )
    if artifact["arch"] != arch:
        raise ValueError(
            f"calib artifact arch {artifact['arch']} != {arch}"
        )
    if artifact["base_group_size"] != args.base_group_size:
        raise ValueError(
            f"calib artifact base_group_size {artifact['base_group_size']} != {args.base_group_size}"
        )

    if args.disable_permutation:
        # Permutation을 identity로 override 시, mask용 per_group_max_abs도 자연 순서 기준으로 재계산해야
        # mask와 runtime group 정렬이 일치한다. raw channel_max_abs는 artifact에 항상 저장되어 있다.
        print("    [override] disable_permutation=True → perm=identity + per_group_max_abs 재계산")
        gs = args.base_group_size
        for key, entry in artifact["modules"].items():
            channel_max_abs = entry["channel_max_abs"]
            n = channel_max_abs.numel()
            if n % gs != 0:
                raise ValueError(
                    f"channel count {n} not divisible by base_group_size {gs} for {key}"
                )
            G = n // gs
            entry["perm"] = torch.arange(n, dtype=entry["perm"].dtype)
            entry["per_group_max_abs"] = channel_max_abs.reshape(G, gs).amax(dim=-1)

    lines = []
    lines.append("[Config]")
    lines.append(f"model_name           : {args.model_name}")
    lines.append(f"model_source         : {args.model_source}")
    lines.append(f"arch                 : {arch}")
    lines.append(f"load_path            : {load_path}")
    lines.append(f"weight_backend       : {weight_backend}")
    lines.append(f"effective_weight_mode: {effective_weight_mode}")
    lines.append(f"model_seqlen         : {model_seqlen}")
    lines.append(f"eval_dataset         : {args.eval_dataset}")
    lines.append(f"eval_split           : {args.eval_split}")
    if args.eval_dataset in ("c4", "c4_new", "c4_omni"):
        lines.append(f"eval_nsamples        : {args.eval_nsamples}")
        lines.append(f"c4_cache_dir         : {args.c4_cache_dir}")
    lines.append(f"eval_seqlen          : {args.seqlen}")
    lines.append(f"replace_scope        : {args.replace_scope}")
    lines.append(f"baseline_only        : {args.baseline_only}")
    lines.append(f"target_modules       : {module_names}")
    lines.append(f"calib_artifact       : {args.calib_artifact}")
    lines.append(f"calib_dataset        : {artifact['calib_dataset']}")
    lines.append(f"calib_nsamples       : {artifact['calib_nsamples']}")
    lines.append(f"disable_permutation  : {args.disable_permutation}")
    lines.append(f"use_static_scales    : {args.use_static_scales}")
    lines.append(f"selective_base_threshold        : {args.selective_base_threshold}")
    lines.append(f"module_selective_base_thresholds: {module_selective_base_thresholds}")
    lines.append(f"enable_weight_quant  : {args.enable_weight_quant}")
    lines.append(f"q_bits               : {args.q_bits}")
    lines.append(f"r_bits               : {args.r_bits}")
    lines.append(f"base_group_size      : {args.base_group_size}")
    lines.append(f"r_group_size         : {args.r_group_size}")
    lines.append(f"residual_clip_alpha  : {args.residual_clip_alpha}")
    lines.append(f"act_bits             : {args.act_bits}")
    lines.append(f"act_group_size       : {args.act_group_size}")
    lines.append("")

    print(f"[3] Loading {args.eval_dataset} ...")
    tok_path = load_path
    _, testenc = load_eval_tokens(
        tok_path,
        dataset_name=args.eval_dataset,
        split=args.eval_split,
        seqlen=args.seqlen,
        nsamples=args.eval_nsamples,
        seed=args.seed,
        c4_cache_dir=args.c4_cache_dir,
    )

    dev = get_model_device(model, arch)

    print("[4] Computing baseline PPL ...")
    baseline_ppl = compute_perplexity(model, testenc, dev, args.seqlen)
    lines.append("[Baseline]")
    lines.append(f"baseline_ppl : {baseline_ppl:.8f}")
    lines.append("")
    print(f"    baseline PPL = {baseline_ppl:.4f}")

    if args.baseline_only:
        print("\n".join(lines))
        summary_path = output_dir / "summary.txt"
        dataset_summary_path = output_dir / f"summary_{args.eval_dataset}_{args.eval_split}_{weight_backend}_baseline.txt"
        save_txt(lines, summary_path)
        save_txt(lines, dataset_summary_path)
        print(f"[Done] Summary saved to {summary_path}")
        print(f"[Done] Dataset summary saved to {dataset_summary_path}")
        return

    print("[5] Replacing modules with QuotRemLinearV3 (perm + static mask) ...")
    layer_indices = resolve_target_layers(
        model, arch, args.replace_scope, args.one_layer_idx, args.custom_layer_indices
    )
    replaced = replace_modules_v3(
        model=model,
        arch=arch,
        layer_indices=layer_indices,
        target_modules=module_names,
        calib_modules=artifact["modules"],
        selective_base_threshold=args.selective_base_threshold,
        module_selective_base_thresholds=module_selective_base_thresholds,
        enable_weight_quant=args.enable_weight_quant,
        weight_bits=args.weight_bits,
        weight_quant_mode=args.weight_quant_mode,
        weight_ch_axis=args.weight_ch_axis,
        weight_group_size=args.weight_group_size,
        weight_scale_method=args.weight_scale_method,
        weight_scale_shrink_factors=weight_scale_shrink_factors,
        q_bits=args.q_bits,
        r_bits=args.r_bits,
        base_group_size=args.base_group_size,
        r_group_size=args.r_group_size,
        residual_clip_alpha=args.residual_clip_alpha,
        act_bits=args.act_bits,
        act_group_size=args.act_group_size,
        use_static_scales=args.use_static_scales,
    )

    total_qr   = sum(q for _, _, q, _ in replaced)
    total_grps = sum(t for _, _, _, t in replaced)
    lines.append("[Replacement]")
    lines.append(f"replaced_count   : {len(replaced)}")
    lines.append(f"total_qr_groups  : {total_qr}")
    lines.append(f"total_groups     : {total_grps}")
    lines.append(f"global_qr_ratio  : {total_qr / max(total_grps, 1):.4f}")
    for name, thr, qr, tot in replaced:
        lines.append(f"  - {name}  threshold={thr}  qr_groups={qr}/{tot} ({qr/max(tot,1):.3f})")
    lines.append("")

    print("[6] Computing quantized PPL ...")
    modified_ppl = compute_perplexity(model, testenc, dev, args.seqlen)
    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines.append("[Result]")
    lines.append(f"modified_ppl : {modified_ppl:.8f}")
    lines.append(f"ppl_diff     : {ppl_diff:.12e}")
    lines.append(f"relative_diff: {rel_diff:.12e}")
    lines.append("")

    print("\n".join(lines))
    summary_path = output_dir / "summary.txt"
    dataset_summary_path = output_dir / f"summary_{args.eval_dataset}_{args.eval_split}_{weight_backend}.txt"
    save_txt(lines, summary_path)
    save_txt(lines, dataset_summary_path)
    print(f"[Done] Summary saved to {summary_path}")
    print(f"[Done] Dataset summary saved to {dataset_summary_path}")


if __name__ == "__main__":
    main()
