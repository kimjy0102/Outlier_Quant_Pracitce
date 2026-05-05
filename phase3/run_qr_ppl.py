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
from replace_modules import replace_modules_with_quotrem_linear


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
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"모델 이름. 선택지: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--model_source", type=str, default="auto",
                        choices=["auto", "fp16", "omni"],
                        help="auto: omni_ckpt가 있으면 사용, fp16: 원본 모델 강제, omni: OmniQuant checkpoint 강제")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_only", action="store_true",
                        help="baseline PPL만 계산하고 QR module 교체/modified PPL 계산은 건너뜁니다.")

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=0)
    parser.add_argument("--custom_layer_indices", type=str, default="")
    # 비워두면 arch에 따라 자동으로 전체 모듈 사용 (OPT/Llama 혼용 에러 방지)
    parser.add_argument("--target_modules", type=str, default="",
                        help="쉼표로 구분. 비우면 arch 전체 모듈 자동 선택. "
                             "OPT 예: fc1,fc2  Llama 예: mlp.gate_proj,mlp.up_proj,mlp.down_proj")

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

    # QR bits
    parser.add_argument("--q_bits", type=int, default=1)
    parser.add_argument("--r_bits", type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--r_group_size", type=int, default=128)
    parser.add_argument("--selective_base_threshold", type=float, default=1.0)
    parser.add_argument("--module_selective_base_thresholds", type=str, default="",
                        help="module별 threshold 덮어쓰기. 예: fc1:8,fc2:4")
    parser.add_argument("--selective_int_bits", type=int, default=4)
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0)

    # eval
    parser.add_argument("--eval_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "c4_new", "c4_omni"])
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_nsamples", type=int, default=256,
                        help="C4처럼 sampling 평가를 할 때 사용할 window 개수")
    parser.add_argument("--c4_cache_dir", type=str, default="/home/dataset/allenai_c4")
    # 다른 논문들과 동일한 기준(2048)으로 고정. Llama native seqlen(4096)과 무관하게 유지.
    parser.add_argument("--seqlen", type=int, default=2048)

    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.model_name
    ensure_dir(output_dir)

    print(f"[1] Loading model: {args.model_name} ...")
    model, tokenizer, model_seqlen, arch, load_path = load_model_and_tokenizer(
        args.model_name,
        model_source=args.model_source,
    )

    # target_modules: 비어있으면 arch 전체 모듈 자동 선택
    if not args.target_modules.strip():
        module_names = get_module_names(arch)
    else:
        module_names = parse_module_names(args.target_modules, arch)

    weight_scale_shrink_factors = parse_float_list(args.weight_scale_shrink_factors)
    module_selective_base_thresholds = parse_module_float_map(
        args.module_selective_base_thresholds, arch
    )

    cfg = get_model_config(args.model_name)
    # omni_ckpt에서 로드됐으면 "omni", hf_id(FP16 원본)에서 로드됐으면 "fp16"
    weight_backend = "omni" if load_path == cfg["omni_ckpt"] else "fp16"
    # enable_weight_quant까지 합산한 실제 weight 처리 방식
    if weight_backend == "omni" and not args.enable_weight_quant:
        effective_weight_mode = "omni_preweight"           # OmniQuant W4A16 그대로 사용
    elif weight_backend == "omni" and args.enable_weight_quant:
        effective_weight_mode = "fake_weight_from_omni_double_quant"  # 이미 양자화된 weight에 재양자화 (비권장)
    elif weight_backend == "fp16" and not args.enable_weight_quant:
        effective_weight_mode = "fp16_weight"              # FP16 원본 weight 그대로 사용
    else:
        effective_weight_mode = "fake_weight_from_fp16"    # FP16 weight에 fake quant 적용

    lines = []
    lines.append("[Config]")
    lines.append(f"model_name         : {args.model_name}")
    lines.append(f"model_source       : {args.model_source}")
    lines.append(f"arch               : {arch}")
    lines.append(f"load_path          : {load_path}")
    lines.append(f"weight_backend     : {weight_backend}")
    lines.append(f"effective_weight_mode: {effective_weight_mode}")
    lines.append(f"model_seqlen       : {model_seqlen}")
    lines.append(f"eval_dataset       : {args.eval_dataset}")
    lines.append(f"eval_split         : {args.eval_split}")
    if args.eval_dataset in ("c4", "c4_new", "c4_omni"):
        lines.append(f"eval_nsamples      : {args.eval_nsamples}")
        lines.append(f"c4_cache_dir       : {args.c4_cache_dir}")
    lines.append(f"eval_seqlen        : {args.seqlen}")
    lines.append(f"replace_scope      : {args.replace_scope}")
    lines.append(f"baseline_only      : {args.baseline_only}")
    lines.append(f"target_modules     : {module_names}")
    lines.append(f"enable_weight_quant: {args.enable_weight_quant}")
    lines.append(f"q_bits             : {args.q_bits}")
    lines.append(f"r_bits             : {args.r_bits}")
    lines.append(f"base_group_size    : {args.base_group_size}")
    lines.append(f"r_group_size       : {args.r_group_size}")
    lines.append(f"selective_base_threshold       : {args.selective_base_threshold}")
    lines.append(f"module_selective_base_thresholds: {module_selective_base_thresholds}")
    lines.append(f"selective_int_bits : {args.selective_int_bits}")
    lines.append(f"residual_clip_alpha: {args.residual_clip_alpha}")
    lines.append("")

    print(f"[2] Loading {args.eval_dataset} ...")
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

    print("[3] Computing baseline PPL ...")
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

    print("[4] Replacing modules with QuotRemLinear ...")
    layer_indices = resolve_target_layers(
        model, arch, args.replace_scope, args.one_layer_idx, args.custom_layer_indices
    )
    replaced_names = replace_modules_with_quotrem_linear(
        model=model,
        arch=arch,
        layer_indices=layer_indices,
        module_names=module_names,
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

    print("[5] Computing quantized PPL ...")
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
