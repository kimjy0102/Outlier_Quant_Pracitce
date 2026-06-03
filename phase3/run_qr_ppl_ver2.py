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
from replace_modules_ver2 import replace_modules_v2, get_default_qr_modules


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


def parse_qr_modules(s, arch, target_modules):
    """
    qr_modules CLI 파싱.
      - 비어있으면 arch별 데이터 기반 default 사용.
      - 'all'이면 target_modules 전체에 QR 적용.
      - 'none'이면 QR 없음 (전부 INT).
      - 그 외에는 쉼표로 분리해서 검증.
    """
    valid = set(get_module_names(arch))
    s_strip = s.strip().lower()

    if s_strip == "":
        return get_default_qr_modules(arch)
    if s_strip == "all":
        return list(target_modules)
    if s_strip == "none":
        return []

    names = [x.strip() for x in s.split(",") if x.strip()]
    for n in names:
        if n not in valid:
            raise ValueError(
                f"Invalid qr_module '{n}' for arch '{arch}'. Valid: {sorted(valid)}"
            )
    return names


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model_source", type=str, default="auto",
                        choices=["auto", "fp16", "omni"])
    parser.add_argument("--output_dir", type=str, default="results_ver2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_only", action="store_true")

    # 적용 범위
    parser.add_argument("--replace_scope", type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=0)
    parser.add_argument("--custom_layer_indices", type=str, default="")
    parser.add_argument("--target_modules", type=str, default="",
                        help="양자화 대상 전체 모듈. 비우면 arch 전체.")
    parser.add_argument("--qr_modules", type=str, default="",
                        help="QR 적용 모듈 (쉼표 구분). "
                             "비우면 arch별 default, 'all'이면 target 전체, 'none'이면 QR 없음. "
                             "target_modules 중 qr_modules에 없는 것은 IntActLinear로 처리.")

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

    # QR params (selective 관련 모두 제거됨)
    parser.add_argument("--q_bits", type=int, default=1)
    parser.add_argument("--r_bits", type=int, default=3)
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--r_group_size", type=int, default=128)
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0)

    # INT activation params (QR 미적용 모듈용)
    parser.add_argument("--int_act_bits", type=int, default=4)
    parser.add_argument("--int_act_group_size", type=int, default=128)

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
        args.model_name,
        model_source=args.model_source,
    )

    # target_modules와 qr_modules 결정
    if not args.target_modules.strip():
        target_modules = get_module_names(arch)
    else:
        target_modules = parse_module_names(args.target_modules, arch)

    qr_modules  = parse_qr_modules(args.qr_modules, arch, target_modules)
    int_modules = [m for m in target_modules if m not in set(qr_modules)]

    weight_scale_shrink_factors = parse_float_list(args.weight_scale_shrink_factors)

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
    lines.append(f"target_modules       : {target_modules}")
    lines.append(f"qr_modules           : {qr_modules}")
    lines.append(f"int_modules          : {int_modules}")
    lines.append(f"enable_weight_quant  : {args.enable_weight_quant}")
    lines.append(f"q_bits               : {args.q_bits}")
    lines.append(f"r_bits               : {args.r_bits}")
    lines.append(f"base_group_size      : {args.base_group_size}")
    lines.append(f"r_group_size         : {args.r_group_size}")
    lines.append(f"residual_clip_alpha  : {args.residual_clip_alpha}")
    lines.append(f"int_act_bits         : {args.int_act_bits}")
    lines.append(f"int_act_group_size   : {args.int_act_group_size}")
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

    print("[4] Replacing modules (QR for qr_modules, INT for others) ...")
    layer_indices = resolve_target_layers(
        model, arch, args.replace_scope, args.one_layer_idx, args.custom_layer_indices
    )
    replaced = replace_modules_v2(
        model=model,
        arch=arch,
        layer_indices=layer_indices,
        target_modules=target_modules,
        qr_modules=qr_modules,
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
        int_act_bits=args.int_act_bits,
        int_act_group_size=args.int_act_group_size,
    )

    n_qr  = sum(1 for _, k in replaced if k == "QR")
    n_int = sum(1 for _, k in replaced if k == "INT")
    lines.append("[Replacement]")
    lines.append(f"replaced_count : {len(replaced)} (QR={n_qr}, INT={n_int})")
    for name, kind in replaced:
        lines.append(f"  - [{kind}] {name}")
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
