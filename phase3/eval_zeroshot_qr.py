import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from model_configs import MODEL_REGISTRY, get_model_config, get_module_names
from model_adapter import (
    load_model_and_tokenizer,
    parse_module_names,
    resolve_target_layers,
)
from replace_modules import replace_modules_with_quotrem_linear


# BlockDialect repo 안의 lm-evaluation-harness 사본은 evaluator.py에 BlockDialect 자체
# 양자화 hook(`from fake_quant import quantize_model`)이 추가되어 있어, 외부에서 그대로 import하면
# ModuleNotFoundError가 발생한다. phase3는 QR 모듈 교체로 양자화를 자체 처리한 뒤 HFLM에 넘기므로
# pip로 설치된 vanilla lm-eval 0.4.5를 쓰는 것이 맞다. 따라서 PYTHONPATH 우선 삽입은 default 비활성.
# 굳이 BlockDialect 사본을 강제하려면 --lm_eval_path를 명시.
BLOCKDIALECT_LM_EVAL_PATH = "/home2/juneyeop/blockdialect/lm-evaluation-harness"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


# task별 표준 metric 선택. OA-LAMA / OmniQuant / BlockDialect 모두 동일 관례 사용:
#   piqa, arc_easy, arc_challenge, hellaswag → acc_norm
#   boolq, winogrande                          → acc
# lm-eval 0.4.x는 result dict 키가 'acc,none' / 'acc_norm,none' 형식이다.
NORM_METRIC_TASKS = {"piqa", "arc_easy", "arc_challenge", "hellaswag"}


def pick_metric(results_dict, task_name):
    entry = results_dict.get(task_name, {})
    if task_name in NORM_METRIC_TASKS:
        # 0.4.x: 'acc_norm,none', 0.3.x fallback: 'acc_norm'
        for k in ("acc_norm,none", "acc_norm"):
            if k in entry:
                return entry[k], "acc_norm"
    for k in ("acc,none", "acc"):
        if k in entry:
            return entry[k], "acc"
    return None, None


def build_variant_tag(args, no_replace):
    if no_replace:
        return f"baseline_{args.model_source}"
    parts = [
        f"qr_q{args.q_bits}r{args.r_bits}",
        f"bg{args.base_group_size}_rg{args.r_group_size}",
        f"sel{args.selective_base_threshold:g}",
        f"sib{args.selective_int_bits}",
    ]
    if args.enable_weight_quant:
        parts.append(f"wq{args.weight_bits}g{args.weight_group_size}")
    if args.residual_clip_alpha > 0:
        parts.append(f"rca{args.residual_clip_alpha:g}")
    if args.module_selective_base_thresholds:
        parts.append("modsel")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()

    # 모델/소스 (run_qr_ppl.py와 동일 인터페이스)
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help=f"모델 이름. 선택지: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--model_source", type=str, default="auto",
                        choices=["auto", "fp16", "omni"])
    parser.add_argument("--output_dir", type=str, default="zeroshot_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_replace", action="store_true",
                        help="QR module 교체 없이 FP16/OmniQuant 원본 그대로 zero-shot 평가 (baseline)")

    # 적용 범위 (QR 교체 시)
    parser.add_argument("--replace_scope", type=str, default="all",
                        choices=["one", "all", "custom"])
    parser.add_argument("--one_layer_idx", type=int, default=0)
    parser.add_argument("--custom_layer_indices", type=str, default="")
    parser.add_argument("--target_modules", type=str, default="")

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
    parser.add_argument("--module_selective_base_thresholds", type=str, default="")
    parser.add_argument("--selective_int_bits", type=int, default=4)
    parser.add_argument("--residual_clip_alpha", type=float, default=0.0)

    # zero-shot 평가 인자
    parser.add_argument("--tasks", type=str,
                        default="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande",
                        help="쉼표로 구분된 task 목록")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="lm-eval batch size (auto 금지, 정수만)")
    parser.add_argument("--limit", type=int, default=None,
                        help="task당 sample 제한 (smoke test용). 비우면 전체")
    parser.add_argument("--lm_eval_path", type=str, default="",
                        help="lm-evaluation-harness 사본 경로 (PYTHONPATH 우선 삽입). "
                             "비우면 pip 설치된 vanilla lm-eval 사용 (권장). "
                             f"BlockDialect 사본 강제 시: {BLOCKDIALECT_LM_EVAL_PATH}")

    args = parser.parse_args()
    set_seed(args.seed)

    # lm-eval 0.4.5 (BlockDialect 사본) PYTHONPATH 우선 삽입
    if args.lm_eval_path and os.path.isdir(args.lm_eval_path):
        sys.path.insert(0, args.lm_eval_path)
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    output_dir = Path(args.output_dir) / args.model_name
    ensure_dir(output_dir)

    print(f"[1] Loading model: {args.model_name} (source={args.model_source}) ...")
    model, tokenizer, model_seqlen, arch, load_path = load_model_and_tokenizer(
        args.model_name,
        model_source=args.model_source,
    )

    if not args.target_modules.strip():
        module_names = get_module_names(arch)
    else:
        module_names = parse_module_names(args.target_modules, arch)

    weight_scale_shrink_factors = parse_float_list(args.weight_scale_shrink_factors)
    module_selective_base_thresholds = parse_module_float_map(
        args.module_selective_base_thresholds, arch
    )

    print(f"[2] no_replace={args.no_replace}")
    if not args.no_replace:
        print("[2a] Replacing modules with QuotRemLinear ...")
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
        print(f"    replaced {len(replaced_names)} modules")

    model.eval()

    # HFLM wrap. model_adapter가 device_map="auto"로 dispatched한 model을 그대로 넘긴다.
    # device 인자는 HFLM 내부 input placement용 — 첫 layer device로 명시.
    try:
        first_device = next(model.parameters()).device
    except StopIteration:
        first_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hflm_device = str(first_device)

    print(f"[3] Wrapping with HFLM (device={hflm_device}, batch_size={args.batch_size}) ...")
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=hflm_device,
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[4] Running simple_evaluate on tasks: {task_list}")
    print(f"    num_fewshot={args.num_fewshot}, limit={args.limit}")

    eval_results = simple_evaluate(
        model=hflm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    results_dict = eval_results["results"]

    # CSV 출력
    variant_tag = build_variant_tag(args, args.no_replace)
    csv_path = output_dir / f"zs_{variant_tag}.csv"

    rows = []
    metric_values = []
    print("\n[Result]")
    for task_name in task_list:
        score, metric_name = pick_metric(results_dict, task_name)
        if score is None:
            print(f"  {task_name:<16} : <missing>")
            rows.append({"task": task_name, "metric": "", "score": ""})
            continue
        score_pct = score * 100.0
        print(f"  {task_name:<16} : {score_pct:6.2f}  ({metric_name})")
        rows.append({"task": task_name, "metric": metric_name, "score": f"{score:.6f}"})
        metric_values.append(score)

    if metric_values:
        avg = sum(metric_values) / len(metric_values)
        print(f"  {'average':<16} : {avg*100:6.2f}")
        rows.append({"task": "average", "metric": "mixed", "score": f"{avg:.6f}"})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "metric", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[Done] CSV saved to {csv_path}")
    print(f"        variant={variant_tag}, model={args.model_name}, source={args.model_source}, "
          f"arch={arch}, load_path={load_path}")


if __name__ == "__main__":
    main()
