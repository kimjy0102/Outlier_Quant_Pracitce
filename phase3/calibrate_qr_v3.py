import argparse
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from model_configs import MODEL_REGISTRY, get_module_names
from model_adapter import (
    load_model_and_tokenizer,
    get_decoder_layers,
    get_named_linear,
    get_model_device,
    parse_module_names,
)


# =========================================================
# Calibration token loader (wikitext2 train / c4 train)
#
# Evaluation 쪽 (eval_utils.py) 과 분리해서 둔다:
#   - eval은 wikitext/c4의 test/validation split을 쓰는데,
#     calibration은 같은 도메인의 train split을 쓰는 게 표준이라 의미가 다르다.
#   - 또 calibration 결과는 한 번 만들어 두고 여러 threshold 실험에서 재사용하므로
#     평가 코드와 직접 결합시키지 않는다.
# =========================================================
def load_wikitext2_train_tokens(model_path, nsamples, seqlen, seed):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt").input_ids

    rng = random.Random(seed)
    windows = []
    upper = enc.shape[1] - seqlen - 1
    if upper <= 0:
        raise ValueError(f"wikitext2 train tokens={enc.shape[1]} < seqlen+1={seqlen+1}")
    for _ in range(nsamples):
        i = rng.randint(0, upper)
        windows.append(enc[:, i:(i + seqlen)])
    return torch.hstack(windows)


def load_c4_train_tokens(model_path, nsamples, seqlen, seed, cache_dir):
    # OmniQuant get_c4 train loader 와 동일한 shard 사용 (재현성 일치 목적).
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        cache_dir=cache_dir,
    )

    rng = random.Random(seed)
    windows = []
    attempts = 0
    max_attempts = max(nsamples * 200, 2000)
    while len(windows) < nsamples and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, len(traindata) - 1)
        enc = tokenizer(traindata[idx]["text"], return_tensors="pt").input_ids
        if enc.shape[1] < seqlen:
            continue
        start = rng.randint(0, enc.shape[1] - seqlen - 1)
        windows.append(enc[:, start:(start + seqlen)])
    if len(windows) < nsamples:
        raise ValueError(
            f"c4 train에서 {nsamples} window를 못 만듦. collected={len(windows)}"
        )
    return torch.hstack(windows)


def load_calib_tokens(calib_dataset, model_path, nsamples, seqlen, seed, c4_cache_dir):
    if calib_dataset == "wikitext2":
        return load_wikitext2_train_tokens(model_path, nsamples, seqlen, seed)
    if calib_dataset == "c4":
        return load_c4_train_tokens(model_path, nsamples, seqlen, seed, c4_cache_dir)
    raise ValueError(f"Unsupported calib_dataset: {calib_dataset}")


# =========================================================
# Permutation calculation
#
# max_abs 기준으로 descending sort 후 round-robin 분배:
#   group g, position p 에 sorted_indices[p * G + g] 채널을 배치.
#   reshape (group_size, G).T.reshape(-1) 와 동일.
# 결과: 가장 큰 channel은 group 0의 position 0, 두 번째는 group 1의 position 0, ...
#       → 각 group이 동일 magnitude tier로부터 한 채널씩 받음 (outlier 분산).
# =========================================================
def compute_round_robin_perm(max_abs_per_channel: torch.Tensor, group_size: int) -> torch.Tensor:
    in_features = max_abs_per_channel.numel()
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )
    G = in_features // group_size
    sorted_indices = torch.argsort(max_abs_per_channel, descending=True)
    perm = sorted_indices.reshape(group_size, G).T.contiguous().reshape(-1)
    return perm


def compute_per_group_max_abs(max_abs_per_channel: torch.Tensor, perm: torch.Tensor, group_size: int) -> torch.Tensor:
    in_features = max_abs_per_channel.numel()
    G = in_features // group_size
    perm_grouped = perm.reshape(G, group_size)
    return max_abs_per_channel[perm_grouped].amax(dim=-1)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model_source", type=str, default="auto",
                        choices=["auto", "fp16", "omni"])
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4"])
    parser.add_argument("--calib_nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c4_cache_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "cache"))
    parser.add_argument("--base_group_size", type=int, default=128)
    parser.add_argument("--target_modules", type=str, default="",
                        help="비우면 arch 전체 모듈")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f"[1] Loading model: {args.model_name} ...")
    model, tokenizer, _, arch, load_path = load_model_and_tokenizer(
        args.model_name, model_source=args.model_source
    )

    if not args.target_modules.strip():
        target_modules = get_module_names(arch)
    else:
        target_modules = parse_module_names(args.target_modules, arch)

    print(f"[2] Loading calibration data: {args.calib_dataset} ({args.calib_nsamples} × {args.seqlen}) ...")
    calib_tokens = load_calib_tokens(
        args.calib_dataset, load_path,
        args.calib_nsamples, args.seqlen, args.seed,
        args.c4_cache_dir,
    )

    print(f"[3] Registering hooks on {len(target_modules)} modules × {len(get_decoder_layers(model, arch))} layers ...")
    layers = get_decoder_layers(model, arch)
    accumulators = {}    # (layer_idx, module_name) → CPU tensor [in_features]
    hooks = []

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        for module_name in target_modules:
            module = get_named_linear(layer, module_name)
            in_features = module.in_features
            key = (layer_idx, module_name)
            accumulators[key] = torch.zeros(in_features, dtype=torch.float32)

            def make_hook(k):
                def pre_hook(mod, inputs):
                    x = inputs[0]
                    if x is None:
                        return
                    # max over all dims except last (channel)
                    cur = x.detach().float().abs()
                    if cur.dim() > 1:
                        cur = cur.amax(dim=tuple(range(cur.dim() - 1)))
                    cur = cur.cpu()
                    accumulators[k] = torch.maximum(accumulators[k], cur)
                return pre_hook

            hooks.append(module.register_forward_pre_hook(make_hook(key)))

    print(f"[4] Forwarding {args.calib_nsamples} calibration samples ...")
    dev = get_model_device(model, arch)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            for i in tqdm(range(args.calib_nsamples)):
                batch = calib_tokens[:, i * args.seqlen:(i + 1) * args.seqlen].to(dev)
                _ = model(batch, use_cache=False)
    finally:
        model.config.use_cache = use_cache
        for h in hooks:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[5] Computing permutations and per-group statistics ...")
    artifact = {
        "model_name":      args.model_name,
        "arch":            arch,
        "load_path":       load_path,
        "base_group_size": args.base_group_size,
        "calib_dataset":   args.calib_dataset,
        "calib_nsamples":  args.calib_nsamples,
        "seqlen":          args.seqlen,
        "seed":            args.seed,
        "target_modules":  list(target_modules),
        "modules":         {},
    }

    for key, max_abs in accumulators.items():
        perm = compute_round_robin_perm(max_abs, args.base_group_size)
        per_group_max_abs = compute_per_group_max_abs(max_abs, perm, args.base_group_size)
        artifact["modules"][key] = {
            "perm":              perm.to(torch.int32).cpu(),
            "per_group_max_abs": per_group_max_abs.cpu(),
            "channel_max_abs":   max_abs.cpu(),
        }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_path)
    print(f"[Done] Saved calibration artifact: {args.output_path}")
    print(f"       modules: {len(artifact['modules'])}")


if __name__ == "__main__":
    main()
