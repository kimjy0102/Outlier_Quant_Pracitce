import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM

# =========================
# Experiment config
# =========================
MODEL_ID = "facebook/opt-6.7b"
DATASET_NAME = "wikitext2"

# Pruning toggles
ENABLE_WEIGHT_PRUNING = True
ENABLE_ACT_PRUNING = False

# lm_head 제외 여부
EXCLUDE_LM_HEAD = True

# threshold mode: "std" or "percentile"
PRUNE_MODE = "std"
PRUNE_VALUE_MODE = "clip" # zero or clip
# std mode: threshold = mean(|x|) + K * std(|x|)
PRUNE_STD_K = 10.0

# percentile mode: threshold = quantile(|x|, q)
# ex) 0.999 => 상위 0.1% 절대값을 outlier로 간주
PRUNE_PERCENTILE = 0.999


def load_wikitext2_testenc(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        legacy=False
    )
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    testenc = tokenizer(text, return_tensors="pt")
    return tokenizer, testenc.input_ids


def perplexity_eval(model, testenc, dataset_name, dev):
    print(f"\nEvaluating perplexity for {dataset_name} ...")

    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough tokens: tokens={testenc.numel()}, seqlen={model.seqlen}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    try:
        with torch.no_grad():
            for i in range(nsamples):
                batch = testenc[:, i * model.seqlen:(i + 1) * model.seqlen].to(dev)

                outputs = model(batch)
                lm_logits = outputs.logits

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)
    finally:
        model.config.use_cache = use_cache

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()
    print(f"{dataset_name} perplexity = {ppl:.4f}")
    return ppl


def _get_threshold(abs_x, mode="std", std_k=3.0, q=0.999):
    if mode == "std":
        return abs_x.mean() + std_k * abs_x.std()
    if mode == "percentile":
        return torch.quantile(abs_x.flatten(), q)
    raise ValueError(f"Unsupported PRUNE_MODE: {mode}")


def prune_weight_outliers_inplace(
    model,
    mode="std",
    std_k=3.0,
    percentile=0.999,
    exclude_lm_head=True,
):
    """
    Outlier pruning:
    |w| > threshold 인 weight를 0으로 설정
    """
    total_params = 0
    total_pruned = 0
    layer_stats = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if exclude_lm_head and "lm_head" in name:
                continue

            w = module.weight.data
            abs_w = w.abs().float()

            threshold = _get_threshold(
                abs_w,
                mode=mode,
                std_k=std_k,
                q=percentile,
            )

            mask = abs_w > threshold
            pruned_cnt = int(mask.sum().item())
            param_cnt = w.numel()

            if PRUNE_VALUE_MODE == "zero":
                w[mask] = 0.0
            elif PRUNE_VALUE_MODE == "clip":
                w.copy_(torch.clamp(w, min=-threshold, max=threshold))
            else:
                raise ValueError(f"Unsupported PRUNE_VALUE_MODE: {PRUNE_VALUE_MODE}")

            total_pruned += pruned_cnt
            total_params += param_cnt
            layer_stats.append((name, pruned_cnt, param_cnt, float(threshold.item())))

    global_ratio = 100.0 * total_pruned / max(total_params, 1)
    return global_ratio, layer_stats


def add_activation_prune_hooks(
    model,
    mode="std",
    std_k=3.0,
    percentile=0.999,
    exclude_lm_head=True,
):
    """
    Linear 입력 activation에 대해 outlier pruning hook 등록:
    |x| > threshold 인 activation을 0으로 설정
    """
    handles = []
    stats = {}  # name -> {"pruned": int, "total": int, "calls": int}

    def make_pre_hook(layer_name):
        def pre_hook(module, inputs):
            x = inputs[0]
            abs_x = x.detach().abs().float()

            threshold = _get_threshold(
                abs_x,
                mode=mode,
                std_k=std_k,
                q=percentile,
            )

            mask = abs_x > threshold
            pruned_cnt = int(mask.sum().item())
            total_cnt = mask.numel()

            if layer_name not in stats:
                stats[layer_name] = {"pruned": 0, "total": 0, "calls": 0}
            stats[layer_name]["pruned"] += pruned_cnt
            stats[layer_name]["total"] += total_cnt
            stats[layer_name]["calls"] += 1
            if PRUNE_VALUE_MODE == "zero":
                x_pruned = x.masked_fill(mask, 0.0)
            elif PRUNE_VALUE_MODE == "clip":
                x_pruned = torch.clamp(x, min=-threshold, max=threshold)
            else:
                raise ValueError(f"Unsupported PRUNE_VALUE_MODE: {PRUNE_VALUE_MODE}")
            return (x_pruned,) + tuple(inputs[1:])

        return pre_hook

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_lm_head and "lm_head" in name:
            continue
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))

    return handles, stats


def remove_hooks(handles):
    for h in handles:
        h.remove()


def print_pruning_report(global_ratio, layer_stats, topk=10, title="Weight pruning"):
    print(f"\n[{title}] Global pruned ratio: {global_ratio:.6f}%")

    ranked = sorted(
        layer_stats,
        key=lambda x: (x[1] / max(x[2], 1)),
        reverse=True
    )[:topk]

    print(f"[{title}] Top-{topk} layers by pruned ratio:")
    for name, pruned_cnt, param_cnt, th in ranked:
        ratio = 100.0 * pruned_cnt / max(param_cnt, 1)
        print(
            f"  {name}: pruned={pruned_cnt}/{param_cnt} "
            f"({ratio:.6f}%), threshold={th:.6e}"
        )


def print_act_pruning_report(stats, topk=10):
    if len(stats) == 0:
        print("\n[Activation pruning] No stats collected.")
        return

    rows = []
    total_pruned = 0
    total_elems = 0

    for name, d in stats.items():
        pruned = d["pruned"]
        total = d["total"]
        calls = d["calls"]
        ratio = 100.0 * pruned / max(total, 1)
        rows.append((name, pruned, total, ratio, calls))
        total_pruned += pruned
        total_elems += total

    global_ratio = 100.0 * total_pruned / max(total_elems, 1)
    print(f"\n[Activation pruning] Global pruned ratio: {global_ratio:.6f}%")
    print(f"[Activation pruning] Top-{topk} layers by pruned ratio:")

    rows = sorted(rows, key=lambda x: x[3], reverse=True)[:topk]
    for name, pruned, total, ratio, calls in rows:
        print(
            f"  {name}: pruned={pruned}/{total} ({ratio:.6f}%), calls={calls}"
        )


def main():
    tokenizer, testenc = load_wikitext2_testenc(MODEL_ID)

    model = OPTForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    model.seqlen = model.config.max_position_embeddings

    dev = model.model.decoder.embed_tokens.weight.device

    # 1) baseline
    ppl_base = perplexity_eval(model, testenc, f"{DATASET_NAME}-baseline", dev)

    # 2) apply pruning
    weight_global_ratio = None
    weight_layer_stats = None
    act_handles = []
    act_stats = {}

    if ENABLE_WEIGHT_PRUNING:
        print(
            f"\nApplying weight outlier pruning: mode={PRUNE_MODE}, "
            f"std_k={PRUNE_STD_K}, percentile={PRUNE_PERCENTILE}, "
            f"exclude_lm_head={EXCLUDE_LM_HEAD}"
        )
        weight_global_ratio, weight_layer_stats = prune_weight_outliers_inplace(
            model=model,
            mode=PRUNE_MODE,
            std_k=PRUNE_STD_K,
            percentile=PRUNE_PERCENTILE,
            exclude_lm_head=EXCLUDE_LM_HEAD,
        )
        print_pruning_report(
            weight_global_ratio,
            weight_layer_stats,
            topk=10,
            title="Weight pruning"
        )

    if ENABLE_ACT_PRUNING:
        print(
            f"\nApplying activation outlier pruning: mode={PRUNE_MODE}, "
            f"std_k={PRUNE_STD_K}, percentile={PRUNE_PERCENTILE}, "
            f"exclude_lm_head={EXCLUDE_LM_HEAD}"
        )
        act_handles, act_stats = add_activation_prune_hooks(
            model=model,
            mode=PRUNE_MODE,
            std_k=PRUNE_STD_K,
            percentile=PRUNE_PERCENTILE,
            exclude_lm_head=EXCLUDE_LM_HEAD,
        )

    # 3) pruned eval
    ppl_pruned = perplexity_eval(model, testenc, f"{DATASET_NAME}-pruned", dev)

    # activation hook cleanup
    remove_hooks(act_handles)

    if ENABLE_ACT_PRUNING:
        print_act_pruning_report(act_stats, topk=10)

    print("\n===== Summary =====")
    print(f"Baseline PPL : {ppl_base:.4f}")
    print(f"Pruned PPL   : {ppl_pruned:.4f}")
    print(f"Delta PPL    : {ppl_pruned - ppl_base:+.4f}")
    print(f"Weight pruning enabled    : {ENABLE_WEIGHT_PRUNING}")
    print(f"Activation pruning enabled: {ENABLE_ACT_PRUNING}")


if __name__ == "__main__":
    main()