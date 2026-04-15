### 기능: baseline ppl(FP16)이 잘 나오는 것 확인 및 Grouping하는 코드가 잘 되는지 검증 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# 0. Utility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_model_main_device(model):
    return model.model.decoder.embed_tokens.weight.device


def save_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for line in lines:
                f.write(str(line) + "\n")


# =========================================================
# 1. Dataset / PPL Eval Input Loader
#    [변경점 1]
#    - WikiText2 전체 split을 하나로 이어붙인 뒤 토큰화
# =========================================================
def load_wikitext2_testenc(model_id, split="test"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids

    return tokenizer, testenc


def get_probe_text():
    return "A transformer layer takes an input hidden state, projects it, mixes information, and returns a new hidden state."


# =========================================================
# 2. Exact Grouped Linear
# =========================================================
class GroupedLinear(nn.Module):
    """
    원래 nn.Linear와 수학적으로 같은 연산:
        y = xW + b = sum_g x_g W_g + b

    목적:
    - group 분할 방식 자체가 맞는지 확인
    - module 교체가 잘 되는지 확인
    - PPL이 baseline과 거의 같은지 확인
    """

    def __init__(self, base_linear: nn.Linear, group_size: int):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.group_size = group_size

        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by group_size ({self.group_size})"
            )

        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        """
        x shape: [..., in_features]
        weight shape: [out_features, in_features]
        """
        x_groups = torch.split(x, self.group_size, dim=-1)
        w_groups = torch.split(self.weight, self.group_size, dim=1)

        out = None
        for xg, wg in zip(x_groups, w_groups):
            part = F.linear(xg, wg, bias=None)
            out = part if out is None else out + part

        if self.bias is not None:
            out = out + self.bias

        return out


# =========================================================
# 3. Module Access / Replacement
# =========================================================
def get_named_linear_module(layer, module_name: str):
    mapping = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1": layer.fc1,
        "fc2": layer.fc2,
    }
    if module_name not in mapping:
        raise ValueError(f"Unsupported module_name: {module_name}")
    return mapping[module_name]


def set_named_linear_module(layer, module_name: str, new_module: nn.Module):
    if module_name == "self_attn.q_proj":
        layer.self_attn.q_proj = new_module
    elif module_name == "self_attn.k_proj":
        layer.self_attn.k_proj = new_module
    elif module_name == "self_attn.v_proj":
        layer.self_attn.v_proj = new_module
    elif module_name == "self_attn.out_proj":
        layer.self_attn.out_proj = new_module
    elif module_name == "fc1":
        layer.fc1 = new_module
    elif module_name == "fc2":
        layer.fc2 = new_module
    else:
        raise ValueError(f"Unsupported module_name: {module_name}")


def parse_module_names(s: str):
    names = [x.strip() for x in s.split(",") if len(x.strip()) > 0]
    valid = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    }
    for n in names:
        if n not in valid:
            raise ValueError(f"Invalid module name: {n}")
    return names


def resolve_target_layers(model, replace_scope: str, one_layer_idx: int):
    num_layers = len(model.model.decoder.layers)

    if replace_scope == "one":
        if one_layer_idx < 0 or one_layer_idx >= num_layers:
            raise ValueError(f"one_layer_idx must be in [0, {num_layers-1}]")
        return [one_layer_idx]
    elif replace_scope == "all":
        return list(range(num_layers))
    else:
        raise ValueError(f"Unsupported replace_scope: {replace_scope}")


def replace_modules_with_grouped_linear(model, layer_indices, module_names, group_size):
    """
    지정된 layer / module을 GroupedLinear로 교체
    """
    replaced_names = []

    for layer_idx in layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear_module(layer, module_name)
            new_module = GroupedLinear(old_module, group_size=group_size)
            set_named_linear_module(layer, module_name, new_module)
            replaced_names.append(f"layer{layer_idx}.{module_name}")

    return replaced_names


# =========================================================
# 4. Probe 비교용 Hook
# =========================================================
def collect_module_outputs(model, tokenizer, text, target_layer_indices, target_module_names):
    """
    교체 전/후에 같은 입력을 넣었을 때
    선택한 모듈들의 output을 저장해서 비교하기 위한 함수
    """
    outputs_dict = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            outputs_dict[name] = x.detach().float().cpu()
        return hook

    for layer_idx in target_layer_indices:
        layer = model.model.decoder.layers[layer_idx]
        for module_name in target_module_names:
            module = get_named_linear_module(layer, module_name)
            name = f"layer{layer_idx}.{module_name}"
            h = module.register_forward_hook(make_hook(name))
            hook_handles.append(h)

    device = get_model_main_device(model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits.detach().float().cpu()

    for h in hook_handles:
        h.remove()

    return outputs_dict, logits


def compare_tensor_dicts(ref_dict, new_dict):
    """
    모듈별 output 차이 계산
    """
    lines = []
    common_keys = sorted(set(ref_dict.keys()) & set(new_dict.keys()))

    for k in common_keys:
        a = ref_dict[k]
        b = new_dict[k]

        diff = a - b
        mse = (diff ** 2).mean().item()
        max_abs = diff.abs().max().item()
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

        lines.append(f"{k}")
        lines.append(f"  mse      : {mse:.12e}")
        lines.append(f"  max_abs  : {max_abs:.12e}")
        lines.append(f"  cosine   : {cos:.12f}")
        lines.append("")

    return lines


def compare_logits(ref_logits, new_logits):
    diff = ref_logits - new_logits
    mse = (diff ** 2).mean().item()
    max_abs = diff.abs().max().item()
    cos = F.cosine_similarity(ref_logits.flatten(), new_logits.flatten(), dim=0).item()

    lines = [
        "[Probe Logits Comparison]",
        f"mse      : {mse:.12e}",
        f"max_abs  : {max_abs:.12e}",
        f"cosine   : {cos:.12f}",
        "",
    ]
    return lines


# =========================================================
# 5. Perplexity
#    [변경점 2, 3]
#    - model.seqlen 기준 non-overlap chunk 평가
#    - model.config.use_cache를 평가 중 명시적으로 False
# =========================================================
def compute_perplexity(model, testenc, dev):
    print("\nEvaluating perplexity ...")

    nsamples = testenc.numel() // model.seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough tokens for one eval chunk: tokens={testenc.numel()}, seqlen={model.seqlen}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)

            outputs = model(batch, use_cache=False)
            lm_logits = outputs.logits

            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache
    return ppl.item()


# =========================================================
# 6. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="facebook/opt-6.7b")
    parser.add_argument("--output_dir", type=str, default="grouped_ppl_results")
    parser.add_argument("--seed", type=int, default=42)

    # grouped linear 관련
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--replace_scope", type=str, default="one", choices=["one", "all"])
    parser.add_argument("--one_layer_idx", type=int, default=10)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="fc1",
        help="Comma-separated module names, e.g. fc1 or self_attn.q_proj,fc1",
    )

    # PPL 관련
    parser.add_argument("--eval_split", type=str, default="test")

    # probe 비교용
    parser.add_argument("--do_probe_compare", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    module_names = parse_module_names(args.target_modules)

    lines = []
    lines.append("[Config]")
    lines.append(f"model_id        : {args.model_id}")
    lines.append(f"group_size      : {args.group_size}")
    lines.append(f"replace_scope   : {args.replace_scope}")
    lines.append(f"one_layer_idx   : {args.one_layer_idx}")
    lines.append(f"target_modules  : {module_names}")
    lines.append(f"eval_split      : {args.eval_split}")
    lines.append("")

    print("[1] Loading WikiText2 + tokenizer...")
    tokenizer, testenc = load_wikitext2_testenc(args.model_id, split=args.eval_split)

    print("[2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # 논문/레포 스타일 평가를 위해 seqlen 명시
    model.seqlen = model.config.max_position_embeddings

    # probe 비교용 text
    probe_text = get_probe_text()

    # -----------------------------------------------------
    # Baseline probe
    # -----------------------------------------------------
    if args.do_probe_compare:
        print("[3] Collecting baseline probe outputs/logits...")
        target_layers_for_probe = resolve_target_layers(model, args.replace_scope, args.one_layer_idx)
        ref_module_outputs, ref_logits = collect_module_outputs(
            model=model,
            tokenizer=tokenizer,
            text=probe_text,
            target_layer_indices=target_layers_for_probe,
            target_module_names=module_names,
        )
    else:
        ref_module_outputs, ref_logits = None, None

    # -----------------------------------------------------
    # Baseline PPL
    # -----------------------------------------------------
    print("[4] Computing baseline PPL...")
    input_device = get_model_main_device(model)
    baseline_ppl = compute_perplexity(
        model=model,
        testenc=testenc,
        dev=input_device,
    )

    lines.append("[Baseline]")
    lines.append(f"baseline_ppl : {baseline_ppl:.8f}")
    lines.append("")

    # -----------------------------------------------------
    # Replace modules
    # -----------------------------------------------------
    print("[5] Replacing modules with GroupedLinear...")
    target_layer_indices = resolve_target_layers(model, args.replace_scope, args.one_layer_idx)
    replaced_names = replace_modules_with_grouped_linear(
        model=model,
        layer_indices=target_layer_indices,
        module_names=module_names,
        group_size=args.group_size,
    )

    lines.append("[Replacement]")
    lines.append(f"replaced_count : {len(replaced_names)}")
    for name in replaced_names:
        lines.append(f"  - {name}")
    lines.append("")

    # -----------------------------------------------------
    # Modified probe
    # -----------------------------------------------------
    if args.do_probe_compare:
        print("[6] Collecting modified probe outputs/logits...")
        new_module_outputs, new_logits = collect_module_outputs(
            model=model,
            tokenizer=tokenizer,
            text=probe_text,
            target_layer_indices=target_layer_indices,
            target_module_names=module_names,
        )

        lines.extend(compare_logits(ref_logits, new_logits))
        lines.append("[Module Output Comparison]")
        lines.extend(compare_tensor_dicts(ref_module_outputs, new_module_outputs))

    # -----------------------------------------------------
    # Modified PPL
    # -----------------------------------------------------
    print("[7] Computing modified PPL...")
    modified_ppl = compute_perplexity(
        model=model,
        testenc=testenc,
        dev=input_device,
    )

    ppl_diff = modified_ppl - baseline_ppl
    rel_diff = ppl_diff / max(abs(baseline_ppl), 1e-12)

    lines.append("[Modified]")
    lines.append(f"modified_ppl : {modified_ppl:.8f}")
    lines.append(f"ppl_diff     : {ppl_diff:.12e}")
    lines.append(f"relative_diff: {rel_diff:.12e}")
    lines.append("")

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------
    print("\n".join(lines))
    save_txt(lines, Path(args.output_dir) / "summary.txt")
    print(f"[Done] Summary saved to {Path(args.output_dir) / 'summary.txt'}")


if __name__ == "__main__":
    main()