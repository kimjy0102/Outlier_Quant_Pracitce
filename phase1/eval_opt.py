import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM


# =========================
# Experiment config (one run = one setting)
# =========================
MODEL_ID = "facebook/opt-6.7b"

# Baseline: both False
# W8A16: ENABLE_WEIGHT_QUANT=True, WEIGHT_BITS=8, ENABLE_ACT_QUANT=False
# W4A16: ENABLE_WEIGHT_QUANT=True, WEIGHT_BITS=4, ENABLE_ACT_QUANT=False
# W8A8 : ENABLE_WEIGHT_QUANT=True, WEIGHT_BITS=8, ENABLE_ACT_QUANT=True, ACT_BITS=8
ENABLE_WEIGHT_QUANT = True
ENABLE_ACT_QUANT = False

WEIGHT_BITS = 4
ACT_BITS = 8

# weight quant mode: "tensor" or "per_channel"
WEIGHT_QUANT_MODE = "per_channel"
WEIGHT_CH_AXIS = 0   # Linear weight [out_features, in_features] -> 보통 row-wise면 0

# activation quant mode: "tensor", "per_token", "per_channel"
ACT_QUANT_MODE = "tensor"
ACT_CH_AXIS = -1     # per_channel일 때 hidden dim 기준
# lm_head exclude
EXCLUDE_LM_HEAD = True

DATASET_NAME = "wikitext2"

def load_wikitext2_testenc(model_id):
    # 논문/레포 방식과 맞추기 위해 전체 test split을 하나로 이어붙임
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        legacy=False
    )

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return tokenizer, testenc.input_ids


def fake_quant_symmetric(x, n_bits=8, mode="tensor", ch_axis=0, eps=1e-8):
    qmax = (2 ** (n_bits - 1)) - 1
    qmin = -qmax - 1

    x_fp = x.float()

    if mode == "tensor":
        max_abs = x_fp.abs().max()
        scale = (max_abs / qmax).clamp_min(eps)

    elif mode == "per_channel":
        if ch_axis < 0:
            ch_axis = x_fp.dim() + ch_axis
        if ch_axis < 0 or ch_axis >= x_fp.dim():
            raise ValueError(
                f"Invalid ch_axis={ch_axis} for input dim={x_fp.dim()} in per_channel mode"
            )
        reduce_dims = tuple(d for d in range(x_fp.dim()) if d != ch_axis)
        max_abs = x_fp.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (max_abs / qmax).clamp_min(eps)

    elif mode == "per_token":
        # activation input이 [B,S,H] 또는 [B,H]일 때,
        # 마지막 dim(H)마다 token/vector별 scale 1개
        if x_fp.dim() == 1:
            max_abs = x_fp.abs().max()
            scale = (max_abs / qmax).clamp_min(eps)
        else:
            max_abs = x_fp.abs().amax(dim=-1, keepdim=True)
            scale = (max_abs / qmax).clamp_min(eps)

    else:
        raise ValueError(f"Unsupported quant mode: {mode}")

    q = torch.round(x_fp / scale).clamp(qmin, qmax)
    return (q * scale).to(x.dtype)


def apply_weight_quant_inplace(model, n_bits=8, mode="per_channel", ch_axis=0):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if EXCLUDE_LM_HEAD and "lm_head" in name:  # ← lm_head 체크 추가
                    print(f"  Skipping {name}")
                    continue

                w = module.weight.data
                w_q = fake_quant_symmetric(
                    w,
                    n_bits=n_bits,
                    mode=mode,
                    ch_axis=ch_axis,
                )
                module.weight.data.copy_(w_q)


def add_activation_quant_hooks(model, n_bits=8, mode="tensor", ch_axis=-1):
    handles = []

    def pre_hook(module, inputs):
        x = inputs[0]
        x_q = fake_quant_symmetric(
            x,
            n_bits=n_bits,
            mode=mode,
            ch_axis=ch_axis,
        )
        return (x_q,) + tuple(inputs[1:])

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if EXCLUDE_LM_HEAD and "lm_head" in name:
                print(f"  Skipping {name}")
                continue
                
            handles.append(module.register_forward_pre_hook(pre_hook))

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()

def perplexity_eval(model, testenc, dataset_name, dev):
    print(f"\nEvaluating perplexity for {dataset_name} dataset ...")

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

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{dataset_name} perplexity = {ppl.item():.4f}\n")

    model.config.use_cache = use_cache
    return ppl.item()

tokenizer, testenc = load_wikitext2_testenc(MODEL_ID)

model = OPTForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
model.seqlen = model.config.max_position_embeddings

if ENABLE_WEIGHT_QUANT:
    print(
        f"Applying weight quantization: {WEIGHT_BITS}-bit, "
        f"mode={WEIGHT_QUANT_MODE}, ch_axis={WEIGHT_CH_AXIS}"
    )
    apply_weight_quant_inplace(
    model,
    n_bits=WEIGHT_BITS,
    mode=WEIGHT_QUANT_MODE,
    ch_axis=WEIGHT_CH_AXIS,
)

act_handles = []

if ENABLE_ACT_QUANT:
    print(
        f"Applying activation quantization: {ACT_BITS}-bit, "
        f"mode={ACT_QUANT_MODE}, ch_axis={ACT_CH_AXIS}"
    )
    act_handles = add_activation_quant_hooks(
        model,
        n_bits=ACT_BITS,
        mode=ACT_QUANT_MODE,
        ch_axis=ACT_CH_AXIS,
    )

# 입력을 모델 첫 장치로 보냄
input_device = model.model.decoder.embed_tokens.weight.device

ppl = perplexity_eval(
    model=model,
    testenc=testenc,
    dataset_name=DATASET_NAME,
    dev=input_device,
)

remove_hooks(act_handles)

tag = "FP16"
if ENABLE_WEIGHT_QUANT and not ENABLE_ACT_QUANT:
    tag = f"W{WEIGHT_BITS}A16"
elif ENABLE_WEIGHT_QUANT and ENABLE_ACT_QUANT:
    tag = f"W{WEIGHT_BITS}A{ACT_BITS}"

print(f"{tag} PPL:", ppl)