import random
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def load_wikitext2_testenc(model_path, split="test"):
    # OmniQuant datautils.py와 동일하게 use_fast=False 고정 (OPT/Llama 모두)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    full_text = "\n\n".join(ds["text"])
    testenc = tokenizer(full_text, return_tensors="pt").input_ids
    return tokenizer, testenc


def _find_c4_arrow_files(cache_dir, split):
    cache_dir = Path(cache_dir)
    c4_split = "validation" if split in ("test", "val", "validation") else split
    if c4_split not in ("train", "validation"):
        raise ValueError(f"C4 split은 train/validation만 지원합니다. got '{split}'")

    files = sorted(str(p) for p in cache_dir.rglob(f"c4-{c4_split}-*.arrow"))
    if not files:
        raise FileNotFoundError(
            f"C4 arrow 파일을 찾지 못했습니다: cache_dir={cache_dir}, split={c4_split}"
        )
    return files, c4_split


def load_c4_evalenc(
    model_path,
    split="validation",
    seqlen=2048,
    nsamples=256,
    seed=0,
    cache_dir="/home/dataset/allenai_c4",
):
    # /home/dataset/allenai_c4는 HuggingFace cache 형태라 load_from_disk가 아니라 arrow로 직접 읽는다.
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    files, c4_split = _find_c4_arrow_files(cache_dir, split)
    ds = load_dataset("arrow", data_files=files, split="train")

    rng = random.Random(seed)
    windows = []
    attempts = 0
    max_attempts = max(nsamples * 100, 1000)

    while len(windows) < nsamples and attempts < max_attempts:
        attempts += 1
        text = ds[rng.randrange(len(ds))]["text"]
        enc = tokenizer(text, return_tensors="pt").input_ids
        if enc.shape[1] < seqlen:
            continue

        start = rng.randint(0, enc.shape[1] - seqlen)
        windows.append(enc[:, start:(start + seqlen)])

    if len(windows) < nsamples:
        raise ValueError(
            f"C4 {c4_split}에서 {nsamples}개 window를 만들지 못했습니다. "
            f"collected={len(windows)}, seqlen={seqlen}, attempts={attempts}"
        )

    testenc = torch.hstack(windows)
    return tokenizer, testenc


def load_c4_new_evalenc(
    model_path,
    split="validation",
    seqlen=2048,
    nsamples=256,
    cache_dir="/home/dataset/allenai_c4",
):
    # OmniQuant datautils.py의 get_c4_new와 같이 앞쪽 validation text를 이어 붙인 뒤 앞에서 자른다.
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    files, c4_split = _find_c4_arrow_files(cache_dir, split)
    if c4_split != "validation":
        raise ValueError("c4_new 평가는 validation split만 지원합니다.")

    ds = load_dataset("arrow", data_files=files, split="train")
    n_texts = min(1100, len(ds))
    enc = tokenizer(" ".join(ds[:n_texts]["text"]), return_tensors="pt").input_ids
    ntokens = nsamples * seqlen
    if enc.shape[1] < ntokens:
        raise ValueError(
            f"C4-new token이 부족합니다: tokens={enc.shape[1]}, required={ntokens}"
        )
    testenc = enc[:, :ntokens]
    return tokenizer, testenc


def load_c4_omni_evalenc(
    model_path,
    split="validation",
    seqlen=2048,
    nsamples=256,
    cache_dir=None,
):
    # OmniQuant/GPTQ datautils.py의 get_c4 평가 shard를 그대로 사용한다.
    if split not in ("test", "val", "validation"):
        raise ValueError("c4_omni 평가는 validation split만 지원합니다.")
    if cache_dir is None:
        cache_dir = str(Path(__file__).resolve().parent / "cache")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        cache_dir=cache_dir,
    )

    rng = random.Random(0)
    windows = []
    while len(windows) < nsamples:
        while True:
            i = rng.randint(0, len(valdata) - 1)
            enc = tokenizer(valdata[i]["text"], return_tensors="pt").input_ids
            if enc.shape[1] >= seqlen:
                break
        start = rng.randint(0, enc.shape[1] - seqlen - 1)
        windows.append(enc[:, start:(start + seqlen)])

    testenc = torch.hstack(windows)
    return tokenizer, testenc


def load_eval_tokens(
    model_path,
    dataset_name="wikitext2",
    split="test",
    seqlen=2048,
    nsamples=256,
    seed=0,
    c4_cache_dir="/home/dataset/allenai_c4",
):
    if dataset_name == "wikitext2":
        return load_wikitext2_testenc(model_path, split=split)
    if dataset_name == "c4":
        return load_c4_evalenc(
            model_path,
            split=split,
            seqlen=seqlen,
            nsamples=nsamples,
            seed=seed,
            cache_dir=c4_cache_dir,
        )
    if dataset_name == "c4_new":
        return load_c4_new_evalenc(
            model_path,
            split=split,
            seqlen=seqlen,
            nsamples=nsamples,
            cache_dir=c4_cache_dir,
        )
    if dataset_name == "c4_omni":
        return load_c4_omni_evalenc(
            model_path,
            split=split,
            seqlen=seqlen,
            nsamples=nsamples,
            cache_dir=c4_cache_dir,
        )
    raise ValueError(f"지원하지 않는 eval dataset입니다: {dataset_name}")


def compute_perplexity(model, testenc, dev, seqlen):
    nsamples = testenc.numel() // seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough tokens: tokens={testenc.numel()}, seqlen={seqlen}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
            lm_logits = model(batch, use_cache=False).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache
    return ppl.item()
