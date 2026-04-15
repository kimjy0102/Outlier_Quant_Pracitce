import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# User Config
# =========================================================
MODEL_ID = "facebook/opt-6.7b"

OUTPUT_ROOT = "opt67b_analysis_results"
RUN_NAME = "opt67b_qkv_input_once_qkv_output_analysis"

LAYER_INDICES = [0, 10, 20, 31]

# activation 분석용:
# q/k/v input은 중복이므로 q_proj input만 사용
# q/k/v output은 각각 따로 봄
ACTIVATION_TARGET_SPECS = [
    {"alias": "qkv_shared_input", "module_name": "self_attn.q_proj", "hook_type": "input"},
    {"alias": "q_proj_output",   "module_name": "self_attn.q_proj", "hook_type": "output"},
    {"alias": "k_proj_output",   "module_name": "self_attn.k_proj", "hook_type": "output"},
    {"alias": "v_proj_output",   "module_name": "self_attn.v_proj", "hook_type": "output"},
    {"alias": "out_proj_input",  "module_name": "self_attn.out_proj", "hook_type": "input"},
    {"alias": "fc1_input",       "module_name": "fc1", "hook_type": "input"},
    {"alias": "fc2_input",       "module_name": "fc2", "hook_type": "input"},
]

# weight 분석용
WEIGHT_TARGET_MODULE_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
]

NUM_CALIBRATION_SAMPLES = 64
MAX_SEQ_LEN = 128
USE_WIKITEXT2 = True

MAX_SAMPLE_VALUES_PER_MODULE = 50000
TOPK = 10
SEED = 42


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_txt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for line in lines:
                f.write(str(line) + "\n")


def plot_vector(vec, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(vec)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_histogram(values, title, xlabel, ylabel, save_path, bins=100, logy=False):
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, log=logy)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def format_topk(vec, topk=10, name="value"):
    k = min(topk, len(vec))
    idx = np.argsort(vec)[-k:][::-1]
    lines = [f"Top-{k} {name} indices and values:"]
    for rank, i in enumerate(idx):
        lines.append(f"{rank+1:2d}. idx={int(i):5d}, {name}={float(vec[i]):.8f}")
    return lines


# =========================================================
# Calibration Text Loader
# =========================================================
def get_calibration_texts(num_samples=64, use_wikitext2=True):
    if not use_wikitext2:
        raise RuntimeError("This script expects WikiText2 calibration data.")

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw_texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 20]
    texts = raw_texts[:num_samples]

    if len(texts) == 0:
        raise RuntimeError("WikiText2 loaded but no valid calibration texts were found.")

    print(f"[Info] Loaded {len(texts)} calibration samples from WikiText2.")
    return texts


# =========================================================
# Running Stats
# =========================================================
class RunningActivationStats:
    def __init__(self, max_sample_values=50000):
        self.total_elems = 0
        self.total_rows = 0
        self.sum_abs = 0.0
        self.sum_abs_sq = 0.0
        self.max_abs = 0.0

        self.channel_sum_abs = None
        self.channel_max_abs = None

        self.max_sample_values = max_sample_values
        self.sample_values = []

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.detach().float()

            if x.dim() == 1:
                flat = x.abs().unsqueeze(0)
            else:
                flat = x.abs().reshape(-1, x.shape[-1])

            n_rows, hidden_dim = flat.shape

            if self.channel_sum_abs is None:
                self.channel_sum_abs = torch.zeros(hidden_dim, dtype=torch.float64)
                self.channel_max_abs = torch.zeros(hidden_dim, dtype=torch.float32)

            self.total_elems += flat.numel()
            self.total_rows += n_rows
            self.sum_abs += flat.sum().item()
            self.sum_abs_sq += (flat * flat).sum().item()
            self.max_abs = max(self.max_abs, flat.max().item())

            self.channel_sum_abs += flat.sum(dim=0).cpu().double()
            self.channel_max_abs = torch.maximum(
                self.channel_max_abs,
                flat.amax(dim=0).cpu().float()
            )

            flat_1d = flat.reshape(-1)
            if len(self.sample_values) < self.max_sample_values:
                remaining = self.max_sample_values - len(self.sample_values)
                if flat_1d.numel() <= remaining:
                    sampled = flat_1d.cpu().numpy()
                else:
                    stride = max(1, flat_1d.numel() // remaining)
                    sampled = flat_1d[::stride][:remaining].cpu().numpy()
                self.sample_values.extend(sampled.tolist())

    def finalize(self):
        global_mean = self.sum_abs / max(self.total_elems, 1)
        global_var = self.sum_abs_sq / max(self.total_elems, 1) - global_mean ** 2
        global_var = max(global_var, 0.0)
        global_std = math.sqrt(global_var)

        channel_mean = (self.channel_sum_abs / max(self.total_rows, 1)).numpy()
        channel_max = self.channel_max_abs.numpy()

        sample_values = np.array(self.sample_values, dtype=np.float32) if len(self.sample_values) > 0 else np.array([], dtype=np.float32)

        result = {
            "global_mean_abs": float(global_mean),
            "global_std_abs": float(global_std),
            "global_max_abs": float(self.max_abs),
            "channel_mean_abs": channel_mean,
            "channel_max_abs": channel_max,
            "sample_values_abs": sample_values,
            "threshold_mean_plus_3std": float(global_mean + 3.0 * global_std),
        }

        if len(sample_values) > 0:
            result["approx_p99"] = float(np.quantile(sample_values, 0.99))
            result["approx_p999"] = float(np.quantile(sample_values, 0.999))
            result["approx_p9999"] = float(np.quantile(sample_values, 0.9999))
        else:
            result["approx_p99"] = None
            result["approx_p999"] = None
            result["approx_p9999"] = None

        return result


class RunningOutlierStats:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.total_elems = 0
        self.total_rows = 0
        self.outlier_count = 0
        self.channel_outlier_count = None

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            x = x.detach().float()

            if x.dim() == 1:
                flat = x.abs().unsqueeze(0)
            else:
                flat = x.abs().reshape(-1, x.shape[-1])

            n_rows, hidden_dim = flat.shape

            if self.channel_outlier_count is None:
                self.channel_outlier_count = torch.zeros(hidden_dim, dtype=torch.float64)

            mask = flat > self.threshold

            self.total_elems += flat.numel()
            self.total_rows += n_rows
            self.outlier_count += mask.sum().item()
            self.channel_outlier_count += mask.sum(dim=0).cpu().double()

    def finalize(self):
        global_ratio = self.outlier_count / max(self.total_elems, 1)
        channel_ratio = (self.channel_outlier_count / max(self.total_rows, 1)).numpy()

        return {
            "global_outlier_ratio": float(global_ratio),
            "channel_outlier_ratio": channel_ratio,
            "threshold": float(self.threshold),
        }


# =========================================================
# Model / Module Helpers
# =========================================================
def get_layer_module(layer, module_name: str):
    mapping = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.out_proj": layer.self_attn.out_proj,
        "fc1": layer.fc1,
        "fc2": layer.fc2,
    }
    return mapping[module_name]


def get_activation_targets(model, layer_indices, activation_target_specs):
    targets = {}
    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]
        for spec in activation_target_specs:
            alias = spec["alias"]
            module_name = spec["module_name"]
            hook_type = spec["hook_type"]
            full_key = f"layer{idx}.{alias}"
            targets[full_key] = {
                "module": get_layer_module(layer, module_name),
                "module_name": module_name,
                "hook_type": hook_type,
                "alias": alias,
                "layer_idx": idx,
            }
    return targets


def get_weight_targets(model, layer_indices, weight_target_module_names):
    targets = {}
    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]
        for module_name in weight_target_module_names:
            full_name = f"layer{idx}.{module_name}"
            targets[full_name] = get_layer_module(layer, module_name)
    return targets


def get_input_device(model):
    return model.model.decoder.embed_tokens.weight.device


# =========================================================
# Hook Builders
# =========================================================
def extract_hook_tensor(hook_type, inputs, output):
    if hook_type == "input":
        if len(inputs) == 0:
            return None
        return inputs[0]
    elif hook_type == "output":
        if isinstance(output, tuple):
            return output[0]
        return output
    else:
        raise ValueError(f"Unsupported hook_type: {hook_type}")


def make_collect_hook(stats_obj: RunningActivationStats, hook_type: str):
    def hook(module, inputs, output):
        x = extract_hook_tensor(hook_type, inputs, output)
        if x is None:
            return
        stats_obj.update(x)
    return hook


def make_outlier_hook(stats_obj: RunningOutlierStats, hook_type: str):
    def hook(module, inputs, output):
        x = extract_hook_tensor(hook_type, inputs, output)
        if x is None:
            return
        stats_obj.update(x)
    return hook


# =========================================================
# Weight Analysis
# =========================================================
def analyze_weight(module):
    with torch.no_grad():
        w = module.weight.detach().float().cpu()
        absw = w.abs()

        global_mean = absw.mean().item()
        global_std = absw.std().item()
        global_max = absw.max().item()
        threshold = global_mean + 3.0 * global_std
        global_outlier_ratio = (absw > threshold).float().mean().item()

        input_channel_mean = absw.mean(dim=0).numpy()
        input_channel_max = absw.amax(dim=0).numpy()
        output_channel_mean = absw.mean(dim=1).numpy()
        output_channel_max = absw.amax(dim=1).numpy()

        sample_values = absw.flatten().numpy()
        if sample_values.size > 50000:
            stride = max(1, sample_values.size // 50000)
            sample_values = sample_values[::stride][:50000]

        return {
            "global_mean_abs": float(global_mean),
            "global_std_abs": float(global_std),
            "global_max_abs": float(global_max),
            "threshold_mean_plus_3std": float(threshold),
            "global_outlier_ratio": float(global_outlier_ratio),
            "input_channel_mean_abs": input_channel_mean,
            "input_channel_max_abs": input_channel_max,
            "output_channel_mean_abs": output_channel_mean,
            "output_channel_max_abs": output_channel_max,
            "sample_values_abs": sample_values,
        }


# =========================================================
# Save Results
# =========================================================
def save_activation_results(target_name, hook_type, module_name, pass1_result, pass2_result, out_dir):
    safe_name = sanitize_name(target_name)
    module_dir = Path(out_dir) / "activation_targets" / safe_name
    ensure_dir(module_dir)

    channel_mean = pass1_result["channel_mean_abs"]
    channel_max = pass1_result["channel_max_abs"]
    sample_values = pass1_result["sample_values_abs"]
    channel_outlier_ratio = pass2_result["channel_outlier_ratio"]

    np.save(module_dir / "channel_mean_abs.npy", channel_mean)
    np.save(module_dir / "channel_max_abs.npy", channel_max)
    np.save(module_dir / "channel_outlier_ratio.npy", channel_outlier_ratio)
    np.save(module_dir / "sample_values_abs.npy", sample_values)

    plot_vector(
        channel_mean,
        title=f"{target_name} ({hook_type} of {module_name}): channel-wise mean |x|",
        xlabel="Channel",
        ylabel="Mean |activation|",
        save_path=module_dir / "channel_mean_abs.png",
    )

    plot_vector(
        channel_max,
        title=f"{target_name} ({hook_type} of {module_name}): channel-wise max |x|",
        xlabel="Channel",
        ylabel="Max |activation|",
        save_path=module_dir / "channel_max_abs.png",
    )

    plot_vector(
        channel_outlier_ratio,
        title=f"{target_name} ({hook_type} of {module_name}): channel-wise outlier ratio",
        xlabel="Channel",
        ylabel="Outlier ratio",
        save_path=module_dir / "channel_outlier_ratio.png",
    )

    if len(sample_values) > 0:
        plot_histogram(
            sample_values,
            title=f"{target_name} ({hook_type} of {module_name}): histogram of |x|",
            xlabel="|activation|",
            ylabel="Count",
            save_path=module_dir / "histogram_abs_linear.png",
            bins=100,
            logy=False,
        )

        plot_histogram(
            sample_values,
            title=f"{target_name} ({hook_type} of {module_name}): histogram of |x| (log-y)",
            xlabel="|activation|",
            ylabel="Count (log)",
            save_path=module_dir / "histogram_abs_logy.png",
            bins=100,
            logy=True,
        )

    summary_lines = [
        f"target_name: {target_name}",
        f"source_module: {module_name}",
        f"hook_type: {hook_type}",
        f"global_mean_abs: {pass1_result['global_mean_abs']:.8f}",
        f"global_std_abs: {pass1_result['global_std_abs']:.8f}",
        f"global_max_abs: {pass1_result['global_max_abs']:.8f}",
        f"threshold_mean_plus_3std: {pass1_result['threshold_mean_plus_3std']:.8f}",
        f"approx_p99: {pass1_result['approx_p99']}",
        f"approx_p999: {pass1_result['approx_p999']}",
        f"approx_p9999: {pass1_result['approx_p9999']}",
        f"global_outlier_ratio: {pass2_result['global_outlier_ratio']:.8f}",
        "",
    ]
    summary_lines += format_topk(channel_mean, TOPK, "channel_mean_abs")
    summary_lines += [""]
    summary_lines += format_topk(channel_max, TOPK, "channel_max_abs")

    save_txt(summary_lines, module_dir / "summary.txt")


def save_weight_results(module_name, weight_result, out_dir):
    safe_name = sanitize_name(module_name)
    module_dir = Path(out_dir) / "weight_targets" / safe_name
    ensure_dir(module_dir)

    in_mean = weight_result["input_channel_mean_abs"]
    in_max = weight_result["input_channel_max_abs"]
    out_mean = weight_result["output_channel_mean_abs"]
    out_max = weight_result["output_channel_max_abs"]
    sample_values = weight_result["sample_values_abs"]

    np.save(module_dir / "input_channel_mean_abs.npy", in_mean)
    np.save(module_dir / "input_channel_max_abs.npy", in_max)
    np.save(module_dir / "output_channel_mean_abs.npy", out_mean)
    np.save(module_dir / "output_channel_max_abs.npy", out_max)
    np.save(module_dir / "sample_values_abs.npy", sample_values)

    plot_vector(
        in_mean,
        title=f"{module_name} weight: input-channel mean |w|",
        xlabel="Input channel",
        ylabel="Mean |weight|",
        save_path=module_dir / "input_channel_mean_abs.png",
    )

    plot_vector(
        in_max,
        title=f"{module_name} weight: input-channel max |w|",
        xlabel="Input channel",
        ylabel="Max |weight|",
        save_path=module_dir / "input_channel_max_abs.png",
    )

    plot_vector(
        out_mean,
        title=f"{module_name} weight: output-channel mean |w|",
        xlabel="Output channel",
        ylabel="Mean |weight|",
        save_path=module_dir / "output_channel_mean_abs.png",
    )

    plot_vector(
        out_max,
        title=f"{module_name} weight: output-channel max |w|",
        xlabel="Output channel",
        ylabel="Max |weight|",
        save_path=module_dir / "output_channel_max_abs.png",
    )

    if len(sample_values) > 0:
        plot_histogram(
            sample_values,
            title=f"{module_name} weight: histogram of |w|",
            xlabel="|weight|",
            ylabel="Count",
            save_path=module_dir / "histogram_abs_linear.png",
            bins=100,
            logy=False,
        )

        plot_histogram(
            sample_values,
            title=f"{module_name} weight: histogram of |w| (log-y)",
            xlabel="|weight|",
            ylabel="Count (log)",
            save_path=module_dir / "histogram_abs_logy.png",
            bins=100,
            logy=True,
        )

    summary_lines = [
        f"module_name: {module_name}",
        f"global_mean_abs: {weight_result['global_mean_abs']:.8f}",
        f"global_std_abs: {weight_result['global_std_abs']:.8f}",
        f"global_max_abs: {weight_result['global_max_abs']:.8f}",
        f"threshold_mean_plus_3std: {weight_result['threshold_mean_plus_3std']:.8f}",
        f"global_outlier_ratio: {weight_result['global_outlier_ratio']:.8f}",
        "",
    ]
    summary_lines += format_topk(in_mean, TOPK, "input_channel_mean_abs")
    summary_lines += [""]
    summary_lines += format_topk(out_mean, TOPK, "output_channel_mean_abs")

    save_txt(summary_lines, module_dir / "summary.txt")


# =========================================================
# Forward Pass Helper
# =========================================================
def run_calibration_pass(model, tokenizer, texts, max_seq_len=128):
    input_device = get_input_device(model)

    for text_idx, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )

        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs, use_cache=False)

        if (text_idx + 1) % 10 == 0 or (text_idx + 1) == len(texts):
            print(f"[Info] Processed {text_idx + 1}/{len(texts)} calibration samples.")


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)

    run_dir = Path(OUTPUT_ROOT) / RUN_NAME
    ensure_dir(run_dir)
    ensure_dir(run_dir / "activation_targets")
    ensure_dir(run_dir / "weight_targets")
    ensure_dir(run_dir / "meta")

    print("[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    print("[2] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("[3] Collecting activation targets...")
    activation_targets = get_activation_targets(model, LAYER_INDICES, ACTIVATION_TARGET_SPECS)
    print(f"[Info] Number of activation targets: {len(activation_targets)}")

    print("[4] Collecting weight targets...")
    weight_targets = get_weight_targets(model, LAYER_INDICES, WEIGHT_TARGET_MODULE_NAMES)
    print(f"[Info] Number of weight targets: {len(weight_targets)}")

    print("[5] Preparing calibration texts...")
    calibration_texts = get_calibration_texts(
        num_samples=NUM_CALIBRATION_SAMPLES,
        use_wikitext2=USE_WIKITEXT2,
    )

    save_txt(
        [
            f"MODEL_ID={MODEL_ID}",
            f"LAYER_INDICES={LAYER_INDICES}",
            f"NUM_CALIBRATION_SAMPLES={NUM_CALIBRATION_SAMPLES}",
            f"MAX_SEQ_LEN={MAX_SEQ_LEN}",
            f"USE_WIKITEXT2={USE_WIKITEXT2}",
            "",
            "ACTIVATION_TARGET_SPECS:",
            *[str(x) for x in ACTIVATION_TARGET_SPECS],
            "",
            "WEIGHT_TARGET_MODULE_NAMES:",
            *[str(x) for x in WEIGHT_TARGET_MODULE_NAMES],
            "",
            "Calibration samples preview:",
            *[f"{i+1}. {t}" for i, t in enumerate(calibration_texts[:10])],
        ],
        run_dir / "meta" / "run_config.txt",
    )

    # -----------------------------------------------------
    # Weight analysis
    # -----------------------------------------------------
    print("[6] Analyzing weights...")
    for module_name, module in weight_targets.items():
        weight_result = analyze_weight(module)
        save_weight_results(module_name, weight_result, run_dir)

    # -----------------------------------------------------
    # Pass 1: activation stats collection
    # -----------------------------------------------------
    print("[7] Pass 1: collecting activation statistics...")
    pass1_stats = {name: RunningActivationStats(MAX_SAMPLE_VALUES_PER_MODULE) for name in activation_targets.keys()}
    hook_handles = []

    for target_name, info in activation_targets.items():
        h = info["module"].register_forward_hook(
            make_collect_hook(pass1_stats[target_name], info["hook_type"])
        )
        hook_handles.append(h)

    run_calibration_pass(model, tokenizer, calibration_texts, max_seq_len=MAX_SEQ_LEN)

    for h in hook_handles:
        h.remove()

    pass1_results = {}
    for target_name, stats_obj in pass1_stats.items():
        pass1_results[target_name] = stats_obj.finalize()

    # -----------------------------------------------------
    # Pass 2: activation outlier ratio collection
    # -----------------------------------------------------
    print("[8] Pass 2: collecting activation outlier ratios...")
    pass2_stats = {}
    hook_handles = []

    for target_name, info in activation_targets.items():
        threshold = pass1_results[target_name]["threshold_mean_plus_3std"]
        pass2_stats[target_name] = RunningOutlierStats(threshold)
        h = info["module"].register_forward_hook(
            make_outlier_hook(pass2_stats[target_name], info["hook_type"])
        )
        hook_handles.append(h)

    run_calibration_pass(model, tokenizer, calibration_texts, max_seq_len=MAX_SEQ_LEN)

    for h in hook_handles:
        h.remove()

    pass2_results = {}
    for target_name, stats_obj in pass2_stats.items():
        pass2_results[target_name] = stats_obj.finalize()

    # -----------------------------------------------------
    # Save activation results
    # -----------------------------------------------------
    print("[9] Saving activation analysis results...")
    for target_name, info in activation_targets.items():
        save_activation_results(
            target_name=target_name,
            hook_type=info["hook_type"],
            module_name=info["module_name"],
            pass1_result=pass1_results[target_name],
            pass2_result=pass2_results[target_name],
            out_dir=run_dir,
        )

    print(f"[Done] All results saved under: {run_dir}")


if __name__ == "__main__":
    main()