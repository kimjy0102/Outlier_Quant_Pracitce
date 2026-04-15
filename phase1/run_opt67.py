import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "facebook/opt-6.7b"

# -----------------------------
# 1. activation 저장용 dict
# -----------------------------
saved_acts = {}

# -----------------------------
# 2. hook 함수 정의
# -----------------------------
def save_input_hook(name):
    def hook(module, inputs, output):
        x = inputs[0].detach().float().cpu()
        saved_acts[name] = x
    return hook

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

# -----------------------------
# 3. 보고 싶은 layer 선택
# -----------------------------
layer_idx = 10
target_module = model.model.decoder.layers[layer_idx].self_attn.q_proj

# hook 등록
hook_handle = target_module.register_forward_hook(
    save_input_hook(f"layer{layer_idx}_q_proj_input")
)

prompt = "Explain what a transformer layer does in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")

# 입력은 임베딩이 있는 장치로 보냄
if torch.cuda.is_available():
    input_device = model.model.decoder.embed_tokens.weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

print("Generating...")
with torch.no_grad():
    outputs = model(**inputs, use_cache=False)


# hook 제거
hook_handle.remove()

# -----------------------------
# 4. 저장된 activation 꺼내기
# -----------------------------
x = saved_acts[f"layer{layer_idx}_q_proj_input"]
print("activation shape:", x.shape)

abs_x = x.abs()

print("global mean(|x|):", abs_x.mean().item())
print("global max(|x|):", abs_x.max().item())
print("global std(|x|):", abs_x.std().item())

if abs_x.dim() == 3:
    channel_mean = abs_x.mean(dim=(0, 1))
    channel_max = abs_x.amax(dim=(0, 1))
elif abs_x.dim() == 2:
    channel_mean = abs_x.mean(dim=0)
    channel_max = abs_x.amax(dim=0)
elif abs_x.dim() == 1:
    channel_mean = abs_x
    channel_max = abs_x
else:
    raise ValueError(f"Unexpected activation shape: {abs_x.shape}")

k = min(10, channel_mean.numel())

print("top-k channel mean values:", torch.topk(channel_mean, k).values)
print("top-k channel mean idx:", torch.topk(channel_mean, k).indices)

print("top-k channel max values:", torch.topk(channel_max, k).values)
print("top-k channel max idx:", torch.topk(channel_max, k).indices)

threshold = abs_x.mean() + 3 * abs_x.std()
outlier_mask = abs_x > threshold

global_outlier_ratio = outlier_mask.float().mean().item()
channel_outlier_ratio = outlier_mask.float().mean(dim=(0, 1))

print("threshold:", threshold.item())
print("global outlier ratio:", global_outlier_ratio)

os.makedirs("input_outlier_results", exist_ok=True)

np.save("input_outlier_results/channel_mean_q_proj.npy", channel_mean.numpy())
np.save("input_outlier_results/channel_max_q_proj.npy", channel_max.numpy())
np.save("input_outlier_results/channel_outlier_ratio_q_proj.npy", channel_outlier_ratio.numpy())

plt.figure(figsize=(10, 4))
plt.plot(channel_mean.numpy())
plt.xlabel("Channel")
plt.ylabel("Mean |activation|")
plt.title(f"OPT-6.7B Layer {layer_idx} q_proj input: channel-wise mean |x|")
plt.tight_layout()
plt.savefig("input_outlier_results/channel_mean_plot_q_proj.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(channel_max.numpy())
plt.xlabel("Channel")
plt.ylabel("Max |activation|")
plt.title(f"OPT-6.7B Layer {layer_idx} q_proj input: channel-wise max |x|")
plt.tight_layout()
plt.savefig("input_outlier_results/channel_max_plot_q_proj.png")
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(abs_x.flatten().numpy(), bins=100)
plt.xlabel("|activation|")
plt.ylabel("Count")
plt.title(f"OPT-6.7B Layer {layer_idx} q_proj input: histogram")
plt.tight_layout()
plt.savefig("input_outlier_results/histogram_abs_x_q_proj.png")
plt.close()

print("Saved results to input_outlier_results/")