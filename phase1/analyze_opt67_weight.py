import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# =========================
# 설정
# =========================
model_id = "facebook/opt-6.7b"

layer_idx = 10
module_name = "fc2"   # 선택지: q_proj, k_proj, v_proj, out_proj, fc1, fc2
save_dir = f"weight_outlier_results/layer{layer_idx}_{module_name}"

os.makedirs(save_dir, exist_ok=True)

# =========================
# 모델 로드
# =========================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

# =========================
# 모듈 선택
# =========================
layer = model.model.decoder.layers[layer_idx]

if module_name == "q_proj":
    target_module = layer.self_attn.q_proj
elif module_name == "k_proj":
    target_module = layer.self_attn.k_proj
elif module_name == "v_proj":
    target_module = layer.self_attn.v_proj
elif module_name == "out_proj":
    target_module = layer.self_attn.out_proj
elif module_name == "fc1":
    target_module = layer.fc1
elif module_name == "fc2":
    target_module = layer.fc2
else:
    raise ValueError(f"Unsupported module_name: {module_name}")

# =========================
# weight 가져오기
# Linear weight shape: [out_features, in_features]
# =========================
W = target_module.weight.detach().float().cpu()
abs_W = W.abs()

print(f"Selected module: layer {layer_idx} / {module_name}")
print("weight shape:", W.shape)

# =========================
# 전체 통계
# =========================
global_mean = abs_W.mean().item()
global_std = abs_W.std().item()
global_max = abs_W.max().item()

print("global mean(|W|):", global_mean)
print("global std(|W|):", global_std)
print("global max(|W|):", global_max)

# threshold 기준
threshold = abs_W.mean() + 3 * abs_W.std()
outlier_mask = abs_W > threshold
global_outlier_ratio = outlier_mask.float().mean().item() * 100.0
normal_ratio = 100.0 - global_outlier_ratio

print("threshold:", threshold.item())
print("global outlier ratio:", global_outlier_ratio)
print(f"outlier ratio = {global_outlier_ratio:.6f}%")
print(f"normal ratio  = {normal_ratio:.6f}%")
# =========================
# row-wise / col-wise 통계
# =========================
# row = output channel 방향
row_mean = abs_W.mean(dim=1)
row_max = abs_W.amax(dim=1)
row_outlier_ratio = outlier_mask.float().mean(dim=1)

# col = input channel 방향
col_mean = abs_W.mean(dim=0)
col_max = abs_W.amax(dim=0)
col_outlier_ratio = outlier_mask.float().mean(dim=0)

print(f"row outlier ratio mean = {row_outlier_ratio.mean().item() * 100.0:.6f}%")
print(f"row outlier ratio max  = {row_outlier_ratio.max().item() * 100.0:.6f}%")

print(f"col outlier ratio mean = {col_outlier_ratio.mean().item() * 100.0:.6f}%")
print(f"col outlier ratio max  = {col_outlier_ratio.max().item() * 100.0:.6f}%")

k = min(10, row_mean.numel(), col_mean.numel())

print("\n[top-k row mean]")
print(torch.topk(row_mean, k).values)
print(torch.topk(row_mean, k).indices)

print("\n[top-k row max]")
print(torch.topk(row_max, k).values)
print(torch.topk(row_max, k).indices)

print("\n[top-k col mean]")
print(torch.topk(col_mean, k).values)
print(torch.topk(col_mean, k).indices)

print("\n[top-k col max]")
print(torch.topk(col_max, k).values)
print(torch.topk(col_max, k).indices)

# =========================
# 저장
# =========================
#np.save(os.path.join(save_dir, "weight.npy"), W.numpy())
#np.save(os.path.join(save_dir, "row_mean.npy"), row_mean.numpy())
#np.save(os.path.join(save_dir, "row_max.npy"), row_max.numpy())
#np.save(os.path.join(save_dir, "row_outlier_ratio.npy"), row_outlier_ratio.numpy())
#np.save(os.path.join(save_dir, "col_mean.npy"), col_mean.numpy())
#np.save(os.path.join(save_dir, "col_max.npy"), col_max.numpy())
#np.save(os.path.join(save_dir, "col_outlier_ratio.npy"), col_outlier_ratio.numpy())

# =========================
# plot 1: 전체 |W| histogram
# =========================
plt.figure(figsize=(8, 4))
plt.hist(abs_W.flatten().numpy(), bins=100)
plt.xlabel("|weight|")
plt.ylabel("Count")
plt.title(f"OPT-6.7B Layer {layer_idx} {module_name}: histogram of |W|")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "histogram_abs_weight.png"))
plt.close()

# =========================
# plot 2: row-wise mean
# =========================
plt.figure(figsize=(10, 4))
plt.plot(row_mean.numpy())
plt.xlabel("Row (output channel)")
plt.ylabel("Mean |W|")
plt.title(f"OPT-6.7B Layer {layer_idx} {module_name}: row-wise mean |W|")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "row_mean_plot.png"))
plt.close()

# =========================
# plot 3: row-wise max
# =========================
plt.figure(figsize=(10, 4))
plt.plot(row_max.numpy())
plt.xlabel("Row (output channel)")
plt.ylabel("Max |W|")
plt.title(f"OPT-6.7B Layer {layer_idx} {module_name}: row-wise max |W|")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "row_max_plot.png"))
plt.close()

# =========================
# plot 4: col-wise mean
# =========================
plt.figure(figsize=(10, 4))
plt.plot(col_mean.numpy())
plt.xlabel("Column (input channel)")
plt.ylabel("Mean |W|")
plt.title(f"OPT-6.7B Layer {layer_idx} {module_name}: col-wise mean |W|")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "col_mean_plot.png"))
plt.close()

# =========================
# plot 5: col-wise max
# =========================
plt.figure(figsize=(10, 4))
plt.plot(col_max.numpy())
plt.xlabel("Column (input channel)")
plt.ylabel("Max |W|")
plt.title(f"OPT-6.7B Layer {layer_idx} {module_name}: col-wise max |W|")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "col_max_plot.png"))
plt.close()

print(f"\nSaved all results to: {save_dir}")