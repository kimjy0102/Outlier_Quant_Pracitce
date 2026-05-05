OMNI_CKPT_BASE = "/home2/juneyeop/OmniQuant/checkpoint"

# hf_id  : 토크나이저 로드 및 FP16 원본 로드용. Llama-1은 로컬 경로 직접 지정.
# omni_ckpt : OmniQuant W4A16g128 save_pretrained 체크포인트 경로 (직접 실행 후 생성됨).
# arch    : "opt" 또는 "llama" — model_adapter.py에서 분기 키로 사용.
MODEL_REGISTRY = {
    "opt-6.7b": {
        "hf_id":     "facebook/opt-6.7b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/opt-6.7b-w4a16g128",
        "arch":      "opt",
    },
    "opt-13b": {
        "hf_id":     "facebook/opt-13b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/opt-13b-w4a16g128",
        "arch":      "opt",
    },
    "opt-30b": {
        "hf_id":     "facebook/opt-30b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/opt-30b-w4a16g128",
        "arch":      "opt",
    },
    "opt-66b": {
        "hf_id":     "facebook/opt-66b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/opt-66b-w4a16g128",
        "arch":      "opt",
    },
    # Llama-1: OmniQuant 스크립트가 로컬 경로를 요구하므로 hf_id도 로컬 경로.
    # 모델 다운로드 후 아래 경로를 실제 위치로 수정할 것.
    "llama-7b": {
        "hf_id":     "huggyllama/llama-7b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/llama-7b-w4a16g128",
        "arch":      "llama",
    },
    "llama-13b": {
        "hf_id":     "huggyllama/llama-13b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/llama-13b-w4a16g128",
        "arch":      "llama",
    },
    "llama-30b": {
        "hf_id":     "huggyllama/llama-30b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/llama-30b-w4a16g128",
        "arch":      "llama",
    },
    "llama-65b": {
        "hf_id":     "huggyllama/llama-65b",
        "omni_ckpt": f"{OMNI_CKPT_BASE}/llama-65b-w4a16g128",
        "arch":      "llama",
    },
}

# arch별 양자화 대상 Linear 모듈 이름.
# OPT : fc1/fc2 (FFN), out_proj (attn output)
# Llama : gate_proj/up_proj/down_proj (SwiGLU FFN), o_proj (attn output)
ARCH_MODULE_NAMES = {
    "opt": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ],
    "llama": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
}


def get_model_config(model_name: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_module_names(arch: str) -> list:
    if arch not in ARCH_MODULE_NAMES:
        raise ValueError(f"Unknown arch: '{arch}'. Available: {list(ARCH_MODULE_NAMES.keys())}")
    return ARCH_MODULE_NAMES[arch]
