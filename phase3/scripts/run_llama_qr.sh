#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=4

MODEL_NAME="llama-7b"
MODEL_SOURCE="${MODEL_SOURCE:-omni}"  # fp16 또는 omni
Q_BITS=1
R_BITS=3
BASE_GROUP_SIZE=128
R_GROUP_SIZE=128
SELECTIVE_BASE_THRESHOLD=4.0
MODULE_SELECTIVE_BASE_THRESHOLDS=""   # Llama는 fc2 없음. 필요 시 mlp.down_proj:64 등으로 지정
SELECTIVE_INT_BITS=4
RESIDUAL_CLIP_ALPHA=0
REPLACE_SCOPE="all"
# 비워두면 Llama arch 전체 모듈 자동 선택:
# self_attn.q_proj, k_proj, v_proj, o_proj, mlp.gate_proj, up_proj, down_proj
TARGET_MODULES=""

OUTPUT_DIR="${SCRIPT_DIR}/../results"
EVAL_DATASET="${EVAL_DATASET:-c4_omni}" # c4_omni, c4, c4_new, or wikitext2
EVAL_SPLIT="${EVAL_SPLIT:-validation}" # c4는 validation, wikitext2는 test
EVAL_NSAMPLES="${EVAL_NSAMPLES:-256}" # c4는 256, wikitext2는 2048 권장
C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../cache}" # c4_omni는 writable cache가 필요하다.
BASELINE_ONLY="${BASELINE_ONLY:-0}"  # 1이면 baseline만 실행, 0이면 QR/OmniQuant 실행. QR/OmniQuant 실행 시 baseline도 같이 실행한다.
BASELINE_ONLY_ARGS=()
if [ "${BASELINE_ONLY}" = "1" ]; then
    BASELINE_ONLY_ARGS=(--baseline_only)
fi

# model_configs.py의 omni_ckpt가 있으면 OmniQuant preweight를 우선 사용한다.
# --enable_weight_quant를 켜지 않으면 double quant 없이 QR activation만 적용한다.

echo "======================================================"
echo "Running Phase3 QR on ${MODEL_NAME}"
echo "MODEL_SOURCE: ${MODEL_SOURCE}, BASELINE_ONLY: ${BASELINE_ONLY}"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "SELECTIVE_BASE_THRESHOLD: ${SELECTIVE_BASE_THRESHOLD}"
echo "SELECTIVE_INT_BITS: ${SELECTIVE_INT_BITS}"
echo "RESIDUAL_CLIP_ALPHA: ${RESIDUAL_CLIP_ALPHA}"
echo "EVAL_DATASET: ${EVAL_DATASET}, EVAL_SPLIT: ${EVAL_SPLIT}, EVAL_NSAMPLES: ${EVAL_NSAMPLES}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/home2/juneyeop/anaconda3/envs/opt67/bin/python "${SCRIPT_DIR}/../run_qr_ppl.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --output_dir "${OUTPUT_DIR}" \
    --replace_scope "${REPLACE_SCOPE}" \
    --target_modules "${TARGET_MODULES}" \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --selective_base_threshold ${SELECTIVE_BASE_THRESHOLD} \
    --module_selective_base_thresholds "${MODULE_SELECTIVE_BASE_THRESHOLDS}" \
    --selective_int_bits ${SELECTIVE_INT_BITS} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --eval_nsamples ${EVAL_NSAMPLES} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --seqlen 2048 \
    "${BASELINE_ONLY_ARGS[@]}"
