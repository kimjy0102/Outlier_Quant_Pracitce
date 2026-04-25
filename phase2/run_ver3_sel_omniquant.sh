#!/bin/bash

# =================================================================
# OmniQuant W4A16g128 weight + Selective Activation QR 설정
# =================================================================
MODEL_ID="/home2/juneyeop/OmniQuant/checkpoint/opt-6.7b-w4a16g128"
Q_BITS=1
R_BITS=3
BASE_GROUP_SIZE=128
R_GROUP_SIZE=128
SELECTIVE_BASE_THRESHOLD=8.0
MODULE_SELECTIVE_BASE_THRESHOLDS="" # 예: "fc1:8,fc2:4,self_attn.v_proj:8" (비우면 전역 threshold 사용)
SELECTIVE_INT_BITS=4
RESIDUAL_CLIP_ALPHA=0.5     # 0이면 기존 R scale, >0이면 alpha*base로 R scale cap
REPLACE_SCOPE="all"

MODULE_THRESHOLD_TAG=""
if [ -n "${MODULE_SELECTIVE_BASE_THRESHOLDS}" ]; then
    MODULE_THRESHOLD_TAG="_mthr${MODULE_SELECTIVE_BASE_THRESHOLDS//,/_}"
    MODULE_THRESHOLD_TAG="${MODULE_THRESHOLD_TAG//:/}"
fi

RESIDUAL_CLIP_TAG=""
if [ "${RESIDUAL_CLIP_ALPHA}" != "0.0" ]; then
    RESIDUAL_CLIP_TAG="_rclip${RESIDUAL_CLIP_ALPHA}"
fi

OUTPUT_DIR="results/results_ver3_sel_omniweight_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}_thr${SELECTIVE_BASE_THRESHOLD}${MODULE_THRESHOLD_TAG}_int${SELECTIVE_INT_BITS}${RESIDUAL_CLIP_TAG}"

echo "======================================================"
echo "Running Selective QR with OmniQuant W4A16g128 Weight"
echo "MODEL_ID: ${MODEL_ID}"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "SELECTIVE_BASE_THRESHOLD: ${SELECTIVE_BASE_THRESHOLD}"
echo "MODULE_SELECTIVE_BASE_THRESHOLDS: ${MODULE_SELECTIVE_BASE_THRESHOLDS}"
echo "SELECTIVE_INT_BITS: ${SELECTIVE_INT_BITS}"
echo "RESIDUAL_CLIP_ALPHA: ${RESIDUAL_CLIP_ALPHA}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver3_sel_omniweight.py \
    --model_id "${MODEL_ID}" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --selective_base_threshold ${SELECTIVE_BASE_THRESHOLD} \
    --module_selective_base_thresholds "${MODULE_SELECTIVE_BASE_THRESHOLDS}" \
    --selective_int_bits ${SELECTIVE_INT_BITS} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --output_dir ${OUTPUT_DIR}
