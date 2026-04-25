#!/bin/bash

# =================================================================
# 파라미터 설정
# =================================================================
Q_BITS=1                    # Q(몫) 비트 수
R_BITS=3                    # R(나머지) 비트 수
BASE_GROUP_SIZE=128         # base(나눌 값) 결정 group 크기 (-1 이면 per-tensor)
R_GROUP_SIZE=128            # R 양자화 group 크기 (-1 이면 per-tensor) — base와 독립
SELECTIVE_BASE_THRESHOLD=8.0 # base가 이 값보다 작으면 QR 대신 INT fallback
MODULE_SELECTIVE_BASE_THRESHOLDS="" # 예: "fc1:8,fc2:4,self_attn.v_proj:8" (비우면 전역 threshold 사용)
SELECTIVE_INT_BITS=4        # fallback group에서 사용할 activation INT bit
RESIDUAL_CLIP_ALPHA=0.75     # 0이면 기존 R scale, >0이면 alpha*base로 R scale cap
WEIGHT_BITS=4               # Weight 비트 수
WEIGHT_GROUP_SIZE=128        # Weight quant group 크기
WEIGHT_SCALE_METHOD="mse"   # max 또는 mse
WEIGHT_SCALE_SHRINK_FACTORS="1.0,0.95,0.9,0.85,0.8"
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
OUTPUT_DIR="results/results_ver3_selective_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}_thr${SELECTIVE_BASE_THRESHOLD}${MODULE_THRESHOLD_TAG}_int${SELECTIVE_INT_BITS}${RESIDUAL_CLIP_TAG}_wgs${WEIGHT_GROUP_SIZE}_${WEIGHT_SCALE_METHOD}"

echo "======================================================"
echo "Running Selective QR Quantization"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "SELECTIVE_BASE_THRESHOLD: ${SELECTIVE_BASE_THRESHOLD}"
echo "MODULE_SELECTIVE_BASE_THRESHOLDS: ${MODULE_SELECTIVE_BASE_THRESHOLDS}"
echo "SELECTIVE_INT_BITS: ${SELECTIVE_INT_BITS}"
echo "RESIDUAL_CLIP_ALPHA: ${RESIDUAL_CLIP_ALPHA}"
echo "WEIGHT_BITS: ${WEIGHT_BITS}, WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "WEIGHT_SCALE_METHOD: ${WEIGHT_SCALE_METHOD}"
echo "WEIGHT_SCALE_SHRINK_FACTORS: ${WEIGHT_SCALE_SHRINK_FACTORS}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver3_selective.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --weight_scale_method ${WEIGHT_SCALE_METHOD} \
    --weight_scale_shrink_factors ${WEIGHT_SCALE_SHRINK_FACTORS} \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --selective_base_threshold ${SELECTIVE_BASE_THRESHOLD} \
    --module_selective_base_thresholds "${MODULE_SELECTIVE_BASE_THRESHOLDS}" \
    --selective_int_bits ${SELECTIVE_INT_BITS} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --output_dir ${OUTPUT_DIR}
