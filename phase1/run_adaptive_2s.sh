#!/bin/bash

# =================================================================
# 파라미터 설정
# =================================================================
Q_BITS=1             # Q(몫) 비트 수
R_BITS=3             # R(나머지) 비트 수
R_GROUP_SIZE=16     # R 양자화 그룹 사이즈 (base 결정 단위와 동일)
WEIGHT_BITS=4        # Weight 비트 수
WEIGHT_GROUP_SIZE=128
REPLACE_SCOPE="all"
OUTPUT_DIR="results_adaptive_q${Q_BITS}_r${R_BITS}_gs${R_GROUP_SIZE}"

echo "======================================================"
echo "Running Adaptive QR Quantization"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "WEIGHT_BITS: ${WEIGHT_BITS}, WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ppl_test_qr_ver2.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --r_group_size ${R_GROUP_SIZE} \
    --output_dir ${OUTPUT_DIR}
