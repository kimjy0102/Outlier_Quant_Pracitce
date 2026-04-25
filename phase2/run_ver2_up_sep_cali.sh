#!/bin/bash

# =================================================================
# 파라미터 설정
# =================================================================
Q_BITS=1                # Q(몫) 비트 수
R_BITS=4                # R(나머지) 비트 수
BASE_GROUP_SIZE=128      # base(나눌 값) 결정 group 크기 (-1이면 per-tensor)
R_GROUP_SIZE=128         # R 양자화 group 크기 (-1이면 per-tensor)
WEIGHT_BITS=4           # Weight 비트 수
WEIGHT_GROUP_SIZE=128    # Weight quant group 크기
REPLACE_SCOPE="all"
N_CALIB_SAMPLES=128     # 캘리브레이션 샘플 수
CALIB_SEQLEN=2048       # 캘리브레이션 시퀀스 길이
OUTPUT_DIR="results/results_ver2_up_sep_cali_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}"

echo "======================================================"
echo "Running Calibration-based QR Quantization"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "WEIGHT_BITS: ${WEIGHT_BITS}, WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "N_CALIB_SAMPLES: ${N_CALIB_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver2_up_sep_cali.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --n_calib_samples ${N_CALIB_SAMPLES} \
    --calib_seqlen ${CALIB_SEQLEN} \
    --output_dir ${OUTPUT_DIR}
