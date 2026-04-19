#!/bin/bash

# =================================================================
# 파라미터 설정
# =================================================================
Q_BITS=2
R_BITS=4
R_GROUP_SIZE=16
WEIGHT_BITS=4
WEIGHT_GROUP_SIZE=16
REPLACE_SCOPE="all"
N_CALIB_SAMPLES=128
CALIB_SEQLEN=2048
OUTPUT_DIR="results_cali_q${Q_BITS}_r${R_BITS}_gs${R_GROUP_SIZE}"

echo "======================================================"
echo "Running Calibration-based Static QR Quantization"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "WEIGHT_BITS: ${WEIGHT_BITS}, WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "N_CALIB_SAMPLES: ${N_CALIB_SAMPLES}, CALIB_SEQLEN: ${CALIB_SEQLEN}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ppl_test_qr_ver2_cali.py \
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
    --n_calib_samples ${N_CALIB_SAMPLES} \
    --calib_seqlen ${CALIB_SEQLEN} \
    --output_dir ${OUTPUT_DIR}
