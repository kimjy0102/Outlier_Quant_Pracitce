#!/bin/bash

SMOOTH_ALPHA=0.5
N_CALIB_SAMPLES=128
CALIB_SEQLEN=2048
WEIGHT_BITS=4          # 논문: INT8
ACT_BITS=4             # 논문: INT8
ACT_MODE="per_token"   # per_token=O1(논문 권장), per_tensor=O2
OUTPUT_DIR="results_smoothquant_alpha${SMOOTH_ALPHA}_w${WEIGHT_BITS}a${ACT_BITS}_${ACT_MODE}"

echo "======================================================"
echo "Running SmoothQuant PPL test"
echo "SMOOTH_ALPHA: ${SMOOTH_ALPHA}"
echo "WEIGHT_BITS: ${WEIGHT_BITS},  ACT_BITS: ${ACT_BITS},  ACT_MODE: ${ACT_MODE}"
echo "N_CALIB_SAMPLES: ${N_CALIB_SAMPLES}, CALIB_SEQLEN: ${CALIB_SEQLEN}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python smoothquant_ppl_test.py \
    --model_id "facebook/opt-6.7b" \
    --smooth_alpha ${SMOOTH_ALPHA} \
    --n_calib_samples ${N_CALIB_SAMPLES} \
    --calib_seqlen ${CALIB_SEQLEN} \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "tensor" \
    --enable_act_quant \
    --act_bits ${ACT_BITS} \
    --act_quant_mode ${ACT_MODE} \
    --output_dir ${OUTPUT_DIR}
