#!/bin/bash

ACT_GROUP_SIZE=16

WEIGHT_GROUP_SIZE=16
REORDER=true          # true: channel reordering 활성화 (outlier 8-bit 처리 포함)
N_CALIB_SAMPLES=128
CALIB_SEQLEN=2048
ACT_SORT_METRIC="abs_mean"
OUTPUT_DIR="results_oa_lama_gs${ACT_GROUP_SIZE}_reorder${REORDER}"

echo "======================================================"
echo "Running OA-LAMA style quantization"
echo "ACT_GROUP_SIZE: ${ACT_GROUP_SIZE}, REORDER: ${REORDER}"
echo "WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "N_CALIB_SAMPLES: ${N_CALIB_SAMPLES}, ACT_SORT_METRIC: ${ACT_SORT_METRIC}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

REORDER_FLAG=""
if [ "${REORDER}" = "true" ]; then
    REORDER_FLAG="--reorder"
fi

python quant_oa_lama.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope all \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --act_group_size ${ACT_GROUP_SIZE} \
    ${REORDER_FLAG} \
    --n_calib_samples ${N_CALIB_SAMPLES} \
    --calib_seqlen ${CALIB_SEQLEN} \
    --act_sort_metric ${ACT_SORT_METRIC} \
    --enable_weight_quant \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --output_dir ${OUTPUT_DIR}
