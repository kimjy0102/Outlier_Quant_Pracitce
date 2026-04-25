#!/bin/bash

# =================================================================
# 파라미터 설정
# =================================================================
Q_BITS=1                # Q(몫) 비트 수
R_BITS=3                # R(나머지) 비트 수
BASE_GROUP_SIZE=128      # base 결정 group 크기 (-1 이면 per-token)
R_GROUP_SIZE=128         # R 양자화 group 크기 (-1 이면 per-token)
WEIGHT_BITS=4           # Weight 비트 수
WEIGHT_GROUP_SIZE=128    # Weight quant group 크기
REPLACE_SCOPE="all"
N_BATCHES=10            # R 수집용 forward pass 횟수
N_BINS=100              # 히스토그램 bin 수
OUTPUT_DIR="results/r_analysis_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}"

echo "======================================================"
echo "Running R Distribution Analysis"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "N_BATCHES: ${N_BATCHES}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver2_up_sep_data.py \
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
    --n_batches ${N_BATCHES} \
    --n_bins ${N_BINS} \
    --output_dir ${OUTPUT_DIR}
