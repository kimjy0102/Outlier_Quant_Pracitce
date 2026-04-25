#!/bin/bash

# =================================================================
# quant_ver3_kv.py 실행 스크립트
#   - Linear QR (ver2_up_sep 재사용) + KV cache QR 를 모두 켤 수 있음
#   - Linear / KV 각 플래그로 On/Off 독립 제어
# =================================================================

# -------- Linear QR (activation + weight) --------
Q_BITS=1                  # Q(몫) 비트 수
R_BITS=3                  # R(나머지) 비트 수
BASE_GROUP_SIZE=128      # base 결정 group 크기 (-1 이면 per-tensor)
R_GROUP_SIZE=16           # R 양자화 group 크기 (-1 이면 per-tensor)
WEIGHT_BITS=4             # Weight 비트 수
WEIGHT_GROUP_SIZE=16      # Weight quant group 크기

# -------- KV cache QR --------
KV_Q_BITS=1               # KV Q 비트 수
KV_R_BITS=3               # KV R 비트 수
KV_BASE_GROUP_SIZE=16     # head_dim(=128)의 약수여야 함 (16/32/64/128)
KV_R_GROUP_SIZE=16
KV_TARGET="both"          # k / v / both

# -------- 공통 --------
REPLACE_SCOPE="all"
OUTPUT_DIR="results/results_ver3_kv_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}_kv${KV_TARGET}_kvq${KV_Q_BITS}_kvr${KV_R_BITS}_kvbgs${KV_BASE_GROUP_SIZE}_kvrgs${KV_R_GROUP_SIZE}"

echo "======================================================"
echo "Running Adaptive QR (Linear + KV cache)"
echo "Linear  : Q=${Q_BITS}, R=${R_BITS}, base_gs=${BASE_GROUP_SIZE}, r_gs=${R_GROUP_SIZE}"
echo "Weight  : W=${WEIGHT_BITS}, W_gs=${WEIGHT_GROUP_SIZE}"
echo "KV      : target=${KV_TARGET}, Q=${KV_Q_BITS}, R=${KV_R_BITS}, base_gs=${KV_BASE_GROUP_SIZE}, r_gs=${KV_R_GROUP_SIZE}"
echo "Scope   : ${REPLACE_SCOPE}"
echo "Output  : ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver3_kv.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --enable_linear_quant \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --enable_kv_quant \
    --kv_q_bits ${KV_Q_BITS} \
    --kv_r_bits ${KV_R_BITS} \
    --kv_base_group_size ${KV_BASE_GROUP_SIZE} \
    --kv_r_group_size ${KV_R_GROUP_SIZE} \
    --kv_target ${KV_TARGET} \
    --output_dir ${OUTPUT_DIR}
    
