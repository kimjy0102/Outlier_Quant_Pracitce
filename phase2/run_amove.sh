#!/bin/bash

# =================================================================
# quant_amove.py 실행 스크립트 (Amove-Conservative reproduce)
#   - Weight  : 일반 fake_quant_symmetric (QuotRem 조건과 동일)
#   - Activation: Amove residual approximation (K=32, C=4, E=2, FP16 scale)
#   - 적용 범위: Linear (self_attn.q/k/v/out_proj, fc1, fc2), attention 경로 제외
#   - 평가 환경: quant_ver2_up_sep.compute_perplexity 동일 (wikitext2, seqlen=2048)
# =================================================================

# -------- Weight Quant (QuotRem 과 동일) --------
WEIGHT_BITS=4             # Weight 비트 수
WEIGHT_GROUP_SIZE=16      # Weight quant group 크기

# -------- Amove Activation Config (Amove-Conservative) --------
A_BITS=4                  # Activation 비트 수
A_GROUP_SIZE=256        # Coarse group size K
A_CLUSTER_SIZE=64          # Fine cluster size C (K 의 약수여야 함)
A_ENCODING_BITS=2         # Per-cluster encoding 비트 E

# -------- 공통 --------
REPLACE_SCOPE="all"
OUTPUT_DIR="results/results_amove_cons_K${A_GROUP_SIZE}_C${A_CLUSTER_SIZE}_E${A_ENCODING_BITS}_W${WEIGHT_BITS}g${WEIGHT_GROUP_SIZE}"

echo "======================================================"
echo "Running Amove-Conservative Reproduce"
echo "Weight     : W=${WEIGHT_BITS}, W_gs=${WEIGHT_GROUP_SIZE}"
echo "Activation : A=${A_BITS}, K=${A_GROUP_SIZE}, C=${A_CLUSTER_SIZE}, E=${A_ENCODING_BITS}"
echo "Scope      : ${REPLACE_SCOPE}"
echo "Output     : ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=7 python quant_amove.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --a_bits ${A_BITS} \
    --a_group_size ${A_GROUP_SIZE} \
    --a_cluster_size ${A_CLUSTER_SIZE} \
    --a_encoding_bits ${A_ENCODING_BITS} \
    --output_dir ${OUTPUT_DIR}
