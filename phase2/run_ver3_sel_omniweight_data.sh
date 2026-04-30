#!/bin/bash

# =================================================================
# quant_ver3_sel_omniweight_data.py 실행 스크립트
#   - OmniQuant W4A16g128 weight + Selective QR 모듈에서
#     R / Q / base / selective routing 분포 수집 및 분석
#   - run_ver3_sel_omniquant.sh 와 동일한 설정을 사용
# =================================================================

MODEL_ID="/home2/juneyeop/OmniQuant/checkpoint/opt-6.7b-w4a16g128"
Q_BITS=1
R_BITS=3
BASE_GROUP_SIZE=128
R_GROUP_SIZE=128
SELECTIVE_BASE_THRESHOLD=8.0
MODULE_SELECTIVE_BASE_THRESHOLDS="fc2:4"
SELECTIVE_INT_BITS=4
RESIDUAL_CLIP_ALPHA=0

N_BATCHES=10      # forward pass 횟수 (많을수록 분포가 더 정확, 시간 ↑)
N_BINS=100        # 히스토그램 bin 수
REPLACE_SCOPE="all"

OUTPUT_DIR="results/r_analysis_ver3_sel_omni_q${Q_BITS}_r${R_BITS}_basegs${BASE_GROUP_SIZE}_rgs${R_GROUP_SIZE}_thr${SELECTIVE_BASE_THRESHOLD}"

echo "======================================================"
echo "Running Data Analysis: ver3 Selective + OmniQuant"
echo "MODEL_ID   : ${MODEL_ID}"
echo "Q_BITS     : ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GS    : ${BASE_GROUP_SIZE}, R_GS: ${R_GROUP_SIZE}"
echo "SEL_THR    : ${SELECTIVE_BASE_THRESHOLD}"
echo "N_BATCHES  : ${N_BATCHES}"
echo "Output     : ${OUTPUT_DIR}"
echo "======================================================"

python quant_ver3_sel_omniweight_data.py \
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
    --n_batches ${N_BATCHES} \
    --n_bins ${N_BINS} \
    --output_dir ${OUTPUT_DIR}
