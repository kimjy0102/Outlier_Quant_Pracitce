#!/bin/bash

# =================================================================
# 1. 파라미터 설정 (여기 숫자만 바꿔가며 테스트하시면 됩니다!)
# =================================================================
QR_BASE=4.0        # 추천값: 8.0, 9.0, 또는 21.0 등
Q_BITS=3             # Q(몫) 비트 수
R_BITS=5             # R(나머지) 비트 수
WEIGHT_BITS=4        # Weight 비트 수
GROUP_SIZE=64     # Weight 그룹 사이즈
#OUTPUT_DIR="quotrem_results_base${QR_BASE}_q${Q_BITS}_r${R_BITS}"
REPLACE_SCOPE="all"           # "all" 또는 "one"
QUANT_GROUP_SIZE=128           # 양자화 그룹 사이즈 (weight의 두배로 설정)
# =================================================================
# 2. 실행 명령어 (멀티라인으로 보기 쉽게 정리)
# =================================================================
echo "======================================================"
echo "Running QR Quantization with QR_BASE = ${QR_BASE}"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}, WEIGHT_BITS: ${WEIGHT_BITS}, WEIGHT_GROUP_SIZE: ${GROUP_SIZE}"
echo "QUANT_GROUP_SIZE: ${QUANT_GROUP_SIZE}, REPLACE_SCOPE: ${REPLACE_SCOPE}"
#echo "Output will be saved to: ${OUTPUT_DIR}"
echo "======================================================"
#self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2
# --custom_layer_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# --one_layer_idx 15
# --collect_q_stats
python quant_ppl_test_qr.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope ${REPLACE_SCOPE} \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --qr_base ${QR_BASE} \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode "group" \
    --weight_group_size ${GROUP_SIZE} \
    --q_bits ${Q_BITS} \
    --q_quant_mode "group" \
    --q_group_size ${QUANT_GROUP_SIZE} \
    --r_bits ${R_BITS} \
    --r_quant_mode "group" \
    --r_group_size ${QUANT_GROUP_SIZE} \
    --do_probe_compare
