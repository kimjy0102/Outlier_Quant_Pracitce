#!/bin/bash

# =================================================================
# 1. 파라미터 설정 (여기 숫자/문자만 바꿔가며 테스트하시면 됩니다!)
# =================================================================
ACT_BITS=7             # Activation 비트 수 (보통 8)
WEIGHT_BITS=4          # Weight 비트 수 (보통 4 또는 8)
ACT_MODE="group"       # Activation 양자화 모드 (tensor, per_token, group 중 택 1)
WEIGHT_MODE="group"    # Weight 양자화 모드 (tensor, per_channel, group 중 택 1)
ACT_GROUP_SIZE=128     # Activation 그룹 사이즈
WEIGHT_GROUP_SIZE=128   # Weight 그룹 사이즈

# 출력 폴더명을 설정값에 따라 자동으로 생성되게 세팅
#OUTPUT_DIR="base_results_a${ACT_BITS}w${WEIGHT_BITS}_${ACT_MODE}_g${GROUP_SIZE}"

# =================================================================
# 2. 실행 명령어
# =================================================================
echo "======================================================"
echo "Running Baseline Quantization (A${ACT_BITS}W${WEIGHT_BITS} ${ACT_MODE})"
echo "Group Size: ${ACT_GROUP_SIZE}"
#echo "Output will be saved to: ${OUTPUT_DIR}"
echo "======================================================"

python quant_ppl_test.py \
    --model_id "facebook/opt-6.7b" \
    --replace_scope "all" \
    --target_modules "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,fc1,fc2" \
    --enable_weight_quant \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode ${WEIGHT_MODE} \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --enable_act_quant \
    --act_bits ${ACT_BITS} \
    --act_quant_mode ${ACT_MODE} \
    --act_group_size ${ACT_GROUP_SIZE} \
    --quant_impl "fake" \
    --do_probe_compare