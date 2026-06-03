#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES=4

# =================================================================
# 1. 파라미터 설정 (여기 숫자/문자만 바꿔가며 테스트하시면 됩니다!)
# =================================================================
MODEL_ID="${MODEL_ID:-huggyllama/llama-7b}"
ARCH="${ARCH:-auto}"               # auto / opt / llama (auto는 model_id에서 추론)
DATASET="${DATASET:-c4_omni}"    # wikitext2 / c4 / c4_new / c4_omni
EVAL_SPLIT="${EVAL_SPLIT:-validation}"   # wikitext2는 test, c4 계열은 validation 권장
EVAL_NSAMPLES="${EVAL_NSAMPLES:-256}"  # c4 계열에서만 의미 있음
EVAL_SEQLEN="${EVAL_SEQLEN:-2048}"
C4_CACHE_DIR="${C4_CACHE_DIR:-/home2/juneyeop/opt67/phase3/cache}"
TARGET_MODULES="${TARGET_MODULES:-}"  # 빈 값이면 quant_ppl_test.py가 arch별 전체 모듈 자동 사용

QUANT_TARGET="both"       # weight_only, act_only, both 중 선택
ACT_BITS=4             # Activation 비트 수 (보통 8)
WEIGHT_BITS=4          # Weight 비트 수 (보통 4 또는 8)
ACT_MODE="group"       # Activation 양자화 모드 (tensor, per_token, group 중 택 1)
WEIGHT_MODE="group"    # Weight 양자화 모드 (tensor, per_channel, group 중 택 1)
ACT_GROUP_SIZE=128     # Activation 그룹 사이즈
WEIGHT_GROUP_SIZE=128   # Weight 그룹 사이즈
DO_PROBE_COMPARE=false  # true면 module/logit 비교까지 저장

# 출력 폴더명에 dataset도 포함해서 wikitext2/c4 결과가 섞이지 않도록 분리.
OUTPUT_DIR="base_results_${QUANT_TARGET}_a${ACT_BITS}w${WEIGHT_BITS}_${ACT_MODE}_ag${ACT_GROUP_SIZE}_${WEIGHT_MODE}_wg${WEIGHT_GROUP_SIZE}_${DATASET}"

WEIGHT_FLAGS=()
ACT_FLAGS=()
PROBE_FLAGS=()
TARGET_MODULES_FLAG=()

if [ "${QUANT_TARGET}" = "weight_only" ] || [ "${QUANT_TARGET}" = "both" ]; then
    WEIGHT_FLAGS+=(--enable_weight_quant)
fi

if [ "${QUANT_TARGET}" = "act_only" ] || [ "${QUANT_TARGET}" = "both" ]; then
    ACT_FLAGS+=(--enable_act_quant)
fi

if [ "${DO_PROBE_COMPARE}" = "true" ]; then
    PROBE_FLAGS+=(--do_probe_compare)
fi

# TARGET_MODULES가 빈 값이면 --target_modules 인자 자체를 안 넘김 → quant_ppl_test.py가 arch별 전체 모듈 사용.
if [ -n "${TARGET_MODULES}" ]; then
    TARGET_MODULES_FLAG+=(--target_modules "${TARGET_MODULES}")
fi

if [ "${QUANT_TARGET}" != "weight_only" ] && [ "${QUANT_TARGET}" != "act_only" ] && [ "${QUANT_TARGET}" != "both" ]; then
    echo "Unsupported QUANT_TARGET=${QUANT_TARGET}. Use weight_only, act_only, or both."
    exit 1
fi

# =================================================================
# 2. 실행 명령어
# =================================================================
echo "======================================================"
echo "Running Baseline Quantization"
echo "MODEL_ID     : ${MODEL_ID}"
echo "ARCH         : ${ARCH}"
echo "DATASET      : ${DATASET} (${EVAL_SPLIT}, nsamples=${EVAL_NSAMPLES}, seqlen=${EVAL_SEQLEN})"
echo "QUANT_TARGET : ${QUANT_TARGET}"
echo "A${ACT_BITS} ${ACT_MODE} group=${ACT_GROUP_SIZE}"
echo "W${WEIGHT_BITS} ${WEIGHT_MODE} group=${WEIGHT_GROUP_SIZE}"
echo "TARGET_MODULES: ${TARGET_MODULES:-<arch default>}"
echo "Output       : ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python quant_ppl_test.py \
    --model_id "${MODEL_ID}" \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --eval_nsamples ${EVAL_NSAMPLES} \
    --eval_seqlen ${EVAL_SEQLEN} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --replace_scope "all" \
    "${TARGET_MODULES_FLAG[@]}" \
    "${WEIGHT_FLAGS[@]}" \
    --weight_bits ${WEIGHT_BITS} \
    --weight_quant_mode ${WEIGHT_MODE} \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    "${ACT_FLAGS[@]}" \
    --act_bits ${ACT_BITS} \
    --act_quant_mode ${ACT_MODE} \
    --act_group_size ${ACT_GROUP_SIZE} \
    --quant_impl "fake" \
    --output_dir "${OUTPUT_DIR}" \
    "${PROBE_FLAGS[@]}"
