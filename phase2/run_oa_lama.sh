#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="${MODEL_NAME:-llama-13b}"
MODEL_ID="${MODEL_ID:-huggyllama/llama-13b}"
TARGET_MODULES="${TARGET_MODULES:-auto}"
EVAL_DATASET="${EVAL_DATASET:-c4_omni}"  # c4_omni, c4, c4_new, or wikitext2
EVAL_SPLIT="${EVAL_SPLIT:-validation}" # c4는 validation, wikitext2는 test
EVAL_NSAMPLES="${EVAL_NSAMPLES:-256}" # c4는 256, wikitext2는 2048 권장
C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../phase3/cache}" # c4_omni는 writable cache가 필요하다.

ACT_GROUP_SIZE=16

WEIGHT_GROUP_SIZE=16
REORDER=true          # true: channel reordering 활성화 (outlier 8-bit 처리 포함)
N_CALIB_SAMPLES=128
CALIB_SEQLEN=2048
ACT_SORT_METRIC="abs_mean"
OUTPUT_DIR="${OUTPUT_DIR:-results_oa_lama_gs${ACT_GROUP_SIZE}_reorder${REORDER}_${EVAL_DATASET}}"

echo "======================================================"
echo "Running OA-LAMA style quantization"
echo "MODEL_NAME: ${MODEL_NAME}, MODEL_ID: ${MODEL_ID}"
echo "TARGET_MODULES: ${TARGET_MODULES}"
echo "ACT_GROUP_SIZE: ${ACT_GROUP_SIZE}, REORDER: ${REORDER}"
echo "WEIGHT_GROUP_SIZE: ${WEIGHT_GROUP_SIZE}"
echo "N_CALIB_SAMPLES: ${N_CALIB_SAMPLES}, ACT_SORT_METRIC: ${ACT_SORT_METRIC}"
echo "EVAL_DATASET: ${EVAL_DATASET}, EVAL_SPLIT: ${EVAL_SPLIT}, EVAL_NSAMPLES: ${EVAL_NSAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================"

REORDER_FLAG=""
if [ "${REORDER}" = "true" ]; then
    REORDER_FLAG="--reorder"
fi

MODEL_ARGS=(--model_id "${MODEL_ID}")
if [ -n "${MODEL_NAME}" ]; then
    MODEL_ARGS=(--model_name "${MODEL_NAME}")
fi

python "${SCRIPT_DIR}/quant_oa_lama.py" \
    "${MODEL_ARGS[@]}" \
    --replace_scope all \
    --target_modules "${TARGET_MODULES}" \
    --act_group_size ${ACT_GROUP_SIZE} \
    ${REORDER_FLAG} \
    --n_calib_samples ${N_CALIB_SAMPLES} \
    --calib_seqlen ${CALIB_SEQLEN} \
    --act_sort_metric ${ACT_SORT_METRIC} \
    --enable_weight_quant \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --eval_nsamples ${EVAL_NSAMPLES} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --output_dir ${OUTPUT_DIR}
