#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=7

MODEL_NAME="${MODEL_NAME:-opt-6.7b}"
MODEL_SOURCE="${MODEL_SOURCE:-omni}"

# 표준 calibration: 128 samples × 2048 seqlen (OmniQuant/GPTQ 관례)
CALIB_DATASET="${CALIB_DATASET:-wikitext2}"   # wikitext2 또는 c4
CALIB_NSAMPLES="${CALIB_NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-42}"
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-128}"

C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../cache}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/../calib_artifacts}"
OUTPUT_PATH="${OUTPUT_PATH:-${ARTIFACT_DIR}/${MODEL_NAME}_${CALIB_DATASET}_n${CALIB_NSAMPLES}_s${SEQLEN}_gs${BASE_GROUP_SIZE}.pt}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "======================================================"
echo "Calibration for ver3"
echo "MODEL_NAME       : ${MODEL_NAME}"
echo "MODEL_SOURCE     : ${MODEL_SOURCE}"
echo "CALIB_DATASET    : ${CALIB_DATASET}"
echo "CALIB_NSAMPLES   : ${CALIB_NSAMPLES}"
echo "SEQLEN           : ${SEQLEN}"
echo "BASE_GROUP_SIZE  : ${BASE_GROUP_SIZE}"
echo "OUTPUT_PATH      : ${OUTPUT_PATH}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/home2/juneyeop/anaconda3/envs/opt67/bin/python "${SCRIPT_DIR}/../calibrate_qr_v3.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --calib_dataset "${CALIB_DATASET}" \
    --calib_nsamples ${CALIB_NSAMPLES} \
    --seqlen ${SEQLEN} \
    --seed ${SEED} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --output_path "${OUTPUT_PATH}"
