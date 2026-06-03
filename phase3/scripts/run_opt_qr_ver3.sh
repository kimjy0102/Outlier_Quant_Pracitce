#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=7

MODEL_NAME="${MODEL_NAME:-opt-6.7b}"
MODEL_SOURCE="${MODEL_SOURCE:-omni}"

# ver1 selective과 동일 시맨틱:
#   group base >= SELECTIVE_BASE_THRESHOLD → QR
#   미만 → INT4 fake quant
SELECTIVE_BASE_THRESHOLD="${SELECTIVE_BASE_THRESHOLD:-16.0}"
MODULE_SELECTIVE_BASE_THRESHOLDS="${MODULE_SELECTIVE_BASE_THRESHOLDS:-}"   # 예: "fc2:64.0"

# QR params
Q_BITS="${Q_BITS:-1}"
R_BITS="${R_BITS:-3}"
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-128}"
R_GROUP_SIZE="${R_GROUP_SIZE:-128}"
RESIDUAL_CLIP_ALPHA="${RESIDUAL_CLIP_ALPHA:-0}"

# INT4 (mask=False group) params
ACT_BITS="${ACT_BITS:-4}"
ACT_GROUP_SIZE="${ACT_GROUP_SIZE:-128}"

REPLACE_SCOPE="${REPLACE_SCOPE:-all}"
TARGET_MODULES="${TARGET_MODULES:-}"   # 비우면 arch 전체 모듈

# Calibration artifact 경로 (calibrate.sh 결과)
CALIB_DATASET="${CALIB_DATASET:-wikitext2}"
CALIB_NSAMPLES="${CALIB_NSAMPLES:-128}"
CALIB_SEQLEN="${CALIB_SEQLEN:-2048}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/../calib_artifacts}"
CALIB_ARTIFACT="${CALIB_ARTIFACT:-${ARTIFACT_DIR}/${MODEL_NAME}_${CALIB_DATASET}_n${CALIB_NSAMPLES}_s${CALIB_SEQLEN}_gs${BASE_GROUP_SIZE}.pt}"

# 1이면 channel permutation 비활성 (mask만 적용). 같은 calibration artifact 재사용 가능.
# OUTPUT_DIR 태그에 사용되므로 OUTPUT_DIR 계산 전에 결정해야 한다.
DISABLE_PERMUTATION="${DISABLE_PERMUTATION:-1}"
DISABLE_PERM_ARGS=()
PERM_TAG="on"
if [ "${DISABLE_PERMUTATION}" = "1" ]; then
    DISABLE_PERM_ARGS=(--disable_permutation)
    PERM_TAG="off"
fi

# Phase 1: 1이면 base / INT4 scale 정적화 (runtime max reduction 제거).
USE_STATIC_SCALES="${USE_STATIC_SCALES:-0}"
STATIC_SCALES_ARGS=()
STATIC_TAG="dyn"
if [ "${USE_STATIC_SCALES}" = "1" ]; then
    STATIC_SCALES_ARGS=(--use_static_scales)
    STATIC_TAG="static"
fi

# 결과 폴더 (ver1/ver2 와 분리). DISABLE_PERMUTATION / USE_STATIC_SCALES도 폴더명에 반영.
THRESH_TAG="${SELECTIVE_BASE_THRESHOLD//./p}"
OUTPUT_DIR="${SCRIPT_DIR}/../results_ver3/thr${THRESH_TAG}_calib_${CALIB_DATASET}_perm${PERM_TAG}_${STATIC_TAG}"

EVAL_DATASET="${EVAL_DATASET:-wikitext2}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-256}"
C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../cache}"
EVAL_SEQLEN="${EVAL_SEQLEN:-2048}"

BASELINE_ONLY="${BASELINE_ONLY:-0}"
BASELINE_ONLY_ARGS=()
if [ "${BASELINE_ONLY}" = "1" ]; then
    BASELINE_ONLY_ARGS=(--baseline_only)
fi

echo "======================================================"
echo "Phase3 QR-ver3 (perm + static mask)"
echo "MODEL_NAME           : ${MODEL_NAME}"
echo "MODEL_SOURCE         : ${MODEL_SOURCE}"
echo "CALIB_ARTIFACT       : ${CALIB_ARTIFACT}"
echo "SELECTIVE_BASE_THRESHOLD: ${SELECTIVE_BASE_THRESHOLD}"
echo "MODULE_SELECTIVE_BASE_THRESHOLDS: ${MODULE_SELECTIVE_BASE_THRESHOLDS}"
echo "Q_BITS=${Q_BITS}, R_BITS=${R_BITS}"
echo "BASE_GROUP_SIZE=${BASE_GROUP_SIZE}, R_GROUP_SIZE=${R_GROUP_SIZE}, RESIDUAL_CLIP_ALPHA=${RESIDUAL_CLIP_ALPHA}"
echo "ACT_BITS=${ACT_BITS}, ACT_GROUP_SIZE=${ACT_GROUP_SIZE}"
echo "EVAL_DATASET=${EVAL_DATASET}, EVAL_SPLIT=${EVAL_SPLIT}, EVAL_NSAMPLES=${EVAL_NSAMPLES}"
echo "DISABLE_PERMUTATION  : ${DISABLE_PERMUTATION} (perm tag: ${PERM_TAG})"
echo "USE_STATIC_SCALES    : ${USE_STATIC_SCALES} (scales tag: ${STATIC_TAG})"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/home2/juneyeop/anaconda3/envs/opt67/bin/python "${SCRIPT_DIR}/../run_qr_ppl_ver3.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --output_dir "${OUTPUT_DIR}" \
    --replace_scope "${REPLACE_SCOPE}" \
    --target_modules "${TARGET_MODULES}" \
    --calib_artifact "${CALIB_ARTIFACT}" \
    --selective_base_threshold ${SELECTIVE_BASE_THRESHOLD} \
    --module_selective_base_thresholds "${MODULE_SELECTIVE_BASE_THRESHOLDS}" \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --act_bits ${ACT_BITS} \
    --act_group_size ${ACT_GROUP_SIZE} \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --eval_nsamples ${EVAL_NSAMPLES} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --seqlen ${EVAL_SEQLEN} \
    "${BASELINE_ONLY_ARGS[@]}" \
    "${DISABLE_PERM_ARGS[@]}" \
    "${STATIC_SCALES_ARGS[@]}"
