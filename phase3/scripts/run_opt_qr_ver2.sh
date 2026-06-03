#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=7

MODEL_NAME="opt-6.7b"
MODEL_SOURCE="${MODEL_SOURCE:-omni}"

# QR params (selective 관련 모두 ver2에서 제거됨)
Q_BITS=1
R_BITS=3
BASE_GROUP_SIZE=128
R_GROUP_SIZE=128
RESIDUAL_CLIP_ALPHA=0

# INT activation params (QR 미적용 모듈용)
INT_ACT_BITS=4
INT_ACT_GROUP_SIZE=128

REPLACE_SCOPE="all"
TARGET_MODULES="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,fc1"
# QR_MODULES 옵션:
#   ""    → arch별 default (OPT: q/k/v_proj + fc1)
#   "all" → target_modules 전체 QR
#   "none"→ 전부 INT (QR 없음)
#   직접 지정도 가능 예: "self_attn.q_proj,fc1"
QR_MODULES="${QR_MODULES:-"all"}"

# ver1과 분리해서 저장 (결과 비교용).
# 추가로 qr_modules 태그를 폴더명에 붙여 매핑별 결과를 구분.
QR_MODULES_TAG="${QR_MODULES:-default}"
QR_MODULES_TAG="${QR_MODULES_TAG//,/_}"
QR_MODULES_TAG="${QR_MODULES_TAG//\./}"
OUTPUT_DIR="${SCRIPT_DIR}/../results_ver2/qr_${QR_MODULES_TAG}"

EVAL_DATASET="${EVAL_DATASET:-wikitext2}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-2048}"
C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../cache}"

BASELINE_ONLY="${BASELINE_ONLY:-0}"
BASELINE_ONLY_ARGS=()
if [ "${BASELINE_ONLY}" = "1" ]; then
    BASELINE_ONLY_ARGS=(--baseline_only)
fi

echo "======================================================"
echo "Running Phase3 QR-ver2 on ${MODEL_NAME}"
echo "MODEL_SOURCE: ${MODEL_SOURCE}, BASELINE_ONLY: ${BASELINE_ONLY}"
echo "QR_MODULES: '${QR_MODULES}' (default='' uses arch default)"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "RESIDUAL_CLIP_ALPHA: ${RESIDUAL_CLIP_ALPHA}"
echo "INT_ACT_BITS: ${INT_ACT_BITS}, INT_ACT_GROUP_SIZE: ${INT_ACT_GROUP_SIZE}"
echo "EVAL_DATASET: ${EVAL_DATASET}, EVAL_SPLIT: ${EVAL_SPLIT}, EVAL_NSAMPLES: ${EVAL_NSAMPLES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/home2/juneyeop/anaconda3/envs/opt67/bin/python "${SCRIPT_DIR}/../run_qr_ppl_ver2.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --output_dir "${OUTPUT_DIR}" \
    --replace_scope "${REPLACE_SCOPE}" \
    --target_modules "${TARGET_MODULES}" \
    --qr_modules "${QR_MODULES}" \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --int_act_bits ${INT_ACT_BITS} \
    --int_act_group_size ${INT_ACT_GROUP_SIZE} \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --eval_nsamples ${EVAL_NSAMPLES} \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    "${BASELINE_ONLY_ARGS[@]}"
