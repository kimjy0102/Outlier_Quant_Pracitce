#!/bin/bash
# phase3 QR ver1 zero-shot launcher (Llama 기본, OPT도 MODEL_NAME만 바꾸면 사용 가능)
# 평가 task: piqa, arc_easy, arc_challenge, boolq, hellaswag, winogrande (num_fewshot=0)
# lm-eval 0.4.5는 BlockDialect repo 안의 사본을 PYTHONPATH로 우선 사용.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"   # 단일 GPU 권장 (Llama-7B는 1장이면 충분)
PYBIN="${PYBIN:-/home2/juneyeop/anaconda3/envs/opt67/bin/python}"

# 모델
MODEL_NAME="${MODEL_NAME:-llama-7b}"
MODEL_SOURCE="${MODEL_SOURCE:-omni}"   # auto / fp16 / omni

# QR setting (run_llama_qr.sh와 동일 default)
Q_BITS="${Q_BITS:-1}"
R_BITS="${R_BITS:-3}"
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-128}"
R_GROUP_SIZE="${R_GROUP_SIZE:-128}"
SELECTIVE_BASE_THRESHOLD="${SELECTIVE_BASE_THRESHOLD:-8.0}"
MODULE_SELECTIVE_BASE_THRESHOLDS="${MODULE_SELECTIVE_BASE_THRESHOLDS:-}"
SELECTIVE_INT_BITS="${SELECTIVE_INT_BITS:-4}"
RESIDUAL_CLIP_ALPHA="${RESIDUAL_CLIP_ALPHA:-0}"
REPLACE_SCOPE="${REPLACE_SCOPE:-all}"
TARGET_MODULES="${TARGET_MODULES:-}"

# weight quant (필요 시)
ENABLE_WEIGHT_QUANT="${ENABLE_WEIGHT_QUANT:-0}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-128}"

# zero-shot 평가 인자
TASKS="${TASKS:-piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-}"                     # 비어있으면 전체. smoke test 시 LIMIT=10 등
NO_REPLACE="${NO_REPLACE:-0}"          # 1이면 baseline (QR 미적용)
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../zeroshot_results}"

# optional 인자 빌드
EXTRA_ARGS=()
if [ "${NO_REPLACE}" = "1" ]; then
    EXTRA_ARGS+=(--no_replace)
fi
if [ "${ENABLE_WEIGHT_QUANT}" = "1" ]; then
    EXTRA_ARGS+=(--enable_weight_quant)
fi
if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi

echo "======================================================"
echo "Phase3 QR zero-shot"
echo "MODEL_NAME       : ${MODEL_NAME}"
echo "MODEL_SOURCE     : ${MODEL_SOURCE}"
echo "NO_REPLACE       : ${NO_REPLACE}"
echo "TASKS            : ${TASKS}"
echo "NUM_FEWSHOT      : ${NUM_FEWSHOT}"
echo "BATCH_SIZE       : ${BATCH_SIZE}"
echo "LIMIT            : ${LIMIT:-<full>}"
echo "QR              : q${Q_BITS}r${R_BITS} bg=${BASE_GROUP_SIZE} rg=${R_GROUP_SIZE} sel=${SELECTIVE_BASE_THRESHOLD}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
"${PYBIN}" "${SCRIPT_DIR}/../eval_zeroshot_qr.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --output_dir "${OUTPUT_DIR}" \
    --replace_scope "${REPLACE_SCOPE}" \
    --target_modules "${TARGET_MODULES}" \
    --q_bits ${Q_BITS} \
    --r_bits ${R_BITS} \
    --base_group_size ${BASE_GROUP_SIZE} \
    --r_group_size ${R_GROUP_SIZE} \
    --selective_base_threshold ${SELECTIVE_BASE_THRESHOLD} \
    --module_selective_base_thresholds "${MODULE_SELECTIVE_BASE_THRESHOLDS}" \
    --selective_int_bits ${SELECTIVE_INT_BITS} \
    --residual_clip_alpha ${RESIDUAL_CLIP_ALPHA} \
    --weight_bits ${WEIGHT_BITS} \
    --weight_group_size ${WEIGHT_GROUP_SIZE} \
    --tasks "${TASKS}" \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size ${BATCH_SIZE} \
    "${EXTRA_ARGS[@]}"
