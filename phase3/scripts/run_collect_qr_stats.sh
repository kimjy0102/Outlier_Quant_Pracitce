#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# QR stats collection 전용 runner.
# 일반 PPL 평가는 run_opt_qr.sh를 쓰고, 분포/CSV/PNG 분석은 이 파일로 분리한다.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

MODEL_NAME="${MODEL_NAME:-llama-7b}"
MODEL_SOURCE="${MODEL_SOURCE:-omni}" # omni / fp16 / 

Q_BITS="${Q_BITS:-1}"
R_BITS="${R_BITS:-3}"
BASE_GROUP_SIZE="${BASE_GROUP_SIZE:-128}"
R_GROUP_SIZE="${R_GROUP_SIZE:-128}"
SELECTIVE_BASE_THRESHOLD="${SELECTIVE_BASE_THRESHOLD:-8.0}"
SELECTIVE_INT_BITS="${SELECTIVE_INT_BITS:-4}"
RESIDUAL_CLIP_ALPHA="${RESIDUAL_CLIP_ALPHA:-0}"

INT_BITS="${INT_BITS:-4}"
INT_GROUP_SIZE="${INT_GROUP_SIZE:-128}"

REPLACE_SCOPE="${REPLACE_SCOPE:-all}"
ONE_LAYER_IDX="${ONE_LAYER_IDX:-0}"
CUSTOM_LAYER_INDICES="${CUSTOM_LAYER_INDICES:-}"
TARGET_MODULES="${TARGET_MODULES:-auto}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../results_qr_stats}"
EVAL_DATASET="${EVAL_DATASET:-wikitext2}"  # c4_omni, c4, c4_new, or wikitext2
EVAL_SPLIT="${EVAL_SPLIT:-test}"
STATS_NSAMPLES="${STATS_NSAMPLES:-128}"
SEED="${SEED:-42}"
C4_CACHE_DIR="${C4_CACHE_DIR:-${SCRIPT_DIR}/../cache}" # c4_omni는 writable cache가 필요하다.

MAX_SAMPLES_PER_MODULE="${MAX_SAMPLES_PER_MODULE:-200000}"
MAX_Q1_SAMPLES_PER_BASE="${MAX_Q1_SAMPLES_PER_BASE:-50000}"
SAMPLE_CHUNK="${SAMPLE_CHUNK:-20000}"
HIST_BINS="${HIST_BINS:-100}"

NO_PLOTS="${NO_PLOTS:-0}"
NO_PLOTS_ARGS=()
if [ "${NO_PLOTS}" = "1" ]; then
    NO_PLOTS_ARGS=(--no_plots)
fi

echo "======================================================"
echo "Collecting Phase3 QR stats on ${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_SOURCE: ${MODEL_SOURCE}"
echo "Q_BITS: ${Q_BITS}, R_BITS: ${R_BITS}"
echo "BASE_GROUP_SIZE: ${BASE_GROUP_SIZE}, R_GROUP_SIZE: ${R_GROUP_SIZE}"
echo "SELECTIVE_BASE_THRESHOLD: ${SELECTIVE_BASE_THRESHOLD}"
echo "SELECTIVE_INT_BITS: ${SELECTIVE_INT_BITS}"
echo "RESIDUAL_CLIP_ALPHA: ${RESIDUAL_CLIP_ALPHA}"
echo "INT_BITS: ${INT_BITS}, INT_GROUP_SIZE: ${INT_GROUP_SIZE}"
echo "REPLACE_SCOPE: ${REPLACE_SCOPE}, TARGET_MODULES: ${TARGET_MODULES}"
echo "EVAL_DATASET: ${EVAL_DATASET}, EVAL_SPLIT: ${EVAL_SPLIT}, STATS_NSAMPLES: ${STATS_NSAMPLES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NO_PLOTS: ${NO_PLOTS}"
echo "======================================================"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/home2/juneyeop/anaconda3/envs/opt67/bin/python "${SCRIPT_DIR}/../collect_qr_stats.py" \
    --model_name "${MODEL_NAME}" \
    --model_source "${MODEL_SOURCE}" \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_split "${EVAL_SPLIT}" \
    --stats_nsamples "${STATS_NSAMPLES}" \
    --seed "${SEED}" \
    --c4_cache_dir "${C4_CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --replace_scope "${REPLACE_SCOPE}" \
    --one_layer_idx "${ONE_LAYER_IDX}" \
    --custom_layer_indices "${CUSTOM_LAYER_INDICES}" \
    --target_modules "${TARGET_MODULES}" \
    --q_bits "${Q_BITS}" \
    --r_bits "${R_BITS}" \
    --base_group_size "${BASE_GROUP_SIZE}" \
    --r_group_size "${R_GROUP_SIZE}" \
    --selective_base_threshold "${SELECTIVE_BASE_THRESHOLD}" \
    --selective_int_bits "${SELECTIVE_INT_BITS}" \
    --residual_clip_alpha "${RESIDUAL_CLIP_ALPHA}" \
    --int_bits "${INT_BITS}" \
    --int_group_size "${INT_GROUP_SIZE}" \
    --max_samples_per_module "${MAX_SAMPLES_PER_MODULE}" \
    --max_q1_samples_per_base "${MAX_Q1_SAMPLES_PER_BASE}" \
    --sample_chunk "${SAMPLE_CHUNK}" \
    --hist_bins "${HIST_BINS}" \
    "${NO_PLOTS_ARGS[@]}"
