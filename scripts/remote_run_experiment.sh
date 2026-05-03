#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/featurize/nemotron}"
PY="${PY:-$PROJECT_DIR/envs/py312/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/external/hf_models/nemotron_30b_bf16}"
VARIANT="${1:?variant required}"
MAX_LENGTH="${2:?max_length required}"
LR="${3:?lr required}"
SEED="${4:-42}"

cd "$PROJECT_DIR"

export PATH="$PROJECT_DIR/envs/py312/bin:$HOME/.local/bin:/home/featurize/work/.local/bin:$PATH"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
export KAGGLEHUB_CACHE="$PROJECT_DIR/external/kagglehub"
export HF_HOME="$PROJECT_DIR/external/hf"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p runs/logs runs/adapters runs/submissions runs/locks
exec 9>"$PROJECT_DIR/runs/locks/experiment.lock"
if ! flock -n 9; then
  echo "another experiment is already running; exiting"
  exit 0
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LR_SLUG="${LR//./p}"
LR_SLUG="${LR_SLUG//-e-/em}"
OUT_DIR="runs/adapters/${VARIANT}_len${MAX_LENGTH}_lr${LR_SLUG}_seed${SEED}_${STAMP}"

echo "experiment start: variant=$VARIANT max_length=$MAX_LENGTH lr=$LR seed=$SEED output=$OUT_DIR"
"$PY" scripts/remote_train_lora.py \
  --project-dir "$PROJECT_DIR" \
  --model-path "$MODEL_PATH" \
  --variant "$VARIANT" \
  --output-dir "$OUT_DIR" \
  --max-length "$MAX_LENGTH" \
  --epochs "${EPOCHS:-1}" \
  --lr "$LR" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-64}" \
  --rank "${RANK:-32}" \
  --alpha "${ALPHA:-32}" \
  --seed "$SEED" \
  --packing \
  --save-steps "${SAVE_STEPS:-10}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"

MSG="${VARIANT} maxlen${MAX_LENGTH} lr${LR} seed${SEED} rank${RANK:-32} ${STAMP}"
"$PY" scripts/remote_submit_adapter.py \
  --project-dir "$PROJECT_DIR" \
  --adapter-dir "$OUT_DIR" \
  --message "$MSG" \
  --poll \
  --poll-seconds "${POLL_SECONDS:-1800}"

cp runs/submissions_latest.txt "runs/submissions/submissions_latest_${STAMP}.txt" || true
echo "experiment done: $OUT_DIR"
