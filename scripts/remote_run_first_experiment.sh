#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/featurize/nemotron}"
PY="${PY:-$PROJECT_DIR/envs/py312/bin/python}"
cd "$PROJECT_DIR"

export PATH="$PROJECT_DIR/envs/py312/bin:$HOME/.local/bin:/home/featurize/work/.local/bin:$PATH"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
export KAGGLEHUB_CACHE="$PROJECT_DIR/external/kagglehub"
export HF_HOME="$PROJECT_DIR/external/hf"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p runs/logs runs/adapters runs/submissions
mkdir -p runs/locks
exec 9>"$PROJECT_DIR/runs/locks/first_experiment.lock"
if ! flock -n 9; then
  echo "another first experiment is already running; exiting"
  exit 0
fi

MODEL_ARGS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model-path "$MODEL_PATH")
  echo "using explicit model path: $MODEL_PATH"
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
SMOKE_DIR="runs/adapters/smoke_tong_s012_8192_$STAMP"
FULL_DIR_8192="runs/adapters/tong_s012_len8192_lr2p4e4_$STAMP"
FULL_DIR_4096="runs/adapters/tong_s012_len4096_lr2p4e4_$STAMP"

echo "python=$PY"
"$PY" - <<'PY'
import torch, transformers, peft, trl
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("libs", transformers.__version__, peft.__version__, trl.__version__)
PY

echo "smoke test: tong_s012 max_length=8192"
if "$PY" scripts/remote_train_lora.py \
  --project-dir "$PROJECT_DIR" \
  "${MODEL_ARGS[@]}" \
  --variant tong_s012 \
  --output-dir "$SMOKE_DIR" \
  --max-length 8192 \
  --epochs 0.01 \
  --lr 2.4e-4 \
  --batch-size 1 \
  --grad-accum 8 \
  --packing \
  --limit-records 16; then
  MAX_LENGTH=8192
  FULL_DIR="$FULL_DIR_8192"
else
  echo "8192 smoke failed, falling back to 4096"
  MAX_LENGTH=4096
  FULL_DIR="$FULL_DIR_4096"
fi

rm -rf "$SMOKE_DIR"

echo "full train: tong_s012 max_length=$MAX_LENGTH output=$FULL_DIR"
"$PY" scripts/remote_train_lora.py \
  --project-dir "$PROJECT_DIR" \
  "${MODEL_ARGS[@]}" \
  --variant tong_s012 \
  --output-dir "$FULL_DIR" \
  --max-length "$MAX_LENGTH" \
  --epochs 1 \
  --lr 2.4e-4 \
  --batch-size 1 \
  --grad-accum 64 \
  --packing \
  --save-steps 10 \
  --save-total-limit 3

MSG="tong_s012 maxlen${MAX_LENGTH} lr2.4e-4 rank32 ${STAMP}"
"$PY" scripts/remote_submit_adapter.py \
  --project-dir "$PROJECT_DIR" \
  --adapter-dir "$FULL_DIR" \
  --message "$MSG" \
  --poll \
  --poll-seconds 1800

cp runs/submissions_latest.txt "runs/submissions/submissions_latest_${STAMP}.txt" || true
echo "done: $FULL_DIR"
