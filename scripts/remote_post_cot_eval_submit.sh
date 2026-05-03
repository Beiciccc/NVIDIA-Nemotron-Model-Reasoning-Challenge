#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/featurize/nemotron}"
PY="${PY:-$PROJECT_DIR/envs/py312/bin/python}"
BASE_ADAPTER="${BASE_ADAPTER:-$PROJECT_DIR/external/kienngx_models/tinker-adapter-backbone}"
COMPETITION="${COMPETITION:-nvidia-nemotron-model-reasoning-challenge}"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 ADAPTER_DIR SUBMIT_MESSAGE" >&2
  exit 2
fi

ADAPTER_DIR="$1"
SUBMIT_MESSAGE="$2"
GEN_LIMIT="${GEN_LIMIT:-120}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
MIN_EXACT_DELTA="${MIN_EXACT_DELTA:-0.02}"
MIN_EXACT_ABS="${MIN_EXACT_ABS:-0.70}"
POLL_SECONDS="${POLL_SECONDS:-1200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

cd "$PROJECT_DIR"
mkdir -p runs/eval runs/logs

ts="$(date -u +%Y%m%d_%H%M%S)"
tag="$(basename "$ADAPTER_DIR")"
base_json="runs/eval/base_gen${GEN_LIMIT}_len${MAX_PROMPT_LENGTH}_new${MAX_NEW_TOKENS}_${ts}.json"
cand_json="runs/eval/${tag}_gen${GEN_LIMIT}_len${MAX_PROMPT_LENGTH}_new${MAX_NEW_TOKENS}_${ts}.json"
decision_json="runs/eval/${tag}_decision_${ts}.json"

echo "POST_COT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "adapter=$ADAPTER_DIR"
echo "gen_limit=$GEN_LIMIT max_prompt_length=$MAX_PROMPT_LENGTH max_new_tokens=$MAX_NEW_TOKENS"

has_python_proc() {
  local pattern="$1"
  ps -eo pid=,comm=,args= | awk -v self="$$" -v pat="$pattern" '
    $1 != self && $2 ~ /python/ && $0 ~ pat { found=1 }
    END { exit found ? 0 : 1 }
  '
}

while has_python_proc "scripts/(remote_train_masked_lora|remote_train_lora)[.]py"; do
  echo "WAIT_TRAIN $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "$SLEEP_SECONDS"
done

while has_python_proc "scripts/remote_eval_adapter[.]py"; do
  echo "WAIT_EVAL $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep "$SLEEP_SECONDS"
done

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "missing adapter weights: $ADAPTER_DIR/adapter_model.safetensors" >&2
  exit 1
fi

echo "BASE_GEN_EVAL_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" scripts/remote_eval_adapter.py \
  --model-path "$PROJECT_DIR/external/hf_models/nemotron_30b_bf16" \
  --adapter "$BASE_ADAPTER" \
  --split official_eval \
  --boxed-target \
  --max-length "$MAX_PROMPT_LENGTH" \
  --loss-batch-size 8 \
  --generate \
  --generation-limit "$GEN_LIMIT" \
  --gen-batch-size "$GEN_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-json "$base_json"

echo "CAND_GEN_EVAL_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" scripts/remote_eval_adapter.py \
  --model-path "$PROJECT_DIR/external/hf_models/nemotron_30b_bf16" \
  --adapter "$ADAPTER_DIR" \
  --split official_eval \
  --boxed-target \
  --max-length "$MAX_PROMPT_LENGTH" \
  --loss-batch-size 8 \
  --generate \
  --generation-limit "$GEN_LIMIT" \
  --gen-batch-size "$GEN_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-json "$cand_json"

"$PY" - "$base_json" "$cand_json" "$decision_json" "$MIN_EXACT_DELTA" "$MIN_EXACT_ABS" <<'PY'
import json
import sys
from pathlib import Path

base_path, cand_path, out_path = map(Path, sys.argv[1:4])
min_delta = float(sys.argv[4])
min_abs = float(sys.argv[5])
base = json.loads(base_path.read_text())
cand = json.loads(cand_path.read_text())
base_exact = float(base.get("exact", -1))
cand_exact = float(cand.get("exact", -1))
decision = {
    "base_json": str(base_path),
    "candidate_json": str(cand_path),
    "base_exact": base_exact,
    "candidate_exact": cand_exact,
    "delta": cand_exact - base_exact,
    "min_delta": min_delta,
    "min_abs": min_abs,
    "submit": cand_exact >= min_abs and (cand_exact - base_exact) >= min_delta,
}
out_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
print(json.dumps(decision, indent=2))
PY

submit="$("$PY" - "$decision_json" <<'PY'
import json, sys
print("1" if json.loads(open(sys.argv[1]).read()).get("submit") else "0")
PY
)"

if [[ "$submit" != "1" ]]; then
  echo "SUBMIT_SKIPPED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

pending="$("$PY" - <<PY
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions("$COMPETITION")[:3]
first = subs[0] if subs else None
print("1" if first and "PENDING" in str(getattr(first, "_status", "")) else "0")
PY
)"

if [[ "$pending" == "1" ]]; then
  echo "SUBMIT_SKIPPED_PENDING $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

echo "SUBMIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" scripts/remote_submit_adapter.py \
  --adapter-dir "$ADAPTER_DIR" \
  --message "$SUBMIT_MESSAGE" \
  --poll \
  --poll-seconds "$POLL_SECONDS"
echo "POST_COT_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
