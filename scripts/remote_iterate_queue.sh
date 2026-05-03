#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/featurize/nemotron}"
QUEUE_FILE="${QUEUE_FILE:-$PROJECT_DIR/runs/experiment_queue.tsv}"
DONE_FILE="${DONE_FILE:-$PROJECT_DIR/runs/experiment_queue.done}"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"

cd "$PROJECT_DIR"
mkdir -p runs/logs runs/locks

if [[ ! -f "$QUEUE_FILE" ]]; then
  cat > "$QUEUE_FILE" <<'EOF'
tong_s012	8192	2.0e-4	43
tong_s011	8192	2.4e-4	42
tong_s006	8192	2.4e-4	42
tong_s012	4096	2.4e-4	44
EOF
fi
touch "$DONE_FILE"

exec 8>"$PROJECT_DIR/runs/locks/iterate_queue.lock"
if ! flock -n 8; then
  echo "iterate queue is already running; exiting"
  exit 0
fi

wait_for_idle() {
  while pgrep -af "remote_run_first_experiment|remote_run_experiment|remote_train_lora.py" | grep -v remote_iterate_queue >/dev/null; do
    echo "training still active at $(date -u); sleeping ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
  done
}

while IFS=$'\t' read -r variant max_length lr seed; do
  [[ -z "${variant:-}" || "$variant" == \#* ]] && continue
  key="${variant}_${max_length}_${lr}_${seed}"
  if grep -qxF "$key" "$DONE_FILE"; then
    continue
  fi

  wait_for_idle
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  log="runs/logs/queue_${key}_${stamp}.log"
  echo "queue launch: $key log=$log"
  if scripts/remote_run_experiment.sh "$variant" "$max_length" "$lr" "$seed" > "$log" 2>&1; then
    echo "$key" >> "$DONE_FILE"
    echo "queue done: $key"
  else
    echo "queue failed: $key; see $log"
    exit 1
  fi
done < "$QUEUE_FILE"

echo "queue exhausted at $(date -u)"
