#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_USER="${REMOTE_USER:-featurize}"
REMOTE_HOST="${REMOTE_HOST:-workspace.featurize.cn}"
REMOTE_PORT="${REMOTE_PORT:-14114}"
REMOTE_DIR="${REMOTE_DIR:-/home/featurize/nemotron}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/codex_featurize_nemotron}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"

run_sync() {
  mkdir -p "$LOCAL_DIR/runs/logs"
  local scope="${1:-small}"
  local paths=("data" "scripts" "runs")
  if [[ "$scope" == "full" ]]; then
    paths=("data" "scripts" "runs" "external/datasets")
  fi

  ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -p "$REMOTE_PORT" \
    "$REMOTE_USER@$REMOTE_HOST" \
    "cd '$REMOTE_DIR' && tar --ignore-failed-read \
      --exclude='data/*.zip' \
      --exclude='./data/*.zip' \
      --exclude='./envs' \
      --exclude='./external/kagglehub' \
      --exclude='./external/hf' \
      --exclude='./external/hf_models' \
      --exclude='./.cache' \
      --exclude='*/__pycache__' \
      --exclude='*.archive' \
      -czf - ${paths[*]}" | tar -xzf - -C "$LOCAL_DIR"
}

if [[ "${1:-}" == "--full" ]]; then
  run_sync full
elif [[ "${1:-}" == "--loop" ]]; then
  while true; do
    echo "sync start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_sync small
    echo "sync done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep "$SYNC_INTERVAL"
  done
else
  run_sync small
fi
