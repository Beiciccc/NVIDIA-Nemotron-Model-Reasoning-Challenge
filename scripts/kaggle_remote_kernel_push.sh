#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <local-kernel-dir>" >&2
  exit 2
fi

LOCAL_KERNEL_DIR="$1"
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:?set REMOTE_PROJECT}"
REMOTE_PASS="${REMOTE_PASS:?set REMOTE_PASS}"

if [[ ! -d "$LOCAL_KERNEL_DIR" ]]; then
  echo "kernel dir not found: $LOCAL_KERNEL_DIR" >&2
  exit 2
fi

KERNEL_BASENAME="$(basename "$LOCAL_KERNEL_DIR")"
REMOTE_KERNEL_DIR="$REMOTE_PROJECT/kaggle_kernels/$KERNEL_BASENAME"

/usr/bin/expect <<EOF
set timeout 60
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST powershell -NoProfile -Command New-Item -ItemType Directory -Force -Path $REMOTE_KERNEL_DIR
expect {
  "password:" { send "$REMOTE_PASS\r"; exp_continue }
  eof
}
EOF

/usr/bin/expect <<EOF
set timeout 180
spawn scp -r -P $REMOTE_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$LOCAL_KERNEL_DIR/." "$REMOTE_USER@$REMOTE_HOST:$REMOTE_KERNEL_DIR/"
expect {
  "password:" { send "$REMOTE_PASS\r"; exp_continue }
  eof
}
EOF

ENCODED_COMMAND="$(python3 - <<PY
import base64
cmd = r"Set-Location -Path '$REMOTE_PROJECT'; kaggle kernels push -p '$REMOTE_KERNEL_DIR'"
print(base64.b64encode(cmd.encode('utf-16le')).decode())
PY
)"

/usr/bin/expect <<EOF
set timeout 240
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST powershell -NoProfile -EncodedCommand $ENCODED_COMMAND
expect {
  "password:" { send "$REMOTE_PASS\r"; exp_continue }
  eof
}
EOF
