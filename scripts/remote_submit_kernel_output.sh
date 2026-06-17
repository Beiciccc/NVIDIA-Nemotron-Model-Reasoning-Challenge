#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <kernel-ref> <candidate-id> <message>" >&2
  exit 2
fi

KERNEL_REF="$1"
CANDIDATE_ID="$2"
MESSAGE="$3"

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER}"
REMOTE_PROJECT="${REMOTE_PROJECT:?set REMOTE_PROJECT}"
REMOTE_PASS="${REMOTE_PASS:?set REMOTE_PASS}"
COMPETITION="${COMPETITION:-nvidia-nemotron-model-reasoning-challenge}"

ENCODED_COMMAND="$(KERNEL_REF_ENV="$KERNEL_REF" CANDIDATE_ID_ENV="$CANDIDATE_ID" MESSAGE_ENV="$MESSAGE" COMPETITION_ENV="$COMPETITION" REMOTE_PROJECT_ENV="$REMOTE_PROJECT" python3 - <<'PY'
import base64
import os


def ps_single(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


kernel = os.environ["KERNEL_REF_ENV"]
candidate = os.environ["CANDIDATE_ID_ENV"]
message = os.environ["MESSAGE_ENV"]
competition = os.environ["COMPETITION_ENV"]
base = os.environ["REMOTE_PROJECT_ENV"]
cmd = f'''
$ErrorActionPreference = "Stop"
$base = {ps_single(base)}
$kernel = {ps_single(kernel)}
$candidate = {ps_single(candidate)}
$message = {ps_single(message)}
$competition = {ps_single(competition)}
$out = Join-Path $base ("outputs/kaggle_submit_cycles_20260524/" + $candidate)
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location -Path $base
$downloadScript = Join-Path $base "scripts/download_kernel_output_streaming.py"
python $downloadScript $kernel $out
$report = Join-Path $out "kaggle_rtx_train_report.json"
if (!(Test-Path $report)) {{
  throw "kaggle_rtx_train_report.json not found in $out"
}}
python -c "import json,sys; r=json.load(open(sys.argv[1], encoding='utf-8')); print(json.dumps({{k:r.get(k) for k in ['status','candidate_id','elapsed_seconds','train_result']}}, default=str)); assert r.get('status') == 'ok'" $report
$zip = Join-Path $out "submission.zip"
if (!(Test-Path $zip)) {{
  throw "submission.zip not found in $out"
}}
python -c "import zipfile,sys,os,json; z=sys.argv[1]; names=zipfile.ZipFile(z).namelist(); print(json.dumps({{'zip':z,'size':os.path.getsize(z),'names':names[:20]}})); assert 'adapter_config.json' in names and 'adapter_model.safetensors' in names" $zip
kaggle competitions submit -c $competition -f $zip -m $message
$deadline = (Get-Date).AddMinutes(90)
do {{
  Start-Sleep -Seconds 60
  $subs = kaggle competitions submissions -c $competition
  $subs | Out-File -Encoding utf8 (Join-Path $out "submissions_after_submit.txt")
  $first = ($subs | Select-String -SimpleMatch $message | Select-Object -First 1).Line
  if ($first) {{
    Write-Output $first
    if ($first -match "COMPLETE|ERROR") {{ break }}
  }} else {{
    $subs | Select-Object -First 8
  }}
}} while ((Get-Date) -lt $deadline)
'''
print(base64.b64encode(cmd.encode("utf-16le")).decode())
PY
)"

export ENCODED_COMMAND REMOTE_PASS REMOTE_PORT REMOTE_USER REMOTE_HOST
/usr/bin/expect <<'EOF'
log_user 0
set timeout 7200
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $env(REMOTE_PORT) $env(REMOTE_USER)@$env(REMOTE_HOST) powershell -NoProfile -EncodedCommand $env(ENCODED_COMMAND)
expect {
  "password:" { send "$env(REMOTE_PASS)\r"; log_user 1; exp_continue }
  eof
}
EOF
