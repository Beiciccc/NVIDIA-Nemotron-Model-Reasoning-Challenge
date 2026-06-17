#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


COMP = "nvidia-nemotron-model-reasoning-challenge"
TODAY = "2026-06-08"
TARGET_COMPLETE = 5
POLL_SECONDS = 300
STATE_FILE = Path("reports/2026-06-08_file_submit_loop_state.json")
RUN_DIR = Path("outputs/jun08_file_submissions/file_loop")

PRIMARY_ZIP = Path("outputs/2026-05-24_submit5_cycle02_kien_original/submission.zip")
ALPHA001_ZIP = Path("outputs/2026-05-24_submit5_cycle03_alpha001/submission.zip")
ALPHA0005_ZIP = Path("outputs/2026-05-24_submit6_cycle06_alpha0005/submission.zip")

CANDIDATES = [
    {
        "message": "jun08_file_cycle01_kien_original_086",
        "zip": PRIMARY_ZIP,
        "reason": "Historical file submission scored 0.86 on 2026-05-24 as cycle02_kien086_original_control.",
    },
    {
        "message": "jun08_file_cycle02_kien_original_086_repeat",
        "zip": PRIMARY_ZIP,
        "reason": "Repeat of the strongest locally available 0.86 file after Code submission API returned 403.",
    },
    {
        "message": "jun08_file_cycle03_kien_original_086_repeat",
        "zip": PRIMARY_ZIP,
        "reason": "Repeat of the strongest locally available 0.86 file after Code submission API returned 403.",
    },
    {
        "message": "jun08_file_cycle04_kien_original_086_repeat",
        "zip": PRIMARY_ZIP,
        "reason": "Repeat of the strongest locally available 0.86 file after Code submission API returned 403.",
    },
    {
        "message": "jun08_file_cycle05_kien_original_086_repeat",
        "zip": PRIMARY_ZIP,
        "reason": "Repeat of the strongest locally available 0.86 file after Code submission API returned 403.",
    },
    {
        "message": "jun08_file_cycle06_kien_alpha001_fallback",
        "zip": ALPHA001_ZIP,
        "reason": "Fallback only; same 2026-05-24 family scored 0.85.",
    },
    {
        "message": "jun08_file_cycle07_kien_alpha0005_fallback",
        "zip": ALPHA0005_ZIP,
        "reason": "Fallback only; same 2026-05-24 family scored 0.85.",
    },
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def log(event, **payload):
    print(json.dumps({"atUtc": now_utc(), "event": event, **payload}, ensure_ascii=False), flush=True)


def run(cmd, timeout=None):
    log("command_start", cmd=cmd)
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    log("command_finish", cmd=cmd[:3], returncode=proc.returncode)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout


def load_state():
    if not STATE_FILE.exists():
        return {"failedMessages": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as exc:
        log("state_load_failed", error=repr(exc))
        return {"failedMessages": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def mark_failed(state, message, error):
    state["failedMessages"] = sorted(set(state.get("failedMessages", [])) | {message})
    state.setdefault("failures", []).append({"atUtc": now_utc(), "message": message, "error": error})
    save_state(state)


def submissions_snapshot():
    text = run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"], timeout=90)
    rows = list(csv.DictReader(text.splitlines()))
    today = [r for r in rows if r.get("date", "").startswith(TODAY)]
    return {
        "todayRows": today,
        "completeToday": [r for r in today if r.get("status") == "SubmissionStatus.COMPLETE"],
        "pendingToday": [r for r in today if r.get("status") == "SubmissionStatus.PENDING"],
        "errorToday": [r for r in today if r.get("status") == "SubmissionStatus.ERROR"],
    }


def submit_file(candidate):
    zip_path = candidate["zip"]
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing zip: {zip_path}")

    out_dir = RUN_DIR / candidate["message"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMP,
        "-f",
        str(zip_path),
        "-m",
        candidate["message"],
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    (out_dir / "submit_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "submit_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    log("file_submission_requested", message=candidate["message"], zip=str(zip_path), bytes=zip_path.stat().st_size)


def main():
    state = load_state()
    log("file_loop_start", date=TODAY, targetComplete=TARGET_COMPLETE, pollSeconds=POLL_SECONDS)

    while True:
        snapshot = submissions_snapshot()
        complete_count = len(snapshot["completeToday"])
        log(
            "submissions_snapshot",
            completeToday=complete_count,
            remaining=max(0, TARGET_COMPLETE - complete_count),
            pendingToday=snapshot["pendingToday"],
            errorToday=snapshot["errorToday"],
            todayRows=snapshot["todayRows"][:12],
            failedMessages=state.get("failedMessages", []),
        )

        if complete_count >= TARGET_COMPLETE:
            log("target_complete", completeToday=complete_count)
            return 0

        if snapshot["pendingToday"]:
            log("waiting_for_pending", sleepSeconds=POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue

        submitted_messages = {r.get("description", "") for r in snapshot["todayRows"]}
        failed_messages = set(state.get("failedMessages", []))
        next_candidate = None
        for candidate in CANDIDATES:
            if candidate["message"] in submitted_messages:
                continue
            if candidate["message"] in failed_messages:
                continue
            next_candidate = candidate
            break

        if not next_candidate:
            log("no_candidates_left", completeToday=complete_count)
            return 2

        log("candidate_selected", candidate={**next_candidate, "zip": str(next_candidate["zip"])})
        try:
            submit_file(next_candidate)
        except Exception as exc:
            log("file_submit_failed", message=next_candidate["message"], error=repr(exc))
            mark_failed(state, next_candidate["message"], repr(exc))
            time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
