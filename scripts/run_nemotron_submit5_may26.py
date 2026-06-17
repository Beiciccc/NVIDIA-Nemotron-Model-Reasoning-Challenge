#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kaggle


PROJECT = Path(__file__).resolve().parents[1]
KAGGLE_BIN = Path("/Users/beici/Library/Python/3.9/bin/kaggle")
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
TARGET_COMPLETE = 5
POLL_SECONDS = 120
TODAY_UTC = datetime.now(timezone.utc).date()
REPORT_PATH = PROJECT / "reports/2026-05-26_submit5_live.json"


CANDIDATES = [
    {
        "message": "may26_cycle03_sangmin_dream_kernel",
        "kernel": "beicicc/nemotron-sangmin-dream26",
        "version": "1",
        "file": "submission.zip",
        "require_kernel_complete": True,
        "block_until_ready_seconds": 1800,
        "reason": "fresh 4.25GB Sangmin adapter; high-variance candidate after minyam/syu settle",
    },
    {
        "message": "may26_cycle04_finding_nemo_public_kernel",
        "kernel": "ayomide2000/finding-nemo",
        "version": "1",
        "file": "submission.zip",
        "reason": "popular public Kien/SVD-style output; may recover stronger Kien conversion",
    },
    {
        "message": "may26_cycle05_kien_tinker_anchor_kernel",
        "kernel": "beicicc/nemotron-direct-kien-tinker-anchor-may26",
        "version": "1",
        "file": "submission.zip",
        "reason": "known strong Kien anchor family; expected around 0.86",
    },
    {
        "message": "may26_cycle06_huikang_v20_mirror_kernel",
        "kernel": "beicicc/nemotron-direct-huikang-v20-may26",
        "version": "2",
        "file": "submission.zip",
        "reason": "Huikang/backtracking v20 mirror; known 0.84-0.85 family",
    },
    {
        "message": "may26_cycle07_backtracking_v20_repeat_kernel",
        "kernel": "beicicc/nemotron-direct-backtracking-v20-c5",
        "version": "1",
        "file": "submission.zip",
        "reason": "validated server-side fallback; scored 0.85 on 2026-05-25",
    },
    {
        "message": "may26_cycle08_lopure_adapter3_repeat_kernel",
        "kernel": "beicicc/nemotron-direct-lopure-adapter3-c4",
        "version": "1",
        "file": "submission.zip",
        "reason": "validated fallback; scored 0.84 on 2026-05-25",
    },
    {
        "message": "may26_cycle09_lopure_root_repeat_kernel",
        "kernel": "beicicc/nemotron-direct-lopure-root-c3",
        "version": "1",
        "file": "submission.zip",
        "reason": "validated fallback; scored 0.84 on 2026-05-25",
    },
    {
        "message": "may26_cycle10_jeffrey_sft_public_kernel",
        "kernel": "jeffreypikeai/nemotron-supervised-tuning",
        "version": "1",
        "file": "submission.zip",
        "reason": "public SFT output exists; high risk because notebook is not clearly final-submission oriented",
    },
]


def status_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def submission_to_dict(sub: Any) -> dict[str, Any]:
    return {
        "ref": getattr(sub, "ref", None),
        "date": getattr(sub, "date", None).isoformat(sep=" ") if getattr(sub, "date", None) else "",
        "description": getattr(sub, "description", ""),
        "status": status_name(getattr(sub, "status", "")),
        "publicScore": getattr(sub, "public_score", ""),
        "privateScore": getattr(sub, "private_score", ""),
        "errorDescription": getattr(sub, "error_description", ""),
        "totalBytes": getattr(sub, "total_bytes", None),
        "url": getattr(sub, "url", ""),
    }


def get_submissions() -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            return [submission_to_dict(s) for s in kaggle.api.competition_submissions(COMPETITION)]
        except Exception as exc:
            last_exc = exc
            print(
                json.dumps(
                    {
                        "event": "submissions_query_retry",
                        "attempt": attempt,
                        "error": repr(exc),
                        "atUtc": datetime.now(timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"failed to query submissions after retries: {last_exc!r}")


def get_max_daily_submissions() -> int | None:
    try:
        comps = kaggle.api.competitions_list(search=COMPETITION)
        for comp in comps:
            if getattr(comp, "ref", "").endswith(COMPETITION) or getattr(comp, "url", "").endswith(COMPETITION):
                return getattr(comp, "max_daily_submissions", None)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "max_daily_query_failed",
                    "error": repr(exc),
                    "atUtc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return None


def is_today(row: dict[str, Any]) -> bool:
    return str(row.get("date", "")).startswith(TODAY_UTC.isoformat())


def summarize(submissions: list[dict[str, Any]], max_daily: int | None, state: dict[str, Any]) -> dict[str, Any]:
    today = [row for row in submissions if is_today(row)]
    complete = [row for row in today if row["status"] == "COMPLETE"]
    error = [row for row in today if row["status"] == "ERROR"]
    pending = [row for row in today if row["status"] == "PENDING"]
    non_error = [row for row in today if row["status"] != "ERROR"]
    remaining_estimate = None
    if max_daily is not None:
        remaining_estimate = max(0, max_daily - len(non_error))
    return {
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "todayUtc": TODAY_UTC.isoformat(),
        "competition": COMPETITION,
        "targetComplete": TARGET_COMPLETE,
        "maxDailySubmissions": max_daily,
        "todayCompleteCount": len(complete),
        "todayErrorCount": len(error),
        "todayPendingCount": len(pending),
        "todayNonErrorCount": len(non_error),
        "remainingEstimateExcludingErrors": remaining_estimate,
        "bestTodayPublicScore": max([float(r["publicScore"]) for r in complete if r.get("publicScore")] or [0.0]),
        "todayRows": today,
        "candidateMessages": [c["message"] for c in CANDIDATES],
        "state": state,
    }


def write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [str(KAGGLE_BIN), *args]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)
    return proc


def kernel_status(kernel: str) -> str:
    proc = run_cli(["kernels", "status", kernel], check=False)
    text = proc.stdout
    if "KernelWorkerStatus.COMPLETE" in text:
        return "COMPLETE"
    if "KernelWorkerStatus.ERROR" in text:
        return "ERROR"
    if "KernelWorkerStatus.RUNNING" in text:
        return "RUNNING"
    if "KernelWorkerStatus.QUEUED" in text:
        return "QUEUED"
    return "UNKNOWN"


def choose_candidate(submissions: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any] | None:
    seen_messages = {row["description"] for row in submissions}
    failed_requests = set(state.setdefault("failedSubmitRequests", []))
    now = time.time()
    for cand in CANDIDATES:
        msg = cand["message"]
        if msg in seen_messages or msg in failed_requests:
            continue
        if cand.get("require_kernel_complete"):
            first_seen = state.setdefault("kernelWaitStarted", {}).setdefault(msg, now)
            status = kernel_status(cand["kernel"])
            state.setdefault("kernelStatuses", {})[msg] = {"status": status, "checkedAtUtc": datetime.now(timezone.utc).isoformat()}
            if status == "COMPLETE":
                return cand
            if status == "ERROR":
                state.setdefault("failedSubmitRequests", []).append(msg)
                continue
            waited = now - first_seen
            if waited < cand.get("block_until_ready_seconds", 0):
                state["blockedOnKernel"] = {
                    "message": msg,
                    "kernel": cand["kernel"],
                    "status": status,
                    "waitedSeconds": round(waited),
                }
                return None
            continue
        return cand
    return None


def submit_candidate(cand: dict[str, Any], state: dict[str, Any]) -> bool:
    args = [
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        cand["file"],
        "-k",
        cand["kernel"],
        "-v",
        cand["version"],
        "-m",
        cand["message"],
    ]
    proc = run_cli(args, check=False)
    event = {
        "atUtc": datetime.now(timezone.utc).isoformat(),
        "candidate": cand,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
    }
    state.setdefault("submitEvents", []).append(event)
    if proc.returncode == 0:
        return True
    if "daily" in proc.stdout.lower() or "maximum" in proc.stdout.lower():
        state["stoppedReason"] = "Kaggle daily submission limit or maximum reached"
        raise RuntimeError(proc.stdout)
    state.setdefault("failedSubmitRequests", []).append(cand["message"])
    return False


def main() -> None:
    max_daily = get_max_daily_submissions()
    state: dict[str, Any] = {"startedAtUtc": datetime.now(timezone.utc).isoformat()}
    print(
        json.dumps(
            {
                "event": "start",
                "todayUtc": TODAY_UTC.isoformat(),
                "targetComplete": TARGET_COMPLETE,
                "maxDailySubmissions": max_daily,
                "pollSeconds": POLL_SECONDS,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while True:
        try:
            submissions = get_submissions()
        except Exception as exc:
            state["lastQueryError"] = {"atUtc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}
            write_report(
                {
                    "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                    "todayUtc": TODAY_UTC.isoformat(),
                    "competition": COMPETITION,
                    "targetComplete": TARGET_COMPLETE,
                    "maxDailySubmissions": max_daily,
                    "state": state,
                }
            )
            time.sleep(POLL_SECONDS)
            continue
        report = summarize(submissions, max_daily, state)
        write_report(report)
        print(
            json.dumps(
                {
                    "event": "poll",
                    "updatedAtUtc": report["updatedAtUtc"],
                    "todayComplete": report["todayCompleteCount"],
                    "todayPending": report["todayPendingCount"],
                    "todayError": report["todayErrorCount"],
                    "bestTodayPublicScore": report["bestTodayPublicScore"],
                    "remainingEstimateExcludingErrors": report["remainingEstimateExcludingErrors"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if report["todayCompleteCount"] >= TARGET_COMPLETE:
            state["stoppedReason"] = "target complete submissions reached"
            write_report(summarize(get_submissions(), max_daily, state))
            return
        if report["todayPendingCount"] > 0:
            time.sleep(POLL_SECONDS)
            continue
        cand = choose_candidate(submissions, state)
        write_report(summarize(submissions, max_daily, state))
        if cand is None:
            print(json.dumps({"event": "wait_no_candidate_ready", "state": state}, sort_keys=True), flush=True)
            time.sleep(POLL_SECONDS)
            continue
        print(json.dumps({"event": "submit", "candidate": cand}, sort_keys=True), flush=True)
        submitted = submit_candidate(cand, state)
        write_report(summarize(get_submissions(), max_daily, state))
        if not submitted:
            print(json.dumps({"event": "submit_request_failed_continue", "message": cand["message"]}, sort_keys=True), flush=True)
            time.sleep(10)
            continue
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
