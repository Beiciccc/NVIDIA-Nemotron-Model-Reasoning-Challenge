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
TODAY_UTC = datetime.now(timezone.utc).date()
TARGET_COMPLETE = 5
POLL_SECONDS = 120
REPORT_PATH = PROJECT / f"reports/{TODAY_UTC.isoformat()}_submit5_live.json"


CANDIDATES = [
    {
        "message": "may27_cycle01_finding_nemo_latest",
        "kernel": "ayomide2000/finding-nemo",
        "version": "11",
        "file": "submission.zip",
        "reason": "latest public Kien/SVD-style output; untested after May26 10:35 update",
    },
    {
        "message": "may27_cycle02_replay_data_086",
        "kernel": "mohamedamr992/nemotron-replay-data-0-86",
        "version": "4",
        "file": "submission.zip",
        "reason": "public replay-data output titled 0.86; not yet tested locally",
    },
    {
        "message": "may27_cycle03_unsloth_4hours",
        "kernel": "chogunter/training-with-unsloth-4hours",
        "version": "3",
        "file": "submission.zip",
        "reason": "fresh 3.24GB trained adapter output; possible non-Kien direction",
    },
    {
        "message": "may27_cycle04_kien_anchor_086",
        "kernel": "beicicc/nemotron-direct-kien-tinker-anchor-may26",
        "version": "1",
        "file": "submission.zip",
        "reason": "known 0.86 control anchor; preserves a high baseline in today's five",
    },
    {
        "message": "may27_cycle05_atlas_public",
        "kernel": "habanwer/nemotron-atlas",
        "version": "49",
        "file": "submission.zip",
        "reason": "novel ATLAS/trace direction; small adapter but different from blind public sweeps",
    },
    {
        "message": "may27_cycle06_backtracking_v20_fallback",
        "kernel": "beicicc/nemotron-direct-backtracking-v20-c5",
        "version": "1",
        "file": "submission.zip",
        "reason": "validated fallback; scored 0.85",
    },
    {
        "message": "may27_cycle07_huikang_v20_mirror_fallback",
        "kernel": "beicicc/nemotron-direct-huikang-v20-may26",
        "version": "2",
        "file": "submission.zip",
        "reason": "fallback Huikang/backtracking mirror",
    },
]


def status_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name or value).rsplit(".", 1)[-1]


def submission_to_dict(sub: Any) -> dict[str, Any]:
    date = getattr(sub, "date", None)
    return {
        "ref": getattr(sub, "ref", None),
        "date": date.isoformat(sep=" ") if date else "",
        "description": getattr(sub, "description", ""),
        "status": status_name(getattr(sub, "status", "")),
        "publicScore": getattr(sub, "public_score", ""),
        "privateScore": getattr(sub, "private_score", ""),
        "errorDescription": getattr(sub, "error_description", ""),
        "totalBytes": getattr(sub, "total_bytes", None),
        "url": getattr(sub, "url", ""),
    }


def query_submissions() -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            return [submission_to_dict(s) for s in kaggle.api.competition_submissions(COMPETITION)]
        except Exception as exc:
            last_exc = exc
            print(json.dumps({"event": "query_retry", "attempt": attempt, "error": repr(exc)}, sort_keys=True), flush=True)
            time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"submission query failed after retries: {last_exc!r}")


def max_daily() -> int | None:
    try:
        comps = kaggle.api.competitions_list(search=COMPETITION)
        for comp in comps:
            if getattr(comp, "url", "").endswith(COMPETITION) or getattr(comp, "ref", "").endswith(COMPETITION):
                return getattr(comp, "max_daily_submissions", None)
    except Exception as exc:
        print(json.dumps({"event": "max_daily_query_failed", "error": repr(exc)}, sort_keys=True), flush=True)
    return None


def is_today(row: dict[str, Any]) -> bool:
    return str(row.get("date", "")).startswith(TODAY_UTC.isoformat())


def summarize(submissions: list[dict[str, Any]], daily_limit: int | None, state: dict[str, Any]) -> dict[str, Any]:
    today = [row for row in submissions if is_today(row)]
    complete = [row for row in today if row["status"] == "COMPLETE"]
    error = [row for row in today if row["status"] == "ERROR"]
    pending = [row for row in today if row["status"] == "PENDING"]
    non_error = [row for row in today if row["status"] != "ERROR"]
    scores = [float(row["publicScore"]) for row in complete if row.get("publicScore")]
    return {
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "todayUtc": TODAY_UTC.isoformat(),
        "competition": COMPETITION,
        "targetComplete": TARGET_COMPLETE,
        "maxDailySubmissions": daily_limit,
        "todayCompleteCount": len(complete),
        "todayErrorCount": len(error),
        "todayPendingCount": len(pending),
        "todayNonErrorCount": len(non_error),
        "remainingEstimateExcludingErrors": None if daily_limit is None else max(0, daily_limit - len(non_error)),
        "bestTodayPublicScore": max(scores or [0.0]),
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
    proc = subprocess.run(cmd, cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)
    return proc


def choose_candidate(submissions: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, str] | None:
    seen = {row["description"] for row in submissions}
    failed = set(state.setdefault("failedSubmitRequests", []))
    for cand in CANDIDATES:
        if cand["message"] not in seen and cand["message"] not in failed:
            return cand
    return None


def submit(cand: dict[str, str], state: dict[str, Any]) -> bool:
    proc = run_cli(
        [
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
        ],
        check=False,
    )
    event = {
        "atUtc": datetime.now(timezone.utc).isoformat(),
        "candidate": cand,
        "returncode": proc.returncode,
        "stdoutTail": proc.stdout[-4000:],
    }
    state.setdefault("submitEvents", []).append(event)
    if proc.returncode == 0:
        return True
    if "daily" in proc.stdout.lower() or "maximum" in proc.stdout.lower():
        state["stoppedReason"] = "daily limit reached"
        raise RuntimeError(proc.stdout)
    state.setdefault("failedSubmitRequests", []).append(cand["message"])
    return False


def main() -> None:
    daily_limit = max_daily()
    state: dict[str, Any] = {"startedAtUtc": datetime.now(timezone.utc).isoformat()}
    print(
        json.dumps(
            {"event": "start", "todayUtc": TODAY_UTC.isoformat(), "targetComplete": TARGET_COMPLETE, "maxDailySubmissions": daily_limit},
            sort_keys=True,
        ),
        flush=True,
    )
    while True:
        try:
            submissions = query_submissions()
        except Exception as exc:
            state["lastQueryError"] = {"atUtc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}
            write_report({"updatedAtUtc": datetime.now(timezone.utc).isoformat(), "state": state})
            time.sleep(POLL_SECONDS)
            continue
        report = summarize(submissions, daily_limit, state)
        write_report(report)
        print(
            json.dumps(
                {
                    "event": "poll",
                    "updatedAtUtc": report["updatedAtUtc"],
                    "complete": report["todayCompleteCount"],
                    "pending": report["todayPendingCount"],
                    "error": report["todayErrorCount"],
                    "remaining": report["remainingEstimateExcludingErrors"],
                    "best": report["bestTodayPublicScore"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if report["todayCompleteCount"] >= TARGET_COMPLETE:
            state["stoppedReason"] = "target complete submissions reached"
            write_report(summarize(query_submissions(), daily_limit, state))
            return
        if report["todayPendingCount"] > 0:
            time.sleep(POLL_SECONDS)
            continue
        cand = choose_candidate(submissions, state)
        if cand is None:
            state["stoppedReason"] = "candidate queue exhausted"
            write_report(summarize(submissions, daily_limit, state))
            raise RuntimeError("candidate queue exhausted before five complete submissions")
        print(json.dumps({"event": "submit", "candidate": cand}, sort_keys=True), flush=True)
        ok = submit(cand, state)
        write_report(summarize(query_submissions(), daily_limit, state))
        time.sleep(POLL_SECONDS if ok else 10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
