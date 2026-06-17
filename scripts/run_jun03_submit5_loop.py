#!/usr/bin/env python3
import json
import signal
import sys
import time
from datetime import datetime, timezone

import kaggle


COMP = "nvidia-nemotron-model-reasoning-challenge"
TODAY = "2026-06-03"
POLL_SECONDS = 120
TARGET_COMPLETE = 5

INITIAL_PENDING = "jun03_cycle03_afr1ste_tinker_adapter_guide_v4"

CANDIDATES = [
    {
        "cycle": 4,
        "ref": "siao112/thk-public-fork-2026-05-15-v14-kienngx",
        "version": 12,
        "file": "submission.zip",
        "message": "jun03_cycle04_siao_thk_public_kienngx_v12_gpu",
        "reason": "new 2026-06-03 saved version, COMPLETE with submission.zip and 0.86 metadata, despite v7 scoring 0.85 today",
    },
    {
        "cycle": 5,
        "ref": "mirzayasirabdullah07/nvidia-nemotron-model-reasoning-notebook",
        "version": 2,
        "file": "submission.zip",
        "message": "jun03_cycle05_mirza_reasoning_notebook_v2_gpu",
        "reason": "unsubmitted today, GPU-enabled public output with submission.zip and 0.86 metadata",
    },
    {
        "cycle": 6,
        "ref": "rohanrk1813/nvidia-comp",
        "version": 4,
        "file": "submission.zip",
        "message": "jun03_cycle06_rohan_nvidia_comp_v4_gpu_fallback",
        "reason": "fallback only: GPU-enabled high-visibility output with submission.zip, no local public score history",
    },
    {
        "cycle": 7,
        "ref": "mirzayasirabdullah07/best-nvidia-nemotron-model-notebook",
        "version": 14,
        "file": "submission.zip",
        "message": "jun03_cycle07_mirza_best_notebook_v14_fallback",
        "reason": "fallback only: unsubmitted today and has submission.zip/0.86 metadata, but latest metadata is no-GPU",
    },
    {
        "cycle": 8,
        "ref": "ayomide2000/finding-nemo",
        "version": 16,
        "file": "submission.zip",
        "message": "jun03_cycle08_finding_nemo_v16_gpu_fallback",
        "reason": "fallback only: GPU output with 0.86 metadata, but local history scored 0.85",
    },
    {
        "cycle": 9,
        "ref": "biohack44/nemotron-v62-d3-sparse-trust-finisher-attack",
        "version": 64,
        "file": "submission.zip",
        "message": "jun03_cycle09_biohack_sparse_trust_v64_gpu_fallback",
        "reason": "fallback only: GPU output with 0.86 metadata, but local history scored 0.85",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(event: str, **payload):
    print(json.dumps({"atUtc": now_utc(), "event": event, **payload}, ensure_ascii=False), flush=True)


def with_alarm(seconds, func):
    def handler(signum, frame):
        raise TimeoutError(f"operation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def status_name(s):
    return str(getattr(s, "name", s)).rsplit(".", 1)[-1]


def submissions_snapshot():
    def query():
        rows = []
        for s in kaggle.api.competition_submissions(COMP)[:30]:
            d = getattr(s, "date", None)
            rows.append(
                {
                    "date": d.isoformat(sep=" ") if d else "",
                    "description": getattr(s, "description", ""),
                    "status": status_name(getattr(s, "status", "")),
                    "publicScore": getattr(s, "public_score", ""),
                    "errorDescription": getattr(s, "error_description", ""),
                }
            )
        return rows

    rows = with_alarm(40, query)
    today = [r for r in rows if r["date"].startswith(TODAY)]
    complete = [r for r in today if r["status"] == "COMPLETE"]
    pending = [r for r in today if r["status"] == "PENDING"]
    errors = [r for r in today if r["status"] == "ERROR"]
    return {
        "rows": rows,
        "todayRows": today,
        "completeToday": complete,
        "pendingToday": pending,
        "errorToday": errors,
    }


def find_submission(snapshot, description):
    for row in snapshot["todayRows"]:
        if row["description"] == description:
            return row
    return None


def refresh_public_context():
    context = {"code": [], "topics": [], "leaderboard": []}

    def code_query():
        return kaggle.api.kernels_list(competition=COMP, page=1, page_size=12, sort_by="dateRun") or []

    try:
        kernels = with_alarm(45, code_query)
        for k in kernels[:12]:
            context["code"].append(
                {
                    "ref": getattr(k, "ref", ""),
                    "title": getattr(k, "title", ""),
                    "author": getattr(k, "author", ""),
                    "lastRunTime": str(getattr(k, "last_run_time", "")),
                    "votes": getattr(k, "total_votes", None),
                }
            )
    except Exception as exc:
        context["codeError"] = repr(exc)

    def topic_query():
        return kaggle.api.competition_list_topics(COMP, sort_by="recent", page=1)

    try:
        resp = with_alarm(45, topic_query)
        for t in getattr(resp, "topics", [])[:8]:
            d = t.to_dict() if hasattr(t, "to_dict") else {}
            context["topics"].append(
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "lastCommentPostDate": d.get("lastCommentPostDate"),
                    "commentCount": d.get("commentCount"),
                    "topicUrl": d.get("topicUrl"),
                }
            )
    except Exception as exc:
        context["topicsError"] = repr(exc)

    def lb_query():
        return kaggle.api.competition_leaderboard_view(COMP, page_size=18) or []

    try:
        leaders = with_alarm(45, lb_query)
        for x in leaders[:18]:
            context["leaderboard"].append(
                {
                    "rank": getattr(x, "rank", None),
                    "teamName": getattr(x, "team_name", ""),
                    "score": getattr(x, "score", ""),
                    "submissionDate": str(getattr(x, "submission_date", "")),
                }
            )
    except Exception as exc:
        context["leaderboardError"] = repr(exc)

    log("public_context_refreshed", context=context)


def submit_candidate(candidate):
    log("candidate_selected", candidate=candidate)

    def do_submit():
        return kaggle.api.competition_submit_code(
            file_name=candidate["file"],
            message=candidate["message"],
            competition=COMP,
            kernel=candidate["ref"],
            kernel_version=candidate["version"],
            quiet=False,
        )

    response = with_alarm(90, do_submit)
    log("submission_requested", message=candidate["message"], response=str(response))
    return candidate["message"]


def main():
    log("loop_start", pollSeconds=POLL_SECONDS, targetComplete=TARGET_COMPLETE, initialPending=INITIAL_PENDING)
    current_target = INITIAL_PENDING
    candidate_index = 0

    while True:
        snapshot = submissions_snapshot()
        complete_count = len(snapshot["completeToday"])
        log(
            "submissions_snapshot",
            completeToday=complete_count,
            remaining=max(0, TARGET_COMPLETE - complete_count),
            pendingToday=snapshot["pendingToday"],
            errorToday=snapshot["errorToday"],
            todayRows=snapshot["todayRows"][:8],
        )

        if complete_count >= TARGET_COMPLETE:
            log("target_complete", completeToday=complete_count)
            return 0

        target_row = find_submission(snapshot, current_target) if current_target else None
        if current_target and target_row and target_row["status"] == "PENDING":
            log("waiting_for_target", target=target_row, sleepSeconds=POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue

        if current_target and target_row:
            log("target_finished", target=target_row)
        elif current_target:
            log("target_missing", target=current_target)

        # If the prior target completed, complete_count may still be below 5 and we can submit the next candidate.
        refresh_public_context()

        submitted_messages = {r["description"] for r in snapshot["todayRows"]}
        next_candidate = None
        while candidate_index < len(CANDIDATES):
            cand = CANDIDATES[candidate_index]
            candidate_index += 1
            if cand["message"] in submitted_messages:
                log("candidate_skip_already_submitted", candidate=cand)
                continue
            next_candidate = cand
            break

        if not next_candidate:
            log("no_candidates_left", completeToday=complete_count)
            return 2

        try:
            current_target = submit_candidate(next_candidate)
        except Exception as exc:
            log("submit_request_failed", candidate=next_candidate, error=repr(exc))
            current_target = None
            time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
