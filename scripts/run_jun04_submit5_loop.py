#!/usr/bin/env python3
import json
import signal
import sys
import time
from datetime import datetime, timezone

import kaggle


COMP = "nvidia-nemotron-model-reasoning-challenge"
TODAY = "2026-06-04"
POLL_SECONDS = 120
TARGET_COMPLETE = 5

INITIAL_PENDING = "jun04_cycle01_siao_kienngx_fork_v18_gpu"

CANDIDATES = [
    {
        "cycle": 2,
        "ref": "mirzayasirabdullah07/best-nvidia-nemotron-notebook",
        "version": 15,
        "file": "submission.zip",
        "message": "jun04_cycle02_mirza_best_nemotron_notebook_v15",
        "reason": "Dalton scan: v15 COMPLETE with submission.zip and 0.86 metadata; not locally disproven yet",
    },
    {
        "cycle": 3,
        "ref": "ayomide2000/finding-nemo",
        "version": 17,
        "file": "submission.zip",
        "message": "jun04_cycle03_finding_nemo_v17_gpu",
        "reason": "fresh 2026-06-03/04 GPU output with submission.zip; v16 was 0.85, v17 is a new upside probe",
    },
    {
        "cycle": 4,
        "ref": "mirzayasirabdullah07/nvidia-nemotron-model-reasoning-notebook",
        "version": 2,
        "file": "submission.zip",
        "message": "jun04_cycle04_mirza_reasoning_notebook_v2_gpu",
        "reason": "known GPU-enabled output; scored 0.86 on 2026-06-03 and is a stable anchor",
    },
    {
        "cycle": 5,
        "ref": "afr1ste/nemotron-0-86-tinker-adapter-guide",
        "version": 4,
        "file": "submission.zip",
        "message": "jun04_cycle05_afr1ste_tinker_adapter_guide_v4",
        "reason": "0.86 metadata; scored 0.86 on 2026-06-02 and 0.85 on 2026-06-03, still better than low-score new experiments",
    },
    {
        "cycle": 6,
        "ref": "wethepeople918/steinifrank",
        "version": 6,
        "file": "submission.zip",
        "message": "jun04_cycle06_wethepeople_steinifrank_v6_gpu_fallback",
        "reason": "fallback/high-variance probe: new SteinIFrank fork has submission.zip, but old SteinIFrank lines were risky",
    },
    {
        "cycle": 7,
        "ref": "siao112/thk-public-fork-2026-05-15-v14-kienngx",
        "version": 12,
        "file": "submission.zip",
        "message": "jun04_cycle07_siao_thk_public_kienngx_v12_fallback",
        "reason": "fallback: valid zip but scored 0.85 on 2026-06-03",
    },
    {
        "cycle": 8,
        "ref": "rohanrk1813/nvidia-comp",
        "version": 4,
        "file": "submission.zip",
        "message": "jun04_cycle08_rohan_nvidia_comp_v4_fallback",
        "reason": "fallback: GPU-enabled high-visibility output with submission.zip, no recent local score",
    },
    {
        "cycle": 9,
        "ref": "biohack44/nemotron-v62-d3-sparse-trust-finisher-attack",
        "version": 64,
        "file": "submission.zip",
        "message": "jun04_cycle09_biohack_sparse_trust_v64_fallback",
        "reason": "fallback: valid GPU output, but previous local submission scored 0.85",
    },
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def log(event, **payload):
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
        for s in kaggle.api.competition_submissions(COMP)[:40]:
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

    rows = with_alarm(45, query)
    today = [r for r in rows if r["date"].startswith(TODAY)]
    return {
        "todayRows": today,
        "completeToday": [r for r in today if r["status"] == "COMPLETE"],
        "pendingToday": [r for r in today if r["status"] == "PENDING"],
        "errorToday": [r for r in today if r["status"] == "ERROR"],
    }


def find_submission(snapshot, description):
    for row in snapshot["todayRows"]:
        if row["description"] == description:
            return row
    return None


def refresh_public_context():
    context = {"code": [], "topics": [], "leaderboard": []}
    try:
        kernels = with_alarm(
            45,
            lambda: kaggle.api.kernels_list(
                competition=COMP, page=1, page_size=12, sort_by="dateRun"
            )
            or [],
        )
        for k in kernels[:12]:
            context["code"].append(
                {
                    "ref": getattr(k, "ref", ""),
                    "title": getattr(k, "title", ""),
                    "lastRunTime": str(getattr(k, "last_run_time", "")),
                    "votes": getattr(k, "total_votes", None),
                }
            )
    except Exception as exc:
        context["codeError"] = repr(exc)

    try:
        resp = with_alarm(45, lambda: kaggle.api.competition_list_topics(COMP, sort_by="recent", page=1))
        for t in getattr(resp, "topics", [])[:8]:
            d = t.to_dict() if hasattr(t, "to_dict") else {}
            context["topics"].append(
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "lastCommentPostDate": d.get("lastCommentPostDate"),
                    "topicUrl": d.get("topicUrl"),
                }
            )
    except Exception as exc:
        context["topicsError"] = repr(exc)

    try:
        leaders = with_alarm(45, lambda: kaggle.api.competition_leaderboard_view(COMP, page_size=18) or [])
        for x in leaders[:18]:
            context["leaderboard"].append(
                {
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
        kwargs = dict(
            file_name=candidate["file"],
            message=candidate["message"],
            competition=COMP,
            kernel=candidate["ref"],
            quiet=False,
        )
        if candidate.get("version"):
            kwargs["kernel_version"] = candidate["version"]
        return kaggle.api.competition_submit_code(**kwargs)

    response = with_alarm(120, do_submit)
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
            todayRows=snapshot["todayRows"][:10],
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
