#!/usr/bin/env python3
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import kaggle


COMP = "nvidia-nemotron-model-reasoning-challenge"
TODAY = "2026-06-08"
POLL_SECONDS = 120
RATE_LIMIT_SLEEP_SECONDS = 300
TARGET_COMPLETE = 5
RECENTLY_SUBMITTED_GRACE_SECONDS = 600
STATE_FILE = Path("reports/2026-06-08_submit5_loop_state.json")

CANDIDATES = [
    {
        "cycle": 1,
        "ref": "wethepeople918/megatronthedecepticon",
        "file": "submission.zip",
        "message": "jun08_cycle01_wethepeople_megatron_probe",
        "reason": "Fresh 2026-06-08 high-vote public Code output with submission.zip plus many cryptarithm/adapter audit artifacts; best public upside probe.",
    },
    {
        "cycle": 2,
        "ref": "denglonghang/nemotron-tinker-noop-bridge-v20",
        "file": "submission.zip",
        "message": "jun08_cycle02_deng_noop_bridge_v20_probe",
        "reason": "Fresh 2026-06-08 Tinker/noop bridge output with submission.zip and adapter files; exploratory upside before falling back to known 0.86 anchors.",
    },
    {
        "cycle": 3,
        "ref": "debatreyabiswas/nemotroncomp-best0-86-solution-nvidia-under-5min",
        "file": "submission.zip",
        "message": "jun08_cycle03_debatreya_best086plus_fresh",
        "reason": "Fresh 2026-06-08 Debatreya best0.86+ output with submission.zip; reliable 0.86-family control after exploratory probes.",
    },
    {
        "cycle": 4,
        "ref": "mirzayasirabdullah07/best-nvidia-nemotron-0-86",
        "file": "submission.zip",
        "message": "jun08_cycle04_mirza_best_nemotron_086_fresh",
        "reason": "Fresh 2026-06-08 Mirza 0.86 public output with submission.zip; likely stable anchor, possibly updated packaging.",
    },
    {
        "cycle": 5,
        "ref": "debatreyabiswas/nemotroncomp-best-0-86-solution-nvidia-under-5min",
        "version": 1,
        "scriptVersionId": 324572187,
        "file": "submission.zip",
        "message": "jun08_cycle05_debatreya_best086_anchor",
        "reason": "Old stable Debatreya 0.86 anchor; kept as measured sanity control.",
    },
    {
        "cycle": 6,
        "ref": "mirzayasirabdullah07/best-nvidia-nemotron-notebook-0-86",
        "version": 16,
        "scriptVersionId": 324524084,
        "file": "submission.zip",
        "message": "jun08_cycle06_mirza_best086_anchor",
        "reason": "Old stable Mirza 0.86 anchor; reproduced repeatedly including 2026-06-07 repeat.",
    },
    {
        "cycle": 7,
        "ref": "mirzayasirabdullah07/best-nvidia-nemotron-notebook-0-86",
        "version": 16,
        "scriptVersionId": 324524084,
        "file": "submission.zip",
        "message": "jun08_cycle07_mirza_best086_repeat_anchor",
        "reason": "Repeat of the strongest accepted anchor, only used if fresh candidates are blocked or error.",
    },
    {
        "cycle": 8,
        "ref": "koushikrudra/nemotron-verify-finding-nemo",
        "file": "submission.zip",
        "message": "jun08_cycle08_koushik_verify_finding_nemo_probe",
        "reason": "Verification-flavored finding-nemo output from 2026-06-07; lower priority because finding-nemo family weakened to 0.85 yesterday.",
    },
    {
        "cycle": 9,
        "ref": "ayomide2000/finding-nemo",
        "file": "submission.zip",
        "message": "jun08_cycle09_ayomide_finding_nemo_demoted",
        "reason": "Finding-nemo-family fallback; 2026-06-06 result was only 0.85.",
    },
    {
        "cycle": 10,
        "ref": "danielsleiman/finding-nemo",
        "version": 1,
        "scriptVersionId": 324380050,
        "file": "submission.zip",
        "message": "jun08_cycle10_daniel_finding_nemo_demoted",
        "reason": "Demoted fallback: known 0.86 on 2026-06-05 but 0.85 on 2026-06-07.",
    },
    {
        "cycle": 11,
        "ref": "afr1ste/nemotron-0-86-tinker-adapter-guide",
        "version": 4,
        "scriptVersionId": 323628912,
        "file": "submission.zip",
        "message": "jun08_cycle11_afr1ste_tinker_demoted",
        "reason": "Demoted fallback: previously 0.86-capable but returned 0.85 on 2026-06-07.",
    },
    {
        "cycle": 12,
        "ref": "kuangyicheng/nemotron-087-training",
        "file": "submission.zip",
        "message": "jun08_cycle12_kuang_087_training_demoted",
        "reason": "Demoted last-resort candidate: 2026-06-08 run advertises 0.87, but 2026-06-06 v91 scored only 0.62.",
    },
    {
        "cycle": 13,
        "ref": "habanwer/nemotron-atlas",
        "file": "submission.zip",
        "message": "jun08_cycle13_habanwer_atlas_last_resort",
        "reason": "Last-resort candidate: fresh run has zip files, but historical signals were weak around the 0.61 range.",
    },
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def log(event, **payload):
    print(json.dumps({"atUtc": now_utc(), "event": event, **payload}, ensure_ascii=False), flush=True)


def load_state():
    if not STATE_FILE.exists():
        return {"failedMessages": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as exc:
        log("state_load_failed", path=str(STATE_FILE), error=repr(exc))
        return {"failedMessages": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def mark_failed(state, message, error):
    failed = set(state.get("failedMessages", []))
    failed.add(message)
    state["failedMessages"] = sorted(failed)
    failures = state.setdefault("failures", [])
    failures.append({"atUtc": now_utc(), "message": message, "error": error})
    save_state(state)


def is_rate_limit(exc):
    text = repr(exc)
    return "429" in text or "Too Many Requests" in text


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


def status_name(status):
    return str(getattr(status, "name", status)).rsplit(".", 1)[-1]


def submissions_snapshot():
    def query():
        rows = []
        for s in kaggle.api.competition_submissions(COMP)[:90]:
            d = getattr(s, "date", None)
            rows.append(
                {
                    "ref": getattr(s, "ref", ""),
                    "fileName": getattr(s, "file_name", ""),
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


def safe_snapshot():
    while True:
        try:
            return submissions_snapshot()
        except Exception as exc:
            sleep = RATE_LIMIT_SLEEP_SECONDS if is_rate_limit(exc) else POLL_SECONDS
            log("snapshot_failed", error=repr(exc), sleepSeconds=sleep)
            time.sleep(sleep)


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
        for t in getattr(resp, "topics", [])[:12]:
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

    try:
        leaders = with_alarm(45, lambda: kaggle.api.competition_leaderboard_view(COMP, page_size=20) or [])
        for x in leaders[:20]:
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
    log("loop_start", date=TODAY, pollSeconds=POLL_SECONDS, targetComplete=TARGET_COMPLETE)
    state = load_state()
    current_target = None
    current_target_requested_at = None
    candidate_index = 0

    while True:
        snapshot = safe_snapshot()
        complete_count = len(snapshot["completeToday"])
        log(
            "submissions_snapshot",
            completeToday=complete_count,
            remaining=max(0, TARGET_COMPLETE - complete_count),
            pendingToday=snapshot["pendingToday"],
            errorToday=snapshot["errorToday"],
            failedMessages=state.get("failedMessages", []),
            todayRows=snapshot["todayRows"][:15],
        )

        if complete_count >= TARGET_COMPLETE:
            log("target_complete", completeToday=complete_count)
            return 0

        if not current_target and snapshot["pendingToday"]:
            current_target = snapshot["pendingToday"][0]["description"]
            current_target_requested_at = time.time()
            log("adopt_pending_target", target=snapshot["pendingToday"][0])

        target_row = find_submission(snapshot, current_target) if current_target else None
        if current_target and target_row and target_row["status"] == "PENDING":
            log("waiting_for_target", target=target_row, sleepSeconds=POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue

        if current_target and target_row:
            log("target_finished", target=target_row)
            current_target = None
            current_target_requested_at = None
        elif current_target:
            age = time.time() - current_target_requested_at if current_target_requested_at else 0
            if age < RECENTLY_SUBMITTED_GRACE_SECONDS:
                log("target_not_visible_yet", target=current_target, ageSeconds=round(age, 1), sleepSeconds=POLL_SECONDS)
                time.sleep(POLL_SECONDS)
                continue
            log("target_missing_after_grace", target=current_target, ageSeconds=round(age, 1))
            current_target = None
            current_target_requested_at = None

        refresh_public_context()

        submitted_messages = {r["description"] for r in snapshot["todayRows"]}
        failed_messages = set(state.get("failedMessages", []))
        next_candidate = None
        while candidate_index < len(CANDIDATES):
            cand = CANDIDATES[candidate_index]
            candidate_index += 1
            if cand["message"] in submitted_messages:
                log("candidate_skip_already_submitted", candidate=cand)
                continue
            if cand["message"] in failed_messages:
                log("candidate_skip_failed_state", candidate=cand)
                continue
            next_candidate = cand
            break

        if not next_candidate:
            log("no_candidates_left", completeToday=complete_count)
            return 2

        try:
            current_target = submit_candidate(next_candidate)
            current_target_requested_at = time.time()
        except Exception as exc:
            if is_rate_limit(exc):
                candidate_index -= 1
                log("submit_rate_limited_retry_same_candidate", candidate=next_candidate, error=repr(exc), sleepSeconds=RATE_LIMIT_SLEEP_SECONDS)
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            else:
                log("submit_request_failed", candidate=next_candidate, error=repr(exc))
                mark_failed(state, next_candidate["message"], repr(exc))
                current_target = None
                current_target_requested_at = None
                time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
