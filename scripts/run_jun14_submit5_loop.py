#!/usr/bin/env python3
import csv
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import kaggle


COMP = "nvidia-nemotron-model-reasoning-challenge"
TODAY = "2026-06-14"
TARGET_COMPLETE = 5
POLL_SECONDS = 300
RATE_LIMIT_SLEEP_SECONDS = 300
RECENTLY_SUBMITTED_GRACE_SECONDS = 900
FILE_UPLOAD_TIMEOUT_SECONDS = 7200

STATE_FILE = Path("reports/2026-06-14_submit5_loop_state.json")
RUN_DIR = Path("outputs/jun14_submit5_loop")

KIEN086_REBUILT_ZIP = Path("outputs/jun14_public_downloads/beicicc_kien_anchor/submission.zip")
SMALL_SMOKE_ZIP = Path("outputs/2026-05-26_submit5_cycle01_minyam_kien_v4/submission.zip")

CANDIDATES = [
    {
        "kind": "code",
        "ref": "nicholas33/nb153-nvidia-nemotron-inference",
        "file": "submission.zip",
        "message": "jun14_cycle00_nicholas_nb153_fresh_probe",
        "reason": "Fresh 2026-06-14 COMPLETE run; unknown score, quick public Code probe before anchors.",
    },
    {
        "kind": "code",
        "ref": "evgendvorkin/nemotron-3-nano-lora-adapter-submission",
        "file": "submission.zip",
        "message": "jun14_cycle01_evgendvorkin_latest_probe",
        "reason": "Fresh 2026-06-14 COMPLETE run with likely LoRA adapter output; quick Code probe only.",
    },
    {
        "kind": "code",
        "ref": "mirzayasirabdullah07/nvidia-nemotron-lora-adapter-model-safetensors",
        "file": "submission.zip",
        "message": "jun14_cycle02_mirza_safetensors_probe",
        "reason": "Fresh 2026-06-14 Mirza-family adapter model safetensors output; Mirza line historically reaches 0.86.",
    },
    {
        "kind": "code",
        "ref": "amit393/nvidia-nemotron-score-0-85",
        "file": "submission.zip",
        "message": "jun14_cycle03_amit_score085_probe",
        "reason": "Fresh 2026-06-14 COMPLETE public Code titled score 0.85; only a probe behind 0.86 candidates.",
    },
    {
        "kind": "code",
        "ref": "johnsonhk88/nvidia-nemotron-model-fine-tuning",
        "file": "submission.zip",
        "message": "jun14_cycle04_johnson_finetuning_probe",
        "reason": "Fresh 2026-06-14 COMPLETE training output; quick Code probe only.",
    },
    {
        "kind": "code",
        "ref": "gdataranger/nemotron-v0-14-sft-training-dgx-spark-gb10",
        "file": "submission.zip",
        "message": "jun14_cycle05_gdataranger_v014_probe",
        "reason": "Fresh 2026-06-14 SFT line; low-vote but conceptually different, quick probe only.",
    },
    {
        "kind": "code",
        "ref": "vngnguynhuy/refine",
        "file": "submission.zip",
        "message": "jun14_cycle06_vng_refine_probe",
        "reason": "Recent 2026-06-13 Refine output; high-vote public adapter probe.",
    },
    {
        "kind": "code",
        "ref": "nikitagajbhiye30/nvidia-cpu-00",
        "file": "submission.zip",
        "message": "jun14_cycle07_nikita_cpu00_probe",
        "reason": "Recent public output with many votes; unknown measured score, quick probe only.",
    },
    {
        "kind": "code",
        "ref": "beicicc/nemotron-direct-kien-tinker-anchor-may26",
        "file": "submission.zip",
        "message": "jun14_cycle08_private_kien_tinker_anchor",
        "reason": "Private kernel-output route for historical 0.86 Kien/Tinker anchor; likely 403 after 2026-06-09 but worth a fast probe.",
    },
    {
        "kind": "code",
        "ref": "beicicc/nemotron-direct-kien-tinker-anchor-may26",
        "file": "submission.zip",
        "message": "jun14_cycle09_private_kien_tinker_anchor_repeat",
        "reason": "Repeat private Kien 0.86 anchor probe before falling back to local rebuilt zip.",
    },
    {
        "kind": "file",
        "zip": KIEN086_REBUILT_ZIP,
        "message": "jun14_file_cycle10_kien086_rebuilt_anchor",
        "reason": "Rebuilt local Kien original 0.86 anchor from adapter files; strongest locally available fallback.",
    },
    {
        "kind": "file",
        "zip": KIEN086_REBUILT_ZIP,
        "message": "jun14_file_cycle11_kien086_rebuilt_anchor_repeat",
        "reason": "Repeat of rebuilt Kien 0.86 anchor; public score historically fluctuates 0.85-0.86.",
    },
    {
        "kind": "file",
        "zip": KIEN086_REBUILT_ZIP,
        "message": "jun14_file_cycle12_kien086_rebuilt_anchor_repeat2",
        "reason": "Repeat of rebuilt Kien 0.86 anchor after Code routes fail.",
    },
    {
        "kind": "file",
        "zip": KIEN086_REBUILT_ZIP,
        "message": "jun14_file_cycle13_kien086_rebuilt_anchor_repeat3",
        "reason": "Repeat of rebuilt Kien 0.86 anchor after Code routes fail.",
    },
    {
        "kind": "file",
        "zip": KIEN086_REBUILT_ZIP,
        "message": "jun14_file_cycle14_kien086_rebuilt_anchor_repeat4",
        "reason": "Final rebuilt Kien 0.86 anchor repeat before small low-score fallback.",
    },
    {
        "kind": "file",
        "zip": SMALL_SMOKE_ZIP,
        "message": "jun14_file_cycle15_minyam_small_upload_smoke",
        "reason": "Small 51MB fallback only if Kien rebuilt upload fails; historically 0.51-0.54.",
    },
    {
        "kind": "file",
        "zip": SMALL_SMOKE_ZIP,
        "message": "jun14_file_cycle16_minyam_small_upload_repeat",
        "reason": "Small fallback to complete requested count if higher-score routes are blocked.",
    },
    {
        "kind": "file",
        "zip": SMALL_SMOKE_ZIP,
        "message": "jun14_file_cycle17_minyam_small_upload_repeat2",
        "reason": "Small fallback to complete requested count if higher-score routes are blocked.",
    },
    {
        "kind": "file",
        "zip": SMALL_SMOKE_ZIP,
        "message": "jun14_file_cycle18_minyam_small_upload_repeat3",
        "reason": "Small fallback to complete requested count if higher-score routes are blocked.",
    },
    {
        "kind": "file",
        "zip": SMALL_SMOKE_ZIP,
        "message": "jun14_file_cycle19_minyam_small_upload_repeat4",
        "reason": "Small fallback to complete requested count if higher-score routes are blocked.",
    },
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def log(event, **payload):
    print(json.dumps({"atUtc": now_utc(), "event": event, **payload}, ensure_ascii=False), flush=True)


def load_state():
    if not STATE_FILE.exists():
        return {"failedMessages": [], "failures": [], "submittedMessages": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as exc:
        log("state_load_failed", path=str(STATE_FILE), error=repr(exc))
        return {"failedMessages": [], "failures": [], "submittedMessages": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def mark_failed(state, message, error, candidate=None):
    failed = set(state.get("failedMessages", []))
    failed.add(message)
    state["failedMessages"] = sorted(failed)
    state.setdefault("failures", []).append(
        {"atUtc": now_utc(), "message": message, "error": error, "candidate": clean_candidate(candidate)}
    )
    save_state(state)


def mark_requested(state, message, candidate):
    requested = set(state.get("submittedMessages", []))
    requested.add(message)
    state["submittedMessages"] = sorted(requested)
    state.setdefault("requests", []).append({"atUtc": now_utc(), "message": message, "candidate": clean_candidate(candidate)})
    save_state(state)


def clean_candidate(candidate):
    if candidate is None:
        return None
    return {key: str(value) if isinstance(value, Path) else value for key, value in candidate.items()}


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


def query_submissions_api():
    rows = []
    for s in kaggle.api.competition_submissions(COMP)[:100]:
        d = getattr(s, "date", None)
        rows.append(
            {
                "ref": str(getattr(s, "ref", "")),
                "fileName": getattr(s, "file_name", ""),
                "date": d.isoformat(sep=" ") if d else "",
                "description": getattr(s, "description", ""),
                "status": status_name(getattr(s, "status", "")),
                "publicScore": getattr(s, "public_score", ""),
                "privateScore": getattr(s, "private_score", ""),
                "errorDescription": getattr(s, "error_description", ""),
            }
        )
    return rows


def query_submissions_cli():
    proc = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    rows = []
    for row in csv.DictReader(proc.stdout.splitlines()):
        rows.append(
            {
                "ref": row.get("ref", ""),
                "fileName": row.get("fileName", ""),
                "date": row.get("date", ""),
                "description": row.get("description", ""),
                "status": row.get("status", "").rsplit(".", 1)[-1],
                "publicScore": row.get("publicScore", ""),
                "privateScore": row.get("privateScore", ""),
                "errorDescription": row.get("errorDescription", ""),
            }
        )
    return rows


def submissions_snapshot():
    try:
        rows = with_alarm(60, query_submissions_api)
        source = "api"
    except Exception as api_exc:
        log("submissions_api_failed_try_cli", error=repr(api_exc))
        rows = query_submissions_cli()
        source = "cli"
    today = [r for r in rows if r["date"].startswith(TODAY)]
    return {
        "source": source,
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


def submit_code(candidate):
    def do_submit():
        return kaggle.api.competition_submit_code(
            file_name=candidate["file"],
            message=candidate["message"],
            competition=COMP,
            kernel=candidate["ref"],
            quiet=False,
        )

    response = with_alarm(180, do_submit)
    log("code_submission_requested", message=candidate["message"], ref=candidate["ref"], response=str(response))


def submit_file(candidate):
    zip_path = Path(candidate["zip"])
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing zip: {zip_path}")

    out_dir = RUN_DIR / candidate["message"]
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMP,
            "-f",
            str(zip_path),
            "-m",
            candidate["message"],
        ],
        text=True,
        capture_output=True,
        timeout=FILE_UPLOAD_TIMEOUT_SECONDS,
    )
    (out_dir / "submit_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "submit_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    log("file_submission_requested", message=candidate["message"], zip=str(zip_path), bytes=zip_path.stat().st_size)


def submit_candidate(candidate):
    log("candidate_selected", candidate=clean_candidate({**candidate, "zip": str(candidate.get("zip", ""))}))
    if candidate["kind"] == "code":
        submit_code(candidate)
    elif candidate["kind"] == "file":
        submit_file(candidate)
    else:
        raise ValueError(f"unknown candidate kind: {candidate['kind']}")


def find_submission(snapshot, description):
    for row in snapshot["todayRows"]:
        if row["description"] == description:
            return row
    return None


def candidate_visible_or_failed(snapshot, message):
    return any(row["description"] == message for row in snapshot["todayRows"])


def main():
    state = load_state()
    log("loop_start", date=TODAY, targetComplete=TARGET_COMPLETE, pollSeconds=POLL_SECONDS)
    current_target = None
    current_target_requested_at = None

    while True:
        snapshot = safe_snapshot()
        complete_count = len(snapshot["completeToday"])
        log(
            "submissions_snapshot",
            source=snapshot["source"],
            completeToday=complete_count,
            remaining=max(0, TARGET_COMPLETE - complete_count),
            pendingToday=snapshot["pendingToday"],
            errorToday=snapshot["errorToday"],
            todayRows=snapshot["todayRows"][:15],
            failedMessages=state.get("failedMessages", []),
        )

        if complete_count >= TARGET_COMPLETE:
            log("target_complete", completeToday=complete_count)
            return 0

        if snapshot["pendingToday"]:
            current_target = snapshot["pendingToday"][0]["description"]
            current_target_requested_at = time.time()
            log("waiting_for_pending", pending=snapshot["pendingToday"], sleepSeconds=POLL_SECONDS)
            time.sleep(POLL_SECONDS)
            continue

        if current_target:
            row = find_submission(snapshot, current_target)
            if row:
                log("target_finished", target=row)
                current_target = None
                current_target_requested_at = None
            else:
                age = time.time() - current_target_requested_at if current_target_requested_at else 0
                if age < RECENTLY_SUBMITTED_GRACE_SECONDS:
                    log("target_not_visible_yet", target=current_target, ageSeconds=round(age, 1), sleepSeconds=POLL_SECONDS)
                    time.sleep(POLL_SECONDS)
                    continue
                log("target_missing_after_grace", target=current_target, ageSeconds=round(age, 1))
                current_target = None
                current_target_requested_at = None

        submitted_messages = {r["description"] for r in snapshot["todayRows"]}
        failed_messages = set(state.get("failedMessages", []))
        next_candidate = None
        for candidate in CANDIDATES:
            if candidate["message"] in submitted_messages:
                continue
            if candidate["message"] in failed_messages:
                continue
            if candidate["message"] in state.get("submittedMessages", []) and not candidate_visible_or_failed(snapshot, candidate["message"]):
                continue
            next_candidate = candidate
            break

        if not next_candidate:
            log("no_candidates_left", completeToday=complete_count)
            return 2

        try:
            submit_candidate(next_candidate)
            mark_requested(state, next_candidate["message"], {**next_candidate, "zip": str(next_candidate.get("zip", ""))})
            current_target = next_candidate["message"]
            current_target_requested_at = time.time()
        except Exception as exc:
            if is_rate_limit(exc):
                log("submit_rate_limited_retry_later", candidate=next_candidate, error=repr(exc), sleepSeconds=RATE_LIMIT_SLEEP_SECONDS)
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue
            log("submit_request_failed", candidate=clean_candidate(next_candidate), error=repr(exc))
            mark_failed(state, next_candidate["message"], repr(exc), {**next_candidate, "zip": str(next_candidate.get("zip", ""))})
            current_target = None
            current_target_requested_at = None
            time.sleep(30)
            continue

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
