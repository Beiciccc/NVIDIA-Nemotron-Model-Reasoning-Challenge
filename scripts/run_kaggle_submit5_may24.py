#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
KAGGLE_BIN = Path("/Users/beici/Library/Python/3.9/bin/kaggle")
COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
KERNEL_REF = "beicicc/nemotron-rtx6000-submit5-may24"
KERNEL_DIR = PROJECT / "kaggle_kernels/nemotron_rtx6000_submit5_may24"
CONFIG_PATH = KERNEL_DIR / "variant_config.json"
OUT_ROOT = PROJECT / "outputs/2026-05-24_submit5_kaggle_rtx"
LOG_ROOT = PROJECT / "logs/2026-05-24/submit5_kaggle_rtx"


CANDIDATES = [
    {
        "variant_id": "cycle01_cryptarithm16_lr5e7",
        "seed": 461,
        "train_limit": 512,
        "max_steps": 16,
        "max_length": 512,
        "learning_rate": 5e-7,
        "grad_accum": 1,
        "families": ["equation_symbolic"],
        "categories": ["cryptarithm_deduce", "cryptarithm_guess"],
        "eval_rows_per_family": 8,
        "scheduler": "constant",
    },
    {
        "variant_id": "cycle02_equation_all32_lr8e7",
        "seed": 462,
        "train_limit": 768,
        "max_steps": 32,
        "max_length": 512,
        "learning_rate": 8e-7,
        "grad_accum": 1,
        "families": ["equation_numeric", "equation_symbolic"],
        "categories": [],
        "eval_rows_per_family": 8,
        "scheduler": "constant",
    },
    {
        "variant_id": "cycle03_numeric32_lr1e6",
        "seed": 463,
        "train_limit": 640,
        "max_steps": 32,
        "max_length": 512,
        "learning_rate": 1e-6,
        "grad_accum": 1,
        "families": ["equation_numeric"],
        "categories": ["equation_numeric_deduce", "equation_numeric_guess"],
        "eval_rows_per_family": 8,
        "scheduler": "constant",
    },
    {
        "variant_id": "cycle04_hardmix48_lr7e7",
        "seed": 464,
        "train_limit": 1024,
        "max_steps": 48,
        "max_length": 512,
        "learning_rate": 7e-7,
        "grad_accum": 1,
        "families": ["equation_numeric", "equation_symbolic", "bit_manipulation"],
        "categories": [],
        "eval_rows_per_family": 8,
        "scheduler": "constant",
    },
    {
        "variant_id": "cycle05_balanced_format24_lr5e7",
        "seed": 465,
        "train_limit": 1024,
        "max_steps": 24,
        "max_length": 512,
        "learning_rate": 5e-7,
        "grad_accum": 1,
        "families": [],
        "categories": [],
        "eval_rows_per_family": 8,
        "scheduler": "constant",
    },
]


def run(cmd: list[str], *, cwd: Path = PROJECT, env: dict | None = None, timeout: int | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout)
    return proc.stdout


def try_run(cmd: list[str], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = run(cmd)
    except Exception as exc:
        text = f"ERROR: {exc!r}\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def write_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshot_public_signals(cycle: int) -> dict:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = LOG_ROOT / f"cycle{cycle:02d}"
    submissions = try_run(
        [str(KAGGLE_BIN), "competitions", "submissions", COMPETITION, "-v"],
        prefix.with_name(prefix.name + "_submissions_before.csv"),
    )
    try_run(
        [str(KAGGLE_BIN), "competitions", "leaderboard", COMPETITION, "--show"],
        prefix.with_name(prefix.name + "_leaderboard_before.txt"),
    )
    try_run(
        [str(KAGGLE_BIN), "kernels", "list", "--competition", COMPETITION, "--sort-by", "dateRun", "--page-size", "20"],
        prefix.with_name(prefix.name + "_latest_kernels_before.txt"),
    )
    try_run(
        [str(KAGGLE_BIN), "datasets", "list", "--search", "nemotron", "--sort-by", "updated", "--max-size", "20"],
        prefix.with_name(prefix.name + "_latest_datasets_before.txt"),
    )
    today_used = count_today_submissions(submissions)
    return {"today_used": today_used, "remaining_estimate": max(0, 5 - today_used)}


def count_today_submissions(csv_text: str) -> int:
    rows = list(csv.DictReader(csv_text.splitlines()))
    return sum(1 for row in rows if str(row.get("date", "")).startswith("2026-05-24"))


def wait_kernel_complete(cycle_dir: Path, poll_seconds: int = 90, timeout_seconds: int = 6 * 3600) -> str:
    deadline = time.time() + timeout_seconds
    status_path = cycle_dir / "kernel_status.log"
    seen = []
    while True:
        text = run([str(KAGGLE_BIN), "kernels", "status", KERNEL_REF])
        seen.append(f"\n[{datetime.utcnow().isoformat()}Z]\n{text}")
        status_path.write_text("".join(seen), encoding="utf-8")
        if "COMPLETE" in text:
            return text
        if "ERROR" in text or "CANCELLED" in text or "FAILED" in text:
            raise RuntimeError(text)
        if time.time() >= deadline:
            raise TimeoutError(f"kernel timed out after {timeout_seconds}s")
        time.sleep(poll_seconds)


def download_outputs(cycle_dir: Path) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(PROJECT / "scripts/download_kaggle_kernel_output.py"),
            KERNEL_REF,
            "--pattern",
            "submission\\.zip$",
            "--out-dir",
            str(cycle_dir),
            "--force",
            "--workers",
            "8",
            "--report-every",
            "30",
        ],
        timeout=None,
    )
    run(
        [
            sys.executable,
            str(PROJECT / "scripts/download_kaggle_kernel_output.py"),
            KERNEL_REF,
            "--pattern",
            "submit5_report\\.json$|variant_config_used\\.json$",
            "--out-dir",
            str(cycle_dir),
            "--force",
            "--workers",
            "1",
            "--report-every",
            "30",
        ],
        timeout=None,
    )


def validate_zip(cycle_dir: Path) -> dict:
    zip_path = cycle_dir / "submission.zip"
    text = run([sys.executable, str(PROJECT / "scripts/validate_adapter_submission.py"), str(zip_path)])
    (cycle_dir / "validation.json").write_text(text, encoding="utf-8")
    return json.loads(text)


def submit_and_wait(cycle: int, cfg: dict, cycle_dir: Path, poll_seconds: int = 90, timeout_seconds: int = 2400) -> dict:
    msg = f"20260524_{cfg['variant_id']}"
    zip_path = cycle_dir / "submission.zip"
    submit_out = run(
        [
            str(KAGGLE_BIN),
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(zip_path),
            "-m",
            msg,
        ],
        timeout=None,
    )
    (cycle_dir / "submit_stdout.txt").write_text(submit_out, encoding="utf-8")
    deadline = time.time() + timeout_seconds
    history = []
    while True:
        text = run([str(KAGGLE_BIN), "competitions", "submissions", COMPETITION, "-v"])
        history.append(f"\n[{datetime.utcnow().isoformat()}Z]\n{text}")
        (cycle_dir / "submissions_poll.csvlog").write_text("".join(history), encoding="utf-8")
        row = latest_submission_for_message(text, msg)
        if row and ("COMPLETE" in row.get("status", "") or "ERROR" in row.get("status", "")):
            result = {"cycle": cycle, "message": msg, "row": row}
            (cycle_dir / "submission_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return result
        if time.time() >= deadline:
            result = {"cycle": cycle, "message": msg, "row": row, "timeout": True}
            (cycle_dir / "submission_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return result
        time.sleep(poll_seconds)


def latest_submission_for_message(csv_text: str, msg: str) -> dict | None:
    rows = list(csv.DictReader(csv_text.splitlines()))
    for row in rows:
        if row.get("description") == msg:
            return row
    return rows[0] if rows else None


def push_kernel() -> None:
    env = dict(os.environ)
    if "REMOTE_PASS" not in env:
        raise RuntimeError("REMOTE_PASS is required for remote Kaggle CLI push")
    run([str(PROJECT / "scripts/kaggle_remote_kernel_push.sh"), str(KERNEL_DIR)], env=env, timeout=None)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for idx, cfg in enumerate(CANDIDATES, start=1):
        cycle_dir = OUT_ROOT / f"cycle{idx:02d}_{cfg['variant_id']}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        signals = snapshot_public_signals(idx)
        if signals["remaining_estimate"] <= 0:
            raise RuntimeError(f"no estimated submissions remaining before cycle {idx}: {signals}")
        (cycle_dir / "public_signals_before.json").write_text(json.dumps(signals, indent=2, sort_keys=True) + "\n")
        write_config(cfg)
        push_kernel()
        wait_kernel_complete(cycle_dir)
        download_outputs(cycle_dir)
        validation = validate_zip(cycle_dir)
        result = submit_and_wait(idx, cfg, cycle_dir)
        item = {
            "cycle": idx,
            "variant_id": cfg["variant_id"],
            "signals_before": signals,
            "validation": validation,
            "submission": result,
        }
        summary.append(item)
        (OUT_ROOT / "summary_partial.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
