from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
REQUIRED = ("adapter_config.json", "adapter_model.safetensors")


def package_adapter(adapter_dir: Path, zip_path: Path) -> None:
    for name in REQUIRED:
        path = adapter_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(config.get("r", -1))
    if not 1 <= rank <= 32:
        raise ValueError(f"adapter rank must be in [1, 32], got {rank}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in REQUIRED:
            zf.write(adapter_dir / name, arcname=name)
    print(f"created {zip_path}")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    print(proc.stdout)
    return proc.stdout


def latest_status() -> str:
    code = f"""
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
subs=api.competition_submissions('{COMPETITION}')[:5]
for s in subs:
    print(getattr(s,'_date',None), getattr(s,'_status',None), getattr(s,'_public_score',None), getattr(s,'_description',None))
    err=getattr(s,'_error_description',None)
    if err:
        print('ERROR:', err)
"""
    return run([sys.executable, "-c", code])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path("/home/featurize/nemotron"))
    parser.add_argument("--message", required=True)
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()

    zip_path = args.project_dir / "submission.zip"
    kaggle_bin = shutil.which("kaggle") or str(args.project_dir / "envs/py312/bin/kaggle")
    package_adapter(args.adapter_dir, zip_path)
    run(
        [
            kaggle_bin,
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(zip_path),
            "-m",
            args.message,
        ],
        cwd=args.project_dir,
    )

    if args.poll:
        deadline = time.time() + args.poll_seconds
        while True:
            status = latest_status()
            (args.project_dir / "runs/submissions_latest.txt").write_text(status, encoding="utf-8")
            first_line = next((line for line in status.splitlines() if line.strip()), "")
            if "SubmissionStatus.COMPLETE" in first_line or "SubmissionStatus.ERROR" in first_line:
                break
            if time.time() >= deadline:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
