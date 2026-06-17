#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def model_file_url(model_version: str, file_path: str) -> str:
    owner_slug, model_slug, framework, instance_slug, version_number = model_version.split("/", 4)
    return (
        "https://www.kaggle.com/api/v1/models/"
        f"{owner_slug}/{model_slug}/{framework}/{instance_slug}/{version_number}/download/{file_path}"
    )


def kaggle_auth() -> tuple[str, str]:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    token = json.loads(token_path.read_text(encoding="utf-8"))
    return token["username"], token["key"]


def stream_download(url: str, target: Path, auth: tuple[str, str], chunk_size: int = 8 * 1024 * 1024) -> None:
    tmp = target.with_name(target.name + ".part")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, auth=auth, stream=True, allow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        print(f"download_resolved_url={response.url.split('?')[0]}", flush=True)
        print(f"download_content_length={total}", flush=True)
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                previous_bucket = written // (256 * 1024 * 1024)
                written += len(chunk)
                current_bucket = written // (256 * 1024 * 1024)
                if current_bucket != previous_bucket:
                    if total:
                        print(f"download_bytes={target.name}:{written}/{total}", flush=True)
                    else:
                        print(f"download_bytes={target.name}:{written}", flush=True)
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_version", help="owner/model/framework/instance/version")
    parser.add_argument("out_dir")
    parser.add_argument("--file", action="append", dest="files", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auth = kaggle_auth()

    for file_path in args.files:
        target = out_dir / Path(file_path).name
        print(f"download_start={file_path}->{target}", flush=True)
        stream_download(model_file_url(args.model_version, file_path), target, auth)
        print(f"download_done={target}:{target.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
