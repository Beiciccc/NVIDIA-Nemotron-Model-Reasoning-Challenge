#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import time

import requests
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi


def stream_download(
    url: str,
    target: Path,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 8,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")

    for attempt in range(1, max_retries + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as response:
                if response.status_code == 416:
                    tmp.replace(target)
                    print(f"stream_download_done={target.name}:{target.stat().st_size}", flush=True)
                    return
                if resume_from and response.status_code != 206:
                    print(
                        f"stream_download_resume_unsupported={target.name}:"
                        f"status={response.status_code}; restarting",
                        flush=True,
                    )
                    tmp.unlink(missing_ok=True)
                    resume_from = 0
                    mode = "wb"
                    response.close()
                    continue
                response.raise_for_status()
                written = resume_from
                with tmp.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        previous_bucket = written // (256 * 1024 * 1024)
                        written += len(chunk)
                        current_bucket = written // (256 * 1024 * 1024)
                        if current_bucket != previous_bucket:
                            print(f"stream_download_bytes={target.name}:{written}", flush=True)
                tmp.replace(target)
                print(f"stream_download_done={target.name}:{target.stat().st_size}", flush=True)
                return
        except Exception as exc:
            print(
                f"stream_download_retry={target.name}:attempt={attempt}:"
                f"bytes={tmp.stat().st_size if tmp.exists() else 0}:error={exc!r}",
                flush=True,
            )
            if attempt == max_retries:
                raise
            time.sleep(min(60, 2 ** attempt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kaggle kernel ref, e.g. owner/slug")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        default=None,
        help="Output file to download. Can be passed multiple times.",
    )
    args = parser.parse_args()

    owner, slug = args.kernel.split("/", 1)
    out_dir = Path(args.out_dir)
    wanted = list(dict.fromkeys(args.files or ["kaggle_rtx_train_report.json", "submission.zip"]))

    api = KaggleApi()
    api.authenticate()
    request = ApiListKernelSessionOutputRequest()
    request.user_name = owner
    request.kernel_slug = slug

    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

    available = {item.file_name: item.url for item in response.files}
    print("available_outputs=" + ",".join(sorted(available)), flush=True)

    missing = [name for name in wanted if name not in available]
    if missing:
        raise SystemExit(f"missing required output files: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if response.log:
        (out_dir / f"{slug}.log").write_text(response.log, encoding="utf-8")

    for name in wanted:
        print(f"stream_download_start={name}", flush=True)
        stream_download(available[name], out_dir / name)


if __name__ == "__main__":
    main()
