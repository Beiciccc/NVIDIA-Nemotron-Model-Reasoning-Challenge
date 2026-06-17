#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi


def _request_type():
    module = sys.modules["kaggle.api.kaggle_api_extended"]
    return module.ApiListKernelSessionOutputRequest


def _kernel_files(kernel):
    owner, slug = kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    req_type = _request_type()
    with api.build_kaggle_client() as client:
        req = req_type()
        req.user_name = owner
        req.kernel_slug = slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(req)
    return list(response.files or [])


def _remote_size(url):
    headers = {"Range": "bytes=0-0"}
    response = requests.get(url, stream=True, headers=headers, timeout=(20, 60))
    try:
        response.raise_for_status()
        content_range = response.headers.get("content-range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
        length = response.headers.get("content-length")
        return int(length) if length else None
    finally:
        response.close()


def _fmt_bytes(value):
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(value)
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:.1f}{unit}"
        n /= 1024


def _fmt_eta(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _download_range(url, part_path, start, end, chunk_size):
    part_path = Path(part_path)
    existing = part_path.stat().st_size if part_path.exists() else 0
    expected = end - start + 1
    if existing == expected:
        return
    if existing > expected:
        part_path.unlink()
        existing = 0
    headers = {"Range": f"bytes={start + existing}-{end}"}
    with requests.get(url, stream=True, headers=headers, timeout=(20, 120)) as response:
        response.raise_for_status()
        with open(part_path, "ab") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    actual = part_path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"incomplete part {part_path}: got {actual}, expected {expected}")


def _download_parallel(url, dest, total, workers, chunk_size, report_every):
    dest = Path(dest)
    part_dir = dest.with_name(dest.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    spans = []
    block = (total + workers - 1) // workers
    for idx in range(workers):
        start = idx * block
        if start >= total:
            break
        end = min(total - 1, start + block - 1)
        spans.append((idx, start, end, part_dir / f"{idx:03d}.part"))

    print(f"parallel workers: {len(spans)}", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(spans)) as pool:
        futures = [
            pool.submit(_download_range, url, part_path, start, end, chunk_size)
            for _, start, end, part_path in spans
        ]
        while True:
            done = sum(1 for future in futures if future.done())
            downloaded = sum(path.stat().st_size for *_, path in spans if path.exists())
            now = time.time()
            elapsed = max(now - started, 1e-6)
            speed = downloaded / elapsed
            eta = (total - downloaded) / speed if speed > 0 else None
            print(
                f"progress: {_fmt_bytes(downloaded)} / {_fmt_bytes(total)} "
                f"({downloaded / total * 100:.1f}%), speed {_fmt_bytes(speed)}/s, "
                f"ETA {_fmt_eta(eta)}, parts {done}/{len(futures)}",
                flush=True,
            )
            if done == len(futures):
                break
            time.sleep(report_every)
        for future in as_completed(futures):
            future.result()

    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as out:
        for _, _, _, path in spans:
            with open(path, "rb") as in_file:
                while True:
                    chunk = in_file.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"assembled file size mismatch: {tmp.stat().st_size} != {total}")
    tmp.replace(dest)
    for _, _, _, path in spans:
        path.unlink(missing_ok=True)
    part_dir.rmdir()
    print(f"complete: {dest} ({_fmt_bytes(total)}) in {_fmt_eta(time.time() - started)}")


def download(url, dest, force=False, chunk_size=8 * 1024 * 1024, report_every=30, workers=1):
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    total = _remote_size(url)
    if dest.exists() and not force:
        local_size = dest.stat().st_size
        if total is None or local_size == total:
            print(f"already complete: {dest} ({_fmt_bytes(local_size)})")
            return

    if force:
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        part_dir = dest.with_name(dest.name + ".parts")
        if part_dir.exists():
            for child in part_dir.iterdir():
                child.unlink()
            part_dir.rmdir()

    downloaded = part.stat().st_size if part.exists() else 0
    headers = {}
    mode = "ab"
    if downloaded and total and downloaded < total:
        headers["Range"] = f"bytes={downloaded}-"
    elif downloaded and total and downloaded >= total:
        part.replace(dest)
        print(f"already complete: {dest} ({_fmt_bytes(total)})")
        return
    elif downloaded:
        part.unlink(missing_ok=True)
        downloaded = 0

    print(f"downloading: {dest}")
    print(f"remote size: {_fmt_bytes(total)}; resume from: {_fmt_bytes(downloaded)}")
    if workers > 1 and total is not None and downloaded == 0:
        _download_parallel(url, dest, total, workers, chunk_size, report_every)
        return
    start = time.time()
    last_report = start
    last_bytes = downloaded
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, headers=headers, timeout=(20, 120)) as response:
        response.raise_for_status()
        with open(part, mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= report_every:
                    interval_speed = (downloaded - last_bytes) / max(now - last_report, 1e-6)
                    avg_speed = (downloaded - (part.stat().st_size if False else 0)) / max(now - start, 1e-6)
                    speed = interval_speed if interval_speed > 0 else avg_speed
                    eta = ((total - downloaded) / speed) if total and speed > 0 else None
                    pct = f"{downloaded / total * 100:.1f}%" if total else "?"
                    print(
                        f"progress: {_fmt_bytes(downloaded)} / {_fmt_bytes(total)} "
                        f"({pct}), speed {_fmt_bytes(speed)}/s, ETA {_fmt_eta(eta)}",
                        flush=True,
                    )
                    last_report = now
                    last_bytes = downloaded

    if total is not None and downloaded != total:
        raise RuntimeError(f"incomplete download: got {downloaded}, expected {total}")
    part.replace(dest)
    elapsed = max(time.time() - start, 1e-6)
    print(f"complete: {dest} ({_fmt_bytes(downloaded)}) in {_fmt_eta(elapsed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("--pattern", default="submission\\.zip$")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-every", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    pattern = re.compile(args.pattern)
    matches = [f for f in _kernel_files(args.kernel) if pattern.search(f.file_name)]
    if not matches:
        raise SystemExit(f"no output files matched {args.pattern!r} for {args.kernel}")
    for item in matches:
        download(
            item.url,
            Path(args.out_dir) / item.file_name,
            force=args.force,
            report_every=args.report_every,
            workers=max(1, args.workers),
        )


if __name__ == "__main__":
    main()
