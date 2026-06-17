#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_zip")
    parser.add_argument("output_zip")
    parser.add_argument(
        "--base-model",
        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )
    parser.add_argument("--inference-mode", action="store_true")
    args = parser.parse_args()

    input_zip = Path(args.input_zip)
    output_zip = Path(args.output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    changed = False
    with zipfile.ZipFile(input_zip, "r") as src, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.endswith("adapter_config.json"):
                cfg = json.loads(data)
                cfg["base_model_name_or_path"] = args.base_model
                if args.inference_mode:
                    cfg["inference_mode"] = True
                data = (json.dumps(cfg, indent=2) + "\n").encode("utf-8")
                changed = True
            dst.writestr(info, data)

    if not changed:
        raise SystemExit("adapter_config.json not found")
    print(f"repacked={output_zip}:{output_zip.stat().st_size}")


if __name__ == "__main__":
    main()
