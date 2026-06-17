from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def main() -> None:
    report: dict = {
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "env": {
            k: os.environ.get(k)
            for k in [
                "KAGGLE_URL_BASE",
                "KAGGLE_KERNEL_RUN_TYPE",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
            ]
        },
        "paths": {
            "/kaggle/input": sorted(str(p) for p in Path("/kaggle/input").glob("*"))[:50],
            "/kaggle/working": str(Path("/kaggle/working").resolve()),
        },
        "disk": {
            "/kaggle/input": shutil.disk_usage("/kaggle/input")._asdict(),
            "/kaggle/working": shutil.disk_usage("/kaggle/working")._asdict(),
        },
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvidia_smi_query": run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version,cuda_version",
                    "--format=csv,noheader",
                ]
            ),
        },
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        report["torch_error"] = repr(exc)

    out = Path("/kaggle/working/gpu_probe_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
