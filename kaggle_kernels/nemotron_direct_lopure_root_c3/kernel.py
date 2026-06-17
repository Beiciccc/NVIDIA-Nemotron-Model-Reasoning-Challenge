from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path


BASE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16"
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def find_file(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name), key=lambda p: p.stat().st_size, reverse=True)
    if not matches:
        raise FileNotFoundError(name)
    for match in matches:
        print(f"candidate={match}:{match.stat().st_size}")
    return matches[0]


src = find_file("submission.zip")
dst = WORKING / "submission.zip"
shutil.copyfile(src, dst)

with zipfile.ZipFile(dst) as zf:
    config_name = next(name for name in zf.namelist() if name.endswith("adapter_config.json"))
    cfg = json.loads(zf.read(config_name))

cfg["base_model_name_or_path"] = BASE_MODEL
cfg["inference_mode"] = True
config_path = WORKING / "adapter_config.json"
config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
subprocess.run(["zip", "-j", "-q", str(dst), str(config_path)], check=True)

with zipfile.ZipFile(dst) as zf:
    cfg = json.loads(zf.read(config_name))
    names = zf.namelist()
    print(
        json.dumps(
            {
                "output": str(dst),
                "size": dst.stat().st_size,
                "config": config_name,
                "base": cfg.get("base_model_name_or_path"),
                "inference": cfg.get("inference_mode"),
                "rank": cfg.get("r"),
                "tensor_files": sum(name.endswith(".safetensors") for name in names),
            },
            indent=2,
        )
    )
