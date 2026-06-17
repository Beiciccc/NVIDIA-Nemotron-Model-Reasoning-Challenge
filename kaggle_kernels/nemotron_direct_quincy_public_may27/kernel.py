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
    matches = [
        path
        for path in INPUT_ROOT.rglob(name)
        if path.is_file() and path.stat().st_size > 0
    ]
    matches = sorted(matches, key=lambda path: path.stat().st_size, reverse=True)
    if not matches:
        raise FileNotFoundError(name)
    for match in matches:
        print(f"candidate={match}:{match.stat().st_size}")
    return matches[0]


cfg_path = find_file("adapter_config.json")
model_path = find_file("adapter_model.safetensors")

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
cfg["base_model_name_or_path"] = BASE_MODEL
cfg["inference_mode"] = True

(WORKING / "adapter_config.json").write_text(
    json.dumps(cfg, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
shutil.copyfile(model_path, WORKING / "adapter_model.safetensors")

subprocess.run(
    ["zip", "-q", "-0", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    cwd=WORKING,
    check=True,
)

with zipfile.ZipFile(WORKING / "submission.zip") as zf:
    out_cfg = json.loads(zf.read("adapter_config.json"))
    print(
        json.dumps(
            {
                "output": str(WORKING / "submission.zip"),
                "size": (WORKING / "submission.zip").stat().st_size,
                "base": out_cfg.get("base_model_name_or_path"),
                "inference": out_cfg.get("inference_mode"),
                "rank": out_cfg.get("r"),
                "alpha": out_cfg.get("lora_alpha"),
                "target_modules": out_cfg.get("target_modules"),
                "tensor_files": sum(name.endswith(".safetensors") for name in zf.namelist()),
            },
            indent=2,
        )
    )
