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


cfg = json.loads(find_file("adapter_config.json").read_text(encoding="utf-8"))
cfg["base_model_name_or_path"] = BASE_MODEL
cfg["inference_mode"] = True
(WORKING / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
shutil.copyfile(find_file("adapter_model.safetensors"), WORKING / "adapter_model.safetensors")
subprocess.run(
    ["zip", "-q", "-9", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    cwd=WORKING,
    check=True,
)

dst = WORKING / "submission.zip"
with zipfile.ZipFile(dst) as zf:
    out_cfg = json.loads(zf.read("adapter_config.json"))
    print(
        json.dumps(
            {
                "output": str(dst),
                "size": dst.stat().st_size,
                "base": out_cfg.get("base_model_name_or_path"),
                "inference": out_cfg.get("inference_mode"),
                "rank": out_cfg.get("r"),
                "tensor_files": sum(name.endswith(".safetensors") for name in zf.namelist()),
            },
            indent=2,
        )
    )
