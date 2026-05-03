from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    base_path = args.base / "adapter_model.safetensors"
    target_path = args.target / "adapter_model.safetensors"
    base_state = load_file(str(base_path), device="cpu")
    target_state = load_file(str(target_path), device="cpu")
    if set(base_state) != set(target_state):
        missing = sorted(set(base_state) ^ set(target_state))[:20]
        raise ValueError(f"adapter key mismatch, first differing keys: {missing}")

    out_state = {}
    for key, base_value in base_state.items():
        target_value = target_state[key]
        if torch.is_floating_point(base_value):
            out_state[key] = base_value + args.alpha * (target_value - base_value)
        else:
            out_state[key] = target_value

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(out_state, str(args.output / "adapter_model.safetensors"))
    shutil.copy2(args.target / "adapter_config.json", args.output / "adapter_config.json")
    metadata = {
        "base": str(args.base),
        "target": str(args.target),
        "alpha": args.alpha,
    }
    (args.output / "interpolation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
