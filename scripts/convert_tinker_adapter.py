from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import safe_open, save_file


OLD_PREFIX = "base_model.model.model.layers."
NEW_PREFIX = "base_model.model.backbone.layers."


def convert_key(key: str) -> str:
    if key.startswith(OLD_PREFIX):
        return NEW_PREFIX + key[len(OLD_PREFIX) :]
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    tensors = {}
    source_model = args.source / "adapter_model.safetensors"
    with safe_open(source_model, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            new_key = convert_key(key)
            if new_key in tensors:
                raise ValueError(f"duplicate converted key: {new_key}")
            tensors[new_key] = f.get_tensor(key)

    save_file(tensors, args.output / "adapter_model.safetensors", metadata=metadata)

    for path in args.source.iterdir():
        if path.name == "adapter_model.safetensors":
            continue
        dest = args.output / path.name
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)

    meta = {"converted_key_prefix": {"from": OLD_PREFIX, "to": NEW_PREFIX}}
    (args.output / "conversion_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    changed = sum(1 for key in tensors if key.startswith(NEW_PREFIX))
    print(f"converted tensors: {len(tensors)}")
    print(f"backbone-prefixed tensors: {changed}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
