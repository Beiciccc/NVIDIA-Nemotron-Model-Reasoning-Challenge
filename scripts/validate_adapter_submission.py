#!/usr/bin/env python3
import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path


REQUIRED = ("adapter_config.json", "adapter_model.safetensors")


def _common_prefix(paths):
    split = [p.split("/")[:-1] for p in paths]
    if not split:
        return ""
    prefix = []
    for parts in zip(*split):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return "/".join(prefix) + ("/" if prefix else "")


def _find_required(names):
    root = {name: name for name in names if name in REQUIRED}
    if len(root) == len(REQUIRED):
        return root, ""
    candidates = [name for name in names if name.endswith(REQUIRED)]
    mapping = {}
    for req in REQUIRED:
        hits = [name for name in names if name.endswith("/" + req) or name == req]
        if len(hits) == 1:
            mapping[req] = hits[0]
    if len(mapping) == len(REQUIRED):
        return mapping, _common_prefix(list(mapping.values()))
    raise ValueError(f"missing required files; found matches: {candidates[:20]}")


def _read_safetensors_header(zf, name):
    with zf.open(name) as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("safetensors file is too small for header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len > 100_000_000:
            raise ValueError(f"unreasonable safetensors header length: {header_len}")
        header = handle.read(header_len)
        if len(header) != header_len:
            raise ValueError("truncated safetensors header")
        payload = json.loads(header)
        tensor_keys = [key for key in payload if key != "__metadata__"]
        if not tensor_keys:
            raise ValueError("safetensors header contains no tensor keys")
        return len(tensor_keys)


def validate(path, repack_to=None):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty zip: {path}")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"corrupt zip member: {bad}")
        names = [name for name in zf.namelist() if not name.endswith("/")]
        mapping, prefix = _find_required(names)
        config = json.loads(zf.read(mapping["adapter_config.json"]))
        rank = config.get("r")
        if rank is None:
            rank = max(config.get("rank_pattern", {"_": 0}).values())
        if int(rank) > 32:
            raise ValueError(f"LoRA rank exceeds competition limit: {rank}")
        info = zf.getinfo(mapping["adapter_model.safetensors"])
        if info.file_size < 1_000_000:
            raise ValueError(f"adapter_model.safetensors too small: {info.file_size}")
        tensor_count = _read_safetensors_header(zf, mapping["adapter_model.safetensors"])

        if repack_to:
            repack_to = Path(repack_to)
            repack_to.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(repack_to, "w", compression=zipfile.ZIP_STORED) as out:
                for req, src in mapping.items():
                    with zf.open(src) as in_file, out.open(req, "w", force_zip64=True) as out_file:
                        shutil.copyfileobj(in_file, out_file, length=8 * 1024 * 1024)
            path = repack_to

    result = {
        "path": str(path),
        "size": path.stat().st_size,
        "rank": int(rank),
        "prefix": prefix,
        "repacked": bool(repack_to),
        "tensor_count": tensor_count,
        "target_modules": config.get("target_modules"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--repack-to")
    args = parser.parse_args()
    validate(args.zip_path, args.repack_to)


if __name__ == "__main__":
    main()
