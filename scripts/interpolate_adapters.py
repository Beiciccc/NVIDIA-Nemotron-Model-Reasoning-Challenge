from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def _tensor_keys(header: dict) -> list[str]:
    return [key for key in header.keys() if key != "__metadata__"]


def _canonical_key(key: str) -> str:
    return (
        key.replace("base_model.model.model.layers.", "base_model.model.backbone.layers.")
        .replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head.")
    )


def _match_keys(base_keys: list[str], target_keys: list[str]) -> dict[str, str]:
    base_by_canonical = {_canonical_key(key): key for key in base_keys}
    target_canonical = {_canonical_key(key) for key in target_keys}
    if set(base_by_canonical) != target_canonical:
        missing = sorted(set(base_by_canonical) ^ target_canonical)[:20]
        raise ValueError(f"adapter key mismatch after canonicalization, first differing keys: {missing}")
    return {target_key: base_by_canonical[_canonical_key(target_key)] for target_key in target_keys}


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _write_safetensors_f32_interpolation(
    base_path: Path,
    target_path: Path,
    output_path: Path,
    alpha: float,
    chunk_floats: int = 8 * 1024 * 1024,
) -> None:
    base_header_len, base_header = _read_safetensors_header(base_path)
    target_header_len, target_header = _read_safetensors_header(target_path)

    base_keys = _tensor_keys(base_header)
    target_keys = _tensor_keys(target_header)
    base_key_for_target = _match_keys(base_keys, target_keys)

    output_header: dict[str, dict] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key in target_keys:
        base_key = base_key_for_target[key]
        base_info = base_header[base_key]
        target_info = target_header[key]
        if base_info["shape"] != target_info["shape"]:
            raise ValueError(f"shape mismatch for {key}: {base_info['shape']} vs {target_info['shape']}")
        if base_info["dtype"] != "F32" or target_info["dtype"] != "F32":
            raise ValueError(
                f"numpy fallback only supports F32 tensors, got {key}: "
                f"{base_info['dtype']} / {target_info['dtype']}"
            )
        nbytes = _numel(base_info["shape"]) * 4
        if base_info["data_offsets"][1] - base_info["data_offsets"][0] != nbytes:
            raise ValueError(f"unexpected byte span for base tensor {key}")
        if target_info["data_offsets"][1] - target_info["data_offsets"][0] != nbytes:
            raise ValueError(f"unexpected byte span for target tensor {key}")
        output_header[key] = {
            "dtype": "F32",
            "shape": base_info["shape"],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    base_data_start = 8 + base_header_len
    target_data_start = 8 + target_header_len
    with base_path.open("rb") as base_file, target_path.open("rb") as target_file, output_path.open("wb") as out_file:
        out_file.write(struct.pack("<Q", len(header_bytes)))
        out_file.write(header_bytes)
        for key in target_keys:
            base_key = base_key_for_target[key]
            base_begin, base_end = base_header[base_key]["data_offsets"]
            target_begin, target_end = target_header[key]["data_offsets"]
            remaining = base_end - base_begin
            base_file.seek(base_data_start + base_begin)
            target_file.seek(target_data_start + target_begin)
            if target_end - target_begin != remaining:
                raise ValueError(f"byte span mismatch for {key}")

            while remaining:
                take = min(remaining, chunk_floats * 4)
                take -= take % 4
                base_buf = base_file.read(take)
                target_buf = target_file.read(take)
                if len(base_buf) != take or len(target_buf) != take:
                    raise ValueError(f"short read while interpolating {key}")
                base_arr = np.frombuffer(base_buf, dtype="<f4")
                target_arr = np.frombuffer(target_buf, dtype="<f4")
                out_arr = base_arr + alpha * (target_arr - base_arr)
                out_file.write(out_arr.astype("<f4", copy=False).tobytes())
                remaining -= take


def _interpolate_with_torch(base_path: Path, target_path: Path, output_path: Path, alpha: float) -> bool:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ModuleNotFoundError:
        return False

    base_state = load_file(str(base_path), device="cpu")
    target_state = load_file(str(target_path), device="cpu")
    base_key_for_target = _match_keys(list(base_state), list(target_state))

    out_state = {}
    for key, target_value in target_state.items():
        base_value = base_state[base_key_for_target[key]]
        if torch.is_floating_point(base_value):
            out_state[key] = base_value + alpha * (target_value - base_value)
        else:
            out_state[key] = target_value

    save_file(out_state, str(output_path), metadata={"format": "pt"})
    return True


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
    args.output.mkdir(parents=True, exist_ok=True)
    output_model = args.output / "adapter_model.safetensors"
    used_torch = _interpolate_with_torch(base_path, target_path, output_model, args.alpha)
    if not used_torch:
        _write_safetensors_f32_interpolation(base_path, target_path, output_model, args.alpha)
    shutil.copy2(args.target / "adapter_config.json", args.output / "adapter_config.json")
    metadata = {
        "base": str(args.base),
        "target": str(args.target),
        "alpha": args.alpha,
        "backend": "torch" if used_torch else "numpy_f32_stream",
    }
    (args.output / "interpolation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
