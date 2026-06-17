from __future__ import annotations

import json
import os
import random
import re
import shutil
import site
import subprocess
import stat
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


OUTPUT_DIR = Path("/kaggle/working")
REPORT_PATH = OUTPUT_DIR / "kaggle_rtx_train_report.json"
LEGACY_COMPETITION_TRAIN = Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")
OFFICIAL_SUFFIX = "\nPut your final answer inside \\boxed{}."

# Smoke defaults: validate the RTX6000 training path without producing a multi-GB adapter.
SMOKE_MAX_STEPS = 2
SMOKE_LIMIT_RECORDS = 8
SMOKE_MAX_LENGTH = 512
SAVE_ADAPTER = False


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def run(cmd: list[str], timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def add_utility_paths() -> list[str]:
    added = []
    for path in [
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages",
        "/kaggle/input/nvidia-utility-script/nvidia_cutlass_dsl/python_packages",
        "/kaggle/input/nvidia_utility_script/nvidia_cutlass_dsl/python_packages",
    ]:
        if Path(path).exists():
            site.addsitedir(path)
            added.append(path)
    return added


def fix_triton_ptxas() -> dict:
    report: dict = {"applied": False}
    utility_root = Path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script")
    if utility_root.exists():
        sys.path.insert(0, str(utility_root))
    ptxas_src = utility_root / "triton/backends/nvidia/bin/ptxas-blackwell"
    ptxas_dst = Path("/tmp/ptxas-blackwell")
    if ptxas_src.exists():
        shutil.copy2(ptxas_src, ptxas_dst)
        ptxas_dst.chmod(ptxas_dst.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        dst_bin = Path("/tmp/triton_nvidia_bin")
        shutil.copytree(ptxas_src.parent, dst_bin, dirs_exist_ok=True)
        for fp in dst_bin.iterdir():
            if fp.is_file():
                fp.chmod(fp.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas_dst)
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas_dst)
        report.update(
            {
                "applied": True,
                "ptxas_src": str(ptxas_src),
                "ptxas_dst": str(ptxas_dst),
                "ptxas_dst_mode": oct(ptxas_dst.stat().st_mode),
                "dst_bin": str(dst_bin),
            }
        )

    try:
        import triton.backends.nvidia.compiler as nv_compiler

        nv_compiler.get_ptxas_version = lambda arch: "12.0"
        report["patched_get_ptxas_version"] = True
    except Exception as exc:
        report["patched_get_ptxas_version"] = False
        report["patch_error"] = repr(exc)
    return report


def discover_adapter_path() -> Path:
    candidates = []
    for cfg in Path("/kaggle/input").rglob("adapter_config.json"):
        folder = cfg.parent
        if (folder / "adapter_model.safetensors").is_file():
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError("No adapter_config.json + adapter_model.safetensors pair found under /kaggle/input")

    def score(path: Path) -> tuple[int, str]:
        lowered = str(path).lower()
        bonus = 0
        if "tinker" in lowered or "kien" in lowered:
            bonus += 100
        if "0-86" in lowered or "086" in lowered:
            bonus += 50
        return (-bonus, str(path))

    return sorted(candidates, key=score)[0]


def build_prompt_text(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize_masked(tokenizer, record: dict, max_length: int) -> dict:
    user_content = record["messages"][0]["content"]
    prompt_text = build_prompt_text(tokenizer, user_content)
    full_text = apply_chat_template(tokenizer, record["messages"])

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        input_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[:max_length]
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(x == -100 for x in labels):
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "labels": labels}


def discover_train_csv() -> Path:
    known = [
        LEGACY_COMPETITION_TRAIN,
        Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"),
        Path("/kaggle/input/competitions/nvidia-nemotron-3-reasoning-challenge/train.csv"),
    ]
    for path in known:
        if path.is_file():
            return path
    for path in sorted(Path("/kaggle/input").rglob("train.csv")):
        try:
            head = pd.read_csv(path, nrows=2)
        except Exception:
            continue
        cols = set(head.columns)
        if {"prompt", "answer"}.issubset(cols):
            return path
    raise FileNotFoundError("Could not find a train.csv with prompt and answer columns under /kaggle/input")


def make_records(train_csv: Path, limit: int, seed: int) -> list[dict]:
    df = pd.read_csv(train_csv)
    df = df[df["prompt"].notna() & df["answer"].notna()].copy()
    if "type" in df.columns:
        hard_types = {"Equation Transformation", "Text Encryption", "Numeral Conversion"}
        hard = df[df["type"].isin(hard_types)]
        if len(hard) >= limit:
            df = hard
    df = df.sample(frac=1.0, random_state=seed).head(limit)
    records = []
    for _, row in df.iterrows():
        answer = str(row["answer"]).strip()
        records.append(
            {
                "id": str(row.get("id", "")),
                "messages": [
                    {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
                    {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
                ],
            }
        )
    return records


def package_adapter(adapter_dir: Path) -> Path:
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            zf.write(adapter_dir / name, arcname=name)
    return zip_path


def write_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    start = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(46)
    torch.manual_seed(46)

    report: dict = {
        "mode": "smoke",
        "save_adapter": SAVE_ADAPTER,
        "utility_paths": add_utility_paths(),
        "triton_ptxas_fix": fix_triton_ptxas(),
        "nvidia_smi": run(["nvidia-smi"]),
        "input_roots": sorted(str(p) for p in Path("/kaggle/input").glob("*")),
        "legacy_train_csv_exists": LEGACY_COMPETITION_TRAIN.exists(),
        "train_csv_candidates": sorted(str(p) for p in Path("/kaggle/input").rglob("train.csv"))[:20],
    }

    try:
        import kagglehub

        model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
        adapter_path = discover_adapter_path()
        report.update({"model_path": str(model_path), "adapter_path": str(adapter_path)})

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_csv = discover_train_csv()
        records = make_records(train_csv, SMOKE_LIMIT_RECORDS, seed=46)
        tokenized = [tokenize_masked(tokenizer, rec, SMOKE_MAX_LENGTH) for rec in records]
        loss_tokens = sum(sum(x != -100 for x in row["labels"]) for row in tokenized)
        report.update(
            {
                "records": len(records),
                "train_csv": str(train_csv),
                "max_length": SMOKE_MAX_LENGTH,
                "loss_tokens": loss_tokens,
                "record_ids": [r["id"] for r in records],
            }
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        model.config.use_cache = False

        for name, module in sys.modules.items():
            if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
                module.is_fast_path_available = False

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        report.update({"trainable_params": trainable, "total_params": total})

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "smoke_trainer"),
            max_steps=SMOKE_MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-6,
            lr_scheduler_type="constant",
            warmup_ratio=0.0,
            weight_decay=0.0,
            max_grad_norm=1.0,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            optim="adamw_torch_fused",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=CausalCollator(tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        report["train_metrics"] = train_result.metrics

        if SAVE_ADAPTER:
            adapter_dir = OUTPUT_DIR / "adapter"
            model.save_pretrained(adapter_dir, save_embedding_layers=False)
            zip_path = package_adapter(adapter_dir)
            report["adapter_dir"] = str(adapter_dir)
            report["submission_zip"] = str(zip_path)
            report["submission_zip_bytes"] = zip_path.stat().st_size

        report["elapsed_seconds"] = round(time.time() - start, 2)
        report["status"] = "ok"
        write_report(report)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        report["elapsed_seconds"] = round(time.time() - start, 2)
        write_report(report)
        raise


if __name__ == "__main__":
    main()
