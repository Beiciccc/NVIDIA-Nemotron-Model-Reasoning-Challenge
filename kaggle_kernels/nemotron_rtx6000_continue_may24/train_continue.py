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
CONFIG_PATH = Path(__file__).with_name("candidate_config.json")
LEGACY_COMPETITION_TRAIN = Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")
OFFICIAL_SUFFIX = "\nPut your final answer inside \\boxed{}."

DEFAULT_CONFIG = {
    "candidate_id": "cycle_candidate",
    "seed": 501,
    "selection": "hard",
    "limit_records": 128,
    "max_length": 512,
    "max_steps": 16,
    "learning_rate": 5e-7,
    "batch_size": 1,
    "grad_accum": 1,
    "save_adapter": True,
}

CONFIG_OVERRIDES = {
    "candidate_id": "cycle02_kien086_equation_symbolic_lr3e7_s12",
    "seed": 502,
    "selection": "equation_symbolic",
    "limit_records": 128,
    "max_length": 512,
    "max_steps": 12,
    "learning_rate": 3e-7,
    "batch_size": 1,
    "grad_accum": 1,
    "save_adapter": True,
}


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


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    for path in [CONFIG_PATH, Path.cwd() / "candidate_config.json", OUTPUT_DIR / "candidate_config.json"]:
        if path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8")))
    config.update(CONFIG_OVERRIDES)
    return config


def classify_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if "bit manipulation" in lowered:
        return "bit"
    if "secret encryption rules" in lowered:
        return "text"
    if "different numeral system" in lowered:
        return "numeral"
    if "secret unit conversion" in lowered:
        return "unit"
    if "gravitational constant" in lowered:
        return "gravity"
    if "transformation rules is applied to equations" in lowered:
        match = re.search(r"now,\s*determine\s+the\s+result\s+for:\s*(.+)", prompt, re.I | re.S)
        target = match.group(1).strip() if match else prompt
        digits = sum(ch.isdigit() for ch in target)
        symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in target)
        if digits >= max(2, symbols):
            return "equation_numeric"
        return "equation_symbolic"
    return "other"


def filter_by_selection(df: pd.DataFrame, selection: str, limit: int) -> pd.DataFrame:
    selection = selection.lower().strip()
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = df["prompt"].astype(str).map(classify_prompt)

    groups = {
        "equation": {"equation_numeric", "equation_symbolic"},
        "hard": {"equation_numeric", "equation_symbolic", "bit", "text", "numeral"},
        "nonperfect": {"equation_numeric", "equation_symbolic", "bit", "text", "numeral"},
        "format": {"equation_numeric", "equation_symbolic", "bit", "text", "numeral"},
        "bit_text": {"bit", "text"},
        "equation_numeral": {"equation_numeric", "equation_symbolic", "numeral"},
        "equation_symbolic": {"equation_symbolic"},
        "equation_numeric": {"equation_numeric"},
        "measure_gravity": {"unit", "gravity"},
        "balanced": set(df["category"].unique()),
    }
    wanted = groups.get(selection)
    if wanted is None:
        wanted = {part.strip() for part in re.split(r"[,|+]", selection) if part.strip()}

    if wanted:
        filtered = df[df["category"].isin(wanted)]
        if len(filtered) >= min(limit, max(1, len(df) // 100)):
            return filtered
    return df


def make_records(train_csv: Path, limit: int, seed: int, selection: str) -> list[dict]:
    df = pd.read_csv(train_csv)
    df = df[df["prompt"].notna() & df["answer"].notna()].copy()
    df["category"] = df["prompt"].astype(str).map(classify_prompt)
    if "type" in df.columns:
        type_col = df["type"].astype(str)
        selection = selection.lower()
        if selection == "hard":
            hard_types = {"Equation Transformation", "Text Encryption", "Numeral Conversion"}
            filtered = df[type_col.isin(hard_types)]
            if len(filtered) >= limit:
                df = filtered
        elif selection == "equation":
            filtered = df[type_col.str.contains("Equation", case=False, na=False)]
            if len(filtered) >= limit:
                df = filtered
        elif selection == "format":
            filtered = df[type_col.isin({"Equation Transformation", "Text Encryption", "Numeral Conversion"})]
            if len(filtered) >= limit:
                df = filtered
        elif selection == "nonperfect":
            easy_types = {"Gravity Physics", "Unit Conversion"}
            filtered = df[~type_col.isin(easy_types)]
            if len(filtered) >= limit:
                df = filtered
    else:
        df = filter_by_selection(df, selection, limit)
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
    config = load_config()
    seed = int(config["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(seed)
    torch.manual_seed(seed)

    report: dict = {
        "mode": "continue_submit_candidate",
        "config": config,
        "candidate_id": config["candidate_id"],
        "save_adapter": bool(config["save_adapter"]),
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
        records = make_records(
            train_csv,
            int(config["limit_records"]),
            seed=seed,
            selection=str(config["selection"]),
        )
        max_length = int(config["max_length"])
        tokenized = [tokenize_masked(tokenizer, rec, max_length) for rec in records]
        loss_tokens = sum(sum(x != -100 for x in row["labels"]) for row in tokenized)
        record_categories = [classify_prompt(rec["messages"][0]["content"]) for rec in records]
        report.update(
            {
                "records": len(records),
                "train_csv": str(train_csv),
                "max_length": max_length,
                "loss_tokens": loss_tokens,
                "record_ids": [r["id"] for r in records],
                "record_category_counts": {name: record_categories.count(name) for name in sorted(set(record_categories))},
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
            output_dir=str(OUTPUT_DIR / "continue_trainer"),
            max_steps=int(config["max_steps"]),
            per_device_train_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=int(config["grad_accum"]),
            learning_rate=float(config["learning_rate"]),
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

        if bool(config["save_adapter"]):
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
