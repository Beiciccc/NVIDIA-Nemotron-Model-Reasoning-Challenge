from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import site
import stat
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


OUTPUT_DIR = Path("/kaggle/working")
CONFIG_PATH = Path(__file__).with_name("variant_config.json")
REPORT_PATH = OUTPUT_DIR / "submit5_report.json"
USED_CONFIG_PATH = OUTPUT_DIR / "variant_config_used.json"
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"
OFFICIAL_SUFFIX = "\nPut your final answer inside \\boxed{}."
BASE_MODEL_NAME = "metric/nemotron-3-nano-30b-a3b-bf16"


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
        report.update({"applied": True, "ptxas_src": str(ptxas_src), "ptxas_dst": str(ptxas_dst)})
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


def discover_train_csv() -> Path:
    known = [
        Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"),
        Path("/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv"),
        Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv"),
    ]
    for path in known:
        if path.is_file():
            return path
    for path in sorted(Path("/kaggle/input").rglob("train.csv")):
        try:
            head = pd.read_csv(path, nrows=2)
        except Exception:
            continue
        if {"prompt", "answer"}.issubset(set(head.columns)):
            return path
    raise FileNotFoundError("Could not find labeled train.csv under /kaggle/input")


def parse_equation_prompt(prompt: str) -> tuple[str, str]:
    text = str(prompt)
    marker = "Below are a few examples:\n"
    question_marker = "\nNow, determine the result for: "
    if marker not in text or question_marker not in text:
        return "", ""
    after_header = text.split(marker, 1)[1]
    examples_text, rest = after_header.split(question_marker, 1)
    return examples_text, rest.strip()


def infer_task_family(prompt: str) -> str:
    text = str(prompt).lower()
    if "bit manipulation" in text or "8-bit binary" in text:
        return "bit_manipulation"
    if "decrypt the following text" in text or "secret encryption" in text:
        return "cipher"
    if "write the number" in text and ("wonderland" in text or "numeral" in text):
        return "numeral"
    if "unit conversion" in text or "measurement" in text or " m becomes " in text:
        return "unit_conversion"
    if "gravity" in text or "falling distance" in text or "gravitational" in text:
        return "gravity"
    if "equation" in text or "determine the result for" in text:
        examples_text, _ = parse_equation_prompt(prompt)
        if any(ch.isdigit() for ch in examples_text):
            return "equation_numeric"
        return "equation_symbolic"
    return "unknown"


def infer_validation_category(prompt: str) -> str:
    task = infer_task_family(prompt)
    if task not in {"equation_numeric", "equation_symbolic"}:
        return task
    examples_text, question_text = parse_equation_prompt(prompt)
    if task == "equation_numeric":
        q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
        if q_match and re.search(r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text):
            return "equation_numeric_deduce"
        return "equation_numeric_guess"
    if len(question_text) == 5:
        q_op = question_text[2]
        for ex_line in examples_text.strip().splitlines():
            inp = ex_line.split(" = ", 1)[0].strip()
            if len(inp) == 5 and inp[2] == q_op:
                return "cryptarithm_deduce"
    return "cryptarithm_guess"


def prepare_frame(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv, dtype={"id": str, "answer": str})
    df = df[df["prompt"].notna() & df["answer"].notna()].copy()
    df["family"] = df["prompt"].map(infer_task_family)
    df["category"] = df["prompt"].map(infer_validation_category)
    return df


def select_eval_ids(df: pd.DataFrame, rows_per_family: int, seed: int) -> set[str]:
    if rows_per_family <= 0:
        return set()
    ids = []
    for _, group in df.groupby("family"):
        n = min(rows_per_family, len(group))
        if n:
            ids.extend(group.sample(n=n, random_state=seed)["id"].tolist())
    return set(ids)


def select_train_rows(df: pd.DataFrame, cfg: dict, eval_ids: set[str]) -> pd.DataFrame:
    work = df[~df["id"].isin(eval_ids)].copy()
    families = set(cfg.get("families") or [])
    categories = set(cfg.get("categories") or [])
    if families:
        work = work[work["family"].isin(families)]
    if categories:
        work = work[work["category"].isin(categories)]
    if work.empty:
        raise ValueError(f"empty training set for families={families} categories={categories}")
    limit = int(cfg.get("train_limit") or len(work))
    return work.sample(frac=1.0, random_state=int(cfg.get("seed", 46))).head(limit).reset_index(drop=True)


def select_eval_rows(df: pd.DataFrame, eval_ids: set[str]) -> pd.DataFrame:
    if not eval_ids:
        return df.sample(n=min(64, len(df)), random_state=2026).reset_index(drop=True)
    return df[df["id"].isin(eval_ids)].reset_index(drop=True)


def build_prompt_text(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def row_to_record(row: pd.Series) -> dict:
    answer = str(row["answer"]).strip()
    return {
        "id": str(row["id"]),
        "family": str(row["family"]),
        "category": str(row["category"]),
        "messages": [
            {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ],
    }


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


def make_loss_batch(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    batch = CausalCollator(pad_token_id)(features)
    return {k: v.to("cuda") for k, v in batch.items()}


def compute_answer_nll(model, tokenizer, records: list[dict], max_length: int, batch_size: int = 4) -> dict:
    tokenized = [tokenize_masked(tokenizer, rec, max_length) for rec in records]
    total_loss = 0.0
    total_tokens = 0
    by_family = defaultdict(lambda: [0.0, 0])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokenized), batch_size):
            rows = tokenized[start : start + batch_size]
            recs = records[start : start + batch_size]
            batch = make_loss_batch(rows, tokenizer.pad_token_id)
            outputs = model(**batch)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, 1:].contiguous()
            vocab = logits.shape[-1]
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
            masks = labels.ne(-100)
            for idx, rec in enumerate(recs):
                n = int(masks[idx].sum().item())
                loss_sum = float(losses[idx][masks[idx]].sum().item()) if n else 0.0
                total_loss += loss_sum
                total_tokens += n
                by_family[rec["family"]][0] += loss_sum
                by_family[rec["family"]][1] += n
    return {
        "answer_nll": total_loss / max(total_tokens, 1),
        "answer_tokens": total_tokens,
        "loss_by_family": {k: v[0] / max(v[1], 1) for k, v in sorted(by_family.items())},
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_adapter(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"
    config = json.loads(cfg_path.read_text())
    config["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(cfg_path, arcname="adapter_config.json")
        zf.write(model_path, arcname="adapter_model.safetensors")
    return {
        "path": str(SUBMISSION_ZIP),
        "bytes": SUBMISSION_ZIP.stat().st_size,
        "sha256": sha256_path(SUBMISSION_ZIP),
        "zip_contents": sorted(zipfile.ZipFile(SUBMISSION_ZIP).namelist()),
    }


def write_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    start = time.time()
    cfg = json.loads(CONFIG_PATH.read_text())
    USED_CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(int(cfg.get("seed", 46)))
    torch.manual_seed(int(cfg.get("seed", 46)))

    report = {
        "variant_config": cfg,
        "utility_paths": add_utility_paths(),
        "triton_ptxas_fix": fix_triton_ptxas(),
        "nvidia_smi": run(["nvidia-smi"]),
        "input_roots": sorted(str(p) for p in Path("/kaggle/input").glob("*")),
    }
    try:
        import kagglehub

        model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
        adapter_path = discover_adapter_path()
        train_csv = discover_train_csv()
        report.update({"model_path": str(model_path), "adapter_path": str(adapter_path), "train_csv": str(train_csv)})

        df = prepare_frame(train_csv)
        eval_ids = select_eval_ids(df, int(cfg.get("eval_rows_per_family", 8)), int(cfg.get("seed", 46)) + 999)
        train_rows = select_train_rows(df, cfg, eval_ids)
        eval_rows = select_eval_rows(df, eval_ids)
        train_records = [row_to_record(row) for _, row in train_rows.iterrows()]
        eval_records = [row_to_record(row) for _, row in eval_rows.iterrows()]
        max_length = int(cfg.get("max_length", 512))

        report["data"] = {
            "train_rows": len(train_records),
            "eval_rows": len(eval_records),
            "train_family_counts": train_rows["family"].value_counts().sort_index().to_dict(),
            "train_category_counts": train_rows["category"].value_counts().sort_index().to_dict(),
            "eval_family_counts": eval_rows["family"].value_counts().sort_index().to_dict(),
            "train_ids_head": [r["id"] for r in train_records[:20]],
            "eval_ids_head": [r["id"] for r in eval_records[:20]],
        }

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenized = [tokenize_masked(tokenizer, rec, max_length) for rec in train_records]
        report["tokenized"] = {
            "max_length": max_length,
            "loss_tokens": sum(sum(x != -100 for x in row["labels"]) for row in tokenized),
            "min_len": min(len(row["input_ids"]) for row in tokenized),
            "max_len": max(len(row["input_ids"]) for row in tokenized),
        }

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
        report["params"] = {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        }

        report["eval_before"] = compute_answer_nll(model, tokenizer, eval_records, max_length, batch_size=4)
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "trainer_logs"),
            max_steps=int(cfg.get("max_steps", 16)),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
            learning_rate=float(cfg.get("learning_rate", 5e-7)),
            lr_scheduler_type=str(cfg.get("scheduler", "constant")),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
            weight_decay=0.0,
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            optim="adamw_torch_fused",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=CausalCollator(tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        report["train_metrics"] = train_result.metrics
        report["eval_after"] = compute_answer_nll(model, tokenizer, eval_records, max_length, batch_size=4)
        report["eval_delta"] = report["eval_before"]["answer_nll"] - report["eval_after"]["answer_nll"]

        adapter_dir = Path("/tmp/submit5_adapter")
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        model.save_pretrained(adapter_dir, save_embedding_layers=False)
        report["submission"] = package_adapter(adapter_dir)
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
