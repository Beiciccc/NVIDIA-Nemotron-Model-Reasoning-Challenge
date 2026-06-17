from __future__ import annotations

import gc
import json
import math
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
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


OUTPUT_DIR = Path("/kaggle/working")
REPORT_PATH = OUTPUT_DIR / "kaggle_rtx_submit_report.json"
OFFICIAL_SUFFIX = "\nPut your final answer inside \\boxed{}."


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    max_steps: int
    limit_records: int
    max_length: int
    learning_rate: float
    selection: str
    answer_style: str = "boxed_only"
    eval_per_category: int = 1


CONFIGS = {
    "cycle01": ExperimentConfig(
        name="cycle01_kien_format_balanced_lr2e7_s8",
        seed=2401,
        max_steps=8,
        limit_records=96,
        max_length=512,
        learning_rate=2e-7,
        selection="balanced_all",
        answer_style="boxed_only",
        eval_per_category=1,
    ),
    "cycle02": ExperimentConfig(
        name="cycle02_kien_bit_text_lr3e7_s12",
        seed=2402,
        max_steps=12,
        limit_records=128,
        max_length=512,
        learning_rate=3e-7,
        selection="bit_text",
        answer_style="boxed_only",
        eval_per_category=1,
    ),
    "cycle03": ExperimentConfig(
        name="cycle03_kien_equation_numeral_lr3e7_s12",
        seed=2403,
        max_steps=12,
        limit_records=128,
        max_length=512,
        learning_rate=3e-7,
        selection="equation_numeral",
        answer_style="boxed_only",
        eval_per_category=1,
    ),
    "cycle04": ExperimentConfig(
        name="cycle04_kien_measure_gravity_lr4e7_s16",
        seed=2404,
        max_steps=16,
        limit_records=160,
        max_length=512,
        learning_rate=4e-7,
        selection="measure_gravity",
        answer_style="boxed_only",
        eval_per_category=1,
    ),
    "cycle05": ExperimentConfig(
        name="cycle05_kien_all_shortcot_lr2e7_s16",
        seed=2405,
        max_steps=16,
        limit_records=192,
        max_length=768,
        learning_rate=2e-7,
        selection="balanced_all",
        answer_style="short_cot",
        eval_per_category=1,
    ),
}

RUN_CONFIG_NAME = os.environ.get("NEMOTRON_RUN_CONFIG", "cycle01")
CONFIG = CONFIGS[RUN_CONFIG_NAME]


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


def discover_train_csv() -> Path:
    known = [
        Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"),
        Path("/kaggle/input/competitions/nvidia-nemotron-3-reasoning-challenge/train.csv"),
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
    raise FileNotFoundError("Could not find train.csv with prompt and answer columns")


def discover_adapter_path() -> Path:
    candidates = []
    for cfg in Path("/kaggle/input").rglob("adapter_config.json"):
        folder = cfg.parent
        if (folder / "adapter_model.safetensors").is_file():
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError("No adapter_config.json + adapter_model.safetensors pair found")

    def score(path: Path) -> tuple[int, str]:
        lowered = str(path).lower()
        bonus = 0
        if "kien" in lowered:
            bonus += 100
        if "tinker" in lowered:
            bonus += 100
        return (-bonus, str(path))

    return sorted(candidates, key=score)[0]


def categorize_prompt(prompt: str) -> str:
    p = prompt.lower()
    if "secret bit manipulation" in p or "8-bit binary" in p:
        return "bit"
    if "secret encryption rules" in p or "decrypt the following text" in p:
        return "text"
    if "different numeral system" in p:
        return "numeral"
    if "secret set of transformation rules is applied to equations" in p:
        return "equation"
    if "gravitational constant" in p:
        return "gravity"
    if "secret unit conversion" in p:
        return "measure"
    return "other"


def select_train_rows(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    df = df[df["prompt"].notna() & df["answer"].notna()].copy()
    df["category"] = df["prompt"].map(categorize_prompt)
    rng = random.Random(cfg.seed)
    categories_by_selection = {
        "balanced_all": ["bit", "text", "numeral", "equation", "gravity", "measure"],
        "bit_text": ["bit", "text"],
        "equation_numeral": ["equation", "numeral"],
        "measure_gravity": ["measure", "gravity"],
    }
    categories = categories_by_selection[cfg.selection]
    per_cat = max(1, math.ceil(cfg.limit_records / len(categories)))
    chosen = []
    for category in categories:
        part = df[df["category"] == category]
        if len(part) == 0:
            continue
        seeds = [cfg.seed + i for i in range(3)]
        part = part.sample(frac=1.0, random_state=seeds[len(chosen) % len(seeds)])
        chosen.append(part.head(per_cat))
    out = pd.concat(chosen, ignore_index=True)
    out = out.sample(frac=1.0, random_state=cfg.seed).head(cfg.limit_records).reset_index(drop=True)
    order = list(range(len(out)))
    rng.shuffle(order)
    return out.iloc[order].reset_index(drop=True)


def select_eval_rows(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    df = df[df["prompt"].notna() & df["answer"].notna()].copy()
    df["category"] = df["prompt"].map(categorize_prompt)
    selected = []
    for category in ["bit", "text", "numeral", "equation", "gravity", "measure"]:
        part = df[df["category"] == category]
        if len(part) == 0:
            continue
        selected.append(part.sample(n=min(cfg.eval_per_category, len(part)), random_state=cfg.seed + 99))
    return pd.concat(selected, ignore_index=True).reset_index(drop=True)


def assistant_text(answer: str, style: str) -> str:
    ans = str(answer).strip()
    if style == "short_cot":
        return f"The requested output is \\boxed{{{ans}}}."
    return f"\\boxed{{{ans}}}"


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


def make_record(row: pd.Series, cfg: ExperimentConfig) -> dict:
    answer = str(row["answer"]).strip()
    return {
        "id": str(row.get("id", "")),
        "category": str(row.get("category", categorize_prompt(str(row["prompt"])))),
        "messages": [
            {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
            {"role": "assistant", "content": assistant_text(answer, cfg.answer_style)},
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
    return {
        "input_ids": input_ids,
        "labels": labels,
        "id": record["id"],
        "category": record["category"],
    }


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


@torch.no_grad()
def eval_generation(model, tokenizer, eval_df: pd.DataFrame, max_new_tokens: int = 48) -> dict:
    model.eval()
    rows = []
    exact = 0
    boxed = 0
    for _, row in eval_df.iterrows():
        prompt = str(row["prompt"]).strip() + OFFICIAL_SUFFIX
        prompt_text = build_prompt_text(tokenizer, prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_boxed(decoded)
        target = str(row["answer"]).strip()
        ok = normalize_answer(pred) == normalize_answer(target)
        exact += int(ok)
        boxed += int("\\boxed{" in decoded)
        rows.append(
            {
                "id": str(row.get("id", "")),
                "category": str(row.get("category", categorize_prompt(prompt))),
                "target": target,
                "prediction": pred,
                "raw": decoded[-500:],
                "exact": ok,
                "has_boxed": "\\boxed{" in decoded,
            }
        )
    model.train()
    n = max(1, len(rows))
    return {"n": len(rows), "exact": exact, "exact_rate": exact / n, "boxed_rate": boxed / n, "rows": rows}


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
    random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

    report: dict = {
        "mode": "submit_train",
        "run_config_name": RUN_CONFIG_NAME,
        "config": asdict(CONFIG),
        "utility_paths": add_utility_paths(),
        "triton_ptxas_fix": fix_triton_ptxas(),
        "nvidia_smi": run(["nvidia-smi"]),
        "input_roots": sorted(str(p) for p in Path("/kaggle/input").glob("*")),
        "train_csv_candidates": sorted(str(p) for p in Path("/kaggle/input").rglob("train.csv"))[:20],
    }

    try:
        import kagglehub

        model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
        adapter_path = discover_adapter_path()
        train_csv = discover_train_csv()
        report.update({"model_path": str(model_path), "adapter_path": str(adapter_path), "train_csv": str(train_csv)})

        df = pd.read_csv(train_csv)
        train_df = select_train_rows(df, CONFIG)
        eval_df = select_eval_rows(df, CONFIG)
        train_records = [make_record(row, CONFIG) for _, row in train_df.iterrows()]
        report["train_category_counts"] = train_df["category"].value_counts().to_dict()
        report["eval_category_counts"] = eval_df["category"].value_counts().to_dict()
        report["train_record_ids"] = train_df["id"].astype(str).head(30).tolist()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenized = [tokenize_masked(tokenizer, rec, CONFIG.max_length) for rec in train_records]
        loss_tokens = sum(sum(x != -100 for x in row["labels"]) for row in tokenized)
        report.update({"records": len(tokenized), "max_length": CONFIG.max_length, "loss_tokens": loss_tokens})

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

        if len(eval_df) > 0:
            report["pre_eval"] = eval_generation(model, tokenizer, eval_df)
            gc.collect()
            torch.cuda.empty_cache()

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "trainer"),
            max_steps=CONFIG.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=CONFIG.learning_rate,
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

        if len(eval_df) > 0:
            report["post_eval"] = eval_generation(model, tokenizer, eval_df)

        adapter_dir = OUTPUT_DIR / "adapter"
        model.save_pretrained(adapter_dir, save_embedding_layers=False)
        zip_path = package_adapter(adapter_dir)
        report["adapter_dir"] = str(adapter_dir)
        report["submission_zip"] = str(zip_path)
        report["submission_zip_bytes"] = zip_path.stat().st_size
        report["submission_zip_names"] = zipfile.ZipFile(zip_path).namelist()
        report["validation_gate"] = {
            "zip_exists": zip_path.is_file(),
            "zip_min_bytes": zip_path.stat().st_size > 100_000_000,
            "loss_finite": math.isfinite(float(report["train_metrics"].get("train_loss", 0.0))),
            "eval_boxed_rate_post": report.get("post_eval", {}).get("boxed_rate"),
        }
        report["status"] = "ok"
        report["elapsed_seconds"] = round(time.time() - start, 2)
        write_report(report)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        report["elapsed_seconds"] = round(time.time() - start, 2)
        write_report(report)
        raise


if __name__ == "__main__":
    main()
