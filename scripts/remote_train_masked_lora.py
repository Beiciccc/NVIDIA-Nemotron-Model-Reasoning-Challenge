from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from remote_train_lora import (
    PROMPT_SUFFIX,
    TARGET_MODULES,
    TONG_VARIANTS,
    apply_chat_template,
    load_dgx_records,
    load_konbu_records,
    load_tong_records,
    resolve_model_path,
)


OFFICIAL_SUFFIX = "\nPut your final answer inside \\boxed{}."
OFFICIAL_HARD_TYPES = {"Equation Transformation", "Text Encryption", "Numeral Conversion"}
TONG_OFFICIAL_SUFFIX_VARIANTS = {f"{name}_official_suffix": name for name in TONG_VARIANTS}
RECENT_REASONING_VARIANTS = {
    "recent_reasoning",
    "recent_reasoning_hard",
    "recent_reasoning_hard_plus_perfect",
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
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def tokenize_masked(tokenizer, record: dict, max_length: int) -> dict:
    messages = record["messages"]
    user_content = messages[0]["content"]
    prompt_text = build_prompt_text(tokenizer, user_content)
    full_text = apply_chat_template(tokenizer, messages)

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


def make_official_records(df: pd.DataFrame, boxed_answer: bool = False) -> list[dict]:
    def assistant_answer(answer: object) -> str:
        text = str(answer).strip()
        return f"\\boxed{{{text}}}" if boxed_answer else text

    return [
        {
            "messages": [
                {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
                {"role": "assistant", "content": assistant_answer(row["answer"])},
            ]
        }
        for _, row in df.iterrows()
    ]


def build_reasoning_answer(reasoning: object, answer: object) -> str:
    text = str(reasoning).strip()
    answer_text = str(answer).strip()
    if "\\boxed{" in text:
        return text
    return f"{text}\n\n\\boxed{{{answer_text}}}" if text else f"\\boxed{{{answer_text}}}"


def make_reasoning_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
                {"role": "assistant", "content": build_reasoning_answer(row["reasoning"], row["answer"])},
            ]
        }
        for _, row in df.iterrows()
    ]


def load_recent_reasoning_records(project_dir: Path, variant: str, seed: int) -> list[dict]:
    recent_dir = project_dir / "external/datasets/public_recent"
    reasoning_path = recent_dir / "nemotron-reasoning-data2/all_reasoning_training.csv"
    df = pd.read_csv(reasoning_path)
    df = df[df["reasoning"].notna() & (df["reasoning"].astype(str).str.len() > 5)]

    if variant in {"recent_reasoning_hard", "recent_reasoning_hard_plus_perfect"}:
        traj_path = recent_dir / "nemotron-reasoning-traj/nemotron_traj.csv"
        traj = pd.read_csv(traj_path, usecols=["prompt", "correctness"])
        hard_prompts = set(traj.loc[traj["correctness"].isin(["false", "partial"]), "prompt"].astype(str))
        df = df[df["prompt"].astype(str).isin(hard_prompts)]

    df = df.sample(frac=1.0, random_state=seed)
    records = make_reasoning_records(df)

    if variant == "recent_reasoning_hard_plus_perfect":
        perfect_path = recent_dir / "nemotron-perfect-synthetic-dataset/perfect_train.csv"
        perfect = pd.read_csv(perfect_path).rename(columns={"answer": "reasoning"})
        perfect["answer"] = ""
        perfect_records = make_reasoning_records(perfect)
        tiny_path = recent_dir / "nemotron-reasoning-train-data/train_reasoning_30.csv"
        tiny = pd.read_csv(tiny_path)
        tiny_records = make_reasoning_records(tiny)
        records.extend(perfect_records)
        records.extend(tiny_records)
        random.Random(seed).shuffle(records)

    print(f"{variant} records: {len(records)}")
    return records


def use_official_suffix(records: list[dict]) -> list[dict]:
    retargeted: list[dict] = []
    for rec in records:
        copied = json.loads(json.dumps(rec))
        user = copied["messages"][0]["content"]
        if user.endswith(PROMPT_SUFFIX):
            user = user[: -len(PROMPT_SUFFIX)]
        copied["messages"][0]["content"] = user.rstrip() + OFFICIAL_SUFFIX
        retargeted.append(copied)
    return retargeted


def load_records(project_dir: Path, variant: str, seed: int) -> list[dict]:
    if variant in RECENT_REASONING_VARIANTS:
        return load_recent_reasoning_records(project_dir, variant, seed)
    if variant in TONG_OFFICIAL_SUFFIX_VARIANTS:
        base_variant = TONG_OFFICIAL_SUFFIX_VARIANTS[variant]
        records = use_official_suffix(load_tong_records(project_dir, base_variant, seed))
        print(f"{variant} records: {len(records)}")
        return records
    if variant.startswith("tong_"):
        return load_tong_records(project_dir, variant, seed)
    if variant == "dgx":
        return load_dgx_records(project_dir, seed)
    if variant in {"official", "official_boxed"}:
        path = project_dir / "data/train.csv"
        df = pd.read_csv(path)
        df = df.sample(frac=1.0, random_state=seed)
        records = make_official_records(df, boxed_answer=variant.endswith("_boxed"))
        print(f"{variant} records: {len(records)}")
        return records
    if variant in {
        "official_train_split",
        "official_hard_train_split",
        "official_train_split_boxed",
        "official_hard_train_split_boxed",
    }:
        path = project_dir / "external/datasets/exp024_tong/train_split.csv"
        df = pd.read_csv(path)
        if variant in {"official_hard_train_split", "official_hard_train_split_boxed"}:
            df = df[df["type"].isin(OFFICIAL_HARD_TYPES)]
        df = df.sample(frac=1.0, random_state=seed)
        records = make_official_records(df, boxed_answer=variant.endswith("_boxed"))
        print(f"{variant} records: {len(records)}")
        return records
    return load_konbu_records(project_dir, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("/home/featurize/nemotron"))
    parser.add_argument("--model-path")
    parser.add_argument("--init-adapter")
    parser.add_argument(
        "--variant",
        choices=[
            *TONG_VARIANTS.keys(),
            *TONG_OFFICIAL_SUFFIX_VARIANTS.keys(),
            *sorted(RECENT_REASONING_VARIANTS),
            "dgx",
            "konbu",
            "official",
            "official_boxed",
            "official_train_split",
            "official_hard_train_split",
            "official_train_split_boxed",
            "official_hard_train_split_boxed",
        ],
        default="tong_s012",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(args.project_dir / "external/hf"))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    records = load_records(args.project_dir, args.variant, args.seed)
    if args.limit_records:
        records = records[: args.limit_records]

    model_path = resolve_model_path(args.project_dir, args.model_path)
    print(f"model path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = [tokenize_masked(tokenizer, rec, args.max_length) for rec in records]
    dataset = Dataset.from_list(tokenized)
    loss_tokens = sum(sum(1 for x in row["labels"] if x != -100) for row in tokenized)
    total_tokens = sum(len(row["input_ids"]) for row in tokenized)
    print(f"records: {len(dataset)} total_tokens={total_tokens} loss_tokens={loss_tokens}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if args.init_adapter:
        print(f"init adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    save_strategy = "steps" if args.save_steps > 0 else "no"

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=5,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
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
        args=train_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "variant": args.variant,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "records": len(dataset),
        "total_tokens": total_tokens,
        "loss_tokens": loss_tokens,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "init_adapter": args.init_adapter,
        "model_path": model_path,
        "masking": "prompt_masked_completion_loss",
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
