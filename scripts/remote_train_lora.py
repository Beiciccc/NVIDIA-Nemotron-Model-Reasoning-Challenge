from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]


TONG_VARIANTS = {
    "tong_s005": {
        "samples": {
            "numeral_system.csv": 600,
            "gravity_physics.csv": 1000,
            "unit_conversion.csv": 1000,
            "text_decryption.csv": 1492,
            "bit_manipulation_including_wrong.csv": 1508,
            "equation_numeric.csv": 535,
            "cryptarithm.csv": 69,
        },
        "priority": None,
    },
    "tong_s006": {
        "samples": {
            "numeral_system.csv": 600,
            "gravity_physics.csv": 1000,
            "unit_conversion.csv": 1000,
            "text_decryption.csv": 1492,
            "bit_manipulation_including_wrong.csv": 1508,
            "equation_numeric.csv": 535,
            "cryptarithm.csv": 69,
        },
        "priority": "exp026_s005_priority.txt",
    },
    "tong_s011": {
        "samples": {
            "numeral_system.csv": 600,
            "gravity_physics.csv": 1200,
            "unit_conversion.csv": 1150,
            "text_decryption.csv": 1492,
            "bit_manipulation_including_wrong.csv": 1508,
            "bit_manipulation_synth_including_wrong_v2.csv": 500,
            "equation_numeric.csv": 535,
            "cryptarithm.csv": 69,
        },
        "priority": None,
    },
    "tong_s012": {
        "samples": {
            "numeral_system.csv": 600,
            "gravity_physics.csv": 1200,
            "unit_conversion.csv": 1150,
            "text_decryption.csv": 1492,
            "bit_manipulation_including_wrong.csv": 1508,
            "bit_manipulation_synth_including_wrong_v2.csv": 500,
            "equation_numeric.csv": 535,
            "cryptarithm.csv": 69,
        },
        "priority": "exp026_s011_priority.txt",
    },
}


def build_assistant(answer: str, cot: str | None) -> str:
    answer = str(answer).strip()
    cot = "" if cot is None else str(cot).strip()
    boxed = f"\\boxed{{{answer}}}"
    if boxed in cot and "</think>" in cot:
        return cot
    cot = re.sub(r"\\boxed\{[^}]*\}", "", cot).strip()
    if "</think>" in cot:
        cot = cot.split("</think>", 1)[0].strip()
    if cot:
        return f"{cot}\n</think>\n{boxed}"
    return boxed


def make_message(prompt: str, answer: str, cot: str | None) -> dict:
    return {
        "messages": [
            {"role": "user", "content": str(prompt).strip() + PROMPT_SUFFIX},
            {"role": "assistant", "content": build_assistant(str(answer), cot)},
        ]
    }


def read_priority_ids(project_dir: Path, priority_name: str | None) -> set[str]:
    if not priority_name:
        return set()
    path = project_dir / "external/datasets/exp024_tong/priority" / priority_name
    if not path.exists():
        print(f"priority file missing: {path}")
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_tong_records(project_dir: Path, variant: str, seed: int) -> list[dict]:
    cfg = TONG_VARIANTS[variant]
    data_dir = project_dir / "external/datasets/exp024_tong/type_tong"
    priority_ids = read_priority_ids(project_dir, cfg["priority"])
    records: list[dict] = []
    id_records: list[tuple[str, dict]] = []
    rng = random.Random(seed)

    for fname, count in cfg["samples"].items():
        path = data_dir / fname
        df = pd.read_csv(path)
        df = df[df["generated_cot"].notna() & (df["generated_cot"].astype(str).str.len() > 5)]
        sample = df.sample(n=min(count, len(df)), random_state=seed)
        print(f"{fname}: {len(sample)}/{count}")
        for _, row in sample.iterrows():
            rec = make_message(row["prompt"], row["answer"], row["generated_cot"])
            records.append(rec)
            id_records.append((str(row.get("id", "")), rec))

    dup_count = 0
    if priority_ids:
        for pid, rec in list(id_records):
            if pid in priority_ids:
                records.append(json.loads(json.dumps(rec)))
                dup_count += 1
        print(f"priority duplicates: {dup_count}")

    rng.shuffle(records)
    return records


def load_dgx_records(project_dir: Path, seed: int) -> list[dict]:
    path = project_dir / "external/datasets/dgx_cot/problem_ids_matched.csv"
    df = pd.read_csv(path)
    cot_col = "generated_cot" if "generated_cot" in df.columns else "cot"
    if cot_col not in df.columns:
        candidates = [c for c in df.columns if "cot" in c.lower() or "reason" in c.lower()]
        if not candidates:
            raise ValueError(f"cannot find CoT column in {path}: {df.columns.tolist()}")
        cot_col = candidates[0]
    df = df[df[cot_col].notna() & (df[cot_col].astype(str).str.len() > 5)]
    records = [make_message(r["prompt"], r["answer"], r[cot_col]) for _, r in df.iterrows()]
    random.Random(seed).shuffle(records)
    print(f"dgx records: {len(records)}")
    return records


def load_konbu_records(project_dir: Path, seed: int) -> list[dict]:
    path = project_dir / "external/datasets/konbu_cot/train_split_with_cot.csv"
    df = pd.read_csv(path)
    cot_col = "cot" if "cot" in df.columns else "generated_cot"
    if cot_col not in df.columns:
        candidates = [c for c in df.columns if "cot" in c.lower() or "reason" in c.lower()]
        if not candidates:
            raise ValueError(f"cannot find CoT column in {path}: {df.columns.tolist()}")
        cot_col = candidates[0]
    df = df[df[cot_col].notna() & (df[cot_col].astype(str).str.len() > 5)]
    records = [make_message(r["prompt"], r["answer"], r[cot_col]) for _, r in df.iterrows()]
    random.Random(seed).shuffle(records)
    print(f"konbu records: {len(records)}")
    return records


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def resolve_model_path(project_dir: Path, model_path: str | None) -> str:
    if model_path:
        return model_path
    import kagglehub

    os.environ.setdefault("KAGGLEHUB_CACHE", str(project_dir / "external/kagglehub"))
    return kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("/home/featurize/nemotron"))
    parser.add_argument("--model-path")
    parser.add_argument("--init-adapter")
    parser.add_argument("--variant", choices=[*TONG_VARIANTS.keys(), "dgx", "konbu"], default="tong_s012")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2.4e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--assistant-only-loss", action="store_true")
    parser.add_argument("--completion-only-loss", action="store_true")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1e9)
    parser.add_argument("--save-embedding-layers", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(args.project_dir / "external/hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.project_dir / "external/hf/transformers"))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.variant.startswith("tong_"):
        records = load_tong_records(args.project_dir, args.variant, args.seed)
    elif args.variant == "dgx":
        records = load_dgx_records(args.project_dir, args.seed)
    else:
        records = load_konbu_records(args.project_dir, args.seed)
    if args.limit_records:
        records = records[: args.limit_records]

    model_path = resolve_model_path(args.project_dir, args.model_path)
    print(f"model path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [apply_chat_template(tokenizer, rec["messages"]) for rec in records]
    dataset = Dataset.from_dict({"text": texts})
    print(f"training texts: {len(dataset)}")

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
    train_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
        max_length=args.max_length,
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        completion_only_loss=args.completion_only_loss if args.completion_only_loss else None,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, save_embedding_layers=args.save_embedding_layers)
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
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "packing": args.packing,
        "assistant_only_loss": args.assistant_only_loss,
        "completion_only_loss": args.completion_only_loss,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "init_adapter": args.init_adapter,
        "save_embedding_layers": args.save_embedding_layers,
        "model_path": model_path,
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
