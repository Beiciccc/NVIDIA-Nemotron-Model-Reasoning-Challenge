from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from remote_train_lora import apply_chat_template, resolve_model_path
from remote_train_masked_lora import OFFICIAL_SUFFIX, build_prompt_text, tokenize_masked


def resolve_eval_model_path(project_dir: Path, model_path: str | None) -> str:
    if model_path:
        return model_path
    local_path = project_dir / "external/hf_models/nemotron_30b_bf16"
    if local_path.exists():
        return str(local_path)
    return resolve_model_path(project_dir, None)


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        text = boxed[-1]
    text = text.replace("<|endoftext|>", "").replace("</s>", "").strip()
    text = text.split("\n")[0].strip()
    text = text.strip("` $")
    return re.sub(r"\s+", " ", text)


def load_eval_records(project_dir: Path, split: str, seed: int, limit: int, boxed_target: bool) -> list[dict]:
    if split == "official_eval":
        path = project_dir / "external/datasets/exp024_tong/eval_split.csv"
    elif split == "official_train":
        path = project_dir / "external/datasets/exp024_tong/train_split.csv"
    else:
        path = project_dir / "data/train.csv"

    df = pd.read_csv(path)
    df = df.sample(frac=1.0, random_state=seed)
    if limit:
        df = df.head(limit)

    records: list[dict] = []
    for _, row in df.iterrows():
        answer = str(row["answer"]).strip()
        assistant = f"\\boxed{{{answer}}}" if boxed_target else answer
        records.append(
            {
                "id": str(row.get("id", "")),
                "type": str(row.get("type", "unknown")),
                "answer": answer,
                "messages": [
                    {"role": "user", "content": str(row["prompt"]).strip() + OFFICIAL_SUFFIX},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    return records


def make_loss_batch(features: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        ids = list(f["input_ids"])
        lab = list(f["labels"])
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device="cuda"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
    }


def compute_loss(model, tokenizer, records: list[dict], max_length: int, batch_size: int) -> dict:
    tokenized = [tokenize_masked(tokenizer, r, max_length) for r in records]
    total_loss = 0.0
    total_tokens = 0
    by_type = defaultdict(lambda: [0.0, 0])

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokenized), batch_size):
            batch_records = records[start : start + batch_size]
            batch_rows = tokenized[start : start + batch_size]
            batch = make_loss_batch(batch_rows, tokenizer.pad_token_id)
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
            for i, rec in enumerate(batch_records):
                n = int(masks[i].sum().item())
                loss_sum = float(losses[i][masks[i]].sum().item()) if n else 0.0
                total_loss += loss_sum
                total_tokens += n
                by_type[rec["type"]][0] += loss_sum
                by_type[rec["type"]][1] += n

    return {
        "answer_nll": total_loss / max(total_tokens, 1),
        "answer_tokens": total_tokens,
        "loss_by_type": {k: v[0] / max(v[1], 1) for k, v in sorted(by_type.items())},
    }


def build_generation_prompts(tokenizer, records: list[dict]) -> list[str]:
    prompts = []
    for rec in records:
        user_content = rec["messages"][0]["content"]
        prompts.append(build_prompt_text(tokenizer, user_content))
    return prompts


def compute_generation_accuracy(
    model,
    tokenizer,
    records: list[dict],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
) -> dict:
    tokenizer.padding_side = "left"
    prompts = build_generation_prompts(tokenizer, records)
    correct = 0
    by_type = defaultdict(lambda: [0, 0])
    examples = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_records = records[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            ).to("cuda")
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
            for i, rec in enumerate(batch_records):
                prompt_width = enc["input_ids"].shape[1]
                generated_ids = out[i, prompt_width:]
                raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
                pred = normalize_answer(raw)
                gold = normalize_answer(rec["answer"])
                ok = pred == gold
                correct += int(ok)
                by_type[rec["type"]][0] += int(ok)
                by_type[rec["type"]][1] += 1
                if len(examples) < 20 and not ok:
                    examples.append(
                        {
                            "id": rec["id"],
                            "type": rec["type"],
                            "gold": gold,
                            "pred": pred,
                            "raw": raw[:300],
                        }
                    )

            print(f"generated {min(start + len(batch_prompts), len(prompts))}/{len(prompts)}", flush=True)

    return {
        "exact": correct / max(len(records), 1),
        "correct": correct,
        "total": len(records),
        "exact_by_type": {k: v[0] / max(v[1], 1) for k, v in sorted(by_type.items())},
        "wrong_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("/home/featurize/nemotron"))
    parser.add_argument("--model-path")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", choices=["official_eval", "official_train", "official_all"], default="official_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--loss-batch-size", type=int, default=16)
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--boxed-target", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generation-limit", type=int, default=0)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(args.project_dir / "external/hf"))

    model_path = resolve_eval_model_path(args.project_dir, args.model_path)
    records = load_eval_records(args.project_dir, args.split, args.seed, args.limit_records, args.boxed_target)
    print(f"records={len(records)} split={args.split} adapter={args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    for name, module in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    result = {
        "adapter": args.adapter,
        "split": args.split,
        "records": len(records),
        "target_format": "boxed" if args.boxed_target else "raw",
    }
    if not args.skip_loss:
        result.update(compute_loss(model, tokenizer, records, args.max_length, args.loss_batch_size))

    if args.generate:
        gen_records = records[: args.generation_limit] if args.generation_limit else records
        result.update(
            compute_generation_accuracy(
                model,
                tokenizer,
                gen_records,
                args.max_length,
                args.max_new_tokens,
                args.gen_batch_size,
            )
        )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
