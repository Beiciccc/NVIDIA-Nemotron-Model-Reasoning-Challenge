
# %% markdown
# ## Acknowledgements
# 
# To start, I’d like to thank **Tong Hui Kang** and **konbu17**.
# 

# %% markdown
# ## Mode Selection

# %% cell 2
# ============================================================
# MODE SELECTION — set exactly one to 1
# ============================================================

import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")

# Mode A: Train LoRA from scratch on Kaggle GPU
TRAIN_ON_KAGGLE = 1

# Mode B: Use pre-trained LoRA weights from dataset and just package them
USE_PRETRAINED = 0

assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1, \
    "Set exactly one of TRAIN_ON_KAGGLE / USE_PRETRAINED to 1."

PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/konbu17/nemotron-sft-lora-cot-selection"
BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

print({
    "TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE,
    "USE_PRETRAINED": USE_PRETRAINED,
    "PRETRAINED_ADAPTER_DATASET_PATH": PRETRAINED_ADAPTER_DATASET_PATH,
})

# %% markdown
# ## Setup & Model Loading

# %% cell 4
import os, glob, sys, subprocess, site

candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)
print("Found Triton wheels:", candidates)

if not candidates:
    raise FileNotFoundError("No Triton wheel found under /kaggle/input")
wheel = candidates[0]

target = "/kaggle/working/pydeps"
os.makedirs(target, exist_ok=True)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--no-deps",
        "--target", target,
        "--upgrade",
        "--ignore-installed",
        wheel,
    ],
    check=True,
)

if target not in sys.path:
    sys.path.insert(0, target)

site.addsitedir(target)

print("Custom target added:", target)

import importlib.util
print("triton spec：", importlib.util.find_spec("triton"))

# %% cell 5
if TRAIN_ON_KAGGLE:
    import sys, os, shutil, stat

    # Add utility script to Python path (provides helper binaries)
    sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')

    # Copy ptxas-blackwell to /tmp with execute permissions
    ptxas_src = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell'
    ptxas_dst = '/tmp/ptxas-blackwell'
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        src_bin = os.path.dirname(ptxas_src)
        dst_bin = '/tmp/triton_nvidia_bin'
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for f in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, f)
            if os.path.isfile(fp):
                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = ptxas_dst

        import triton.backends.nvidia as nv_backend
        nv_backend.__file__ = os.path.join(dst_bin, '..', '__init__.py')
        os.environ['TRITON_PTXAS_PATH'] = ptxas_dst

    import triton.backends.nvidia.compiler as nv_compiler
    nv_compiler.get_ptxas_version = lambda arch: '12.0'

    print('Training environment fixes applied.')
else:
    print("USE_PRETRAINED=1: skipping Triton / ptxas environment fixes.")

# %% cell 6
# trl installation is handled by the Unsloth offline setup cell below.
if TRAIN_ON_KAGGLE:
    print("Skip standalone trl install/import here; the Unsloth setup cell will install compatible packages.")

# %% cell 7
if TRAIN_ON_KAGGLE:
    import glob
    import os
    import subprocess
    import sys

    def recursive_wheels(pattern: str):
        return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

    packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
    all_mamba = recursive_wheels("mamba_ssm-*.whl")
    all_causal = recursive_wheels("causal*conv1d*.whl")

    print("Found mamba wheels:", all_mamba)
    print("Found causal-conv1d wheels:", all_causal)

    import torch
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA:", torch.version.cuda)

    if not torch.cuda.is_available():
        raise RuntimeError("TRAIN_ON_KAGGLE=1 requires a GPU runtime because Nemotron depends on CUDA wheels.")

    if not os.path.isdir(packages_dir):
        raise FileNotFoundError(f"Offline wheel directory not found: {packages_dir}")

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--no-index", "--find-links", packages_dir,
            "unsloth", "trl", "peft", "transformers", "datasets", "accelerate", "bitsandbytes",
        ],
        check=True,
    )

    def pick_last(wheels):
        return wheels[-1] if wheels else None

    causal_wheel = pick_last(all_causal)
    mamba_wheel = pick_last(all_mamba)
    print("Selected causal wheel:", causal_wheel)
    print("Selected mamba wheel:", mamba_wheel)

    if causal_wheel:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", causal_wheel], check=True)
    if mamba_wheel:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", mamba_wheel], check=True)
    else:
        raise FileNotFoundError("Could not find a compatible mamba_ssm wheel under /kaggle/input.")

    print("Offline package installation finished. Restart the kernel if Kaggle keeps stale imports from earlier runs.")
else:
    print("USE_PRETRAINED=1: skipping datasets / trl / mamba_ssm / unsloth installation.")

# %% cell 8
if TRAIN_ON_KAGGLE:
    import torch
    import kagglehub
    from unsloth import FastLanguageModel

    MAX_SEQ_LEN = 8192
    MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"Model path: {MODEL_PATH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded with Unsloth.")
else:
    print("USE_PRETRAINED=1: skipping base model and tokenizer loading.")

# %% cell 9
if TRAIN_ON_KAGGLE:
    from unsloth import FastLanguageModel

    LORA_RANK = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.0
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "up_proj", "down_proj",
    ]

    print("Creating trainable LoRA wrapper via FastLanguageModel.get_peft_model ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()
else:
    print("USE_PRETRAINED=1: skipping trainable LoRA construction.")

# %% markdown
# ## Mode A: Train on Kaggle

# %% cell 11
if TRAIN_ON_KAGGLE:
    import pandas as pd
    import random
    import gc, time
    from datasets import Dataset as HFDataset
    from trl import SFTTrainer, SFTConfig

    SEED = 42
    PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


    DATASET_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
    df = pd.read_csv(DATASET_PATH)
    print(f"Full dataset: {len(df)} rows")


    train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Full dataset: {len(df)} rows")


    import re
    import math
    from collections import defaultdict
    from torch.utils.data import DataLoader, Sampler
    records = []
    record_types = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
        user_content = prompt + PROMPT_SUFFIX
        assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
        records.append({"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]})
        record_types.append(str(row["type"]))
    dataset = HFDataset.from_list(records)
    print(f"SFT records: {len(records)}")

    def formatting_prompts_func(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts


    training_args = SFTConfig(
        output_dir="/kaggle/working/sft_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_length=8192,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        packing=False,
    )

    def build_stratified_index_order(labels, batch_size, seed):
        """Approximate nemotron-master's stratified batching over effective batches."""
        by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[label].append(idx)

        rng = random.Random(seed)
        for idx_list in by_label.values():
            rng.shuffle(idx_list)

        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches = [[] for _ in range(n_batches)]
        batch_order = list(range(n_batches))
        rng.shuffle(batch_order)

        assigned = 0
        for label in sorted(by_label.keys()):
            for idx in by_label[label]:
                batches[batch_order[assigned % n_batches]].append(idx)
                assigned += 1

        order = [idx for batch in batches for idx in batch]
        if len(order) != len(labels):
            raise ValueError("Stratified order size mismatch")
        return order

    class PrecomputedOrderSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            if len(self.stratified_order) != len(self.train_dataset):
                raise ValueError("Stratified order length does not match train dataset")

            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return DataLoader(self.train_dataset, **dataloader_kwargs)

    effective_batch_size = max(
        1,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    )
    stratified_order = build_stratified_index_order(record_types, effective_batch_size, SEED)
    print(f"Approx stratified effective batch size: {effective_batch_size}")
    print("Stratified batching by type:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        stratified_order=stratified_order,
    )

    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")


    ADAPTER_DIR = "/kaggle/working/sft_adapter"
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")



# %% markdown
# ## Mode B: Load Pre-trained LoRA（Temporarily unavailable）

# %% cell 13
if USE_PRETRAINED:
    import os

    SRC_ADAPTER_DIR = PRETRAINED_ADAPTER_DATASET_PATH
    required_files = ["adapter_config.json", "adapter_model.safetensors"]

    print("Using pre-trained adapter from:", SRC_ADAPTER_DIR)
    for fname in required_files:
        fpath = os.path.join(SRC_ADAPTER_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required adapter file: {fpath}")
        print(f"  {fname}: {os.path.getsize(fpath)/1024/1024:.1f} MB")
else:
    print("TRAIN_ON_KAGGLE=1: pretrained adapter path check skipped.")

# %% markdown
# ## Create submission.zip

# %% cell 15
import json, os, shutil, zipfile

OUTPUT_DIR = "/kaggle/working"
SUBMISSION_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "submission_adapter")
os.makedirs(SUBMISSION_ADAPTER_DIR, exist_ok=True)

required_files = ["adapter_config.json", "adapter_model.safetensors"]

if TRAIN_ON_KAGGLE:
    src_adapter_dir = "/kaggle/working/sft_adapter"
    print("Packaging freshly trained adapter from:", src_adapter_dir)
else:
    src_adapter_dir = PRETRAINED_ADAPTER_DATASET_PATH
    print("Packaging pre-trained adapter directly from:", src_adapter_dir)

for fname in required_files:
    src = os.path.join(src_adapter_dir, fname)
    dst = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required adapter file: {src}")
    shutil.copy2(src, dst)
    print(f"Copied {fname} ({os.path.getsize(dst)/1024/1024:.1f} MB)")

config_path = os.path.join(SUBMISSION_ADAPTER_DIR, "adapter_config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in required_files:
        fpath = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
        zf.write(fpath, fname)
        print(f"  Added {fname}")

zip_sz = os.path.getsize(zip_path) / 1024 / 1024
print(f"\nsubmission.zip: {zip_sz:.1f} MB")
print("Done! Ready to submit.")
