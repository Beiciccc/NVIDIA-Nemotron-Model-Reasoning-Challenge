
# %% markdown
# # Nemotron-3-Nano LoRA SFT with CoT-Selected Training Data
# 
# ## Approach
# 

# %% cell 1
# ============================================================
# MODE SELECTION — set exactly one to 1
# ============================================================

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

# %% cell 3
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
print("triton spec:", importlib.util.find_spec("triton"))

# %% cell 4
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


# %% cell 5
# Install trl if needed (for training mode)
if TRAIN_ON_KAGGLE:
    try:
        import trl
        print(f"trl already installed: {trl.__version__}")
    except ImportError:
        # Try offline install first, then online
        offline_path = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages/"
        if os.path.exists(offline_path):
            subprocess.run(f"pip install --no-index --find-links={offline_path} trl", shell=True)
        else:
            subprocess.run("pip install trl", shell=True)
        import trl
        print(f"trl installed: {trl.__version__}")

# %% cell 6
if TRAIN_ON_KAGGLE:
    import glob, importlib.util, os, subprocess, sys, types

    def sh(cmd: str, check: bool = True):
        print("+", cmd)
        return subprocess.run(cmd, shell=True, check=check)

    def find_spec(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def recursive_wheels(pattern: str):
        return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

    all_mamba = recursive_wheels("mamba_ssm-*.whl")
    all_causal = recursive_wheels("causal*conv1d*.whl")
    all_datasets = recursive_wheels("datasets-*.whl")
    all_trl = recursive_wheels("trl-*.whl")
    all_multiprocess = recursive_wheels("multiprocess-*.whl")
    all_dill = recursive_wheels("dill-*.whl")
    all_xxhash = recursive_wheels("xxhash-*.whl")

    print("Found mamba wheels:", all_mamba)
    print("Found causal-conv1d wheels:", all_causal)

    import torch
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    abi_tag = "cxx11abiTRUE" if torch.compiled_with_cxx11_abi() else "cxx11abiFALSE"

    print("Python:", sys.version)
    print("Torch: ", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA:", torch.version.cuda)
    print("Wheel selector:", {"py_tag": py_tag, "torch": torch_mm, "abi": abi_tag})

    if not torch.cuda.is_available():
        raise RuntimeError(
            "TRAIN_ON_KAGGLE=1 requires GPU runtime because mamba_ssm wheel is CUDA-based."
        )

    def pick_best(wheels):
        exact = [w for w in wheels if py_tag in w and f"torch{torch_mm}" in w and abi_tag in w]
        if exact:
            return exact[-1]
        py_only = [w for w in wheels if py_tag in w]
        if py_only:
            return py_only[-1]
        return None

    if not find_spec("datasets"):
        w = pick_best(all_datasets)
        if w:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{w}"')
    if not find_spec("trl"):
        w = pick_best(all_trl)
        if w:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{w}"')
    for pkg, wheels in [("multiprocess", all_multiprocess), ("dill", all_dill), ("xxhash", all_xxhash)]:
        if not find_spec(pkg):
            w = pick_best(wheels)
            if w:
                sh(f'{sys.executable} -m pip install --no-index --no-deps "{w}"', check=False)

    if not find_spec("mamba_ssm"):
        causal_wheel = pick_best(all_causal)
        mamba_wheel = pick_best(all_mamba)

        print("Selected causal wheel:", causal_wheel)
        print("Selected mamba wheel:", mamba_wheel)

        if causal_wheel:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{causal_wheel}"')
        if mamba_wheel:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{mamba_wheel}"')
        else:
            raise FileNotFoundError(
                f"No compatible mamba_ssm wheel found under /kaggle/input for "
                f"py={py_tag}, torch={torch_mm}, abi={abi_tag}."
            )

    import datasets
    import trl

    for _mod_name in [
        'mamba_ssm.modules.mamba3',
        'mamba_ssm.ops.cute',
        'mamba_ssm.ops.cute.mamba3',
        'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
    ]:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)
    sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

    import mamba_ssm

    print(f'datasets:  {datasets.__version__}')
    print(f'trl:       {trl.__version__}')
    print(f'mamba_ssm: {mamba_ssm.__version__}')
else:
    print("USE_PRETRAINED=1: skipping datasets / trl / mamba_ssm installation.")


# %% cell 7
if TRAIN_ON_KAGGLE:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import kagglehub

    MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"Model path: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded.")
else:
    print("USE_PRETRAINED=1: skipping base model/tokenizer loading.")


# %% cell 8
if TRAIN_ON_KAGGLE:
    from peft import LoraConfig, get_peft_model, TaskType

    LORA_RANK = 32
    LORA_ALPHA = 32

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    print("USE_PRETRAINED=1: skipping PEFT model construction.")


# %% markdown
# ## Mode A: Train on Kaggle

# %% cell 10
if TRAIN_ON_KAGGLE:
    import pandas as pd
    import random
    import gc, time
    from datasets import Dataset as HFDataset
    from trl import SFTTrainer, SFTConfig

    SEED = 123
    PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

    # --- Dataset path ---
    DATASET_PATH = "/kaggle/input/datasets/konbu17/nemotron-sft-lora-cot-selection/train_split_with_cot.csv"
    df = pd.read_csv(DATASET_PATH)
    print(f"Full dataset: {len(df)} rows")
    print(df["type"].value_counts().sort_index())

    # --- Type-based sampling ---
    TYPE_SAMPLES = {
        "Numeral Conversion": 300,
        "Gravitational Constant": 400,
        "Unit Conversion": 700,
        "Text Encryption": 700,
        "Bit Manipulation": 607,        # all available
        "Equation Transformation": 200,  # all available
    }

    sampled_dfs = []
    for ptype, n_samples in TYPE_SAMPLES.items():
        subset = df[df["type"] == ptype]
        if n_samples >= len(subset):
            sampled = subset
        else:
            sampled = subset.sample(n=n_samples, random_state=SEED)
        print(f"  {ptype}: {len(subset)} -> {len(sampled)}")
        sampled_dfs.append(sampled)

    train_df = pd.concat(sampled_dfs, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"\nTraining samples: {len(train_df)}")

    # --- Build SFT dataset ---
    import re
    records = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
        user_content = prompt + PROMPT_SUFFIX
        # chat template auto-adds <think>\n, so assistant starts with CoT directly
        assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
        records.append({"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]})
    dataset = HFDataset.from_list(records)
    print(f"SFT records: {len(records)}")

    # --- Training ---
    training_args = SFTConfig(
        output_dir="/kaggle/working/sft_output",
        num_train_epochs=2, # 1 # 2
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4, # 5e-5 # 1e-4
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=4096, # 7680 # 4096
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")

    # Save adapter
    ADAPTER_DIR = "/kaggle/working/sft_adapter"
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")

# %% markdown
# ## Mode B: Load Pre-trained LoRA

# %% cell 12
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

# %% cell 14
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

