
# %% cell 0
import polars as pl

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')

train.head()

# %% cell 1
import site

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32  # Can be set to a maximum of 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model loaded successfully.")

# Initialize LoRA Adapter
print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# YOUR CODE HERE
# --------------
# model.train() 
# --------------


# Save Adapter
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

# %% cell 2
import subprocess

subprocess.run("zip -m submission.zip *", shell=True, check=True)

# %% cell 3
print('Done.')
