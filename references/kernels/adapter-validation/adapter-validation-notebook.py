
# %% markdown
# This is standalone validation notebook that takes in a `submission.zip` and returns the same `submission.zip` along with evaluation metrics.
# 
# These validation concepts were taken from Kh0a's [notebook](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2).
# 

# %% cell 1
SUBMISSION_ZIP_PATH = (
    "/kaggle/input/notebooks/huikang/tinker-submission-notebook/submission.zip"
)
RUN_EVALUATION = True
EVALUATION_SAMPLE_SIZE = 950

# %% cell 2
import zipfile

with zipfile.ZipFile(SUBMISSION_ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall()

# %% markdown
# # Print configs

# %% cell 4
import json

with open("adapter_config.json") as f:
    trained_adapter_config = json.load(f)

print(trained_adapter_config)

# %% markdown
# # Load model

# %% cell 6
"""Metric for NVIDIA (129716)."""

import subprocess
import sys

# Set up environment
commands = [
    "uv pip uninstall torch torchvision torchaudio",
    "tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp",
    "chmod +x /tmp/triton/backends/nvidia/bin/ptxas",
    "chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell",
]
if RUN_EVALUATION:
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
sys.path.insert(0, "/tmp")

# %% cell 7
import glob
import math
import multiprocessing
import os
import re
import time
from pathlib import Path

import kagglehub
import pandas as pd
from tqdm import tqdm

# Configuration
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

# %% cell 8
class ParticipantVisibleError(Exception):
    pass


def cache_model(
    path: str | Path,
    exts: tuple[str, ...] = (".bin", ".pt", ".safetensors"),
    num_workers: int | None = None,
    chunk_mb: int = 256,
) -> int:
    """Pre-read model weight files into the OS page cache to speed up later loads.

    Args:
        path        : Directory containing model files, or a single file path.
        exts        : File extensions treated as model weight files.
        num_workers : Number of threads (default = min(CPU cores, 8)).
        chunk_mb    : Size of each read chunk in MB.

    Returns:
        Total bytes read (int).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath: Path) -> tuple[Path, int]:
        """Sequentially read an entire file in chunks."""
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        try:
            with open(fpath, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    total += len(data)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
        return fpath, total

    path = Path(path)
    # Collect files to read
    files: list[Path] = []
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file() and str(p).endswith(exts)]
        files.sort()
    else:
        files = [path] if path.exists() else []

    if not files:
        print(f"No model files found to cache at: {path}")
        return 0

    # Decide number of worker threads
    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0
    # Read files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {fpath.name}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f"[cache_model] total read ≈ {gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")
    return total_bytes


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return "NOT_FOUND"

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()



def generate_predictions(
    test_df: pd.DataFrame,
    lora_path: str,
    row_id_col: str,
    max_lora_rank: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    debug: bool = False,
) -> pd.DataFrame:
    """Load the model and generate predictions for the provided test data.

    Args:
        debug: If True, writes a CSV file with raw model outputs and extracted predictions.
    """
    # Cache Model
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Initialize vLLM Offline inference Engine
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_df.itertuples(index=False):
        user_content = (
            item.prompt
            + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        )
        # Format using the tokenizer's chat template directly
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            # Fallback if chat template fails
            prompt = user_content
        prompts.append(prompt)

    # Generate predictions using continuous batching
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path),
    )

    predictions = []
    debug_records = []
    for item, output in zip(test_df.itertuples(index=False), outputs):
        raw_text = output.outputs[0].text
        extracted_answer = extract_final_answer(raw_text)

        row_id_val = getattr(item, row_id_col)

        predictions.append(
            {
                row_id_col: row_id_val,
                "prediction": extracted_answer,
            }
        )

        if debug:
            debug_records.append(
                {
                    row_id_col: row_id_val,
                    "raw_output": raw_text,
                    "extracted_prediction": extracted_answer,
                }
            )

    # Write debug CSV if requested
    if debug and debug_records:
        debug_df = pd.DataFrame(debug_records)
        debug_df.to_csv("debug_predictions.csv", index=False)
        print("Debug data saved to debug_predictions.csv")

    return pd.DataFrame(predictions)

# %% cell 9
# Cache Model
if RUN_EVALUATION:
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

# %% markdown
# # Init vLLM

# %% cell 11
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# %% cell 12
# www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/evaluation
max_model_len = 8192
max_lora_rank = 32
max_tokens = 7680
top_p = 1.0
temperature = 0.0
max_num_seqs = 64
gpu_memory_utilization = 0.85
max_model_len = 8192

# %% cell 13
# Initialize vLLM Offline inference Engine

if RUN_EVALUATION:
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=1,
    )

# %% markdown
# # Test generation

# %% cell 15
import pandas as pd

df = pd.read_csv("/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")
df = df.head(EVALUATION_SAMPLE_SIZE).copy()

# %% cell 16
problem_texts = list(df["prompt"])

if RUN_EVALUATION:
    tokenizer = llm.get_tokenizer()
    prompts = []
    for problem_text in problem_texts:
        # Format using the tokenizer's chat template directly
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

# %% cell 17
def detect_category(prompt: str) -> str:
    if "secret bit manipulation rule transforms 8-bit binary numbers" in prompt:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in prompt:
        return "cipher"
    if "secret set of transformation rules is applied to equations" in prompt:
        after_header = prompt.split("Below are a few examples:\n", 1)[1]
        examples_text, rest = after_header.split("\nNow, determine the result for: ", 1)
        question_text = rest.strip()
        if any(c.isdigit() for c in examples_text):
            q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question_text)
            if q_match and re.search(
                r"\d" + re.escape(q_match.group(2)) + r"\d", examples_text
            ):
                return "equation_numeric_deduce"
            return "equation_numeric_guess"
        if len(question_text) == 5:
            q_op = question_text[2]
            for ex_line in examples_text.strip().splitlines():
                inp = ex_line.split(" = ")[0].strip()
                if len(inp) == 5 and inp[2] == q_op:
                    return "cryptarithm_deduce"
        return "cryptarithm_guess"
    if "gravitational constant has been secretly changed" in prompt:
        return "gravity"
    if "converted into a different numeral system" in prompt:
        return "numeral"
    if "secret unit conversion is applied to measurements" in prompt:
        return "unit_conversion"
    raise ValueError("unknown")

# %% cell 18
# Generate predictions using continuous batching
lora_path = "/kaggle/working"

if RUN_EVALUATION:
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path),
    )

# %% markdown
# # Produce submission

# %% cell 20
import zipfile as _zf

print(os.listdir("."))
with _zf.ZipFile("submission.zip", "w", _zf.ZIP_DEFLATED) as zf:
    for file in os.listdir("."):
        if "adapter" not in file:
            continue
        if not os.path.isfile(file):
            continue
        zf.write(file)
        os.remove(file)

# %% cell 21
print(os.listdir("."))

# %% markdown
# # Calculate statistics

# %% cell 23
if RUN_EVALUATION:
    df["output"] = [output.outputs[0].text for output in outputs]
    df["category"] = [detect_category(problem_text) for problem_text in problem_texts]
    df["predicted"] = df["output"].apply(extract_final_answer)
    df["correct"] = df.apply(
        lambda row: verify(str(row["answer"]), str(row["predicted"])), axis=1
    )
    df["minlogprob"] = [
        min(
            lp.logprob
            for token_dict in output.outputs[0].logprobs
            for lp in token_dict.values()
        )
        if output.outputs[0].logprobs
        else None
        for output in outputs
    ]

# %% cell 24
# Save mistakes per category
if RUN_EVALUATION:
    os.makedirs("mistakes", exist_ok=True)
    for category in df["category"].unique():
        cat_mistakes = df[(df["category"] == category) & (~df["correct"])]
        if not cat_mistakes.empty:
            cat_mistakes.to_csv(f"mistakes/{category}.csv", index=False)
    
    # Print results table
    stats = df.groupby("category")["correct"].agg(correct="sum", total="count").sort_index()
    stats["correct"] = stats["correct"].astype("int")
    grand_total = stats["total"].sum()
    stats["weightage"] = (stats["total"] / grand_total * 100).map("{:.1f}%".format)
    stats["percentage"] = (stats["correct"] / stats["total"] * 100).map("{:.1f}%".format)
    stats["contribution"] = (stats["correct"] / grand_total * 100).map("{:.1f}%".format)
    overall_pct = stats["correct"].sum() / grand_total * 100
    totals = pd.DataFrame(
        {
            "correct": [stats["correct"].sum()],
            "total": [grand_total],
            "weightage": ["100.0%"],
            "percentage": [f"{overall_pct:.1f}%"],
            "contribution": [f"{overall_pct:.1f}%"],
        },
        index=["TOTAL"],
    )
    results = pd.concat([stats, totals])
    print(results.to_string())
    results.to_csv("results.csv")
    df.to_csv("validation.csv", index=False)

# %% cell 25

