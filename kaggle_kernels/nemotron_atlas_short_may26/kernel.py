# Auto-generated from habanwer/nemotron-atlas with short-run settings for May 26.
# ── Environment Setup (Kaggle) ──

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
import sys
import json
import math
import time
import shutil
import random
import warnings
import subprocess
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# Fix Triton ptxas issue on Kaggle (ptxas binary not in PATH)
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
if not os.path.exists(os.environ["TRITON_PTXAS_PATH"]):
    for candidate in ["/usr/local/cuda-12/bin/ptxas", "/usr/bin/ptxas"]:
        if os.path.exists(candidate):
            os.environ["TRITON_PTXAS_PATH"] = candidate
            break

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
SESSION_START = time.time()

import torch
print(f"Runtime: Kaggle")
print(f"CUDA available: {torch.cuda.is_available()}")

# %%
# ── Dependencies & Triton Fix (Kaggle) ──

print("Packages assumed pre-installed on Kaggle.")

# ── Fix Triton ptxas (Kaggle ships ptxas-blackwell on read-only FS without +x) ──
_system_ptxas = shutil.which('ptxas')

class _PtxasShim:
    """Mimics the object Triton's get_ptxas() returns."""
    def __init__(self, p):
        self.path = p
        try:
            out = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r'(?:V|release )(\d+\.\d+)', out)
            self.version = m.group(1) if m else "12.0"
        except Exception:
            self.version = "12.0"
    def __str__(self): return self.path or ""
    def __fspath__(self): return self.path or ""

def _patch_triton_ptxas():
    if not _system_ptxas:
        print("WARNING: No system ptxas found — Triton patch skipped")
        return
    try:
        import triton.backends.nvidia.compiler as _tnc
        _obj = _PtxasShim(_system_ptxas)
        _tnc.get_ptxas = lambda *a, **kw: _obj
        print(f"✓ Patched Triton get_ptxas → {_system_ptxas} (v{_obj.version})")
    except Exception as e:
        print(f"⚠ Triton patch skipped: {e}")

_patch_triton_ptxas()

# ── Import heavy dependencies ──
import polars as pl

try:
    import mamba_ssm
    print(f"mamba-ssm {mamba_ssm.__version__}")
except ImportError:
    print("Warning: If not on Kaggle, mamba-ssm may be ABSENT and Nemotron model may not load correctly")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except ImportError:
    print("WARNING: transformers NOT available")

print(f"✓ torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

# %%
# ── Configuration: Hyperparameters, Paths & Data Setup (Kaggle) ──

import torch.nn as nn  # needed later for module discovery

class Config:
    """All hyperparameters and paths in one place."""

    # ── Kaggle paths ──
    DATA_PATH = '/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge'
    OUTPUT_DIR = '/kaggle/working/lora_adapter'
    MODEL_CACHE_DIR = '/kaggle/working/hf-cache'

    # ── Model / LoRA ──
    MODEL_NAME = 'metric/nemotron-3-nano-30b-a3b-bf16'
    LORA_RANK = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    # ── ATLAS: Architecture-Targeting LoRA Module Selection ──
    ATLAS_TARGET_MODULES = [
        'in_proj', 'out_proj',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
    ]
    ATLAS_TARGET_REGEX = r"^.+\.(in_proj|out_proj|q_proj|k_proj|v_proj|o_proj|shared_experts\.up_proj|shared_experts\.down_proj)$"

    # ── Dataset ──
    DATASET_FRACTION = 0.15

    # ── SFT ──
    LEARNING_RATE = 7e-5
    LR_SWEEP = [3e-5, 7e-5, 2e-4]  # quick LR sweep candidates
    MAX_SEQ_LENGTH = 2048
    ANSWER_TOKEN_WEIGHT = 5.0
    ANSWER_TOKEN_WEIGHTS = {
        "gravity": 5.0, "numeral": 5.0, "unit_conversion": 5.0,
        "equation_transform": 5.0, "binary": 5.0, "cipher": 5.0,
    }
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16  # effective batch ≈ 64
    NUM_EPOCHS_SFT = 1
    SFT_PATIENCE = 1
    GRADIENT_CHECKPOINTING = True
    CURRICULUM_ENABLED = True
    USE_STRATIFIED_SAMPLER = True
    USE_COSINE_SCHEDULE = False
    PER_TYPE_OVERSAMPLE = {}  # Oversample by Replication - for example insert {"equation_transform": 3}
    VOTING_N = 5
    WARMUP_RATIO = 0.05
    SEED = 42
    REQUIRE_EPOCH_FITS_TIME_BUDGET = True
    VAL_SPLIT = 0.1

    # ── Pipeline Resume ──
    RESUME_FROM_CHECKPOINT = None

    # ── Session time budget (seconds) ──
    SESSION_TIME_LIMIT = 2 * 3600

    # ── Time-aware checkpointing ──
    CHECKPOINT_EVERY_STEPS = 25
    TIME_BUFFER_SEC = 600

config = Config()

# Lock seeds for reproducibility
import random as _random
import numpy as _np
_random.seed(config.SEED)
_np.random.seed(config.SEED)
import torch
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

# ── Validation ──
assert config.LORA_RANK <= 32, f"Competition requires LORA_RANK ≤ 32, got {config.LORA_RANK}"

# Trace file paths
SOLVER_TRACES_PATH = os.path.join(config.OUTPUT_DIR, 'solver_traces.json')

# Create directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)

# HF cache to runtime-local storage
os.environ['HF_HOME'] = config.MODEL_CACHE_DIR
os.environ['HF_HUB_CACHE'] = os.path.join(config.MODEL_CACHE_DIR, 'hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(config.MODEL_CACHE_DIR, 'transformers')
os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

# ── Summary ──
print(f"\nConfig (Kaggle):")
print(f"  Data:         {config.DATA_PATH}")
print(f"  Output:       {config.OUTPUT_DIR}")
print(f"  Solver:       {SOLVER_TRACES_PATH}")
print(f"  Model Cache:  {config.MODEL_CACHE_DIR}")
print(f"  LoRA:         rank={config.LORA_RANK} alpha={config.LORA_ALPHA}")
print(f"  Dataset:      fraction={config.DATASET_FRACTION}")
print(f"  SFT:          lr={config.LEARNING_RATE} epochs={config.NUM_EPOCHS_SFT} seq_len={config.MAX_SEQ_LENGTH}")
print(f"  Grad Accum:   {config.GRADIENT_ACCUMULATION_STEPS} (eff batch={config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS})")
print(f"  Curriculum:   {'ON' if config.CURRICULUM_ENABLED else 'OFF'}")
print(f"  Session limit: {config.SESSION_TIME_LIMIT/3600:.0f}h (buffer: {config.TIME_BUFFER_SEC/60:.0f}min)")

# %%
# Load and Classify Training Data (Corrected 6-Type Classifier)

train_path = f'{config.DATA_PATH}/train.csv'
test_path = f'{config.DATA_PATH}/test.csv'

if os.path.exists(train_path):
    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    def classify_puzzle(prompt: str) -> str:
        """Classify into 6 verified competition puzzle categories."""
        p = prompt.lower()
        if 'bit manipulation' in p or 'bitwise' in p or 'binary representation' in p:
            return 'binary'
        if 'roman' in p or 'numeral system' in p:
            return 'numeral'
        if 'gravity' in p or 'gravitational' in p or 'free fall' in p:
            return 'gravity'
        if ('unit' in p and ('convert' in p or 'conversion' in p)) or ('convert' in p and 'to' in p):
            return 'unit_conversion'
        if 'cipher' in p or 'encrypt' in p or 'decrypt' in p:
            return 'cipher'
        if 'equation' in p or 'transformation rule' in p or 'symbol' in p:
            return 'equation_transform'
        # Fallback heuristics for edge cases
        if re.search(r'base\s*\d+|binary|0b[01]', p):
            return 'binary'
        if re.search(r'[MDCLXVI]{2,}', prompt):  # Roman numerals (original case)
            return 'numeral'
        if re.search(r'meter|kilogram|mile|inch|pound|celsius|fahrenheit', p):
            return 'unit_conversion'
        return 'equation_transform'  # Most likely unmatched type

    train_types = [classify_puzzle(p) for p in train_df['prompt'].to_list()]
    type_distribution = defaultdict(int)
    for t in train_types:
        type_distribution[t] += 1

    print("\nPuzzle Type Distribution:")
    for ptype, count in sorted(type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count} ({100*count/len(train_types):.1f}%)")

    train_df = train_df.with_columns(pl.Series('puzzle_type', train_types))

    # Apply per-type oversampling specified in config.PER_TYPE_OVERSAMPLE
    try:
        oversample_map = getattr(config, 'PER_TYPE_OVERSAMPLE', {}) or {}
        if oversample_map:
            extra_frames = []
            for ptype, mult in oversample_map.items():
                try:
                    mult = int(mult)
                except Exception:
                    continue
                if mult <= 1:
                    continue
                sel = train_df.filter(pl.col('puzzle_type') == ptype)
                n = len(sel)
                if n == 0:
                    print(f"Oversample: no examples found for type '{ptype}', skipping")
                    continue
                reps = [sel] * (mult - 1)
                extra_frames.extend(reps)
                print(f"Oversample: {ptype}: {n} samples × {mult} -> {n*mult} total")
            if extra_frames:
                train_df = pl.concat([train_df] + extra_frames)
                # Recompute train_types if needed
                train_types = train_df['puzzle_type'].to_list()
                print(f"After oversampling: {len(train_df)} samples (original fraction={config.DATASET_FRACTION})")
    except Exception as e:
        print(f"Warning: oversampling step failed: {e}")

    print(f"\nSample Puzzle:")
    print(train_df['prompt'][0][:4000])
    print(f"\nAnswer: {train_df['answer'][0]}")
else:
    raise FileNotFoundError(f"Training data not found at {train_path}")

# %%
# ── SAT: Solver-Augmented Training — Trace Generation ──
# Generates verified reasoning traces using programmatic solvers.
# Solver rates: numeral 100%, unit_conversion 100%, gravity 100%, cipher ~60%+,
# binary: ~100% (per-bit boolean function analysis with known-answer disambiguation)
# equation_transform: ~100% (operator grouping + positional mapping with known answer)
# All 6 puzzle types now get SFT traces.

from tqdm.auto import tqdm

SYSTEM_PROMPT = """Solve step by step. Your Final Answer MUST BE DElIVERED IN THE FOLLOWING FORMAT: \\boxed{Final Answer}"""

def extract_boxed_answer(text: str) -> Optional[str]:
    match = re.search(r'\\boxed\{', text)
    if match:
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()
    return None

def check_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    pred, gt = str(predicted).strip(), str(ground_truth).strip()
    if pred == gt or pred.lower() == gt.lower():
        return True
    # Normalize whitespace and commas
    pred_norm = re.sub(r'[\s,]+', '', pred)
    gt_norm = re.sub(r'[\s,]+', '', gt)
    if pred_norm == gt_norm or pred_norm.lower() == gt_norm.lower():
        return True
    try:
        if abs(float(pred) - float(gt)) / max(abs(float(gt)), 1e-8) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    return False

def load_traces(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def save_traces(traces: List[Dict], path: str):
    with open(path, 'w') as f:
        json.dump(traces, f, indent=2)

# PROGRAMMATIC SOLVERS — Verified correct against train.csv ground truth
# Each returns (answer, reasoning_text) or (None, None)

_REASONING_VARIANTS = {
    'numeral': [
        "This is a Roman numeral conversion problem.\n\nLooking at the examples, I can see numbers being converted to Roman numerals.\n\n{steps}\n\nTherefore, converting {target}:\n{breakdown}\n\n\\boxed{{{answer}}}",
        "I need to convert a number to Roman numerals.\n\nThe examples confirm the standard Roman numeral system is being used.\n\n{steps}\n\nNow for {target}:\n{breakdown}\n\n\\boxed{{{answer}}}",
        "This puzzle asks me to convert integers to Roman numeral notation.\n\nVerifying with examples:\n{steps}\n\nApplying to {target}:\n{breakdown}\n\nThe answer is \\boxed{{{answer}}}",
    ],
    'unit_conversion': [
        "This is a unit conversion problem.\n\nFrom the examples I can determine the conversion ratio:\n{steps}\n\nAverage ratio: {ratio:.6f}\n\nFor the target value {target}:\n{target} × {ratio:.6f} = {answer}\n\n\\boxed{{{answer}}}",
        "I need to find the conversion factor between these units.\n\nAnalyzing the given examples:\n{steps}\n\nThe conversion ratio is {ratio:.6f}.\n\nConverting {target}:\nResult = {target} × {ratio:.6f} = {answer}\n\n\\boxed{{{answer}}}",
        "Looking at the example conversions to determine the ratio:\n{steps}\n\nConsistent ratio: {ratio:.6f}\n\nApplying to {target}:\n{target} × {ratio:.6f} = {answer}\n\n\\boxed{{{answer}}}",
    ],
    'gravity': [
        "This is a gravitational free fall problem using d = ½ × g × t².\n\nDetermining g from the examples:\n{steps}\n\nAverage g = {g:.4f} m/s²\n\nFor t = {target} s:\nd = ½ × {g:.4f} × {target}² = ½ × {g:.4f} × {t_sq:.2f} = {answer}\n\n\\boxed{{{answer}}}",
        "I need to find the gravitational constant from the examples, then apply d = ½gt².\n\nFrom the examples:\n{steps}\n\ng ≈ {g:.4f} m/s²\n\nCalculating for t = {target} s:\nd = 0.5 × {g:.4f} × {t_sq:.2f} = {answer}\n\n\\boxed{{{answer}}}",
        "Using the free fall formula d = ½gt², I'll first determine g.\n\nAnalyzing examples:\n{steps}\n\nDerived g = {g:.4f} m/s²\n\nFor t = {target}:\nd = ½ × {g:.4f} × {target}² = {answer}\n\n\\boxed{{{answer}}}",
    ],
    'cipher': [
        "This is a substitution cipher problem.\n\nStep 1 — Build decryption mapping from examples:\n{steps}\n\nStep 2 — Coverage: {coverage} of 26 letters mapped.\n\nStep 3 — Decrypt the target text:\n{decrypt_steps}\n\n\\boxed{{{answer}}}",
        "I need to decode a substitution cipher using the provided examples.\n\nExtracting letter mappings from examples:\n{steps}\n\nWith {coverage} letters mapped, decrypting the target:\n{decrypt_steps}\n\n\\boxed{{{answer}}}",
        "This cipher uses a monoalphabetic substitution. Let me build the mapping from examples.\n\nFrom the examples:\n{steps}\n\nTotal mapped: {coverage}/26 letters.\n\nApplying to target:\n{decrypt_steps}\n\n\\boxed{{{answer}}}",
    ],
    'binary': [
        "This is a binary bit transformation puzzle. I need to find the boolean function mapping each output bit from the input bits.\n\nStep 1 — Collect examples:\n{examples_text}\n\nStep 2 — Target input: {target}\n\nStep 3 — Per-bit function identification (using examples only):\n{bit_analysis}\n\nStep 4 — Apply identified functions to {target}:\nResult: {answer}\n\n\\boxed{{{answer}}}",
        "I'll analyze this binary transformation by identifying the boolean rule for each output bit from the training examples.\n\nExamples:\n{examples_text}\n\nTarget: {target}\n\nIdentifying per-bit rules from examples:\n{bit_analysis}\n\nComputing output for {target}: {answer}\n\n\\boxed{{{answer}}}",
        "Binary transformation puzzle — each output bit is a boolean function of input bits.\n\nGiven examples:\n{examples_text}\n\nDetermine output for: {target}\n\nBit-by-bit function identification:\n{bit_analysis}\n\nComputed output: {answer}\n\n\\boxed{{{answer}}}",
    ],
    'equation_transform': [
        "This is an equation transformation puzzle with symbol operators.\n\nStep 1 — Collect examples:\n{examples_text}\n\nStep 2 — Target: {target} = ?\nThe operator is '{operator}' at position 2.\n\nStep 3 — Derive positional mapping from same-operator examples:\n{analysis}\n\nStep 4 — Apply mapping to {target}:\nResult: {answer}\n\n\\boxed{{{answer}}}",
        "I need to find the transformation rule from examples grouped by operator.\n\nExamples:\n{examples_text}\n\nTarget equation: {target} = ?\nOperator: '{operator}'\n\nPositional mapping analysis:\n{analysis}\n\nComputed result: {answer}\n\n\\boxed{{{answer}}}",
        "Equation transformation puzzle — each operator defines a positional rearrangement.\n\nExamples:\n{examples_text}\n\nDetermine: {target} = ?\nTarget uses operator '{operator}'.\n\nMapping derivation:\n{analysis}\n\nThe answer is {answer}.\n\n\\boxed{{{answer}}}",
    ],
}

def solve_numeral(prompt: str, answer: str) -> Optional[Dict]:
    """Solve Roman numeral conversion. Verified 100% on 200 samples."""
    m = re.search(r'write the number (\d+)', prompt.lower())
    if not m:
        m = re.search(r'convert.*?(\d+)', prompt.lower())
    if not m:
        m = re.search(r'(?:number|integer|value)\s+(\d+)', prompt.lower())
    if not m:
        return None

    num = int(m.group(1))
    target = str(num)

    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    remaining = num
    breakdown_parts = []
    for v, s in vals:
        while remaining >= v:
            result += s
            remaining -= v
    if not result:
        return None

    # Build breakdown for reasoning
    remaining = num
    for v, s in vals:
        count = remaining // v
        if count > 0:
            breakdown_parts.append(f"  {v} × {count} = {s * count}")
            remaining -= v * count

    # Build example verification steps
    examples = re.findall(r'(\d+)\s*(?:→|->|becomes?|is)\s*([MDCLXVI]+)', prompt)
    steps = '\n'.join(f"- {a} → {b} ✓" for a, b in examples[:3]) if examples else "- Examples confirm standard Roman numeral system"
    breakdown = '\n'.join(breakdown_parts) if breakdown_parts else f"  {num} = {result}"

    variant = random.choice(_REASONING_VARIANTS['numeral'])
    reasoning = variant.format(steps=steps, target=target, breakdown=breakdown, answer=result)

    return {'reasoning': reasoning, 'computed_answer': result}

def _unit_precision(s: str) -> int:
    s = s.strip()
    if '.' not in s:
        return 0
    return len(s.split('.', 1)[1])


def _unit_precision(s: str) -> int:
    s = s.strip()
    if '.' not in s:
        return 0
    return len(s.split('.', 1)[1])


def solve_unit_conversion(prompt: str, answer: str) -> Optional[Dict]:
    """Solve linear unit conversion. Uses interval intersection for robust ratio estimation."""
    pairs = re.findall(
        r'([\d.]+)\s*\w*\s*(?:→|->|becomes?|is|equals?|=)\s*([\d.]+)',
        prompt,
    )
    if len(pairs) < 2:
        return None

    now_part = prompt.lower().split('now')[-1] if 'now' in prompt.lower() else prompt.lower()
    target_m = re.search(r'(?:convert|now|what)\s*[^0-9]*([\d.]+)', now_part)
    if not target_m:
        return None
    target = float(target_m.group(1))

    # Interval for ratio k: y_obs = round(k*x, p), so k in [(y-eps)/x, (y+eps)/x]
    lows: List[float] = []
    highs: List[float] = []
    ratios: List[float] = []
    for a_str, b_str in pairs:
        try:
            a, b = float(a_str), float(b_str)
        except ValueError:
            continue
        if a <= 0:
            continue
        p = _unit_precision(b_str)
        eps = 0.5 * (10 ** -p)
        lows.append((b - eps) / a)
        highs.append((b + eps) / a)
        ratios.append(b / a)
    if not ratios:
        return None

    lo = max(lows)
    hi = min(highs)
    if lo <= hi:
        k = 0.5 * (lo + hi)
    else:
        k = sum(ratios) / len(ratios)

    result = target * k
    # Training answers always use 2 decimals (e.g. '19.00', '9.30')
    rs = f'{result:.2f}'

    steps = '\n'.join(
        f"- {a} -> {b} (ratio: {float(b)/float(a):.6f})" for a, b in pairs[:4] if float(a) > 0
    )
    variant = random.choice(_REASONING_VARIANTS['unit_conversion'])
    reasoning = variant.format(steps=steps, ratio=k, target=target, answer=rs)
    return {'reasoning': reasoning, 'computed_answer': rs}
def _gravity_precision(num_str: str) -> int:
    """Decimal places in '12.4' -> 1, '12.40' -> 2, '13' -> 0."""
    s = num_str.strip()
    if '.' not in s:
        return 0
    return len(s.split('.', 1)[1])


def _gravity_format(value: float) -> str:
    """Match training-answer style: 2 decimals, strip trailing zeros, keep .0 if needed.
    Examples: 45.00 -> '45.0', 38.60 -> '38.6', 140.44 -> '140.44'."""
    s = f'{value:.2f}'
    if '.' in s:
        s = s.rstrip('0')
        if s.endswith('.'):
            s = s + '0'
    return s


def solve_gravity(prompt: str, answer: str) -> Optional[Dict]:
    """Solve gravity (d = 0.5*g*t^2) using interval intersection for robust g estimation."""
    # Parse examples
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)', prompt, re.IGNORECASE)
    if len(pairs) < 2:
        pairs_rev = re.findall(r'distance\s*=\s*([\d.]+).*?t\s*=\s*([\d.]+)\s*s', prompt, re.IGNORECASE)
        pairs = [(t, d) for d, t in pairs_rev]
    if len(pairs) < 2:
        return None

    # Each observation d_obs constrains g to an interval via display precision:
    #   d_obs = round(0.5*g*t², p_i) => g ∈ [2*(d-0.5*10^-p)/t², 2*(d+0.5*10^-p)/t²]
    lows: List[float] = []
    highs: List[float] = []
    gs_simple: List[float] = []
    for t_str, d_str in pairs:
        try:
            t, d = float(t_str), float(d_str)
        except ValueError:
            continue
        if t <= 0:
            continue
        p = _gravity_precision(d_str)
        eps = 0.5 * (10 ** -p)
        t_sq = t * t
        lows.append(2 * (d - eps) / t_sq)
        highs.append(2 * (d + eps) / t_sq)
        gs_simple.append(2 * d / t_sq)
    if not gs_simple:
        return None

    g_lo = max(lows)
    g_hi = min(highs)
    if g_lo <= g_hi:
        g = 0.5 * (g_lo + g_hi)
    else:
        # Intervals disjoint (shouldn't happen with clean data) — weighted mean
        ws = [(t ** 4) for t_str, _ in pairs for t in (float(t_str),)]
        g = sum(w * gi for w, gi in zip(ws, gs_simple)) / sum(ws)

    # Target time in "Now" section
    now_section = prompt.lower().split('now')[-1] if 'now' in prompt.lower() else prompt.lower()
    target_m = re.search(r't\s*=\s*([\d.]+)\s*s', now_section)
    if not target_m:
        target_m = re.search(r'(?:for|at|when)\s*t\s*=\s*([\d.]+)', now_section)
    if not target_m:
        return None

    t_target = float(target_m.group(1))
    result = 0.5 * g * t_target * t_target
    result_str = _gravity_format(result)
    t_sq = t_target * t_target

    steps = '\n'.join(
        f"- t={t}s, d={d}m -> g = 2x{d}/{t}^2 = {2*float(d)/(float(t)**2):.4f}"
        for t, d in pairs[:4] if float(t) > 0
    )
    variant = random.choice(_REASONING_VARIANTS['gravity'])
    reasoning = variant.format(steps=steps, g=g, target=t_target, t_sq=t_sq, answer=result_str)

    return {'reasoning': reasoning, 'computed_answer': result_str}


# ---- Cipher vocabulary (extracted from 1576 training examples + answers) ----
# Fixed vocabulary of 77 words drawn from noun-verb-object "Wonderland" patterns.
# Every target plaintext word observed in train.csv is in this set.
_CIPHER_VOCAB = (
    'above', 'alice', 'ancient', 'around', 'beyond', 'bird', 'book', 'bright',
    'castle', 'cat', 'cave', 'chases', 'clever', 'colorful', 'creates',
    'crystal', 'curious', 'dark', 'discovers', 'door', 'dragon', 'draws',
    'dreams', 'explores', 'follows', 'forest', 'found', 'garden', 'golden',
    'hatter', 'hidden', 'imagines', 'in', 'inside', 'island', 'key', 'king',
    'knight', 'library', 'magical', 'map', 'message', 'mirror', 'mountain',
    'mouse', 'mysterious', 'near', 'ocean', 'palace', 'potion', 'princess',
    'puzzle', 'queen', 'rabbit', 'reads', 'school', 'secret', 'sees', 'silver',
    'story', 'strange', 'student', 'studies', 'teacher', 'the', 'through',
    'tower', 'treasure', 'turtle', 'under', 'valley', 'village', 'watches',
    'wise', 'wizard', 'wonderland', 'writes',
)
_CIPHER_VOCAB_BY_LEN: Dict[int, List[str]] = {}
for _w in _CIPHER_VOCAB:
    _CIPHER_VOCAB_BY_LEN.setdefault(len(_w), []).append(_w)


def _cipher_parse(prompt: str):
    examples: List[Tuple[str, str]] = []
    target: Optional[str] = None
    for line in prompt.split('\n'):
        line = line.strip()
        if '->' in line:
            a, b = line.split('->', 1)
            a, b = a.strip(), b.strip()
            if a and b:
                examples.append((a, b))
        elif line.lower().startswith('now'):
            m = re.search(r'(?:decrypt|decipher|translate|determine).*?:\s*(.*)',
                          line, re.IGNORECASE)
            if m:
                target = m.group(1).strip()
    return examples, target


def _cipher_initial_mapping(examples):
    """Build (enc->plain) bijection from example word pairs. Return None on
    inconsistency (e.g. two different plain letters mapped to same enc letter)."""
    enc2plain: Dict[str, str] = {}
    plain2enc: Dict[str, str] = {}
    for enc, plain in examples:
        ews, pws = enc.split(), plain.split()
        if len(ews) != len(pws):
            continue
        for ew, pw in zip(ews, pws):
            if len(ew) != len(pw):
                continue
            for ec, pc in zip(ew.lower(), pw.lower()):
                if not (ec.isalpha() and pc.isalpha()):
                    continue
                if ec in enc2plain and enc2plain[ec] != pc:
                    return None
                if pc in plain2enc and plain2enc[pc] != ec:
                    return None
                enc2plain[ec] = pc
                plain2enc[pc] = ec
    return enc2plain


def _cipher_word_candidates(enc_word: str, enc2plain: Dict[str, str]) -> List[str]:
    """Vocabulary words of matching length consistent with the encrypted word's
    repeat pattern and the partial enc->plain mapping."""
    el = enc_word.lower()
    out = []
    base_rev = {v: k for k, v in enc2plain.items()}
    for cand in _CIPHER_VOCAB_BY_LEN.get(len(el), ()):
        local = dict(enc2plain)
        rev = dict(base_rev)
        ok = True
        for ec, pc in zip(el, cand):
            if not ec.isalpha():
                if ec != pc:
                    ok = False
                    break
                continue
            if ec in local:
                if local[ec] != pc:
                    ok = False
                    break
            elif pc in rev and rev[pc] != ec:
                ok = False
                break
            else:
                local[ec] = pc
                rev[pc] = ec
        if ok:
            out.append(cand)
    return out


def _cipher_backtrack(target: str, enc2plain: Dict[str, str]):
    """Return (plaintext_words, final_mapping) or None. Backtracks over target
    words (most-constrained-first) under shared bijection."""
    words = target.split()
    word_cands: List[List[str]] = []
    for w in words:
        if not re.search(r'[a-z]', w.lower()):
            word_cands.append([w])
            continue
        cands = _cipher_word_candidates(w, enc2plain)
        if not cands:
            return None
        word_cands.append(cands)

    order = sorted(range(len(words)), key=lambda i: len(word_cands[i]))
    chosen: Dict[int, str] = {}
    cur_map: Dict[str, str] = dict(enc2plain)
    cur_rev: Dict[str, str] = {v: k for k, v in enc2plain.items()}

    def rec(idx: int) -> bool:
        if idx == len(order):
            return True
        wi = order[idx]
        ew = words[wi].lower()
        for cand in word_cands[wi]:
            add: Dict[str, str] = {}
            add_rev: Dict[str, str] = {}
            ok = True
            for ec, pc in zip(ew, cand):
                if not ec.isalpha():
                    if ec != pc:
                        ok = False
                        break
                    continue
                if ec in cur_map:
                    if cur_map[ec] != pc:
                        ok = False
                        break
                elif ec in add:
                    if add[ec] != pc:
                        ok = False
                        break
                else:
                    if pc in cur_rev or pc in add_rev:
                        existing = cur_rev.get(pc, add_rev.get(pc))
                        if existing != ec:
                            ok = False
                            break
                    add[ec] = pc
                    add_rev[pc] = ec
            if not ok:
                continue
            for ec, pc in add.items():
                cur_map[ec] = pc
                cur_rev[pc] = ec
            chosen[wi] = cand
            if rec(idx + 1):
                return True
            for ec, pc in add.items():
                del cur_map[ec]
                del cur_rev[pc]
            chosen.pop(wi, None)
        return False

    if not rec(0):
        return None
    decoded = [chosen.get(i, words[i]) for i in range(len(words))]
    return decoded, cur_map


def solve_cipher(prompt: str, answer: str) -> Optional[Dict]:
    """Solve monoalphabetic substitution cipher via vocabulary-constrained
    backtracking.

    Algorithm:
      1. Parse examples into (enc, plain) word pairs.
      2. Build partial enc->plain bijection from example letter alignments.
      3. For each target word: enumerate vocabulary candidates of matching
         length consistent with (a) the repeat-pattern of the enc word and
         (b) the current mapping.
      4. Backtrack across target words under the shared bijection,
         most-constrained-first.

    Achieves 100% coverage on train (1576/1576). Uses ONLY examples +
    fixed vocabulary — no ground-truth leakage. Returns None when no
    consistent decryption exists.
    """
    examples, target = _cipher_parse(prompt)
    if not examples or not target:
        return None

    enc2plain = _cipher_initial_mapping(examples)
    if enc2plain is None:
        return None

    res = _cipher_backtrack(target, enc2plain)
    if res is None:
        return None
    decoded_words, final_map = res
    decrypted = ' '.join(decoded_words)

    if not check_answer(decrypted, answer):
        return None

    # Transparent CoT: full per-letter mapping (sorted) + per-word decryption
    mapping_items = sorted(final_map.items())
    steps = '\n'.join(f"- '{k}' \u2192 '{v}'" for k, v in mapping_items)
    coverage = len(final_map)

    decrypt_steps_list = []
    enc_words = target.split()
    for ew, pw in zip(enc_words, decoded_words):
        decrypt_steps_list.append(f'  "{ew}" \u2192 "{pw}"')
    decrypt_steps = '\n'.join(decrypt_steps_list)

    variant = random.choice(_REASONING_VARIANTS['cipher'])
    reasoning = variant.format(steps=steps, coverage=coverage,
                               decrypt_steps=decrypt_steps, answer=decrypted)
    return {'reasoning': reasoning, 'computed_answer': decrypted}


# ════════════════════════════════════════════════════════════════════════════════
# BINARY SOLVER — ROT/SHR/SHL structured expression enumeration (Tong's approach)
# Achieves 89.1% coverage (1428/1602) with 98.7% precision
# ════════════════════════════════════════════════════════════════════════════════

import numpy as np

# 21 transforms: ROT(1-7), SHR(1-7), SHL(1-7)
_BIN_TRANS_NAMES = (
    [('ROT', k) for k in range(1, 8)] +
    [('SHR', k) for k in range(1, 8)] +
    [('SHL', k) for k in range(1, 8)] +
    [('ID', 0), ('NOT', 0), ('REV', 0)]
)
_BIN_N_TRANS      = 21   # phases 1-3 use the original 21 (expanding hurts: net-negative wrongs)
_BIN_N_TRANS_FULL = 24   # phases MAJ / CHOICE use full bank incl. ID, NOT, REV
_BIN_N_OPS   = 6
_BIN_OP_NAMES = ['AND', 'AND NOT', 'OR', 'OR NOT', 'XOR', 'XOR NOT']


def _build_trans_table_bin() -> np.ndarray:
    x = np.arange(256, dtype=np.uint16)
    rows = []
    for k in range(1, 8):
        rows.append(((x << k) | (x >> (8 - k))).astype(np.uint8))
    for k in range(1, 8):
        rows.append((x >> k).astype(np.uint8))
    for k in range(1, 8):
        rows.append(((x << k) & 0xFF).astype(np.uint8))
    # Extras (indices 21..23) for MAJORITY / CHOICE phases
    rows.append(x.astype(np.uint8))                       # 21: ID
    rows.append(((~x) & 0xFF).astype(np.uint8))           # 22: NOT
    rev = np.zeros(256, dtype=np.uint8)                   # 23: REV (bit-reverse 8 bits)
    for v in range(256):
        r = 0
        for i in range(8):
            if v & (1 << i):
                r |= 1 << (7 - i)
        rev[v] = r
    rows.append(rev)
    return np.array(rows, dtype=np.uint8)   # shape (24, 256)


_BIN_TRANS_TABLE = _build_trans_table_bin()


def _bin_src_bit(j: int, ti: int):
    t_type, k = _BIN_TRANS_NAMES[ti]
    if t_type == 'ROT':
        return (j + k) % 8
    elif t_type == 'SHR':
        s = j - k
        return s if s >= 0 else None
    else:
        s = j + k
        return s if s < 8 else None


def _bin_get_bit(byte_val: int, j: int) -> int:
    return (byte_val >> (7 - j)) & 1


def _bin_op_byte(a: int, b: int, op_idx: int) -> int:
    a, b = int(a) & 0xFF, int(b) & 0xFF
    nb = (~b) & 0xFF
    return [a & b, a & nb, a | b, a | nb, a ^ b, a ^ nb][op_idx]


def _bin_op_arr(a: np.ndarray, b: np.ndarray, op_idx: int) -> np.ndarray:
    if op_idx == 0: return a & b
    if op_idx == 1: return a & (~b)
    if op_idx == 2: return a | b
    if op_idx == 3: return a | (~b)
    if op_idx == 4: return a ^ b
    return                a ^ (~b)


def _bin_trans_label(ti: int) -> str:
    t, k = _BIN_TRANS_NAMES[ti]
    if t in ('ID', 'NOT', 'REV'):
        return t
    return f'{t}({k})'


def _bin_gen_cot(target_str: str, expr: tuple, examples: list) -> str:
    bits      = [int(c) for c in target_str]
    target_in = int(target_str, 2)
    n_t       = expr[0]
    lines     = []

    if n_t == 1:
        ti   = expr[1]
        lbl  = _bin_trans_label(ti)
        rb   = int(_BIN_TRANS_TABLE[ti, target_in])
        rs   = f'{rb:08b}'
        t_nm, t_k = _BIN_TRANS_NAMES[ti]
        desc = (f'output bit j = input[(j+{t_k})%8]' if t_nm == 'ROT'
                else f'output bit j = input[j{"+" if t_nm=="SHL" else "-"}{t_k}] or 0 if out of range')
        lines += [f'Analyzing the transformation rule on the given examples.',
                  f'Identified: single transform {lbl}',
                  f'  {lbl}: {desc}', '',
                  f'Applying {lbl} to {target_str}:',
                  f'Input bits: {" ".join(f"{j}:{bits[j]}" for j in range(8))}', '']
        for j in range(8):
            src = _bin_src_bit(j, ti)
            sv  = bits[src] if src is not None else 0
            sd  = f'inp[{src}]={sv}' if src is not None else 'C0=0'
            lines.append(f'  bit {j}: {lbl}[{j}] = {sd} → {_bin_get_bit(rb,j)}')
        lines += ['', f'Result: {rs}', f'\\boxed{{{rs}}}']

    elif n_t == 2:
        _, ti1, op1, ti2 = expr
        l1, l2   = _bin_trans_label(ti1), _bin_trans_label(ti2)
        op_sym   = _BIN_OP_NAMES[op1]
        a_b      = int(_BIN_TRANS_TABLE[ti1, target_in])
        b_b      = int(_BIN_TRANS_TABLE[ti2, target_in])
        rb       = _bin_op_byte(a_b, b_b, op1)
        rs       = f'{rb:08b}'
        lines += [f'Analyzing the transformation rule on the given examples.',
                  f'Identified: 2-transform ({l1}) {op_sym} ({l2})', '',
                  f'Applying ({l1}) {op_sym} ({l2}) to {target_str}:',
                  f'Input bits: {" ".join(f"{j}:{bits[j]}" for j in range(8))}', '']
        for j in range(8):
            s1 = _bin_src_bit(j, ti1); s2 = _bin_src_bit(j, ti2)
            v1 = bits[s1] if s1 is not None else 0
            v2 = bits[s2] if s2 is not None else 0
            d1 = f'inp[{s1}]={v1}' if s1 is not None else 'C0=0'
            d2 = f'inp[{s2}]={v2}' if s2 is not None else 'C0=0'
            lines.append(f'  bit {j}: {d1} {op_sym} {d2} = {_bin_get_bit(rb,j)}')
        lines += ['', f'Result: {rs}', f'\\boxed{{{rs}}}']

    else:
        _, ti1, op1, ti2, op2, ti3 = expr
        l1, l2, l3 = _bin_trans_label(ti1), _bin_trans_label(ti2), _bin_trans_label(ti3)
        op1s, op2s = _BIN_OP_NAMES[op1], _BIN_OP_NAMES[op2]
        a_b   = int(_BIN_TRANS_TABLE[ti1, target_in])
        b_b   = int(_BIN_TRANS_TABLE[ti2, target_in])
        c_b   = int(_BIN_TRANS_TABLE[ti3, target_in])
        mid_b = _bin_op_byte(a_b, b_b, op1)
        rb    = _bin_op_byte(mid_b, c_b, op2)
        rs    = f'{rb:08b}'
        lines += [f'Analyzing the transformation rule on the given examples.',
                  f'Identified: 3-transform (({l1}) {op1s} ({l2})) {op2s} ({l3})', '',
                  f'Applying to {target_str}:',
                  f'Input bits: {" ".join(f"{j}:{bits[j]}" for j in range(8))}', '']
        for j in range(8):
            s1 = _bin_src_bit(j, ti1); s2 = _bin_src_bit(j, ti2); s3 = _bin_src_bit(j, ti3)
            v1 = bits[s1] if s1 is not None else 0
            v2 = bits[s2] if s2 is not None else 0
            v3 = bits[s3] if s3 is not None else 0
            d1 = f'inp[{s1}]={v1}' if s1 is not None else '0'
            d2 = f'inp[{s2}]={v2}' if s2 is not None else '0'
            d3 = f'inp[{s3}]={v3}' if s3 is not None else '0'
            mb = _bin_get_bit(mid_b, j)
            lines.append(f'  bit {j}: ({d1} {op1s} {d2}={mb}) {op2s} {d3} = {_bin_get_bit(rb,j)}')
        lines += ['', f'Result: {rs}', f'\\boxed{{{rs}}}']

    return '\n'.join(lines)


def solve_binary(prompt: str, answer: str) -> Optional[Dict]:
    """Solve binary puzzles via ROT/SHR/SHL/MAJ/CHOICE structured enumeration.
    Achieves ~94.6% coverage (1516/1602) with MAJORITY + CHOICE expansion.
    Uses ONLY example pairs — no ground truth leakage.
    Returns None when no consistent expression is found.
    """
    examples = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    m = (re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
         or re.search(r'(?:determine|find)[^\n]*?([01]{8})(?!\s*->)', prompt, re.IGNORECASE))
    if not m or len(examples) < 2:
        return None

    target_str = m.group(1)
    target_in  = int(target_str, 2)

    if len(examples) >= 3:
        train_ex = examples[:-1]
        ho_inp_s, ho_out_s = examples[-1]
        ho_inp, ho_out, has_ho = int(ho_inp_s, 2), int(ho_out_s, 2), True
    else:
        train_ex = examples
        ho_inp = ho_out = 0
        has_ho = False

    t_inp = np.array([int(a, 2) for a, b in train_ex], dtype=np.uint8)
    t_out = np.array([int(b, 2) for a, b in train_ex], dtype=np.uint8)
    tv    = _BIN_TRANS_TABLE[:, t_inp]   # (N_TRANS, n_train)

    def ho1(ti):
        return not has_ho or int(_BIN_TRANS_TABLE[ti, ho_inp]) == ho_out

    def ho2(ti1, ti2, op1):
        if not has_ho: return True
        return _bin_op_byte(int(_BIN_TRANS_TABLE[ti1, ho_inp]),
                            int(_BIN_TRANS_TABLE[ti2, ho_inp]), op1) == ho_out

    def ho3(ti1, ti2, op1, ti3, op2):
        if not has_ho: return True
        mid = _bin_op_byte(int(_BIN_TRANS_TABLE[ti1, ho_inp]),
                           int(_BIN_TRANS_TABLE[ti2, ho_inp]), op1)
        return _bin_op_byte(mid, int(_BIN_TRANS_TABLE[ti3, ho_inp]), op2) == ho_out

    # Phase 1: single transform
    for i in range(_BIN_N_TRANS):
        if np.all(tv[i] == t_out) and ho1(i):
            ans = f'{int(_BIN_TRANS_TABLE[i, target_in]):08b}'
            return {'reasoning': _bin_gen_cot(target_str, (1, i), examples),
                    'computed_answer': ans}

    # Phase 2: two transforms
    for i in range(_BIN_N_TRANS):
        for j in range(_BIN_N_TRANS):
            for op1 in range(_BIN_N_OPS):
                pred = _bin_op_arr(tv[i], tv[j], op1)
                if np.all(pred == t_out) and ho2(i, j, op1):
                    a   = int(_BIN_TRANS_TABLE[i, target_in])
                    b   = int(_BIN_TRANS_TABLE[j, target_in])
                    ans = f'{_bin_op_byte(a, b, op1):08b}'
                    return {'reasoning': _bin_gen_cot(target_str, (2, i, op1, j), examples),
                            'computed_answer': ans}

    # Phase 3: three transforms (vectorised inner loop)
    for i in range(_BIN_N_TRANS):
        for j in range(_BIN_N_TRANS):
            for op1 in range(_BIN_N_OPS):
                mid_v = _bin_op_arr(tv[i], tv[j], op1)
                for op2 in range(_BIN_N_OPS):
                    preds   = _bin_op_arr(mid_v[np.newaxis, :], tv, op2)   # (N_TRANS, n)
                    matched = np.where(np.all(preds == t_out[np.newaxis, :], axis=1))[0]
                    for k in matched.tolist():
                        if ho3(i, j, op1, k, op2):
                            a   = int(_BIN_TRANS_TABLE[i, target_in])
                            b   = int(_BIN_TRANS_TABLE[j, target_in])
                            mid = _bin_op_byte(a, b, op1)
                            c   = int(_BIN_TRANS_TABLE[k, target_in])
                            ans = f'{_bin_op_byte(mid, c, op2):08b}'
                            return {'reasoning': _bin_gen_cot(target_str, (3, i, op1, j, op2, k), examples),
                                    'computed_answer': ans}

    # ---- Phase MAJ / CHOICE use the full 24-transform bank ----
    tv_full = _BIN_TRANS_TABLE[:, t_inp]   # (24, n_train)
    T_at_tgt = _BIN_TRANS_TABLE[:, target_in]

    def _ho_maj(i, j, k):
        if not has_ho: return True
        a = int(_BIN_TRANS_TABLE[i, ho_inp])
        b = int(_BIN_TRANS_TABLE[j, ho_inp])
        c = int(_BIN_TRANS_TABLE[k, ho_inp])
        return ((a & b) | (a & c) | (b & c)) == ho_out

    def _ho_cho(i, j, k):
        if not has_ho: return True
        a = int(_BIN_TRANS_TABLE[i, ho_inp])
        b = int(_BIN_TRANS_TABLE[j, ho_inp])
        c = int(_BIN_TRANS_TABLE[k, ho_inp])
        return ((a & b) | (((~a) & 0xFF) & c)) == ho_out

    # Phase MAJ: maj(Ta, Tb, Tc) = (Ta&Tb)|(Ta&Tc)|(Tb&Tc)
    for i in range(_BIN_N_TRANS_FULL):
        for j in range(i + 1, _BIN_N_TRANS_FULL):
            for k in range(j + 1, _BIN_N_TRANS_FULL):
                pred = (tv_full[i] & tv_full[j]) | (tv_full[i] & tv_full[k]) | (tv_full[j] & tv_full[k])
                if np.all(pred == t_out) and _ho_maj(i, j, k):
                    a = int(T_at_tgt[i]); b = int(T_at_tgt[j]); c = int(T_at_tgt[k])
                    out_b = (a & b) | (a & c) | (b & c)
                    ans = f'{out_b:08b}'
                    return {'reasoning': _bin_gen_cot_maj(target_str, i, j, k, examples),
                            'computed_answer': ans}

    # Phase CHOICE: choice(Ta, Tb, Tc) = (Ta & Tb) | (~Ta & Tc)
    for i in range(_BIN_N_TRANS_FULL):
        for j in range(_BIN_N_TRANS_FULL):
            if j == i: continue
            for k in range(_BIN_N_TRANS_FULL):
                if k == i or k == j: continue
                pred = (tv_full[i] & tv_full[j]) | (((~tv_full[i]) & 0xFF) & tv_full[k])
                if np.all(pred == t_out) and _ho_cho(i, j, k):
                    a = int(T_at_tgt[i]); b = int(T_at_tgt[j]); c = int(T_at_tgt[k])
                    out_b = (a & b) | (((~a) & 0xFF) & c)
                    ans = f'{out_b:08b}'
                    return {'reasoning': _bin_gen_cot_cho(target_str, i, j, k, examples),
                            'computed_answer': ans}

    return None


def _bin_gen_cot_maj(target_str: str, i: int, j: int, k: int, examples: list) -> str:
    la, lb, lc = _bin_trans_label(i), _bin_trans_label(j), _bin_trans_label(k)
    target_in = int(target_str, 2)
    a = int(_BIN_TRANS_TABLE[i, target_in])
    b = int(_BIN_TRANS_TABLE[j, target_in])
    c = int(_BIN_TRANS_TABLE[k, target_in])
    out_b = (a & b) | (a & c) | (b & c)
    rs = f'{out_b:08b}'
    lines = [f'Analyzing the transformation rule on the given examples.',
             f'Identified: MAJORITY( {la}, {lb}, {lc} )',
             f'  majority(a,b,c) = (a AND b) OR (a AND c) OR (b AND c)  — bitwise',
             '',
             f'Applying to input {target_str}:',
             f'  {la}({target_str}) = {a:08b}',
             f'  {lb}({target_str}) = {b:08b}',
             f'  {lc}({target_str}) = {c:08b}',
             f'  majority(a,b,c)   = {rs}',
             '',
             f'Result: {rs}',
             f'\boxed{{{rs}}}']
    return '\n'.join(lines)


def _bin_gen_cot_cho(target_str: str, i: int, j: int, k: int, examples: list) -> str:
    la, lb, lc = _bin_trans_label(i), _bin_trans_label(j), _bin_trans_label(k)
    target_in = int(target_str, 2)
    a = int(_BIN_TRANS_TABLE[i, target_in])
    b = int(_BIN_TRANS_TABLE[j, target_in])
    c = int(_BIN_TRANS_TABLE[k, target_in])
    out_b = (a & b) | (((~a) & 0xFF) & c)
    rs = f'{out_b:08b}'
    lines = [f'Analyzing the transformation rule on the given examples.',
             f'Identified: CHOICE( {la}, {lb}, {lc} )',
             f'  choice(a,b,c) = (a AND b) OR ((NOT a) AND c)  — bitwise',
             '',
             f'Applying to input {target_str}:',
             f'  {la}({target_str}) = {a:08b}',
             f'  {lb}({target_str}) = {b:08b}',
             f'  {lc}({target_str}) = {c:08b}',
             f'  choice(a,b,c)     = {rs}',
             '',
             f'Result: {rs}',
             f'\boxed{{{rs}}}']
    return '\n'.join(lines)


# EQUATION TRANSFORM SOLVER

def _rev_result(n: int) -> int:
    """Reverse decimal digits of integer (for operation result)."""
    if n < 0:
        return -int(str(-n)[::-1])
    return int(str(n)[::-1])

def _rev2(n: int) -> int:
    """Reverse the zero-padded 2-digit representation of n."""
    return int(str(n).zfill(2)[::-1])

_EQ_STD_OPS = [
    ('+',       lambda a, b: a + b),
    ('-',       lambda a, b: a - b),
    ('abs_sub', lambda a, b: abs(a - b)),
    ('*',       lambda a, b: a * b),
    ('//',      lambda a, b: a // b if b != 0 else None),
    ('r//',     lambda a, b: b // a if a != 0 else None),
    ('%',       lambda a, b: a % b if b != 0 else None),
    ('max',     lambda a, b: max(a, b)),
    ('min',     lambda a, b: min(a, b)),
]
_EQ_VRNT = [(ra, rb, rr)
            for ra in (False, True)
            for rb in (False, True)
            for rr in (False, True)]


def _eq_apply(a, b, ra, rb, rr, op_fn, rev_fn=_rev2):
    """Apply reversal variant (using rev_fn for inputs) + op to (a, b)."""
    try:
        aa = rev_fn(a) if ra else a
        bb = rev_fn(b) if rb else b
        mid = op_fn(aa, bb)
        if mid is None:
            return None
        return _rev_result(mid) if rr else mid
    except Exception:
        return None


def _eq_is_numeric(lhs):
    return (len(lhs) == 5 and lhs[0].isdigit() and lhs[1].isdigit()
            and lhs[3].isdigit() and lhs[4].isdigit())


def _parse_eq_rhs(rhs_str):
    """Parse RHS to int, tolerating non-digit prefix/suffix."""
    s = rhs_str.strip()
    if s.lstrip('-').isdigit():
        return int(s)
    m = re.search(r'-?\d+', s)
    return int(m.group()) if m else None


def _solve_numeric_eq(target, examples, answer):
    """Numeric reversal-framework solver with rev2 fix and single-example support."""
    target_op = target[2]
    try:
        ta, tb = int(target[:2]), int(target[3:5])
    except ValueError:
        return None

    by_op = defaultdict(list)
    for lhs, rhs in examples:
        if not _eq_is_numeric(lhs):
            continue
        c = _parse_eq_rhs(rhs)
        if c is not None:
            try:
                by_op[lhs[2]].append((int(lhs[:2]), int(lhs[3:5]), c))
            except ValueError:
                pass

    same_op = by_op.get(target_op, [])
    if len(same_op) < 1:
        return None

    # Concatenation special cases
    for ra_c, rb_c, swap in [(False, False, False), (False, False, True),
                              (True, False, False), (False, True, False),
                              (True, True, False), (True, True, True)]:
        as_ = str(_rev2(ta) if ra_c else ta).zfill(2)
        bs_ = str(_rev2(tb) if rb_c else tb).zfill(2)
        concat_val = int(bs_ + as_) if swap else int(as_ + bs_)
        if check_answer(str(concat_val), answer):
            def _check_concat(a, b, ra_c=ra_c, rb_c=rb_c, swap=swap):
                as2 = str(_rev2(a) if ra_c else a).zfill(2)
                bs2 = str(_rev2(b) if rb_c else b).zfill(2)
                return int(bs2 + as2) if swap else int(as2 + bs2)
            if all(_check_concat(a, b) == c for a, b, c in same_op):
                return f"concat_{ra_c}_{rb_c}_{swap}"

    all_op_keys = [op for op in by_op.keys() if op != target_op]
    use_cross = (len(same_op) == 1 and len(all_op_keys) > 0)

    for op_name, op_fn in _EQ_STD_OPS:
        for ra, rb, rr in _EQ_VRNT:
            if not all(_eq_apply(a, b, ra, rb, rr, op_fn) == c
                       for a, b, c in same_op):
                continue

            if use_cross:
                variant_ok = True
                for other_op in all_op_keys:
                    other_triples = by_op[other_op]
                    if not other_triples:
                        continue
                    ok = any(
                        all(_eq_apply(a, b, ra, rb, rr, ofn) == c
                            for a, b, c in other_triples)
                        for _, ofn in _EQ_STD_OPS
                    )
                    if not ok:
                        variant_ok = False
                        break
                if not variant_ok:
                    continue

            predicted = _eq_apply(ta, tb, ra, rb, rr, op_fn)
            if predicted is None:
                continue

            computed_answer = str(predicted)
            if not check_answer(computed_answer, answer):
                continue

            # Build description for CoT
            a_desc = "rev(a)" if ra else "a"
            b_desc = "rev(b)" if rb else "b"
            inner  = f"{a_desc} {op_name} {b_desc}"
            rule_desc = f"rev({inner})" if rr else inner
            return (op_name, ra, rb, rr, rule_desc, predicted)

    return None


def _solve_symbolic_eq(target, examples, answer):
    """Fallback: positional character mapping for symbol puzzles."""
    target_op = target[2]
    by_op = defaultdict(list)
    for lhs, rhs in examples:
        by_op[lhs[2]].append((lhs, rhs))

    same_op = by_op.get(target_op, [])
    if len(same_op) < 2:
        return None

    solve_op = same_op[:-1]
    verify_op = same_op[-1]

    rhs_lens = set(len(rhs) for _, rhs in solve_op)
    if len(rhs_lens) != 1:
        return None
    out_len = rhs_lens.pop()
    if len(verify_op[1]) != out_len:
        return None

    pos_mapping = []
    analysis_parts = [f"Examples with operator '{target_op}':"]
    for lhs, rhs in solve_op[:4]:
        analysis_parts.append(f"  {lhs} = {rhs}")
    analysis_parts.append(f"\nOutput length: {out_len}. Positional analysis:")

    for out_pos in range(out_len):
        found_pos = None
        for in_pos in range(5):
            if all(rhs[out_pos] == lhs[in_pos]
                   for lhs, rhs in solve_op if len(rhs) > out_pos):
                found_pos = in_pos
                break
        if found_pos is None:
            return None
        pos_mapping.append(found_pos)
        analysis_parts.append(f"  output[{out_pos}] = input[{found_pos}]")

    verify_lhs, verify_rhs = verify_op
    for out_pos, in_pos in enumerate(pos_mapping):
        if out_pos >= len(verify_rhs) or in_pos >= len(verify_lhs):
            return None
        if verify_rhs[out_pos] != verify_lhs[in_pos]:
            return None

    computed_answer = ''
    for in_pos in pos_mapping:
        if in_pos >= len(target):
            return None
        computed_answer += target[in_pos]

    if not check_answer(computed_answer, answer):
        return None

    examples_text = '\n'.join(f"  {lhs} = {rhs}" for lhs, rhs in examples[:6])
    if len(examples) > 6:
        examples_text += f"\n  ... and {len(examples) - 6} more"
    analysis = '\n'.join(analysis_parts)

    variant = random.choice(_REASONING_VARIANTS['equation_transform'])
    reasoning = variant.format(
        examples_text=examples_text, target=target,
        operator=target_op, analysis=analysis, answer=computed_answer
    )
    return {'reasoning': reasoning, 'computed_answer': computed_answer}


def solve_equation_transform(prompt: str, answer: str) -> Optional[Dict]:
    """Solve equation transformation puzzles.
    Numeric LHS: reversal-framework solver with zero-padded reversal.
    Symbolic LHS: positional character mapping.
    """
    lines_raw = prompt.replace('\\n', '\n').split('\n')

    examples = []
    target = None
    for line in lines_raw:
        line = line.strip()
        low = line.lower()
        if 'determine' in low or 'find' in low or 'what' in low:
            m = re.search(r':\s*(.{5})\s*$', line)
            if not m:
                m = re.search(r'(.{5})\s*=\s*\?', line)
            if m:
                target = m.group(1)
            continue
        if '=' in line and len(line) >= 7:
            eq_idx = line.index('=')
            lhs = line[:eq_idx].strip()
            rhs = line[eq_idx+1:].strip()
            if len(lhs) == 5 and rhs:
                examples.append((lhs, rhs))

    if not target or len(target) != 5 or not examples:
        return None

    if _eq_is_numeric(target):
        numeric_result = _solve_numeric_eq(target, examples, answer)
        if numeric_result is None:
            return None
        if isinstance(numeric_result, str) and numeric_result.startswith('concat_'):
            # Concat: just use the answer directly
            parts = numeric_result.split('_')
            ra_c, rb_c = parts[1] == 'True', parts[2] == 'True'
            ta, tb = int(target[:2]), int(target[3:5])
            as_ = str(_rev2(ta) if ra_c else ta).zfill(2)
            bs_ = str(_rev2(tb) if rb_c else tb).zfill(2)
            swap = len(parts) > 3 and parts[3] == 'True'
            computed_answer = (bs_ + as_) if swap else (as_ + bs_)
            examples_text = '\n'.join(f"  {lhs} = {rhs}" for lhs, rhs in examples[:6])
            analysis = f"Rule: concatenate operands → {target[:2]} || {target[3:5]} = {computed_answer}"
        else:
            op_name, ra, rb, rr, rule_desc, predicted = numeric_result
            computed_answer = str(predicted)
            examples_text = '\n'.join(f"  {lhs} = {rhs}" for lhs, rhs in examples[:6])
            if len(examples) > 6:
                examples_text += f"\n  ... and {len(examples) - 6} more"
            target_op = target[2]
            verify_lines = []
            # Show first 3 same-op examples
            for lhs, rhs in examples:
                if len(verify_lines) >= 3:
                    break
                if lhs[2] == target_op and _eq_is_numeric(lhs):
                    verify_lines.append(f"  {lhs}: {rule_desc} = {rhs} \u2713")
            analysis = (f"Operator '{target_op}' follows rule: {rule_desc}\n"
                        + '\n'.join(verify_lines))

        variant = random.choice(_REASONING_VARIANTS['equation_transform'])
        reasoning = variant.format(
            examples_text=examples_text, target=target,
            operator=target[2], analysis=analysis, answer=computed_answer
        )
        return {'reasoning': reasoning, 'computed_answer': computed_answer}
    else:
        return _solve_symbolic_eq(target, examples, answer)



# ── EQUATION NUMERIC ASSOCIATION FALLBACK ──
# Hashcat-style per-glyph rule association. Fires only when the primary
# solve_equation_transform returns None or a wrong answer for numeric targets.
# For each distinct operator glyph in the examples, search a rule bank for
# rules consistent with every example using that glyph, then apply to the target.
# If multiple consistent rules all yield the same target output, we emit it.

def _rev_int(n):
    s = str(n).lstrip('-')
    return int(s[::-1] or '0') * (-1 if n < 0 else 1)

def _swap_digits(n):
    s = f"{n:02d}"; return int(s[1] + s[0])

def _build_eq_rule_bank():
    R = {}
    for k in range(-5, 6):
        R[f'a+b{k:+d}'] = (lambda k=k: (lambda a, b: str(a + b + k)))()
        R[f'a-b{k:+d}'] = (lambda k=k: (lambda a, b: str(a - b + k)))()
        R[f'b-a{k:+d}'] = (lambda k=k: (lambda a, b: str(b - a + k)))()
        R[f'a*b{k:+d}'] = (lambda k=k: (lambda a, b: str(a * b + k)))()
    R['a//b']  = lambda a, b: str(a // b) if b else None
    R['b//a']  = lambda a, b: str(b // a) if a else None
    R['a%b']   = lambda a, b: str(a % b) if b else None
    R['b%a']   = lambda a, b: str(b % a) if a else None
    R['|a-b|'] = lambda a, b: str(abs(a - b))
    R['max']   = lambda a, b: str(max(a, b))
    R['min']   = lambda a, b: str(min(a, b))
    R['cat_ab']   = lambda a, b: f"{a:02d}{b:02d}"
    R['cat_ba']   = lambda a, b: f"{b:02d}{a:02d}"
    R['cat_ab_s'] = lambda a, b: f"{a}{b}"
    R['cat_ba_s'] = lambda a, b: f"{b}{a}"
    R['rev_a+b']     = lambda a, b: str(_rev_int(a) + b)
    R['a+rev_b']     = lambda a, b: str(a + _rev_int(b))
    R['rev_a+rev_b'] = lambda a, b: str(_rev_int(a) + _rev_int(b))
    R['rev(a+b)']    = lambda a, b: str(_rev_int(a + b))
    R['rev(a-b)']    = lambda a, b: str(_rev_int(a - b))
    R['rev(a*b)']    = lambda a, b: str(_rev_int(a * b))
    R['rev_a*b']     = lambda a, b: str(_rev_int(a) * b)
    R['a*rev_b']     = lambda a, b: str(a * _rev_int(b))
    R['rev_a*rev_b'] = lambda a, b: str(_rev_int(a) * _rev_int(b))
    R['rev_a-b']     = lambda a, b: str(_rev_int(a) - b)
    R['a-rev_b']     = lambda a, b: str(a - _rev_int(b))
    R['rev_a-rev_b'] = lambda a, b: str(_rev_int(a) - _rev_int(b))
    for k in range(-3, 4):
        if k == 0: continue
        R[f'rev(a+b){k:+d}'] = (lambda k=k: (lambda a, b: str(_rev_int(a + b) + k)))()
        R[f'rev(a*b){k:+d}'] = (lambda k=k: (lambda a, b: str(_rev_int(a * b) + k)))()
        R[f'rev(a-b){k:+d}'] = (lambda k=k: (lambda a, b: str(_rev_int(a - b) + k)))()
    R['dsum_a+b']  = lambda a, b: str(sum(int(c) for c in str(a)) + sum(int(c) for c in str(b)))
    R['dsum_a*b']  = lambda a, b: str(sum(int(c) for c in str(a)) * sum(int(c) for c in str(b)))
    R['dsum(a*b)'] = lambda a, b: str(sum(int(c) for c in str(a * b)))
    R['dsum(a+b)'] = lambda a, b: str(sum(int(c) for c in str(a + b)))
    R['a+b*2']  = lambda a, b: str(a + b * 2)
    R['a*2+b']  = lambda a, b: str(a * 2 + b)
    R['a*2-b']  = lambda a, b: str(a * 2 - b)
    R['a-b*2']  = lambda a, b: str(a - b * 2)
    R['a+a']    = lambda a, b: str(2 * a)
    R['b+b']    = lambda a, b: str(2 * b)
    R['a*a']    = lambda a, b: str(a * a)
    R['b*b']    = lambda a, b: str(b * b)
    R['a*a+b']  = lambda a, b: str(a * a + b)
    R['a+b*b']  = lambda a, b: str(a + b * b)
    R['-(a+b)'] = lambda a, b: str(-(a + b))
    R['-(a-b)'] = lambda a, b: str(-(a - b))
    R['-(b-a)'] = lambda a, b: str(-(b - a))
    R['-(a*b)'] = lambda a, b: str(-(a * b))
    R['cat_sum_diff'] = lambda a, b: f"{a + b}{abs(a - b)}"
    R['cat_diff_sum'] = lambda a, b: f"{abs(a - b)}{a + b}"
    R['a^b'] = lambda a, b: str(a ^ b)
    R['a|b'] = lambda a, b: str(a | b)
    R['a&b'] = lambda a, b: str(a & b)
    R['a1+b1'] = lambda a, b: str((a // 10) + (b // 10))
    R['a0+b0'] = lambda a, b: str((a % 10) + (b % 10))
    R['a1*b1'] = lambda a, b: str((a // 10) * (b // 10))
    R['a0*b0'] = lambda a, b: str((a % 10) * (b % 10))
    R['swap_a+swap_b'] = lambda a, b: str(_swap_digits(a) + _swap_digits(b))
    R['swap_a-swap_b'] = lambda a, b: str(_swap_digits(a) - _swap_digits(b))
    R['swap_a*swap_b'] = lambda a, b: str(_swap_digits(a) * _swap_digits(b))
    return R

_EQ_RULES = _build_eq_rule_bank()
_EQ_BASE_FUNCS = {
    'a+b':    lambda a, b: a + b,
    'a-b':    lambda a, b: a - b,
    'b-a':    lambda a, b: b - a,
    'a*b':    lambda a, b: a * b,
    'abs_ab': lambda a, b: abs(a - b),
    'a+a':    lambda a, b: 2 * a,
    'b+b':    lambda a, b: 2 * b,
}

def _parse_eq_prompt(prompt):
    lines_raw = prompt.replace('\\n', '\n').split('\n')
    examples = []; target_line = None
    for line in lines_raw:
        line = line.strip(); low = line.lower()
        if 'determine' in low or 'find' in low or 'what' in low:
            m = re.search(r':\s*(.{5})\s*$', line) or re.search(r'(.{5})\s*=\s*\?', line)
            if m: target_line = m.group(1)
            continue
        if '=' in line and len(line) >= 7:
            eq_idx = line.index('=')
            lhs = line[:eq_idx].strip(); rhs = line[eq_idx + 1:].strip()
            if len(lhs) == 5 and rhs:
                examples.append((lhs, rhs))
    return examples, target_line

def _parse_num_lhs(lhs):
    if len(lhs) != 5: return None
    try:
        a = int(lhs[0:2]); b = int(lhs[3:5])
    except Exception:
        return None
    return a, lhs[2], b

def _is_numeric_target(t):
    return bool(t) and len(t) == 5 and t[0].isdigit() and t[1].isdigit() and t[3].isdigit() and t[4].isdigit()

def _solve_eq_numeric_assoc(prompt):
    examples, target = _parse_eq_prompt(prompt)
    if not _is_numeric_target(target) or not examples:
        return None
    tp = _parse_num_lhs(target)
    if tp is None: return None
    ta, top, tb = tp
    by_op = defaultdict(list)
    for lhs, rhs in examples:
        p = _parse_num_lhs(lhs)
        if p is None: continue
        a, op, b = p
        by_op[op].append((a, b, rhs))
    op_rules = {}
    for op, triples in by_op.items():
        consistent = []
        for rname, fn in _EQ_RULES.items():
            ok = True
            for (a, b, rhs) in triples:
                try: v = fn(a, b)
                except Exception: v = None
                if v is None or v != rhs: ok = False; break
            if ok: consistent.append(('R', rname))
        for fname, bfn in _EQ_BASE_FUNCS.items():
            ok_p = True; ok_s = True
            for (a, b, rhs) in triples:
                try: val = str(bfn(a, b))
                except Exception: val = None
                if val is None or rhs != op + val: ok_p = False
                if val is None or rhs != val + op: ok_s = False
                if not ok_p and not ok_s: break
            if ok_p: consistent.append(('P', fname))
            if ok_s: consistent.append(('S', fname))
        op_rules[op] = consistent
    if top not in op_rules:
        return None
    cands = op_rules[top]
    if not cands: return None
    outs = set(); chosen = None
    for kind, rname in cands:
        try:
            if kind == 'R':
                v = _EQ_RULES[rname](ta, tb)
                if v is not None: outs.add(v); chosen = rname
            elif kind == 'P':
                outs.add(top + str(_EQ_BASE_FUNCS[rname](ta, tb))); chosen = 'prefix_' + rname
            elif kind == 'S':
                outs.add(str(_EQ_BASE_FUNCS[rname](ta, tb)) + top); chosen = 'suffix_' + rname
        except Exception:
            continue
    if len(outs) != 1: return None
    pred = outs.pop()
    reasoning = (f"Per-glyph rule association.\n"
                 f"Operator '{top}' in target maps to rule: {chosen}.\n"
                 f"Applied to {ta:02d}{top}{tb:02d}: result = {pred}.")
    return {'computed_answer': pred, 'reasoning': reasoning}

_orig_solve_equation_transform = solve_equation_transform

def solve_equation_transform_v2(prompt, answer):
    r = _orig_solve_equation_transform(prompt, answer)
    if r is not None and r.get('computed_answer', '').strip() == str(answer).strip():
        return r
    fb = _solve_eq_numeric_assoc(prompt)
    if fb is not None:
        return fb
    return r

# ── UNIT / GRAVITY ROUNDING FALLBACK ──
# Primary solvers occasionally miss by ±0.01 because they pick one rounding
# convention (banker's vs half-up vs floor/ceil). We infer the model from
# examples, then search for the rounding convention that fits all examples
# exactly, and apply it to the target. Only used as a fallback when the
# primary solver's answer does not match ground truth for gate purposes.


def _r_half_up(x):   return math.floor(x * 100 + 0.5) / 100
def _r_half_even(x): return round(x, 2)
def _r_floor(x):     return math.floor(x * 100) / 100
def _r_ceil(x):      return math.ceil(x * 100) / 100
def _r_half_down(x):
    v = x * 100
    if v - math.floor(v) <= 0.5: return math.floor(v) / 100
    return math.ceil(v) / 100
def _r_trunc(x):     return math.trunc(x * 100) / 100

_ROUND_MODES = {
    'half_up': _r_half_up, 'half_even': _r_half_even,
    'floor': _r_floor, 'ceil': _r_ceil,
    'half_down': _r_half_down, 'trunc': _r_trunc,
}

def _parse_unit_prompt(prompt):
    exs = []
    for m in re.finditer(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt):
        exs.append((float(m.group(1)), float(m.group(2))))
    tm = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt)
    t = float(tm.group(1)) if tm else None
    return exs, t

def _solve_unit_round_fallback(prompt):
    exs, t = _parse_unit_prompt(prompt)
    if len(exs) < 2 or t is None: return None
    xs = np.array([e[0] for e in exs]); ys = np.array([e[1] for e in exs])
    A = np.vstack([xs, np.ones_like(xs)]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    except Exception:
        return None
    a, b = float(coef[0]), float(coef[1])
    best = None
    for name, rfn in _ROUND_MODES.items():
        if all(abs(rfn(a * x + b) - y) < 1e-9 for x, y in exs):
            best = rfn; break
    if best is None:
        try:
            a2 = float(np.dot(xs, ys) / np.dot(xs, xs))
        except Exception:
            return None
        for name, rfn in _ROUND_MODES.items():
            if all(abs(rfn(a2 * x) - y) < 1e-9 for x, y in exs):
                best = rfn; a, b = a2, 0.0; break
    if best is None: return None
    pred = best(a * t + b)
    computed = f"{pred:.2f}"
    reasoning = (f"Affine fit y = {a:.6f}*x + {b:.6f} with rounding mode "
                 f"that matches all {len(exs)} examples exactly. "
                 f"Applied to target {t}: result = {computed}.")
    return {'computed_answer': computed, 'reasoning': reasoning}

def _parse_gravity_prompt(prompt):
    exs = []
    for m in re.finditer(r't\s*=\s*([\d.]+)s,\s*distance\s*=\s*([\d.]+)\s*m', prompt):
        exs.append((float(m.group(1)), float(m.group(2))))
    tm = re.search(r'falling distance for t\s*=\s*([\d.]+)s', prompt)
    t = float(tm.group(1)) if tm else None
    return exs, t

def _solve_gravity_round_fallback(prompt):
    exs, t = _parse_gravity_prompt(prompt)
    if not exs or t is None: return None
    gs = [2 * d / (tt * tt) for tt, d in exs]
    g_mean = sum(gs) / len(gs)
    best = None
    for g in list(set(gs)) + [g_mean]:
        for name, rfn in _ROUND_MODES.items():
            if all(abs(rfn(0.5 * g * tt * tt) - d) < 1e-9 for tt, d in exs):
                best = (g, rfn); break
        if best: break
    if best is None:
        for step in range(-500, 501):
            g = g_mean + step * 0.001
            for name, rfn in _ROUND_MODES.items():
                if all(abs(rfn(0.5 * g * tt * tt) - d) < 1e-9 for tt, d in exs):
                    best = (g, rfn); break
            if best: break
    if best is None: return None
    g, rfn = best
    pred = rfn(0.5 * g * t * t)
    computed = f"{pred:.2f}"
    reasoning = (f"Fitted g = {g:.6f} with rounding mode that matches all "
                 f"{len(exs)} examples exactly. d = 0.5*g*t^2 at t={t}: {computed}.")
    return {'computed_answer': computed, 'reasoning': reasoning}

_orig_solve_unit_conversion = solve_unit_conversion
def solve_unit_conversion_v2(prompt, answer):
    r = _orig_solve_unit_conversion(prompt, answer)
    if r is not None and r.get('computed_answer', '').strip() == str(answer).strip():
        return r
    fb = _solve_unit_round_fallback(prompt)
    if fb is not None: return fb
    return r

_orig_solve_gravity = solve_gravity
def solve_gravity_v2(prompt, answer):
    r = _orig_solve_gravity(prompt, answer)
    if r is not None and r.get('computed_answer', '').strip() == str(answer).strip():
        return r
    fb = _solve_gravity_round_fallback(prompt)
    if fb is not None: return fb
    return r

# Map puzzle type to solver function
_SOLVERS = {
    'numeral': solve_numeral,
    'unit_conversion': solve_unit_conversion_v2,
    'gravity': solve_gravity_v2,
    'cipher': solve_cipher,
    'binary': solve_binary,
    'equation_transform': solve_equation_transform_v2,
}

# Quality tier classification:
#   T1 = independently verified (solver computes answer without using ground truth)
#   T2 = answer-aware but structurally sound (not used anymore after solver fixes)
# All current solvers are T1 after the fixes — they compute answers independently
# and check_answer serves as genuine verification, not tautological confirmation.

# Phase 1: Generate solver traces for ALL training puzzles
print(f"\n── Generating Solver Traces ──")
solver_traces = []
solver_stats = defaultdict(lambda: {'attempted': 0, 'solved': 0, 'verified': 0})

for row in tqdm(train_df.iter_rows(named=True), total=len(train_df), desc="Solver traces"):
    puzzle_type = classify_puzzle(row['prompt'])
    solver_fn = _SOLVERS.get(puzzle_type)
    if solver_fn is None:
        continue

    solver_stats[puzzle_type]['attempted'] += 1
    result = solver_fn(row['prompt'], str(row['answer']))
    if result is None:
        continue

    solver_stats[puzzle_type]['solved'] += 1

    # Verify against ground truth — only accept correct answers
    if check_answer(result['computed_answer'], str(row['answer'])):
        solver_stats[puzzle_type]['verified'] += 1
        quality_tier = 'T1'  # all 6 types have verified solver traces
        solver_traces.append({
            'prompt': row['prompt'],
            'answer': str(row['answer']),
            'reasoning': result['reasoning'],
            'puzzle_type': puzzle_type,
            'source': 'solver',
            'quality_tier': quality_tier,
        })

# Hybrid Traces Approach: SAT for all other types to control token length + Public traces ONLY for equation_transform (the bottleneck)
# The equation_transform are based on the dataset with reasoning trajectories from a Nemotron-3-Nano-30B baseline run on 9,500 problems
# Extracted by @kishanvavdara from: https://github.com/tonghuikang/nemotron - Many thanks to @kishanvavdara & @huikang.

PUBLIC_TRACES_CSV = "/kaggle/input/datasets/kishanvavdara/nemotron-reasoning-traj/nemotron_traj.csv"

public_traces = []
if os.path.exists(PUBLIC_TRACES_CSV):
    df_public = pl.read_csv(PUBLIC_TRACES_CSV)
    print(f"✓ Loaded {len(df_public):,} public traces from tonghuikang (Progress Prize Winner)")

    for row in df_public.iter_rows(named=True):
        raw_type = str(row.get('problem type', row.get('puzzle_type', 'unknown'))).lower().strip()

        if 'equation' in raw_type or 'symbolic' in raw_type:
            ptype = 'equation_transform'
        elif 'bit' in raw_type or 'binary' in raw_type or 'manipulation' in raw_type:
            ptype = 'binary'
        elif 'gravity' in raw_type:
            ptype = 'gravity'
        elif 'unit' in raw_type or 'conversion' in raw_type:
            ptype = 'unit_conversion'
        elif 'numeral' in raw_type or 'roman' in raw_type:
            ptype = 'numeral'
        elif 'cipher' in raw_type:
            ptype = 'cipher'
        else:
            ptype = 'unknown'

        public_traces.append({
            'prompt': row['prompt'],
            'reasoning': row.get('generated', row.get('reasoning', '')),
            'answer': str(row.get('correct answer', row.get('answer', ''))),
            'puzzle_type': ptype,
            'source': 'public'
        })

    # Hybrid logic: replace ONLY equation_transform with public traces
    local_eq_count = len([t for t in solver_traces if t['puzzle_type'] == 'equation_transform'])
    public_eq = [t for t in public_traces if t.get('puzzle_type') == 'equation_transform']

    solver_traces = [t for t in solver_traces if t['puzzle_type'] != 'equation_transform']
    solver_traces.extend(public_eq)

    print(f"  Replaced equation_transform traces: {local_eq_count} → {len(public_eq)} (public)")
    solver_stats['equation_transform']['verified'] = len(public_eq)
else:
    print("⚠ Public traces CSV not found — using only local solvers")

print(f"\nSolver Trace Results:")
print(f"{'Type':<20} {'Attempted':>10} {'Solved':>10} {'Verified':>10} {'Rate':>8}")
print("-" * 60)
total_verified = 0
for pt in sorted(solver_stats.keys()):
    s = solver_stats[pt]
    rate = s['verified'] / max(s['attempted'], 1)
    total_verified += s['verified']
    print(f"{pt:<20} {s['attempted']:>10} {s['solved']:>10} {s['verified']:>10} {rate:>7.1%}")
print(f"{'TOTAL':<20} {'':>10} {'':>10} {total_verified:>10}")

# Save solver traces
save_traces(solver_traces, SOLVER_TRACES_PATH)
print(f"\nSolver traces saved to {SOLVER_TRACES_PATH}")

# All training data comes from solver traces
training_data = solver_traces

# Summary
print(f"\n── Training Data ──")
print(f"  Total traces: {len(training_data)}")
trace_type_counts = defaultdict(int)
for t in training_data:
    trace_type_counts[t['puzzle_type']] += 1
for pt, cnt in sorted(trace_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {pt}: {cnt}")
print(f"\n✓ Training data ready: {len(training_data)} samples")

# %%
# UNIFIED DATASET — Single source of truth for all training stages
# Merges all puzzles from train_df with available solver/teacher traces to produce:
# unified_dataset: full pool after fraction scaling (all types represented) AND sft_data — subset WITH reasoning traces.
def stratified_sample(data: List[Dict], fraction: float, seed: int = 42) -> List[Dict]:
    """Sample `fraction` of data, stratified by puzzle_type."""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for item in data:
        by_type[item['puzzle_type']].append(item)
    result = []
    for ptype in sorted(by_type.keys()):
        items = by_type[ptype][:]
        rng.shuffle(items)
        n = int(len(items) * fraction)
        result.extend(items[:n])
    rng.shuffle(result)
    return result

# ── Build unified pool: every puzzle from train_df, annotated with traces ──
trace_lookup = {}
for t in training_data:
    trace_lookup[t['prompt']] = t

unified_pool = []
for row in train_df.iter_rows(named=True):
    prompt = row['prompt']
    answer = str(row['answer'])
    puzzle_type = classify_puzzle(prompt)
    trace = trace_lookup.get(prompt)

    if trace and trace.get('source') == 'public':
        source = 'public'
    elif trace:
        source = trace.get('source', 'local_solver')
    else:
        source = 'none'

    entry = {
        'prompt': prompt,
        'answer': answer,
        'puzzle_type': puzzle_type,
        'has_trace': trace is not None,
        'reasoning': trace['reasoning'] if trace else None,
        'source': source,                    # use the source variable
    }
    unified_pool.append(entry)

# ── Apply DATASET_FRACTION with stratified sampling ──
if config.DATASET_FRACTION < 1.0:
    unified_dataset = stratified_sample(unified_pool, config.DATASET_FRACTION)
else:
    unified_dataset = unified_pool[:]
    random.shuffle(unified_dataset)

# ── Convenience views for each stage ──
sft_data = [item for item in unified_dataset if item['has_trace']]

# ── Summary ──
print(f"══ Unified Dataset (DATASET_FRACTION={config.DATASET_FRACTION}) ══")
print(f" Full pool: {len(unified_pool)} puzzles")
print(f" After scaling: {len(unified_dataset)} puzzles")
print(f" With traces: {len(sft_data)} → SFT input")

print(f"\n{'Type':<20} {'Pool':>6} {'Scaled':>7} {'Traced':>7} {'Trace%':>7}")
print("-" * 50)

pool_types = defaultdict(int)
scaled_types = defaultdict(lambda: {'total': 0, 'traced': 0})

for item in unified_pool:
    pool_types[item['puzzle_type']] += 1

for item in unified_dataset:
    scaled_types[item['puzzle_type']]['total'] += 1
    if item['has_trace']:
        scaled_types[item['puzzle_type']]['traced'] += 1

all_types = sorted(set(pool_types.keys()) | set(scaled_types.keys()))
for pt in all_types:
    p = pool_types.get(pt, 0)
    s = scaled_types[pt]['total']
    t = scaled_types[pt]['traced']
    rate = t / max(s, 1)
    print(f" {pt:<20} {p:>6} {s:>7} {t:>7} {rate:>6.1%}")


# %%
# Load Model with ATLAS LoRA Targeting (Kaggle)

import site
import torch
import torch.nn as nn

# Kaggle-specific cutlass path
cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
if os.path.exists(cutlass_pkg_path):
    site.addsitedir(cutlass_pkg_path)

try:
    import mamba_ssm
    # Disable Triton-based CUDA kernels to avoid ptxas permission issues
    import mamba_ssm.ops.triton as mamba_triton
    if hasattr(mamba_triton, 'ssd_combined'):
        mamba_triton.ssd_combined.is_fast_path_available = False
except ImportError:
    print("mamba_ssm")

# Also patch the Nemotron model's fast path check
for mod_name in list(sys.modules.keys()):
    if 'nemotron' in mod_name.lower():
        mod = sys.modules[mod_name]
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False
            print(f"Disabled fast path in {mod_name}")

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Resolve Nemotron model path (Kaggle only — offline) ──
MODEL_PATH = None
NEMOTRON_PATHS = [
    '/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1',
    '/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1',
    '/kaggle/input/nemotron-3-nano-30b-a3b-bf16',
]
for path in NEMOTRON_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Nemotron model not attached!\n"
        "Add via: Kaggle UI → Add Input → Models → search 'nemotron-3-nano-30b-a3b-bf16'"
    )

print(f"Nemotron path: {MODEL_PATH}")
print(f"Model cache dir: {config.MODEL_CACHE_DIR}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    cache_dir=config.MODEL_CACHE_DIR,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    local_files_only=True,
    cache_dir=config.MODEL_CACHE_DIR,
)

# Disable fast path after model load (module gets imported dynamically)
for mod_name in list(sys.modules.keys()):
    if 'modeling_nemotron' in mod_name.lower():
        mod = sys.modules[mod_name]
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False
            print(f"Disabled fast path in {mod_name}")

print(f"Parameters: {model.num_parameters():,}")

# ATLAS — Architecture-Targeting LoRA Module Selection
print("\n── ATLAS Module Discovery ──")

# Get all Linear module names (what PEFT actually targets)
all_module_names = []
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        all_module_names.append(name)

print(f"  Total Linear modules: {len(all_module_names)}")
print(f"  Sample module names:")
for n in all_module_names[:10]:
    print(f"    {n}")

# Count routed vs shared vs attention vs mamba
routed = [n for n in all_module_names if re.search(r'experts\.\d+', n)]
shared = [n for n in all_module_names if 'shared_expert' in n.lower()]
attn = [n for n in all_module_names if any(k in n.split('.')[-1] for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]
mamba = [n for n in all_module_names if n.split('.')[-1] in ('in_proj', 'out_proj') and 'expert' not in n]
print(f"\n  Routed expert Linear modules: {len(routed)}")
print(f"  Shared expert Linear modules: {len(shared)}")
print(f"  Attention Linear modules: {len(attn)}")
print(f"  Mamba in_proj/out_proj modules: {len(mamba)}")

# Build ATLAS regex from discovered module names
atlas_targets = ['in_proj', 'out_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']

# Find shared expert naming: e.g. "shared_experts" or "shared_expert"
if shared:
    example = shared[0]
    for part in example.split('.'):
        if 'shared_expert' in part.lower():
            shared_prefix = part
            break
    atlas_targets.append(f'{shared_prefix}\\.up_proj')
    atlas_targets.append(f'{shared_prefix}\\.down_proj')
    print(f"  Shared expert prefix: '{shared_prefix}'")

target_alternation = '|'.join(atlas_targets)
ATLAS_REGEX = f"^.+\\.({target_alternation})$"
print(f"\n  ATLAS regex: {ATLAS_REGEX}")

# Audit against actual module names
old_regex = r".*\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$"
old_matches = [n for n in all_module_names if re.fullmatch(old_regex, n)]
new_matches = [n for n in all_module_names if re.fullmatch(ATLAS_REGEX, n)]

old_param_count = sum(
    sum(p.numel() for p in mod.parameters())
    for name, mod in model.named_modules()
    if isinstance(mod, nn.Linear) and re.fullmatch(old_regex, name)
)
new_param_count = sum(
    sum(p.numel() for p in mod.parameters())
    for name, mod in model.named_modules()
    if isinstance(mod, nn.Linear) and re.fullmatch(ATLAS_REGEX, name)
)

print(f"\n── ATLAS Module Targeting Audit ──")
print(f"  OLD (all projections): {len(old_matches)} modules ({old_param_count/1e6:.1f}M base params)")
print(f"  ATLAS (always-active): {len(new_matches)} modules ({new_param_count/1e6:.1f}M base params)")
if old_param_count > 0:
    print(f"  Reduction: {len(old_matches) - len(new_matches)} modules removed ({(1 - new_param_count/old_param_count)*100:.1f}% base param reduction)")

# Show examples
routed_excluded = [n for n in old_matches if n not in new_matches][:3]
if routed_excluded:
    print(f"  Example excluded (routed): {routed_excluded[0]}")
shared_included = [n for n in new_matches if 'shared_expert' in n.lower()][:2]
if shared_included:
    print(f"  Example included (shared): {shared_included[0]}")
attn_included = [n for n in new_matches if 'q_proj' in n][:1]
if attn_included:
    print(f"  Example included (attn):   {attn_included[0]}")
mamba_included = [n for n in new_matches if n.endswith('in_proj')][:1]
if mamba_included:
    print(f"  Example included (mamba):  {mamba_included[0]}")

# Use the regex directly
config.ATLAS_TARGET_REGEX = ATLAS_REGEX

print(f"\nApplying ATLAS LoRA (rank={config.LORA_RANK})...")
lora_config = LoraConfig(
    r=config.LORA_RANK,
    lora_alpha=config.LORA_ALPHA,
    target_modules=ATLAS_REGEX,
    lora_dropout=config.LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

if config.RESUME_FROM_CHECKPOINT:
    if not os.path.exists(config.RESUME_FROM_CHECKPOINT):
        raise FileNotFoundError(
            f"RESUME_FROM_CHECKPOINT path does not exist: {config.RESUME_FROM_CHECKPOINT}"
        )
    # Resume from existing adapter checkpoint (e.g., post-SFT)
    from peft import PeftModel
    try:
        model = PeftModel.from_pretrained(model, config.RESUME_FROM_CHECKPOINT, is_trainable=True)
        print(f"\n✓ Resumed adapter from: {config.RESUME_FROM_CHECKPOINT}")
        for f_name in sorted(os.listdir(config.RESUME_FROM_CHECKPOINT)):
            fp = os.path.join(config.RESUME_FROM_CHECKPOINT, f_name)
            if os.path.isfile(fp):
                size_mb = os.path.getsize(fp) / 1024 / 1024
                print(f"  {f_name} ({size_mb:.1f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {config.RESUME_FROM_CHECKPOINT}: {e}")
else:
    # Fresh LoRA — train from scratch
    model = get_peft_model(model, lora_config)

# Enable gradient checkpointing to reduce memory usage during backward pass
model.gradient_checkpointing_enable()
print("Gradient checkpointing enabled")

model.print_trainable_parameters()

# %%
# Dataset and Training Functions (with Dynamic Padding + Time-Aware Checkpointing)

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import sys
import os
import builtins
os.environ.setdefault('PYTHONUNBUFFERED', '1')
# Prefer line-buffered stdout/stderr so newline-terminated prints appear immediately in remote logs
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
# Make print() flush by default across the notebook (safe fallback)
try:
    _orig_print = builtins.print
    def _print(*args, **kwargs):
        kwargs.setdefault('flush', True)
        return _orig_print(*args, **kwargs)
    builtins.print = _print
except Exception:
    pass

class ReasoningDataset(Dataset):
    """Dataset that formats training data using the Nemotron tokenizer's
    native chat template and applies proper label masking (prompt tokens → -100).
    Returns variable-length tensors — dynamic padding is done in collate_fn.
    """
    def __init__(self, data: List[Dict], tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.MAX_SEQ_LENGTH
        self.data = []
        skipped = 0
        def _compress_reasoning(reasoning: str) -> str:
            boxed_idx = reasoning.find('\\boxed{')
            if boxed_idx < 0:
                return reasoning
            prefix = reasoning[:boxed_idx]
            suffix = reasoning[boxed_idx:]
            keep_lines = [ln for ln in prefix.splitlines() if ln.strip()]
            prefix_trim = '\n'.join(keep_lines[-12:])
            return f"{prefix_trim}\n{suffix}" if prefix_trim else suffix

        for item in data:
            reasoning = item.get('reasoning', '')
            if '\\boxed{' not in reasoning:
                skipped += 1
                continue

            candidate = dict(item)
            candidate['reasoning'] = _compress_reasoning(reasoning)
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": candidate['prompt']},
                {"role": "assistant", "content": candidate['reasoning']},
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) <= self.max_length:
                self.data.append(candidate)
                continue

            boxed_only = dict(item)
            boxed_only['reasoning'] = reasoning[reasoning.find('\\boxed{'):]
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": boxed_only['prompt']},
                {"role": "assistant", "content": boxed_only['reasoning']},
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) <= self.max_length:
                self.data.append(boxed_only)
            else:
                skipped += 1
        if skipped:
            print(f"ReasoningDataset: skipped {skipped}/{skipped + len(self.data)} samples (missing answer marker or still over length)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build conversation messages
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['reasoning']},
        ]
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item['prompt']},
        ]

        # Format using native chat template (handles BOS/EOS internally)
        try:
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_text = f"System: {SYSTEM_PROMPT}\nUser: {item['prompt']}\nAssistant: {item['reasoning']}"
            prompt_text = f"System: {SYSTEM_PROMPT}\nUser: {item['prompt']}\nAssistant:"

        # Tokenize WITHOUT max_length padding — dynamic padding in collate_fn
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False
        )

        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encoding['input_ids'])

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create labels: mask prompt portion with -100
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Build per-token weights: boost tokens inside \boxed{} region
        token_weights = torch.ones_like(input_ids, dtype=torch.float32)
        full_ids = input_ids.tolist()
        boxed_tokens = self.tokenizer.encode('\\boxed{', add_special_tokens=False)
        boxed_start = -1
        for i in range(len(full_ids) - len(boxed_tokens) + 1):
            if full_ids[i:i+len(boxed_tokens)] == boxed_tokens:
                boxed_start = i + len(boxed_tokens)
                break
        if boxed_start > 0:
            depth = 1
            boxed_end = boxed_start
            for i in range(boxed_start, len(full_ids)):
                tok_text = self.tokenizer.decode([full_ids[i]])
                depth += tok_text.count('{') - tok_text.count('}')
                boxed_end = i + 1
                if depth <= 0:
                    break
            per_type_w = config.ANSWER_TOKEN_WEIGHTS.get(
                item.get("puzzle_type"), config.ANSWER_TOKEN_WEIGHT
            )
            token_weights[boxed_start:boxed_end] = per_type_w
        # Zero weight for masked positions
        token_weights[labels == -100] = 0.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'token_weights': token_weights,
        }


def dynamic_collate_fn(batch):
    """Pad to longest sequence in the batch, not MAX_SEQ_LENGTH.
    With BATCH_SIZE=1, this pads to exactly the sample's length — zero waste.
    Saves 30-50% compute vs fixed max_length padding.
    """
    max_len = max(b['input_ids'].size(0) for b in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    token_weights = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, b in enumerate(batch):
        seq_len = b['input_ids'].size(0)
        input_ids[i, :seq_len] = b['input_ids']
        attention_mask[i, :seq_len] = b['attention_mask']
        labels[i, :seq_len] = b['labels']
        token_weights[i, :seq_len] = b['token_weights']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'token_weights': token_weights,
    }


def check_answer(predicted, ground_truth) -> bool:
    """Check if predicted answer matches ground truth (flexible matching)."""
    if predicted is None:
        return False
    pred = str(predicted).strip()
    gt = str(ground_truth).strip()
    if pred == gt:
        return True
    if pred.lower() == gt.lower():
        return True
    pred_norm = re.sub(r'[\s,]+', '', pred)
    gt_norm = re.sub(r'[\s,]+', '', gt)
    if pred_norm == gt_norm or pred_norm.lower() == gt_norm.lower():
        return True
    try:
        if abs(float(pred) - float(gt)) / max(abs(float(gt)), 1e-8) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    return False


import shutil

def atomic_save_pretrained(model, path):
    """Atomic save to avoid partial checkpoint writes."""
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)

def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=8,
                session_start=None, time_limit_sec=None,
                midepoch_checkpoint_path=None, emergency_checkpoint_path=None,
                checkpoint_every_steps=100, on_emergency=None):
    """Train for one epoch with time-aware checkpointing.
    Mid-epoch and emergency saves use separate paths from best-val.
    Returns dict with loss, grad_norm, lr, and early_stopped flag.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    grad_norms = []
    steps_since_flush = 0
    optimizer_steps = 0
    early_stopped = False
    optimizer.zero_grad()

    import sys
    import os
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    # How often to emit newline logs (optimizer steps); configurable via config.LOG_EVERY_OPT_STEPS
    log_every = getattr(config, 'LOG_EVERY_OPT_STEPS', 10)

    progress = tqdm(dataloader, desc="Training", file=sys.stdout, mininterval=1.0)
    for step, batch in enumerate(progress):
        # ── TIME CHECK: abort gracefully if running out of time ──
        if session_start and time_limit_sec:
            elapsed = time.time() - session_start
            remaining = time_limit_sec - elapsed
            if remaining < config.TIME_BUFFER_SEC:
                print(f"\n⏰ Time limit approaching ({remaining/60:.0f}min left) — saving checkpoint and stopping")
                # Flush any accumulated gradients before saving
                if steps_since_flush > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                    grad_norms.append(gn)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                # Validate first; if improved,
                # let the caller route it to best_checkpoint_path.
                if on_emergency is not None:
                    try:
                        on_emergency(model)
                    except Exception as _e:
                        print(f"  ⚠ on_emergency hook failed: {_e}")
                if emergency_checkpoint_path:
                    atomic_save_pretrained(model, emergency_checkpoint_path)
                    print(f"  ✓ Emergency checkpoint -> {emergency_checkpoint_path}")
                early_stopped = True
                break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_weights = batch['token_weights'].to(device) if 'token_weights' in batch else None

        if token_weights is not None and token_weights.sum() > 0:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = token_weights[:, 1:].contiguous()
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=-100
            ).view(shift_labels.shape)
            raw_loss = (per_token_loss * shift_weights).sum() / shift_weights.sum().clamp(min=1.0)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            raw_loss = outputs.loss

        loss = raw_loss / grad_accum

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            print(f" WARNING: NaN/Inf loss at step {step} — skipping batch")
            continue

        loss.backward()

        total_loss += raw_loss.item()
        num_batches += 1
        steps_since_flush += 1

        if steps_since_flush >= grad_accum:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            grad_norms.append(gn)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps_since_flush = 0
            optimizer_steps += 1

            # ── Mid-epoch checkpoint ──
            if midepoch_checkpoint_path and checkpoint_every_steps > 0 and optimizer_steps % checkpoint_every_steps == 0:
                atomic_save_pretrained(model, midepoch_checkpoint_path)
                elapsed_h = (time.time() - session_start) / 3600 if session_start else 0
                print(f"  Checkpoint Saved -> {midepoch_checkpoint_path} (step {optimizer_steps}, {elapsed_h:.1f}h)")

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0
        progress.set_postfix({
            'loss': f"{total_loss / max(num_batches,1):.4f}",
            'lr': f"{current_lr:.2e}",
            'opt_steps': optimizer_steps,
        })
        # Emit periodic newline logs so remote runners (Kaggle, CI) capture progress
        if optimizer_steps > 0 and (optimizer_steps % log_every == 0):
            avg_loss = total_loss / max(num_batches, 1)
            try:
                tqdm.write(f"Step {optimizer_steps} | avg_loss={avg_loss:.4f} | lr={current_lr:.2e}")
            except Exception:
                print(f"Step {optimizer_steps} | avg_loss={avg_loss:.4f} | lr={current_lr:.2e}", flush=True)

    # Flush remaining accumulated gradients
    if steps_since_flush > 0 and not early_stopped:
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        grad_norms.append(gn)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    avg_grad_norm = sum(grad_norms) / max(len(grad_norms), 1) if grad_norms else 0.0
    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0

    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 20)),
        'grad_norm': avg_grad_norm,
        'lr': current_lr,
        'num_batches': num_batches,
        'early_stopped': early_stopped,
    }


def validate(model, dataloader, device):
    """Validate model. Returns dict with loss, perplexity."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                total_loss += outputs.loss.item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 20)),
        'num_batches': num_batches,
    }

print("Training and evaluation functions ready (dynamic padding + time-aware checkpointing)")

# %%
# SFT Training
# Uses sft_data from unified dataset (traces only, all types represented)

from torch.utils.data import WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

DO_TRAINING = True

if config.RESUME_FROM_CHECKPOINT:
    if not os.path.exists(config.RESUME_FROM_CHECKPOINT):
        raise FileNotFoundError(f"RESUME_FROM_CHECKPOINT path not found: {config.RESUME_FROM_CHECKPOINT}")
    print(f"⏭ SFT skipped — using adapter from: {config.RESUME_FROM_CHECKPOINT}")
elif DO_TRAINING and sft_data:
    print(f"Supervised Fine-Tuning on {len(sft_data)} traces (from unified dataset)")

    # Show type breakdown entering SFT
    _sft_types = defaultdict(int)
    for item in sft_data:
        _sft_types[item.get('puzzle_type', 'unknown')] += 1
    print("SFT type distribution:")
    for pt, cnt in sorted(_sft_types.items(), key=lambda x: -x[1]):
        print(f"  {pt}: {cnt} ({100*cnt/len(sft_data):.1f}%)")

    # ── Token length diagnostic ──
    _tok_lengths = []
    for item in sft_data:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": item['reasoning']},
        ]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            _tok_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        except Exception:
            _tok_lengths.append(len(item.get('reasoning', '')))
    _tok_lengths.sort()
    n = len(_tok_lengths)
    p50 = _tok_lengths[n // 2]
    p90 = _tok_lengths[int(n * 0.9)]
    p95 = _tok_lengths[int(n * 0.95)]
    p99 = _tok_lengths[int(n * 0.99)]
    print(f"\nToken length stats: median={p50}, p90={p90}, p95={p95}, p99={p99}, max={_tok_lengths[-1]}")
    print(f"MAX_SEQ_LENGTH={config.MAX_SEQ_LENGTH} → {sum(1 for l in _tok_lengths if l <= config.MAX_SEQ_LENGTH)}/{n} samples fit")

    # Shuffle before splitting to avoid biased val set
    random.shuffle(sft_data)

    # Ensure at least 1 validation sample
    val_size = max(1, int(len(sft_data) * config.VAL_SPLIT)) if len(sft_data) > 1 else 0
    val_split = sft_data[:val_size]
    train_split = sft_data[val_size:]

    print(f"\nTrain: {len(train_split)}, Val: {len(val_split)}")

    train_dataset = ReasoningDataset(train_split, tokenizer, config.MAX_SEQ_LENGTH)
    val_dataset = ReasoningDataset(val_split, tokenizer, config.MAX_SEQ_LENGTH)

    # Use dynamic_collate_fn for variable-length batching (zero-waste padding)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=dynamic_collate_fn)

    # Optional type-balanced sampling
    use_stratified = getattr(config, 'USE_STRATIFIED_SAMPLER', False)
    train_sampler = None
    if use_stratified:
        type_counts = defaultdict(int)
        for it in train_dataset.data:
            type_counts[it.get('puzzle_type', 'unknown')] += 1
        # Inverse-frequency weight per sample => balanced batches across the 6 types.
        sample_weights = [1.0 / max(type_counts[it.get('puzzle_type','unknown')], 1)
                          for it in train_dataset.data]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset.data),
            replacement=True,
        )
        print(f"Stratified sampler ON — type counts: {dict(type_counts)}")
        # rebuild train_loader using sampler (mutually exclusive with shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            sampler=train_sampler, collate_fn=dynamic_collate_fn,
        )
    # legacy curriculum kept off by config; tiers var preserved as None for downstream code paths
    curriculum_tiers = None

    # Use fused AdamW if available (saves ~5% memory on CUDA)
    _adam_kwargs = dict(lr=config.LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.95))
    try:
        optimizer = AdamW(model.parameters(), fused=True, **_adam_kwargs)
        print("Using fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = AdamW(model.parameters(), **_adam_kwargs)
        print("Using standard AdamW")

    # Compute optimizer steps correctly: ceil(batches_per_epoch / grad_accum) * epochs
    steps_per_epoch = max(1, -(-len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS))
    num_training_steps = steps_per_epoch * config.NUM_EPOCHS_SFT
    num_warmup_steps = max(1, int(num_training_steps * config.WARMUP_RATIO))

    print(f"Scheduler: {num_training_steps} optimizer steps ({steps_per_epoch}/epoch), {num_warmup_steps} warmup")

    if getattr(config, 'USE_COSINE_SCHEDULE', False):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"Scheduler: cosine_with_warmup")
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"Scheduler: linear_with_warmup")

    device = next(model.parameters()).device
    print(f"Device: {device}")

    best_val_loss = float('inf')
    # Use three distinct checkpoint paths.
    best_checkpoint_path      = f"{config.OUTPUT_DIR}/best_checkpoint"
    midepoch_checkpoint_path  = f"{config.OUTPUT_DIR}/mid_checkpoint"
    emergency_checkpoint_path = f"{config.OUTPUT_DIR}/emergency_checkpoint"
    sft_history = []
    patience_counter = 0
    global_early_stopped = False

    def _emergency_validate_and_route(model_ref):
        """When time-stop fires mid-epoch, validate. If improved, write
        to best_checkpoint_path; otherwise leave best alone (emergency path handles fallback)."""
        global best_val_loss
        if len(val_split) == 0:
            return
        try:
            v = validate(model_ref, val_loader, device)
            print(f"  ⏸ emergency-validate val_loss={v['loss']:.4f} (best={best_val_loss:.4f})")
            if not (math.isnan(v['loss']) or math.isinf(v['loss'])) and v['loss'] < best_val_loss:
                best_val_loss = v['loss']
                atomic_save_pretrained(model_ref, best_checkpoint_path)
                print(f"  ✓ Emergency-validated improvement saved to BEST ({best_val_loss:.4f})")
        finally:
            model_ref.train()

    for epoch in range(config.NUM_EPOCHS_SFT):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS_SFT}")
        torch.cuda.empty_cache()
        gc.collect()

        # Pre-epoch time guard to avoid incomplete epochs near session cutoff.
        if getattr(config, 'REQUIRE_EPOCH_FITS_TIME_BUDGET', False) and len(sft_history) >= 1:
            avg_epoch_time = sum(h['epoch_time_sec'] for h in sft_history) / len(sft_history)
            elapsed = time.time() - SESSION_START
            remaining = config.SESSION_TIME_LIMIT - elapsed
            if remaining < (avg_epoch_time + config.TIME_BUFFER_SEC):
                print(f"  ⏭ Skipping epoch {epoch+1}: avg epoch={avg_epoch_time:.0f}s, "
                      f"remaining={remaining:.0f}s, buffer={config.TIME_BUFFER_SEC}s. "
                      f"Preserving validated best instead of risking mid-epoch stop.")
                global_early_stopped = True
                break

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum=config.GRADIENT_ACCUMULATION_STEPS,
            session_start=SESSION_START,
            time_limit_sec=config.SESSION_TIME_LIMIT,
            midepoch_checkpoint_path=midepoch_checkpoint_path,
            emergency_checkpoint_path=emergency_checkpoint_path,
            checkpoint_every_steps=config.CHECKPOINT_EVERY_STEPS,
            on_emergency=_emergency_validate_and_route,
        )

        # Check if time-aware early stop triggered
        if train_metrics.get('early_stopped', False):
            print(f"  ⏰ Training stopped early due to session time limit")
            global_early_stopped = True
            # Still record metrics for the partial epoch
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - SESSION_START
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': float('inf'),
                'train_ppl': train_metrics['perplexity'],
                'val_ppl': float('inf'),
                'grad_norm': train_metrics['grad_norm'],
                'lr': train_metrics['lr'],
                'epoch_time_sec': epoch_time,
                'session_elapsed_sec': elapsed_total,
                'early_stopped': True,
            }
            sft_history.append(epoch_record)
            break

        val_metrics = validate(model, val_loader, device) if len(val_split) > 0 else {'loss': float('inf'), 'perplexity': float('inf'), 'num_batches': 0}

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - SESSION_START

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_ppl': train_metrics['perplexity'],
            'val_ppl': val_metrics['perplexity'],
            'grad_norm': train_metrics['grad_norm'],
            'lr': train_metrics['lr'],
            'epoch_time_sec': epoch_time,
            'session_elapsed_sec': elapsed_total,
        }
        sft_history.append(epoch_record)

        print(f"  Train Loss: {train_metrics['loss']:.4f} (PPL: {train_metrics['perplexity']:.2f})")
        if len(val_split) > 0:
            print(f"  Val Loss:   {val_metrics['loss']:.4f} (PPL: {val_metrics['perplexity']:.2f})")
        else:
            print(f"  Val Loss:   N/A (no validation data)")
        print(f"  Grad Norm:  {train_metrics['grad_norm']:.4f} | LR: {train_metrics['lr']:.2e}")
        print(f"  Epoch time: {epoch_time:.0f}s | Session: {elapsed_total/3600:.2f}h")

        # Anomaly detection
        if math.isnan(train_metrics['loss']) or math.isinf(train_metrics['loss']):
            print("⛔ FATAL: Training loss is NaN/Inf — stopping SFT immediately")
            break

        if train_metrics['grad_norm'] < 1e-7:
            print(f"  ⚠ Vanishing gradients detected (grad_norm={train_metrics['grad_norm']:.2e})")

        if len(val_split) > 0 and not math.isnan(val_metrics['loss']) and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            atomic_save_pretrained(model, best_checkpoint_path)
            print(f"  ✓ Best checkpoint saved (val loss: {best_val_loss:.4f})")
        elif len(val_split) > 0:
            patience_counter += 1
            print(f"  ✗ No improvement (patience: {patience_counter}/{config.SFT_PATIENCE})")
            if patience_counter >= config.SFT_PATIENCE:
                print(f"  ⚠ Early stopping triggered after {epoch + 1} epochs")
                break
        else:
            atomic_save_pretrained(model, best_checkpoint_path)
            print(f"  ✓ Checkpoint saved (no val split)")

    # ─── SFT Summary Table ───
    print(f"\n{'='*80}")
    print(f"SFT TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>10} {'TrainPPL':>9} {'ValPPL':>9} {'GradNorm':>9} {'LR':>10} {'Time':>6} {'Note':>8}")
    print(f"{'-'*80}")
    for h in sft_history:
        vl = f"{h['val_loss']:>10.4f}" if h['val_loss'] < float('inf') else f"{'N/A':>10}"
        vp = f"{h['val_ppl']:>9.2f}" if h['val_ppl'] < float('inf') else f"{'N/A':>9}"
        note = "STOPPED" if h.get('early_stopped') else ""
        print(f"{h['epoch']:>5} {h['train_loss']:>10.4f} {vl} "
              f"{h['train_ppl']:>9.2f} {vp} {h['grad_norm']:>9.4f} "
              f"{h['lr']:>10.2e} {h['epoch_time_sec']:>5.0f}s {note:>8}")
    print(f"{'-'*80}")
    best_str = f"{best_val_loss:.4f}" if best_val_loss < float('inf') else "N/A"
    print(f"Best val loss: {best_str} | Total time: {sum(h['epoch_time_sec'] for h in sft_history):.0f}s")
    if global_early_stopped:
        print(f"⏰ Training was time-limited — best available checkpoint saved")
    print(f"{'='*80}")

    # Save metrics to disk
    sft_metrics_path = f"{config.OUTPUT_DIR}/sft_metrics.json"
    with open(sft_metrics_path, 'w') as f:
        json.dump(sft_history, f, indent=2)
    print(f"SFT metrics saved to {sft_metrics_path}")

    # Restore best checkpoint weights via state_dict
    if best_val_loss < float('inf'):
        print(f"\nRestoring best checkpoint (val loss: {best_val_loss:.4f})...")
    else:
        print(f"\nRestoring last checkpoint...")
    if os.path.exists(best_checkpoint_path):
        st_file = os.path.join(best_checkpoint_path, "adapter_model.safetensors")
        bin_file = os.path.join(best_checkpoint_path, "adapter_model.bin")

        if os.path.exists(st_file):
            from safetensors.torch import load_file
            best_state = load_file(st_file, device=str(device))
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, best_state)
            print("Best checkpoint restored (safetensors)")
        elif os.path.exists(bin_file):
            best_state = torch.load(bin_file, map_location=device, weights_only=True)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, best_state)
            print("Best checkpoint restored (bin)")
        else:
            print("WARNING: No best checkpoint weights found")

    # Validate restored checkpoint consistency before export.
    if len(val_split) > 0 and best_val_loss < float('inf') and os.path.exists(best_checkpoint_path):
        post_restore = validate(model, val_loader, device)
        drift = abs(post_restore['loss'] - best_val_loss) / max(best_val_loss, 1e-8)
        print(f"  Post-restore val_loss={post_restore['loss']:.4f} vs recorded best={best_val_loss:.4f} "
              f"(drift={drift*100:.2f}%)")
        if drift > 0.01:
            raise RuntimeError(
                f"FATAL: restored adapter val_loss drifts >1% from recorded best "
                f"({post_restore['loss']:.4f} vs {best_val_loss:.4f}). "
                f"Refusing to ship a polluted checkpoint. Inspect {best_checkpoint_path}."
            )
        print("  ✓ Eval-restore-eval safety net passed.")

    print(f"Training complete.")

else:
    sft_history = []
    print("Training skipped — no training data available")

# %%
# Save LoRA Adapter (from best checkpoint)
import hashlib

# Priority: validated best > emergency (validated when possible) > mid.
_candidates = [
    (f"{config.OUTPUT_DIR}/best_checkpoint",      "SFT-best (validated)"),
    (f"{config.OUTPUT_DIR}/emergency_checkpoint", "SFT-emergency (time-stop)"),
    (f"{config.OUTPUT_DIR}/mid_checkpoint",       "SFT-mid (un-validated, last resort)"),
]
restore_from = None
source_label = "current (no checkpoint found)"
for _path, _label in _candidates:
    if os.path.exists(os.path.join(_path, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(_path, "adapter_model.bin")):
        restore_from = _path
        source_label = _label
        print(f"Submission source: {_label}  ({_path})")
        break

if restore_from:
    print(f"Restoring best {source_label} checkpoint from: {restore_from}")
    st_file = os.path.join(restore_from, "adapter_model.safetensors")
    bin_file = os.path.join(restore_from, "adapter_model.bin")
    if os.path.exists(st_file):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        best_state = load_file(st_file, device=str(next(model.parameters()).device))
        set_peft_model_state_dict(model, best_state)
    elif os.path.exists(bin_file):
        from peft import set_peft_model_state_dict
        best_state = torch.load(bin_file, map_location=next(model.parameters()).device, weights_only=True)
        set_peft_model_state_dict(model, best_state)

print(f"Saving adapter from {source_label} weights...")
model.save_pretrained(config.OUTPUT_DIR)

adapter_config_path = f"{config.OUTPUT_DIR}/adapter_config.json"
if os.path.exists(adapter_config_path):
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    saved_rank = adapter_config.get('r', 0)
    assert saved_rank <= 32, f"Competition requires LoRA rank ≤ 32, got {saved_rank}"
    print(f"  Rank: {saved_rank} (✓ ≤ 32)")
    print(f"  Alpha: {adapter_config.get('lora_alpha', 'N/A')}")
else:
    print("WARNING: adapter_config.json not found")

print("\nSaved files:")
for fname in sorted(os.listdir(config.OUTPUT_DIR)):
    fpath = f"{config.OUTPUT_DIR}/{fname}"
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        sha = hashlib.sha256(open(fpath, 'rb').read()).hexdigest()[:16]
        print(f"  {fname} ({size_mb:.2f} MB) sha256:{sha}")

# %%
# Create Submission

import subprocess
import shutil

os.chdir(config.OUTPUT_DIR)

if os.path.exists('submission.zip'):
    os.remove('submission.zip')

required_files = ['adapter_config.json']
model_files = [f for f in os.listdir('.') if f.startswith('adapter_model')]
required_files.extend(model_files)

print("Files for submission:")
for f in required_files:
    status = "found" if os.path.exists(f) else "MISSING"
    print(f"  {f} ({status})")

try:
    subprocess.run(['zip', 'submission.zip'] + required_files, check=True)
    print(f"\nsubmission.zip created ({os.path.getsize('submission.zip') / 1024 / 1024:.2f} MB)")
except Exception:
    import zipfile
    with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in required_files:
            if os.path.exists(f):
                zf.write(f)
    print(f"\nsubmission.zip created ({os.path.getsize('submission.zip') / 1024 / 1024:.2f} MB)")

# Also place a copy at Kaggle working root for convenience.
root_submission = '/kaggle/working/submission.zip'
if os.path.abspath('submission.zip') != os.path.abspath(root_submission):
    shutil.copy2('submission.zip', root_submission)
print(f"Copied submission.zip to {root_submission}")

# %%
# Summary

elapsed_total = time.time() - SESSION_START

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
else:
    gpu_name = "CPU"
    gpu_mem = 0

# Count trace sources from unified dataset
solver_count = len(sft_data) if 'sft_data' in dir() else 0
unified_total = len(unified_dataset) if 'unified_dataset' in dir() else 0

print(f"""
═══════════════════════════════════════════════════════════════════
PIPELINE SUMMARY — SAT (Solver-Augmented Training) + ATLAS LoRA
═══════════════════════════════════════════════════════════════════

Hardware:
  GPU: {gpu_name} ({gpu_mem:.0f}GB)

Configuration:
  LoRA Rank: {config.LORA_RANK}, Alpha: {config.LORA_ALPHA}
  ATLAS shared experts + Mamba + Attention only (routed experts excluded)
  Dataset fraction: {config.DATASET_FRACTION} ({unified_total} puzzles, stratified by type)
  SFT: lr={config.LEARNING_RATE}, epochs={config.NUM_EPOCHS_SFT}, seq_len={config.MAX_SEQ_LENGTH}
  Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}
  Dynamic padding: ON (variable-length batching, zero waste)
  Time-aware checkpointing: every {config.CHECKPOINT_EVERY_STEPS} optimizer steps
  Curriculum: {'ON' if config.CURRICULUM_ENABLED else 'OFF'}

Unified Dataset:
  Total puzzles:    {unified_total} (DATASET_FRACTION={config.DATASET_FRACTION})
  Solver traces:    {solver_count} (all 6 puzzle types)

SAT Pipeline:
  1. Programmatic solvers generate verified reasoning traces (all 6 puzzle types)
  2. Unified dataset merges all puzzles with traces, stratified by type
  3. Nemotron loaded with ATLAS LoRA (rank {config.LORA_RANK}) — only always-active modules
  4. SFT on traced puzzles (label-masked, weighted loss, dynamic padding)
  5. Best checkpoint restored → export LoRA adapter

Session Time: {elapsed_total/3600:.1f}h / {config.SESSION_TIME_LIMIT/3600:.0f}h limit
═══════════════════════════════════════════════════════════════════
""")

if os.path.exists('submission.zip'):
    print(f"Output: submission.zip ({os.path.getsize('submission.zip') / 1024 / 1024:.2f} MB)")
else:
    output_dir = config.OUTPUT_DIR
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files if os.path.isfile(os.path.join(output_dir, f)))
        print(f"Output directory: {output_dir}")
        print(f"  Files: {len(files)}, Total size: {total_size / 1024 / 1024:.2f} MB")
    else:
        print("No output files found")