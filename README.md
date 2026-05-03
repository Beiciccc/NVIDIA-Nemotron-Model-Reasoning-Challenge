# NVIDIA Nemotron Model Reasoning Challenge

Remote-first Kaggle workflow for the NVIDIA Nemotron Model Reasoning Challenge.

The current reproducible pipeline:

1. Downloads the competition files with the Kaggle API.
2. Builds an exact `id + prompt` lookup model from the released training rows.
3. Generates `submission.csv` and `submission.zip` because this competition requires the uploaded file to be named `submission.zip`.
4. Submits through the Kaggle CLI.
5. Stores run metadata under `runs/` and timestamped submissions under `submissions/`.

The competition evaluator expects a PEFT LoRA adapter at the root of `submission.zip`
with at least:

```text
adapter_config.json
adapter_model.safetensors
```

`scripts/submit_public_adapter.ps1` downloads and submits a public LoRA adapter
dataset used by strong public notebooks, giving us a valid adapter-submission loop
while local 16GB-GPU training experiments are developed.

Remote project directory:

```text
C:\Users\Kun\Desktop\kaggle\NVIDIA Nemotron Model Reasoning Challenge
```
