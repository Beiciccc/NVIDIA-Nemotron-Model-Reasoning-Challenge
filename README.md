# NVIDIA Nemotron Model Reasoning Challenge

Remote-first Kaggle workflow for the NVIDIA Nemotron Model Reasoning Challenge.

## Final result

The competition is complete. Kaggle reports the official deadline as `2026-06-15 23:59:00` UTC.

Final archive files:

- `reports/2026-06-17_final_announcement.md`
- `reports/2026-06-17_final_results.json`
- `solutions/mirza_best086_anchor/`

Team `Kun Zhang` / `beicicc` appears in the downloaded leaderboard snapshot with rank field `363 / 4355` and score `0.864`. In that snapshot, 157 teams scored higher and 408 teams were tied at `0.864`.

The best individual submission by private score was:

```text
ref: 53447857
message: jun07_cycle11_mirza_best086_repeat_anchor
public: 0.860
private: 0.864
source: mirzayasirabdullah07/best-nvidia-nemotron-notebook-0-86 v16 repeat
```

The solution package for that line is published under `solutions/mirza_best086_anchor/`. The large Kaggle `submission.zip` adapter binary is not stored in Git history because it is a multi-GB artifact and exceeds normal GitHub repository limits.

## Workflow

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
