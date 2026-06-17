# Mirza best 0.86 anchor

This directory publishes the highest-scoring solution line from our NVIDIA Nemotron Model Reasoning Challenge run.

Final displayed competition result: Solo Silver Medal, rank `101 / 4182`, score `0.860`.

## Scores

Best private-score submission:

| Field | Value |
| --- | --- |
| Submission ref | `53447857` |
| Message | `jun07_cycle11_mirza_best086_repeat_anchor` |
| Public score | `0.860` |
| Private score | `0.864` |
| Source kernel | `mirzayasirabdullah07/best-nvidia-nemotron-notebook-0-86` |
| Source version | `v16`, scriptVersionId `324524084` |

Best public-score tie:

| Field | Value |
| --- | --- |
| Submission ref | `53443877` |
| Message | `jun07_cycle04_mirza_best086_anchor` |
| Public score | `0.864` |
| Private score | `0.860` |

## Contents

- `manifest.json`: score and provenance manifest.
- `public_final_code/`: currently public Mirza final-code notebook pulled from Kaggle on `2026-06-17`.
- `public_adapter_safetensors/`: currently public Mirza adapter-safetensors notebook pulled from Kaggle on `2026-06-17`.
- `../../scripts/run_jun07_submit5_loop.py`: local submission loop that selected and repeated this anchor.
- `../../reports/2026-06-07_submit5_kaggle_gpu_summary.json`: run summary for the day this anchor was submitted.

## Binary artifact status

The scored `submission.zip` is not committed here. The adapter output is multi-GB, while GitHub rejects normal repository files above 100 MiB. The original scored Kaggle slug returned `404` when pulled on `2026-06-17`, so this publication keeps the source lineage, currently available public notebooks, and exact score records rather than an unrecoverable binary.
