# 2026-05-23 Submit-5 Cycle Summary

Competition: NVIDIA Nemotron Model Reasoning Challenge

Loop rule followed: checked remaining submissions, refreshed public Code/Dataset signals, validated candidate structure, submitted, waited for public score, then moved to the next cycle.

## Results

| Cycle | Candidate | Kaggle message | Public score | Outcome |
|---|---|---|---:|---|
| 1 | cocoaai/nvidia-nemotron-huikang-0-87-svd-submit | cycle1_cocoaai_huikang_087_svd_rank32 | 0.84 | Failed; title score did not reproduce. |
| 2 | syu21125/nemotron-v62-submit-v3 | cycle2_syu_v62_submit_v3_rank32 | 0.44 | Failed badly; syu/boxed-loss direction rejected. |
| 3 | sethmoudry/nemotron-sft-v32-adapter | cycle3_sethmoudry_v32_rank32 | 0.85 | Best of today's cycle, but no improvement over historical 0.86. |
| 4 | kingkong153/nemotron-adapter-v31 | cycle4_kingkong_v31_rank32 | 0.83 | Failed; similar public SFT adapter direction weakened. |
| 5 | nybbler/nemotron-v6-aux-adapter | cycle5_nybbler_v6_aux_rank32 | 0.84 | Failed; aux/no-lm_head diversity did not break through. |

Historical best visible in submissions remains 0.86 from `kienngx tinker adapter` on 2026-04-24.

## Current Leaderboard Context

Snapshot after cycle 5 showed the top public scores at 0.87, with many teams at 0.86. A new 0.86 would not be enough for a reliable top-ten position because of tie volume and submission-time ordering.

## Direction Notes

The main negative result is that publicly named high-score artifacts are not reliable. The 0.87-titled Huikang/SVD candidate scored 0.84, and the Syu v62 candidate scored 0.44 despite passing structural validation. The regular large SFT adapters tested today mostly clustered at 0.83-0.85.

The next real direction should not be another blind public adapter sweep. It should be either:

1. recover or reproduce the actual 0.87 Huikang-continuation recipe discussed publicly, with exact post-metric formatting behavior checked;
2. audit top public kernels for adapters that are genuinely new and not Kien/Huikang repacks;
3. train a controlled continuation from the known 0.86 Kien/Tinker base with explicit format-contract fixes for the metric update.

## Saved Artifacts

Local logs:

- `logs/2026-05-23/submissions_after_5_cycles.csv`
- `logs/2026-05-23/latest_kernels_after_5_cycles.txt`
- `logs/2026-05-23/latest_adapter_datasets_after_5_cycles.txt`
- `logs/2026-05-23/leaderboard_snapshot_after_5_cycles.txt`
- `logs/2026-05-23/artifact_manifest.csv`

Large artifact sync:

- `outputs/cycle01_cocoaai_huikang_087_svd/submission.zip` was copied locally.
- The remaining submitted zip files remain on the remote server and are recorded in the manifest. Direct SCP and local Kaggle CDN download were both too slow for full same-turn mirroring without turning the sync into a multi-hour transfer.
