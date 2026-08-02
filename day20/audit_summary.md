# Day 20 — pre-publication repo audit

Date: 2026-08-02 (UTC). Full command outputs saved alongside this file.

## Tracked-file size scan (`audit_tracked_sizes.txt`)

Largest tracked blobs — nothing anywhere near the 50MB line:

| file | size |
|---|---|
| day3/train.log | 7.2MB |
| day3/eval_sweep.log | 6.9MB |
| eval500.log | 5.7MB |
| day5/act_pusht_train.log | 4.0MB |
| day16/oracle_demos_800.npz | 3.3MB |
| (everything else) | <1MB, mostly ~100-300KB rollout mp4s |

Total tracked content: **40MB**. No checkpoints, no venv, no node_modules tracked.

## Secrets scan (`audit_secrets_scan.txt`)

Grepped tracked files AND untracked day9/day12/day19/dashboard for: `hf_*`,
`sk-*`, `ghp_*`, `github_pat_*`, AWS `AKIA*`, Slack `xox*`, private-key
headers, and quoted `api_key/secret/token/password` assignments.
**Zero matches.**

## Problems found → fixed in .gitignore this day

1. `dashboard/frontend/node_modules/` existed with no ignore rule → added `node_modules/`.
2. `day12/hand_video.MOV` is **59MB** (over the 50MB line) → ignored. Raw phone
   capture; stays on local disk, day12 ships the derived artifacts
   (trajectories, retargeted paths, side-by-side mp4s).
3. `day12/hand_landmarker.task` (7.5MB) → ignored; re-downloadable MediaPipe
   model, effectively a checkpoint.
4. `outputs/` was ignored wholesale, but the dashboard leaderboard reads the
   pretrained-diffusion 61.0% baseline from `outputs/eval/.../22-35-58_pusht_diffusion`.
   Deployed as-was, the headline row would vanish. → narrowed to `outputs/*`
   with `!outputs/eval/` (2.7MB, evidence only — the protected dir is added to
   git untouched, never modified).

## Untracked work committed as part of going public

`day9/` (provenance notes), `day12/` (hand-retargeting evidence, minus the two
ignored files), `day19/` (frontend verification), `dashboard/frontend/` +
built `dashboard/static/`, `day20/` (this audit + smoke evidence).
