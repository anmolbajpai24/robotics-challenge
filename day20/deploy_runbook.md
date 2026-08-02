# Day 20 — Render deploy runbook

Written before the deploy; the two account actions (git push, Render clicks)
are Anmol's. Verified locally beforehand: `uvicorn dashboard.main:app` on
:8011 answers /summary (15 runs, 3-row frozen table), serves all 6 curated
clips as video/mp4, 404s a `--path-as-is` traversal probe, and serves the
built React UI at `/` (see smoke_local.txt).

## Deploy config (the part that's Render's)

- Language: **Python 3**, branch **main**, root directory: blank
- Build command: `pip install -r requirements.txt`
  (frontend is pre-built and committed at dashboard/static/ — no node needed)
- Start command: `uvicorn dashboard.main:app --host 0.0.0.0 --port $PORT`
- Env var: `PYTHON_VERSION=3.12.3` (parity with the WSL venv)
- Instance: Free — spins down after ~15 min idle, ~1 min cold start after

## Why the deployed data is complete

- The leaderboard's 61.0% row reads `outputs/eval/.../22-35-58_pusht_diffusion/`,
  which was gitignored until today's audit narrowed `outputs/` to
  `outputs/*` + `!outputs/eval/` (2.7MB, committed untouched).
- All 15 `eval_info.json` runs are in git, so /summary and /runs match local.
- Videos shipped: the 6 curated clips in VIDEO_CLIPS total ~800KB, well within
  free tier. The full 40MB repo clones fine on Render.

## Post-deploy smoke test (run against the live URL)

1. `GET /summary` → counts.runs == 15, frozen_table rows 61.0 / 39.8 / 0.8
2. `GET /` → the leaderboard UI renders (embedded tables + live panel agree)
3. `GET /videos` → 6 clips; GET one `/videos/file/...` → 200 video/mp4
4. Save outputs to day20/smoke_live.txt
