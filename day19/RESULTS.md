# Day 19 — dashboard frontend integration

**Take-over note:** Anmol authorized take-over for this session ("dashboard
frontend integration"). Claude ran the commands, edited the files, and wrote
this log. Spotter mode resumes after this session.

## What shipped

One process now serves the whole dashboard: `uvicorn dashboard.main:app --port
8011` answers every API route AND serves the built React UI at `/`. The
two-terminal dev loop (uvicorn + `npm run dev` on 5173) still works for
frontend changes.

## What the frontend contains

Prebuilt React + Vite app (unzipped at `dashboard/frontend/`, see its
`INTEGRATION.md`), two tabs:

- **Leaderboard** — three embedded frozen tables that render even with the API
  down (PushT 61.0 / 39.8 / 0.8; xarm lift 0.0 / 4.2 / 99.0; off-ruler results
  incl. the 20/20 privileged oracle, labeled), plus a "Live from the API"
  panel showing what `/summary` actually serves, for at-a-glance comparison.
- **Episodes** — a player for saved rollout mp4s from `/videos`. Nothing
  re-runs to serve these; it reads evidence files already on disk.

Build tooling: `npm run dev` proxies `/summary /runs /curves /videos /health`
to 8011; `npm run build` emits `dashboard/static/`.

## Work done

1. **Dev-loop bring-up.** `npm install` + `npm run dev`, leaderboard confirmed
   rendering against the live API in a real browser (headless Chrome).
   Environment fix required first: WSL had no Linux Node — `npm` resolved to
   the *Windows* install, which dies on `\\wsl.localhost` UNC paths (esbuild's
   postinstall can't run). Installed Node v22.23.2 user-space (tarball,
   sha256-verified) into `~/.local/opt/`, symlinked into `~/.local/bin/`
   (already first on PATH). No sudo.

2. **`normalizeSummary()` fix** in `dashboard/frontend/src/api.js`. The live
   panel showed the raw-JSON banner exactly as INTEGRATION.md predicted: the
   real `/summary` shape is `{ counts, frozen_table: [...] }` and
   `frozen_table` wasn't among the normalizer's guesses. Added an explicit
   mapping for the real shape. One subtlety mattered: `success_rate` is
   LeRobot's `pc_success`, **already a percentage** — the generic formatter's
   `v <= 1` heuristic would have shown act-pusht's 0.8 as "80.0%". The
   explicit mapping formats it as arrived: 0.8%. API untouched, per the rules.
   Before/after screenshots saved.

3. **`/videos` + `/videos/file/{path}`** added to `dashboard/main.py`.
   Curated allowlist (`VIDEO_CLIPS`): six titled clips, one per storyline —
   pretrained-diffusion, diffusion-scratch 90k, act-pusht's rare success,
   ALOHA transfer cube best episode, and the day-16 sighted vs blind-control
   pair. The allowlist doubles as the path check: `/videos/file` serves an
   exact allowlisted relative path or 404s. Read-only on evidence dirs.
   Verified: traversal probe (`day16/../../CLAUDE.md`, `--path-as-is`) → 404;
   on-disk but non-allowlisted mp4 → 404.

4. **Build + static mount.** `npm run build` → `dashboard/static/` (164K).
   Mounted with `StaticFiles(html=True)` at `/` as the LAST route in main.py —
   the mount is a catch-all, so anything registered after it would be
   unreachable (the Day 10 ordering lesson). Guarded with `is_dir()` so the
   bare API still runs without a build. All routes re-verified after mounting.

## Evidence in this folder

- `summary_live_response.json` — the real `/summary` payload that drove the fix
- `dev_leaderboard_before_normalizer_fix.png` — raw-JSON banner, api: up
- `dev_leaderboard_after_normalizer_fix.png` — live table rendering, 0.8% correct
- `prod_leaderboard_8011.png` / `prod_episodes_player_8011.png` — the
  single-process app on 8011: leaderboard, and the player mid-playback of a
  PushT rollout with all six curated clips listed
- `videos_endpoint_verification.txt` — /videos listing, mp4 200s, guard 404s,
  all API routes answering with the static mount live
- `single_process_verification.txt` — final endpoint sweep, everything 200

## Gotchas for the log

- `pkill -f`/`pgrep -f` will match the *current shell* if the pattern string —
  or the launch command itself — appears in the same compound command; a
  restart one-liner killed its own shell twice (exit 144) before splitting
  kill and launch into separate invocations. Extends the day-3 pgrep lesson.
- Headless Chrome on Windows can screenshot WSL-hosted apps via localhost
  forwarding, but its CDP port is Windows-loopback only — the CDP client had
  to run on Windows node.exe too.
- `dashboard/frontend/node_modules` was recreated from scratch after the
  Windows-npm false start; `node_modules/` is now in `.gitignore`, and
  `outputs/eval` (2.7M) is un-ignored so the leaderboard's day-1/2 baseline
  evidence and the pretrained-diffusion clip live in the repo the dashboard
  reads them from.
