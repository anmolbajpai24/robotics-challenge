# Day 20 deploy log

Session date: 2026-08-03. Running log, appended as each step lands.

## Authorization

I waived the CLAUDE.md no-push rule for this session only, in writing,
scope: git push to origin for the dashboard release. Nothing else changed.
The push, the Render service creation, and the smoke test all ran under
that waiver. This file is the record.

## Step 1: push

Preflight outputs, pasted:

```
$ git remote -v
origin  https://github.com/anmolbajpai24/robotics-challenge.git (fetch)
origin  https://github.com/anmolbajpai24/robotics-challenge.git (push)

$ git status
On branch main
Your branch is ahead of 'origin/main' by 5 commits.
nothing to commit, working tree clean

$ git log --oneline origin/main..HEAD
71bbc2b Day 20: public release
de99ae6 Days 9 and 12: track evidence that never landed
cf9e597 Day 19: frontend integration
00acdbe Day 16: sight ablation on xarm lift
5681aee Day 14: hand-written BC training loop on xarm lift
```

One surprise: 5 unpushed commits, not the 3 expected. 5681aee (day 14) and
00acdbe (day 16) had never been pushed either. They are ancestors of cf9e597,
so pushing the expected 3 pushes them regardless. The day 20 audit had already
recorded this state (audit_git_status.txt says "ahead by 2 commits", taken
before the day 19 and day 20 commits existed). Two extra checks before
pushing, both clean:

- Blob scan across the whole unpushed range: largest objects are
  day12/sim_replay.mp4 at 3.5MB and day16/oracle_demos_800.npz at 3.2MB.
  No checkpoints, no 59MB hand video.
- Secret scan across the full unpushed patch history (hf_, ghp_, sk-, rnd_,
  AKIA, xox, github_pat_, private key headers, quoted api_key/password/secret
  assignments, lockfiles and built bundles excluded): zero matches.

Pushed:

```
$ git push origin main
To https://github.com/anmolbajpai24/robotics-challenge.git
   2ad2636..71bbc2b  main -> main
```

## Step 2: Render

Anmol supplied a Render API key in chat (key lives in the conversation only,
not in any file, not in this one). Service created through the API,
HTTP 201:

- name: robotics-challenge-dashboard
- id: srv-d9ofete417fc73f74rug, region oregon, plan free
- repo: https://github.com/anmolbajpai24/robotics-challenge, branch main,
  autodeploy on commit
- build: `pip install -r requirements.txt` (fastapi, uvicorn, pydantic,
  nothing heavy)
- start: `uvicorn dashboard.main:app --host 0.0.0.0 --port $PORT`
- env: PYTHON_VERSION=3.12.3
- url: https://robotics-challenge-dashboard.onrender.com, same string
  README.md already prints, so no README edit needed
- first deploy dep-d9ofetu417fc73f74slg started on creation, polled every
  20s until terminal

Deploy timeline, polled through the API:

```
20:18:27 update_in_progress
20:18:44 update_in_progress
20:19:05 live
```

Live in about a minute after creation. The build is small, three pip
packages and a 40MB clone.

## Step 3: live smoke test

Full transcript in smoke_live.txt. All GET requests, no HEAD. Results:

- /health: {"ok":true}
- /summary: counts {runs: 15, policies: 4, envs: 2}. Frozen table has the
  three rows, 61.0 / 39.8 / 0.8, every row labeled frozen-500 with
  n_episodes 500 and its provenance source. Matches smoke_local.txt.
- /: 200 text/html, the built React UI.
- /videos/file/ on the pretrained diffusion ep0 clip: 200 video/mp4,
  69978 bytes, byte for byte the same size as the local serve.
- Traversal probe with --path-as-is: 404.

## Close out

README.md line 10 already prints
https://robotics-challenge-dashboard.onrender.com and that is the exact URL
Render assigned, so no README change was needed. This log and smoke_live.txt
committed and pushed under the same waiver. Note: autodeploy is on, so that
push triggers one more Render deploy of identical code.

Free tier behavior a visitor will hit: the instance spins down after about
15 minutes idle and the next request takes around a minute to answer while
it cold starts. Videos and /summary are instant once warm.
