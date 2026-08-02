# Eval-run provenance — which policy made each result

**Problem:** `eval_info.json` records success %, episodes, and rewards but **not** the
policy. Confirmed: its only top-level keys are `per_task`, `per_group`, `overall` — no
policy field anywhere. So the policy for each run has to be recovered from the launch
command in the logs.

**The 15 runs** = the 15 `eval_info.json` files in the repo (day3×5, day5×5, day7×2,
outputs/eval×3). The three `outputs/eval` dirs with *no* `eval_info.json`
(`22-30-30`, `22-31-28`, plus `22-30-30/videos`) are aborted attempts, not runs, and are
excluded.

## Proof sources, strongest first
- **[LOG]** = LeRobot's parsed-config dump captured in a committed `.log`. This is what the
  process *actually ran with* — `output_dir` names the run, `pretrained_path` names the
  policy weights. Strongest possible proof. Only 6 runs have this.
- **[HIST]** = line in `~/.bash_history`. The command as typed; disambiguated to its run by
  the `--output_dir=` flag (day3-final / day5 / day7) or by episode count (the two early
  `outputs/eval` runs, which used the default output path).

## Policy type is provable from the config fingerprint (not just the path)
- **diffusion**: policy block carries `noise_scheduler_type: 'DDPM'`, `beta_schedule`,
  `down_dims`, `num_train_timesteps` (see eval500.log:32–75).
- **ACT**: policy block carries `chunk_size: 100`, `n_action_steps: 100`
  (see day5/act_pusht_train.log — the checkpoint these day5 evals load).

---

## The table

| # | run id | policy | weights (pretrained_path) | eps | proof |
|---|--------|--------|---------------------------|-----|-------|
| 1 | outputs/eval/2026-07-16/12-01-14_pusht_diffusion | **pretrained diffusion** | `lerobot/diffusion_pusht` **or** `./diffusion_pusht_migrated` (same weights) | 3 | [HIST] |
| 2 | outputs/eval/2026-07-16/22-12-02_pusht_diffusion | **pretrained diffusion** | `./diffusion_pusht_migrated` | 4 | [HIST] |
| 3 | outputs/eval/2026-07-16/22-35-58_pusht_diffusion | **pretrained diffusion** | `diffusion_pusht_migrated` | 500 | **[LOG]** eval500.log |
| 4 | day3/eval_010000 | **from-scratch diffusion** | day3 checkpoint `010000` | 500 | **[LOG]** eval_sweep.log |
| 5 | day3/eval_030000 | **from-scratch diffusion** | day3 checkpoint `030000` | 500 | **[LOG]** eval_sweep.log |
| 6 | day3/eval_050000 | **from-scratch diffusion** | day3 checkpoint `050000` | 500 | **[LOG]** eval_sweep.log |
| 7 | day3/eval_070000 | **from-scratch diffusion** | day3 checkpoint `070000` | 500 | **[LOG]** eval_sweep.log |
| 8 | day3/eval_final | **from-scratch diffusion** | day3 checkpoint `last` | 500 | [HIST] |
| 9 | day5/eval_100k | **ACT (PushT)** | day5 checkpoint `100000` | 500 | [HIST] |
| 10 | day5/eval_100k_nas8_probe | **ACT (PushT)** | day5 `100000`, `n_action_steps=8` | 50 | [HIST] |
| 11 | day5/eval_100k_tempens_probe | **ACT (PushT)** | day5 `100000`, `temporal_ensemble_coeff=0.01` | 50 | [HIST] |
| 12 | day5/eval_100k_nas1_probe | **ACT (PushT)** | day5 `100000`, `n_action_steps=1` | 50 | [HIST] |
| 13 | day5/eval_100k_nas8 | **ACT (PushT)** | day5 `100000`, `n_action_steps=8` | 500 | [HIST] |
| 14 | day7/eval_rehearsal | **ACT (ALOHA transfer)** | `./act_aloha_transfer_migrated` | 2 | [HIST] |
| 15 | day7/eval_aloha | **ACT (ALOHA transfer)** | `./act_aloha_transfer_migrated` | 10 | [HIST] |

**Four distinct policies across the 15 runs:** pretrained diffusion (1–3),
from-scratch diffusion (4–8), ACT on PushT (9–13), ACT on ALOHA (14–15).

---

## Proof lines (verbatim)

### Runs 3–7 — captured in committed logs [LOG]

**Run 3 — outputs/eval/.../22-35-58_pusht_diffusion — pretrained diffusion**
`eval500.log`:
```
31:  'output_dir': PosixPath('outputs/eval/2026-07-16/22-35-58_pusht_diffusion'),
71:            'pretrained_path': PosixPath('diffusion_pusht_migrated'),
```
(policy block eval500.log:32–75 shows DDPM/beta_schedule/down_dims → diffusion.)
This is the **61.0% pretrained-diffusion baseline**.

**Runs 4–7 — day3 checkpoint sweep — from-scratch diffusion**
`day3/eval_sweep.log` (four evals concatenated in one log):
```
30:     'output_dir': PosixPath('day3/eval_010000'),
70:            'pretrained_path': PosixPath('/mnt/d/robotics-challenge/day3/train_run/checkpoints/010000/pretrained_model'),
26044:  'output_dir': PosixPath('day3/eval_030000'),
26084:            'pretrained_path': PosixPath('/mnt/d/robotics-challenge/day3/train_run/checkpoints/030000/pretrained_model'),
52264:  'output_dir': PosixPath('day3/eval_050000'),
52304:            'pretrained_path': PosixPath('/mnt/d/robotics-challenge/day3/train_run/checkpoints/050000/pretrained_model'),
78412:  'output_dir': PosixPath('day3/eval_070000'),
78452:            'pretrained_path': PosixPath('/mnt/d/robotics-challenge/day3/train_run/checkpoints/070000/pretrained_model'),
```
The `day3/train_run` checkpoints are the from-scratch diffusion training (day3/train.log
line 106: `pretrained_path: None` at train time = trained from scratch).

### Runs 1, 2, 8–15 — bash history only [HIST]

No eval `.log` was saved for these — the only record of the command is `~/.bash_history`.
Line numbers are the history entry numbers.

**Run 1 — 12-01-14 (n_episodes=3), pretrained diffusion.** One of the three n=3 commands
(exact line ambiguous; all load the identical pretrained weights, hub == migrated):
```
13: lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.n_episodes=3 --eval.batch_size=1 --policy.device=cuda
15: lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.n_episodes=3 --eval.batch_size=1 --policy.device=cuda
16: lerobot-eval --policy.path=./diffusion_pusht_migrated --env.type=pusht --eval.n_episodes=3 --eval.batch_size=1 --policy.device=cuda
```

**Run 2 — 22-12-02 (n_episodes=4), pretrained diffusion.** One of the two n=4 commands:
```
21: lerobot-eval   --policy.path=./diffusion_pusht_migrated   --env.type=pusht   --eval.n_episodes=4   --eval.batch_size=4   --policy.device=cuda
22: lerobot-eval   --policy.path=./diffusion_pusht_migrated   --env.type=pusht   --eval.n_episodes=4   --eval.batch_size=4   --eval.use_async_envs=false   --policy.device=cuda
```

**Run 8 — day3/eval_final, from-scratch diffusion.** THE 39.8% baseline:
```
67: lerobot-eval   --policy.path=/mnt/d/robotics-challenge/day3/train_run/checkpoints/last/pretrained_model   --env.type=pusht   --eval.n_episodes=500   --eval.batch_size=4   --eval.use_async_envs=false   --policy.device=cuda   --output_dir=day3/eval_final
71: (same command, re-run)
```

**Runs 9–13 — day5, ACT on PushT** (each loads the day5 `100000` checkpoint; probes only
change inference-time flags):
```
113: lerobot-eval   --policy.path=/mnt/d/robotics-challenge/day5/train_run/checkpoints/100000/pretrained_model   --env.type=pusht   --eval.n_episodes=500 --eval.batch_size=4 --eval.use_async_envs=false --seed=1000 --policy.device=cuda --output_dir=day5/eval_100k
114: ...checkpoints/100000/... --policy.n_action_steps=8 ... --eval.n_episodes=50  ... --output_dir=day5/eval_100k_nas8_probe
116: ...checkpoints/100000/... --policy.n_action_steps=8 ... --eval.n_episodes=50  ... --output_dir=day5/eval_100k_nas8_probe   (re-run)
117: ...checkpoints/100000/... --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 ... --eval.n_episodes=50 ... --output_dir=day5/eval_100k_tempens_probe
118: ...checkpoints/100000/... --policy.n_action_steps=1 ... --eval.n_episodes=50  ... --output_dir=day5/eval_100k_nas1_probe
119: ...checkpoints/100000/... --policy.n_action_steps=8 ... --eval.n_episodes=500 ... --output_dir=day5/eval_100k_nas8
```

**Runs 14–15 — day7, ACT on ALOHA transfer cube:**
```
128: MUJOCO_GL=egl lerobot-eval --policy.path=./act_aloha_transfer_migrated --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=2  --eval.batch_size=1 --eval.use_async_envs=false --policy.device=cuda --output_dir=day7/eval_rehearsal --seed=1000
129: MUJOCO_GL=egl lerobot-eval --policy.path=./act_aloha_transfer_migrated --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=10 --eval.batch_size=1 --eval.use_async_envs=false --policy.device=cuda --output_dir=day7/eval_aloha    --seed=1000
```

---

## Notes for the dashboard edit

- `dashboard/main.py:assert_policy()` currently collapses everything to `diffusion` / `act`
  by directory prefix. That is coarse: it can't tell **pretrained** diffusion (runs 1–3)
  from **from-scratch** diffusion (runs 4–8) — both live under different prefixes but are
  genuinely different policies, and it labels ALOHA ACT (day7) the same as PushT ACT (day5).
  Consider four labels: `pretrained-diffusion`, `diffusion-scratch` (day3),
  `act-pusht` (day5), `act-aloha` (day7).
- Proof strength differs. Runs 3–7 are pinned by a committed config dump. Runs 1, 2, 8–15
  rest on bash history — solid (disambiguated by `--output_dir` or episode count) but not a
  committed artifact. If you want them equally durable, that's a reason to keep this file.
- Aside (not tonight's job): day5/eval_100k and day5/eval_100k_nas8 were also run at
  500 ep / batch 4 / seed 1000 / async off — numerically the frozen ruler — yet main.py's
  `FROZEN_500` set marks only run 3 and run 8. Flagging so the "frozen" column doesn't
  mislead later.
