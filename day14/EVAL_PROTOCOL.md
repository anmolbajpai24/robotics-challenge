# xarm lift — frozen eval protocol (the ruler for this environment)

Established day 14. Never change it; results across days are only comparable on
this exact protocol. Companion to the PushT frozen-500 protocol from day 1.

## Protocol

- Environment: `gym_xarm/XarmLift-v0` (gym-xarm 0.1.1 + `day14/xarm_compat.py`
  patches — REQUIRED, see below), `obs_type="state"`.
- Episodes: **500**
- Seeding: episode i is `env.reset(seed=1000 + i)` for i in 0..499.
- Step cap: 300 (the registered TimeLimit).
- Success: episode terminates with the patched `is_success` — cube center at
  least 0.15 m above its spawn height. Success rate = successes / 500.
- Actions clamped to [-1, 1], float32.
- Policy input is the policy's own business (state, pixels, anything the env
  exposes) — the ruler fixes only env, seeds, episode count, step cap, success.
- Videos: first 3 episodes, 25 fps, evidence only (not part of the metric).
- Runner: `python eval_bc.py --episodes 500 --seed 1000 --video-episodes 3`

## Why xarm_compat.py is mandatory

gym-xarm 0.1.1 (dormant upstream) breaks against gymnasium 1.x renderer APIs,
and its shipped `is_success` compares table-relative cube height against an
absolute-frame threshold — unsatisfiable, always 0%. The compat module patches
both at import time. Without it the benchmark is meaningless.

## Note for state-based policies

Feed the model `[grasp-site xyz (absolute), gripper angle]` read from the
physics state (`read_model_frame_state` in `day14/eval_bc.py`) — this is the
dataset's coordinate frame. Do NOT use the env's `agent_pos`: it subtracts the
table-center offset twice and does not match `lerobot/xarm_lift_medium`.

## Baselines on this exact protocol

| day | policy | success |
|-----|--------|---------|
| 14  | hand-written MLP behavior clone (state-only, 4->128->128->4, 40 epochs, `bc_mlp.pt`) | **0.0% (0/500)** |

Benchmark validity: a scripted oracle with privileged cube position scores
20/20 (`day14/probe_scripted_lift.txt`) — success is reachable and the check
fires; the 0% is the policy's failure, not the bench's. Root causes analyzed
day 14: state lacks cube position (partial observability) + MSE regression
averages the teacher's bang-bang actions into timid mid-range commands.

Teacher reference (not this protocol): demos in `lerobot/xarm_lift_medium` come
from a TD-MPC RL agent ("medium" tier); final-reward analysis in
`day14/explore_dataset.txt` shows mixed demo quality.
