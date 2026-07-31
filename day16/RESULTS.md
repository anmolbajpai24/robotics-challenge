# Day 16 — giving the network sight (xarm lift)

Pre-registered this morning: the Day 14/15 blind BC MLP scored 0/500 because
the cube's position was never in its input. Tonight: confirm the dataset really
lacks the cube, then rerun the exact same benchmark with sight added.

## Recon (step 0)

`lerobot/xarm_lift_medium` stores NO numeric cube position anywhere — the only
observation channels are 84x84 pixels and a 4-dim state (hand xyz + gripper,
generically named `motor_0..3`). Checked at three levels: manifest
(`dataset.features`), the on-disk table (`hf_dataset.column_names`), and one
loaded sample. Evidence: `recon_dataset_keys.txt`. Episode metadata holds no
seeds either, so cube spawns cannot be reconstructed by replay. Branch B.

## The demos (new teacher)

The Day 15 scripted oracle (privileged cube position, 4-phase controller,
copied unchanged from `day14/probe_scripted_lift.py`) demonstrated 800
episodes at seeds 5000..5799 — deliberately disjoint from the frozen eval's
1000..1499 so no exam seed is ever trained on.

- Teacher success while demonstrating: **793/800 = 99.1%**; the 7 failed
  episodes are excluded from training.
- 145,912 (state, action) pairs total; 143,812 used.
- State is 7-dim: [grasp-site xyz (absolute), gripper angle, cube xyz
  (absolute)] — both positions read from the physics state, same frame as
  eval, never the env's double-offset `agent_pos`.
- Data: `oracle_demos_800.npz`, stats: `oracle_demos_stats.txt`.

## Training (both runs share one script, `train_bc.py`)

Same recipe as Day 14 exactly: MLP input->128->128->4, seed 1000, Adam 1e-3,
batch 256, MSE, 40 epochs, states normalized to mean 0 / std 1, actions raw.

| run | input | final MSE |
|-----|-------|-----------|
| sighted | 7-dim (with cube xyz) | 0.005 |
| blind control | 4-dim (cube stripped, same demos) | 0.025 |

The 5x loss gap is partial observability made visible: without the cube, the
approach direction is unpredictable from the input, and MSE settles on the
average cube position.

## Frozen 500 (the ruler — identical protocol to day14/EVAL_PROTOCOL.md)

500 episodes, `reset(seed=1000+i)`, 300-step cap, patched success check
(cube >= 15 cm above spawn), actions clamped [-1, 1].

| policy | teacher | sees cube? | success |
|--------|---------|-----------|---------|
| Day 14 MLP (`day14/bc_mlp.pt`) | TD-MPC demos | no | 0/500 = **0.0%** |
| Day 16 blind control (`bc_mlp_blind.pt`) | scripted oracle | no | 21/500 = **4.2%** |
| Day 16 sighted (`bc_mlp_sighted.pt`) | scripted oracle | yes | 495/500 = **99.0%** |
| scripted oracle (probe, not a policy) | — | privileged | 20/20 |

Evidence: `eval_500_sighted.txt`, `eval_500_blind.txt`, first-3-episode videos
per run.

## Attribution — what actually caused the jump

Because the two Day 16 runs share the teacher, the demos, the architecture,
and the training recipe, cube-in-the-input is the ONLY difference between
4.2% and 99.0%. The better teacher alone (Day 14 0.0% -> blind control 4.2%)
buys almost nothing; sight buys the task. Pre-registered hypothesis confirmed.

## Chart numbers

blind (day 14): 0.0% -> sighted (day 16): 99.0%
control (same demos, no cube): 4.2% | oracle ceiling: 20/20 probe, 99.1% as
demonstrator

## Bonus observation

The sighted clone lifts in ~130 steps; its teacher takes ~181 because the
oracle's phases are keyed to a step timer. The network, which cannot see time,
blended the phases into a smooth state-conditioned policy and finishes ~50
steps faster than its teacher — imitation compressed the choreography.

## Honest limits

- The sighted policy reads the cube's xyz from the simulator — "sight" here is
  ground-truth state, not pixels. Learning it from the camera is future work.
- The oracle's step-timer phase logic was a pre-registered risk (states near
  the descend->squeeze boundary demand opposite gripper commands); at this
  state dimensionality it did not materialize as a failure mode.
