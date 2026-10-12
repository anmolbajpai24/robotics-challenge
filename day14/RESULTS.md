# Day 14 / Day 15: writing a training loop by hand, and the 0% it produced

## Day 14: the hand-written loop

The idea for day 14 was to stop treating "training" as a button on a CLI and
actually write the loop: show the network a situation, guess a move, measure
how wrong the guess was, nudge the weights, repeat. Target environment:
`gym_xarm/XarmLift-v0` (single arm, lift a cube), using LeRobot's ready-made
dataset `lerobot/xarm_lift_medium`.

Dataset, checked before writing any training code (`explore_dataset.txt`):
800 episodes, 20,000 frames. Each frame has an 84x84 image, a 4-dim state
(hand xyz + gripper angle), and a 4-dim action. These demos come from a
TD-MPC RL agent at the "medium" skill tier, not a human, per
`EVAL_PROTOCOL.md`. Final-episode-reward stats confirm "medium": min -0.05,
mean 0.39, max 1.35, a mixed bag, not a clean expert.

The model: a plain MLP, state-only, `4 -> 128 -> 128 -> 4`, trained 40 epochs
with Adam and MSE loss on the 4-dim state (no image). Final training loss
0.358257 (`train_losses.txt`, `train_bc_output.txt`). Saved as `bc_mlp.pt`.

## Day 15: the first benchmark, and 0/500

Day 15 set up this environment's frozen eval (500 episodes, `reset(seed=1000+i)`,
300-step cap, success = cube at least 15cm above its spawn height -- see
`EVAL_PROTOCOL.md`) and ran the new MLP through it.

**Result: 0/500 = 0.0% success.** (`eval_500_seed1000.txt`, 85.3s total,
0.17s/episode.)

Before blaming the environment, the benchmark itself got checked with a
scripted policy that is allowed to read the cube's true position straight
from the simulator (a cheat, not a real policy). That oracle scored
**20/20 = 100%** (`probe_scripted_lift.txt`). So the env works and the check
fires correctly; the 0% is the trained policy failing, not a broken bench.

## Two causes, from the day 15 post

**1. The network can't see the cube.** The dataset's state is only
`[hand xyz, gripper angle]` (`explore_dataset.txt`, confirmed again in
`day16/recon_dataset_keys.txt` at the byte level: no cube position anywhere
in `lerobot/xarm_lift_medium`). Every cube spawns somewhere random. Without
its position in the input, the best the network can learn from 800 demos is
the average routine: reach toward where cubes tend to sit, close the
gripper, lift. When a given cube isn't near that average spot, the network
reaches for the usual place, grabs air, and has no signal to correct with,
because it never knew where the cube actually was.

**2. The teacher only used extremes, and MSE averaged them into mush.** The
TD-MPC teacher's gripper actions sit at the extremes, exactly +1 or -1 (RL
policies tend to saturate their action bounds). But two states that look
nearly identical to the network sometimes got taught +1 and sometimes -1.
Because training minimizes squared error, and squared error punishes big
misses hard, the network's safest bet on a split-decision input is the
average of the two, a half-closed command right in the middle. A gripper
told to close halfway grips nothing.

Both causes point the same direction: the network wasn't stupid, it was
blind and its teacher's own signal was self-contradictory given what little
it could see. That diagnosis is what day 16 went and tested directly (see
`day16/RESULTS.md`): add the cube's position to the input, keep everything
else the same, and the same benchmark went from 0.0% to 99.0%.
