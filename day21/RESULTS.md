# Day 21 — the day 15/16 method, speedrun on xarm push

One night, one task swap: same oracle-demos -> BC-MLP -> frozen-500 pipeline
that cracked lift on day 16, pointed at the push task. The catch: the push
task doesn't exist anymore.

## The env had to be resurrected (phase 1)

The installed gym-xarm 0.1.1 push.py is a stub — `"""DEPRECATED: use only
Lift for now"""` with the entire class commented out, written for the old
mujoco-py API (`self.sim.*`), and not registered. The `push.xml` scene still
ships, and the modern `Base` class already provides everything the commented
code needs (`obj`, `eef`, `obj_rot`, ... as properties), so
`xarm_push_compat.py` ports the commented implementation onto modern Base and
registers `gym_xarm/XarmPush-v0` (300-step cap, like lift's registration).
The installed package is never edited. Evidence with pasted source lines:
`recon_push.txt`.

And yes — the commented-out `is_success` has the SAME cross-frame bug day 14
found in lift: it compares `self.obj` (table-relative in modern Base) against
`self.goal` (absolute world coords). Shipped math says the cube is forever
~1.94 m from the goal. The patch keeps the shipped semantics (3D distance
<= 0.05 m) and only unifies the frame:

    success = || cube_absolute - goal_absolute || <= 0.05

The cube rests at z 0.566 and the goal disc sits at z 0.545 (constant), so
the 5 cm ball leaves ~4.5 cm of xy tolerance — almost exactly the red target
disc's 4.75 cm radius. Satisfiability was proven in recon by teleporting the
cube onto the goal (`is_success() -> True`) before any training ran.

Two more things the modern Base forced (documented in the compat file):
action space declared "xyzw" because `Base.step` hard-asserts 4-dim actions
(the original's 3-dim "xyz" + padding path no longer exists — gripper is
simply commanded closed at 1.0, the original's padding value), and
`_reset_sim` must return a bool or `Base.reset` spins forever.

## The oracle (phase 2)

Day 14 pattern (proportional offsets, gain 20, clip [-1, 1]) but the phases
are state-based, not step-timer based — a push can knock the cube sideways
and need a re-approach, which a timer can't rewind. Rise -> travel to a
standoff 6 cm behind the cube along the cube->goal line -> descend -> press
through the cube with forward speed ramped down as the cube closes in
(gentle commands; these become the BC training targets).

The first cut scored 2/20 with a lesson in it: the "am I behind the cube"
test was xy-only, so the hand switched to press mode while still at travel
height and flew clean over the cube forever — every failed episode ended
with the cube EXACTLY where it spawned. Fix: press mode requires being down
at push height, and the behind-threshold sits below steady-press contact
distance so contact doesn't flap the state machine.

After the fix: **20/20 on seeds 1000..1019** (`probe_scripted_push.txt`) —
which doubles as the oracle ceiling on the first 20 exam episodes.

## The demos (phase 3)

800 episodes, seeds 5000..5799 — disjoint from the frozen eval's 1000..1499,
no exam seed is ever trained on.

- Teacher success while demonstrating: **786/800 = 98.2%**; failures excluded
  from training.
- 89,086 (state, action) pairs total; 84,886 used.
- State is 10-dim: [grasp-site xyz, gripper angle, cube xyz, goal xyz], all
  absolute, all read from the physics state. The goal joins the input because
  — unlike lift, where "up 15 cm" is baked into the task — the push target
  moves every episode.
- Failure flavor: in a handful of episodes the closed gripper pinches the
  cube and flings it (cube z reached 1.25 m in one); those episodes time out
  and get dropped. One episode succeeded in 1 step — the cube can spawn
  almost inside the 5 cm ball.
- Data: `oracle_demos_800.npz`, stats: `oracle_demos_stats.txt`.

## Training (phase 4)

Same recipe as day 14/16 exactly: MLP input->128->128->4, seed 1000, Adam
1e-3, batch 256, MSE, 40 epochs, states normalized to mean 0 / std 1,
actions raw. Only the input width changed: 10.

One real bug: goal z is constant (0.545 every episode), so its std is
exactly 0 and normalization produced 0/0 = NaN across the entire dataset —
epoch 1 loss was NaN before a single useful gradient. Constant dims now get
std 1 (they normalize to ~0 and carry no signal, which is right for a
constant). Final MSE **0.0033** (`train_losses_push.txt`).

## Frozen 500 (phase 5 — same ruler, new task)

500 episodes, `reset(seed=1000+i)`, 300-step cap, patched absolute-frame
success check, actions clamped [-1, 1]. Evidence: `eval_500_push.txt`,
first-3-episode videos.

**494/500 = 98.8%.** All 6 failures are 300-step timeouts, no crashes.
5 of the 494 successes were near-spawn (<= 5 steps).

## Lift vs push — the method, not the task

| | lift (day 16) | push (day 21) |
|---|---|---|
| env as shipped | works, is_success frame bug | deprecated stub, class commented out |
| fix | 1 monkeypatched method | class resurrected + same frame fix |
| success check | cube >= 15 cm above spawn | cube within 5 cm of goal |
| oracle phases | 4, step-timer keyed | state-based (re-approach possible) |
| teacher success | 793/800 = 99.1% | 786/800 = 98.2% |
| pairs used | 143,812 | 84,886 |
| state input | 7-dim (no goal — task fixes it) | 10-dim (goal varies, joins input) |
| final MSE | 0.005 | 0.0033 |
| **frozen-500** | **495/500 = 99.0%** | **494/500 = 98.8%** |
| oracle ceiling (20 exam seeds) | 20/20 | 20/20 |
| clone speed vs teacher | ~50 steps faster (timer teacher) | matches it (110.2 vs 111.4 steps) |

That last row closes day 16's "bonus observation": the lift clone finished
~50 steps faster than its teacher because the teacher's phases were keyed to
a step timer the network couldn't see, so it blended them. The push teacher
is purely state-conditioned — nothing hidden to compress — and the clone
reproduces its timing almost exactly. The speedup was never the network
being clever; it was the teacher being non-Markov.

## Honest limits

- "Sight" is still ground-truth simulator state, not pixels; same as day 16.
- The 5 cm success threshold is inherited from the commented-out original,
  not chosen by me; with the fixed 2.1 cm z-offset it is effectively a
  4.5 cm xy disc.
- The 6 eval failures are uncharacterized beyond "timed out" — no video was
  recorded for them (only episodes 0-2 get filmed).
- The gripper action dim is a constant 1.0 in every demo, so the network
  learns a constant there; it does no work on this task.
- The resurrected env is my port of dead upstream code. Faithful to the
  commented original where the modern Base allows, but it is not
  upstream-blessed — the deviations are listed in `xarm_push_compat.py`.
