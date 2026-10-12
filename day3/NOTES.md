# Day 3 / Day 4: training a diffusion policy from scratch, overnight

## Day 3: the training run

Day 3 trained a diffusion policy from scratch on PushT, no pretrained
checkpoint this time, same 43 minutes of human demos the pretrained model
used on day 1-2. 263M params (per the day 3 update post), target 90,000
steps, launched to run overnight on the laptop's RTX 4060. The reference
training recipe for this policy is about 200k steps, so 90k is roughly
half of that.

One near miss early on: checkpoints were pointed at the external SSD
(`/mnt/d`), but WSL doesn't auto-mount a drive plugged in after boot, so the
target folder didn't exist yet. Nothing had crashed, because the first
checkpoint save only comes after an hour. The drive got mounted live, while
the trainer kept running, before that first save hit (day 3 update post).

The run finished in **10 hours 41 minutes** (`train.log`: started
2026-07-18 12:54:23, "End of training" at 23:35:15), saving 9 checkpoints
(every 10k steps, confirmed by counting "Checkpoint policy after step"
lines in `train.log`). Final training loss **0.002** at step 90,000
(`train.log`, matches `loss_curve.png`).

## Day 4: the skill curve, and the 39.8% exam

Day 4 ran a full 500-episode frozen-protocol eval on the finished policy
(the same ruler the pretrained model scored 61% on): **39.8% success**
(`day3/eval_final/eval_info.json`, matches the long-standing baseline table).

Day 4 also went back and evaluated 4 earlier checkpoints (150 episodes each,
a cheaper probe, not the frozen ruler) to see how skill changed over the
course of training. Reading the actual numbers out of
`day3/eval_*/eval_info.json` (same data `skill_curve.png` plots):

| step   | success (n episodes) |
|--------|-----------------------|
| 10,000 | 10.0% (n=150)         |
| 30,000 | 34.7% (n=150)         |
| 50,000 | 43.3% (n=150)         |
| 70,000 | 43.3% (n=150)         |
| 90,000 | 39.8% (n=500, the frozen exam) |

That matches the day 4 post's framing closely: roughly 10%, 35%, 43%, then
flat. Skill climbed hard early, peaked by 50k steps, and never moved for the
next 20k steps of training (70k reads the identical 43.3%). Training loss
kept falling the entire time regardless: `train.log` shows loss at 50k was
roughly 0.005-0.006, and by 90k it had dropped to 0.002, a fall of about
2.5x on top of a curve that had already stopped buying any game skill.
Watching the loss curve alone would have said training was still paying
off. The eval curve says otherwise.

The 90k-step number (39.8%) runs on the full 500-episode frozen protocol,
while the four earlier checkpoints only ran 150 episodes each as a cheaper
probe, so they're not directly the same sample size, worth keeping in mind
when comparing the flat middle of the curve to the final point.

## A timing note

The day 4 update post describes the curve in hours of training ("10% after
one hour, 35% after three, 43% after five and a half"). The real checkpoint
timestamps in `train.log`: step 10,000 at 1h21m, step 30,000 at 3h53m, step
50,000 (the 43% point) at 6h20m, step 70,000 at 8h27m. The post's hour
labels are rounded for a tweet; the step numbers and percentages above are
the exact ones.
