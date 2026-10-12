# Day 12 / Day 13: replaying a human hand video on the sim robot, and why it misses

## Day 12: filming hands, replaying on the ALOHA sim

The idea: film 30 seconds of my own hands picking up a matchbox and passing
it from one hand to the other, no robot involved. Run MediaPipe hand
tracking on the footage to get wrist paths and pinch timing, then retarget
that motion onto the two-armed ALOHA transfer-cube sim and see how close the
sim robot gets to actually doing it.

Tracking quality (`tracking_summary.txt`): 1979-frame video at 59.976fps,
both hands detected in 79.1% of frames. The right hand (the one doing the
pinching in the video) drops below a 6cm thumb-to-index gap in 56.4% of
frames; the left hand almost never closes that tight (0.5% of frames), which
matches a matchbox pass where one hand grips and the other mostly just
receives.

The replay (`replay_log.txt`): 1649 steps at 50Hz = 33.0 seconds of
retargeted motion (`aloha_actions.npy`), played back on the sim with the
cube at a fixed, disclosed start position.

**Result: reward 0 for all 1649 steps (100.0%).** The sim gripper never
touched the cube, not once in the full 33 seconds. The motion is
recognizable, when the right hand grips in the video the sim's right
gripper closes and moves in, and both arms converge at the center line for
the pass, but "looks like the gesture" and "touches the object" are
different thresholds, and the replay never crossed the second one.

The closest the sim gripper got to the cube, per the day 12 update post, was
about 2cm, and it happened at the same moment the real hands made the pass.
That lines up with `tracking_summary.txt`'s own timing: the minimum gap
between the tracked wrists is 0.470 (at t=22.06s), the moment the hands are
closest together in the source video. Both the video and the replay agree
on *when* the closest approach happens; the gap is in translating "close" on
video into "close enough" in the sim.

## Day 13: why the sim arm doesn't go where it's told

Every replay and experiment through the two-armed sim's end-effector mode
assumes commanded position equals reached position: tell the gripper to go
to a point, assume it got there. Day 13 checked that assumption directly.

Method (`weld_offset_probe.txt`): command the right-arm mocap target to 6
known points, step the physics for 250 steps, and read back where the
fingertip geom actually settled.

**It doesn't go where it's told.** Across the 6 tested points, the
fingertip settles consistently to the right, slightly up, and below the
commanded target, with the 3D offset magnitude ranging from about 7.9cm to
12.4cm (and averaging around 9.5cm). The reason is in the sim's own task
file: the end-effector task couples the arm to its mocap target with a
*soft* weld (`solimp=".25 .25 0.001"`), not a rigid lock, so gravity and the
arm's own geometry pull the real fingertip away from the commanded point.

The error also isn't constant. It changes with the commanded pose; moving
the target in x, y, or z each shifted the offset by a different amount
(see the 6-row table in `weld_offset_probe.txt`), which is exactly what
"soft weld, not a rigid lock" predicts. `retarget.py` applies a fixed
average correction, `WELD_OFFSET_RIGHT = (+0.070, +0.042, -0.040)` (and the
mirrored version for the left arm), good to roughly +/-3cm. That residual
is pose-dependent, which is part of why day 12's replay misses the cube by
a couple of centimetres right at the moment it matters most, the pass.

## The thread between the two days

Day 12 found a gap of a couple of centimetres at the one moment the task
needed precision. Day 13 found a real, measured source for gaps of exactly
that size, baked into how the sim's own end-effector mode moves. Moving
like a human and doing what the human did are separated by a few
centimetres for more than one reason; this is one of them, with a number
attached.
