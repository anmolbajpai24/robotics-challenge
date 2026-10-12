# Day 6: trying SmolVLA on PushT, and why it doesn't crash

Day 6's question: SmolVLA is a vision-language-action model pretrained on a
real robot arm with many joints. PushT only needs a policy that steers a dot
on a screen with 2 numbers. Can a brain trained for one body even run on the
other, or does the mismatch stop it outright?

The expectation going in was a crash: the arm-trained model outputs 6
numbers, PushT's action space is 2-dim, so something should refuse to run.

That's not what happened. SmolVLA is built to survive exactly this kind of
mismatch: it pads every robot's state and action vectors up to a fixed size
of 32 internally, and trims the output back down to whatever the target
robot needs (per the day 6 evening post). Hand it a 2-number body and
nothing crashes. The shapes line up fine.

The actual wall is in what those numbers mean, not how many there are.
`mismatch.txt` lays out the three places SmolVLA's pretrained brain and
PushT disagree:

| feature    | SmolVLA expects | PushT delivers |
|------------|-----------------|-----------------|
| state dim  | 6-dim           | 2-dim           |
| cameras    | 3 x 256x256     | 1 x 384x384     |
| action dim | 6-dim           | 2-dim           |

SmolVLA's pretraining expects 6 joint-angle readings as state and 6
joint-angle commands as action, from 3 cameras at 256x256. PushT hands it an
x/y position as state, an x/y push as action, and one camera at 384x384.
Padding makes all of these "fit" in the tensor-shape sense. None of them
match in the sense that matters: a padded-out joint angle is still being
read by a network that learned joint angles mean something specific, and
PushT's x/y was never that.

So the body problem isn't a shape error the computer catches for you before
it lets you run. The numbers flow through fine and mean nothing useful. The
body lives in what the numbers mean, not in how many there are.
