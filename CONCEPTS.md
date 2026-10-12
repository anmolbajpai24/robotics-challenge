# Concepts

Short write-ups of the research idea posted each research morning, in day
order. Content here is drawn only from the posts in
[docs/X_POSTS.md](docs/X_POSTS.md); this file just pulls the ideas into one
place and links where (if anywhere) the idea actually got tried in this
repo.

## Day 1: Vision-Language-Action models (VLAs)

A VLA is a robot policy that follows plain-English instructions instead of
running hand-written motion code. The approach borrows the LLM playbook:
start from a model that already understands images and language, then train
it to output robot actions instead of text. A camera feed plus a text
instruction go in, motor movements come out. The target model named on day 1
was SmolVLA, a 450M param model from Hugging Face.

Post: [day 1 morning](https://x.com/anmol_bajpai24/status/2077671346696577392).
Used in: [day6/](day6/NOTES.md), the day SmolVLA actually got pointed at a task.

## Day 2: imitation learning

The policy that played PushT on day 1 was never told the rules of the game,
no goal, no score. It learned from about 200 recorded human games, where
every frame pairs what the screen showed with what the human did next.
Training tunes the network until, shown a screen, it outputs roughly the
move a human would have made there. Everything that looks like strategy in
a replay (circling the block, correcting an overshoot) is really one good
reflex applied over and over, ten times a second.

Post: [day 2 morning](https://x.com/anmol_bajpai24/status/2078049330519564633),
[update](https://x.com/anmol_bajpai24/status/2078191223379247318).
Used in: [day2/](day2/), where the 200-demo dataset itself got opened up and inspected.

## Day 3: training by denoising

The surprising part of training a diffusion policy: the network is never
asked for the right move. Instead, you take the human's real move, add
random noise to it, and the network's only job is to guess what noise got
added. That works because when the policy actually plays, it starts from
pure noise and cleans it up step by step into a move, the same math as
Stable Diffusion, pointed at motion instead of pixels. Refining noise over
many steps also lets the policy commit to one answer (push left, or push
right) instead of averaging two valid answers into a move that drives
straight into the block. Cleaning noise is the skill the policy needs at
play time, so cleaning noise is what gets trained.

Post: [day 3 morning](https://x.com/anmol_bajpai24/status/2078454656704069734).
Used in: [day3/](day3/NOTES.md), training this exact kind of policy from scratch overnight.

## Day 5: action chunking and ACT

ACT is a different kind of policy: instead of deciding one move, looking
again, and deciding the next, it predicts a whole chunk of future moves in
one network call and executes them before re-planning. That's the
chunking part, one network call standing in for what would otherwise be a
hundred. It's a good fit for the slow, precise bimanual arm ACT was
designed for, but on a task where the world reacts to every move, executing
a long chunk blind means acting on a world that's already changed.

Post: [day 5 morning](https://x.com/anmol_bajpai24/status/2079171442688876656),
[update](https://x.com/anmol_bajpai24/status/2079286668428038312).
Used in: [day5/](day5/), training and evaluating ACT on PushT.

## Day 6: embodiment

Embodiment is the reason you can't just download a robot brain and run it
on any robot. A policy trained on one body outputs numbers shaped like that
body, a dot's x/y, or a 6-joint arm's angles, and a brain trained for one
body can't drive another: like handing someone fluent in piano a set of
drums and asking them to play the same song. The twist found on day 6 is
that this isn't always a crash. SmolVLA pads every robot's numbers to a
fixed size internally, so a mismatched body can flow through without
erroring and still produce nothing useful: the body problem lives in what
the numbers mean, not in how many of them there are.

Post: [day 6 morning](https://x.com/anmol_bajpai24/status/2079461377769615851),
[update](https://x.com/anmol_bajpai24/status/2079649463820558795).
Used in: [day6/](day6/NOTES.md).

## Day 9: sim2real (SimGAN, domain randomization)

Sim2real is the gap between the simulator and the real world: everything
measured in this challenge happens inside a physics engine, and a real
robot adds motor lag, uneven friction, changing light, camera shake, an
object slightly heavier than expected, things policies trained purely in
sim often fall apart on contact with. Researchers attack the gap from both
ends. One approach makes the sim more like reality: a Stanford paper called
SimGAN tunes the simulator itself until it matches recordings of a real
robot. The other approach makes the policy tougher instead, training it
across thousands of randomized versions of the sim so that the real world
just looks like one more variation it already saw.

Post: [day 9 morning](https://x.com/anmol_bajpai24/status/2080636744140296256).
Not applied in this challenge, every day here stayed inside simulation.

## Day 10: Human2Robot

Every robot dataset in this challenge exists because a human teleoperated a
robot, driving it by hand, one demo at a time, which is the real bottleneck
of imitation learning: the data doesn't exist until someone manufactures it
slowly. Human2Robot's bet is skipping that step entirely by learning from
the internet's millions of hours of humans just doing things: watch a human
video, extract what the hands did, translate it onto a robot's body, no
teleoperation at all. The hard part is the translation itself, a human hand
isn't a gripper, a human arm isn't a robot arm, and video shows what
happened without showing the forces behind it.

Post: [day 10 morning](https://x.com/anmol_bajpai24/status/2081029985272492296).
Used in: [day12/](day12/NOTES.md), the small piece of this idea actually tried (hand video -> MediaPipe -> retargeted sim replay).

## Day 11: a paper from Karen Liu's lab at Stanford

Karen Liu runs a robotics lab at Stanford whose recent work is humanoids,
robots learning whole-body skills from human motion. Day 11's plan was to
pick one paper from that lab and read it closely.

Post: [day 11 morning](https://x.com/anmol_bajpai24/status/2081281867500920977).
No evening update was posted. Paper read and takeaway: TODO(anmol).

## Day 17: the Imperial paper, 1,000 tasks in a day

A lab at Imperial reported teaching one robot arm 1,000 tasks in a day, one
demonstration per task. The trick, per the day 17 update, is that it mostly
stops training networks at all: every task splits into aligning the
gripper to the object, then replaying what the human's single demo did, and
when a new task shows up the robot retrieves the closest demo it already
has instead of learning from scratch. The numbers posted: one demo per
task, 1,000 tasks, under 24 hours of total human demo time, checked against
over 5,000 real rollouts, with a reported 10x better data efficiency than
behavior cloning when demos are scarce.

Post: [day 17 morning](https://x.com/anmol_bajpai24/status/2083459466117468550),
[update](https://x.com/anmol_bajpai24/status/2083610492313166036).
Paper title and link: TODO(anmol) (flagged uncaptured in the post itself).
Not applied in this repo.

## Day 18: π0.7

π0.7, Physical Intelligence's robot model, reportedly operated a kitchen
appliance it was never trained on: you give it the task, then correct it
mid-run in plain language or show it a picture of the subgoal, and it
adjusts. Its training data for that specific task traced back to just two
episodes of someone closing an air fryer, plus general open robot footage,
and with coaching it loaded a sweet potato into a machine it had never used
before. The day 18 post contrasts that directly with this challenge's own
numbers: this project's hand-written network needed 800 demos to lift one
cube, and had to be handed the cube's coordinates first.

Post: [day 18 morning](https://x.com/anmol_bajpai24/status/2083797946617708840),
[update](https://x.com/anmol_bajpai24/status/2084021734479728897).
Not applied in this repo.
