# 22-Day Robotics Challenge: X posts (source of truth for docs)

Pulled from https://x.com/anmol_bajpai24 on 2026-10-12. Text is verbatim. Dates are UTC.
Posts marked [TRUNCATED] were cut off by X and need the rest pasted in by hand.
No Day 8 post and no Day 11 evening update were found on the timeline.

## Kickoff (2026-07-15)
https://x.com/anmol_bajpai24/status/2077513446527287595

My friend challenged me: 22 days until my London flight, one robotics research idea every morning, one build update every night. No skipping.
I've done game dev for 3 years. Time to learn how robots learn, in public.
Day 1 tomorrow.

## Day 1 (2026-07-16)
Morning thread: https://x.com/anmol_bajpai24/status/2077671346696577392

Day 1/22

First idea: Vision-Language-Action models. Robots that follow instructions in plain English instead of hand-written motion code.

VLAs borrow the LLM playbook. Start from a model that already understands images and language, then train it to output robot actions. The robot's camera feed plus a text instruction go in. Motor movements come out.

I'm working toward SmolVLA, a 450M param model from Hugging Face. No robot arm, just a simulator.

Tonight's job: install LeRobot and get a simpler pretrained policy moving in sim.

Update: https://x.com/anmol_bajpai24/status/2077828178295607482

Day 1 - Update

A pretrained robot policy now runs on my laptop, playing PushT, getting 2 wins out of 3. Getting there took a detour, these tools want Linux, so LeRobot lives in WSL2, in its own Python environment.

Official score is ~64% over 500 episodes, so that's about right.

How it decides is the interesting part. At each step, the model starts from random noise and refines it in 100 passes into a short movement plan. Same math as Stable Diffusion, pointed at motion instead of pixels. The roundabout method exists because sometimes pushing from the left or the right would both work, and a network that outputs one direct answer averages them and drives straight into the block. Refining noise lets it commit to one, either left or right. Each decision means running the network 100 times, which is why each 30-second episode took 48 real seconds to compute. The brain runs slower than the game it plays.

GIF is episode 0, a clean win in 14.5 seconds. The red cross is the model's latest command, the blue dot chases it.

Follow-up (2026-07-17): https://x.com/anmol_bajpai24/status/2078016169114563021

Follow-up to day 1. Ran the full benchmark overnight: 500 episodes, same protocol as the model card.

Result: 61% success. The official number is 64%. At 500 samples that difference is within sampling error, so the pipeline checks out end to end, from the migrated checkpoint to the simulator.

Batching 4 episodes at a time cut compute from 48 to 16 seconds per episode, so the whole thing took about 2h12m.

## Day 2 (2026-07-17)
Morning: https://x.com/anmol_bajpai24/status/2078049330519564633

Day 2/22
Today's idea: imitation learning.

The model from yesterday was never told the rules of PushT. Nobody gave it a goal or a score. It was trained on about 200 recorded human gameplays of PushT, where every frame has 2 things, what the screen showed, and what the human did next. Training tunes the network until, shown a screen, it outputs roughly the move a human would have made.

That's the whole method. Everything in yesterday's GIF is compressed human behavior, replayed ten times a second. What looks like strategy, circling the block, correcting an overshoot, is one good reflex applied over and over.

Today I'll go through the training data itself, load the 200 human demonstrations, watch one, and pull out exactly what the network receives from each frame.

Update: https://x.com/anmol_bajpai24/status/2078191223379247318

Day 2/22, update.

Opened the dataset the model learned from: 206 recorded human games, about 43 minutes of play in total. That's the entire education behind yesterday's 61%.

The network never sees the crisp render from my GIFs. It sees a 96x96 thumbnail. Left is actual size, right is the same pixels blown up. I didn't smooth the blow-up, because smoothing would invent detail the model never had. Every decision it makes goes through that square.

The first demo in the dataset isn't a clean solution either. The human overshoots and has to circle the whole block to come at it from the other side. 16 seconds, where the average demo is 12.5 seconds. The policy learned from someone fumbling and recovering, and it works anyway.

## Day 3 (2026-07-18)
Morning: https://x.com/anmol_bajpai24/status/2078454656704069734

Day 3/22
Today I train my own policy. One thing surprised me about how training works. I assumed you show the network a screen, it guesses a move, and you correct it towards what the human did.

No. You take the human's real move, add random noise to it, and the network's only job is to guess what noise you added. It's never asked for the right move at all.

The reason that works: when the model plays, it starts from pure noise and cleans it up step by step into a move. Cleaning noise is the skill. So cleaning noise is what you train.

Will share an update later today

Update: https://x.com/anmol_bajpai24/status/2078559210544242823

Day 3/22, update.

Training my own policy tonight. A 263M param diffusion policy learning PushT from the 43 minutes of human demos, 90k steps overnight on my laptop's 4060. The full reference recipe is about 200k steps, so this is about half of that.

I have attached the loss graph. The y axis is a log scale and it shows the loss, every gridline is a 10x drop, nearly all the visible improvement happens in the opening minutes, and the rest of the night is a slow grind for smaller and smaller gains. Loss is down about 40x and it's still dropping.

I had a near miss in the beginning, checkpoints were pointed at my external SSD, and the target folder didn't exist. WSL doesn't auto-mount drives you plug in after boot. Nothing had crashed by then, only because the first checkpoint save comes at an hourly delay. Caught it and mounted the drive live, under the running trainer, before the save hit.

Will post the result of all the 90k steps tomorrow after it is finished running.

## Day 4 (2026-07-19)
Morning: https://x.com/anmol_bajpai24/status/2078780417772274042

Day 4/22

The overnight run finished. 90k steps in 10 hours 41 minutes, nine checkpoints saved, final loss 0.002. It was still improving when the step budget ran out.

The exam is running as I post this: the same 500 episodes, same protocol the pretrained policy scored 61% on.

One detail I keep coming back to. My policy has never played a single game of PushT. Training never touches the simulator, it only ever watched the 43 minutes of human demos. Today it plays for the first time, 500 games in a row.

Follow-up [TRUNCATED]: https://x.com/anmol_bajpai24/status/2078873970145992818

Follow-up: 39.8%.
The exam finished. My overnight policy wins 4 games in 10 over 500 episodes. The pretrained policy, trained on the same 43 minutes of demos but for about 200k steps, scores 61% on the exact same exam.

45% of the total training steps (which were 200k) brought [rest not captured]

Update: https://x.com/anmol_bajpai24/status/2078904067875836102

Day 4/22, update.

Tested the policy at five different ages today, one eval per saved checkpoint. The chart has five points: 10% success after one hour of training, 35% after three, 43% after five and a half. Then flat. The last four hours moved the success rate nowhere.

The loss kept falling that entire time. From 50k to 90k steps it dropped another 2.5x, and it bought zero game skill. If I had watched only the loss curve, I would have believed the second half was going great.

I have attached videos from the 500 game exam and the policy's best win, and its worst loss. Same policy, same night of training, both games are in the 500.

My 90k-step policy scores 40%, the 200k step pretrained one scores 61%, on the same 500-episode exam.

## Day 5 (2026-07-20)
Morning: https://x.com/anmol_bajpai24/status/2079171442688876656

Day 5/22

Today's idea: action chunking and a second kind of policy.

Today I will train a different architecture: ACT, which predicts a whole chunk of future moves in one shot. One network call per decision instead of a hundred.

That's the chunking part. Instead of deciding one move, looking again, deciding the next, it commits to a short burst of future moves and executes them before re-planning.

Update: https://x.com/anmol_bajpai24/status/2079286668428038312

Day 5/22, update.

Trained the ACT policy in under 2 hours (diffusion took 10 hours), 52M params in ACT vs diffusion's 263M.

Then the first eval scored basically 0%. A policy whose training loss looked perfect, winning one game in 350.
The fix wasn't retraining. ACT predicts a chunk of future moves, and by default it predicts 100 and executes all 100 blind before looking again. That's fine for the slow bimanual arm ACT was designed for.

On PushT, where the block reacts to every push, executing 100 moves without looking means acting on a world that's already changed.

Same trained model, one setting changed: how many moves it makes before stopping to look again.
Look after every move: 0%, too twitchy to build a push. Fire off all 100 without looking: 0%, the block's moved and it's shoving at empty space. Make 8, then look: 11%. Nothing retrained. Just how often it checks what actually happened.

So the table now reads: pretrained diffusion 61%, my diffusion 40%, my ACT 11%. ACT is a bad fit for this task and that's the honest result.
It's built for high-frequency arm control, not reactive pushing.

(Correction posted on Day 9: the default-settings ACT eval ran the frozen protocol and scored 0.8%. The 11% is off-ruler.)

## Day 6 (2026-07-21)
Morning: https://x.com/anmol_bajpai24/status/2079461377769615851

Day 6/22

Today's idea: embodiment. Why you can't just download a robot brain and run it on any robot.

The policies I trained this week output 2 numbers per step which tell where to move a dot on a screen. That's the entire body they know. A real robot arm has 6 or more joints, and a policy trained on that arm outputs 6+ numbers shaped like that specific arm's angles.

A brain trained for one body can't drive another. The outputs don't line up. It's like handing someone fluent in piano a set of drums and asking them to play the same song. They know the music, but the instrument is wrong.

Today I try SmolVLA, a vision-language-action model from @huggingface. It's the model I named on Day 1 as where this was headed. It's bigger than anything I've trained yet, and it understands language, you can tell it what to do in words, but its pretrained brain learned on a real robot arm with many joints, and my task only asks a policy to steer a dot around a screen, both have very different bodies.

So the question tonight is can this arm-trained brain touch my task at all, or does the body mismatch stop it totally?

Update: https://x.com/anmol_bajpai24/status/2079649463820558795

Day 6/22, update.

The arm-trained brain can't touch my task. But not for the reason I gave this morning. I expected a crash. Arm brain outputs 6 numbers, my task takes 2, they don't fit. Turns out SmolVLA is built to survive exactly that: it pads every robot's numbers up to a fixed size of 32 and trims the output back down. Hand it a 2-number body and nothing crashes.

The real wall is what's baked into the pretrained brain. It expects three cameras at 256x256, my task has one at 384. It expects a state made of 6 arm joint readings, mine is the x and y of a dot. Its actions are shaped like an arm's joints, mine steer a dot. Every row mismatches.

So the body problem isn't a shape error the computer catches for you. The numbers would flow through fine and mean nothing. The body lives in what the numbers mean, not in how many there are.

## Day 7 (2026-07-22)
Morning: https://x.com/anmol_bajpai24/status/2079852457740374311

Day 7/22

New simulator, new robot: two arms that pass a cube from one gripper to the other. On day 5, my ACT policy scored 11% on PushT and I called it a bad fit for the task. ACT wasn't built for pushing a block around a screen. It was built for exactly this robot, two arms, fast control, precise handovers.

Yesterday's lesson was that a brain can't drive the wrong body, so today I am running it on the body it was designed for.

Update: https://x.com/anmol_bajpai24/status/2080000148105785827

Day 7/22 - Update

Ran a pretrained ACT policy on the new robot today: two arms, one cube, and the job is to pick it up and hand it across in the air.

It got 7 wins out of 10, I have attached both the videos of the best win and the failure. The cube gets lifted, passed between grippers mid-air, and caught by the other arm.

## Day 8 (2026-07-23)
No post found. TODO(anmol): was Day 8 skipped, or posted somewhere else?

## Day 9 (2026-07-24)
Morning: https://x.com/anmol_bajpai24/status/2080636744140296256

Day 9/22

Today's idea: Sim2real. The gap between the simulator and the real world.

Everything I've measured in this challenge lives in simulation, all of it happened inside a physics engine. A real robot adds everything which the sim does not have like motor lag, uneven friction, changing light, a camera that shakes, a cube slightly heavier than expected.

Policies trained in sim often fall apart on contact with reality. Researchers attack the gap from both ends. Make the sim more like reality, one Stanford paper (SimGAN) literally tunes the simulator until it matches recordings of the real robot. Or make the policy tougher, train it across thousands of randomized versions of the sim so reality is just one more variation.

Update: https://x.com/anmol_bajpai24/status/2080767319161422215

Day 9/22, update.

Fixed a hole in the eval API tonight. The result files never recorded which policy made each score so I dug through old logs and shell history and reconstructed it for all 15 runs, each with the log line that proves it.

The dig also corrected one of my own published numbers. My day 5 ACT eval at default settings turns out to have run the full frozen protocol, same 500 episodes, same seed as the baselines. So ACT's honest on-ruler score is 0.8%. The 11% I posted came from a tuned inference setting, which is off-ruler.

## Day 10 (2026-07-25)
Morning: https://x.com/anmol_bajpai24/status/2081029985272492296

Day 10/22

Today's idea: Human2Robot. Teaching robots from humans directly.

The 43 minutes of demonstrations my policies learned from required a human to sit there and drive the robot. Every dataset like it is made the same way, a person remotely controlling a robot's movements in real time which is called teleoperating, one demo at a time. That's the bottleneck of imitation learning. Robot data doesn't exist until someone manufactures it, slowly, by hand.

Meanwhile the internet holds millions of hours of humans just doing things. Cooking, folding, fixing, pouring. Human2Robot's bet is learning from that directly: watch a human video, extract what the hands did, translate it to a robot's body. No teleoperation step at all.

The hard part is the translation. My hands aren't grippers, my arm isn't their arm, and video shows what happened, not the forces behind it. I'll try a small piece of it soon.

Update: https://x.com/anmol_bajpai24/status/2081111361355211145

Day 10/22, update.

The eval API is finished. Three new endpoints tonight: /summary serves the headline table, per-run metrics come as chart-ready arrays, and my day 4 skill curve is now served live from the actual result files.

The summary table has three rows, the only runs tested under identical settings: 61.0, 39.8, 0.8. Everything else stays visible in the API, just labeled as tested under different settings, so the table can't accidentally compare unlike things

## Day 11 (2026-07-26)
Morning: https://x.com/anmol_bajpai24/status/2081281867500920977

Day 11/22

Professor Karen Liu runs a robotics lab at Stanford, and I keep running into her name. Her lab's recent work is humanoids: robots learning whole-body skills from human motion.

Today I'll pick one interesting paper from her lab and go through it

No evening update found. TODO(anmol): which paper did you read, and what was the takeaway?

## Day 12 (2026-07-27)
Morning: https://x.com/anmol_bajpai24/status/2081691923090378780

Day 12/22

Two days ago I explained Human2Robot: the bet that robots can learn from videos of people instead of needing a human to drive them. Today I'll try a small part of it

The plan is to film 30 seconds of my hands picking up a matchbox and passing it from one hand to the other. No robot involved, no driving anything, just a phone video of hands.

Then, MediaPipe, Google's hand-tracking library extracts what my hands did from that footage, wrist paths, when the fingers pinch, and I'll translate it onto the two-armed robot simulation. Then the sim robot attempts my motion, and I put the two videos side by side.

The robot replays my motion, it doesn't learn a skill from it. I can maybe try teaching it the skill later once it gets the motion right.

Update: https://x.com/anmol_bajpai24/status/2081858007768654063

Day 12/22, update.

Filmed my hands passing a matchbox. MediaPipe pulled out both wrists' paths and when my fingers pinched, frame by frame. I translated that onto the two-armed sim robot and replayed it. The left one is me picking up the matchstick and right one is the simulator trying to match it.

When my right hand grips, the robot's right gripper closes and moves toward the cube. When my hands meet in the middle for the pass, both arms converge at the center line. It's recognizably my gesture.

The task didn't get completed. The robot scored 0 for all 33 seconds. It never touched the cube. Closest approach: about 2 cm, at the exact moment my real hands made the pass.

Moving like a human and doing what the human did are separated by a few centimeters. Today I have that gap on video. The papers I read yesterday exist because closing those last centimeters takes learning, not replaying.

## Day 13 (2026-07-28)
Morning: https://x.com/anmol_bajpai24/status/2081978960754421789

Day 13/22

The two-armed sim has a mode where each gripper chases a target point in space. Every replay and experiment I run through it assumes commanded position is equal to reached position.

I'll check if the assumption is correct. I'll command the gripper to known spots, read back where the fingertip actually settles, measure the gap at several arm poses. If there's an error, find out whether it's constant or changes with pose, and whether it can be corrected.

Update: https://x.com/anmol_bajpai24/status/2082166122686947397

Day 13/22, update.

The robot doesn't go where you tell it. The two-armed sim has a position mode where each gripper chases a target point in space. I assumed commanding a position means reaching it.

I tested it by sending the gripper to a known spot, reading back where the fingertip actually settled, It was off by 7 centimeters.

The pull towards the target is soft, so gravity and the arm's own geometry shift the resting point away from the commanded one. And the error isn't constant, it changes with the arm's pose.

## Day 14 (2026-07-29)
Morning: https://x.com/anmol_bajpai24/status/2082485597248295152

Day 14/22

Today I'm writing a training loop myself. I want to know what that machinery actually is, so I'm writing it. Show the network a situation, it guesses a move, measure how wrong the guess was, nudge every weight a little, repeat across thousands of examples.

I'm doing it on @LeRobotHF's third simulator, a single arm that lifts a cube, with its ready dataset.

Update: https://x.com/anmol_bajpai24/status/2082609306500001884

Day 14/22, update.

Got the new arm's simulator and dataset loaded, shapes and ranges checked. Spent the rest of the evening on the concepts the loop needs before the code makes sense, what batching is for, why networks want normalized numbers, what the optimizer actually does with a gradient.

## Day 15 (2026-07-30)
Morning: https://x.com/anmol_bajpai24/status/2082827912131248297

Day 15/22

Will complete the training loop today, then the small network trains on the arm dataset. After that I set up a benchmark for this environment, fixed episodes, fixed seed, and run the first exam on it.

Update: https://x.com/anmol_bajpai24/status/2082922951922663768

Day 15/22, update.

Trained the network on the arm dataset today, then ran the first benchmark for this environment: 500 episodes, fixed seeds, success means the cube lifts 15 cm.

Score: 0 out of 500.

Before blaming the environment, I checked it. A scripted cheat policy that's allowed to read the cube's true position scored 20/20 on the same setup, so I knew that the env works but my policy has some issue.

Two causes.

First, the network can't see the cube. The data only ever tells it where the robot's own hand is, never where the cube is. So the best it could learn from 800 demos was the average routine: reach to where cubes usually sit, close, lift. Each new cube spawns somewhere random, the network reaches for the usual spot, grabs air, and can't correct, because it has no idea where the cube actually is. The videos show this.

Second, the teacher only ever used extremes, exactly +1 or -1. But the same looking situation sometimes had a +1 and sometimes a -1, and the way I trained punishes big misses, so the network learned to answer with the safe middle. A gripper commanded to half-close never grips anything.

## Day 16 (2026-07-31)
Morning: https://x.com/anmol_bajpai24/status/2083186440419365314

Day 16/22

My guess on why yesterday's test got a 0 is because the cubes position was not readable. Today I'll test that.

I'll add the cube's position to what the network sees. Then the same 500-episode benchmark, same seeds.
If sight was the problem, the number will increase.

Update: https://x.com/anmol_bajpai24/status/2083317121736712602

Day 16/22, update.

Checked the dataset first. There was no cube position in it anywhere. State is just hand xyz + gripper. So the network was blind, confirmed.

Yesterday's script that asked the sim the cube's position and moved towards it, played 800 episodes with the cube's position included. Then trained two identical networks on it. One sees the cube and the other does not. Seeing here means the sim tells it the cube's true position, not through a camera.

Same 500-episode benchmark, same seeds:
yesterday's network: 0.0%
new network, still blind: 4.2%
new network, sees the cube: 99.0%

## Day 17 (2026-08-01)
Morning: https://x.com/anmol_bajpai24/status/2083459466117468550

Day 17/22

A lab at Imperial says they taught one robot arm 1,000 tasks in a day, one demo per task. Reading their paper today to find out how they achieved this.

Link to the paper - [link not captured. TODO(anmol): paper title and URL]

Update: https://x.com/anmol_bajpai24/status/2083610492313166036

Day 17/22, update.

Read the Imperial paper. The trick is they mostly stopped training networks. Every task splits in two: align the gripper to the object, then replay what the human's demo did.

When a new task shows up, the robot retrieves the closest demo it already has instead of learning from scratch.

One demo per task, 1,000 tasks, under 24 hours of human demo time, checked with over 5,000 real rollouts. They report 10x better data efficiency than behavior cloning when demos are scarce.

## Day 18 (2026-08-02)
Morning: https://x.com/anmol_bajpai24/status/2083797946617708840

Day 18/22

π0.7 came out in April, Physical Intelligence's latest robot model. Apparently, it operated a kitchen appliance it was never trained on. Reading up on how.

Update: https://x.com/anmol_bajpai24/status/2084021734479728897

Day 18/22, update.

π0.7 can successfully operate an air fryer.

You give π0.7 the task, then correct it mid-run in plain language, or show it a picture of the subgoal, and it adjusts. Its training data for the task traced back to two episodes of someone closing an air fryer, plus open robot footage.

With coaching, it loaded a sweet potato into a machine it had never used. Two episodes. My network needed 800 demos to lift one cube, and had to be handed the cube's coordinates first.

## Day 19 (2026-08-03)
Morning: https://x.com/anmol_bajpai24/status/2084240745809277101

Day 19/22

I'm building a dashboard to view the findings of this challenge in a single place. It will also have a leaderboard for the eval numbers and a player for the rollout videos.

Update: https://x.com/anmol_bajpai24/status/2084383085341377007

Day 19/22, update.

Started making the dashboard,

The leaderboard is done, it pulls the frozen numbers live from the eval API, PushT and xarm in their own tables.

## Day 20 (2026-08-04)
Morning: https://x.com/anmol_bajpai24/status/2084521967286390852

Day 20/22

Today, I'll be making the episode player, so the rollout videos sit next to the numbers they produced. One clip per storyline, from the first pretrained run to the sighted network.

Update: https://x.com/anmol_bajpai24/status/2084710802582593752

Day 20/22, update.

Episode player is done. Six clips, one per storyline, from the first pretrained run to the sighted network, playing right next to the numbers they produced.

## Day 21 (2026-08-05)
Morning: https://x.com/anmol_bajpai24/status/2084888885054844963

Day 21/22

New task today: pushing. The arm has to slide the cube to a target spot on the table instead of picking it up.

Dashboard screenshots: https://x.com/anmol_bajpai24/status/2085095816231006584 and https://x.com/anmol_bajpai24/status/2085098138260664599
"This is how the dashboard looks, it has a leaderboard tab and an episodes tab"
"This is the episodes tab, it plays the best rollout from each policy"

Update: https://x.com/anmol_bajpai24/status/2085098686175170935

Day 21/22, update.

The push task worked: 494/500 = 98.8% on the frozen exam.

The scripted teacher gave 800 demos in 21 seconds. Same tiny network as lift, one extra input for the target's position.

## Day 22 (2026-08-06)
Morning: https://x.com/anmol_bajpai24/status/2085242213345136699

Day 22/22

Last day of the challenge. Finishing the dashboard

It goes live today evening

Update: https://x.com/anmol_bajpai24/status/2085455711186964851

Day 22/22, last day.

The dashboard is live at https://robotics-challenge-dashboard.onrender.com

Everything from 22 days on one page, every frozen number, every labeled result, an episode player from the first pretrained run to last night's push network.
