"""Replay retargeted human-hand actions on the ALOHA transfer-cube sim.

Loads day12/aloha_actions.npy (N x 16 end-effector actions at 50 Hz), steps
the MuJoCo physics directly, and writes a rendered video plus a reward trace.

We bypass gym.make on purpose: the registered env caps episodes at 300 steps
and its EE task is gated behind a NotImplementedError, while the underlying
task class works (verified in day12/recon_ee_mode.txt). Driving
task.before_step + physics.step directly also skips the triple-camera render
inside get_observation, which would triple the runtime for nothing.

Run with:  MUJOCO_GL=egl ./lerobot-env/bin/python day12/replay.py
"""

import argparse

import cv2
import numpy as np
from dm_control import mujoco
from dm_control.rl import control

from gym_aloha.constants import ASSETS_DIR, DT
from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask

# Fixed, disclosed cube start. sample_box_pose() would randomise it; a replay
# is not a policy and cannot react, so a random spawn just adds luck. This spot
# sits inside the right arm's retargeted path (see retargeted_paths.png).
CUBE_POSE = np.array([0.10, 0.50, 0.05, 1.0, 0.0, 0.0, 0.0])

REWARD_MEANING = {
    0: "no contact",
    1: "right gripper touched cube",
    2: "right gripper lifted cube",
    3: "left gripper touched cube (transfer attempted)",
    4: "left gripper holds cube off table (transfer!)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="day12/aloha_actions.npy")
    parser.add_argument("--out-video", default="day12/sim_replay.mp4")
    parser.add_argument("--out-log", default="day12/replay_log.txt")
    parser.add_argument("--camera", default="top", choices=["top", "angle", "front_close"])
    parser.add_argument("--video-fps", type=int, default=25,
                        help="output video rate. 25 fps with every-2nd-step rendering "
                             "keeps real time while halving render work.")
    arguments = parser.parse_args()

    actions = np.load(arguments.actions)
    step_count, action_width = actions.shape
    assert action_width == 16, f"expected Nx16 actions, got {actions.shape}"

    render_every = round(50.0 / arguments.video_fps / 1.0)  # sim steps per video frame

    physics = mujoco.Physics.from_xml_path(
        str(ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml")
    )
    task = TransferCubeEndEffectorTask()
    env = control.Environment(
        physics, task, time_limit=float("inf"), control_timestep=DT,
        n_sub_steps=None, flat_observation=False,
    )
    env.reset()

    box_joint_index = physics.model.name2id("red_box_joint", "joint")
    np.copyto(physics.data.qpos[box_joint_index : box_joint_index + 7], CUBE_POSE)
    physics.forward()

    writer = cv2.VideoWriter(
        arguments.out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        arguments.video_fps,
        (640, 480),
    )
    if not writer.isOpened():
        raise SystemExit("cv2.VideoWriter failed to open -- codec problem?")

    best_reward = 0
    best_reward_step = 0
    reward_step_counts = {level: 0 for level in REWARD_MEANING}

    for step_index in range(step_count):
        task.before_step(actions[step_index], physics)
        physics.step()

        reward = task.get_reward(physics)
        reward_step_counts[reward] += 1
        if reward > best_reward:
            best_reward = reward
            best_reward_step = step_index
            print(f"step {step_index} (t={step_index * DT:.2f}s): reward {reward} -- "
                  f"{REWARD_MEANING[reward]}")

        if step_index % render_every == 0:
            frame_rgb = physics.render(height=480, width=640, camera_id=arguments.camera)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if step_index % 250 == 0 and step_index > 0:
            print(f"  ...step {step_index}/{step_count}")

    writer.release()

    lines = [
        "day12 replay: retargeted human hand video -> ALOHA transfer-cube sim",
        f"actions: {arguments.actions}  ({step_count} steps @ 50 Hz = {step_count * DT:.1f} s)",
        f"cube start (fixed, disclosed): x={CUBE_POSE[0]} y={CUBE_POSE[1]} z={CUBE_POSE[2]}",
        f"camera: {arguments.camera}  video: {arguments.out_video} @ {arguments.video_fps} fps",
        "",
        f"BEST REWARD: {best_reward} -- {REWARD_MEANING[best_reward]}"
        + (f" (first at step {best_reward_step}, t={best_reward_step * DT:.2f}s)"
           if best_reward > 0 else ""),
        "",
        "steps spent at each reward level:",
    ]
    for level, meaning in REWARD_MEANING.items():
        count = reward_step_counts[level]
        lines.append(f"  reward {level} ({meaning}): {count} steps"
                     f" ({100.0 * count / step_count:.1f}%)")

    report = "\n".join(lines)
    with open(arguments.out_log, "w") as log_file:
        log_file.write(report + "\n")
    print()
    print(report)


if __name__ == "__main__":
    main()
