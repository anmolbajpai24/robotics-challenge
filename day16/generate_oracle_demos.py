# Day 16: fire the teacher. The day-14/15 scripted oracle (privileged cube
# position, 20/20 on the probe) demonstrates the lift task; we record
# (state, action) pairs where state is 7-dim: [grasp-site xyz (absolute),
# gripper angle, cube xyz (absolute)]. Both positions are read straight from
# the physics state — same frame the eval uses, bypassing the env's
# double-offset agent_pos quirk.
#
# Seeds: BASE_SEED 5000, far away from the frozen eval's 1000..1499 — the
# network must never train on an exam seed.
#
# Controller copied unchanged from day14/probe_scripted_lift.py.
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day14"))
import xarm_compat  # noqa: F401  (renderer + is_success patches, must precede gym.make)
import gymnasium as gym
import gym_xarm  # noqa: F401  (registers gym_xarm/XarmLift-v0)

BASE_SEED = 5000


def read_state_7d(unwrapped_env):
    grasp_site_position = unwrapped_env._utils.get_site_xpos(
        unwrapped_env.model, unwrapped_env.data, "grasp"
    )
    cube_position = unwrapped_env.obj + unwrapped_env.center_of_table  # absolute frame
    return np.concatenate(
        [grasp_site_position, unwrapped_env.gripper_angle, cube_position]
    ).astype(np.float32)


def oracle_action(unwrapped_env, step_index):
    # verbatim logic from day14/probe_scripted_lift.py
    hand_position = unwrapped_env._utils.get_site_xpos(
        unwrapped_env.model, unwrapped_env.data, "grasp"
    )
    cube_position = unwrapped_env.obj + unwrapped_env.center_of_table
    offset = cube_position - hand_position

    if np.linalg.norm(offset[:2]) > 0.01 and step_index < 80:
        action = np.array([offset[0], offset[1], 0.0, -1.0])  # phase 1: line up above
    elif offset[2] < -0.015 and step_index < 120:
        action = np.array([offset[0], offset[1], offset[2], -1.0])  # phase 2: descend
    elif step_index < 150:
        action = np.array([0.0, 0.0, 0.0, 1.0])  # phase 3: squeeze
    else:
        action = np.array([0.0, 0.0, 1.0, 1.0])  # phase 4: lift, keep squeezing

    action = np.clip(action * 20.0, -1.0, 1.0).astype(np.float32)
    action[3] = np.clip(action[3], -1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--output", type=str, default="oracle_demos_800.npz")
    arguments = parser.parse_args()

    environment = gym.make("gym_xarm/XarmLift-v0", obs_type="state")
    unwrapped = environment.unwrapped

    all_states = []
    all_actions = []
    all_episode_indices = []
    episode_seeds = []
    episode_successes = []
    episode_lengths = []

    start_time = time.monotonic()
    for episode_index in range(arguments.episodes):
        seed = BASE_SEED + episode_index
        environment.reset(seed=seed)
        success = False
        steps_taken = 0
        for step_index in range(300):
            state = read_state_7d(unwrapped)
            action = oracle_action(unwrapped, step_index)
            all_states.append(state)
            all_actions.append(action)
            all_episode_indices.append(episode_index)

            observation, reward, terminated, truncated, info = environment.step(action)
            steps_taken += 1
            if terminated:
                success = True  # only the patched success check terminates
                break
            if truncated:
                break

        episode_seeds.append(seed)
        episode_successes.append(success)
        episode_lengths.append(steps_taken)
        if (episode_index + 1) % 50 == 0 or episode_index == 0:
            running_success_rate = np.mean(episode_successes)
            print(
                f"episode {episode_index + 1:3d}/{arguments.episodes}  "
                f"seed {seed}  steps {steps_taken:3d}  success {success}  "
                f"running success rate {running_success_rate:.1%}"
            )

    elapsed_seconds = time.monotonic() - start_time
    environment.close()

    states = np.stack(all_states)
    actions = np.stack(all_actions)
    episode_indices = np.array(all_episode_indices, dtype=np.int64)
    episode_seeds = np.array(episode_seeds, dtype=np.int64)
    episode_successes = np.array(episode_successes, dtype=bool)
    episode_lengths = np.array(episode_lengths, dtype=np.int64)

    np.savez_compressed(
        arguments.output,
        states=states,
        actions=actions,
        episode_indices=episode_indices,
        episode_seeds=episode_seeds,
        episode_successes=episode_successes,
        episode_lengths=episode_lengths,
    )

    success_count = int(episode_successes.sum())
    print(f"\n--- teacher stats ({elapsed_seconds:.1f}s, "
          f"{elapsed_seconds / arguments.episodes:.2f}s/episode) ---")
    print(f"episodes: {arguments.episodes}  seeds {episode_seeds[0]}..{episode_seeds[-1]}")
    print(f"teacher successes: {success_count}/{arguments.episodes} = "
          f"{success_count / arguments.episodes:.1%}")
    print(f"total (state, action) pairs: {len(states)}")
    print(f"episode length: min {episode_lengths.min()}  "
          f"mean {episode_lengths.mean():.1f}  max {episode_lengths.max()}")
    print("state layout: [hand x, hand y, hand z, gripper angle, cube x, cube y, cube z]")
    print("state min per dim:", np.round(states.min(axis=0), 4))
    print("state max per dim:", np.round(states.max(axis=0), 4))
    print("action min per dim:", np.round(actions.min(axis=0), 4))
    print("action max per dim:", np.round(actions.max(axis=0), 4))
    print(f"saved {arguments.output}")


if __name__ == "__main__":
    main()
