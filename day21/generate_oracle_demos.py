# Day 21 phase 3: fire the teacher, push edition. The scripted push oracle
# (privileged cube + goal position, 20/20 on the probe) demonstrates the push
# task; we record (state, action) pairs where state is 10-dim:
# [grasp-site xyz (absolute), gripper angle, cube xyz (absolute),
#  goal xyz (absolute)]. The goal joins the state because — unlike lift, where
# "up 15 cm" is baked into the task — the push target moves every episode, so
# the network cannot succeed without being told where it is.
# All positions read straight from the physics state, same frame the eval
# uses. Structure copied from day16/generate_oracle_demos.py.
#
# Seeds: BASE_SEED 5000, far away from the frozen eval's 1000..1499 — the
# network must never train on an exam seed.
#
# Controller imported from probe_scripted_push.py — single source of truth,
# the probe and the teacher are the same code path.
import argparse
import time

import numpy as np

import xarm_push_compat  # noqa: F401  (registers gym_xarm/XarmPush-v0)
import gymnasium as gym

from probe_scripted_push import oracle_action

BASE_SEED = 5000


def read_state_10d(unwrapped_env):
    grasp_site_position = unwrapped_env._utils.get_site_xpos(
        unwrapped_env.model, unwrapped_env.data, "grasp"
    )
    cube_position = unwrapped_env.obj + unwrapped_env.center_of_table  # absolute frame
    goal_position = unwrapped_env.goal  # already absolute
    return np.concatenate(
        [grasp_site_position, unwrapped_env.gripper_angle, cube_position, goal_position]
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--output", type=str, default="oracle_demos_800.npz")
    arguments = parser.parse_args()

    environment = gym.make("gym_xarm/XarmPush-v0", obs_type="state")
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
        for _ in range(300):
            state = read_state_10d(unwrapped)
            action = oracle_action(unwrapped)
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
                f"running success rate {running_success_rate:.1%}",
                flush=True,
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
    print("state layout: [hand x, hand y, hand z, gripper angle, "
          "cube x, cube y, cube z, goal x, goal y, goal z]")
    print("state min per dim:", np.round(states.min(axis=0), 4))
    print("state max per dim:", np.round(states.max(axis=0), 4))
    print("action min per dim:", np.round(actions.min(axis=0), 4))
    print("action max per dim:", np.round(actions.max(axis=0), 4))
    print(f"saved {arguments.output}")


if __name__ == "__main__":
    main()
