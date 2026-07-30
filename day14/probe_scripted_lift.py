# Validation probe (NOT a baseline policy): scripted lift using privileged cube
# position from the sim. Proves the env + action semantics + patched success
# check can produce success at all. If this scores > 0, the benchmark is sound.
import numpy as np

import xarm_compat  # noqa: F401
import gymnasium as gym
import gym_xarm  # noqa: F401

EPISODES = 20
BASE_SEED = 1000

environment = gym.make("gym_xarm/XarmLift-v0", obs_type="state")
unwrapped = environment.unwrapped

success_count = 0
for episode_index in range(EPISODES):
    environment.reset(seed=BASE_SEED + episode_index)
    success = False
    for step_index in range(300):
        hand_position = unwrapped._utils.get_site_xpos(
            unwrapped.model, unwrapped.data, "grasp"
        )
        cube_position = unwrapped.obj + unwrapped.center_of_table  # back to absolute
        offset = cube_position - hand_position

        if np.linalg.norm(offset[:2]) > 0.01 and step_index < 80:
            # phase 1: line up above the cube
            action = np.array([offset[0], offset[1], 0.0, -1.0])
        elif offset[2] < -0.015 and step_index < 120:
            # phase 2: descend to it
            action = np.array([offset[0], offset[1], offset[2], -1.0])
        elif step_index < 150:
            # phase 3: squeeze
            action = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # phase 4: lift straight up, keep squeezing
            action = np.array([0.0, 0.0, 1.0, 1.0])

        action = np.clip(action * 20.0, -1.0, 1.0).astype(np.float32)
        action[3] = np.clip(action[3], -1.0, 1.0)
        observation, reward, terminated, truncated, info = environment.step(action)
        if terminated:
            success = True
            break
        if truncated:
            break
    success_count += success
    print(f"episode {episode_index + 1:2d}/{EPISODES}  success {success}")

environment.close()
print(f"\nSCRIPTED ORACLE: {success_count}/{EPISODES} = {success_count / EPISODES:.0%}")
