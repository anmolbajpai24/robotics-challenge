# Day 21 phase 2: validation probe for push (NOT a baseline policy) — scripted
# push using privileged cube + goal position from the sim. Proves the
# resurrected env + action semantics + patched success check can produce
# success at all. Same role as day14/probe_scripted_lift.py.
#
# Controller: day14 pattern (proportional offsets, gain 20, clip [-1, 1]) but
# the phases are STATE-based, not step-timer based — a push can knock the cube
# sideways and need a re-approach, which a timer can't rewind. Phases:
#   travel : rise, then move to a standoff point 6 cm behind the cube along
#            the cube->goal line, at a safe height (fingers clear the cube)
#   descend: drop to push height (grasp site 1 cm above cube center — the
#            closed fingertips then press near cube-center height without
#            scraping the table 2.4 cm below)
#   press  : drive the hand through the cube toward the goal; forward speed
#            ramps down with remaining cube->goal distance so the cube is not
#            launched past the target (gentle commands — these become the BC
#            training targets)
# Gripper is commanded closed (+1) throughout, matching the original task's
# constant gripper padding of 1.0.
import argparse

import numpy as np

import xarm_push_compat  # noqa: F401  (registers gym_xarm/XarmPush-v0)
import gymnasium as gym

STANDOFF = 0.06          # hand waits this far behind the cube before pressing
PUSH_HEIGHT_LIFT = 0.01  # grasp site rides 1 cm above cube-center height
SAFE_CLEARANCE = 0.08    # travel height above cube center
BEHIND_MIN = 0.022       # below steady-press contact distance (~0.032 =
                         # cube half 0.024 + fingertip), so pressing is stable
LATERAL_MAX = 0.025      # off-axis corridor for the press


def oracle_action(unwrapped_env):
    hand = unwrapped_env._utils.get_site_xpos(
        unwrapped_env.model, unwrapped_env.data, "grasp"
    )
    cube = unwrapped_env.obj + unwrapped_env.center_of_table  # absolute frame
    goal = unwrapped_env.goal

    to_goal_xy = goal[:2] - cube[:2]
    distance_to_goal = np.linalg.norm(to_goal_xy)
    push_direction = to_goal_xy / max(distance_to_goal, 1e-6)

    push_z = cube[2] + PUSH_HEIGHT_LIFT
    safe_z = cube[2] + SAFE_CLEARANCE
    behind_xy = cube[:2] - push_direction * STANDOFF

    hand_from_cube = hand[:2] - cube[:2]
    along = np.dot(hand_from_cube, push_direction)      # negative = behind cube
    lateral = abs(
        hand_from_cube[0] * push_direction[1] - hand_from_cube[1] * push_direction[0]
    )
    aligned = (along <= -BEHIND_MIN) and (lateral <= LATERAL_MAX)
    hand_is_low = hand[2] <= push_z + 0.015
    hand_at_standoff = np.linalg.norm(hand[:2] - behind_xy) <= 0.02

    if aligned and hand_is_low:
        # press through the cube toward the goal, slowing as the cube closes in
        press_speed = np.clip(distance_to_goal * 8.0, 0.08, 0.5)
        press_point_xy = cube[:2] - push_direction * 0.015
        correction_xy = np.clip((press_point_xy - hand[:2]) * 20.0, -0.3, 0.3)
        correction_z = np.clip((push_z - hand[2]) * 20.0, -0.3, 0.3)
        action = np.clip(
            np.array(
                [
                    push_direction[0] * press_speed + correction_xy[0],
                    push_direction[1] * press_speed + correction_xy[1],
                    correction_z,
                ]
            ),
            -1.0,
            1.0,
        )
    else:
        if hand_at_standoff:
            target = np.array([behind_xy[0], behind_xy[1], push_z])      # descend
        elif hand[2] < safe_z - 0.01:
            target = np.array([hand[0], hand[1], safe_z])                # rise first
        else:
            target = np.array([behind_xy[0], behind_xy[1], safe_z])      # travel
        action = np.clip((target - hand) * 20.0, -1.0, 1.0)
    return np.append(action, 1.0).astype(np.float32)  # gripper closed throughout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    arguments = parser.parse_args()

    environment = gym.make("gym_xarm/XarmPush-v0", obs_type="state")
    unwrapped = environment.unwrapped

    success_count = 0
    for episode_index in range(arguments.episodes):
        environment.reset(seed=arguments.seed + episode_index)
        start_distance = np.linalg.norm(
            (unwrapped.obj + unwrapped.center_of_table) - unwrapped.goal
        )
        success = False
        steps_taken = 0
        for _ in range(300):
            action = oracle_action(unwrapped)
            observation, reward, terminated, truncated, info = environment.step(action)
            steps_taken += 1
            if terminated:
                success = True  # only the patched success check terminates
                break
            if truncated:
                break
        final_distance = np.linalg.norm(
            (unwrapped.obj + unwrapped.center_of_table) - unwrapped.goal
        )
        success_count += success
        print(
            f"episode {episode_index + 1:2d}/{arguments.episodes}  "
            f"seed {arguments.seed + episode_index}  steps {steps_taken:3d}  "
            f"success {success}  cube->goal {start_distance:.3f} -> {final_distance:.3f}"
        )

    environment.close()
    print(
        f"\nSCRIPTED PUSH ORACLE: {success_count}/{arguments.episodes} = "
        f"{success_count / arguments.episodes:.0%}  (seeds {arguments.seed}.."
        f"{arguments.seed + arguments.episodes - 1})"
    )


if __name__ == "__main__":
    main()
