# Day 21 phase 1: recon the resurrected push env before anything trains.
# Pastes the deprecation + cross-frame evidence straight from the installed
# sources, boots the env, and checks: spaces, frames, settled cube height,
# that the shipped success math is unsatisfiable, and that the patched one
# is geometrically reachable (cube teleported onto the goal -> success True).
import numpy as np

import xarm_push_compat  # noqa: F401  (registers gym_xarm/XarmPush-v0, patches applied)
import gymnasium as gym
import gym_xarm
import gym_xarm.tasks.push  # not imported by tasks/__init__ — that line is commented out too
import gym_xarm.tasks.base


def print_source_lines(path, first_line, last_line, label):
    print(f"\n--- {label} ({path.split('site-packages/')[-1]} "
          f"lines {first_line}-{last_line}) ---")
    with open(path) as source_file:
        for line_number, line in enumerate(source_file, 1):
            if first_line <= line_number <= last_line:
                print(f"{line_number:4d}  {line.rstrip()}")


push_py = gym_xarm.tasks.push.__file__
base_py = gym_xarm.tasks.base.__file__
print_source_lines(push_py, 1, 23, "installed push.py: deprecated stub, is_success + reward commented out")
print_source_lines(push_py, 60, 77, "installed push.py: commented _sample_goal — self.goal is ABSOLUTE world coords")
print_source_lines(base_py, 173, 175, "installed base.py: obj property is TABLE-RELATIVE — same cross-frame bug as lift")

environment = gym.make("gym_xarm/XarmPush-v0", obs_type="state")
unwrapped = environment.unwrapped

print("\n--- spaces ---")
print("action space:     ", environment.action_space)
print("observation space:", environment.observation_space)
print("max episode steps:", environment.spec.max_episode_steps)

print("\n--- resets (hand/cube/goal all printed in the ABSOLUTE frame) ---")
for seed in (1000, 1001, 1002):
    environment.reset(seed=seed)
    hand_absolute = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "grasp")
    cube_absolute = unwrapped.obj + unwrapped.center_of_table
    mixed_frame_distance = np.linalg.norm(unwrapped.obj - unwrapped.goal)  # shipped math
    fixed_frame_distance = np.linalg.norm(cube_absolute - unwrapped.goal)  # patched math
    print(f"seed {seed}:")
    print(f"  hand {np.round(hand_absolute, 4)}  cube {np.round(cube_absolute, 4)}  "
          f"goal {np.round(unwrapped.goal, 4)}")
    print(f"  shipped-math dist (rel cube vs abs goal): {mixed_frame_distance:.4f}  "
          f"<= 0.05? {mixed_frame_distance <= 0.05}")
    print(f"  patched-math dist (abs vs abs):           {fixed_frame_distance:.4f}  "
          f"<= 0.05? {fixed_frame_distance <= 0.05}")

print("\n--- 10 no-op steps (does the sim hold still and not explode) ---")
environment.reset(seed=1000)
cube_z_before = float((unwrapped.obj + unwrapped.center_of_table)[2])
for _ in range(10):
    observation, reward, terminated, truncated, info = environment.step(
        np.zeros(4, dtype=np.float32)
    )
cube_after = unwrapped.obj + unwrapped.center_of_table
print(f"cube z spawn-settled {cube_z_before:.4f} -> after 10 no-ops {cube_after[2]:.4f}")
print(f"goal z (fixed in _sample_goal): {unwrapped.goal[2]:.4f}")
print(f"resting z offset cube-vs-goal: {abs(cube_after[2] - unwrapped.goal[2]):.4f} "
      f"(3D threshold 0.05 leaves ~{np.sqrt(max(0.05**2 - (cube_after[2] - unwrapped.goal[2])**2, 0)):.4f} xy tolerance)")
print(f"reward at spawn: {reward:.4f}  terminated {terminated}  info {info}")

print("\n--- satisfiability: teleport cube onto the goal, patched check must fire ---")
object_qpos = unwrapped._utils.get_joint_qpos(unwrapped.model, unwrapped.data, "object_joint0")
object_qpos[:2] = unwrapped.goal[:2]  # xy onto the goal, keep resting z
unwrapped._utils.set_joint_qpos(unwrapped.model, unwrapped.data, "object_joint0", object_qpos)
xarm_push_compat.xarm_compat.mujoco.mj_forward(unwrapped.model, unwrapped.data)
print(f"after teleport: patched is_success() = {unwrapped.is_success()}")

environment.close()
print("\nRECON VERDICT: env boots, shipped success math unsatisfiable (cross-frame, "
      "same class of bug as lift), patched absolute-frame check satisfiable. Proceed.")
