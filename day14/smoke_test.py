import xarm_compat
import gymnasium as gym, gym_xarm

env = gym.make("gym_xarm/XarmLift-v0", obs_type="pixels_agent_pos")
obs, info = env.reset(seed=0)
print(obs["agent_pos"], obs["pixels"].shape, env.action_space)
u = env.unwrapped
print("obj (relative):", u.obj, " z_target:", u.z_target)
print("is_success now:", u.is_success(), " gap to success:", u.z_target - u.obj[2])
env.close()
