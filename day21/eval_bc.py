# Day 21 phase 5: frozen-500 eval for the push BC MLP. Protocol IDENTICAL in
# shape to day14/EVAL_PROTOCOL.md / day16/eval_bc.py: 500 episodes,
# reset(seed=1000+i), 300-step cap (from the env registration), success =
# patched absolute-frame check (cube within 5 cm of the goal), actions clamped
# [-1, 1] float32, first 3 episodes to video.
# Model input read straight from the physics state in the demo dataset's
# frame: [grasp-site xyz, gripper angle, cube xyz, goal xyz], all absolute.
import argparse
import time

import numpy as np
import torch
from torch import nn
import imageio

import xarm_push_compat  # noqa: F401  (registers gym_xarm/XarmPush-v0)
import gymnasium as gym


class PolicyMLP(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, state):
        return self.layers(state)


def read_model_frame_state(unwrapped_env):
    grasp_site_position = unwrapped_env._utils.get_site_xpos(
        unwrapped_env.model, unwrapped_env.data, "grasp"
    )
    cube_position = unwrapped_env.obj + unwrapped_env.center_of_table  # absolute
    return np.concatenate(
        [grasp_site_position, unwrapped_env.gripper_angle, cube_position,
         unwrapped_env.goal]
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="bc_mlp_push.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-episodes", type=int, default=3)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    input_dims = checkpoint["input_dims"]
    mode = checkpoint["mode"]
    model = PolicyMLP(input_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_mean = checkpoint["state_mean"]
    state_std = checkpoint["state_std"]
    print(f"checkpoint {arguments.checkpoint}: mode {mode}, input dims {input_dims}")

    environment = gym.make("gym_xarm/XarmPush-v0", obs_type="state")
    unwrapped = environment.unwrapped

    success_count = 0
    start_time = time.monotonic()
    for episode_index in range(arguments.episodes):
        observation, info = environment.reset(seed=arguments.seed + episode_index)
        video_frames = [] if episode_index < arguments.video_episodes else None
        if episode_index == 0:
            print("frame check — first model input:",
                  read_model_frame_state(unwrapped))

        success = False
        steps_taken = 0
        while True:
            state = torch.from_numpy(read_model_frame_state(unwrapped))
            normalized_state = (state - state_mean) / state_std
            with torch.no_grad():
                predicted_action = model(normalized_state.unsqueeze(0)).squeeze(0)
            action = predicted_action.clamp(-1.0, 1.0).numpy().astype(np.float32)

            observation, reward, terminated, truncated, info = environment.step(action)
            steps_taken += 1
            if video_frames is not None:
                video_frames.append(environment.render())
            if terminated:
                success = True  # only the patched success check terminates
                break
            if truncated:
                break

        success_count += success
        if video_frames is not None:
            video_path = f"rollout_{mode}_ep{episode_index}.mp4"
            imageio.mimsave(video_path, video_frames, fps=25)
            print(f"  saved {video_path} ({len(video_frames)} frames)")
        print(
            f"episode {episode_index + 1:3d}/{arguments.episodes}  "
            f"seed {arguments.seed + episode_index}  steps {steps_taken:3d}  "
            f"success {success}  running rate {success_count / (episode_index + 1):.1%}",
            flush=True,
        )

    elapsed_seconds = time.monotonic() - start_time
    environment.close()
    print(
        f"\nRESULT [{mode}]: {success_count}/{arguments.episodes} = "
        f"{success_count / arguments.episodes:.1%} success | "
        f"seed base {arguments.seed} | {elapsed_seconds:.1f}s "
        f"({elapsed_seconds / arguments.episodes:.2f}s/episode)"
    )


if __name__ == "__main__":
    main()
