import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/xarm_lift_medium")

print("episodes:", dataset.num_episodes)
print("frames:", dataset.num_frames)

sample = dataset[0]
print("\n--- one sample ---")
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"{key}: shape {tuple(value.shape)}, dtype {value.dtype}")
    else:
        print(f"{key}: {value!r} ({type(value).__name__})")

print("\n--- bulk ranges (via hf_dataset, no video decoding) ---")
states = torch.stack(list(dataset.hf_dataset["observation.state"]))
actions = torch.stack(list(dataset.hf_dataset["action"]))
print("states tensor:", tuple(states.shape))
print("state  min:", states.min(dim=0).values)
print("state  max:", states.max(dim=0).values)
print("action min:", actions.min(dim=0).values)
print("action max:", actions.max(dim=0).values)

print("\n--- teacher's report card: final reward of each episode ---")
frame_indices = torch.stack(list(dataset.hf_dataset["frame_index"]))
assert (frame_indices[24::25] == 24).all(), "episodes are not uniform 25-frame blocks"
rewards = torch.stack(list(dataset.hf_dataset["next.reward"]))
final_rewards = rewards[24::25].float()
print("final rewards:", tuple(final_rewards.shape))
print("min:", final_rewards.min().item())
print("mean:", final_rewards.mean().item())
print("max:", final_rewards.max().item())
