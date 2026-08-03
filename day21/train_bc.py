# Day 21 phase 4: behavior cloning on the push oracle's demos. Recipe copied
# unchanged from day14/train_bc.py via day16/train_bc.py: seed 1000, batch 256,
# Adam lr 1e-3, MSE, 40 epochs, states normalized to mean 0 / std 1, actions
# raw (teacher actions already live in [-1, 1]). Only the input width changes:
# 10-dim [hand xyz, gripper, cube xyz, goal xyz] — sighted AND goal-aware,
# because the push target moves every episode.
# Only pairs from SUCCESSFUL teacher episodes are used (failures teach failure).
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--demos", type=str, default="oracle_demos_800.npz")
arguments = parser.parse_args()

torch.manual_seed(1000)

input_dims = 10

# ---- data: oracle demos, successful episodes only ----
demos = np.load(arguments.demos)
successful_episode = demos["episode_successes"][demos["episode_indices"]]
states = torch.from_numpy(demos["states"][successful_episode])
actions = torch.from_numpy(demos["actions"][successful_episode])
print(f"input dims {input_dims}")
print(f"pairs: {len(states)} of {len(demos['states'])} "
      f"(dropped {int((~demos['episode_successes']).sum())} failed episodes)")

# ---- normalization: shift/scale each state dimension to mean 0, std 1 ----
# goal z is CONSTANT on this task (0.545 every episode), so its std is exactly
# 0 and (x - mean) / std would be 0/0 = NaN for the whole dataset — this
# NaN'd the first training attempt. Constant dims get std 1: they normalize
# to ~0 and carry no signal, which is exactly right for a constant.
state_mean = states.mean(dim=0)
state_std = states.std(dim=0)
state_std = torch.where(state_std < 1e-6, torch.ones_like(state_std), state_std)
normalized_states = (states - state_mean) / state_std


class StateActionDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]


pairs = StateActionDataset(normalized_states, actions)
loader = DataLoader(pairs, batch_size=256, shuffle=True)


# ---- model: same 128-128 trunk as day 14/16, only the input width changes ----
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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = PolicyMLP(input_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()
print("model on", device, "| parameters:",
      sum(p.numel() for p in model.parameters()))

# ---- training loop, unchanged from day 14 ----
epochs = 40
epoch_losses = []
for epoch in range(1, epochs + 1):
    total_squared_error = 0.0
    for state_batch, action_batch in loader:
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)

        optimizer.zero_grad()
        predicted_actions = model(state_batch)
        loss = loss_function(predicted_actions, action_batch)
        loss.backward()
        optimizer.step()

        total_squared_error += loss.item() * len(state_batch)
    epoch_loss = total_squared_error / len(pairs)
    epoch_losses.append(epoch_loss)
    print(f"epoch {epoch:3d}/{epochs}  mse loss {epoch_loss:.6f}", flush=True)

# ---- evidence: per-epoch losses + checkpoint (weights AND normalization stats) ----
losses_path = "train_losses_push.txt"
with open(losses_path, "w") as loss_file:
    for epoch_number, loss_value in enumerate(epoch_losses, 1):
        loss_file.write(f"{epoch_number} {loss_value:.8f}\n")

checkpoint_path = "bc_mlp_push.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "input_dims": input_dims,
        "mode": "push",
    },
    checkpoint_path,
)
print(f"saved {checkpoint_path} and {losses_path}")
