# Day 14: behavior cloning on xarm lift, written by hand — no lerobot-train.
# Learns state -> action from lerobot/xarm_lift_medium (teacher: a TD-MPC RL agent).
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

torch.manual_seed(1000)

# ---- data: all 20,000 (state, action) pairs, loaded fully into memory ----
lerobot_dataset = LeRobotDataset("lerobot/xarm_lift_medium")
states = torch.stack(list(lerobot_dataset.hf_dataset["observation.state"]))
actions = torch.stack(list(lerobot_dataset.hf_dataset["action"]))

# ---- normalization: shift/scale each state dimension to mean 0, std 1 ----
state_mean = states.mean(dim=0)
state_std = states.std(dim=0)
normalized_states = (states - state_mean) / state_std
# actions stay raw: the teacher's actions already live in [-1, 1]


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

first_states, first_actions = next(iter(loader))
print("one batch:", tuple(first_states.shape), tuple(first_actions.shape))


# ---- model: 4 numbers in (where the hand is) -> 4 numbers out (how to move) ----
class PolicyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, state):
        return self.layers(state)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = PolicyMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()
print("model on", device, "| parameters:",
      sum(p.numel() for p in model.parameters()))

# ---- the training loop: the whole point of tonight ----
epochs = 40
epoch_losses = []
for epoch in range(1, epochs + 1):
    total_squared_error = 0.0
    for state_batch, action_batch in loader:
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)

        optimizer.zero_grad()                        # 1. wipe last batch's gradients
        predicted_actions = model(state_batch)       # 2. forward: guess actions
        loss = loss_function(predicted_actions, action_batch)  # 3. how wrong?
        loss.backward()                              # 4. gradients: blame each weight
        optimizer.step()                             # 5. nudge every weight downhill

        total_squared_error += loss.item() * len(state_batch)
    epoch_loss = total_squared_error / len(pairs)
    epoch_losses.append(epoch_loss)
    print(f"epoch {epoch:3d}/{epochs}  mse loss {epoch_loss:.6f}")

# ---- evidence: per-epoch losses + checkpoint (weights AND normalization stats) ----
with open("train_losses.txt", "w") as loss_file:
    for epoch_number, loss_value in enumerate(epoch_losses, 1):
        loss_file.write(f"{epoch_number} {loss_value:.8f}\n")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
    },
    "bc_mlp.pt",
)
print("saved bc_mlp.pt and train_losses.txt")
