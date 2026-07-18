from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = Path(__file__).parent / "trajectory_ep0.png"

ds = LeRobotDataset("lerobot/pusht", episodes=[0])
actions = torch.stack([ds[i]["action"] for i in range(ds.num_frames)])
x, y = actions[:, 0], actions[:, 1]

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(x, y, color="0.85", linewidth=1, zorder=1)
dots = ax.scatter(x, y, c=range(len(actions)), cmap="viridis", s=14, zorder=2)
fig.colorbar(dots, ax=ax, label="frame index (10 per second)")

ax.annotate("start", (float(x[0]), float(y[0])), xytext=(8, -8),
            textcoords="offset points", fontsize=9, color="0.3")
ax.annotate("end", (float(x[-1]), float(y[-1])), xytext=(8, -8),
            textcoords="offset points", fontsize=9, color="0.3")

ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.invert_yaxis()  # image coords, y goes down
ax.set_aspect("equal")
ax.set_xlabel("action x (canvas pixels)")
ax.set_ylabel("action y (canvas pixels)")
ax.set_title("Episode 0: commanded pusher positions over time")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
