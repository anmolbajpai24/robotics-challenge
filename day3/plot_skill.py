"""Plot success rate vs training steps from the day3 checkpoint eval sweep.

Reads every day3/eval_*/eval_info.json, takes overall pc_success, and plots it
against the checkpoint step (parsed from the directory name; eval_final = 90k).
Rerun anytime: python day3/plot_skill.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAY3 = Path(__file__).parent
PRETRAINED_PC = 61.0  # day-1 baseline: lerobot pretrained diffusion policy, 500 eps
FINAL_STEP = 90_000

points = []  # (step, pc_success, n_episodes)
for info_path in sorted(DAY3.glob("eval_*/eval_info.json")):
    name = info_path.parent.name
    step = FINAL_STEP if name == "eval_final" else int(re.sub(r"\D", "", name))
    d = json.load(open(info_path))
    ov = d["overall"]
    points.append((step, ov["pc_success"], ov["n_episodes"]))

if not points:
    raise SystemExit("No day3/eval_*/eval_info.json files found yet.")

points.sort()
steps = [p[0] for p in points]
pcs = [p[1] for p in points]

plt.figure(figsize=(8, 5))
plt.axhline(PRETRAINED_PC, ls="--", color="gray",
            label=f"Pretrained baseline ({PRETRAINED_PC:.0f}%, 500 eps)")
plt.plot(steps, pcs, marker="o", label="From scratch (this run)")
for s, pc, n in points:
    plt.annotate(f"{pc:.0f}%", (s, pc), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)
plt.ylim(0, 100)
plt.xlabel("Training step")
plt.ylabel("Success rate (%, 500-episode eval at 90k, 150 at intermediates)")
plt.title("PushT diffusion policy — skill vs training time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(DAY3 / "skill_curve.png", dpi=150)

for s, pc, n in points:
    print(f"step {s:>6}: {pc:5.1f}% success  (n={n})")
print(f"wrote {DAY3 / 'skill_curve.png'}")
