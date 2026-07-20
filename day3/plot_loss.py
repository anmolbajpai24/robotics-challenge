"""Parse day3/train.log and plot the training loss curve to day3/loss_curve.png.

Rerun anytime (mid-run or after completion) from the repo root:
    python day3/plot_loss.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file; no display available under WSL
import matplotlib.pyplot as plt

LOG_PATH = Path(__file__).parent / "train.log"
OUT_PATH = Path(__file__).parent / "loss_curve.png"

# The log interleaves tqdm progress-bar writes (stderr) with logging (stdout),
# tearing most INFO lines mid-token. Reconstruct the stdout stream: delete the
# bar chunks, then drop the \r separators so torn INFO fragments rejoin.
BAR = re.compile(r"Training:\s*\d+%\|[^|]*\|\s*\d+/\d+ \[[^\]]*\]?")
# Past step 1000 the logger prints rounded K/M notation ("step:10K"), so several
# consecutive records can display the same step; the dict keeps the last one.
pat = re.compile(r"step:([0-9.]+)([KM]?) smpl:.*?loss:([0-9.]+) grdn:")
MULT = {"": 1, "K": 1_000, "M": 1_000_000}

with open(LOG_PATH) as f:
    text = BAR.sub("", f.read()).replace("\r", "")

points = {}
for line in text.splitlines():
    m = pat.search(line)
    if m:
        points[int(float(m.group(1)) * MULT[m.group(2)])] = float(m.group(3))

if not points:
    raise SystemExit(f"No 'step:... loss:...' lines found in {LOG_PATH} — log format changed?")

steps = sorted(points)
losses = [points[s] for s in steps]

plt.figure(figsize=(8, 5))
plt.plot(steps, losses)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Loss (noise-prediction MSE)")
plt.title("Diffusion policy on PushT — day 3 overnight run")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)

print(f"{len(steps)} points parsed; last: step {steps[-1]}, loss {losses[-1]:.4f}")
print(f"wrote {OUT_PATH}")
