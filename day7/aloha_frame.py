"""Single labeled ALOHA frame for the day-7 post.
Pulls a late-episode (handoff) frame from the best eval video and annotates it
with ALOHA's own data-model dims. PIL idiom borrowed from day2/keyhole.py.
"""
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

DAY7 = Path(__file__).parent
MP4 = DAY7 / "eval_aloha" / "videos" / "aloha_0" / "eval_episode_3.mp4"
OUT = DAY7 / "aloha_transfer_frame.png"
FRAME_FRAC = 0.85  # near the end -> cube received by second arm (episode ends on success)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(sz, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"{FONT_DIR}/{name}", sz)


frames = iio.imread(MP4)  # (T, 480, 640, 3)
idx = int(len(frames) * FRAME_FRAC)
photo = Image.fromarray(frames[idx])
W, H = photo.size  # 640 x 480

MARGIN = 28
TITLE_H = 84
CAP_H = 210
canvas_w = W + 2 * MARGIN
canvas_h = TITLE_H + H + CAP_H + MARGIN
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
canvas.paste(photo, (MARGIN, TITLE_H))

d = ImageDraw.Draw(canvas)
ink, sub = (17, 17, 17), (90, 90, 90)

# title
d.text((MARGIN, 22), "ALOHA sim — bimanual transfer-cube", font=font(30, bold=True), fill=ink)
d.text((MARGIN, 58),
       f"pretrained ACT · gym-aloha / MuJoCo · frame {idx}/{len(frames)} of eval episode 3 (success)",
       font=font(16), fill=sub)

# caption rows: (label, value)
rows = [
    ("Observation", "480×640 RGB  ·  1 camera (top-down)"),
    ("State", "14-dim  —  2 arms × 7  (6 joints + 1 gripper)"),
    ("Action", "14-dim  —  target joint positions"),
    ("Control", "50 fps  ·  up to 400 steps / episode"),
    ("This run", "70% success (7/10 episodes)  ·  seed 1000"),
]
y = TITLE_H + H + 18
lab_f, val_f = font(19, bold=True), font(19)
for label, value in rows:
    d.text((MARGIN, y), label, font=lab_f, fill=ink)
    d.text((MARGIN + 150, y), value, font=val_f, fill=ink)
    y += 36

canvas.save(OUT)
print(f"wrote {OUT} ({canvas_w}x{canvas_h}) from frame {idx}/{len(frames)}")
