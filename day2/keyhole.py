from pathlib import Path

from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCALE = 5
MARGIN, GAP, CAPTION_H = 20, 40, 30
OUT = Path(__file__).parent / "keyhole.png"

ds = LeRobotDataset("lerobot/pusht", episodes=[0])
mid = ds.num_frames // 2

img = ds[mid]["observation.image"]
arr = img.mul(255).round().byte().permute(1, 2, 0).numpy()

small = Image.fromarray(arr)
big = small.resize((96 * SCALE, 96 * SCALE), Image.NEAREST)

width = MARGIN + 96 + GAP + big.width + MARGIN
height = MARGIN + big.height + CAPTION_H + MARGIN
canvas = Image.new("RGB", (width, height), "white")
canvas.paste(small, (MARGIN, MARGIN))
canvas.paste(big, (MARGIN + 96 + GAP, MARGIN))

draw = ImageDraw.Draw(canvas)
draw.text((MARGIN, MARGIN + 96 + 8), "96x96\n(actual size)", fill="black")
draw.text((MARGIN + 96 + GAP, MARGIN + big.height + 8),
          f"same image, nearest-neighbor x{SCALE} (frame {mid} of episode 0)", fill="black")

canvas.save(OUT)
print(f"wrote {OUT} ({width}x{height})")
