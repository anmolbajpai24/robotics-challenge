from pathlib import Path

from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCALE = 4
OUT = Path(__file__).parent / "episode0.gif"

ds = LeRobotDataset("lerobot/pusht", episodes=[0])

frames = []
for i in range(ds.num_frames):
    img = ds[i]["observation.image"]
    arr = img.mul(255).round().byte().permute(1, 2, 0).numpy()  # chw float -> hwc uint8
    frames.append(Image.fromarray(arr).resize((96 * SCALE, 96 * SCALE), Image.NEAREST))

frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=1000 // ds.fps, loop=0)
print(f"wrote {OUT} ({len(frames)} frames at {ds.fps} fps)")
