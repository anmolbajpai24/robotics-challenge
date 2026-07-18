import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht")

print("== Q1: size ==")
print(f"episodes: {ds.num_episodes}   frames: {ds.num_frames}   fps: {ds.fps}")
print(f"average demo length: {ds.num_frames / ds.num_episodes / ds.fps:.1f} s")

print("\n== Q2: features ==")
for name, spec in ds.features.items():
    print(f"{name:20s} dtype={spec['dtype']:8s} shape={spec['shape']}")

print("\n== Q3: frame 0 ==")
frame = ds[0]
for key, value in frame.items():
    if isinstance(value, torch.Tensor):
        print(f"{key:20s} shape={tuple(value.shape)} dtype={value.dtype}")
    else:
        print(f"{key:20s} {value!r}")
img = frame["observation.image"]
print(f"image value range: {img.min():.3f} .. {img.max():.3f}")
print(f"pusher at {frame['observation.state'].tolist()}, commanded to {frame['action'].tolist()}")

print("\n== Q4: episode 0 ==")
ep0 = ds.meta.episodes[0]
start, end = int(ep0["dataset_from_index"]), int(ep0["dataset_to_index"])
n = end - start
print(f"rows {start}..{end} -> {n} frames = {n / ds.fps:.1f} s")

print("\n== Q5: action range (episode 0) ==")
actions = torch.stack([ds[i]["action"] for i in range(start, end)])
print(f"shape of stacked actions: {tuple(actions.shape)}")
print(f"x range: {actions[:, 0].min():.0f} .. {actions[:, 0].max():.0f}")
print(f"y range: {actions[:, 1].min():.0f} .. {actions[:, 1].max():.0f}")
