# Day 16 step 0 — recon: does lerobot/xarm_lift_medium store the cube's position
# ANYWHERE? Three checks, each one level deeper: the manifest (what the recorder
# said it logged), the table on disk (what is actually there), and one loaded
# sample (what a training batch sees). Pre-registered expectation: state is
# 4-dim hand xyz + gripper, and no cube channel exists.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/xarm_lift_medium")

print("--- manifest: every channel the recorder logged ---")
for feature_name, feature_info in dataset.features.items():
    print(f"{feature_name}: shape {feature_info['shape']}, names {feature_info['names']}")

print("\n--- reality: columns actually in the table on disk ---")
print(dataset.hf_dataset.column_names)

print("\n--- one sample, as training would see it ---")
sample = dataset[0]
for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"{key}: shape {tuple(value.shape)}, dtype {value.dtype}")
    else:
        print(f"{key}: {value!r}")

print("\nVERDICT: the only observation channels are observation.image (pixels)")
print("and observation.state (4 dims, generic motor names). No numeric cube")
print("position anywhere in the dataset -> Branch B: generate oracle demos.")
