# Renders day14 loss curve from train_losses.txt -> loss_curve.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs, losses = [], []
with open("train_losses.txt") as loss_file:
    for line in loss_file:
        epoch_number, loss_value = line.split()
        epochs.append(int(epoch_number))
        losses.append(float(loss_value))

figure, axes = plt.subplots(figsize=(7, 4), dpi=150)
axes.plot(epochs, losses, color="#2a78d6", linewidth=2)
axes.set_title("Behavior cloning on xarm lift — training loss", color="#1a1a19", pad=12)
axes.set_xlabel("epoch", color="#5f5e56")
axes.set_ylabel("MSE loss", color="#5f5e56")
axes.grid(True, color="#e8e7e0", linewidth=0.8)
for spine_name in ["top", "right"]:
    axes.spines[spine_name].set_visible(False)
for spine_name in ["left", "bottom"]:
    axes.spines[spine_name].set_color("#c3c2b7")
axes.tick_params(colors="#5f5e56")
axes.annotate(
    f"{losses[-1]:.3f}",
    xy=(epochs[-1], losses[-1]),
    xytext=(-6, 10),
    textcoords="offset points",
    ha="right",
    color="#1a1a19",
)
figure.tight_layout()
figure.savefig("loss_curve.png")
print("saved loss_curve.png")
