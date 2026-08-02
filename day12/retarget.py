"""Turn tracked hand trajectories into ALOHA end-effector actions.

Input : day12/hand_trajectory.csv   (from track_hands.py, 59.976 fps, image coords)
Output: day12/aloha_actions.npy     (N x 16, 50 Hz, sim metres)
        day12/retargeted_paths.png  (what the arms will actually be told to do)

The 16-wide action layout is fixed by gym_aloha's TransferCubeEndEffectorTask:
    [0:3]   left  mocap xyz (metres, world frame)
    [3:7]   left  quaternion (w, x, y, z)
    [7]     left  gripper, 0 = closed, 1 = open
    [8:11]  right mocap xyz
    [11:15] right quaternion
    [15]    right gripper

Run with the lerobot-env python (needs numpy + matplotlib only).
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIM_HZ = 50.0  # gym_aloha DT = 0.02

# Where we want the FINGERTIPS to travel, in sim metres. The cube spawns at
# x in [0.0, 0.2], y in [0.4, 0.6], so both boxes straddle reachable ground.
FINGERTIP_X_LEFT = (-0.32, -0.05)
FINGERTIP_X_RIGHT = (0.05, 0.32)
FINGERTIP_Y = (0.40, 0.62)

# Measured offset between the commanded mocap target and where the fingertip
# geom actually ends up. The weld constraint is soft (solimp=".25 .25 0.001"),
# so the gripper trails its target, and the error varies with arm pose --
# these are averages over a probe grid, good to roughly +/-3 cm. See
# day12/weld_offset_probe.txt.
WELD_OFFSET_RIGHT = np.array([0.070, 0.042, -0.040])
WELD_OFFSET_LEFT = np.array([-0.070, 0.042, -0.040])

# Gripper orientation. [1,0,0,0] is the identity the task resets to; holding it
# fixed means we control position only, which is the scoped claim being made.
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def interpolate_gaps(values, times):
    """Fill NaN runs by linear interpolation; hold the edges flat.

    Tracking dropped 234 of 1979 frames. The sim needs a number at every step,
    so gaps get bridged rather than skipped -- but only gaps. Leading and
    trailing NaNs are filled with the nearest real value rather than
    extrapolated, since extrapolating a hand position is inventing motion.
    """
    values = np.asarray(values, dtype=float)
    is_valid = ~np.isnan(values)
    if not is_valid.any():
        raise ValueError("column is entirely NaN -- nothing to interpolate")
    return np.interp(times, times[is_valid], values[is_valid])


def smooth(values, window):
    """Centred moving average. Frame-to-frame jitter is tracking noise, not motion."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def map_range(values, source_low, source_high, target_low, target_high):
    """Linearly stretch one range onto another, then clamp to the target."""
    if source_high - source_low < 1e-9:
        return np.full_like(values, (target_low + target_high) / 2.0)
    fraction = (values - source_low) / (source_high - source_low)
    return np.clip(target_low + fraction * (target_high - target_low),
                   min(target_low, target_high), max(target_low, target_high))


def build_arm_track(
    wrist_x, wrist_y, pinch_metres, times_source, times_sim,
    fingertip_x_range, weld_offset, height, pinch_threshold_cm, smooth_window,
):
    """One arm: image coordinates in, mocap targets + gripper command out."""
    # 1. Clamp. Values outside 0..1 are MediaPipe extrapolating a hand that had
    #    left the frame -- guesses, not measurements. Left alone they become
    #    sudden lunges to the edge of the workspace.
    wrist_x = np.clip(wrist_x, 0.0, 1.0)
    wrist_y = np.clip(wrist_y, 0.0, 1.0)

    # 2. Bridge dropouts, 3. de-jitter, 4. change clock 59.976 Hz -> 50 Hz.
    wrist_x = smooth(interpolate_gaps(wrist_x, times_source), smooth_window)
    wrist_y = smooth(interpolate_gaps(wrist_y, times_source), smooth_window)
    pinch_cm = smooth(interpolate_gaps(pinch_metres, times_source), smooth_window) * 100.0

    wrist_x = np.interp(times_sim, times_source, wrist_x)
    wrist_y = np.interp(times_sim, times_source, wrist_y)
    pinch_cm = np.interp(times_sim, times_source, pinch_cm)

    # 5. Stretch each hand's own observed range across its half of the table.
    #    Per-hand normalisation (rather than mapping the full 0..1 frame) is
    #    what makes the arms actually move -- each hand only occupied about a
    #    fifth of the frame width.
    observed_x_low, observed_x_high = wrist_x.min(), wrist_x.max()
    observed_y_low, observed_y_high = wrist_y.min(), wrist_y.max()

    fingertip_x = map_range(wrist_x, observed_x_low, observed_x_high, *fingertip_x_range)
    # Image y grows downward but sim +y renders upward, so this mapping inverts.
    fingertip_y = map_range(wrist_y, observed_y_low, observed_y_high,
                            FINGERTIP_Y[1], FINGERTIP_Y[0])
    fingertip_z = np.full_like(fingertip_x, height)

    # Undo the soft-weld lag so the fingertip lands near the intended point.
    mocap = np.stack([fingertip_x, fingertip_y, fingertip_z], axis=1) - weld_offset

    # 6. Pinch -> gripper. Sim wants 0 = closed, 1 = open.
    gripper = np.where(pinch_cm < pinch_threshold_cm, 0.0, 1.0)

    return mocap, gripper, pinch_cm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="day12/hand_trajectory.csv")
    parser.add_argument("--out-actions", default="day12/aloha_actions.npy")
    parser.add_argument("--out-plot", default="day12/retargeted_paths.png")
    parser.add_argument("--height", type=float, default=0.10,
                        help="fingertip height in metres. 0.05 is cube centre height; "
                             "the probe showed mocap_z=0.10 lands the fingertip at ~0.054.")
    parser.add_argument("--pinch-threshold-cm", type=float, default=6.0,
                        help="below this the gripper closes. Right hand median was 5.77, "
                             "left hand sat at 7.95 and never grasped.")
    parser.add_argument("--smooth-window", type=int, default=9,
                        help="moving-average width in source frames (9 ~= 0.15 s)")
    parser.add_argument("--no-weld-correction", action="store_true",
                        help="command raw fingertip targets without the measured offset")
    arguments = parser.parse_args()

    data = np.genfromtxt(arguments.csv, delimiter=",", names=True)
    times_source = data["time_seconds"]
    duration = times_source[-1]
    times_sim = np.arange(0.0, duration, 1.0 / SIM_HZ)

    offset_left = np.zeros(3) if arguments.no_weld_correction else WELD_OFFSET_LEFT
    offset_right = np.zeros(3) if arguments.no_weld_correction else WELD_OFFSET_RIGHT

    left_mocap, left_gripper, left_pinch = build_arm_track(
        data["left_wrist_x"], data["left_wrist_y"], data["left_pinch_m"],
        times_source, times_sim, FINGERTIP_X_LEFT, offset_left,
        arguments.height, arguments.pinch_threshold_cm, arguments.smooth_window,
    )
    right_mocap, right_gripper, right_pinch = build_arm_track(
        data["right_wrist_x"], data["right_wrist_y"], data["right_pinch_m"],
        times_source, times_sim, FINGERTIP_X_RIGHT, offset_right,
        arguments.height, arguments.pinch_threshold_cm, arguments.smooth_window,
    )

    step_count = len(times_sim)
    actions = np.zeros((step_count, 16))
    actions[:, 0:3] = left_mocap
    actions[:, 3:7] = IDENTITY_QUAT
    actions[:, 7] = left_gripper
    actions[:, 8:11] = right_mocap
    actions[:, 11:15] = IDENTITY_QUAT
    actions[:, 15] = right_gripper

    np.save(arguments.out_actions, actions)

    figure, (path_axis, grip_axis) = plt.subplots(1, 2, figsize=(14, 6))
    path_axis.plot(left_mocap[:, 0], left_mocap[:, 1], linewidth=0.8, label="left arm")
    path_axis.plot(right_mocap[:, 0], right_mocap[:, 1], linewidth=0.8, label="right arm")
    path_axis.add_patch(plt.Rectangle((0.0, 0.4), 0.2, 0.2, fill=False,
                                      linestyle="--", color="red", label="cube spawn zone"))
    path_axis.set_xlim(-0.45, 0.45)
    path_axis.set_ylim(0.30, 0.70)
    path_axis.set_xlabel("sim x (metres)")
    path_axis.set_ylabel("sim y (metres)")
    path_axis.set_title("retargeted mocap paths, top-down")
    path_axis.legend()
    path_axis.grid(alpha=0.3)

    grip_axis.plot(times_sim, left_pinch, linewidth=0.7, alpha=0.5, label="left pinch (cm)")
    grip_axis.plot(times_sim, right_pinch, linewidth=0.7, alpha=0.5, label="right pinch (cm)")
    grip_axis.plot(times_sim, left_gripper * 2 + 10, linewidth=1.2, label="left gripper cmd")
    grip_axis.plot(times_sim, right_gripper * 2 + 13, linewidth=1.2, label="right gripper cmd")
    grip_axis.axhline(arguments.pinch_threshold_cm, linestyle="--", color="grey",
                      label=f"threshold {arguments.pinch_threshold_cm} cm")
    grip_axis.set_xlabel("time (seconds)")
    grip_axis.set_title("pinch signal and resulting gripper commands (offset for legibility)")
    grip_axis.legend(fontsize=8)
    grip_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(arguments.out_plot, dpi=120)
    plt.close(figure)

    left_closed = int((left_gripper == 0).sum())
    right_closed = int((right_gripper == 0).sum())
    print(f"source frames      : {len(times_source)} @ {len(times_source)/duration:.3f} fps")
    print(f"sim steps written  : {step_count} @ {SIM_HZ:.0f} Hz ({duration:.2f} s)")
    print(f"weld correction    : {'OFF' if arguments.no_weld_correction else 'ON'}")
    print(f"fingertip height   : {arguments.height:.3f} m")
    print(f"left  mocap x range: {left_mocap[:,0].min():+.3f} .. {left_mocap[:,0].max():+.3f}"
          f"   y {left_mocap[:,1].min():.3f} .. {left_mocap[:,1].max():.3f}")
    print(f"right mocap x range: {right_mocap[:,0].min():+.3f} .. {right_mocap[:,0].max():+.3f}"
          f"   y {right_mocap[:,1].min():.3f} .. {right_mocap[:,1].max():.3f}")
    print(f"left  gripper closed : {left_closed}/{step_count} steps ({100*left_closed/step_count:.1f}%)")
    print(f"right gripper closed : {right_closed}/{step_count} steps ({100*right_closed/step_count:.1f}%)")
    print(f"actions -> {arguments.out_actions}")
    print(f"plot    -> {arguments.out_plot}")


if __name__ == "__main__":
    main()
